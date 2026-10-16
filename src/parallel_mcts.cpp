#include "parallel_mcts.hpp"
#include "action_encoding.hpp"
#include "self_play.hpp" // ParallelRunner

#include <c10/core/DeviceGuard.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <thread>

namespace az {

namespace {

// Terminal value for a finished game from the side-to-move's view
// (mirrors MCTS::terminal_value).
float terminal_value(const Board& board) {
    std::string res = board.result();
    if (res == "1/2-1/2") return 0.0f;
    return -1.0f; // side to move has lost
}

// Root Dirichlet noise (identical to MCTS::inject_noise / self_play's helper).
void inject_dirichlet_noise(MCTSNode* root, float alpha, float epsilon) {
    if (root->children.empty()) return;

    static thread_local std::mt19937 rng(std::random_device{}());
    std::gamma_distribution<float> gamma(alpha, 1.0f);

    std::vector<float> noise;
    noise.reserve(root->children.size());
    float sum = 0.0f;
    for (size_t i = 0; i < root->children.size(); ++i) {
        float g = gamma(rng);
        noise.push_back(g);
        sum += g;
    }
    if (sum <= 0.0f) return;

    size_t idx = 0;
    for (auto& [action, child] : root->children) {
        float n = noise[idx++] / sum;
        child->prior = (1.0f - epsilon) * child->prior + epsilon * n;
    }
}

// Insert children into a node using precomputed legal actions (already
// encoded). Caller holds exclusive rights to the leaf via expand_state==1, so
// no synchronization is needed.
void insert_children(MCTSNode* node, const std::vector<int>& actions,
                     const float* policy) {
    if (actions.empty()) return;
    float prior_sum = 0.0f;
    for (int act : actions) prior_sum += policy[act];
    for (int act : actions) {
        float prior = (prior_sum > 0.0f) ? (policy[act] / prior_sum)
                                         : (1.0f / actions.size());
        node->children[act] = std::make_unique<MCTSNode>(prior);
    }
}

} // anonymous namespace

ParallelMCTS::ParallelMCTS(const Config& cfg)
    : c_puct_(cfg.c_puct),
      num_simulations_(cfg.num_simulations),
      dirichlet_alpha_(cfg.dirichlet_alpha),
      dirichlet_epsilon_(cfg.dirichlet_epsilon),
      virtual_loss_(cfg.virtual_loss),
      max_batch_(cfg.parallel_games > 0 ? cfg.parallel_games : 32) {}

// ── Public API ───────────────────────────────────────────────────────────
std::vector<float> ParallelMCTS::search(
    const Board& board,
    ChessNet& model,
    torch::Device device,
    int num_threads,
    bool add_noise
) {
    c10::DeviceGuard guard(device);
    torch::NoGradGuard no_grad;

    auto root = std::make_unique<MCTSNode>(0.0f);

    // Synchronous root expansion — one batched-size-1 forward, before workers
    // start, so the shared tree begins with a populated root.
    {
        EvalRequest req;
        req.leaf = root.get();
        req.board = std::make_shared<Board>(board);
        req.legal_moves = board.generate_legal_moves();
        req.actions.reserve(req.legal_moves.size());
        for (const auto& m : req.legal_moves) {
            req.actions.push_back(encode_action(m, board));
        }
        auto promise = std::make_shared<std::promise<EvalResult>>();
        auto future = promise->get_future();
        req.promise = promise;
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            queue_.push_back(std::make_unique<EvalRequest>(std::move(req)));
        }
        queue_cv_.notify_one();

        // Run the evaluator inline on this thread for the root only.
        std::vector<std::unique_ptr<EvalRequest>> batch;
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            batch.swap(queue_);
        }
        for (auto& r : batch) {
            // encode -> forward -> expand
            std::vector<float> planes(NUM_CHANNELS * 64);
            encode_board(*r->board, planes.data());
            auto options = torch::TensorOptions().dtype(torch::kFloat32);
            auto inp = torch::from_blob(planes.data(),
                {1, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE}, options)
                .clone().to(device);
            auto [logits, value_t] = model->forward(inp);
            auto probs = torch::softmax(logits, 1).squeeze(0).to(torch::kCPU);
            std::vector<float> pol(NUM_ACTIONS, 0.0f);
            std::memcpy(pol.data(), probs.data_ptr<float>(),
                        NUM_ACTIONS * sizeof(float));
            float v = value_t.item<float>();
            insert_children(r->leaf, r->actions, pol.data());
            r->leaf->expand_state.store(2, std::memory_order_release);
            EvalResult res{std::move(pol), v};
            r->promise->set_value(std::move(res));
        }
        future.get();
    }

    if (root->children.empty()) {
        return std::vector<float>(NUM_ACTIONS, 0.0f);
    }
    if (add_noise) {
        inject_dirichlet_noise(root.get(), dirichlet_alpha_, dirichlet_epsilon_);
    }
    root->visit_count = 1;

    // ── Launch evaluator + workers ───────────────────────────────────
    eval_done_ = false;
    std::atomic<int> sims_done{0};
    int total_sims = num_simulations_;
    int num_workers = std::max(1, num_threads);

    std::thread eval_thread(&ParallelMCTS::eval_thread_fn, this,
                            std::ref(model), device);

    std::vector<std::thread> workers;
    workers.reserve(num_workers);
    for (int w = 0; w < num_workers; ++w) {
        workers.emplace_back(&ParallelMCTS::worker_loop, this,
                             std::cref(board), root.get(),
                             std::ref(sims_done), total_sims);
    }

    for (auto& t : workers) t.join();

    // Signal the evaluator to drain and exit.
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        eval_done_ = true;
    }
    queue_cv_.notify_all();
    eval_thread.join();

    // Extract visit-count distribution (same contract as MCTS::search).
    std::vector<float> probs(NUM_ACTIONS, 0.0f);
    float total = 0.0f;
    for (const auto& [action, child] : root->children) {
        probs[action] = static_cast<float>(
            child->visit_count.load(std::memory_order_relaxed));
        total += probs[action];
    }
    if (total > 0.0f) {
        for (float& p : probs) p /= total;
    }
    return probs;
}

// ── PUCT select with virtual loss ────────────────────────────────────────
std::pair<int, MCTSNode*> ParallelMCTS::select_child_vl(MCTSNode* node) const {
    float sqrt_n = std::sqrt(static_cast<float>(
        node->visit_count.load(std::memory_order_relaxed)
        + node->virtual_loss.load(std::memory_order_relaxed)));

    float best_score = -1e9f;
    int best_action = -1;
    MCTSNode* best_child = nullptr;

    for (const auto& [action, child] : node->children) {
        int vc = child->visit_count.load(std::memory_order_relaxed);
        int vl = child->virtual_loss.load(std::memory_order_relaxed);

        // Q with virtual loss: pretend in-flight visits all lost (value -1).
        float n = static_cast<float>(vc + vl);
        float q = (n > 0.0f)
            ? -((child->value_sum.load(std::memory_order_relaxed)
                 - virtual_loss_ * vl) / n)
            : 0.0f;

        float u = c_puct_ * child->prior * sqrt_n / (1.0f + n);
        float score = q + u;
        if (score > best_score) {
            best_score = score;
            best_action = action;
            best_child = child.get();
        }
    }
    return {best_action, best_child};
}

// ── Backprop + release virtual loss ──────────────────────────────────────
void ParallelMCTS::backprop_and_release(std::vector<MCTSNode*>& path, float value) {
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        add_value_sum(*it, value);
        (*it)->visit_count.fetch_add(1, std::memory_order_relaxed);
        // Virtual loss is applied only to nodes descended INTO (children),
        // never to the path root — release it everywhere except index 0.
        if (*it != path.front()) {
            (*it)->virtual_loss.fetch_sub(1, std::memory_order_relaxed);
        }
        value = -value;
    }
}

// ── Worker loop ──────────────────────────────────────────────────────────
void ParallelMCTS::worker_loop(
    const Board& root_board,
    MCTSNode* root,
    std::atomic<int>& sims_done,
    int total_sims)
{
    while (true) {
        int i = sims_done.fetch_add(1, std::memory_order_relaxed);
        if (i >= total_sims) break;

        MCTSNode* node = root;
        Board scratch = root_board;
        std::vector<MCTSNode*> path;
        path.push_back(node);

        // SELECT — descend while the node is fully expanded (expand_state==2)
        // and the scratch game isn't over by draw rules. is_game_over_fast()
        // skips the expensive full check — a node with children is guaranteed
        // non-terminal anyway; the leaf gets the full check via legal_moves.
        while (node->expand_state.load(std::memory_order_acquire) == 2
               && !scratch.is_game_over_fast()) {
            auto [action, child] = select_child_vl(node);
            if (child == nullptr) break;
            child->virtual_loss.fetch_add(1, std::memory_order_relaxed);
            node = child;
            path.push_back(node);
            scratch.make_move(decode_action(action, scratch));
        }

        float value = 0.0f;
        if (scratch.is_game_over_fast()) {
            value = terminal_value(scratch);
        } else if (node->expand_state.load(std::memory_order_acquire) == 0) {
            // Unexpanded leaf: try to own the expansion.
            uint8_t expected = 0;
            if (node->expand_state.compare_exchange_strong(
                    expected, 1,
                    std::memory_order_acquire, std::memory_order_acquire)) {
                // We own the expansion — compute legal moves, actions, and
                // encoding once on this worker, then submit to the eval queue.
                auto req = std::make_unique<EvalRequest>();
                req->leaf = node;
                req->board = std::make_shared<Board>(scratch);
                req->legal_moves = scratch.generate_legal_moves();
                req->actions.reserve(req->legal_moves.size());
                for (const auto& m : req->legal_moves) {
                    req->actions.push_back(encode_action(m, scratch));
                }
                auto promise = std::make_shared<std::promise<EvalResult>>();
                auto future = promise->get_future();
                req->promise = promise;
                {
                    std::lock_guard<std::mutex> lock(queue_mutex_);
                    queue_.push_back(std::move(req));
                }
                queue_cv_.notify_one();
                EvalResult res = future.get();
                value = res.value;
            } else {
                // Someone else is expanding the same leaf: wait briefly then
                // this sim counts as done (a cheap, deadlock-free choice —
                // its sibling's result is about to land anyway).
                while (node->expand_state.load(std::memory_order_acquire) == 1) {
                    std::this_thread::yield();
                }
                // Leaf is now expanded by the sibling; count this sim with the
                // leaf's (just-published) value estimate.
                value = node->q_value();
            }
        } else {
            // expand_state == 2 but select loop ended (expanded node with no
            // selectable child — essentially unreachable). Value from this
            // node's own perspective matches the backprop convention.
            value = node->q_value();
        }

        backprop_and_release(path, value);
    }
}

// ── Eval queue pop (worker of the dedicated eval thread) ─────────────────
ParallelMCTS::EvalRequest* ParallelMCTS::try_pop_request() {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    if (queue_.empty()) {
        queue_cv_.wait(lock, [&] { return !queue_.empty() || eval_done_; });
    }
    if (queue_.empty()) return nullptr;
    auto req = std::move(queue_.front());
    queue_.erase(queue_.begin());
    auto* raw = req.release();
    return raw;
}

// ── Batched evaluator thread ─────────────────────────────────────────────
void ParallelMCTS::eval_thread_fn(ChessNet& model, torch::Device device) {
    c10::DeviceGuard guard(device);
    torch::NoGradGuard no_grad;
    std::vector<EvalRequest*> batch;
    batch.reserve(max_batch_);

    // Persistent pool shared across every batch in this eval thread — encode
    // and expansion are parallelized without per-batch thread spawn.
    int n_workers = std::min<int>(4, static_cast<int>(std::thread::hardware_concurrency()));
    ParallelRunner runner(n_workers > 0 ? n_workers : 4);

    while (true) {
        EvalRequest* r = try_pop_request();
        if (r == nullptr) {
            // Queue empty and eval_done_ was signalled — but drain any
            // stragglers that arrived between the last check and now.
            if (!batch.empty()) {
                // fallthrough to flush below
            } else {
                break;
            }
        } else {
            batch.push_back(r);
        }

        // Keep draining (non-blocking) until we fill the batch or run dry.
        while (static_cast<int>(batch.size()) < max_batch_) {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if (queue_.empty()) break;
            auto req2 = std::move(queue_.front());
            queue_.erase(queue_.begin());
            batch.push_back(req2.release());
        }

        if (batch.empty()) continue;

        // ── Encode all boards in the batch (persistent pool) ────────
        size_t n = batch.size();
        std::vector<float> planes(n * NUM_CHANNELS * BOARD_SIZE * BOARD_SIZE);
        runner.run(static_cast<int>(n), [&](int i) {
            encode_board(*batch[i]->board,
                         &planes[i * NUM_CHANNELS * BOARD_SIZE * BOARD_SIZE]);
        });

        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        auto tensor = torch::from_blob(planes.data(),
            {static_cast<int64_t>(n), NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE},
            options).to(device);

        auto [logits, values] = model->forward(tensor);
        auto probs = torch::softmax(logits, 1).to(torch::kCPU);
        auto vals = values.to(torch::kCPU);
        float* p_ptr = probs.data_ptr<float>();
        float* v_ptr = vals.data_ptr<float>();

        // ── Expand each leaf + fulfil its promise (persistent pool) ─
        runner.run(static_cast<int>(n), [&](int i) {
            auto* req = batch[i];
            float* leaf_policy = &p_ptr[i * NUM_ACTIONS];
            float val = v_ptr[i];

            insert_children(req->leaf, req->actions, leaf_policy);
            req->leaf->expand_state.store(2, std::memory_order_release);

            std::vector<float> pol(NUM_ACTIONS);
            std::memcpy(pol.data(), leaf_policy, NUM_ACTIONS * sizeof(float));
            req->promise->set_value(EvalResult{std::move(pol), val});
            delete req;
        });
        batch.clear();
    }
}

// ── expand_leaf (kept for API completeness) ──────────────────────────────
void ParallelMCTS::expand_leaf(MCTSNode* leaf, const Board& board,
                               const std::vector<float>& policy) {
    auto legal = board.generate_legal_moves();
    std::vector<int> actions;
    actions.reserve(legal.size());
    for (const auto& m : legal) actions.push_back(encode_action(m, board));
    insert_children(leaf, actions, policy.data());
    leaf->expand_state.store(2, std::memory_order_release);
}

} // namespace az
