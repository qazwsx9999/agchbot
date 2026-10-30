#include "self_play.hpp"
#include "action_encoding.hpp"
#include "mcts.hpp"
#include "parallel_mcts.hpp"

#include <c10/core/DeviceGuard.h>

#if __has_include(<ATen/autocast_mode.h>)
#include <ATen/autocast_mode.h>
#define HAS_AUTOCAST 1
#else
#define HAS_AUTOCAST 0
#endif

#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include <algorithm>
#include <thread>
#include <atomic>
#include <mutex>
#include <iterator>

namespace az {

namespace {

const char* SELF_PLAY_OPENINGS[] = {
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
    "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq d6 0 2",
    "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 1 2",
    "rnbqkbnr/pppp1ppp/8/4p3/2P5/8/PP1PPPPP/RNBQKBNR w KQkq e6 0 2",
    "rnbqkbnr/ppp1pppp/8/3p4/8/5N2/PPPPPPPP/RNBQKB1R w KQkq d6 0 2",
    "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",
    "rnbqkbnr/ppppp1pp/8/5p2/3P4/8/PPP1PPPP/RNBQKBNR w KQkq f6 0 2",
    "rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2"
};
constexpr size_t NUM_SP_OPENINGS = sizeof(SELF_PLAY_OPENINGS) / sizeof(SELF_PLAY_OPENINGS[0]);

void init_self_play_board(Board& board, int game_id) {
    // 75% standard starting position, 25% diverse opening book positions
    // so the network maintains accurate value/policy across all standard openings.
    if (game_id % 4 == 0) {
        size_t idx = static_cast<size_t>((game_id / 4) % NUM_SP_OPENINGS);
        board.set_fen(SELF_PLAY_OPENINGS[idx]);
    } else {
        board.reset();
    }
}

struct ActiveGame {
    int game_id;
    Board board;
    struct HistoryStep {
        std::vector<float> state;
        std::vector<float> policy;
        Color player;
    };
    std::vector<HistoryStep> history;
    int move_count = 0;
    std::unique_ptr<MCTSNode> root;

    explicit ActiveGame(int id) : game_id(id) {
        init_self_play_board(board, id);
    }
};

void inject_dirichlet_noise(MCTSNode* root, float alpha, float epsilon) {
    if (root->children.empty()) return;

    static thread_local std::mt19937 rng(std::random_device{}());
    std::gamma_distribution<float> gamma(alpha, 1.0f);

    std::vector<float> noise;
    noise.reserve(root->children.size());
    float sum_noise = 0.0f;

    for (size_t i = 0; i < root->children.size(); ++i) {
        float g = gamma(rng);
        noise.push_back(g);
        sum_noise += g;
    }

    if (sum_noise <= 0.0f) return;

    size_t idx = 0;
    for (auto& [action, child] : root->children) {
        float n = noise[idx++] / sum_noise;
        child->prior = (1.0f - epsilon) * child->prior + epsilon * n;
    }
}

std::pair<int, MCTSNode*> select_child_puct(MCTSNode* node, float c_puct) {
    int parent_visits = node->visit_count.load(std::memory_order_relaxed)
                      + node->virtual_loss.load(std::memory_order_relaxed);
    float sqrt_n = std::sqrt(static_cast<float>(std::max(1, parent_visits)));

    // FPU Reduction: estimate unvisited children relative to parent's Q-value
    float parent_q = node->q_value();
    constexpr float fpu_reduction = 0.25f;
    float fpu_value = std::max(-1.0f, parent_q - fpu_reduction);

    float best_score = -1e9f;
    int best_action = -1;
    MCTSNode* best_child = nullptr;

    for (const auto& [action, child] : node->children) {
        int vc = child->visit_count.load(std::memory_order_relaxed);
        int vl = child->virtual_loss.load(std::memory_order_relaxed);
        float n = static_cast<float>(vc + vl);

        float q = (n > 0.0f)
            ? -((child->value_sum.load(std::memory_order_relaxed) + static_cast<float>(vl)) / n)
            : fpu_value;
        float u = c_puct * child->prior * sqrt_n / (1.0f + n);
        float score = q + u;
        if (score > best_score) {
            best_score = score;
            best_action = action;
            best_child = child.get();
        }
    }

    return {best_action, best_child};
}

void backprop_path(std::vector<MCTSNode*>& path, float value) {
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        add_value_sum(*it, value);
        (*it)->visit_count += 1;
        value = -value;
    }
}

void apply_virtual_loss(const std::vector<MCTSNode*>& path) {
    for (size_t i = 1; i < path.size(); ++i) {
        path[i]->virtual_loss.fetch_add(1, std::memory_order_relaxed);
    }
}

void revert_virtual_loss(const std::vector<MCTSNode*>& path) {
    for (size_t i = 1; i < path.size(); ++i) {
        path[i]->virtual_loss.fetch_sub(1, std::memory_order_relaxed);
    }
}

void revert_virtual_loss_and_backprop(std::vector<MCTSNode*>& path, float val) {
    revert_virtual_loss(path);
    backprop_path(path, val);
}

struct LeafEval {
    MCTSNode* leaf_node = nullptr;
    std::vector<int> actions;
    std::vector<std::vector<MCTSNode*>> paths;
    float planes[NUM_CHANNELS * 64];
    int batch_idx = -1;
};

} // anonymous namespace

SelfPlayEngine::SelfPlayEngine(const Config& cfg, torch::Device device)
    : config_(cfg), device_(device) {}

std::vector<Sample> SelfPlayEngine::play_one_game(ChessNet& model, int& out_winner,
                                                  int search_threads) {
    // Single-game fallback using standard MCTS
    Board board;
    MCTS mcts(config_);
    ParallelMCTS pmcts(config_);
    if (search_threads < 0) search_threads = config_.search_threads;

    struct IntermediateStep {
        std::vector<float> state;
        std::vector<float> policy;
        Color player;
    };

    std::vector<IntermediateStep> trajectory;
    const int MAX_MOVES = config_.max_game_length;
    int move_count = 0;

    while (!board.is_game_over() && move_count < MAX_MOVES) {
        std::vector<float> state(NUM_CHANNELS * 64);
        encode_board(board, state.data());

        std::vector<float> policy = (search_threads > 1)
            ? pmcts.search(board, model, device_, search_threads, true)
            : mcts.search(board, model, device_, true);
        float temp = (move_count < config_.temperature_moves) ? 1.0f : 0.0f;
        int action = MCTS::select_action(policy, temp);

        trajectory.push_back({std::move(state), std::move(policy), board.turn()});

        Move m = decode_action(action, board);
        if (m.is_null() || !board.make_move(m)) break;
        move_count++;
    }

    std::string res = board.result();
    Color winner = NONE_COLOR;
    if (res == "1-0") {
        winner = WHITE;
        out_winner = 1;
    } else if (res == "0-1") {
        winner = BLACK;
        out_winner = -1;
    } else {
        winner = NONE_COLOR;
        out_winner = 0;
    }

    std::vector<Sample> samples;
    samples.reserve(trajectory.size());
    for (const auto& step : trajectory) {
        float val = 0.0f;
        if (winner != NONE_COLOR) {
            val = (step.player == winner) ? 1.0f : -1.0f;
        }
        samples.push_back({step.state, step.policy, val});
    }

    return samples;
}

SelfPlayStats SelfPlayEngine::generate_games_batched(ChessNet& model, ReplayBuffer& buffer, int num_games) {
    c10::DeviceGuard device_guard(device_);
    model->eval();
    torch::NoGradGuard no_grad;

    SelfPlayStats stats;
    int batch_size = std::min(config_.parallel_games, num_games);
    int games_started = 0;
    int total_moves = 0;

    std::vector<std::unique_ptr<ActiveGame>> active_games;
    active_games.reserve(batch_size);

    for (int i = 0; i < batch_size; ++i) {
        games_started++;
        active_games.push_back(std::make_unique<ActiveGame>(games_started));
    }

    std::cout << "  [Self-Play] Generating " << num_games << " games ("
              << active_games.size() << " batched parallel games on "
              << (device_.is_cuda() ? "CUDA" : "CPU") << ") ...\n";

    int sel_workers = config_.select_workers;
    if (sel_workers <= 0) {
        int hw = static_cast<int>(std::thread::hardware_concurrency());
        if (hw <= 0) hw = 4;
        sel_workers = std::max(1, hw / std::max(1, config_.num_gpus));
        sel_workers = std::min(sel_workers, batch_size);
    }
    ParallelRunner runner(sel_workers);

    constexpr int kVirtualLossBatch = 8;
    const size_t max_total_leaves = static_cast<size_t>(batch_size) * kVirtualLossBatch;
    std::vector<float> batch_planes(max_total_leaves * NUM_CHANNELS * BOARD_SIZE * BOARD_SIZE);
    std::vector<std::vector<LeafEval>> per_game_leaves(batch_size);
    for (auto& pgl : per_game_leaves) {
        pgl.reserve(kVirtualLossBatch);
    }

    while (!active_games.empty()) {
        // ─────────────────────────────────────────────────────────────
        // 1. Batched Root Expansion
        // ─────────────────────────────────────────────────────────────
        std::vector<ActiveGame*> need_root_expansion;
        for (auto& g : active_games) {
            if (!g->root) {
                need_root_expansion.push_back(g.get());
            }
        }

        if (!need_root_expansion.empty()) {
            size_t n_exp = need_root_expansion.size();
            for (size_t i = 0; i < n_exp; ++i) {
                encode_board(need_root_expansion[i]->board, &batch_planes[i * NUM_CHANNELS * 64]);
            }

            auto options = torch::TensorOptions().dtype(torch::kFloat32);
            auto tensor = torch::from_blob(batch_planes.data(),
                {static_cast<int64_t>(n_exp), NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE}, options)
                .to(device_);

#if HAS_AUTOCAST
            bool use_autocast = device_.is_cuda();
            if (use_autocast) {
                at::autocast::set_autocast_enabled(at::kCUDA, true);
            }
#endif
            auto [logits, values] = model->forward(tensor);
#if HAS_AUTOCAST
            if (use_autocast) {
                at::autocast::set_autocast_enabled(at::kCUDA, false);
            }
#endif
            auto probs = torch::softmax(logits.to(torch::kFloat32), 1).to(torch::kCPU);
            auto vals = values.to(torch::kFloat32).to(torch::kCPU);
            float* p_ptr = probs.data_ptr<float>();
            float* v_ptr = vals.data_ptr<float>();

            for (size_t i = 0; i < n_exp; ++i) {
                auto* g = need_root_expansion[i];
                g->root = std::make_unique<MCTSNode>(0.0f);
                float* game_policy = &p_ptr[i * NUM_ACTIONS];
                float root_val = v_ptr[i];

                auto legal_moves = g->board.generate_legal_moves();
                if (!legal_moves.empty()) {
                    std::vector<int> actions;
                    actions.reserve(legal_moves.size());
                    float prior_sum = 0.0f;

                    for (const auto& m : legal_moves) {
                        int act = encode_action(m, g->board);
                        actions.push_back(act);
                        prior_sum += game_policy[act];
                    }

                    for (int act : actions) {
                        float prior = (prior_sum > 0.0f) ? (game_policy[act] / prior_sum) : (1.0f / actions.size());
                        g->root->children[act] = std::make_unique<MCTSNode>(prior);
                    }

                    inject_dirichlet_noise(g->root.get(), config_.dirichlet_alpha, config_.dirichlet_epsilon);
                }
                add_value_sum(g->root.get(), root_val);
                g->root->visit_count = 1;
            }
        }

        // ─────────────────────────────────────────────────────────────
        // 2. Batched Parallel MCTS Simulations (Multi-Leaf Waves)
        // ─────────────────────────────────────────────────────────────
        for (int sim = 0; sim < config_.num_simulations; ) {
            int wave_sims = std::min(kVirtualLossBatch, config_.num_simulations - sim);
            sim += wave_sims;

            // Phase A: Select-down + virtual loss + inline board encoding per game
            runner.run(static_cast<int>(active_games.size()), [&](int gi) {
                per_game_leaves[gi].clear();
                auto& g = active_games[gi];
                if (!g->root || g->root->children.empty()) return;

                for (int s = 0; s < wave_sims; ++s) {
                    thread_local Board scratch;
                    scratch.copy_from(g->board);

                    MCTSNode* node = g->root.get();
                    std::vector<MCTSNode*> path;
                    path.push_back(node);

                    while (!node->children.empty()
                           && !(path.size() > 1 && (scratch.is_game_over_fast() || scratch.repetition_count() >= 1))) {
                        auto [action, child] = select_child_puct(node, config_.c_puct);
                        if (!child) break;
                        scratch.make_move(decode_action(action, scratch));
                        node = child;
                        path.push_back(node);
                    }

                    apply_virtual_loss(path);

                    if (path.size() > 1 && (scratch.is_game_over_fast() || scratch.repetition_count() >= 1)) {
                        revert_virtual_loss_and_backprop(path, 0.0f);
                        continue;
                    }

                    // Check if this leaf was already queued in this wave for this game
                    LeafEval* existing = nullptr;
                    for (auto& le : per_game_leaves[gi]) {
                        if (le.leaf_node == node) {
                            existing = &le;
                            break;
                         }
                    }

                    if (existing != nullptr) {
                        existing->paths.push_back(std::move(path));
                        continue;
                    }

                    if (node->children.empty()) {
                        auto legal = scratch.generate_legal_moves();
                        if (legal.empty()) {
                            // Terminal: checkmate with mate-distance bonus, stalemate -> 0.0f (draw)
                            int depth = std::max(1, static_cast<int>(path.size()) - 1);
                            float val = scratch.is_in_check() ? (-1.001f + 0.001f * static_cast<float>(depth)) : 0.0f;
                            revert_virtual_loss_and_backprop(path, val);
                            continue;
                        }

                        LeafEval le;
                        le.leaf_node = node;
                        le.actions.reserve(legal.size());
                        for (const auto& m : legal) {
                            le.actions.push_back(encode_action(m, scratch));
                        }
                        encode_board(scratch, le.planes);
                        le.paths.push_back(std::move(path));
                        per_game_leaves[gi].push_back(std::move(le));
                    } else {
                        revert_virtual_loss_and_backprop(path, node->q_value());
                    }
                }
            });

            // Flatten leaves across all active games into contiguous batch_planes
            int total_leaves = 0;
            for (size_t gi = 0; gi < active_games.size(); ++gi) {
                for (auto& le : per_game_leaves[gi]) {
                    le.batch_idx = total_leaves;
                    std::memcpy(
                        &batch_planes[total_leaves * NUM_CHANNELS * BOARD_SIZE * BOARD_SIZE],
                        le.planes,
                        NUM_CHANNELS * BOARD_SIZE * BOARD_SIZE * sizeof(float)
                    );
                    total_leaves++;
                }
            }

            if (total_leaves > 0) {
                auto options = torch::TensorOptions().dtype(torch::kFloat32);
                auto tensor = torch::from_blob(batch_planes.data(),
                    {static_cast<int64_t>(total_leaves), NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE}, options)
                    .to(device_);

#if HAS_AUTOCAST
                bool use_autocast = device_.is_cuda();
                if (use_autocast) {
                    at::autocast::set_autocast_enabled(at::kCUDA, true);
                }
#endif
                auto [logits, values] = model->forward(tensor);
#if HAS_AUTOCAST
                if (use_autocast) {
                    at::autocast::set_autocast_enabled(at::kCUDA, false);
                }
#endif

                auto probs = torch::softmax(logits.to(torch::kFloat32), 1).to(torch::kCPU);
                auto vals = values.to(torch::kFloat32).to(torch::kCPU);

                float* p_ptr = probs.data_ptr<float>();
                float* v_ptr = vals.data_ptr<float>();

                // Phase C: Expand leaves + revert virtual loss + backprop per game (partitioned by gi)
                runner.run(static_cast<int>(active_games.size()), [&](int gi) {
                    for (auto& leaf : per_game_leaves[gi]) {
                        int b_idx = leaf.batch_idx;
                        float* leaf_policy = &p_ptr[b_idx * NUM_ACTIONS];
                        float val = v_ptr[b_idx];

                        if (!leaf.actions.empty()) {
                            float prior_sum = 0.0f;
                            for (int act : leaf.actions) prior_sum += leaf_policy[act];
                            for (int act : leaf.actions) {
                                float prior = (prior_sum > 0.0f) ? (leaf_policy[act] / prior_sum) : (1.0f / leaf.actions.size());
                                leaf.leaf_node->children[act] = std::make_unique<MCTSNode>(prior);
                            }
                        }

                        for (auto& p : leaf.paths) {
                            revert_virtual_loss_and_backprop(p, val);
                        }
                    }
                });
            }
        }

        // ─────────────────────────────────────────────────────────────
        // 3. Make Moves & Advance Active Games
        // ─────────────────────────────────────────────────────────────
        std::vector<size_t> completed_indices;

        for (size_t i = 0; i < active_games.size(); ++i) {
            auto& g = active_games[i];

            std::vector<float> probs(NUM_ACTIONS, 0.0f);
            float total_visits = 0.0f;
            if (g->root) {
                for (const auto& [action, child] : g->root->children) {
                    probs[action] = static_cast<float>(child->visit_count);
                    total_visits += probs[action];
                }
            }

            if (total_visits > 0.0f) {
                for (float& p : probs) p /= total_visits;
            }

            float temp = (g->move_count < config_.temperature_moves) ? 1.0f : 0.0f;
            int action = MCTS::select_action(probs, temp);

            std::vector<float> state(NUM_CHANNELS * 64);
            encode_board(g->board, state.data());
            g->history.push_back({std::move(state), std::move(probs), g->board.turn()});

            Move m = decode_action(action, g->board);
            bool moved = !m.is_null() && g->board.make_move(m);
            g->move_count++;
            g->root.reset(); // Invalidate tree for next turn

            const int MAX_MOVES = config_.max_game_length;
            if (!moved || g->board.is_game_over() || g->move_count >= MAX_MOVES) {
                // Game finished!
                std::string res = g->board.result();
                Color winner = NONE_COLOR;
                std::string outcome_str;

                if (res == "1-0") {
                    winner = WHITE;
                    stats.white_wins++;
                    outcome_str = "White won";
                } else if (res == "0-1") {
                    winner = BLACK;
                    stats.black_wins++;
                    outcome_str = "Black won";
                } else {
                    winner = NONE_COLOR;
                    stats.draws++;
                    outcome_str = "Draw     ";
                }

                // Add to replay buffer
                std::vector<Sample> samples;
                samples.reserve(g->history.size());
                for (const auto& step : g->history) {
                    float val = 0.0f;
                    if (winner != NONE_COLOR) {
                        val = (step.player == winner) ? 1.0f : -1.0f;
                    }
                    samples.push_back({step.state, step.policy, val});
                }
                buffer.push_game(samples);
                total_moves += g->move_count;

                std::cout << "    [Self-Play] Game " << std::setw(3) << g->game_id << "/" << num_games
                          << " finished: " << outcome_str
                          << " (" << std::setw(3) << g->move_count << " moves)\n";

                // Recycle slot if more games need to start
                if (games_started < num_games) {
                    games_started++;
                    g = std::make_unique<ActiveGame>(games_started);
                } else {
                    completed_indices.push_back(i);
                }
            }
        }

        // Remove terminated games in reverse order
        for (auto it = completed_indices.rbegin(); it != completed_indices.rend(); ++it) {
            active_games.erase(active_games.begin() + *it);
        }
    }

    stats.total_samples = total_moves;
    stats.avg_moves = (num_games > 0) ? (static_cast<float>(total_moves) / num_games) : 0.0f;

    std::cout << "  [Self-Play] Summary: White: " << stats.white_wins
              << " | Black: " << stats.black_wins
              << " | Draws: " << stats.draws
              << " | Avg moves: " << std::fixed << std::setprecision(1) << stats.avg_moves << "\n";

    return stats;
}

SelfPlayStats SelfPlayEngine::generate_games_cpu(ChessNet& model, ReplayBuffer& buffer, int num_games) {
    model->eval();
    torch::NoGradGuard no_grad;

    int num_workers = config_.num_workers;
    if (num_workers <= 0) {
        num_workers = static_cast<int>(std::thread::hardware_concurrency());
        if (num_workers <= 0) num_workers = 4;
    }
    num_workers = std::min(num_workers, num_games);

    std::cout << "  [Self-Play] Generating " << num_games << " games with "
              << num_workers << " multi-threaded CPU workers ...\n";

    SelfPlayStats stats;
    std::atomic<int> next_game_id{1};
    std::atomic<int> completed_count{0};
    std::atomic<int> white_wins{0};
    std::atomic<int> black_wins{0};
    std::atomic<int> draws{0};
    std::atomic<int> total_moves{0};
    std::mutex log_mutex;

    std::vector<std::thread> workers;
    workers.reserve(num_workers);

    for (int w = 0; w < num_workers; ++w) {
        workers.emplace_back([&]() {
            torch::set_num_threads(1);
            while (true) {
                int g_id = next_game_id.fetch_add(1);
                if (g_id > num_games) break;

                int winner = 0;
                auto samples = play_one_game(model, winner);

                std::string outcome_str;
                if (winner == 1) {
                    white_wins.fetch_add(1);
                    outcome_str = "White won";
                } else if (winner == -1) {
                    black_wins.fetch_add(1);
                    outcome_str = "Black won";
                } else {
                    draws.fetch_add(1);
                    outcome_str = "Draw     ";
                }

                total_moves.fetch_add(static_cast<int>(samples.size()));
                buffer.push_game(samples);

                int done = completed_count.fetch_add(1) + 1;
                {
                    std::lock_guard<std::mutex> lock(log_mutex);
                    std::cout << "    [Self-Play] Game " << std::setw(3) << done << "/" << num_games
                              << " finished: " << outcome_str
                              << " (" << std::setw(3) << samples.size() << " moves)\n";
                }
            }
        });
    }

    for (auto& th : workers) {
        if (th.joinable()) th.join();
    }

    stats.white_wins = white_wins.load();
    stats.black_wins = black_wins.load();
    stats.draws = draws.load();
    stats.total_samples = total_moves.load();
    stats.avg_moves = (num_games > 0) ? (static_cast<float>(total_moves.load()) / num_games) : 0.0f;

    std::cout << "  [Self-Play] Summary: White: " << stats.white_wins
              << " | Black: " << stats.black_wins
              << " | Draws: " << stats.draws
              << " | Avg moves: " << std::fixed << std::setprecision(1) << stats.avg_moves << "\n";

    return stats;
}

SelfPlayStats SelfPlayEngine::generate_games(ChessNet& model, ReplayBuffer& buffer, int num_games) {
    if (device_.is_cuda() || config_.batch_mcts) {
        return generate_games_batched(model, buffer, num_games);
    } else {
        return generate_games_cpu(model, buffer, num_games);
    }
}

// ── Multi-GPU driver ─────────────────────────────────────────────────────
// Each device gets its own SelfPlayEngine + model copy, plays a share of the
// games on its own std::thread, then stats are merged. The shared
// ReplayBuffer is already mutex-guarded, and log output is serialized with
// log_mutex so per-GPU lines don't interleave.
SelfPlayStats SelfPlayEngine::generate_games_multi_gpu(
    std::vector<ChessNet>& models,
    const std::vector<torch::Device>& devices,
    const Config& cfg,
    ReplayBuffer& buffer,
    int num_games,
    std::mutex& log_mutex)
{
    SelfPlayStats total;
    const int n_dev = std::min(
        static_cast<int>(devices.size()),
        static_cast<int>(models.size()));
    if (n_dev <= 0) return total;

    if (n_dev == 1) {
        SelfPlayEngine eng(cfg, devices[0]);
        return eng.generate_games(models[0], buffer, num_games);
    }

    // Split games across devices (first device gets the remainder).
    int base = num_games / n_dev;
    int rem  = num_games % n_dev;

    std::vector<SelfPlayStats> per(n_dev);
    std::vector<std::thread> threads;
    threads.reserve(n_dev);

    Config dev_cfg = cfg;
    if (cfg.select_workers > 0) {
        dev_cfg.select_workers = std::max(1, cfg.select_workers / n_dev);
    }

    for (int d = 0; d < n_dev; ++d) {
        int share = base + (d == 0 ? rem : 0);
        threads.emplace_back([&, d, share]() {
            c10::DeviceGuard guard(devices[d]);
            SelfPlayEngine eng(dev_cfg, devices[d]);
            per[d] = eng.generate_games(models[d], buffer, share);
            std::lock_guard<std::mutex> lock(log_mutex);
            std::cout << "    [GPU" << d << "] done: W" << per[d].white_wins
                      << " B" << per[d].black_wins
                      << " D" << per[d].draws << "\n";
        });
    }
    for (auto& t : threads) t.join();

    int moves_sum = 0;
    for (int d = 0; d < n_dev; ++d) {
        total.white_wins    += per[d].white_wins;
        total.black_wins    += per[d].black_wins;
        total.draws         += per[d].draws;
        total.total_samples += per[d].total_samples;
        int share = base + (d == 0 ? rem : 0);
        moves_sum += static_cast<int>(per[d].avg_moves * share + 0.5f);
    }
    total.avg_moves = (num_games > 0)
        ? (static_cast<float>(moves_sum) / num_games) : 0.0f;

    return total;
}

} // namespace az
