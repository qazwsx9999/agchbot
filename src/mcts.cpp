#include "mcts.hpp"
#include "action_encoding.hpp"
#include <cmath>
#include <random>
#include <numeric>
#include <algorithm>

namespace az {

MCTS::MCTS(const Config& cfg)
    : c_puct_(cfg.c_puct),
      num_simulations_(cfg.num_simulations),
      dirichlet_alpha_(cfg.dirichlet_alpha),
      dirichlet_epsilon_(cfg.dirichlet_epsilon) {}

std::vector<float> MCTS::search(
    const Board& board,
    ChessNet& model,
    torch::Device device,
    bool add_noise
) {
    torch::NoGradGuard no_grad;
    auto root = std::make_unique<MCTSNode>(0.0f);

    // Initial root expansion
    float root_value = expand(root.get(), board, board.generate_legal_moves(), model, device);
    if (root->children.empty()) {
        return std::vector<float>(NUM_ACTIONS, 0.0f);
    }

    if (add_noise) {
        inject_noise(root.get());
    }

    add_value_sum(root.get(), root_value);
    root->visit_count = 1;

    for (int sim = 0; sim < num_simulations_; ++sim) {
        MCTSNode* node = root.get();
        Board scratch = board;
        std::vector<MCTSNode*> path;
        path.push_back(node);

        // SELECT: Traverse down the tree. Any position repeated inside the
        // search tree (path.size() > 1 && repetition_count() >= 1) is treated
        // as a 0.0 draw so the bot never shuffles its king back and forth when winning.
        while (!node->children.empty()
               && !(path.size() > 1 && (scratch.is_game_over_fast() || scratch.repetition_count() >= 1))) {
            auto [action, child] = select_child(node);
            Move m = decode_action(action, scratch);
            scratch.make_move(m);
            node = child;
            path.push_back(node);
        }

        float value = 0.0f;
        if (path.size() > 1 && (scratch.is_game_over_fast() || scratch.repetition_count() >= 1)) {
            value = 0.0f;
        } else if (node->children.empty()) {
            auto legal = scratch.generate_legal_moves();
            if (legal.empty()) {
                int depth = std::max(1, static_cast<int>(path.size()) - 1);
                value = scratch.is_in_check() ? (-1.001f + 0.001f * static_cast<float>(depth)) : 0.0f;
            } else {
                value = expand(node, scratch, legal, model, device);
            }
        }

        // BACKPROPAGATE
        backprop(path, value);
    }

    last_root_value_ = root->q_value();

    // Extract visit-count distribution
    std::vector<float> probs(NUM_ACTIONS, 0.0f);
    float total_visits = 0.0f;
    for (const auto& [action, child] : root->children) {
        probs[action] = static_cast<float>(child->visit_count);
        total_visits += probs[action];
    }

    if (total_visits > 0.0f) {
        for (float& p : probs) {
            p /= total_visits;
        }
    }

    return probs;
}

float MCTS::expand(MCTSNode* node, const Board& board,
                   const std::vector<Move>& legal_moves,
                   ChessNet& model, torch::Device device) {
    std::vector<float> planes(NUM_CHANNELS * 64);
    encode_board(board, planes.data());

    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    auto inp = torch::from_blob(planes.data(), {1, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE}, options)
                   .clone()
                   .to(device);

    auto [policy_logits, value_t] = model->forward(inp);
    auto policy = torch::softmax(policy_logits, 1).squeeze(0).to(torch::kCPU);
    float* p_ptr = policy.data_ptr<float>();
    float value = value_t.item<float>();

    if (legal_moves.empty()) {
        return value;
    }

    std::vector<int> actions;
    actions.reserve(legal_moves.size());
    float prior_sum = 0.0f;

    for (const auto& m : legal_moves) {
        int act = encode_action(m, board);
        actions.push_back(act);
        prior_sum += p_ptr[act];
    }

    for (int act : actions) {
        float prior = (prior_sum > 0.0f) ? (p_ptr[act] / prior_sum) : (1.0f / actions.size());
        node->children[act] = std::make_unique<MCTSNode>(prior);
    }

    return value;
}

std::pair<int, MCTSNode*> MCTS::select_child(MCTSNode* node) {
    int parent_visits = node->visit_count.load(std::memory_order_relaxed);
    float sqrt_n = std::sqrt(static_cast<float>(std::max(1, parent_visits)));

    // FPU Reduction: estimate unvisited children relative to parent's Q-value
    float parent_q = node->q_value();
    constexpr float fpu_reduction = 0.25f;
    float fpu_value = std::max(-1.0f, parent_q - fpu_reduction);

    float best_score = -1e9f;
    int best_action = -1;
    MCTSNode* best_child = nullptr;

    for (const auto& [action, child] : node->children) {
        int child_visits = child->visit_count.load(std::memory_order_relaxed);
        float q = (child_visits > 0) ? (-child->q_value()) : fpu_value;
        float u = c_puct_ * child->prior * sqrt_n / (1.0f + static_cast<float>(child_visits));
        float score = q + u;
        if (score > best_score) {
            best_score = score;
            best_action = action;
            best_child = child.get();
        }
    }

    return {best_action, best_child};
}

float MCTS::terminal_value(const Board& board) {
    std::string res = board.result();
    if (res == "1/2-1/2") return 0.0f;
    // Side to move has lost
    return -1.0f;
}

void MCTS::backprop(std::vector<MCTSNode*>& path, float value) {
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        add_value_sum(*it, value);
        (*it)->visit_count += 1;
        value = -value;
    }
}

void MCTS::inject_noise(MCTSNode* root) {
    if (root->children.empty()) return;

    static thread_local std::mt19937 rng(std::random_device{}());
    std::gamma_distribution<float> gamma(dirichlet_alpha_, 1.0f);

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
        child->prior = (1.0f - dirichlet_epsilon_) * child->prior + dirichlet_epsilon_ * n;
    }
}

int MCTS::select_action(const std::vector<float>& probs, float temperature) {
    if (temperature < 1e-3f) {
        // Greedy argmax
        return static_cast<int>(std::distance(probs.begin(), std::max_element(probs.begin(), probs.end())));
    }

    std::vector<float> adjusted(probs.size(), 0.0f);
    float sum = 0.0f;
    for (size_t i = 0; i < probs.size(); ++i) {
        if (probs[i] > 0.0f) {
            adjusted[i] = std::pow(probs[i], 1.0f / temperature);
            sum += adjusted[i];
        }
    }

    if (sum <= 0.0f) {
        return static_cast<int>(std::distance(probs.begin(), std::max_element(probs.begin(), probs.end())));
    }

    static thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, sum);
    float r = dist(rng);
    float cumulative = 0.0f;

    for (size_t i = 0; i < adjusted.size(); ++i) {
        cumulative += adjusted[i];
        if (r <= cumulative) return static_cast<int>(i);
    }

    return static_cast<int>(adjusted.size() - 1);
}

} // namespace az
