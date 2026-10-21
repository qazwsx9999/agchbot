#pragma once

#include "board.hpp"
#include "model.hpp"
#include "config.hpp"
#include <memory>
#include <unordered_map>
#include <vector>
#include <atomic>

namespace az {

struct MCTSNode {
    std::atomic<int> visit_count{0};
    std::atomic<float> value_sum{0.0f};
    float prior = 0.0f;
    // Virtual loss: number of in-flight (being-evaluated) visits on this node.
    // Used by ParallelMCTS to discourage sibling worker threads from picking
    // the same leaf. Removed again during backprop — leaves no residue on
    // the final visit_count.
    std::atomic<int> virtual_loss{0};
    // Expansion state for concurrent in-tree search (ParallelMCTS):
    //   0 = unexpanded, 1 = a worker is expanding it now, 2 = fully expanded.
    // Selectors only read `children` when this is 2 (release/acquire), so a
    // worker inserting children can never race a sibling iterating the map.
    std::atomic<uint8_t> expand_state{0};
    std::unordered_map<int, std::unique_ptr<MCTSNode>> children;

    explicit MCTSNode(float p = 0.0f) : prior(p) {}

    float q_value() const {
        int vc = visit_count.load(std::memory_order_relaxed);
        return (vc == 0) ? 0.0f
                        : (value_sum.load(std::memory_order_relaxed)
                           / static_cast<float>(vc));
    }
};

// C++17-compatible atomic float add (fetch_add on float atomics is C++20).
// Only ParallelMCTS shares a node across threads; sequential paths are
// equally correct through the same helper.
inline void add_value_sum(MCTSNode* node, float delta) {
    float old = node->value_sum.load(std::memory_order_relaxed);
    while (!node->value_sum.compare_exchange_weak(
        old, old + delta,
        std::memory_order_relaxed, std::memory_order_relaxed)) {
    }
}

class MCTS {
public:
    explicit MCTS(const Config& cfg);

    // Search and return a 4672-length visit-count distribution
    std::vector<float> search(
        const Board& board,
        ChessNet& model,
        torch::Device device,
        bool add_noise = true
    );

    // Pick action using temperature or argmax
    static int select_action(const std::vector<float>& probs, float temperature);

    void set_num_simulations(int n) { num_simulations_ = n; }
    int num_simulations() const { return num_simulations_; }
    float last_root_value() const { return last_root_value_; }

private:
    float c_puct_;
    int num_simulations_;
    float dirichlet_alpha_;
    float dirichlet_epsilon_;
    float last_root_value_ = 0.0f;

    float expand(MCTSNode* node, const Board& board,
                 const std::vector<Move>& legal_moves,
                 ChessNet& model, torch::Device device);
    std::pair<int, MCTSNode*> select_child(MCTSNode* node);
    static float terminal_value(const Board& board);
    static void backprop(std::vector<MCTSNode*>& path, float value);
    void inject_noise(MCTSNode* root);
};

} // namespace az
