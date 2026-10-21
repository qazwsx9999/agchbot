#pragma once

#include "board.hpp"
#include "mcts.hpp"
#include "model.hpp"
#include "config.hpp"

#include <atomic>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <vector>

namespace az {

// Parallel Monte Carlo Tree Search for a SINGLE game.
//
// Unlike az::MCTS (which runs one simulation at a time on one thread),
// ParallelMCTS splits the num_simulations budget across a pool of worker
// threads that all share ONE search tree. Three mechanisms make the shared
// tree safe and productive:
//
//   * Atomics on every node (visit_count, value_sum) so selection, expansion
//     and backprop never corrupt each other.
//   * expand_state (0/1/2 CAS) so a worker inserting children can never race
//     a sibling iterating the same children map.
//   * Virtual loss — a small penalty applied to nodes along an in-flight
//     evaluation path and removed during backprop. It steers sibling workers
//     toward different leaves so the same branch isn't evaluated K times.
//
// Leaf evaluations are not performed inline: each worker submits its leaf to
// a shared queue and blocks on a promise. A dedicated evaluator thread drains
// the queue in batches, runs ONE batched model->forward per batch, expands
// the leaf nodes with the resulting priors, and fulfils the promises — so
// the GPU stays saturated while every CPU core keeps searching.
class ParallelMCTS {
public:
    explicit ParallelMCTS(const Config& cfg);

    // Run a parallel search and return the 4672-length visit-count
    // distribution (identical contract to MCTS::search).
    std::vector<float> search(
        const Board& board,
        ChessNet& model,
        torch::Device device,
        int num_threads,
        bool add_noise
    );

    void set_num_simulations(int n) { num_simulations_ = n; }
    int num_simulations() const { return num_simulations_; }
    float last_root_value() const { return last_root_value_; }

private:
    struct EvalResult {
        std::vector<float> policy; // full 4672 policy (already softmaxed, CPU)
        float value;
    };

    struct EvalRequest {
        MCTSNode* leaf;
        // The worker's scratch board at this leaf. Shared_ptr so the
        // evaluator can outlive the worker's frame safely.
        std::shared_ptr<Board> board;
        std::shared_ptr<std::promise<EvalResult>> promise;
        // Precomputed legal moves + their encoded action ids at this leaf —
        // computed once on the worker side, reused by the evaluator to insert
        // children without calling generate_legal_moves()/encode_action()
        // again on the eval thread.
        std::vector<Move> legal_moves;
        std::vector<int> actions;
    };

    // ── Worker-side helpers ────────────────────────────────────────
    std::pair<int, MCTSNode*> select_child_vl(MCTSNode* node) const;
    static void backprop_and_release(std::vector<MCTSNode*>& path, float value);
    void worker_loop(
        const Board& root_board,
        MCTSNode* root,
        std::atomic<int>& sims_done,
        int total_sims);

    // ── Evaluator-thread helpers ───────────────────────────────────
    EvalRequest* try_pop_request();
    void eval_thread_fn(ChessNet& model, torch::Device device);
    void expand_leaf(MCTSNode* leaf, const Board& board, const std::vector<float>& policy);

    // Config
    float c_puct_;
    int num_simulations_;
    float dirichlet_alpha_;
    float dirichlet_epsilon_;
    float virtual_loss_;
    int max_batch_;
    float last_root_value_ = 0.0f;

    // Shared eval queue (only used while a search() call is live).
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::vector<std::unique_ptr<EvalRequest>> queue_;
    bool eval_done_ = false;
};

} // namespace az
