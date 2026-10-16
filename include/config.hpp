#pragma once

#include <string>
#include <thread>

namespace az {

// All defaults mirror the Python config.py so the C++ engine reproduces
// the exact same training behaviour (just faster).
struct Config {
    // ── Model architecture ──────────────────────────────────────────
    int num_res_blocks = 10;
    int num_filters = 128;

    // ── MCTS ────────────────────────────────────────────────────────
    int num_simulations = 800;
    float c_puct = 1.5f;
    float dirichlet_alpha = 0.3f;
    float dirichlet_epsilon = 0.25f;
    float temperature = 1.0f;
    int temperature_moves = 30;      // == Python temperature_threshold

    // ── Self-play ───────────────────────────────────────────────────
    int num_self_play_games = 100;
    int num_workers = 0;             // 0 = auto-detect CPU cores
    int max_game_length = 512;
    int parallel_games = 32;         // batched self-play width (games evaluated together)
    bool batch_mcts = false;         // true = use batched MCTS on CPU; false = worker threads
    int select_workers = 0;          // threads splitting batched select-down (0 = auto: min(parallel_games, hw))
    int num_gpus = 1;                // CUDA devices to drive in self-play (1 = single GPU)

    // ── Parallel single-tree search (ParallelMCTS) ──────────────────
    int search_threads = 4;          // threads for a single shared-tree search (eval/play)
    float virtual_loss = 1.0f;       // virtual-loss penalty magnitude on in-flight nodes

    // ── Training ────────────────────────────────────────────────────
    int batch_size = 256;
    float learning_rate = 0.01f;
    float weight_decay = 1e-4f;
    float momentum = 0.9f;
    int num_epochs = 1;              // epochs over the buffer per iteration
    int replay_buffer_size = 50000;
    int min_replay_buffer_size = 2048;

    // ── Training loop ───────────────────────────────────────────────
    int num_iterations = 100;
    int checkpoint_interval = 1;     // save every N iterations

    // ── Evaluation ──────────────────────────────────────────────────
    std::string eval_opponent = "random";  // "random" or "minimax"
    int minimax_depth = 3;
    int eval_interval = 10;          // evaluate every N iterations
    int eval_games = 20;             // == Python num_eval_games

    // ── Paths & device ──────────────────────────────────────────────
    std::string checkpoint_dir = "checkpoints";
    std::string checkpoint_filename = "checkpoint.pt";
    std::string device = "auto";     // "auto", "cuda", or "cpu"
};

} // namespace az
