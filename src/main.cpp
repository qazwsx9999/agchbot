#include "config.hpp"
#include "model.hpp"
#include "replay_buffer.hpp"
#include "trainer.hpp"
#include "self_play.hpp"
#include "evaluate.hpp"
#include "bots.hpp"
#include "action_encoding.hpp"
#include "mcts.hpp"
#include "parallel_mcts.hpp"

#include <c10/core/DeviceGuard.h>

#include <iostream>
#include <iomanip>
#include <filesystem>
#include <chrono>
#include <string>
#include <memory>
#include <algorithm>
#include <mutex>
#include <vector>

namespace fs = std::filesystem;

// Build the evaluation opponent from config (matches Python eval_opponent).
static std::unique_ptr<az::Bot> make_eval_opponent(const az::Config& cfg) {
    if (cfg.eval_opponent == "minimax") {
        return std::make_unique<az::MinimaxBot>(cfg.minimax_depth);
    }
    return std::make_unique<az::RandomBot>();
}

void run_interactive_play(az::ChessNet& model, const az::Config& cfg, torch::Device device) {
    model->eval();
    az::Board board;
    az::MCTS mcts(cfg);
    az::ParallelMCTS pmcts(cfg);

    std::cout << "\n========================================\n";
    std::cout << "   AlphaZero C++ Terminal Chess Match   \n";
    std::cout << "========================================\n";
    std::cout << "You are White. Enter moves in UCI (e.g. e2e4, g1f3, e7e8q).\n\n";

    while (!board.is_game_over()) {
        board.print();

        if (board.turn() == az::WHITE) {
            bool valid = false;
            while (!valid) {
                std::cout << "Your move (UCI): ";
                std::string uci;
                if (!(std::cin >> uci)) return;

                auto legals = board.generate_legal_moves();
                for (const auto& m : legals) {
                    if (m.to_uci() == uci) {
                        board.make_move(m);
                        valid = true;
                        break;
                    }
                }
                if (!valid) {
                    std::cout << "Illegal move! Try again.\n";
                }
            }
        } else {
            std::cout << "AI is thinking (MCTS " << cfg.num_simulations << " sims) ...\n";
            auto probs = (cfg.search_threads > 1)
                ? pmcts.search(board, model, device, cfg.search_threads, false)
                : mcts.search(board, model, device, false);
            int action = az::MCTS::select_action(probs, 0.0f);
            az::Move m = az::decode_action(action, board);
            std::cout << "AI plays: " << m.to_uci() << "\n";
            board.make_move(m);
        }
    }

    board.print();
    std::cout << "Game Over! Result: " << board.result() << "\n";
}

int main(int argc, char* argv[]) {
    // Unbuffered output so progress is visible even if the process dies or is
    // interrupted (also makes stdout/stderr appear immediately when piped).
    std::cout << std::unitbuf;
    std::cerr << std::unitbuf;

    az::Config cfg;
    bool resume = false;
    bool eval_only = false;
    bool interactive_play = false;

    // Command-line parsing
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--resume") {
            resume = true;
        } else if (arg == "--eval-only") {
            eval_only = true;
        } else if (arg == "--play") {
            interactive_play = true;
        } else if ((arg == "--iterations" || arg == "--num-iterations") && i + 1 < argc) {
            cfg.num_iterations = std::stoi(argv[++i]);
        } else if ((arg == "--games" || arg == "--num-self-play-games") && i + 1 < argc) {
            cfg.num_self_play_games = std::stoi(argv[++i]);
        } else if ((arg == "--simulations" || arg == "--num-simulations") && i + 1 < argc) {
            cfg.num_simulations = std::stoi(argv[++i]);
        } else if ((arg == "--eval-games" || arg == "--num-eval-games") && i + 1 < argc) {
            cfg.eval_games = std::stoi(argv[++i]);
        } else if (arg == "--eval-interval" && i + 1 < argc) {
            cfg.eval_interval = std::stoi(argv[++i]);
        } else if (arg == "--minimax-depth" && i + 1 < argc) {
            cfg.minimax_depth = std::stoi(argv[++i]);
        } else if (arg == "--device" && i + 1 < argc) {
            cfg.device = argv[++i];
        } else if ((arg == "--workers" || arg == "--num-workers") && i + 1 < argc) {
            cfg.num_workers = std::stoi(argv[++i]);
        } else if (arg == "--parallel-games" && i + 1 < argc) {
            cfg.parallel_games = std::stoi(argv[++i]);
        } else if (arg == "--c-puct" && i + 1 < argc) {
            cfg.c_puct = std::stof(argv[++i]);
        } else if (arg == "--learning-rate" && i + 1 < argc) {
            cfg.learning_rate = std::stof(argv[++i]);
        } else if (arg == "--num-epochs" && i + 1 < argc) {
            cfg.num_epochs = std::stoi(argv[++i]);
        } else if (arg == "--max-game-length" && i + 1 < argc) {
            cfg.max_game_length = std::stoi(argv[++i]);
        } else if (arg == "--batch-size" && i + 1 < argc) {
            cfg.batch_size = std::stoi(argv[++i]);
        } else if (arg == "--replay-buffer-size" && i + 1 < argc) {
            cfg.replay_buffer_size = std::stoi(argv[++i]);
        } else if (arg == "--min-replay-buffer-size" && i + 1 < argc) {
            cfg.min_replay_buffer_size = std::stoi(argv[++i]);
        } else if (arg == "--checkpoint-interval" && i + 1 < argc) {
            cfg.checkpoint_interval = std::stoi(argv[++i]);
        } else if (arg == "--eval-opponent" && i + 1 < argc) {
            cfg.eval_opponent = argv[++i];
        } else if (arg == "--checkpoint-dir" && i + 1 < argc) {
            cfg.checkpoint_dir = argv[++i];
        } else if (arg == "--batch-mcts") {
            cfg.batch_mcts = true;
        } else if ((arg == "--gpus" || arg == "--num-gpus") && i + 1 < argc) {
            cfg.num_gpus = std::stoi(argv[++i]);
        } else if (arg == "--select-workers" && i + 1 < argc) {
            cfg.select_workers = std::stoi(argv[++i]);
        } else if ((arg == "--search-threads" || arg == "--virtual-loss-threads") && i + 1 < argc) {
            cfg.search_threads = std::stoi(argv[++i]);
        } else if ((arg == "--virtual-loss" || arg == "--vloss") && i + 1 < argc) {
            cfg.virtual_loss = std::stof(argv[++i]);
        }
    }

    std::cout << "=========================================================\n";
    std::cout << "  AlphaZero Chess C++ Engine (High-Performance Engine)   \n";
    std::cout << "=========================================================\n";

    // Device selection ("auto" -> CUDA if available, else CPU; "cuda" falls back to CPU)
    torch::Device device(torch::kCPU);
    bool want_cuda = (cfg.device == "auto" || cfg.device == "cuda");
    if (want_cuda && torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
        cfg.device = "cuda";
        std::cout << "  [System] Using GPU (CUDA Acceleration Enabled)\n";
    } else {
        if (cfg.device == "cuda") {
            std::cout << "  [WARNING] CUDA not available - falling back to CPU.\n";
        }
        cfg.device = "cpu";
        int threads = (cfg.num_workers > 0) ? cfg.num_workers : static_cast<int>(std::thread::hardware_concurrency());
        if (threads <= 0) threads = 4;
        torch::set_num_threads(threads);
        std::cout << "  [System] Using CPU (" << threads << " threads configured for LibTorch)\n";
    }

    fs::create_directories(cfg.checkpoint_dir);
    std::string ckpt_path = cfg.checkpoint_dir + "/" + cfg.checkpoint_filename;
    std::string buffer_path = cfg.checkpoint_dir + "/replay_buffer.bin";

    az::ChessNet model(cfg);
    model->to(device);

    // Model copies for additional CUDA devices (multi-GPU self-play).
    // device_models[0] shares the canonical model handle (always in sync);
    // entries d>=1 are independent ChessNets that must be re-synced after
    // every load / training step.
    int n_devices = 1;
    std::vector<az::ChessNet> device_models;
    std::vector<torch::Device> devices;
    if (device.is_cuda()) {
        int avail = static_cast<int>(torch::cuda::device_count());
        n_devices = std::min(std::max(1, cfg.num_gpus), std::max(1, avail));
        for (int d = 0; d < n_devices; ++d) {
            devices.push_back(torch::Device(torch::kCUDA, d));
        }
        device_models.push_back(model); // shared handle with the trained model
        for (int d = 1; d < n_devices; ++d) {
            c10::DeviceGuard guard(devices[d]);
            az::ChessNet m(cfg);
            m->to(devices[d]);
            device_models.push_back(std::move(m));
        }
    } else {
        devices.push_back(device);
        device_models.push_back(model); // unused on the CPU path
    }

    // Copy the canonical model's params+buffers into the d>=1 copies.
    auto sync_device_models = [&]() {
        torch::NoGradGuard no_grad; // copy_ is in-place; must not track grads
        for (int d = 1; d < n_devices; ++d) {
            c10::DeviceGuard guard(devices[d]);
            auto src = model->named_parameters();
            auto dst = device_models[d]->named_parameters();
            for (auto it = src.begin(), jt = dst.begin();
                 it != src.end() && jt != dst.end(); ++it, ++jt) {
                jt->value().copy_(it->value().to(torch::kCPU));
            }
            auto src_b = model->named_buffers();
            auto dst_b = device_models[d]->named_buffers();
            for (auto it = src_b.begin(), jt = dst_b.begin();
                 it != src_b.end() && jt != dst_b.end(); ++it, ++jt) {
                jt->value().copy_(it->value().to(torch::kCPU));
            }
        }
    };
    sync_device_models();

    az::Trainer trainer(model, cfg, device);
    az::ReplayBuffer buffer(cfg.replay_buffer_size);
    az::SelfPlayEngine self_play(cfg, device);
    az::Evaluator evaluator(cfg, device);
    std::mutex sp_log_mutex;

    int start_iter = 1;
    int total_games = 0;

    if (resume || eval_only || interactive_play) {
        if (fs::exists(ckpt_path)) {
            std::cout << "  [System] Loading checkpoint from '" << ckpt_path << "' ...\n";
            start_iter = trainer.load_checkpoint(ckpt_path, total_games) + 1;
            model->to(device);
            if (fs::exists(buffer_path)) {
                buffer.load(buffer_path);
            }
            sync_device_models(); // push loaded weights to the GPU copies
            std::cout << "  [System] Resumed at iteration " << start_iter
                      << " (total games: " << total_games
                      << ", buffer: " << buffer.size() << " samples)\n";
        } else {
            std::cout << "  [System] No checkpoint found at '" << ckpt_path << "', starting fresh.\n";
        }
    }

    if (interactive_play) {
        run_interactive_play(model, cfg, device);
        return 0;
    }

    if (eval_only) {
        auto opponent = make_eval_opponent(cfg);
        evaluator.evaluate_vs_bot(model, *opponent, cfg.eval_games, cfg.num_simulations);
        return 0;
    }

    std::cout << "\n  Starting Training Loop (" << cfg.num_iterations << " iterations)\n";
    std::cout << "  ---------------------------------------------------------\n";

    for (int iter = start_iter; iter <= cfg.num_iterations; ++iter) {
        auto t0 = std::chrono::high_resolution_clock::now();
        std::cout << "\n=== Iteration " << iter << "/" << cfg.num_iterations << " ===\n";

        // 1. Self-Play (multi-GPU splits the games across devices)
        if (device.is_cuda() && n_devices > 1) {
            az::SelfPlayEngine::generate_games_multi_gpu(
                device_models, devices, cfg, buffer,
                cfg.num_self_play_games, sp_log_mutex);
        } else {
            self_play.generate_games(model, buffer, cfg.num_self_play_games);
        }
        total_games += cfg.num_self_play_games;

        // 2. Training (gated on a minimum buffer size, like Python)
        int min_req = std::min(cfg.min_replay_buffer_size, cfg.batch_size);
        if (static_cast<int>(buffer.size()) >= min_req) {
            std::cout << "  [Train] Training on " << buffer.size() << " samples ...\n";
            auto stats = trainer.train_step(buffer);
            sync_device_models(); // broadcast post-train weights to GPU copies
            std::cout << "  [Train] Loss: " << std::fixed << std::setprecision(4) << stats.total_loss
                      << " (policy: " << stats.policy_loss << ", value: " << stats.value_loss << ")\n";
        } else {
            std::cout << "  [Train] Skipped (buffer " << buffer.size()
                      << " < min " << min_req << ")\n";
        }

        // 3. Evaluation
        if (iter % cfg.eval_interval == 0) {
            auto opponent = make_eval_opponent(cfg);
            evaluator.evaluate_vs_bot(model, *opponent, cfg.eval_games, cfg.num_simulations);
        }

        // 4. Save Checkpoint (model + optimizer + replay buffer)
        if (iter % cfg.checkpoint_interval == 0) {
            trainer.save_checkpoint(ckpt_path, iter, total_games);
            buffer.save(buffer_path);
            std::cout << "  [Save] Checkpoint saved to '" << cfg.checkpoint_dir << "/'\n";
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed_sec = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "  Iteration " << iter << " completed in " << std::fixed
                  << std::setprecision(1) << elapsed_sec << "s\n";
    }

    std::cout << "\n  Training Complete! Checkpoint saved at '" << ckpt_path << "'\n";
    return 0;
}
