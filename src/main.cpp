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

void run_uci(az::ChessNet& model, const az::Config& cfg, torch::Device device) {
    model->eval();
    az::Board board;
    az::ParallelMCTS pmcts(cfg);
    az::MCTS mcts(cfg);

    int current_sims = (cfg.num_simulations > 0) ? cfg.num_simulations : 800;
    int search_threads = (cfg.search_threads > 1) ? cfg.search_threads : 4;

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::string cmd;
        if (!(iss >> cmd)) continue;

        if (cmd == "uci") {
            std::cout << "id name AlphaZero Chess C++\n";
            std::cout << "id author AlphaZero Team\n";
            std::cout << "option name Threads type spin default " << search_threads << " min 1 max 64\n";
            std::cout << "option name Simulations type spin default " << current_sims << " min 10 max 100000\n";
            std::cout << "uciok\n" << std::flush;
        } else if (cmd == "isready") {
            std::cout << "readyok\n" << std::flush;
        } else if (cmd == "ucinewgame") {
            board.reset();
        } else if (cmd == "setoption") {
            std::string name_token, opt_name, val_token;
            iss >> name_token >> opt_name;
            if (iss >> val_token) {
                if (opt_name == "Threads") {
                    int t;
                    if (iss >> t) search_threads = std::max(1, t);
                } else if (opt_name == "Simulations") {
                    int s;
                    if (iss >> s) current_sims = std::max(1, s);
                }
            }
        } else if (cmd == "position") {
            std::string pos_type;
            iss >> pos_type;
            if (pos_type == "startpos") {
                board.reset();
                std::string maybe_moves;
                if (iss >> maybe_moves && maybe_moves == "moves") {
                    std::string m_str;
                    while (iss >> m_str) {
                        auto legals = board.generate_legal_moves();
                        for (const auto& lm : legals) {
                            if (lm.to_uci() == m_str) {
                                board.make_move(lm);
                                break;
                            }
                        }
                    }
                }
            } else if (pos_type == "fen") {
                std::string fen;
                std::string token;
                std::vector<std::string> fen_parts;
                while (iss >> token && token != "moves" && fen_parts.size() < 6) {
                    fen_parts.push_back(token);
                }
                for (size_t p = 0; p < fen_parts.size(); ++p) {
                    if (p > 0) fen += " ";
                    fen += fen_parts[p];
                }
                board.set_fen(fen);
                if (token == "moves") {
                    std::string m_str;
                    while (iss >> m_str) {
                        auto legals = board.generate_legal_moves();
                        for (const auto& lm : legals) {
                            if (lm.to_uci() == m_str) {
                                board.make_move(lm);
                                break;
                            }
                        }
                    }
                }
            }
        } else if (cmd == "go") {
            std::string sub;
            int sims = current_sims;
            while (iss >> sub) {
                if (sub == "nodes" || sub == "sims" || sub == "simulations") {
                    int n;
                    if (iss >> n) sims = std::max(1, n);
                }
            }

            pmcts.set_num_simulations(sims);
            mcts.set_num_simulations(sims);

            std::vector<float> probs;
            float root_q = 0.0f;
            if (search_threads > 1) {
                probs = pmcts.search(board, model, device, search_threads, false);
                root_q = pmcts.last_root_value();
            } else {
                probs = mcts.search(board, model, device, false);
                root_q = mcts.last_root_value();
            }

            int action = az::MCTS::select_action(probs, 0.0f);
            az::Move best = az::decode_action(action, board);

            float winrate = (root_q + 1.0f) * 0.5f;
            int cp = static_cast<int>(root_q * 1000.0f);

            std::cout << "info depth 10 nodes " << sims
                      << " score cp " << cp
                      << " winrate " << std::fixed << std::setprecision(4) << winrate
                      << " pv " << best.to_uci() << "\n";
            std::cout << "bestmove " << best.to_uci() << "\n" << std::flush;
        } else if (cmd == "quit") {
            break;
        }
    }
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
    bool uci_mode = false;
    std::string custom_ckpt_path = "";

    // Command-line parsing
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--resume") {
            resume = true;
        } else if (arg == "--eval-only") {
            eval_only = true;
        } else if (arg == "--play") {
            interactive_play = true;
        } else if (arg == "--uci") {
            uci_mode = true;
        } else if ((arg == "--checkpoint" || arg == "--ckpt") && i + 1 < argc) {
            custom_ckpt_path = argv[++i];
        } else if ((arg == "--iterations" || arg == "--num-iterations") && i + 1 < argc) {
            cfg.num_iterations = std::stoi(argv[++i]);
        } else if ((arg == "--games" || arg == "--num-self-play-games") && i + 1 < argc) {
            cfg.num_self_play_games = std::stoi(argv[++i]);
        } else if ((arg == "--simulations" || arg == "--num-simulations") && i + 1 < argc) {
            cfg.num_simulations = std::stoi(argv[++i]);
        } else if ((arg == "--eval-sims" || arg == "--eval-simulations") && i + 1 < argc) {
            cfg.eval_simulations = std::stoi(argv[++i]);
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

    if (!uci_mode) {
        std::cout << "=========================================================\n";
        std::cout << "  AlphaZero Chess C++ Engine (High-Performance Engine)   \n";
        std::cout << "=========================================================\n";
    }

    // Device selection ("auto" -> CUDA if available, else CPU; "cuda" falls back to CPU)
    torch::Device device(torch::kCPU);
    bool want_cuda = (cfg.device == "auto" || cfg.device == "cuda");
    if (want_cuda && torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
        cfg.device = "cuda";
        torch::set_num_threads(1);
        (uci_mode ? std::cerr : std::cout) << "  [System] Using GPU (CUDA Acceleration Enabled, LibTorch threads=1)\n";
    } else {
        if (cfg.device == "cuda") {
            (uci_mode ? std::cerr : std::cout) << "  [WARNING] CUDA not available - falling back to CPU.\n";
        }
        cfg.device = "cpu";
        int threads = (cfg.num_workers > 0) ? cfg.num_workers : static_cast<int>(std::thread::hardware_concurrency());
        if (threads <= 0) threads = 4;
        if (cfg.select_workers > 1 || cfg.search_threads > 1) {
            torch::set_num_threads(1);
        } else {
            torch::set_num_threads(threads);
        }
        (uci_mode ? std::cerr : std::cout) << "  [System] Using CPU (" << threads << " worker threads configured)\n";
    }

    fs::create_directories(cfg.checkpoint_dir);
    std::string ckpt_path = custom_ckpt_path;
    if (ckpt_path.empty()) {
        std::vector<std::string> candidates = {
            cfg.checkpoint_dir + "/" + cfg.checkpoint_filename,
            "run_ckpt/checkpoint.pt",
            "checkpoints/checkpoint.pt",
            "model.pt"
        };
        for (const auto& c : candidates) {
            if (fs::exists(c)) {
                ckpt_path = c;
                break;
            }
        }
        if (ckpt_path.empty()) {
            ckpt_path = cfg.checkpoint_dir + "/" + cfg.checkpoint_filename;
        }
    }
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

    if (resume || eval_only || interactive_play || uci_mode) {
        if (fs::exists(ckpt_path)) {
            (uci_mode ? std::cerr : std::cout) << "  [System] Loading checkpoint from '" << ckpt_path << "' ...\n";
            start_iter = trainer.load_checkpoint(ckpt_path, total_games) + 1;
            model->to(device);
            if (fs::exists(buffer_path)) {
                buffer.load(buffer_path);
            }
            sync_device_models(); // push loaded weights to the GPU copies
            (uci_mode ? std::cerr : std::cout) << "  [System] Resumed at iteration " << start_iter
                      << " (total games: " << total_games
                      << ", buffer: " << buffer.size() << " samples)\n";
        } else {
            (uci_mode ? std::cerr : std::cout) << "  [System] No checkpoint found at '" << ckpt_path << "', starting fresh.\n";
        }
    }

    if (uci_mode) {
        run_uci(model, cfg, device);
        return 0;
    }

    if (interactive_play) {
        run_interactive_play(model, cfg, device);
        return 0;
    }

    if (eval_only) {
        auto opponent = make_eval_opponent(cfg);
        int eval_sims = (cfg.eval_simulations > 0) ? cfg.eval_simulations : cfg.num_simulations;
        evaluator.evaluate_vs_bot(model, *opponent, cfg.eval_games, eval_sims);
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
            int eval_sims = (cfg.eval_simulations > 0) ? cfg.eval_simulations : cfg.num_simulations;
            evaluator.evaluate_vs_bot(model, *opponent, cfg.eval_games, eval_sims);
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
