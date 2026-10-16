#pragma once

#include "model.hpp"
#include "replay_buffer.hpp"
#include "config.hpp"
#include <vector>
#include <functional>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <algorithm>

namespace az {

struct SelfPlayStats {
    int white_wins = 0;
    int black_wins = 0;
    int draws = 0;
    float avg_moves = 0.0f;
};

// Persistent worker pool: threads are spawned ONCE and kept alive for the
// lifetime of the pool. `run(count, fn)` dispatches `count` job indices
// across them and blocks until all are completed — eliminating the
// per-simulation thread-spawn cost entirely (was ~16k creates per move-batch
// in batched self-play).
//
// Epoch design: every `run` bumps `epoch_`; workers wait for an epoch change,
// claim indices atomically, and report exit. `run` waits for ALL jobs to be
// completed AND every worker to have exited the inner loop before returning —
// that wait is what makes stale-index execution impossible between epochs.
class ParallelRunner {
public:
    explicit ParallelRunner(int num_threads)
        : n_(std::max(1, num_threads)) {
        workers_.reserve(n_);
        for (int i = 0; i < n_; ++i)
            workers_.emplace_back([this]() { worker_loop(); });
    }
    ~ParallelRunner() {
        { std::lock_guard<std::mutex> l(m_); stop_ = true; }
        cv_job_.notify_all();
        for (auto& t : workers_) if (t.joinable()) t.join();
    }
    ParallelRunner(const ParallelRunner&) = delete;
    ParallelRunner& operator=(const ParallelRunner&) = delete;

    // Run `job_count` invocations of `fn(i)` across the pool, wait for all.
    // Serializes with respect to itself — callers must not nest run() calls.
    template <typename F>
    void run(int job_count, F&& fn) {
        if (n_ <= 1 || job_count <= 0) {
            for (int i = 0; i < job_count; ++i) fn(i);
            return;
        }
        {
            std::unique_lock<std::mutex> l(m_);
            cv_idle_.wait(l, [this] { return idle_workers_ == n_; });
            job_ = std::forward<F>(fn);
            next_.store(0, std::memory_order_relaxed);
            completed_.store(0, std::memory_order_relaxed);
            job_count_ = job_count;
            idle_workers_ = 0;
            ++epoch_;
        }
        cv_job_.notify_all();
        {
            std::unique_lock<std::mutex> l(m_);
            cv_done_.wait(l, [this] {
                return completed_.load(std::memory_order_relaxed) >= job_count_
                    && idle_workers_ == n_;
            });
            job_ = nullptr;
        }
    }

private:
    void worker_loop() {
        int seen_epoch = 0;
        for (;;) {
            std::function<void(int)> job;
            {
                std::unique_lock<std::mutex> l(m_);
                ++idle_workers_;
                cv_idle_.notify_one();
                cv_done_.notify_one();
                cv_job_.wait(l, [this, seen_epoch] { return stop_ || epoch_ != seen_epoch; });
                if (stop_) return;
                seen_epoch = epoch_;
                job = job_;
            }
            // Claim indices atomically; `job_count_` can not change until every
            // worker exits this loop (the run() wait on exited_), so reading it
            // inside the loop is consistent.
            while (true) {
                int i = next_.fetch_add(1, std::memory_order_relaxed);
                if (i >= job_count_) break;
                job(i);
                completed_.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }

    int n_;
    std::vector<std::thread> workers_;
    std::mutex m_;
    std::condition_variable cv_job_, cv_done_, cv_idle_;
    std::function<void(int)> job_;
    std::atomic<int> next_{0}, completed_{0};
    int job_count_{0};
    int idle_workers_{0};
    int epoch_{0};
    bool stop_{false};
};

class SelfPlayEngine {
public:
    SelfPlayEngine(const Config& cfg, torch::Device device);

    // Play one complete game and return the collected samples.
    // search_threads: threads for the shared-tree ParallelMCTS
    // (-1 = use config_.search_threads; 1 = plain sequential MCTS).
    std::vector<Sample> play_one_game(ChessNet& model, int& out_winner,
                                      int search_threads = -1);

    // Play num_games and push all samples into buffer
    SelfPlayStats generate_games(ChessNet& model, ReplayBuffer& buffer, int num_games);

    // Multi-GPU helper: run num_games across several device-bound engine+model
    // pairs in parallel, merge the stats. Each element owns its own ChessNet
    // already placed on that device.
    static SelfPlayStats generate_games_multi_gpu(
        std::vector<ChessNet>& models,
        const std::vector<torch::Device>& devices,
        const Config& cfg,
        ReplayBuffer& buffer,
        int num_games,
        std::mutex& log_mutex);

private:
    Config config_;
    torch::Device device_;

    SelfPlayStats generate_games_batched(ChessNet& model, ReplayBuffer& buffer, int num_games);
    SelfPlayStats generate_games_gpu(ChessNet& model, ReplayBuffer& buffer, int num_games) {
        return generate_games_batched(model, buffer, num_games);
    }
    SelfPlayStats generate_games_cpu(ChessNet& model, ReplayBuffer& buffer, int num_games);
};

} // namespace az
