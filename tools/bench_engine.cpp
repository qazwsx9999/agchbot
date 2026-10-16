// Micro-benchmark for the 3 CPU hot paths touched by the optimization pass:
//   1. generate_legal_moves  (leaves_king_in_check vs Board-copy legality)
//   2. make_move/undo_move   (incremental Zobrist vs full recompute)
//   3. is_game_over vs is_game_over_fast (move-gen-free terminal check)
//
// Usage: bench_engine [seconds_per_phase]

#include "board.hpp"
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

using namespace az;
using Clock = std::chrono::steady_clock;

// Play random games from a position, return boards at mid-game positions.
static std::vector<Board> sample_positions(int count, std::mt19937& rng) {
    std::vector<Board> out;
    while ((int)out.size() < count) {
        Board b;
        int plies = 8 + (rng() % 40); // sample mid-game positions
        for (int p = 0; p < plies && !b.is_game_over(); ++p) {
            auto moves = b.generate_legal_moves();
            if (moves.empty()) break;
            b.make_move(moves[rng() % moves.size()]);
        }
        if (!b.is_game_over()) out.push_back(b);
    }
    return out;
}

int main(int argc, char** argv) {
    double secs = argc > 1 ? atof(argv[1]) : 2.0;
    std::mt19937 rng(12345);
    auto positions = sample_positions(200, rng);
    printf("Sampled %d positions\n\n", (int)positions.size());

    // ── 1. generate_legal_moves throughput ─────────────────────────
    {
        uint64_t moves = 0, calls = 0;
        auto t0 = Clock::now();
        while (std::chrono::duration<double>(Clock::now() - t0).count() < secs) {
            for (auto& b : positions) {
                auto mv = b.generate_legal_moves();
                calls++;
                moves += mv.size();
            }
        }
        double el = std::chrono::duration<double>(Clock::now() - t0).count();
        printf("generate_legal_moves : %.0f calls/sec  (%.0f moves/sec)\n",
               calls / el, moves / el);
    }

    // ── 2. make_move + undo_move throughput ────────────────────────
    {
        uint64_t pairs = 0;
        auto t0 = Clock::now();
        while (std::chrono::duration<double>(Clock::now() - t0).count() < secs) {
            for (auto& b : positions) {
                auto mv = b.generate_legal_moves();
                for (auto& m : mv) {
                    b.make_move(m);
                    b.undo_move(m);
                    pairs++;
                }
            }
        }
        double el = std::chrono::duration<double>(Clock::now() - t0).count();
        printf("make+undo pairs      : %.0f/sec\n", pairs / el);
    }

    // ── 3a. is_game_over (full) throughput ─────────────────────────
    {
        uint64_t calls = 0;
        auto t0 = Clock::now();
        while (std::chrono::duration<double>(Clock::now() - t0).count() < secs) {
            for (auto& b : positions) { b.is_game_over(); calls++; }
        }
        double el = std::chrono::duration<double>(Clock::now() - t0).count();
        printf("is_game_over         : %.0f calls/sec\n", calls / el);
    }

    // ── 3b. is_game_over_fast (draw rules only) throughput ─────────
    {
        uint64_t calls = 0;
        auto t0 = Clock::now();
        while (std::chrono::duration<double>(Clock::now() - t0).count() < secs) {
            for (auto& b : positions) { b.is_game_over_fast(); calls++; }
        }
        double el = std::chrono::duration<double>(Clock::now() - t0).count();
        printf("is_game_over_fast    : %.0f calls/sec\n", calls / el);
    }

    // ── 4. Simulated MCTS descent: descend ~30 plies with fast checks ──
    {
        uint64_t plies = 0;
        auto t0 = Clock::now();
        while (std::chrono::duration<double>(Clock::now() - t0).count() < secs) {
            for (auto& pos : positions) {
                Board scratch = pos;
                for (int d = 0; d < 30; ++d) {
                    if (scratch.is_game_over_fast()) break;
                    auto mv = scratch.generate_legal_moves();
                    if (mv.empty()) break;
                    scratch.make_move(mv[rng() % mv.size()]);
                    plies++;
                }
            }
        }
        double el = std::chrono::duration<double>(Clock::now() - t0).count();
        printf("MCTS-descent plies   : %.0f plies/sec\n", plies / el);
    }

    return 0;
}
