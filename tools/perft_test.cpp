// Perft test — verifies generate_legal_moves + make_move/undo_move +
// leaves_king_in_check produce correct node counts against known values.
//
// Usage: perft_test [max_depth]
// Exit 0 = all passed, 1 = any mismatch.

#include "board.hpp"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace az;

static uint64_t perft(Board& b, int depth) {
    if (depth == 0) return 1;
    auto moves = b.generate_legal_moves();
    if (depth == 1) return static_cast<uint64_t>(moves.size());
    uint64_t nodes = 0;
    for (const auto& m : moves) {
        b.make_move(m);
        nodes += perft(b, depth - 1);
        b.undo_move(m);
    }
    return nodes;
}

struct TestCase {
    const char* name;
    const char* fen;
    int depth;
    uint64_t expected;
};

int main(int argc, char** argv) {
    int max_depth = argc > 1 ? atoi(argv[1]) : 99;

    std::vector<TestCase> cases = {
        // ── Startpos ──────────────────────────────────────────────
        {"startpos d1", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 1, 20},
        {"startpos d2", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 2, 400},
        {"startpos d3", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 3, 8902},
        {"startpos d4", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 4, 197281},
        {"startpos d5", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 5, 4865609},
        // ── Kiwipete (position 2) ────────────────────────────────
        {"kiwipete d1", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 1, 48},
        {"kiwipete d2", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 2, 2039},
        {"kiwipete d3", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 3, 97862},
        {"kiwipete d4", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 4, 4085603},
        // ── Position 3 (endgame, ep, pins) ───────────────────────
        {"pos3 d1", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 1, 14},
        {"pos3 d2", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 2, 191},
        {"pos3 d3", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 3, 2812},
        {"pos3 d4", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 4, 43238},
        {"pos3 d5", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 5, 674624},
        // ── Position 4 (promotions, captures) ────────────────────
        {"pos4 d1", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 1, 6},
        {"pos4 d2", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 2, 264},
        {"pos4 d3", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 3, 9467},
        {"pos4 d4", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 4, 422333},
        // ── Position 5 ───────────────────────────────────────────
        {"pos5 d1", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 1, 44},
        {"pos5 d2", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 2, 1486},
        {"pos5 d3", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 3, 62379},
        {"pos5 d4", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 4, 2103487},
    };

    int pass = 0, fail = 0, skip = 0;
    for (const auto& tc : cases) {
        if (tc.depth > max_depth) { skip++; continue; }
        Board b(tc.fen);
        uint64_t got = perft(b, tc.depth);
        bool ok = (got == tc.expected);
        printf("[%s] %-14s depth=%d expected=%-10llu got=%-10llu %s\n",
               ok ? "PASS" : "FAIL", tc.name, tc.depth,
               (unsigned long long)tc.expected,
               (unsigned long long)got,
               ok ? "" : "<-- MISMATCH");
        if (ok) pass++; else fail++;
    }
    printf("\n%d passed, %d failed, %d skipped (max_depth=%d)\n",
           pass, fail, skip, max_depth);
    return fail > 0 ? 1 : 0;
}
