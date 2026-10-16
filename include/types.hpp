#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <iostream>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace az {

// ── Basic Chess Types ────────────────────────────────────────────────
using Bitboard = uint64_t;

// Portable 64-bit bit intrinsics (GCC/Clang builtins vs MSVC).
// Both are only ever called on a non-zero bitboard.
inline int ctz64(uint64_t b) {
#if defined(_MSC_VER)
    unsigned long idx;
    _BitScanForward64(&idx, b);
    return static_cast<int>(idx);
#else
    return __builtin_ctzll(b);
#endif
}

inline int popcount64(uint64_t b) {
#if defined(_MSC_VER)
    return static_cast<int>(__popcnt64(b));
#else
    return __builtin_popcountll(b);
#endif
}

enum Color : int {
    WHITE = 0,
    BLACK = 1,
    NONE_COLOR = 2
};

inline Color operator!(Color c) {
    return (c == WHITE) ? BLACK : WHITE;
}

enum PieceType : int {
    PAWN = 0,
    KNIGHT = 1,
    BISHOP = 2,
    ROOK = 3,
    QUEEN = 4,
    KING = 5,
    NONE_PIECE = 6
};

enum Square : int {
    A1 = 0,  B1, C1, D1, E1, F1, G1, H1,
    A2 = 8,  B2, C2, D2, E2, F2, G2, H2,
    A3 = 16, B3, C3, D3, E3, F3, G3, H3,
    A4 = 24, B4, C4, D4, E4, F4, G4, H4,
    A5 = 32, B5, C5, D5, E5, F5, G5, H5,
    A6 = 40, B6, C6, D6, E6, F6, G6, H6,
    A7 = 48, B7, C7, D7, E7, F7, G7, H7,
    A8 = 56, B8, C8, D8, E8, F8, G8, H8,
    SQ_NONE = 64
};

inline int rank_of(int sq) { return sq / 8; }
inline int file_of(int sq) { return sq % 8; }
inline int make_square(int file, int rank) { return rank * 8 + file; }

inline int square_mirror(int sq) {
    return sq ^ 56; // Mirrors rank 0 <-> 7, 1 <-> 6, etc.
}

// ── Move Representation ──────────────────────────────────────────────
struct Move {
    int from;
    int to;
    PieceType promotion; // NONE_PIECE if not a promotion

    Move() : from(SQ_NONE), to(SQ_NONE), promotion(NONE_PIECE) {}
    Move(int f, int t, PieceType p = NONE_PIECE) : from(f), to(t), promotion(p) {}

    bool is_null() const { return from == SQ_NONE; }
    bool operator==(const Move& o) const {
        return from == o.from && to == o.to && promotion == o.promotion;
    }
    bool operator!=(const Move& o) const {
        return !(*this == o);
    }

    std::string to_uci() const {
        if (is_null()) return "0000";
        static const char f_names[] = "abcdefgh";
        static const char r_names[] = "12345678";
        std::string s;
        s += f_names[file_of(from)];
        s += r_names[rank_of(from)];
        s += f_names[file_of(to)];
        s += r_names[rank_of(to)];
        if (promotion != NONE_PIECE) {
            if (promotion == KNIGHT) s += 'n';
            else if (promotion == BISHOP) s += 'b';
            else if (promotion == ROOK) s += 'r';
            else if (promotion == QUEEN) s += 'q';
        }
        return s;
    }
};

// ── AlphaZero Constants ──────────────────────────────────────────────
constexpr int NUM_CHANNELS = 18;
constexpr int BOARD_SIZE = 8;
constexpr int NUM_ACTIONS = 4672;

// 8 Queen-type directions: (rank_delta, file_delta)
struct Direction {
    int dr;
    int df;
};

const Direction QUEEN_DIRS[8] = {
    { 1,  0}, // 0 N
    { 1,  1}, // 1 NE
    { 0,  1}, // 2 E
    {-1,  1}, // 3 SE
    {-1,  0}, // 4 S
    {-1, -1}, // 5 SW
    { 0, -1}, // 6 W
    { 1, -1}  // 7 NW
};

const Direction KNIGHT_DIRS[8] = {
    { 2,  1}, { 1,  2}, {-1,  2}, {-2,  1},
    {-2, -1}, {-1, -2}, { 1, -2}, { 2, -1}
};

const PieceType UNDERPROMO_PIECES[3] = { KNIGHT, BISHOP, ROOK };
const int UNDERPROMO_FILE_DELTAS[3] = { -1, 0, 1 };

} // namespace az
