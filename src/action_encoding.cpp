#include "action_encoding.hpp"
#include <cstring>
#include <cmath>
#include <algorithm>

namespace az {

namespace {

int get_queen_dir_idx(int dr, int df) {
    for (int i = 0; i < 8; ++i) {
        if (QUEEN_DIRS[i].dr == dr && QUEEN_DIRS[i].df == df) return i;
    }
    return -1;
}

int get_knight_dir_idx(int dr, int df) {
    for (int i = 0; i < 8; ++i) {
        if (KNIGHT_DIRS[i].dr == dr && KNIGHT_DIRS[i].df == df) return i;
    }
    return -1;
}

int get_underpromo_piece_idx(PieceType pt) {
    if (pt == KNIGHT) return 0;
    if (pt == BISHOP) return 1;
    if (pt == ROOK) return 2;
    return -1;
}

int get_underpromo_file_delta_idx(int df) {
    if (df == -1) return 0;
    if (df == 0) return 1;
    if (df == 1) return 2;
    return -1;
}

} // anonymous namespace

void encode_board(const Board& board, float* out_planes) {
    // Zero out all 21 * 8 * 8 = 1344 floats
    std::memset(out_planes, 0, NUM_CHANNELS * 64 * sizeof(float));

    bool flip = (board.turn() == BLACK);
    Color own = board.turn();
    Color opp = !own;

    PieceType ptypes[6] = { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING };

    // Planes 0-5: Current player's pieces
    for (int i = 0; i < 6; ++i) {
        Bitboard b = board.pieces(ptypes[i], own);
        while (b) {
            int sq = ctz64(b);
            b &= (b - 1);
            int msq = flip ? square_mirror(sq) : sq;
            int r = rank_of(msq);
            int f = file_of(msq);
            out_planes[i * 64 + r * 8 + f] = 1.0f;
        }
    }

    // Planes 6-11: Opponent's pieces
    for (int i = 0; i < 6; ++i) {
        Bitboard b = board.pieces(ptypes[i], opp);
        while (b) {
            int sq = ctz64(b);
            b &= (b - 1);
            int msq = flip ? square_mirror(sq) : sq;
            int r = rank_of(msq);
            int f = file_of(msq);
            out_planes[(6 + i) * 64 + r * 8 + f] = 1.0f;
        }
    }

    // Plane 12: Color (1.0 if White to move, else 0.0)
    if (board.turn() == WHITE) {
        for (int sq = 0; sq < 64; ++sq) {
            out_planes[12 * 64 + sq] = 1.0f;
        }
    }

    // Plane 13: Own king-side castling
    if (board.has_kingside_castling_rights(own)) {
        for (int sq = 0; sq < 64; ++sq) out_planes[13 * 64 + sq] = 1.0f;
    }

    // Plane 14: Own queen-side castling
    if (board.has_queenside_castling_rights(own)) {
        for (int sq = 0; sq < 64; ++sq) out_planes[14 * 64 + sq] = 1.0f;
    }

    // Plane 15: Opp king-side castling
    if (board.has_kingside_castling_rights(opp)) {
        for (int sq = 0; sq < 64; ++sq) out_planes[15 * 64 + sq] = 1.0f;
    }

    // Plane 16: Opp queen-side castling
    if (board.has_queenside_castling_rights(opp)) {
        for (int sq = 0; sq < 64; ++sq) out_planes[16 * 64 + sq] = 1.0f;
    }

    // Plane 17: En-passant square
    int ep = board.ep_square();
    if (ep != SQ_NONE) {
        int msq = flip ? square_mirror(ep) : ep;
        int r = rank_of(msq);
        int f = file_of(msq);
        out_planes[17 * 64 + r * 8 + f] = 1.0f;
    }

    // Plane 18: 1st repetition (repetition_count >= 1, i.e. 2nd occurrence)
    if (board.repetition_count() >= 1) {
        for (int sq = 0; sq < 64; ++sq) {
            out_planes[18 * 64 + sq] = 1.0f;
        }
    }

    // Plane 19: 2nd+ repetition (repetition_count >= 2, i.e. 3rd+ occurrence)
    if (board.repetition_count() >= 2) {
        for (int sq = 0; sq < 64; ++sq) {
            out_planes[19 * 64 + sq] = 1.0f;
        }
    }

    // Plane 20: 50-move rule clock normalized [0.0, 1.0]
    float hm = std::min(1.0f, static_cast<float>(board.halfmove_clock()) / 100.0f);
    if (hm > 0.0f) {
        for (int sq = 0; sq < 64; ++sq) {
            out_planes[20 * 64 + sq] = hm;
        }
    }
}

int encode_action(const Move& move, const Board& board) {
    bool flip = (board.turn() == BLACK);

    int from_sq = flip ? square_mirror(move.from) : move.from;
    int to_sq = flip ? square_mirror(move.to) : move.to;

    int from_r = rank_of(from_sq);
    int from_f = file_of(from_sq);
    int to_r = rank_of(to_sq);
    int to_f = file_of(to_sq);

    int dr = to_r - from_r;
    int df = to_f - from_f;

    int from_idx = from_r * 8 + from_f;
    int move_type = 0;

    // Under-promotion (Knight, Bishop, Rook)
    if (move.promotion != NONE_PIECE && move.promotion != QUEEN) {
        int piece_i = get_underpromo_piece_idx(move.promotion);
        int fd_i = get_underpromo_file_delta_idx(df);
        move_type = 64 + piece_i * 3 + fd_i;
    }
    // Knight move
    else if (get_knight_dir_idx(dr, df) != -1) {
        move_type = 56 + get_knight_dir_idx(dr, df);
    }
    // Queen-type (slides, captures, and queen promotions)
    else {
        int unit_r = (dr > 0) ? 1 : ((dr < 0) ? -1 : 0);
        int unit_f = (df > 0) ? 1 : ((df < 0) ? -1 : 0);
        int dist = std::max(std::abs(dr), std::abs(df));
        int dir_i = get_queen_dir_idx(unit_r, unit_f);
        move_type = dir_i * 7 + (dist - 1);
    }

    return from_idx * 73 + move_type;
}

Move decode_action(int action, const Board& board) {
    bool flip = (board.turn() == BLACK);

    int from_idx = action / 73;
    int move_type = action % 73;

    int from_r = from_idx / 8;
    int from_f = from_idx % 8;

    int to_r = 0;
    int to_f = 0;
    PieceType promo = NONE_PIECE;

    if (move_type < 56) {
        // Queen-type
        int dir_i = move_type / 7;
        int dist = (move_type % 7) + 1;
        to_r = from_r + QUEEN_DIRS[dir_i].dr * dist;
        to_f = from_f + QUEEN_DIRS[dir_i].df * dist;
    } else if (move_type < 64) {
        // Knight
        int ki = move_type - 56;
        to_r = from_r + KNIGHT_DIRS[ki].dr;
        to_f = from_f + KNIGHT_DIRS[ki].df;
    } else {
        // Under-promotion
        int pi = move_type - 64;
        int piece_i = pi / 3;
        int fd_i = pi % 3;
        promo = UNDERPROMO_PIECES[piece_i];
        to_r = from_r + 1;
        to_f = from_f + UNDERPROMO_FILE_DELTAS[fd_i];
    }

    int from_sq = make_square(from_f, from_r);
    int to_sq = make_square(to_f, to_r);

    // Detect Queen promotion: pawn advancing to 8th rank
    if (promo == NONE_PIECE && to_r == 7) {
        int real_from = flip ? square_mirror(from_sq) : from_sq;
        Color c;
        PieceType pt = board.piece_at(real_from, c);
        if (pt == PAWN) {
            promo = QUEEN;
        }
    }

    if (flip) {
        from_sq = square_mirror(from_sq);
        to_sq = square_mirror(to_sq);
    }

    return Move(from_sq, to_sq, promo);
}

void get_legal_action_mask(const Board& board, bool* out_mask) {
    std::memset(out_mask, 0, NUM_ACTIONS * sizeof(bool));
    auto legal_moves = board.generate_legal_moves();
    for (const auto& m : legal_moves) {
        int action = encode_action(m, board);
        if (action >= 0 && action < NUM_ACTIONS) {
            out_mask[action] = true;
        }
    }
}

} // namespace az
