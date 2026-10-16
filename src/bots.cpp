#include "bots.hpp"
#include <algorithm>
#include <chrono>

namespace az {

namespace {

// Piece square tables (from White's perspective)
const int PST_PAWN[64] = {
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10,-20,-20, 10, 10,  5,
     5, -5,-10,  0,  0,-10, -5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5,  5, 10, 25, 25, 10,  5,  5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
     0,  0,  0,  0,  0,  0,  0,  0
};

const int PST_KNIGHT[64] = {
   -50,-40,-30,-30,-30,-30,-40,-50,
   -40,-20,  0,  5,  5,  0,-20,-40,
   -30,  5, 10, 15, 15, 10,  5,-30,
   -30,  0, 15, 20, 20, 15,  0,-30,
   -30,  5, 15, 20, 20, 15,  5,-30,
   -30,  0, 10, 15, 15, 10,  0,-30,
   -40,-20,  0,  0,  0,  0,-20,-40,
   -50,-40,-30,-30,-30,-30,-40,-50
};

const int PST_BISHOP[64] = {
   -20,-10,-10,-10,-10,-10,-10,-20,
   -10,  5,  0,  0,  0,  0,  5,-10,
   -10, 10, 10, 10, 10, 10, 10,-10,
   -10,  0, 10, 10, 10, 10,  0,-10,
   -10,  5,  5, 10, 10,  5,  5,-10,
   -10,  0,  5, 10, 10,  5,  0,-10,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -20,-10,-10,-10,-10,-10,-10,-20
};

const int PST_ROOK[64] = {
     0,  0,  0,  5,  5,  0,  0,  0,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     5, 10, 10, 10, 10, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0
};

const int PST_QUEEN[64] = {
   -20,-10,-10, -5, -5,-10,-10,-20,
   -10,  0,  5,  0,  0,  0,  0,-10,
   -10,  5,  5,  5,  5,  5,  0,-10,
     0,  0,  5,  5,  5,  5,  0, -5,
    -5,  0,  5,  5,  5,  5,  0, -5,
   -10,  0,  5,  5,  5,  5,  0,-10,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -20,-10,-10, -5, -5,-10,-10,-20
};

const int PST_KING[64] = {
    20, 30, 10,  0,  0, 10, 30, 20,
    20, 20,  0,  0,  0,  0, 20, 20,
   -10,-20,-20,-20,-20,-20,-20,-10,
   -20,-30,-30,-40,-40,-30,-30,-20,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30
};

const int PIECE_VALUES[6] = { 100, 320, 330, 500, 900, 20000 };

} // anonymous namespace

// ── RandomBot ────────────────────────────────────────────────────────
RandomBot::RandomBot() : rng_(std::random_device{}()) {}

Move RandomBot::select_move(Board& board) {
    auto moves = board.generate_legal_moves();
    if (moves.empty()) return Move();
    std::uniform_int_distribution<size_t> dist(0, moves.size() - 1);
    return moves[dist(rng_)];
}

// ── MinimaxBot ───────────────────────────────────────────────────────
MinimaxBot::MinimaxBot(int depth) : depth_(depth) {}

int MinimaxBot::evaluate(const Board& board) const {
    int score = 0;

    for (int pt = 0; pt < 6; ++pt) {
        // White pieces
        Bitboard wb = board.pieces(static_cast<PieceType>(pt), WHITE);
        while (wb) {
            int sq = ctz64(wb);
            wb &= (wb - 1);
            score += PIECE_VALUES[pt];
            switch (pt) {
                case PAWN:   score += PST_PAWN[sq]; break;
                case KNIGHT: score += PST_KNIGHT[sq]; break;
                case BISHOP: score += PST_BISHOP[sq]; break;
                case ROOK:   score += PST_ROOK[sq]; break;
                case QUEEN:  score += PST_QUEEN[sq]; break;
                case KING:   score += PST_KING[sq]; break;
            }
        }

        // Black pieces
        Bitboard bb = board.pieces(static_cast<PieceType>(pt), BLACK);
        while (bb) {
            int sq = ctz64(bb);
            bb &= (bb - 1);
            int mirrored_sq = square_mirror(sq);
            score -= PIECE_VALUES[pt];
            switch (pt) {
                case PAWN:   score -= PST_PAWN[mirrored_sq]; break;
                case KNIGHT: score -= PST_KNIGHT[mirrored_sq]; break;
                case BISHOP: score -= PST_BISHOP[mirrored_sq]; break;
                case ROOK:   score -= PST_ROOK[mirrored_sq]; break;
                case QUEEN:  score -= PST_QUEEN[mirrored_sq]; break;
                case KING:   score -= PST_KING[mirrored_sq]; break;
            }
        }
    }

    return score;
}

void MinimaxBot::order_moves(std::vector<Move>& moves, const Board& board) const {
    Color them = !board.turn();

    std::vector<std::pair<int, Move>> scored_moves;
    scored_moves.reserve(moves.size());

    for (const auto& m : moves) {
        int score = 0;
        Color cap_c;
        PieceType captured = board.piece_at(m.to, cap_c);
        if (captured != NONE_PIECE && cap_c == them) {
            PieceType attacker = board.piece_type_at(m.from);
            score = 10000 + PIECE_VALUES[captured] - PIECE_VALUES[attacker];
        }

        Board temp = board;
        if (temp.make_move(m)) {
            if (temp.in_check(them)) {
                score += 5000;
            }
        }
        scored_moves.emplace_back(score, m);
    }

    std::sort(scored_moves.begin(), scored_moves.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });

    for (size_t i = 0; i < moves.size(); ++i) {
        moves[i] = scored_moves[i].second;
    }
}

int MinimaxBot::negamax(Board& board, int depth, int alpha, int beta, int color_sign) {
    if (board.is_game_over() || depth == 0) {
        if (board.is_checkmate()) {
            return -100000 - depth;
        }
        if (board.is_game_over()) {
            return 0; // Draw
        }
        return color_sign * evaluate(board);
    }

    auto moves = board.generate_legal_moves();
    order_moves(moves, board);

    int max_eval = -1000000;

    for (const auto& m : moves) {
        board.make_move(m);
        int eval = -negamax(board, depth - 1, -beta, -alpha, -color_sign);
        board.undo_move(m);

        max_eval = std::max(max_eval, eval);
        alpha = std::max(alpha, eval);
        if (alpha >= beta) {
            break; // Beta cutoff
        }
    }

    return max_eval;
}

Move MinimaxBot::select_move(Board& board) {
    auto moves = board.generate_legal_moves();
    if (moves.empty()) return Move();
    order_moves(moves, board);

    int color_sign = (board.turn() == WHITE) ? 1 : -1;
    int best_eval = -1000000;
    Move best_move = moves[0];

    int alpha = -1000000;
    int beta = 1000000;

    for (const auto& m : moves) {
        board.make_move(m);
        int eval = -negamax(board, depth_ - 1, -beta, -alpha, -color_sign);
        board.undo_move(m);

        if (eval > best_eval) {
            best_eval = eval;
            best_move = m;
        }
        alpha = std::max(alpha, eval);
    }

    return best_move;
}

} // namespace az
