#include "board.hpp"
#include <sstream>
#include <cctype>
#include <algorithm>
#include <random>
#include <mutex>

namespace az {

namespace {

// ── Attack & Zobrist Tables ──────────────────────────────────────────
Bitboard PAWN_ATTACKS[2][64];
Bitboard KNIGHT_ATTACKS[64];
Bitboard KING_ATTACKS[64];

uint64_t ZOBRIST_PIECES[2][6][64];
uint64_t ZOBRIST_CASTLING[16];
uint64_t ZOBRIST_EP[8];
uint64_t ZOBRIST_SIDE;

bool TABLES_INITIALIZED = false;

void init_tables_once() {
    // Fixed seed for deterministic Zobrist hashing
    std::mt19937_64 rng(0x123456789ABCDEF0ULL);
    for (int c = 0; c < 2; ++c) {
        for (int pt = 0; pt < 6; ++pt) {
            for (int sq = 0; sq < 64; ++sq) {
                ZOBRIST_PIECES[c][pt][sq] = rng();
            }
        }
    }
    for (int i = 0; i < 16; ++i) {
        ZOBRIST_CASTLING[i] = rng();
    }
    for (int i = 0; i < 8; ++i) {
        ZOBRIST_EP[i] = rng();
    }
    ZOBRIST_SIDE = rng();

    for (int sq = 0; sq < 64; ++sq) {
        int r = rank_of(sq);
        int f = file_of(sq);

        // White Pawn attacks (up-left, up-right)
        Bitboard wp = 0;
        if (r < 7 && f > 0) wp |= (1ULL << make_square(f - 1, r + 1));
        if (r < 7 && f < 7) wp |= (1ULL << make_square(f + 1, r + 1));
        PAWN_ATTACKS[WHITE][sq] = wp;

        // Black Pawn attacks (down-left, down-right)
        Bitboard bp = 0;
        if (r > 0 && f > 0) bp |= (1ULL << make_square(f - 1, r - 1));
        if (r > 0 && f < 7) bp |= (1ULL << make_square(f + 1, r - 1));
        PAWN_ATTACKS[BLACK][sq] = bp;

        // Knight attacks
        Bitboard k = 0;
        for (const auto& d : KNIGHT_DIRS) {
            int nr = r + d.dr;
            int nf = f + d.df;
            if (nr >= 0 && nr < 8 && nf >= 0 && nf < 8) {
                k |= (1ULL << make_square(nf, nr));
            }
        }
        KNIGHT_ATTACKS[sq] = k;

        // King attacks
        Bitboard kg = 0;
        for (const auto& d : QUEEN_DIRS) {
            int nr = r + d.dr;
            int nf = f + d.df;
            if (nr >= 0 && nr < 8 && nf >= 0 && nf < 8) {
                kg |= (1ULL << make_square(nf, nr));
            }
        }
        KING_ATTACKS[sq] = kg;
    }

    TABLES_INITIALIZED = true;
}

// Thread-safe lazy init: worker threads may construct Boards concurrently,
// so the one-time table fill must run exactly once.
void init_tables() {
    static std::once_flag flag;
    std::call_once(flag, init_tables_once);
}

Bitboard sliding_attacks(int sq, Bitboard occ, const Direction* dirs, int num_dirs) {
    Bitboard attacks = 0;
    int r = rank_of(sq);
    int f = file_of(sq);

    for (int d = 0; d < num_dirs; ++d) {
        int dr = dirs[d].dr;
        int df = dirs[d].df;
        int cr = r + dr;
        int cf = f + df;
        while (cr >= 0 && cr < 8 && cf >= 0 && cf < 8) {
            int target_sq = make_square(cf, cr);
            attacks |= (1ULL << target_sq);
            if (occ & (1ULL << target_sq)) {
                break; // Blocked
            }
            cr += dr;
            cf += df;
        }
    }
    return attacks;
}

const Direction ROOK_DIRS[4] = { {1, 0}, {-1, 0}, {0, 1}, {0, -1} };
const Direction BISHOP_DIRS[4] = { {1, 1}, {1, -1}, {-1, 1}, {-1, -1} };

} // anonymous namespace

Board::Board() {
    init_tables();
    reset();
}

Board::Board(const std::string& fen) {
    init_tables();
    set_fen(fen);
}

void Board::reset() {
    set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

Bitboard Board::color_bb(Color c) const {
    Bitboard b = 0;
    for (int pt = 0; pt < 6; ++pt) {
        b |= piece_bb_[c][pt];
    }
    return b;
}

Bitboard Board::occupied() const {
    return color_bb(WHITE) | color_bb(BLACK);
}

PieceType Board::piece_at(int sq, Color& out_color) const {
    Bitboard mask = (1ULL << sq);
    for (int c = 0; c < 2; ++c) {
        for (int pt = 0; pt < 6; ++pt) {
            if (piece_bb_[c][pt] & mask) {
                out_color = static_cast<Color>(c);
                return static_cast<PieceType>(pt);
            }
        }
    }
    out_color = NONE_COLOR;
    return NONE_PIECE;
}

PieceType Board::piece_type_at(int sq) const {
    Color c;
    return piece_at(sq, c);
}

bool Board::has_kingside_castling_rights(Color c) const {
    return (c == WHITE) ? (castling_rights_ & 1) : (castling_rights_ & 4);
}

bool Board::has_queenside_castling_rights(Color c) const {
    return (c == WHITE) ? (castling_rights_ & 2) : (castling_rights_ & 8);
}

bool Board::is_attacked(int sq, Color attacker) const {
    Bitboard occ = occupied();

    // Attacked by pawns
    Color defender = !attacker;
    if (PAWN_ATTACKS[defender][sq] & piece_bb_[attacker][PAWN]) return true;

    // Attacked by knights
    if (KNIGHT_ATTACKS[sq] & piece_bb_[attacker][KNIGHT]) return true;

    // Attacked by king
    if (KING_ATTACKS[sq] & piece_bb_[attacker][KING]) return true;

    // Attacked by bishops / queens
    Bitboard bq = piece_bb_[attacker][BISHOP] | piece_bb_[attacker][QUEEN];
    if (bq && (sliding_attacks(sq, occ, BISHOP_DIRS, 4) & bq)) return true;

    // Attacked by rooks / queens
    Bitboard rq = piece_bb_[attacker][ROOK] | piece_bb_[attacker][QUEEN];
    if (rq && (sliding_attacks(sq, occ, ROOK_DIRS, 4) & rq)) return true;

    return false;
}

bool Board::in_check(Color c) const {
    Bitboard king = piece_bb_[c][KING];
    if (!king) return false;
    int king_sq = ctz64(king);
    return is_attacked(king_sq, !c);
}

bool Board::set_fen(const std::string& fen) {
    for (int c = 0; c < 2; ++c) {
        for (int pt = 0; pt < 6; ++pt) {
            piece_bb_[c][pt] = 0;
        }
    }
    castling_rights_ = 0;
    ep_square_ = SQ_NONE;
    halfmove_clock_ = 0;
    fullmove_number_ = 1;
    history_.clear();

    std::istringstream ss(fen);
    std::string placement, active, castling, ep;
    if (!(ss >> placement >> active >> castling >> ep)) return false;

    // Placement
    int r = 7, f = 0;
    for (char ch : placement) {
        if (ch == '/') {
            r--;
            f = 0;
        } else if (std::isdigit(ch)) {
            f += (ch - '0');
        } else {
            Color c = std::isupper(ch) ? WHITE : BLACK;
            char lower = std::tolower(ch);
            PieceType pt = NONE_PIECE;
            if (lower == 'p') pt = PAWN;
            else if (lower == 'n') pt = KNIGHT;
            else if (lower == 'b') pt = BISHOP;
            else if (lower == 'r') pt = ROOK;
            else if (lower == 'q') pt = QUEEN;
            else if (lower == 'k') pt = KING;

            if (pt != NONE_PIECE && f < 8 && r >= 0) {
                piece_bb_[c][pt] |= (1ULL << make_square(f, r));
                f++;
            }
        }
    }

    turn_ = (active == "b") ? BLACK : WHITE;

    if (castling.find('K') != std::string::npos) castling_rights_ |= 1;
    if (castling.find('Q') != std::string::npos) castling_rights_ |= 2;
    if (castling.find('k') != std::string::npos) castling_rights_ |= 4;
    if (castling.find('q') != std::string::npos) castling_rights_ |= 8;

    if (ep != "-") {
        int ep_f = ep[0] - 'a';
        int ep_r = ep[1] - '1';
        ep_square_ = make_square(ep_f, ep_r);
    }

    ss >> halfmove_clock_ >> fullmove_number_;
    hash_ = compute_hash();
    return true;
}

// Copy only position fields, skipping the move-history vector (~8KB+ when
// long). Used inside MCTS where only the 1-ply make/undo legality test
// matters — the caller pushes a fresh history entry on each pseudo-move
// anyway.
Board Board::clone_lean() const {
    Board b;
    b.piece_bb_ = piece_bb_;
    b.turn_ = turn_;
    b.castling_rights_ = castling_rights_;
    b.ep_square_ = ep_square_;
    b.halfmove_clock_ = halfmove_clock_;
    b.fullmove_number_ = fullmove_number_;
    b.hash_ = hash_;
    return b;
}

// Assign all state from another board. `history_.assign` keeps this board's
// existing vector capacity in the common case — avoids a fresh ~8KB heap
// allocation on every MCTS scratch-board assignment.
void Board::copy_from(const Board& o) {
    piece_bb_ = o.piece_bb_;
    turn_ = o.turn_;
    castling_rights_ = o.castling_rights_;
    ep_square_ = o.ep_square_;
    halfmove_clock_ = o.halfmove_clock_;
    fullmove_number_ = o.fullmove_number_;
    hash_ = o.hash_;
    history_.assign(o.history_.begin(), o.history_.end());
}

uint64_t Board::compute_hash() const {
    uint64_t h = 0;
    for (int c = 0; c < 2; ++c) {
        for (int pt = 0; pt < 6; ++pt) {
            Bitboard b = piece_bb_[c][pt];
            while (b) {
                int sq = ctz64(b);
                b &= (b - 1);
                h ^= ZOBRIST_PIECES[c][pt][sq];
            }
        }
    }
    h ^= ZOBRIST_CASTLING[castling_rights_ & 0xF];
    if (ep_square_ != SQ_NONE) {
        h ^= ZOBRIST_EP[file_of(ep_square_)];
    }
    if (turn_ == BLACK) {
        h ^= ZOBRIST_SIDE;
    }
    return h;
}

std::string Board::to_fen() const {
    std::string s;
    for (int r = 7; r >= 0; --r) {
        int empty = 0;
        for (int f = 0; f < 8; ++f) {
            Color c;
            PieceType pt = piece_at(make_square(f, r), c);
            if (pt == NONE_PIECE) {
                empty++;
            } else {
                if (empty > 0) {
                    s += std::to_string(empty);
                    empty = 0;
                }
                static const char pieces_str[] = "pnbrqk";
                char ch = pieces_str[pt];
                if (c == WHITE) ch = std::toupper(ch);
                s += ch;
            }
        }
        if (empty > 0) s += std::to_string(empty);
        if (r > 0) s += '/';
    }

    s += (turn_ == WHITE) ? " w " : " b ";

    std::string castling;
    if (castling_rights_ & 1) castling += 'K';
    if (castling_rights_ & 2) castling += 'Q';
    if (castling_rights_ & 4) castling += 'k';
    if (castling_rights_ & 8) castling += 'q';
    if (castling.empty()) castling = "-";
    s += castling + " ";

    if (ep_square_ != SQ_NONE) {
        s += static_cast<char>('a' + file_of(ep_square_));
        s += static_cast<char>('1' + rank_of(ep_square_));
    } else {
        s += "-";
    }

    s += " " + std::to_string(halfmove_clock_) + " " + std::to_string(fullmove_number_);
    return s;
}

void Board::generate_pawn_moves(std::vector<Move>& moves) const {
    Color us = turn_;
    Color them = !us;
    Bitboard pawns = piece_bb_[us][PAWN];
    Bitboard occ = occupied();
    Bitboard opp = color_bb(them);

    int forward = (us == WHITE) ? 8 : -8;
    int promo_rank = (us == WHITE) ? 7 : 0;
    int start_rank = (us == WHITE) ? 1 : 6;

    while (pawns) {
        int from = ctz64(pawns);
        pawns &= (pawns - 1);
        int r = rank_of(from);

        // Single push
        int to1 = from + forward;
        if (!(occ & (1ULL << to1))) {
            if (rank_of(to1) == promo_rank) {
                moves.emplace_back(from, to1, QUEEN);
                moves.emplace_back(from, to1, KNIGHT);
                moves.emplace_back(from, to1, ROOK);
                moves.emplace_back(from, to1, BISHOP);
            } else {
                moves.emplace_back(from, to1, NONE_PIECE);
                // Double push
                if (r == start_rank) {
                    int to2 = from + 2 * forward;
                    if (!(occ & (1ULL << to2))) {
                        moves.emplace_back(from, to2, NONE_PIECE);
                    }
                }
            }
        }

        // Captures
        Bitboard attacks = PAWN_ATTACKS[us][from] & opp;
        while (attacks) {
            int to = ctz64(attacks);
            attacks &= (attacks - 1);
            if (rank_of(to) == promo_rank) {
                moves.emplace_back(from, to, QUEEN);
                moves.emplace_back(from, to, KNIGHT);
                moves.emplace_back(from, to, ROOK);
                moves.emplace_back(from, to, BISHOP);
            } else {
                moves.emplace_back(from, to, NONE_PIECE);
            }
        }

        // En-passant capture
        if (ep_square_ != SQ_NONE) {
            if (PAWN_ATTACKS[us][from] & (1ULL << ep_square_)) {
                moves.emplace_back(from, ep_square_, NONE_PIECE);
            }
        }
    }
}

void Board::generate_knight_moves(std::vector<Move>& moves) const {
    Color us = turn_;
    Bitboard knights = piece_bb_[us][KNIGHT];
    Bitboard own_pieces = color_bb(us);

    while (knights) {
        int from = ctz64(knights);
        knights &= (knights - 1);
        Bitboard targets = KNIGHT_ATTACKS[from] & ~own_pieces;
        while (targets) {
            int to = ctz64(targets);
            targets &= (targets - 1);
            moves.emplace_back(from, to, NONE_PIECE);
        }
    }
}

void Board::generate_bishop_moves(std::vector<Move>& moves) const {
    Color us = turn_;
    Bitboard bishops = piece_bb_[us][BISHOP];
    Bitboard occ = occupied();
    Bitboard own_pieces = color_bb(us);

    while (bishops) {
        int from = ctz64(bishops);
        bishops &= (bishops - 1);
        Bitboard targets = sliding_attacks(from, occ, BISHOP_DIRS, 4) & ~own_pieces;
        while (targets) {
            int to = ctz64(targets);
            targets &= (targets - 1);
            moves.emplace_back(from, to, NONE_PIECE);
        }
    }
}

void Board::generate_rook_moves(std::vector<Move>& moves) const {
    Color us = turn_;
    Bitboard rooks = piece_bb_[us][ROOK];
    Bitboard occ = occupied();
    Bitboard own_pieces = color_bb(us);

    while (rooks) {
        int from = ctz64(rooks);
        rooks &= (rooks - 1);
        Bitboard targets = sliding_attacks(from, occ, ROOK_DIRS, 4) & ~own_pieces;
        while (targets) {
            int to = ctz64(targets);
            targets &= (targets - 1);
            moves.emplace_back(from, to, NONE_PIECE);
        }
    }
}

void Board::generate_queen_moves(std::vector<Move>& moves) const {
    Color us = turn_;
    Bitboard queens = piece_bb_[us][QUEEN];
    Bitboard occ = occupied();
    Bitboard own_pieces = color_bb(us);

    while (queens) {
        int from = ctz64(queens);
        queens &= (queens - 1);
        Bitboard targets = (sliding_attacks(from, occ, ROOK_DIRS, 4) |
                            sliding_attacks(from, occ, BISHOP_DIRS, 4)) & ~own_pieces;
        while (targets) {
            int to = ctz64(targets);
            targets &= (targets - 1);
            moves.emplace_back(from, to, NONE_PIECE);
        }
    }
}

void Board::generate_king_moves(std::vector<Move>& moves) const {
    Color us = turn_;
    Color them = !us;
    Bitboard king = piece_bb_[us][KING];
    if (!king) return;
    int from = ctz64(king);
    Bitboard own_pieces = color_bb(us);
    Bitboard occ = occupied();

    // Standard steps
    Bitboard targets = KING_ATTACKS[from] & ~own_pieces;
    while (targets) {
        int to = ctz64(targets);
        targets &= (targets - 1);
        moves.emplace_back(from, to, NONE_PIECE);
    }

    // Castling
    if (us == WHITE && from == E1 && !in_check(WHITE)) {
        // Kingside: E1 -> G1 (F1, G1 must be empty and not attacked)
        if ((castling_rights_ & 1) &&
            !(occ & ((1ULL << F1) | (1ULL << G1))) &&
            !is_attacked(F1, them) && !is_attacked(G1, them)) {
            moves.emplace_back(E1, G1, NONE_PIECE);
        }
        // Queenside: E1 -> C1 (D1, C1, B1 empty; D1, C1 not attacked)
        if ((castling_rights_ & 2) &&
            !(occ & ((1ULL << D1) | (1ULL << C1) | (1ULL << B1))) &&
            !is_attacked(D1, them) && !is_attacked(C1, them)) {
            moves.emplace_back(E1, C1, NONE_PIECE);
        }
    } else if (us == BLACK && from == E8 && !in_check(BLACK)) {
        // Kingside: E8 -> G8
        if ((castling_rights_ & 4) &&
            !(occ & ((1ULL << F8) | (1ULL << G8))) &&
            !is_attacked(F8, them) && !is_attacked(G8, them)) {
            moves.emplace_back(E8, G8, NONE_PIECE);
        }
        // Queenside: E8 -> C8
        if ((castling_rights_ & 8) &&
            !(occ & ((1ULL << D8) | (1ULL << C8) | (1ULL << B8))) &&
            !is_attacked(D8, them) && !is_attacked(C8, them)) {
            moves.emplace_back(E8, C8, NONE_PIECE);
        }
    }
}

std::vector<Move> Board::generate_legal_moves() const {
    std::vector<Move> pseudo;
    pseudo.reserve(64);

    generate_pawn_moves(pseudo);
    generate_knight_moves(pseudo);
    generate_bishop_moves(pseudo);
    generate_rook_moves(pseudo);
    generate_queen_moves(pseudo);
    generate_king_moves(pseudo);

    std::vector<Move> legal;
    legal.reserve(pseudo.size());

    for (const auto& m : pseudo) {
        if (!leaves_king_in_check(m)) {
            legal.push_back(m);
        }
    }
    return legal;
}

bool Board::is_legal_move(const Move& m) const {
    auto legals = generate_legal_moves();
    for (const auto& lm : legals) {
        if (lm == m) return true;
    }
    return false;
}

bool Board::make_move(const Move& m) {
    Color us = turn_;
    Color them = !us;
    Color piece_c;
    PieceType pt = piece_at(m.from, piece_c);

    if (pt == NONE_PIECE || piece_c != us) return false;

    // Save history
    BoardStateHistory hist;
    hist.castling_rights = castling_rights_;
    hist.ep_square = ep_square_;
    hist.halfmove_clock = halfmove_clock_;
    hist.fullmove_number = fullmove_number_;
    hist.mover = us;
    hist.hash = hash_;
    Color cap_c;
    hist.captured_piece = piece_at(m.to, cap_c);
    history_.push_back(hist);

    // Remove moving piece from 'from'
    piece_bb_[us][pt] &= ~(1ULL << m.from);

    // Handle captures
    if (hist.captured_piece != NONE_PIECE) {
        piece_bb_[them][hist.captured_piece] &= ~(1ULL << m.to);
        halfmove_clock_ = 0;
    } else {
        halfmove_clock_++;
    }

    if (pt == PAWN) halfmove_clock_ = 0;

    // Handle en-passant capture
    if (pt == PAWN && m.to == ep_square_) {
        int cap_sq = (us == WHITE) ? (m.to - 8) : (m.to + 8);
        piece_bb_[them][PAWN] &= ~(1ULL << cap_sq);
    }

    // Set new en-passant square if double pawn push
    if (pt == PAWN && std::abs(m.to - m.from) == 16) {
        ep_square_ = (us == WHITE) ? (m.from + 8) : (m.from - 8);
    } else {
        ep_square_ = SQ_NONE;
    }

    // Place piece at 'to' (or promotion)
    PieceType placed_pt = (m.promotion != NONE_PIECE) ? m.promotion : pt;
    piece_bb_[us][placed_pt] |= (1ULL << m.to);

    // Castling rook movement
    if (pt == KING) {
        if (m.from == E1 && m.to == G1) { // White Kingside
            piece_bb_[WHITE][ROOK] &= ~(1ULL << H1);
            piece_bb_[WHITE][ROOK] |= (1ULL << F1);
        } else if (m.from == E1 && m.to == C1) { // White Queenside
            piece_bb_[WHITE][ROOK] &= ~(1ULL << A1);
            piece_bb_[WHITE][ROOK] |= (1ULL << D1);
        } else if (m.from == E8 && m.to == G8) { // Black Kingside
            piece_bb_[BLACK][ROOK] &= ~(1ULL << H8);
            piece_bb_[BLACK][ROOK] |= (1ULL << F8);
        } else if (m.from == E8 && m.to == C8) { // Black Queenside
            piece_bb_[BLACK][ROOK] &= ~(1ULL << A8);
            piece_bb_[BLACK][ROOK] |= (1ULL << D8);
        }
        // King move voids castling
        if (us == WHITE) castling_rights_ &= ~3;
        else castling_rights_ &= ~12;
    }

    // Rook moves/captures void castling rights
    if (m.from == A1 || m.to == A1) castling_rights_ &= ~2;
    if (m.from == H1 || m.to == H1) castling_rights_ &= ~1;
    if (m.from == A8 || m.to == A8) castling_rights_ &= ~8;
    if (m.from == H8 || m.to == H8) castling_rights_ &= ~4;

    // Check legality: did this move leave own king in check?
    if (in_check(us)) {
        undo_move(m);
        return false;
    }

    turn_ = them;
    if (turn_ == WHITE) fullmove_number_++;

    // Incremental Zobrist update — apply the move's deltas (~12 XORs) instead
    // of a full-board recompute (12 bitboards scanned). hist.* holds pre-move
    // state; every field above is already updated. undo restores hash_ from
    // hist.hash, so no delta-unwinding is needed here.
    uint64_t h = hash_;
    h ^= ZOBRIST_PIECES[us][pt][m.from];
    h ^= ZOBRIST_PIECES[us][placed_pt][m.to];
    if (hist.captured_piece != NONE_PIECE) {
        h ^= ZOBRIST_PIECES[them][hist.captured_piece][m.to];
    }
    if (pt == PAWN && m.to == hist.ep_square) {
        int cap_sq = (us == WHITE) ? (m.to - 8) : (m.to + 8);
        h ^= ZOBRIST_PIECES[them][PAWN][cap_sq];
    }
    if (hist.ep_square != SQ_NONE) h ^= ZOBRIST_EP[file_of(hist.ep_square)];
    if (ep_square_ != SQ_NONE) h ^= ZOBRIST_EP[file_of(ep_square_)];
    if (pt == KING) {
        if (m.from == E1 && m.to == G1)      h ^= ZOBRIST_PIECES[WHITE][ROOK][H1] ^ ZOBRIST_PIECES[WHITE][ROOK][F1];
        else if (m.from == E1 && m.to == C1) h ^= ZOBRIST_PIECES[WHITE][ROOK][A1] ^ ZOBRIST_PIECES[WHITE][ROOK][D1];
        else if (m.from == E8 && m.to == G8) h ^= ZOBRIST_PIECES[BLACK][ROOK][H8] ^ ZOBRIST_PIECES[BLACK][ROOK][F8];
        else if (m.from == E8 && m.to == C8) h ^= ZOBRIST_PIECES[BLACK][ROOK][A8] ^ ZOBRIST_PIECES[BLACK][ROOK][D8];
    }
    h ^= ZOBRIST_CASTLING[hist.castling_rights & 0xF] ^ ZOBRIST_CASTLING[castling_rights_ & 0xF];
    h ^= ZOBRIST_SIDE; // side always flips on a successful move
    hash_ = h;
    return true;
}

void Board::undo_move(const Move& m) {
    if (history_.empty()) return;
    const auto& hist = history_.back();

    // The side that made the move is recorded in history, so undo works
    // correctly whether or not turn_ has already been flipped (e.g. when
    // make_move self-reverts an illegal move before flipping the side).
    Color us = hist.mover;
    Color them = !us;

    // Restore state
    castling_rights_ = hist.castling_rights;
    ep_square_ = hist.ep_square;
    halfmove_clock_ = hist.halfmove_clock;
    fullmove_number_ = hist.fullmove_number;
    hash_ = hist.hash;

    PieceType placed_pt = (m.promotion != NONE_PIECE) ? m.promotion : piece_type_at(m.to);
    piece_bb_[us][placed_pt] &= ~(1ULL << m.to);

    PieceType original_pt = (m.promotion != NONE_PIECE) ? PAWN : placed_pt;
    piece_bb_[us][original_pt] |= (1ULL << m.from);

    // Restore captured piece
    if (hist.captured_piece != NONE_PIECE) {
        piece_bb_[them][hist.captured_piece] |= (1ULL << m.to);
    }

    // Restore en-passant captured pawn
    if (original_pt == PAWN && m.to == hist.ep_square) {
        int cap_sq = (us == WHITE) ? (m.to - 8) : (m.to + 8);
        piece_bb_[them][PAWN] |= (1ULL << cap_sq);
    }

    // Restore castling rook
    if (original_pt == KING) {
        if (m.from == E1 && m.to == G1) {
            piece_bb_[WHITE][ROOK] &= ~(1ULL << F1);
            piece_bb_[WHITE][ROOK] |= (1ULL << H1);
        } else if (m.from == E1 && m.to == C1) {
            piece_bb_[WHITE][ROOK] &= ~(1ULL << D1);
            piece_bb_[WHITE][ROOK] |= (1ULL << A1);
        } else if (m.from == E8 && m.to == G8) {
            piece_bb_[BLACK][ROOK] &= ~(1ULL << F8);
            piece_bb_[BLACK][ROOK] |= (1ULL << H8);
        } else if (m.from == E8 && m.to == C8) {
            piece_bb_[BLACK][ROOK] &= ~(1ULL << D8);
            piece_bb_[BLACK][ROOK] |= (1ULL << A8);
        }
    }

    turn_ = us;
    history_.pop_back();
}

bool Board::is_checkmate() const {
    return in_check(turn_) && generate_legal_moves().empty();
}

bool Board::is_stalemate() const {
    return !in_check(turn_) && generate_legal_moves().empty();
}

bool Board::is_insufficient_material() const {
    Bitboard occ = occupied();
    int count = popcount64(occ);
    if (count == 2) return true; // K vs K
    if (count == 3) {
        if (piece_bb_[WHITE][KNIGHT] || piece_bb_[BLACK][KNIGHT] ||
            piece_bb_[WHITE][BISHOP] || piece_bb_[BLACK][BISHOP]) {
            return true; // KB vs K or KN vs K
        }
    }
    return false;
}

bool Board::is_threefold_repetition() const {
    int count = 1; // Current position
    int max_lookback = std::min(halfmove_clock_, static_cast<int>(history_.size()));
    for (int i = static_cast<int>(history_.size()) - 2; i >= static_cast<int>(history_.size()) - max_lookback; i -= 2) {
        if (history_[i].hash == hash_) {
            count++;
            if (count >= 3) return true;
        }
    }
    return false;
}

bool Board::is_game_over() const {
    if (halfmove_clock_ >= 100) return true; // 50-move rule
    if (is_threefold_repetition()) return true; // Threefold repetition
    if (is_insufficient_material()) return true;
    if (generate_legal_moves().empty()) return true;
    return false;
}

// Cheap game-over check: only the draw rules that are cheap to test
// (halfmove clock, threefold, insufficient material). Skips the
// `generate_legal_moves().empty()` term — used in MCTS descent where a node
// with children is guaranteed to have legal moves anyway. At the leaf we use
// is_game_over() with legal_moves already computed.
bool Board::is_game_over_fast() const {
    if (halfmove_clock_ >= 100) return true;
    if (is_threefold_repetition()) return true;
    if (is_insufficient_material()) return true;
    return false;
}

// Does making `m` leave the side-to-move's king in check? ~10x cheaper than
// `Board temp = *this; make_move; undo_move` — clones only the 12-piece
// bitboards (~96 bytes, no heap, no history) and applies only the occupancy
// deltas the move causes, then checks attack on the resulting king square.
bool Board::leaves_king_in_check(const Move& m) const {
    Color us = turn_;
    Color them = !us;

    // Find the moving piece type and any captured piece at `to`.
    PieceType pt = NONE_PIECE;
    PieceType captured = NONE_PIECE;
    {
        Bitboard from_bb = 1ULL << m.from;
        Bitboard to_bb = 1ULL << m.to;
        for (int p = 0; p < 6; ++p) {
            if (piece_bb_[us][p] & from_bb) pt = static_cast<PieceType>(p);
            if (piece_bb_[them][p] & to_bb) captured = static_cast<PieceType>(p);
        }
    }
    if (pt == NONE_PIECE) return true; // not a movable piece — treat as illegal
    PieceType placed_pt = (m.promotion != NONE_PIECE) ? m.promotion : pt;

    // Copy only the piece occupancy bitboards (small, no heap, no history).
    std::array<std::array<Bitboard, 6>, 2> bb = piece_bb_;
    const Bitboard from_mask = ~(1ULL << m.from);
    const Bitboard to_mask = 1ULL << m.to;

    bb[us][pt] &= from_mask;
    if (captured != NONE_PIECE) bb[them][captured] &= ~to_mask;

    // En-passant capture: pawn moved to the ep square diagonally without a
    // piece present — the pawn being removed sits behind the ep target.
    if (pt == PAWN && m.to == ep_square_ && ep_square_ != SQ_NONE) {
        int cap_sq = (us == WHITE) ? (m.to - 8) : (m.to + 8);
        bb[them][PAWN] &= ~(1ULL << cap_sq);
    }

    bb[us][placed_pt] |= to_mask;

    // Castling moves the rook too.
    if (pt == KING) {
        if (m.from == E1 && m.to == G1) {
            bb[WHITE][ROOK] &= ~(1ULL << H1);
            bb[WHITE][ROOK] |= (1ULL << F1);
        } else if (m.from == E1 && m.to == C1) {
            bb[WHITE][ROOK] &= ~(1ULL << A1);
            bb[WHITE][ROOK] |= (1ULL << D1);
        } else if (m.from == E8 && m.to == G8) {
            bb[BLACK][ROOK] &= ~(1ULL << H8);
            bb[BLACK][ROOK] |= (1ULL << F8);
        } else if (m.from == E8 && m.to == C8) {
            bb[BLACK][ROOK] &= ~(1ULL << A8);
            bb[BLACK][ROOK] |= (1ULL << D8);
        }
    }

    // Locate the king square in the post-move bitboard set.
    Bitboard king_bb = bb[us][KING];
    if (!king_bb) return true; // king missing -> treat as unsafe
    int king_sq = ctz64(king_bb);

    // Check whether `them` attacks `king_sq` on the post-move position.
    // Reuse the same static attack tables as is_attacked but against `bb`
    // (not `piece_bb_`).
    auto attacked_by = [&](int sq, Color attacker) -> bool {
        Bitboard occ = 0;
        for (int c = 0; c < 2; ++c) for (int p = 0; p < 6; ++p) occ |= bb[c][p];

        Color defender = !attacker;
        if (PAWN_ATTACKS[defender][sq] & bb[attacker][PAWN]) return true;
        if (KNIGHT_ATTACKS[sq] & bb[attacker][KNIGHT]) return true;
        if (KING_ATTACKS[sq] & bb[attacker][KING]) return true;

        Bitboard bq = bb[attacker][BISHOP] | bb[attacker][QUEEN];
        if (bq && (sliding_attacks(sq, occ, BISHOP_DIRS, 4) & bq)) return true;

        Bitboard rq = bb[attacker][ROOK] | bb[attacker][QUEEN];
        if (rq && (sliding_attacks(sq, occ, ROOK_DIRS, 4) & rq)) return true;
        return false;
    };

    return attacked_by(king_sq, them);
}

std::string Board::result() const {
    if (is_checkmate()) {
        return (turn_ == WHITE) ? "0-1" : "1-0";
    }
    if (is_game_over()) {
        return "1/2-1/2";
    }
    return "*";
}

void Board::print() const {
    std::cout << "\n  +-----------------+\n";
    for (int r = 7; r >= 0; --r) {
        std::cout << (r + 1) << " | ";
        for (int f = 0; f < 8; ++f) {
            Color c;
            PieceType pt = piece_at(make_square(f, r), c);
            if (pt == NONE_PIECE) {
                std::cout << ". ";
            } else {
                static const char pieces_str[] = "pnbrqk";
                char ch = pieces_str[pt];
                if (c == WHITE) ch = std::toupper(ch);
                std::cout << ch << " ";
            }
        }
        std::cout << "|\n";
    }
    std::cout << "  +-----------------+\n";
    std::cout << "    a b c d e f g h\n";
    std::cout << "  Turn: " << ((turn_ == WHITE) ? "White" : "Black") << "\n\n";
}

} // namespace az
