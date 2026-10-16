#pragma once

#include "types.hpp"
#include <string>
#include <vector>
#include <array>

namespace az {

struct BoardStateHistory {
    uint8_t castling_rights;
    int ep_square;
    int halfmove_clock;
    int fullmove_number;
    Color mover;            // Side that made the move (for robust undo)
    PieceType captured_piece;
    Bitboard hash;
};

class Board {
public:
    Board();
    explicit Board(const std::string& fen);

    void reset();
    bool set_fen(const std::string& fen);
    std::string to_fen() const;

    // Piece access
    Bitboard pieces(PieceType pt, Color c) const { return piece_bb_[c][pt]; }
    Bitboard color_bb(Color c) const;
    Bitboard occupied() const;
    PieceType piece_at(int sq, Color& out_color) const;
    PieceType piece_type_at(int sq) const;

    // State getters
    Color turn() const { return turn_; }
    int ep_square() const { return ep_square_; }
    int halfmove_clock() const { return halfmove_clock_; }
    int fullmove_number() const { return fullmove_number_; }
    bool has_kingside_castling_rights(Color c) const;
    bool has_queenside_castling_rights(Color c) const;

    // Checks and Attacks
    bool is_attacked(int sq, Color attacker_color) const;
    bool in_check(Color c) const;
    bool is_in_check() const { return in_check(turn_); }

    // Move logic
    std::vector<Move> generate_legal_moves() const;
    bool is_legal_move(const Move& m) const;
    bool make_move(const Move& m);
    void undo_move(const Move& m);

    // Copy only position fields (no move history). ~10x cheaper than copy
    // ctor when `history_` is long — used inside MCTS where only move-gen
    // legality tests (which push/pop a local entry) matter, not the record.
    Board clone_lean() const;

    // Terminal detection
    bool is_game_over() const;         // full check: draw rules + no legal moves
    bool is_game_over_fast() const;    // cheap path: draw rules only (no move gen)

    // Fast legality: does making `m` leave the side-to-move's king in check?
    // ~10x cheaper than copy-ctor + make_move: applies only occupancy changes
    // (piece placement, captures, ep, castling rook) to local bitboard copies
    // (~96 bytes, no heap, no history) — skipping ep/castling/halfmove/
    // history/hash bookkeeping entirely.
    bool leaves_king_in_check(const Move& m) const;

    // Assign all state from another board. `history_.assign` keeps existing
    // vector capacity in the common case, avoiding a fresh ~8KB heap copy on
    // every MCTS scratch-board assignment.
    void copy_from(const Board& o);
    std::string result() const; // "1-0", "0-1", "1/2-1/2", or "*"
    bool is_checkmate() const;
    bool is_stalemate() const;
    bool is_threefold_repetition() const;
    bool is_insufficient_material() const;

    uint64_t zobrist_hash() const { return hash_; }
    void print() const;

private:
    std::array<std::array<Bitboard, 6>, 2> piece_bb_;
    Color turn_;
    uint8_t castling_rights_; // bit 0: WK, 1: WQ, 2: BK, 3: BQ
    int ep_square_;
    int halfmove_clock_;
    int fullmove_number_;
    uint64_t hash_;

    std::vector<BoardStateHistory> history_;

    // Position hash kept incrementally in sync inside make_move (XOR deltas —
    // ~12 ops instead of a full-board recompute) and restored from history on
    // undo. Always current; is_threefold_repetition / zobrist_hash read it
    // directly.
    uint64_t compute_hash() const;

    // Move generation helpers
    void generate_pawn_moves(std::vector<Move>& moves) const;
    void generate_knight_moves(std::vector<Move>& moves) const;
    void generate_bishop_moves(std::vector<Move>& moves) const;
    void generate_rook_moves(std::vector<Move>& moves) const;
    void generate_queen_moves(std::vector<Move>& moves) const;
    void generate_king_moves(std::vector<Move>& moves) const;
};

} // namespace az
