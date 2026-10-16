#pragma once

#include "board.hpp"
#include <vector>

namespace az {

// Encodes a Board into 18 x 8 x 8 float planes from current player's perspective.
// out_planes must point to a float buffer of size 18 * 8 * 8 = 1152 floats.
void encode_board(const Board& board, float* out_planes);

// Encodes a legal Move into an action index [0, 4671].
int encode_action(const Move& move, const Board& board);

// Decodes an action index [0, 4671] into a legal Move.
Move decode_action(int action, const Board& board);

// Fills an array of size 4672 with true for legal moves, false otherwise.
void get_legal_action_mask(const Board& board, bool* out_mask);

} // namespace az
