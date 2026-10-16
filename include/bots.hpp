#pragma once

#include "board.hpp"
#include <random>

namespace az {

class Bot {
public:
    virtual ~Bot() = default;
    virtual Move select_move(Board& board) = 0;
};

class RandomBot : public Bot {
public:
    RandomBot();
    Move select_move(Board& board) override;

private:
    std::mt19937 rng_;
};

class MinimaxBot : public Bot {
public:
    explicit MinimaxBot(int depth = 1);
    Move select_move(Board& board) override;

    int evaluate(const Board& board) const;

private:
    int depth_;

    int negamax(Board& board, int depth, int alpha, int beta, int color_sign);
    void order_moves(std::vector<Move>& moves, const Board& board) const;
};

} // namespace az
