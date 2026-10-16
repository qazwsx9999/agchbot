#pragma once

#include "model.hpp"
#include "bots.hpp"
#include "config.hpp"
#include <string>
#include <vector>

namespace az {

struct EvalResult {
    int wins = 0;
    int losses = 0;
    int draws = 0;
    float win_rate = 0.0f;
};

struct OpeningPosition {
    std::string name;
    std::string fen;
};

// Returns the standard 12 paired diverse opening book positions
std::vector<OpeningPosition> get_opening_book();

class Evaluator {
public:
    Evaluator(const Config& cfg, torch::Device device);

    EvalResult evaluate_vs_bot(ChessNet& model, Bot& bot, int num_games, int simulations);

private:
    Config config_;
    torch::Device device_;

    int play_match(
        ChessNet& model,
        Bot& bot,
        Color ai_color,
        const std::string& start_fen,
        int simulations,
        int& out_moves
    );
};

} // namespace az
