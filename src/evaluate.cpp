#include "evaluate.hpp"
#include "mcts.hpp"
#include "parallel_mcts.hpp"
#include "action_encoding.hpp"
#include <iostream>
#include <iomanip>

namespace az {

std::vector<OpeningPosition> get_opening_book() {
    return {
        {"Standard            ", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"},
        {"Open Game (e4 e5)   ", "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2"},
        {"Sicilian (e4 c5)    ", "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2"},
        {"French (e4 e6)      ", "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"},
        {"Caro-Kann (e4 c6)   ", "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"},
        {"Queen's Pawn (d4 d5)", "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq d6 0 2"},
        {"Indian Game (d4 Nf6)", "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 1 2"},
        {"English (c4 e5)     ", "rnbqkbnr/pppp1ppp/8/4p3/2P5/8/PP1PPPPP/RNBQKBNR w KQkq e6 0 2"},
        {"Reti (Nf3 d5)       ", "rnbqkbnr/ppp1pppp/8/3p4/8/5N2/PPPPPPPP/RNBQKB1R w KQkq d6 0 2"},
        {"Scandinavian (e4 d5)", "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2"},
        {"Dutch (d4 f5)       ", "rnbqkbnr/ppppp1pp/8/5p2/3P4/8/PPP1PPPP/RNBQKBNR w KQkq f6 0 2"},
        {"Alekhine (e4 Nf6)   ", "rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2"}
    };
}

Evaluator::Evaluator(const Config& cfg, torch::Device device)
    : config_(cfg), device_(device) {}

int Evaluator::play_match(
    ChessNet& model,
    Bot& bot,
    Color ai_color,
    const std::string& start_fen,
    int simulations,
    int& out_moves
) {
    Board board(start_fen);
    Config eval_cfg = config_;
    eval_cfg.num_simulations = simulations;
    MCTS mcts(eval_cfg);
    ParallelMCTS pmcts(eval_cfg);
    int search_threads = config_.search_threads;

    out_moves = 0;
    const int MAX_MOVES = config_.max_game_length;

    while (!board.is_game_over() && out_moves < MAX_MOVES) {
        Move m;
        if (board.turn() == ai_color) {
            auto probs = (search_threads > 1)
                ? pmcts.search(board, model, device_, search_threads, false)
                : mcts.search(board, model, device_, false);
            int action = MCTS::select_action(probs, 0.0f); // greedy
            m = decode_action(action, board);
        } else {
            m = bot.select_move(board);
        }

        if (m.is_null() || !board.make_move(m)) {
            break;
        }
        out_moves++;
    }

    std::string res = board.result();
    if (res == "1/2-1/2" || out_moves >= MAX_MOVES) return 0; // Draw
    if ((res == "1-0" && ai_color == WHITE) || (res == "0-1" && ai_color == BLACK)) {
        return 1; // AI Win
    }
    return -1; // AI Loss
}

EvalResult Evaluator::evaluate_vs_bot(ChessNet& model, Bot& bot, int num_games, int simulations) {
    model->eval();
    EvalResult result;
    auto openings = get_opening_book();

    std::cout << "  [Eval] Playing " << num_games << " games vs bot ("
              << simulations << " MCTS sims/move) ...\n";

    int games_played = 0;
    size_t opening_idx = 0;

    while (games_played < num_games) {
        const auto& op = openings[opening_idx % openings.size()];

        // Game 1: AI as White
        if (games_played < num_games) {
            int moves = 0;
            int outcome = play_match(model, bot, WHITE, op.fen, simulations, moves);
            games_played++;

            std::string out_str;
            if (outcome == 1) { result.wins++; out_str = "AI WIN (White)      "; }
            else if (outcome == -1) { result.losses++; out_str = "AI LOSS (Black won) "; }
            else { result.draws++; out_str = "DRAW                "; }

            std::cout << "    [Game " << std::setw(2) << games_played << "/" << num_games << "] "
                      << "AI=White [" << op.name << "] -> " << out_str
                      << " (" << std::setw(3) << moves << " moves)\n";
        }

        // Game 2: AI as Black
        if (games_played < num_games) {
            int moves = 0;
            int outcome = play_match(model, bot, BLACK, op.fen, simulations, moves);
            games_played++;

            std::string out_str;
            if (outcome == 1) { result.wins++; out_str = "AI WIN (Black)      "; }
            else if (outcome == -1) { result.losses++; out_str = "AI LOSS (White won) "; }
            else { result.draws++; out_str = "DRAW                "; }

            std::cout << "    [Game " << std::setw(2) << games_played << "/" << num_games << "] "
                      << "AI=Black [" << op.name << "] -> " << out_str
                      << " (" << std::setw(3) << moves << " moves)\n";
        }

        opening_idx++;
    }

    // Win rate = wins / total games (matches Python evaluate.py).
    result.win_rate = (games_played > 0)
        ? (static_cast<float>(result.wins) / games_played)
        : 0.0f;

    std::cout << "  [Eval] Score: W " << result.wins << "  L " << result.losses
              << "  D " << result.draws << "  (win rate " << std::fixed << std::setprecision(1)
              << (100.0f * result.win_rate) << "%)\n";

    return result;
}

} // namespace az
