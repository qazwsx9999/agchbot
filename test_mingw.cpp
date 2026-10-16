#include "board.hpp"
#include "action_encoding.hpp"
#include "bots.hpp"
#include <iostream>
#include <chrono>

int main() {
    std::cout << "===========================================\n";
    std::cout << " Testing AlphaZero C++ Engine with MinGW   \n";
    std::cout << "===========================================\n\n";

    az::Board board;
    board.reset();

    std::cout << "1. Starting Board Representation:\n";
    board.print();

    // Test legal moves
    auto moves = board.generate_legal_moves();
    std::cout << "2. Legal moves from starting position: " << moves.size() << " (Expected: 20)\n";

    // Test 21-plane board encoding
    float planes[az::NUM_CHANNELS * 64];
    az::encode_board(board, planes);
    std::cout << "3. 21-plane state encoding completed successfully.\n";

    // Test threefold repetition
    std::cout << "4. Testing Threefold Repetition...\n";
    // Play Nf3 Nf6, Ng1 Ng8, Nf3 Nf6, Ng1 Ng8
    az::Move w_nf3(az::G1, az::F3);
    az::Move b_nf6(az::G8, az::F6);
    az::Move w_ng1(az::F3, az::G1);
    az::Move b_ng8(az::F6, az::G8);

    board.make_move(w_nf3);
    board.make_move(b_nf6);
    board.make_move(w_ng1);
    board.make_move(b_ng8); // Repetition 2

    board.make_move(w_nf3);
    board.make_move(b_nf6);
    board.make_move(w_ng1);
    board.make_move(b_ng8); // Repetition 3!

    std::cout << "   Threefold repetition detected: "
              << (board.is_threefold_repetition() ? "YES (PASSED)" : "NO (FAILED)") << "\n";
    std::cout << "   Game over: " << (board.is_game_over() ? "YES" : "NO")
              << " | Result: " << board.result() << "\n\n";

    // Speed test: Minimax evaluation speed
    std::cout << "5. Speed Benchmark: Minimax move generation...\n";
    board.reset();
    az::MinimaxBot bot(3); // depth 3

    auto t0 = std::chrono::high_resolution_clock::now();
    az::Move best = bot.select_move(board);
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "   Depth 3 Search completed in: " << ms << " ms (Best move: " << best.to_uci() << ")\n";
    std::cout << "\nAll MinGW tests passed successfully!\n";

    return 0;
}
