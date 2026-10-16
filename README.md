# AlphaZero Chess (C++ Engine)

A complete, high-performance C++17 implementation of the AlphaZero Chess reinforcement learning engine, matching the exact algorithms, board encodings, action spaces, and network architecture of the Python project.

---

## ⚡ Key Performance Advantages

1. **64-bit Bitboard Engine**: Move generation and board representation use low-level bitboards with bitwise operations (`popcnt`, `ctzll`), making board simulation **up to 1,000× faster** than `python-chess`.
2. **Native Memory Management**: Zero Python garbage-collection pauses during deep MCTS search trees.
3. **LibTorch Integration**: Uses PyTorch's official C++ frontend (`torch::nn::Module`), providing direct hardware acceleration on NVIDIA GPUs (CUDA) with minimal CPU-GPU dispatch latency.

---

## 📂 Project Architecture

```
cpp/
├── CMakeLists.txt              # CMake build configuration with LibTorch & CUDA
├── README.md                   # Build and usage documentation
├── include/
│   ├── types.hpp               # Bitboard constants, Piece, Square, Move definitions
│   ├── board.hpp               # 64-bit Bitboard chess board, legal move generator, FEN parser
│   ├── action_encoding.hpp     # 18-plane board encoder & 4,672-action encoder/decoder
│   ├── config.hpp              # Configuration parameters matching config.py
│   ├── model.hpp               # Dual-head ResNet (10 ResBlocks, 128 filters) in LibTorch
│   ├── mcts.hpp                # MCTS search with PUCT and Dirichlet noise
│   ├── replay_buffer.hpp       # Ring replay buffer with mini-batch tensor sampling
│   ├── trainer.hpp             # SGD optimizer (momentum 0.9, weight decay 1e-4)
│   ├── bots.hpp                # MinimaxBot with Piece-Square Tables & RandomBot
│   ├── evaluate.hpp            # 12 diverse opening positions benchmark
│   └── self_play.hpp           # Self-play game generator
└── src/
    ├── board.cpp
    ├── action_encoding.cpp
    ├── model.cpp
    ├── mcts.cpp
    ├── replay_buffer.cpp
    ├── trainer.cpp
    ├── bots.cpp
    ├── evaluate.cpp
    ├── self_play.cpp
    └── main.cpp                # CLI entry point (training, evaluation, interactive play)
```

---

## 🛠️ Build Instructions

### 1. Prerequisites
- A modern C++17 compiler (GCC 9+, Clang 10+, or MSVC 2019+).
- **CMake** (3.18 or newer).
- **LibTorch** (PyTorch C++ library).

### 2. Downloading LibTorch
Download the pre-built LibTorch package from [pytorch.org](https://pytorch.org):
* **Linux (CUDA 12.4)**:
  ```bash
  wget https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.6.0%2Bcu124.zip
  unzip libtorch-cxx11-abi-shared-with-deps-2.6.0+cu124.zip
  ```
* **Windows (CUDA 12.4)**:
  Download the Windows Release zip from `pytorch.org` and extract to `C:\libtorch`.

### 3. Compiling with CMake

#### On Linux / Kaggle:
```bash
cd cpp
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release -j$(nproc)
```

#### On Windows:
```powershell
cd cpp
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="C:\libtorch" ..
cmake --build . --config Release
```

The compiled executable `alphazero_chess` (or `alphazero_chess.exe`) will be generated.

---

## 🚀 Usage

### 1. Training from Scratch
```bash
./alphazero_chess --iterations 500 --games 50 --device cuda
```

### 2. Resuming an Existing Checkpoint
```bash
./alphazero_chess --resume --iterations 500 --device cuda
```

### 3. Training on All CPU Cores (Multi-threaded)
```bash
./alphazero_chess --iterations 500 --games 50 --device cpu --workers 12
```

### 4. Evaluating vs Minimax
```bash
./alphazero_chess --resume --eval-only --eval-games 10 --minimax-depth 1
```

### 5. Interactive Terminal Play Against the AI
```bash
./alphazero_chess --resume --play --simulations 200
```
Enter moves in standard UCI format (e.g. `e2e4`, `g1f3`, `e7e8q`).

---

## 🎯 Parity With the Python Reference

All hyperparameter **defaults mirror `config.py`** so the engine reproduces the exact
same training behaviour (just faster): `num_simulations=800`, `c_puct=1.5`,
`num_self_play_games=100`, `num_iterations=100`, `max_game_length=512`,
`min_replay_buffer_size=2048`, `num_epochs=1`, `eval_opponent=random`,
`minimax_depth=3`, `eval_interval=10`, `eval_games=20`, `device=auto`.

Behavioural parity details:
- **Training gate** — training is skipped until the buffer holds `min(min_replay_buffer_size, batch_size)` samples, like Python.
- **Fully resumable** — a checkpoint saves the model (`checkpoint.pt`), the optimizer/momentum state (`checkpoint.pt.opt`), and the replay buffer (`replay_buffer.bin`). `--resume` restores all three. (Resume on the same device type you trained on.)
- **Evaluation** — paired games across the same 12 openings, greedy (no Dirichlet noise), `win_rate = wins / games`.

Additional CLI flags: `--c-puct`, `--learning-rate`, `--num-epochs`,
`--max-game-length`, `--min-replay-buffer-size`, `--checkpoint-interval`,
`--eval-opponent random|minimax`.
