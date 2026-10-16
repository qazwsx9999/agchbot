#!/usr/bin/env bash
# ==============================================================================
# Automated setup and runner for Kaggle (GPU with CUDA or CPU with all cores)
# ==============================================================================
set -e

WORK_DIR="/kaggle/working"
cd "$WORK_DIR"

# 1. Detect if GPU is available
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    DEVICE="cuda"
    echo "=========================================================="
    echo "  AlphaZero C++: NVIDIA GPU Detected! (CUDA Mode)"
    echo "=========================================================="
else
    DEVICE="cpu"
    echo "=========================================================="
    echo "  AlphaZero C++: CPU Mode ($(nproc) cores)"
    echo "=========================================================="
fi

# 2. Setup LibTorch / PyTorch C++ Library
PY_TORCH_DIR=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)" 2>/dev/null || echo "")
PY_TORCH_LIB=$(python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2>/dev/null || echo "")

if [ -n "$PY_TORCH_DIR" ] && [ -d "$PY_TORCH_DIR" ]; then
    echo "[1/4] Using pre-installed PyTorch ($PY_TORCH_DIR)..."
    CMAKE_PREFIX="$PY_TORCH_DIR"
    export LD_LIBRARY_PATH="$PY_TORCH_LIB:$LD_LIBRARY_PATH"
elif [ -d "$WORK_DIR/libtorch" ]; then
    echo "[1/4] Using existing $WORK_DIR/libtorch..."
    CMAKE_PREFIX="$WORK_DIR/libtorch"
    export LD_LIBRARY_PATH="$WORK_DIR/libtorch/lib:$LD_LIBRARY_PATH"
else
    if [ "$DEVICE" = "cuda" ]; then
        echo "[1/4] Downloading LibTorch CUDA 12.4..."
        wget -q https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.6.0%2Bcu124.zip -O libtorch.zip
    else
        echo "[1/4] Downloading LibTorch CPU..."
        wget -q https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.6.0%2Bcpu.zip -O libtorch.zip
    fi
    echo "      Extracting LibTorch..."
    unzip -q -o libtorch.zip
    rm -f libtorch.zip
    CMAKE_PREFIX="$WORK_DIR/libtorch"
    export LD_LIBRARY_PATH="$WORK_DIR/libtorch/lib:$LD_LIBRARY_PATH"
fi

# 3. Build C++ Engine
echo "[2/4] Building AlphaZero C++ Engine..."
mkdir -p "$WORK_DIR/cpp/build"
cd "$WORK_DIR/cpp/build"
cmake -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX" -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release -j"$(nproc)"
cd "$WORK_DIR"

# 4. Ensure checkpoint exists in run_ckpt/
if [ ! -f "$WORK_DIR/run_ckpt/checkpoint.pt" ]; then
    echo "[3/4] Checkpoint run_ckpt not found, converting from checkpoints/..."
    mkdir -p "$WORK_DIR/converted" "$WORK_DIR/run_ckpt"
    python3 "$WORK_DIR/cpp/tools/convert_checkpoint.py" "$WORK_DIR/checkpoints" "$WORK_DIR/converted"
    "$WORK_DIR/cpp/build/import_ckpt" "$WORK_DIR/converted/weights.bin" "$WORK_DIR/converted/meta.txt" "$WORK_DIR/run_ckpt/checkpoint.pt"
    cp "$WORK_DIR/converted/replay_buffer.bin" "$WORK_DIR/run_ckpt/"
else
    echo "[3/4] Checkpoint ready at $WORK_DIR/run_ckpt."
fi

# 5. Start training with user options or default options
# Default matches user configuration:
# 650 iterations, 100 sims, 50 games/iter, minimax depth 1 eval every 5 iters (10 eval games)
DEFAULT_ARGS=(
    --num-iterations 650
    --num-simulations 100
    --num-self-play-games 50
    --eval-opponent minimax
    --minimax-depth 1
    --eval-interval 5
    --num-eval-games 10
)

USER_ARGS=("$@")
if [ ${#USER_ARGS[@]} -eq 0 ]; then
    FINAL_ARGS=("${DEFAULT_ARGS[@]}")
else
    FINAL_ARGS=("${USER_ARGS[@]}")
fi

NUM_CORES=$(nproc)

if [ "$DEVICE" = "cuda" ]; then
    NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
    echo "[4/4] Starting training on $NUM_GPUS GPU(s) (CUDA) with Batched MCTS + parallel select..."
    WORKERS_PER_GPU=$(( NUM_CORES / NUM_GPUS ))
    [ "$WORKERS_PER_GPU" -lt 1 ] && WORKERS_PER_GPU=1
    exec "$WORK_DIR/cpp/build/alphazero_chess" \
        --resume \
        --device cuda \
        --gpus "$NUM_GPUS" \
        --parallel-games 50 \
        --select-workers "$WORKERS_PER_GPU" \
        --search-threads "$NUM_CORES" \
        --virtual-loss 1.0 \
        --checkpoint-dir "$WORK_DIR/run_ckpt" \
        "${FINAL_ARGS[@]}"
else
    echo "[4/4] Starting training on CPU ($NUM_CORES cores) with Batched MCTS + parallel select..."
    exec "$WORK_DIR/cpp/build/alphazero_chess" \
        --resume \
        --device cpu \
        --batch-mcts \
        --parallel-games 16 \
        --select-workers "$NUM_CORES" \
        --search-threads "$NUM_CORES" \
        --virtual-loss 1.0 \
        --workers "$NUM_CORES" \
        --checkpoint-dir "$WORK_DIR/run_ckpt" \
        "${FINAL_ARGS[@]}"
fi
