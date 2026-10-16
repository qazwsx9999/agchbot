# Running the C++ AlphaZero Engine on Kaggle

This setup allows you to train your AlphaZero chess bot on **GPU (CUDA)** or **all CPU cores** on Kaggle, resuming directly from your current training state (**iteration 359, ~18,000 games played**).

> **Why GPU is Recommended**: AlphaZero uses a 10-block deep residual network. On an NVIDIA GPU (T4 / P100), **Batched MCTS evaluates 32 board states in ~2 ms**, making training **50x to 100x faster than CPU**!

The checkpoint is pre-converted into native C++ format in `run_ckpt/`. All bug fixes (move generation, evaluation tables, multi-threading) are applied.

---

## Step 1: Upload the Zip as a Kaggle Dataset

1. On Kaggle, go to **Datasets** → **New Dataset**.
2. Upload **`alphazero_kaggle_all.zip`** (recommended: contains all code + checkpoint) or **`alphazero_kaggle_v4.zip`** (code only).
3. Name it `alphazero-chess` (or any name you like) and click **Create**.

---

## Step 2: Create a Kaggle Notebook

1. Open a new Kaggle Notebook (**Code** → **New Notebook**).
2. In the right sidebar under **Settings**:
   - **Accelerator**: Select **`GPU T4 x2`** (or `GPU P100`) for maximum speed! *(Or `None` for CPU).*
   - **Internet**: Set to **`On`**.
3. Under **Input** (right sidebar), click **Add Input** and select your uploaded dataset.

---

## Step 3: Run Training (Single Code Cell!)

Copy and paste this into a cell in your Kaggle notebook and press **Run**:

```python
%%bash
# 1. Copy files from dataset into writable working directory
cp -r /kaggle/input/*/cpp /kaggle/working/cpp
cp -r /kaggle/input/*/run_ckpt /kaggle/working/run_ckpt 2>/dev/null || true
cp -r /kaggle/input/*/checkpoints /kaggle/working/checkpoints 2>/dev/null || true

# 2. Run the automated runner script
# By default, it runs your exact requested configuration:
# --resume --device cuda --eval-opponent minimax --minimax-depth 1 --eval-interval 5 \
# --num-eval-games 10 --num-simulations 100 --num-self-play-games 50 --num-iterations 650
bash /kaggle/working/cpp/run_kaggle.sh
```

*(You can also pass custom flags directly, e.g.: `bash /kaggle/working/cpp/run_kaggle.sh --num-iterations 700 --num-simulations 200`)*

The script will:
1. Detect whether an NVIDIA GPU is available (falls back to multi-core CPU if not).
2. Build the C++ engine using Kaggle's PyTorch/LibTorch with `-O3` optimization.
3. Load the checkpoint from `run_ckpt/checkpoint.pt` at **iteration 359**.
4. Run self-play and training with **Batched MCTS** on the GPU (or across all CPU cores).
5. Train up to iteration 500 (or whatever number you specify).

---

## Manual Step-by-Step Commands

If you prefer to run each step in separate notebook cells:

### Cell 1: Copy files
```bash
%%bash
cp -r /kaggle/input/*/cpp /kaggle/working/cpp
cp -r /kaggle/input/*/run_ckpt /kaggle/working/run_ckpt
ls -lh /kaggle/working/run_ckpt
```

### Cell 2: Build C++ Engine
```bash
%%bash
cd /kaggle/working/cpp
mkdir -p build && cd build
# Uses pre-installed PyTorch in Kaggle (with CUDA)
TORCH_PREFIX=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")
cmake -DCMAKE_PREFIX_PATH="$TORCH_PREFIX" -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release -j"$(nproc)"
```

### Cell 3: Run on GPU with Batched MCTS (Exact Match to Your Python Command!)
```bash
%%bash
export LD_LIBRARY_PATH="$(python3 -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))'):$LD_LIBRARY_PATH"
cd /kaggle/working
./cpp/build/alphazero_chess \
    --resume \
    --device cuda \
    --eval-opponent minimax \
    --minimax-depth 1 \
    --eval-interval 5 \
    --num-eval-games 10 \
    --num-simulations 100 \
    --num-self-play-games 50 \
    --num-iterations 650 \
    --parallel-games 32 \
    --checkpoint-dir run_ckpt
```

*(Or for CPU-only: change `--device cuda` to `--device cpu --batch-mcts --parallel-games 16 --workers $(nproc)`)*

---

## Key Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--device` | auto | `cuda` (GPU acceleration) or `cpu` |
| `--gpus` / `--num-gpus` | 1 | CUDA devices to drive in parallel — set to 2 on Kaggle T4 x2 |
| `--parallel-games` | 32 | Batch width for batched MCTS — set equal to `--num-self-play-games` so all games evaluate in one batch |
| `--select-workers` | auto | Threads splitting batched PUCT select-down across games (0 = auto: min(parallel_games, cores)) |
| `--search-threads` | 4 | Threads for a single shared-tree search (ParallelMCTS — used for eval & play) |
| `--virtual-loss` | 1.0 | Virtual-loss penalty magnitude on in-flight search nodes |
| `--num-simulations` | 100 | MCTS simulations per move (100 matches your Python run) |
| `--num-iterations` | 500 | Total iteration target (must be > current checkpoint iteration) |
| `--num-self-play-games` | 100 | Self-play games per iteration |
| `--checkpoint-dir` | run_ckpt | Directory where `checkpoint.pt` and `replay_buffer.bin` are stored |
| `--resume` | - | Resumes from `checkpoint.pt` in `--checkpoint-dir` |

The `run_kaggle.sh` script auto-detects the GPU count and passes `--gpus`, plus
`--select-workers`/`--search-threads`/`--virtual-loss` tuned to the CPU core count —
no flags needed for the default multi-GPU run.

---

## Continuing in Future Sessions

Checkpoints and replay buffers are saved in `/kaggle/working/run_ckpt/`.
When the notebook session finishes, Kaggle saves everything in `/kaggle/working` as the **Notebook Output**.

In your next session:
1. Click **Add Input** → **Notebook Output Files** → select your previous notebook run.
2. Copy `run_ckpt` from input into `/kaggle/working/run_ckpt`.
3. Run Step 3 with a higher `--iterations` target (e.g., 600, 700).
