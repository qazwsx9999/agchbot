"""Convert a C++ engine checkpoint (run_ckpt/checkpoint.pt) back to Python dictionary format.

Usage:
  python cpp/tools/convert_cpp_to_python.py <cpp_checkpoint_path> <out_python_checkpoint_path>
"""

import os
import sys

# Ensure root repository directory is on sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch
from config import Config
from model import ChessNet

def main():
    if len(sys.argv) < 3:
        print("Usage: python convert_cpp_to_python.py <cpp_checkpoint.pt> <out_checkpoint.pt>")
        sys.exit(1)

    cpp_path = sys.argv[1]
    out_path = sys.argv[2]

    print(f"Loading C++ checkpoint from {cpp_path}...")
    jit_m = torch.jit.load(cpp_path, map_location="cpu")
    sd = jit_m.state_dict()

    iteration = sd.get("iteration", torch.tensor(0)).item()
    total_games = sd.get("total_games", torch.tensor(0)).item()

    config = Config()
    model = ChessNet(config)

    clean_sd = {k: v for k, v in sd.items() if k in model.state_dict()}
    model.load_state_dict(clean_sd)

    out_dict = {
        "iteration": iteration,
        "total_games": total_games,
        "model_state_dict": model.state_dict(),
        "trainer_state_dict": {"optimizer": {}},
        "config": config.to_dict(),
    }

    torch.save(out_dict, out_path)
    print(f"Saved Python checkpoint to {out_path} (Iteration: {iteration}, Games: {total_games})")

if __name__ == "__main__":
    main()
