"""Convert a Python AlphaZero checkpoint into the C++ engine's format.

Produces, in <out_dir>:
  weights.bin        - flat (name, shape, float32 data) dump of the model
                       parameters/buffers, read by the C++ `import_ckpt` tool.
  meta.txt           - iteration and total_games.
  replay_buffer.bin  - the replay buffer in the C++ ReplayBuffer binary format.

The C++ `import_ckpt` tool then turns weights.bin + meta.txt into a native
C++ `checkpoint.pt`. replay_buffer.bin is already in the final C++ format.

Note: the SGD momentum buffers are intentionally NOT carried over (the C++
engine simply starts with fresh momentum). Everything that matters for
continuing training - the learned weights and the replay buffer - is kept.

Usage:
  python convert_checkpoint.py <python_checkpoint_dir> <out_dir> [--buffer-capacity N]
"""

import argparse
import os
import struct

import numpy as np
import torch


def write_weights(model_state_dict, out_path):
    # Keep only floating-point tensors (skip e.g. BatchNorm num_batches_tracked,
    # which is a Long counter the C++ side does not need).
    items = [(k, v) for k, v in model_state_dict.items()
             if torch.is_tensor(v) and v.dtype.is_floating_point]
    with open(out_path, "wb") as f:
        f.write(struct.pack("<i", len(items)))
        for name, t in items:
            name_b = name.encode("utf-8")
            f.write(struct.pack("<i", len(name_b)))
            f.write(name_b)
            t = t.detach().cpu().contiguous().float()
            f.write(struct.pack("<i", t.dim()))
            for d in t.shape:
                f.write(struct.pack("<i", int(d)))
            f.write(t.numpy().astype("<f4").tobytes())
    return len(items)


def write_replay_buffer(npz_path, out_path, capacity):
    if not os.path.isfile(npz_path):
        print(f"  [warn] {npz_path} not found - skipping replay buffer")
        return 0
    d = np.load(npz_path)
    n = int(d["size"][0]) if "size" in d.files else d["states"].shape[0]
    states = d["states"][:n].astype("<f4")      # (n, 18, 8, 8), C-order = plane,rank,file
    policies = d["policies"][:n].astype("<f4")   # (n, 4672)
    values = d["values"][:n].astype("<f4")       # (n,)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<Q", n))            # size_t cur_sz
        f.write(struct.pack("<Q", int(capacity)))# size_t capacity (placeholder; C++ uses its own)
        for i in range(n):
            f.write(np.ascontiguousarray(states[i]).tobytes())
            f.write(np.ascontiguousarray(policies[i]).tobytes())
            f.write(struct.pack("<f", float(values[i])))
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir", help="dir with checkpoint.pt and replay_buffer.npz")
    ap.add_argument("out_dir")
    ap.add_argument("--buffer-capacity", type=int, default=50000)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.ckpt_dir, "checkpoint.pt")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    msd = ck["model_state_dict"]
    iteration = int(ck.get("iteration", 0))
    total_games = int(ck.get("total_games", 0))

    n_w = write_weights(msd, os.path.join(args.out_dir, "weights.bin"))
    with open(os.path.join(args.out_dir, "meta.txt"), "w") as f:
        f.write(f"iteration {iteration}\n")
        f.write(f"total_games {total_games}\n")
    n_b = write_replay_buffer(
        os.path.join(args.ckpt_dir, "replay_buffer.npz"),
        os.path.join(args.out_dir, "replay_buffer.bin"),
        args.buffer_capacity,
    )

    print(f"Converted checkpoint:")
    print(f"  iteration    = {iteration}")
    print(f"  total_games  = {total_games}")
    print(f"  weights      = {n_w} tensors -> weights.bin")
    print(f"  replay buffer= {n_b} samples -> replay_buffer.bin")
    print(f"  output dir   = {args.out_dir}")


if __name__ == "__main__":
    main()
