"""Load a saved checkpoint and dump tuned per-(lab,action) thresholds.

Usage:
    python cnn_transformer/self_pair_model/scripts/tune_thresholds.py --checkpoint <run>/best_model.pt --data_dir <path/to/processed> \\
        --train_csv data/train.csv --window_size 128 --stride 64
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

_MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
sys.path.insert(0, str(_MODEL_DIR))

from model import (  # noqa: E402
    BehaviorDataset,
    MABeTransformer,
    NUM_FEATURES,
    DEFAULT_THRESHOLD_GRID,
    validate,
)
from metadata import (  # noqa: E402
    build_pair_lookup,
    build_vocabs,
    parse_video_dir,
    slot_to_table_index,
    vocab_sizes_per_table,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--train_csv", default="data/train.csv")
    parser.add_argument("--window_size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--threshold_grid_step", type=float, default=0.02)
    parser.add_argument("--output_path", default=None,
                        help="Where to save thresholds.json (default: alongside checkpoint)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not os.path.exists(args.train_csv):
        raise FileNotFoundError(f"--train_csv not found: {args.train_csv}")
    print(f"Building metadata vocabs from {args.train_csv}")
    vocabs = build_vocabs(args.train_csv)
    meta_vocab_sizes = vocab_sizes_per_table(vocabs)
    meta_slot_to_table = slot_to_table_index()
    pair_lookup = build_pair_lookup(args.train_csv, vocabs)

    index_df = pd.read_csv(os.path.join(args.data_dir, "index.csv"))
    unique_videos = index_df["video_dir"].unique()
    rng = np.random.default_rng(args.seed)
    shuffled = unique_videos.copy()
    rng.shuffle(shuffled)
    if len(shuffled) <= 1 or args.val_split <= 0:
        raise ValueError("Need >=2 videos and val_split > 0 to evaluate.")
    n_val = max(1, int(round(len(shuffled) * float(args.val_split))))
    n_val = min(n_val, len(shuffled) - 1)
    val_videos = set(shuffled[:n_val])
    print(f"val_split={args.val_split}, seed={args.seed} → {len(val_videos)} val videos")

    val_ds = BehaviorDataset(
        args.data_dir,
        window_size=args.window_size,
        stride=args.stride,
        max_videos=0,
        allowed_video_dirs=val_videos,
        pair_lookup=pair_lookup,
        meta_unk_p=0.0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    num_classes = val_ds.num_classes

    model = MABeTransformer(
        input_dim=NUM_FEATURES,
        num_classes=num_classes,
        d_model=256,
        nhead=4,
        num_layers=3,
        dim_feedforward=1024,
        dropout=0.2,
        window_size=args.window_size,
        meta_vocab_sizes=meta_vocab_sizes,
        meta_slot_to_table=meta_slot_to_table,
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    print(f"Loaded checkpoint: {args.checkpoint}")

    grid = (np.arange(0.05, 0.95 + 1e-6, args.threshold_grid_step)
            if args.threshold_grid_step != 0.02 else DEFAULT_THRESHOLD_GRID)
    print(f"Threshold grid: {len(grid)} values ∈ [{grid[0]:.2f}, {grid[-1]:.2f}]")

    val_loss, fb_05, fb_tuned, thresholds = validate(
        model, val_loader, device, beta=args.beta, threshold_grid=grid,
    )
    print(f"\nVal loss: {val_loss:.4f}")
    print(f"Fb @ 0.5 (aggregated): {fb_05:.4f}")
    print(f"Fb @ tuned per-(lab,action): {fb_tuned:.4f}")
    print(f"Δ = {fb_tuned - fb_05:+.4f}")

    out_path = args.output_path or os.path.join(
        os.path.dirname(args.checkpoint) or ".", "thresholds.json"
    )
    with open(out_path, "w") as f:
        json.dump({
            "beta": args.beta,
            "default_threshold": 0.5,
            "overall_fb_at_0_5": float(fb_05),
            "overall_fb_tuned": float(fb_tuned),
            "action_list": [str(a) for a in val_ds.action_list],
            "per_lab_action": [
                {"lab_id": lab, "action_idx": int(a),
                 "action": str(val_ds.action_list[a]), "threshold": float(t)}
                for (lab, a), t in sorted(thresholds.items())
            ],
        }, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
