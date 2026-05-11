"""Ensemble eval supporting mixed feature modes (raw / embed / concat) per run.

Usage:
    python3 cnn_transformer/main_arc_cnn_transformer/scripts/ensemble_eval_mixed.py \
        --runs <run_dir1> <run_dir2> ... \
        --data_dir <path/to/processed> --train_csv <path/to/train.csv> \
        --output_path <out>/thresholds_smoothed.json
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
    feature_input_dim,
    _overall_fbeta_from_pair_data,
    _tune_per_lab_action,
)
from metadata import (  # noqa: E402
    build_pair_lookup,
    build_vocabs,
    slot_to_table_index,
    vocab_sizes_per_table,
)


def read_run_config(run_dir: str) -> dict:
    with open(os.path.join(run_dir, "config.json")) as f:
        c = json.load(f)
    features_mode = c.get("features_mode", "raw")
    embeddings_dir = c.get("embeddings_dir") or None
    embed_dim = 0
    if features_mode != "raw":
        if not embeddings_dir:
            raise ValueError(f"{run_dir}: features_mode={features_mode!r} "
                             "but no embeddings_dir in config.json")
        meta_path = os.path.join(embeddings_dir, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"{run_dir}: features_mode={features_mode!r} requires "
                f"{meta_path} (run extract_embeddings first).")
        with open(meta_path) as f:
            meta = json.load(f)
        embed_dim = int(meta["d_model"])
    return {
        "window_size": int(c["window_size"]),
        "stride": int(c["stride"]),
        "features_mode": features_mode,
        "embeddings_dir": embeddings_dir,
        "embed_dim": embed_dim,
        "input_dim": feature_input_dim(features_mode, embed_dim),
    }


def _smooth_box(probs: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return probs
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(probs, size=k, axis=0,
                            mode="nearest").astype(probs.dtype, copy=False)


def _smooth_gauss(probs: np.ndarray, sigma: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(probs, sigma=sigma, axis=0,
                             mode="nearest").astype(probs.dtype, copy=False)


@torch.no_grad()
def compute_pair_probs(
    checkpoint_path: str,
    val_videos: set,
    data_dir: str,
    pair_lookup,
    window_size: int,
    stride: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    meta_vocab_sizes,
    meta_slot_to_table,
    features_mode: str = "raw",
    embeddings_dir: str | None = None,
    embed_dim: int = 0,
    input_dim: int | None = None,
):
    val_ds = BehaviorDataset(
        data_dir,
        window_size=window_size,
        stride=stride,
        max_videos=0,
        allowed_video_dirs=val_videos,
        pair_lookup=pair_lookup,
        meta_unk_p=0.0,
        features_mode=features_mode,
        embeddings_dir=embeddings_dir,
        embed_dim=embed_dim,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    pair_keys = val_ds.pair_keys
    num_classes = val_ds.num_classes
    windows = val_ds.windows

    model = MABeTransformer(
        input_dim=input_dim if input_dim is not None
                  else feature_input_dim(features_mode, embed_dim),
        num_classes=num_classes,
        d_model=256,
        nhead=4,
        num_layers=3,
        dim_feedforward=1024,
        dropout=0.2,
        window_size=window_size,
        meta_vocab_sizes=meta_vocab_sizes,
        meta_slot_to_table=meta_slot_to_table,
    ).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
             for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    prob_sum = [np.zeros((nf, num_classes), dtype=np.float32) for _, _, nf in pair_keys]
    counts = [np.zeros(nf, dtype=np.int32) for _, _, nf in pair_keys]
    labels_buf = [np.zeros((nf, num_classes), dtype=np.float32) for _, _, nf in pair_keys]
    loss_masks = [None] * len(pair_keys)

    counter = 0
    for batch in val_loader:
        x = batch["features"].to(device)
        labels_t = batch["labels"]
        padding_mask = batch["padding_mask"].to(device)
        loss_mask_t = batch["loss_mask"]
        meta_ids = batch.get("meta_ids")
        if meta_ids is not None:
            meta_ids = meta_ids.to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x, padding_mask, meta_ids)
        probs = logits.float().sigmoid().cpu().numpy()
        pad_np = padding_mask.cpu().numpy()
        labels_np = labels_t.numpy()
        mask_np = loss_mask_t.numpy()

        B = probs.shape[0]
        for b in range(B):
            pair_idx, start, _ = windows[counter + b]
            actual_len = int((~pad_np[b]).sum())
            if actual_len <= 0:
                continue
            end = min(start + actual_len, prob_sum[pair_idx].shape[0])
            actual_len = end - start
            if actual_len <= 0:
                continue
            prob_sum[pair_idx][start:end] += probs[b, :actual_len]
            counts[pair_idx][start:end] += 1
            labels_buf[pair_idx][start:end] = labels_np[b, :actual_len]
            if loss_masks[pair_idx] is None:
                loss_masks[pair_idx] = mask_np[b]
        counter += B

    avg_probs = []
    for pi, _ in enumerate(pair_keys):
        c = counts[pi].clip(min=1).astype(np.float32)
        avg_probs.append(prob_sum[pi] / c[:, None])

    return pair_keys, avg_probs, labels_buf, loss_masks, num_classes, val_ds.action_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True,
                        help="Run directories, each containing best_model.pt and config.json")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--train_csv", default="data/raw/train.csv")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--threshold_grid_step", type=float, default=0.02)
    parser.add_argument("--smooth_windows", type=int, nargs="*", default=[1],
                        help="Box-filter widths to sweep (1 = no smoothing).")
    parser.add_argument("--smooth_sigmas", type=float, nargs="*", default=[],
                        help="Gaussian sigmas to additionally sweep.")
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    vocabs = build_vocabs(args.train_csv)
    meta_vocab_sizes = vocab_sizes_per_table(vocabs)
    meta_slot_to_table = slot_to_table_index()
    pair_lookup = build_pair_lookup(args.train_csv, vocabs)

    index_df = pd.read_csv(os.path.join(args.data_dir, "index.csv"))
    unique_videos = index_df["video_dir"].unique()
    rng = np.random.default_rng(args.seed)
    shuffled = unique_videos.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * float(args.val_split))))
    n_val = min(n_val, len(shuffled) - 1)
    val_videos = set(shuffled[:n_val])
    print(f"val_split={args.val_split}, seed={args.seed} → {len(val_videos)} val videos")

    grid = (np.arange(0.05, 0.95 + 1e-6, args.threshold_grid_step)
            if args.threshold_grid_step != 0.02 else DEFAULT_THRESHOLD_GRID)

    per_run = []
    for run_dir in args.runs:
        ckpt = os.path.join(run_dir, "best_model.pt")
        rc = read_run_config(run_dir)
        tag = f"window={rc['window_size']} stride={rc['stride']} mode={rc['features_mode']}"
        if rc["features_mode"] != "raw":
            tag += f" embed_dim={rc['embed_dim']} input_dim={rc['input_dim']}"
        print(f"\n=== {run_dir}  {tag} ===")
        pair_keys, avg_probs, labels_buf, loss_masks, num_classes, action_list = \
            compute_pair_probs(
                ckpt, val_videos, args.data_dir, pair_lookup,
                rc["window_size"], rc["stride"],
                args.batch_size, args.num_workers, device,
                meta_vocab_sizes, meta_slot_to_table,
                features_mode=rc["features_mode"],
                embeddings_dir=rc["embeddings_dir"],
                embed_dim=rc["embed_dim"],
                input_dim=rc["input_dim"],
            )
        pair_data = list(zip(avg_probs, labels_buf, loss_masks))
        fb05 = _overall_fbeta_from_pair_data(pair_data, pair_keys, 0.5, args.beta, None)
        thr = _tune_per_lab_action(pair_data, pair_keys, num_classes, grid, args.beta)
        fbt = _overall_fbeta_from_pair_data(pair_data, pair_keys, 0.5, args.beta, thr)
        print(f"  fb@0.5 = {fb05:.4f}   fb_tuned = {fbt:.4f}")
        per_run.append({
            "run_dir": run_dir,
            "window": rc["window_size"], "stride": rc["stride"],
            "features_mode": rc["features_mode"],
            "embed_dim": rc["embed_dim"], "input_dim": rc["input_dim"],
            "pair_keys": pair_keys, "avg_probs": avg_probs,
            "labels": labels_buf, "loss_masks": loss_masks,
            "num_classes": num_classes, "action_list": action_list,
            "fb_at_0_5": float(fb05), "fb_tuned": float(fbt),
        })

    ref = per_run[0]
    for r in per_run[1:]:
        if r["pair_keys"] != ref["pair_keys"]:
            raise RuntimeError(f"pair_keys mismatch between {ref['run_dir']} and {r['run_dir']} "
                               f"— val split is not deterministic across runs.")
        if r["num_classes"] != ref["num_classes"]:
            raise RuntimeError("num_classes mismatch between runs.")

    n_runs = len(per_run)
    ens_probs = []
    for pi in range(len(ref["pair_keys"])):
        stack = np.stack([r["avg_probs"][pi] for r in per_run], axis=0)
        ens_probs.append(stack.mean(axis=0))

    print("\n=== Per-run summary ===")
    for r in per_run:
        print(f"  {r['run_dir']:60s} fb@0.5={r['fb_at_0_5']:.4f}  fb_tuned={r['fb_tuned']:.4f}")
    print(f"\n=== Ensemble ({n_runs} models, equal-weight avg of probs) ===")

    smooth_specs = [("box", w) for w in args.smooth_windows] + \
                   [("gauss", s) for s in args.smooth_sigmas]
    best = {"fb_tuned": -1.0}
    for kind, k in smooth_specs:
        if kind == "box" and k <= 1:
            sm_probs = ens_probs
            tag = "no smoothing"
        elif kind == "box":
            sm_probs = [_smooth_box(p, k) for p in ens_probs]
            tag = f"box K={k}"
        else:
            sm_probs = [_smooth_gauss(p, k) for p in ens_probs]
            tag = f"gauss σ={k}"
        pair_data = list(zip(sm_probs, ref["labels"], ref["loss_masks"]))
        fb05 = _overall_fbeta_from_pair_data(
            pair_data, ref["pair_keys"], 0.5, args.beta, None)
        thr = _tune_per_lab_action(
            pair_data, ref["pair_keys"], ref["num_classes"], grid, args.beta)
        fbt = _overall_fbeta_from_pair_data(
            pair_data, ref["pair_keys"], 0.5, args.beta, thr)
        print(f"  [{tag:14s}]  fb@0.5={fb05:.4f}  fb_tuned={fbt:.4f}")
        if fbt > best["fb_tuned"]:
            best = {"tag": tag, "kind": kind, "k": k,
                    "fb_at_0_5": float(fb05), "fb_tuned": float(fbt),
                    "thresholds": thr}

    best_single = max(r["fb_tuned"] for r in per_run)
    print(f"\nBest smoothing: {best['tag']}  fb_tuned={best['fb_tuned']:.4f}")
    print(f"Δ vs best single ({best_single:.4f}): {best['fb_tuned'] - best_single:+.4f}")
    ens_fb05 = best["fb_at_0_5"]
    ens_fbt = best["fb_tuned"]
    ens_thr = best["thresholds"]

    out_path = args.output_path
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    action_list = ref["action_list"]
    with open(out_path, "w") as f:
        json.dump({
            "beta": args.beta,
            "default_threshold": 0.5,
            "ensemble_runs": [
                {"run_dir": r["run_dir"], "window": r["window"], "stride": r["stride"],
                 "features_mode": r["features_mode"], "embed_dim": r["embed_dim"],
                 "input_dim": r["input_dim"],
                 "fb_at_0_5": r["fb_at_0_5"], "fb_tuned": r["fb_tuned"]}
                for r in per_run
            ],
            "smoothing": {"kind": best.get("kind"), "k": best.get("k"),
                          "tag": best.get("tag")},
            "overall_fb_at_0_5": float(ens_fb05),
            "overall_fb_tuned": float(ens_fbt),
            "action_list": [str(a) for a in action_list],
            "per_lab_action": [
                {"lab_id": lab, "action_idx": int(a),
                 "action": str(action_list[a]), "threshold": float(t)}
                for (lab, a), t in sorted(ens_thr.items())
            ],
        }, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
