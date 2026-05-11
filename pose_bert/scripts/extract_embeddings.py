"""Build a per-mouse embedding cache from a PoseBERT checkpoint.

Usage:
    python -m pose_bert.scripts.extract_embeddings --checkpoint <run_dir> --source <data> --output <out> --train_csv <csv>
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pose_bert.model.dataset import InMemoryPretrainDataset, _scan_data_dir  # noqa: E402
from pose_bert.model.metadata import build_lookup  # noqa: E402


VARIANT_MODULES: Dict[str, str] = {
    "raw":          "pose_bert.model.pose_bert",
    "pos_raw":      "pose_bert.model.pose_bert_pos_raw",
    "pos_bins":     "pose_bert.model.pose_bert_pos_bins",
    "vel_bins":     "pose_bert.model.pose_bert_vel_bins",
    "pos_vel_bins": "pose_bert.model.pose_bert_pos_vel_bins",
}


def detect_variant(cfg: dict) -> str:
    """Resolve checkpoint variant from config (prefer model_name, else key heuristics)."""
    name = cfg.get("model_name")
    if name in VARIANT_MODULES:
        return name

    has_pos = "n_pos_bins" in cfg
    has_vel = "n_vel_r_bins" in cfg
    if has_pos and has_vel:
        return "pos_vel_bins"
    if has_vel:
        return "vel_bins"
    if has_pos:
        return "pos_bins"
    return "raw"


def load_model(checkpoint_dir: str, device: torch.device):
    cfg = json.load(open(os.path.join(checkpoint_dir, "config.json")))
    state = torch.load(os.path.join(checkpoint_dir, "best_model.pt"),
                       map_location=device)
    # Prefer config embedded in checkpoint over standalone config.json (may drift).
    if isinstance(state, dict):
        if "config" in state and isinstance(state["config"], dict):
            cfg = state["config"]
        for k in ("model_state_dict", "state_dict"):
            if k in state and isinstance(state[k], dict):
                state = state[k]
                break
    # Strip torch.compile()'s "_orig_mod." prefix so the unwrapped model can load.
    if any(k.startswith("_orig_mod.") for k in state.keys()):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}

    variant = detect_variant(cfg)
    module = importlib.import_module(VARIANT_MODULES[variant])
    model = module.PoseBERT(cfg).to(device).eval()
    model.load_state_dict(state)
    torch.set_float32_matmul_precision("high")
    # torch.compile only optimizes forward(); wrap extract() directly since we call it.
    if device.type == "cuda":
        try:
            model.extract = torch.compile(model.extract, mode="reduce-overhead")
            print("model.extract compiled with torch.compile "
                  "(mode=reduce-overhead, ~30s warmup on first batch)")
        except Exception as e:
            print(f"WARN: torch.compile failed, falling back to eager: {e}")
    return model, cfg, variant


@torch.no_grad()
def extract_all(args) -> None:
    device = torch.device(args.device if args.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    model, cfg, variant = load_model(args.checkpoint, device)
    enc_window = int(cfg["window_size"])
    d_model    = int(cfg["d_model"])
    enc_stride = args.enc_stride or (enc_window // 2)
    print(f"Variant={variant}  d_model={d_model}  encoder_window={enc_window} "
          f"stride={enc_stride}")

    # Filter to annotated videos only; pretraining set has ~7900 unannotated
    # videos that OOM the all-videos load. --no_filter_for_inference bypasses
    # this for Kaggle test-time (test videos aren't in train.csv).
    import gc
    import pandas as pd
    meta_csv_df = pd.read_csv(args.train_csv)
    all_records = _scan_data_dir(args.source)
    if args.no_filter_for_inference:
        records = list(all_records)
        annotated_set = None
        print(f"--no_filter_for_inference: extracting ALL {len(records)} tracks "
              f"from {args.source} (metadata vocabs still come from --train_csv).")
    else:
        annotated_set = set(
            str(v) for v in meta_csv_df.loc[meta_csv_df["behaviors_labeled"].notna(), "video_id"]
        )
        print(f"train.csv: {len(meta_csv_df)} videos, {len(annotated_set)} annotated "
              f"(behaviors_labeled non-empty) → extracting only those.")
        records = [r for r in all_records if r["video_id"] in annotated_set]
    if args.max_tracks > 0:
        records = records[:args.max_tracks]
    print(f"Will extract {len(records)} tracks from {args.source}")

    # Vocabs come from train_csv to match training-time token indices;
    # only the lookup is augmented with test rows.
    test_lookup = None
    if args.test_csv:
        from pose_bert.model.metadata import build_vocabs as _build_vocabs
        shared_vocabs = _build_vocabs(args.train_csv)
        test_lookup = build_lookup(args.test_csv, shared_vocabs)
        print(f"--test_csv: prepared {len(test_lookup)} (video, mouse) lookup rows.")

    # --chunk_videos caps videos loaded into RAM per chunk so peak memory
    # stays bounded. 0 = single chunk (legacy behavior).
    video_ids_in_order: List[str] = []
    seen: Set[str] = set()
    for r in records:
        v = r["video_id"]
        if v not in seen:
            seen.add(v)
            video_ids_in_order.append(v)
    chunk_size = args.chunk_videos if args.chunk_videos > 0 else len(video_ids_in_order)
    chunk_size = max(1, chunk_size)
    chunks = [video_ids_in_order[i:i + chunk_size]
              for i in range(0, len(video_ids_in_order), chunk_size)]
    print(f"Processing {len(video_ids_in_order)} videos in {len(chunks)} "
          f"chunk(s) of up to {chunk_size}.")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    stitch = args.stitch
    print(f"Stitch mode: {stitch}")
    # |offset - center| within a window; "center" mode picks the window whose
    # center is closest to each frame.
    window_dist_local = (
        np.abs(np.arange(enc_window) - (enc_window // 2)).astype(np.int32)
        if stitch == "center" else None
    )

    total_written = 0
    for ci, chunk_vids in enumerate(chunks):
        chunk_set = set(chunk_vids)
        chunk_records = [r for r in records if r["video_id"] in chunk_set]
        if not chunk_records:
            continue
        if annotated_set is not None:
            allowed = chunk_set & annotated_set
        else:
            allowed = chunk_set
        print(f"\n[chunk {ci+1}/{len(chunks)}] {len(chunk_records)} tracks "
              f"from {len(allowed)} videos")

        ds = InMemoryPretrainDataset(
            data_dir=args.source,
            window_size=enc_window,
            stride=enc_stride,
            max_videos=0,
            allowed_videos=allowed,
            mask_mode="none",
            metadata_csv=args.train_csv,
            meta_unk_p=0.0,
        )
        # Fail loud on feature_dim mismatch; otherwise forward() raises a
        # confusing shape error deep inside.
        ckpt_input_dim = int(cfg.get("input_dim", 0))
        if ckpt_input_dim and ckpt_input_dim != ds.feature_dim:
            raise ValueError(
                f"feature_dim mismatch: checkpoint expects input_dim={ckpt_input_dim} "
                f"(n_parts={ckpt_input_dim//3}) but --source {args.source} provides "
                f"feature_dim={ds.feature_dim} (n_parts={ds.feature_dim//3}). "
                f"Re-run preprocessing with the same --parts_version used for training, "
                f"or point --source at the matching processed dir."
            )
        if test_lookup is not None and ds.meta_lookup is not None:
            ds.meta_lookup.update(test_lookup)
            # meta_ids are CACHED on ds.tracks at construction; updating
            # ds.meta_lookup alone is a no-op, so re-resolve each track here —
            # otherwise test videos run with all-UNK metadata and produce OOD
            # embeddings.
            n_fixed = 0
            for ti, (track, old_meta_ids) in enumerate(ds.tracks):
                rec = chunk_records[ti]
                new_meta_ids = ds._resolve_meta_ids(
                    rec["video_id"], rec["mouse_id"]
                )
                if not np.array_equal(old_meta_ids, new_meta_ids):
                    n_fixed += 1
                ds.tracks[ti] = (track, new_meta_ids)
            print(f"[meta] re-resolved {n_fixed}/{len(ds.tracks)} tracks "
                  f"after merging test_lookup.")

        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

        # Stream-write per-track buffers as they finish: windows are emitted
        # in track-index order (shuffle=False), so any earlier track is done
        # once we see a higher track_idx.
        n_tracks = len(ds.tracks)
        n_to_write = min(len(chunk_records), n_tracks)
        active: Dict[int, tuple] = {}
        n_written = 0

        def flush_track(ti: int) -> None:
            nonlocal n_written
            buf_pair = active.pop(ti, None)
            if buf_pair is None or ti >= n_to_write:
                return
            emb_buf, second = buf_pair
            if stitch == "center":
                avg = emb_buf
            else:
                c = second.clip(min=1).astype(np.float32)
                avg = emb_buf / c[:, None]
            rec = chunk_records[ti]
            out_path = output_dir / rec["lab_id"] / rec["video_id"] / f"{rec['mouse_id']}.npy"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, avg.astype(np.float32))
            n_written += 1

        counter = 0
        for batch in tqdm(loader, desc=f"chunk {ci+1}/{len(chunks)}"):
            feats    = batch["features"].to(device, non_blocking=True)
            meta_ids = batch["meta_ids"].to(device, non_blocking=True)
            pad      = batch["padding_mask"].to(device, non_blocking=True)
            h = model.extract(feats, meta_ids, pad)
            h_np   = h.cpu().numpy()
            pad_np = pad.cpu().numpy()
            B = h_np.shape[0]
            for b in range(B):
                track_idx, start = ds.windows[counter + b]
                for done_idx in [k for k in active if k < track_idx]:
                    flush_track(done_idx)
                actual_len = int((~pad_np[b]).sum())
                if actual_len <= 0:
                    continue
                if track_idx not in active:
                    T_track = ds.tracks[track_idx][0].shape[0]
                    if stitch == "center":
                        active[track_idx] = (
                            np.zeros((T_track, d_model), dtype=np.float32),
                            np.full(T_track, fill_value=10**9, dtype=np.int32),
                        )
                    else:
                        active[track_idx] = (
                            np.zeros((T_track, d_model), dtype=np.float32),
                            np.zeros(T_track, dtype=np.int32),
                        )
                emb_buf, second = active[track_idx]
                end = min(start + actual_len, emb_buf.shape[0])
                n = end - start
                if n <= 0:
                    continue
                if stitch == "center":
                    win_d = window_dist_local[:n]
                    cur_d = second[start:end]
                    better = win_d < cur_d
                    if better.any():
                        idx_local = np.where(better)[0]
                        emb_buf[start + idx_local] = h_np[b, idx_local]
                        second[start + idx_local] = win_d[idx_local]
                else:
                    emb_buf[start:end] += h_np[b, :n]
                    second[start:end] += 1
            counter += B

        # Flush trailing tracks (the last track never sees a higher track_idx).
        for ti in sorted(active.keys()):
            flush_track(ti)
        total_written += n_written
        del ds, loader, active
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    meta = {
        "checkpoint_dir":  os.path.abspath(args.checkpoint),
        "source_dir":      os.path.abspath(args.source),
        "train_csv":       os.path.abspath(args.train_csv),
        "variant":         variant,
        "d_model":         d_model,
        "encoder_window":  enc_window,
        "encoder_stride":  enc_stride,
        "n_tracks":        total_written,
        "stitching":       args.stitch,
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nWrote {total_written} tracks to {output_dir}")
    print(f"meta.json: {meta}")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Build a PoseBERT embedding cache for downstream training."
    )
    ap.add_argument("--checkpoint", required=True,
                    help="Run directory with best_model.pt and config.json.")
    ap.add_argument("--source", required=True,
                    help="Processed data root: {lab}/{vid}/{mouse}.npy.")
    ap.add_argument("--output", required=True,
                    help="Output directory (mirrors --source layout).")
    ap.add_argument("--train_csv", required=True,
                    help="train.csv for metadata vocabs (must match pretraining).")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--enc_stride", type=int, default=0,
                    help="Encoder stride; 0 = window//2.")
    ap.add_argument("--max_tracks", type=int, default=0,
                    help="Debug: limit to first N tracks; 0 = all.")
    ap.add_argument("--chunk_videos", type=int, default=0,
                    help="Cap videos loaded per chunk; 0 = single chunk.")
    ap.add_argument("--no_filter_for_inference", action="store_true",
                    help="Skip filter to behaviors_labeled videos (Kaggle test-time).")
    ap.add_argument("--test_csv", type=str, default=None,
                    help="Optional metadata CSV merged into lookup; vocabs stay on --train_csv.")
    ap.add_argument("--stitch", choices=("center", "average"), default="center",
                    help="Overlap combiner: 'center' picks the window whose center is closest; 'average' sums/counts.")
    return ap


def main():
    args = build_arg_parser().parse_args()
    extract_all(args)


if __name__ == "__main__":
    main()
