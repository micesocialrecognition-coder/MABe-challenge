"""Build single-mouse [T, n_parts*3] tensors for PoseBERT pretraining.

Usage:
    python preprocessing/build_pretrain_data.py \\
        --train_csv data/raw/train.csv \\
        --tracking_dir data/raw/train_tracking \\
        --output_dir data/pose_bert_processed_v2/ \\
        --parts_version v2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess_features import (
    BODY_PARTS_REGISTRY,
    pivot_long_to_wide,
    pixels_to_cm,
    position_masks_from_raw,
    fill_missing_positions,
)


def process_single_mouse(
    tracking_one_mouse: pd.DataFrame,
    video_frames: list[int],
    pix_per_cm: float,
    arena_center_x_cm: float,
    arena_center_y_cm: float,
    parts: list[str],
) -> np.ndarray:
    """Pivot + cm + mask + fill + arena-center → [T, 3*len(parts)] float32 array."""
    n_parts = len(parts)

    # parts_order acts as the allow-list — out-of-schema parts are discarded by reindex.
    wide_px = pivot_long_to_wide(
        tracking_one_mouse, parts_order=parts, video_frames=video_frames
    )

    wide_cm = pixels_to_cm(wide_px, pix_per_cm)

    masks_df = position_masks_from_raw(wide_cm)

    filled_cm = fill_missing_positions(wide_cm)

    x_cols = [f"x_{p}" for p in parts]
    y_cols = [f"y_{p}" for p in parts]
    filled_cm[x_cols] = filled_cm[x_cols] - arena_center_x_cm
    filled_cm[y_cols] = filled_cm[y_cols] - arena_center_y_cm

    coord_col_order = [f"{c}_{p}" for p in parts for c in ("x", "y")]
    positions = filled_cm[coord_col_order].to_numpy(dtype=np.float32)

    # Combine m_x and m_y into a single per-part mask (1.0 if EITHER observed).
    mask_per_part = np.zeros((len(video_frames), n_parts), dtype=np.float32)
    for i, part in enumerate(parts):
        m_x = masks_df[f"m_x_{part}"].to_numpy(dtype=np.float32)
        m_y = masks_df[f"m_y_{part}"].to_numpy(dtype=np.float32)
        mask_per_part[:, i] = np.maximum(m_x, m_y)

    tensor = np.concatenate([positions, mask_per_part], axis=1).astype(np.float32)
    expected = (len(video_frames), 3 * n_parts)
    assert tensor.shape == expected, f"Expected shape {expected}, got {tensor.shape}"

    return tensor


def process_video(
    lab_id: str,
    video_id: str,
    tracking_path: Path,
    meta_row: pd.Series,
    output_dir: Path,
    parts: list[str],
) -> list[dict]:
    """Process all mice in a video → one .npy per mouse; returns index records."""
    # Filter to schema parts here (not just via pivot reindex): video_frames is built from
    # surviving rows below, so frames seen only in out-of-schema parts must be dropped or
    # they become all-NaN rows and shift the frame count vs. V1's historical behaviour
    # (AdaptableSnail set 2 had headpiece-only frames).
    tracking_df = pd.read_parquet(tracking_path)
    tracking_df = tracking_df[tracking_df["bodypart"].isin(parts)].copy()

    pix_per_cm = float(meta_row.get("pix_per_cm_approx", 1.0) or 1.0)
    if pix_per_cm == 0:
        pix_per_cm = 1.0
    fps = float(meta_row.get("frames_per_second", 30.0) or 30.0)
    video_width_pix = float(meta_row.get("video_width_pix", 0) or 0)
    video_height_pix = float(meta_row.get("video_height_pix", 0) or 0)

    arena_center_x_cm = (video_width_pix / 2.0) / pix_per_cm
    arena_center_y_cm = (video_height_pix / 2.0) / pix_per_cm

    video_frames = sorted(tracking_df["video_frame"].unique().tolist())
    if len(video_frames) == 0:
        return []

    mouse_ids = sorted(tracking_df["mouse_id"].unique().tolist())

    video_out_dir = output_dir / lab_id / video_id
    video_out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for mouse_id in mouse_ids:
        mouse_tracking = tracking_df[tracking_df["mouse_id"] == mouse_id]
        if len(mouse_tracking) == 0:
            continue

        tensor = process_single_mouse(
            tracking_one_mouse=mouse_tracking,
            video_frames=video_frames,
            pix_per_cm=pix_per_cm,
            arena_center_x_cm=arena_center_x_cm,
            arena_center_y_cm=arena_center_y_cm,
            parts=parts,
        )

        npy_filename = f"{mouse_id}.npy"
        npy_path = video_out_dir / npy_filename
        np.save(npy_path, tensor)

        rel_path = f"{lab_id}/{video_id}/{npy_filename}"
        records.append({
            "npy_path": rel_path,
            "lab_id": lab_id,
            "video_id": video_id,
            "mouse_id": mouse_id,
            "num_frames": tensor.shape[0],
            "fps": fps,
        })

    return records


def build_pretrain_dataset(train_csv: Path, tracking_dir: Path, output_dir: Path, parts_version: str) -> None:
    """Build full PoseBERT pretraining dataset; writes per-mouse .npy + index.csv + parts.json."""
    parts = BODY_PARTS_REGISTRY[parts_version]
    output_dir.mkdir(parents=True, exist_ok=True)

    # parts.json is consumed downstream by extract_embeddings.py to reproduce input layout.
    import json
    with open(output_dir / "parts.json", "w") as f:
        json.dump({"parts_version": parts_version, "parts": parts}, f, indent=2)

    meta_df = pd.read_csv(train_csv)
    print(f"Loaded {len(meta_df)} videos from {train_csv}")
    print(f"parts_version={parts_version} ({len(parts)} parts) -> [T, {3*len(parts)}] tensors")

    all_records: list[dict] = []
    skipped = 0

    for _, row in tqdm(meta_df.iterrows(), total=len(meta_df), desc="Processing videos"):
        lab_id = str(row["lab_id"])
        video_id = str(row["video_id"])

        tracking_path = tracking_dir / lab_id / f"{video_id}.parquet"
        if not tracking_path.exists():
            skipped += 1
            continue

        try:
            records = process_video(
                lab_id=lab_id,
                video_id=video_id,
                tracking_path=tracking_path,
                meta_row=row,
                output_dir=output_dir,
                parts=parts,
            )
            all_records.extend(records)
        except Exception as e:
            print(f"  ERROR processing {lab_id}/{video_id}: {e}")
            skipped += 1
            continue

    index_df = pd.DataFrame(all_records)
    index_path = output_dir / "index.csv"
    index_df.to_csv(index_path, index=False)

    print(f"\nDone! Processed {len(all_records)} mouse tracks "
          f"from {len(meta_df) - skipped} videos ({skipped} skipped).")
    print(f"Index written to {index_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build PoseBERT pretraining data: raw tracking -> [T, 33] .npy per mouse"
    )
    parser.add_argument("--train_csv", type=Path, required=True)
    parser.add_argument(
        "--tracking_dir", type=Path, required=True,
        help="Root dir: {tracking_dir}/{lab_id}/{video_id}.parquet",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--parts_version", choices=sorted(BODY_PARTS_REGISTRY.keys()), default="v1",
        help="Body parts list version (v1=11, v2=16).",
    )
    args = parser.parse_args()

    build_pretrain_dataset(
        train_csv=args.train_csv,
        tracking_dir=args.tracking_dir,
        output_dir=args.output_dir,
        parts_version=args.parts_version,
    )


if __name__ == "__main__":
    main()
