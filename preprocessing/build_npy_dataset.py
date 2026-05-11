"""Run feature + label preprocessing and save per-pair .npy files for mmap loading.

Usage:
    python build_npy_dataset.py --train_csv data_sample/train.csv \\
        --tracking_dir data_sample/train_tracking \\
        --annotation_dir data_sample/train_annotation \\
        --output_dir data_sample/processed
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocess_data import (
    build_action_universe,
    intervals_to_frame_labels,
    build_pair_loss_mask,
)
from preprocess_features import (
    ALL_PARTS,
    DROP_PARTS,
    process_directed_pair_agent_centric,
)


def build_dataset(
    train_csv: str,
    tracking_dir: str,
    annotation_dir: str,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    train_meta = pd.read_csv(train_csv)
    train_meta["video_id"] = train_meta["video_id"].astype(str)

    action_list, per_video_whitelist, lab_action_sets = build_action_universe(train_meta)
    action_to_idx = {a: i for i, a in enumerate(action_list)}
    num_actions = len(action_list)

    np.save(os.path.join(output_dir, "action_list.npy"), np.array(action_list))
    print(f"Action list ({num_actions} actions): {action_list}")

    saved_videos = 0
    saved_pairs = 0
    for _, row in tqdm(train_meta.iterrows(), total=len(train_meta), desc="Videos"):
        lab_id = row["lab_id"]
        video_id = str(row["video_id"])

        pair_whitelist = per_video_whitelist.get((lab_id, video_id), {})
        if not pair_whitelist:
            continue

        tracking_path = os.path.join(tracking_dir, lab_id, f"{video_id}.parquet")
        annotation_path = os.path.join(annotation_dir, lab_id, f"{video_id}.parquet")

        if not os.path.exists(tracking_path) or not os.path.exists(annotation_path):
            continue

        fps = float(row.get("frames_per_second", 25.0) or 25.0)
        pix_per_cm = float(row.get("pix_per_cm_approx", 1.0) or 1.0)

        tracking_df = pd.read_parquet(tracking_path)
        tracking_df = tracking_df[~tracking_df["bodypart"].isin(DROP_PARTS)]
        ann_df = pd.read_parquet(annotation_path)

        video_frames = sorted(tracking_df["video_frame"].unique())
        vf = np.array(video_frames)
        # train.csv whitelists pairs by mouse_id, but some annotated mice are
        # absent from the tracking parquet (e.g. AdaptableSnail mouse 3/4).
        # Without this filter, the per-mouse pivot produces all-NaN → all-zero
        # placeholder features, which (a) waste training compute and (b) break
        # the embedding pipeline that — correctly — has no track for them.
        tracked_mice = set(int(m) for m in tracking_df["mouse_id"].unique())

        video_pair_count = 0
        for (agent_id, target_id), allowed_actions in pair_whitelist.items():
            if int(agent_id) not in tracked_mice or int(target_id) not in tracked_mice:
                continue
            loss_mask = build_pair_loss_mask(
                pair_whitelist=pair_whitelist,
                agent_id=agent_id,
                target_id=target_id,
                action_to_idx=action_to_idx,
                action_dim=num_actions,
            )
            try:
                agent_feat_df, target_feat_df = process_directed_pair_agent_centric(
                    tracking_df=tracking_df,
                    agent_id=agent_id,
                    target_id=target_id,
                    fps=fps,
                    pix_per_cm=pix_per_cm,
                    parts_order=ALL_PARTS,
                    drop_parts=DROP_PARTS,
                    video_frames=video_frames,
                    bodyparts_already_dropped=True,
                )
            except Exception as e:
                print(f"  SKIP {lab_id}/{video_id} pair ({agent_id},{target_id}): {e}")
                continue

            features = np.concatenate(
                [agent_feat_df.values, target_feat_df.values], axis=1
            ).astype(np.float32)

            labels = intervals_to_frame_labels(
                ann_df=ann_df,
                vf=vf,
                agent_id=agent_id,
                target_id=target_id,
                action_to_idx=action_to_idx,
                action_dim=num_actions,
            )

            pair_key = f"{agent_id}_{target_id}"
            pair_dir = os.path.join(output_dir, lab_id, video_id)
            os.makedirs(pair_dir, exist_ok=True)

            np.save(os.path.join(pair_dir, f"{pair_key}_features.npy"), features)
            np.save(os.path.join(pair_dir, f"{pair_key}_labels.npy"), labels)
            np.save(os.path.join(pair_dir, f"{pair_key}_loss_mask.npy"), loss_mask)

            saved_pairs += 1
            video_pair_count += 1

        if video_pair_count > 0:
            saved_videos += 1
            print(
                f"  {lab_id}/{video_id}: {video_pair_count} pairs, "
                f"{len(video_frames)} frames, "
                f"{num_actions} actions"
            )

    index_rows = []
    for lab_id in sorted(os.listdir(output_dir)):
        lab_dir = os.path.join(output_dir, lab_id)
        if not os.path.isdir(lab_dir):
            continue
        for vid_name in sorted(os.listdir(lab_dir)):
            vid_dir = os.path.join(lab_dir, vid_name)
            if not os.path.isdir(vid_dir):
                continue
            pair_keys = set()
            for fname in os.listdir(vid_dir):
                if fname.endswith("_features.npy"):
                    pair_keys.add(fname.replace("_features.npy", ""))
            for pair_key in sorted(pair_keys):
                feat_path = os.path.join(vid_dir, f"{pair_key}_features.npy")
                num_frames = np.load(feat_path, mmap_mode='r').shape[0]
                video_dir = os.path.join(lab_id, vid_name)
                index_rows.append((video_dir, pair_key, num_frames))

    index_df = pd.DataFrame(index_rows, columns=["video_dir", "pair_key", "num_frames"])
    index_df.to_csv(os.path.join(output_dir, "index.csv"), index=False)
    print(f"Saved index.csv: {len(index_rows)} pairs")

    print(f"\nDone. Saved {saved_pairs} pairs across {saved_videos} videos to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="data_sample/train.csv")
    parser.add_argument("--tracking_dir", default="data_sample/train_tracking")
    parser.add_argument("--annotation_dir", default="data_sample/train_annotation")
    parser.add_argument("--output_dir", default="data_sample/processed")
    args = parser.parse_args()
    build_dataset(args.train_csv, args.tracking_dir, args.annotation_dir, args.output_dir)
