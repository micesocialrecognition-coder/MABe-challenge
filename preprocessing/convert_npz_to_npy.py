"""Convert per-video .npz archives to per-pair uncompressed .npy files for mmap loading.

Usage:
    python convert_npz_to_npy.py /path/to/processed_data
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm


def convert(data_dir: str, delete_npz: bool = False) -> None:
    index_rows = []
    npz_files = []

    for lab_id in sorted(os.listdir(data_dir)):
        lab_dir = os.path.join(data_dir, lab_id)
        if not os.path.isdir(lab_dir):
            continue
        for fname in sorted(os.listdir(lab_dir)):
            if fname.endswith(".npz") and not fname.startswith("."):
                npz_files.append((lab_id, fname))

    if not npz_files:
        print("No .npz files found. Already converted?")
        return

    print(f"Found {len(npz_files)} .npz files to convert")

    for lab_id, fname in tqdm(npz_files, desc="Converting"):
        video_id = fname.replace(".npz", "")
        npz_path = os.path.join(data_dir, lab_id, fname)
        out_dir = os.path.join(data_dir, lab_id, video_id)
        os.makedirs(out_dir, exist_ok=True)

        with np.load(npz_path) as data:
            pair_keys = set()
            for key in data.files:
                if key.endswith("_features"):
                    pair_keys.add(key.replace("_features", ""))

            for pair_key in sorted(pair_keys):
                features = data[f"{pair_key}_features"]
                np.save(os.path.join(out_dir, f"{pair_key}_features.npy"), features)

                lk = f"{pair_key}_labels"
                if lk in data:
                    np.save(os.path.join(out_dir, f"{pair_key}_labels.npy"), data[lk])

                mk = f"{pair_key}_loss_mask"
                if mk in data:
                    np.save(os.path.join(out_dir, f"{pair_key}_loss_mask.npy"), data[mk])

                video_dir = os.path.join(lab_id, video_id)
                index_rows.append((video_dir, pair_key, features.shape[0]))

        if delete_npz:
            os.remove(npz_path)

    index_df = pd.DataFrame(index_rows, columns=["video_dir", "pair_key", "num_frames"])
    index_path = os.path.join(data_dir, "index.csv")

    if os.path.exists(index_path):
        backup = index_path + ".bak"
        os.rename(index_path, backup)
        print(f"Backed up old index.csv → index.csv.bak")

    index_df.to_csv(index_path, index=False)
    print(f"Saved new index.csv: {len(index_rows)} pairs across {len(npz_files)} videos")
    if not delete_npz:
        print("Original .npz files kept. Use --delete-npz to remove them after verifying.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .npz to individual .npy files")
    parser.add_argument("data_dir", help="Path to processed_data directory")
    parser.add_argument("--delete-npz", action="store_true",
                        help="Delete original .npz files after conversion")
    args = parser.parse_args()
    convert(args.data_dir, delete_npz=args.delete_npz)
