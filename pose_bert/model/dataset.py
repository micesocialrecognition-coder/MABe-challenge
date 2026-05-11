"""PoseBERT pretraining dataset: InMemory (RAM) and Disk (mmap) variants."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from pose_bert.model.masking import (
    generate_span_mask,
    generate_two_disjoint_span_masks,
)
from pose_bert.model.metadata import (
    META_UNK_P_DEFAULT,
    N_META_FIELDS,
    UNK_IDX,
    build_lookup,
    build_vocabs,
    inject_unk,
    vocab_sizes,
)


WINDOW_SIZE = 128
STRIDE = 64


# Translation is intentionally omitted — pos_raw's MSE loss is translation-equivariant,
# so it teaches no invariance. Pivot is observed-coord centroid to stay arena-agnostic.

def _build_lr_perm(parts: List[str]) -> np.ndarray:
    """Permutation that swaps body parts whose names differ only by '_left'<->'_right'."""
    perm = list(range(len(parts)))
    name_to_idx = {p: i for i, p in enumerate(parts)}
    for i, name in enumerate(parts):
        if "_left" in name:
            mate = name.replace("_left", "_right")
            if mate in name_to_idx:
                j = name_to_idx[mate]
                perm[i], perm[j] = j, i
    return np.asarray(perm, dtype=np.int64)


def _augment_window(
    window: np.ndarray,
    n_parts: int,
    lr_perm: np.ndarray,
    rng: np.random.Generator,
    flip_p: float,
    rot_max_deg: float,
    coord_jitter_std: float,
) -> np.ndarray:
    """Flip+L/R swap, rotation around observed centroid, Gaussian jitter on [T, 3*n_parts]."""
    out = window.copy()
    coord_dim = 2 * n_parts
    coords = out[:, :coord_dim].reshape(-1, n_parts, 2)
    masks = out[:, coord_dim:]

    obs = masks > 0
    if obs.any():
        cx = float(coords[..., 0][obs].mean())
        cy = float(coords[..., 1][obs].mean())
    else:
        cx, cy = 0.0, 0.0

    if rng.random() < flip_p:
        coords[..., 0] = 2.0 * cx - coords[..., 0]
        out[:, :coord_dim] = coords[:, lr_perm, :].reshape(-1, coord_dim)
        out[:, coord_dim:] = masks[:, lr_perm]
        coords = out[:, :coord_dim].reshape(-1, n_parts, 2)
        masks = out[:, coord_dim:]

    if rot_max_deg > 0:
        ang = float(rng.uniform(-rot_max_deg, rot_max_deg)) * np.pi / 180.0
        c, s = np.cos(ang), np.sin(ang)
        dx = coords[..., 0] - cx
        dy = coords[..., 1] - cy
        coords[..., 0] = c * dx - s * dy + cx
        coords[..., 1] = s * dx + c * dy + cy

    if coord_jitter_std > 0:
        coords += rng.normal(0.0, coord_jitter_std, size=coords.shape).astype(np.float32)

    return out


def _scan_data_dir(data_dir: str) -> List[Dict]:
    """Scan {data_dir}/{lab}/{video}/{mouse}.npy and return track records."""
    records = []
    root = Path(data_dir)
    for npy_path in sorted(root.rglob("*.npy")):
        parts = npy_path.relative_to(root).parts
        if len(parts) != 3:
            continue
        lab_id, video_id, mouse_file = parts
        records.append({
            "npy_path": str(npy_path.relative_to(root)),
            "lab_id": lab_id,
            "video_id": video_id,
            "mouse_id": mouse_file.replace(".npy", ""),
            "abs_path": str(npy_path),
        })
    return records


class InMemoryPretrainDataset(Dataset):
    """Pretraining dataset that loads ALL .npy tracks into RAM at __init__."""

    def __init__(
        self,
        data_dir: str,
        window_size: int = WINDOW_SIZE,
        stride: int = STRIDE,
        max_videos: int = 0,
        allowed_videos: Optional[Set[str]] = None,
        mask_ratio_single: float = 0.15,
        min_span: int = 5,
        max_span: int = 30,
        # mask_mode: "single_span" -> "span_mask"; "two_disjoint_spans" ->
        # "span_mask_pos"+"span_mask_vel" (disjoint); "none" -> no mask key (forecast).
        mask_mode: str = "single_span",
        mask_ratio_pos: float = 0.15,
        mask_ratio_vel: float = 0.15,
        # meta_vocabs must match across train/val splits.
        metadata_csv: Optional[str] = None,
        meta_unk_p: float = META_UNK_P_DEFAULT,
        meta_vocabs: Optional[Dict[str, Dict[str, int]]] = None,
        augment: bool = False,
        parts: Optional[List[str]] = None,
        aug_flip_p: float = 0.5,
        aug_rot_max_deg: float = 30.0,
        aug_coord_jitter_std: float = 0.3,
    ):
        self.window_size = window_size
        self.data_dir = data_dir
        self.mask_ratio_single = mask_ratio_single
        self.min_span = min_span
        self.max_span = max_span
        if mask_mode not in {"single_span", "two_disjoint_spans", "none"}:
            raise ValueError(f"Unknown mask_mode={mask_mode!r}")
        self.mask_mode = mask_mode
        self.mask_ratio_pos = mask_ratio_pos
        self.mask_ratio_vel = mask_ratio_vel
        self.meta_unk_p = float(meta_unk_p)
        self.augment = bool(augment)
        self.aug_flip_p = float(aug_flip_p)
        self.aug_rot_max_deg = float(aug_rot_max_deg)
        self.aug_coord_jitter_std = float(aug_coord_jitter_std)
        self._lr_perm = _build_lr_perm(parts) if (augment and parts) else None
        if self.augment and self._lr_perm is None:
            raise ValueError("augment=True requires parts list (load from parts.json)")

        all_records = _scan_data_dir(data_dir)
        if len(all_records) == 0:
            raise FileNotFoundError(f"No .npy files found in {data_dir}")

        if metadata_csv is not None:
            self.meta_vocabs = meta_vocabs if meta_vocabs is not None else build_vocabs(metadata_csv)
            self.meta_lookup = build_lookup(metadata_csv, self.meta_vocabs)
        else:
            self.meta_vocabs = None
            self.meta_lookup = None
        self.meta_vocab_sizes = (
            vocab_sizes(self.meta_vocabs) if self.meta_vocabs else [1] * N_META_FIELDS
        )
        # num_labs kept for backward compat with code that reads it.
        self.num_labs = self.meta_vocab_sizes[0]

        all_videos = sorted({r["video_id"] for r in all_records})
        if max_videos > 0:
            all_videos = all_videos[:max_videos]
            all_records = [r for r in all_records if r["video_id"] in set(all_videos)]
        if allowed_videos is not None:
            all_records = [r for r in all_records if r["video_id"] in allowed_videos]

        self.tracks = []
        total_frames = 0
        meta_unk_tracks = 0
        self.feature_dim: Optional[int] = None
        for rec in all_records:
            track = np.load(rec["abs_path"]).astype(np.float32)
            if self.feature_dim is None:
                self.feature_dim = int(track.shape[-1])
            elif track.shape[-1] != self.feature_dim:
                raise ValueError(
                    f"Inconsistent feature dim: {rec['abs_path']} has "
                    f"{track.shape[-1]} but expected {self.feature_dim}"
                )
            meta_ids = self._resolve_meta_ids(rec["video_id"], rec["mouse_id"])
            if meta_ids[0] == UNK_IDX:
                meta_unk_tracks += 1
            self.tracks.append((track, meta_ids))
            total_frames += track.shape[0]

        self.windows = []
        for track_idx, (track, _) in enumerate(self.tracks):
            T = track.shape[0]
            for start in range(0, T, stride):
                self.windows.append((track_idx, start))

        ram_mb = total_frames * self.feature_dim * 4 / (1024 ** 2)
        print(
            f"InMemoryPretrainDataset: {len(self.tracks)} tracks, "
            f"{total_frames:,} total frames ({ram_mb:.1f} MB in RAM), "
            f"{len(self.windows):,} windows  "
            f"(window={window_size}, stride={stride}, feature_dim={self.feature_dim}, "
            f"meta_vocab_sizes={self.meta_vocab_sizes}, "
            f"meta_unk_p={self.meta_unk_p}, "
            f"unresolved_lab_tracks={meta_unk_tracks})"
        )

    def _resolve_meta_ids(self, video_id: str, mouse_id: str) -> np.ndarray:
        if self.meta_lookup is None:
            return np.zeros(N_META_FIELDS, dtype=np.int64)
        try:
            mid = int(mouse_id)
        except (TypeError, ValueError):
            return np.zeros(N_META_FIELDS, dtype=np.int64)
        return self.meta_lookup.get((str(video_id), mid), np.zeros(N_META_FIELDS, dtype=np.int64))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        track_idx, start = self.windows[idx]
        track_data, meta_ids = self.tracks[track_idx]

        end = min(start + self.window_size, len(track_data))
        actual_len = end - start

        features = torch.zeros(self.window_size, self.feature_dim, dtype=torch.float32)
        padding_mask = torch.ones(self.window_size, dtype=torch.bool)

        window_np = track_data[start:end]
        if self.augment:
            n_parts = self.feature_dim // 3
            window_np = _augment_window(
                window_np, n_parts, self._lr_perm, np.random.default_rng(),
                flip_p=self.aug_flip_p,
                rot_max_deg=self.aug_rot_max_deg,
                coord_jitter_std=self.aug_coord_jitter_std,
            )
        features[:actual_len] = torch.from_numpy(np.asarray(window_np, dtype=np.float32))
        padding_mask[:actual_len] = False

        if self.meta_unk_p > 0.0:
            rng = np.random.default_rng()
            meta_ids_sample = inject_unk(meta_ids, self.meta_unk_p, rng)
        else:
            meta_ids_sample = meta_ids

        # Generate masks in the worker — pure numpy runs in parallel with GPU.
        out = {
            "features": features,
            "meta_ids": torch.from_numpy(meta_ids_sample.copy()),
            "padding_mask": padding_mask,
        }

        if self.mask_mode == "single_span":
            out["span_mask"] = generate_span_mask(
                seq_len=actual_len,
                window_size=self.window_size,
                mask_ratio=self.mask_ratio_single,
                min_span=self.min_span,
                max_span=self.max_span,
            )
        elif self.mask_mode == "two_disjoint_spans":
            mask_pos, mask_vel = generate_two_disjoint_span_masks(
                seq_len=actual_len,
                window_size=self.window_size,
                mask_ratio_pos=self.mask_ratio_pos,
                mask_ratio_vel=self.mask_ratio_vel,
                min_span=self.min_span,
                max_span=self.max_span,
            )
            out["span_mask_pos"] = mask_pos
            out["span_mask_vel"] = mask_vel

        return out


class DiskPretrainDataset(Dataset):
    """Pretraining dataset that memory-maps .npy files on each access."""

    def __init__(
        self,
        data_dir: str,
        window_size: int = WINDOW_SIZE,
        stride: int = STRIDE,
        max_videos: int = 0,
        allowed_videos: Optional[Set[str]] = None,
        metadata_csv: Optional[str] = None,
        meta_unk_p: float = META_UNK_P_DEFAULT,
        meta_vocabs: Optional[Dict[str, Dict[str, int]]] = None,
    ):
        self.window_size = window_size
        self.data_dir = data_dir
        self.meta_unk_p = float(meta_unk_p)

        index_df = pd.read_csv(os.path.join(data_dir, "index.csv"))

        if max_videos > 0:
            unique_vids = index_df["video_id"].unique()[:max_videos]
            index_df = index_df[index_df["video_id"].isin(unique_vids)]

        if allowed_videos is not None:
            index_df = index_df[index_df["video_id"].isin(allowed_videos)]

        index_df = index_df.reset_index(drop=True)

        if metadata_csv is not None:
            self.meta_vocabs = meta_vocabs if meta_vocabs is not None else build_vocabs(metadata_csv)
            self.meta_lookup = build_lookup(metadata_csv, self.meta_vocabs)
        else:
            self.meta_vocabs = None
            self.meta_lookup = None
        self.meta_vocab_sizes = (
            vocab_sizes(self.meta_vocabs) if self.meta_vocabs else [1] * N_META_FIELDS
        )
        self.num_labs = self.meta_vocab_sizes[0]

        self.track_info = []
        total_frames = 0
        for _, row in index_df.iterrows():
            npy_path = os.path.join(data_dir, row["npy_path"])
            num_frames = int(row["num_frames"])
            meta_ids = self._resolve_meta_ids(row["video_id"], row["mouse_id"])
            self.track_info.append((npy_path, meta_ids, num_frames))
            total_frames += num_frames

        if not self.track_info:
            raise FileNotFoundError(f"No tracks loaded from {data_dir}")
        self.feature_dim = int(np.load(self.track_info[0][0], mmap_mode="r").shape[-1])

        self.windows = []
        for track_idx, (_, _, T) in enumerate(self.track_info):
            for start in range(0, T, stride):
                self.windows.append((track_idx, start))

        print(
            f"DiskPretrainDataset: {len(self.track_info)} tracks, "
            f"{total_frames:,} total frames (memory-mapped), "
            f"{len(self.windows):,} windows  "
            f"(window={window_size}, stride={stride}, feature_dim={self.feature_dim}, "
            f"meta_vocab_sizes={self.meta_vocab_sizes}, meta_unk_p={self.meta_unk_p})"
        )

    def _resolve_meta_ids(self, video_id, mouse_id) -> np.ndarray:
        if self.meta_lookup is None:
            return np.zeros(N_META_FIELDS, dtype=np.int64)
        try:
            mid = int(mouse_id)
        except (TypeError, ValueError):
            return np.zeros(N_META_FIELDS, dtype=np.int64)
        return self.meta_lookup.get((str(video_id), mid), np.zeros(N_META_FIELDS, dtype=np.int64))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        track_idx, start = self.windows[idx]
        npy_path, meta_ids, num_frames = self.track_info[track_idx]

        track_data = np.load(npy_path, mmap_mode="r")

        end = min(start + self.window_size, num_frames)
        actual_len = end - start

        features = torch.zeros(self.window_size, self.feature_dim, dtype=torch.float32)
        padding_mask = torch.ones(self.window_size, dtype=torch.bool)

        features[:actual_len] = torch.from_numpy(np.array(track_data[start:end]))
        padding_mask[:actual_len] = False

        if self.meta_unk_p > 0.0:
            rng = np.random.default_rng()
            meta_ids_sample = inject_unk(meta_ids, self.meta_unk_p, rng)
        else:
            meta_ids_sample = meta_ids

        return {
            "features": features,
            "meta_ids": torch.from_numpy(meta_ids_sample.copy()),
            "padding_mask": padding_mask,
        }


def create_pretrain_dataloaders(
    data_dir: str,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
    batch_size: int = 64,
    val_split: float = 0.1,
    num_workers: int = 4,
    seed: int = 42,
    max_videos: int = 0,
    in_memory: bool = True,
    mask_ratio_single: float = 0.15,
    min_span: int = 5,
    max_span: int = 30,
    mask_mode: str = "single_span",
    mask_ratio_pos: float = 0.15,
    mask_ratio_vel: float = 0.15,
    metadata_csv: Optional[str] = None,
    meta_unk_p: float = META_UNK_P_DEFAULT,
    augment: bool = False,
    aug_flip_p: float = 0.5,
    aug_rot_max_deg: float = 30.0,
    aug_coord_jitter_std: float = 0.3,
) -> Tuple[DataLoader, DataLoader]:
    """Create train/val DataLoaders, split by video_id to prevent leakage."""

    if in_memory:
        all_records = _scan_data_dir(data_dir)
        all_videos = sorted({r["video_id"] for r in all_records})
    else:
        index_df = pd.read_csv(os.path.join(data_dir, "index.csv"))
        all_videos = index_df["video_id"].unique().tolist()

    if max_videos > 0:
        all_videos = all_videos[:max_videos]
    rng = np.random.RandomState(seed)
    shuffled = all_videos.copy()
    rng.shuffle(shuffled)

    n_val = max(1, int(len(shuffled) * val_split))
    val_videos = set(shuffled[:n_val])
    train_videos = set(shuffled[n_val:])

    DatasetClass = InMemoryPretrainDataset if in_memory else DiskPretrainDataset

    # Build metadata vocabs once so train and val share the same indices.
    if metadata_csv is not None:
        shared_vocabs = build_vocabs(metadata_csv)
    else:
        shared_vocabs = None

    parts_list: Optional[List[str]] = None
    if augment and in_memory:
        import json as _json
        parts_path = os.path.join(data_dir, "parts.json")
        if not os.path.exists(parts_path):
            raise FileNotFoundError(
                f"--augment requires {parts_path} for L/R swap permutation"
            )
        with open(parts_path) as f:
            parts_list = list(_json.load(f)["parts"])

    in_mem_kwargs = dict(
        mask_ratio_single=mask_ratio_single,
        min_span=min_span,
        max_span=max_span,
        mask_mode=mask_mode,
        mask_ratio_pos=mask_ratio_pos,
        mask_ratio_vel=mask_ratio_vel,
    )
    disk_kwargs: dict = {}
    common_kwargs = in_mem_kwargs if in_memory else disk_kwargs

    train_aug_kwargs = (
        dict(
            augment=True, parts=parts_list,
            aug_flip_p=aug_flip_p,
            aug_rot_max_deg=aug_rot_max_deg,
            aug_coord_jitter_std=aug_coord_jitter_std,
        )
        if (augment and in_memory) else {}
    )

    train_ds = DatasetClass(
        data_dir,
        window_size=window_size,
        stride=stride,
        max_videos=0,
        allowed_videos=train_videos,
        metadata_csv=metadata_csv,
        meta_unk_p=meta_unk_p,
        meta_vocabs=shared_vocabs,
        **common_kwargs,
        **train_aug_kwargs,
    )
    val_ds = DatasetClass(
        data_dir,
        window_size=window_size,
        stride=stride,
        max_videos=0,
        allowed_videos=val_videos,
        metadata_csv=metadata_csv,
        meta_unk_p=0.0,
        meta_vocabs=shared_vocabs,
        **common_kwargs,
    )

    # On Linux fork-based multiprocessing shares RAM copy-on-write,
    # so in-memory tracks are accessible to workers without copying.
    if num_workers == 0:
        num_workers = min(4, os.cpu_count() or 1)

    pin = torch.cuda.is_available()

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin,
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    n_train_w = len(train_ds)
    n_val_w = len(val_ds)
    print(
        f"Video-level split (seed={seed}, val_split={val_split}): "
        f"{len(train_videos)} train videos, {len(val_videos)} val videos | "
        f"{n_train_w:,} train windows, {n_val_w:,} val windows"
    )
    if len(val_videos) == 0 and len(shuffled) > 0:
        print("  (no val videos -- using train loss for checkpointing)")

    return train_loader, val_loader
