"""CNN+Transformer per-frame multi-class behavior model (softmax + 'none' class).

Usage: python model_softmax.py <data_dir> <model_dir> [--num_epochs ...]
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd

# cnn_transformer/ root holds shared modules (metadata, transformer, F_Beta).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from transformer import Encoder
from metadata import (
    META_UNK_P_DEFAULT,
    N_META_SLOTS,
    build_pair_lookup,
    build_vocabs,
    inject_unk,
    parse_pair_key,
    parse_video_dir,
    resolve_meta_ids,
    slot_to_table_index,
    vocab_sizes_per_table,
)


class TeeLogger:
    """Duplicates stdout to both the terminal and a log file."""
    def __init__(self, log_path: str):
        self.terminal = sys.stdout
        self.log_file = open(log_path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()
        sys.stdout = self.terminal


NUM_FEATURES = 88 * 2
WINDOW_SIZE = 64
STRIDE = 32
BETA = 1.0
DEFAULT_TRAIN_CSV = "/mabe/data/train.csv"

FEATURES_MODES = ("raw", "embed", "concat")


def read_embedding_dim(embeddings_dir: str) -> int:
    """Read d_model from {embeddings_dir}/meta.json (written by extract_embeddings.py)."""
    import json as _json
    meta_path = os.path.join(embeddings_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Embeddings dir {embeddings_dir!r} is missing meta.json — "
            f"run pose_bert.scripts.extract_embeddings to produce it."
        )
    with open(meta_path) as f:
        return int(_json.load(f)["d_model"])


def feature_input_dim(features_mode: str, embed_dim: int) -> int:
    if features_mode == "raw":
        return NUM_FEATURES
    if features_mode == "embed":
        return 2 * embed_dim
    if features_mode == "concat":
        return NUM_FEATURES + 2 * embed_dim
    raise ValueError(f"unknown features_mode={features_mode!r}")

# Layout of the 88-d per-mouse feature block: must match preprocess_features.py
# column order x_*,y_* | m_x_*,m_y_* | vx_*,vy_* | m_vx_*,m_vy_* (11 parts × 2 coords).
N_PARTS = 11
POS_X_IDX = np.arange(0, 22, 2, dtype=np.int64)
POS_Y_IDX = np.arange(1, 22, 2, dtype=np.int64)
VEL_X_IDX = POS_X_IDX + 44
VEL_Y_IDX = POS_Y_IDX + 44
MASK_POS_IDX = np.arange(22, 44, dtype=np.int64)
MASK_VEL_IDX = np.arange(66, 88, dtype=np.int64)

# Body part order resolved lazily to avoid an import cycle.
_AUG_PART_ORDER: Optional[List[str]] = None
_AUG_LR_SWAP_PAIRS: Optional[List[Tuple[int, int]]] = None


def _resolve_lr_pairs() -> List[Tuple[int, int]]:
    global _AUG_PART_ORDER, _AUG_LR_SWAP_PAIRS
    if _AUG_LR_SWAP_PAIRS is not None:
        return _AUG_LR_SWAP_PAIRS
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "preprocessing"))
        from preprocess_features import ALL_PARTS  # type: ignore
        _AUG_PART_ORDER = list(ALL_PARTS)
    except Exception:
        _AUG_PART_ORDER = []
    pairs: List[Tuple[int, int]] = []
    parts = _AUG_PART_ORDER
    for i, p in enumerate(parts):
        if p.endswith("_left"):
            mate = p[: -len("_left")] + "_right"
            if mate in parts:
                pairs.append((i, parts.index(mate)))
    _AUG_LR_SWAP_PAIRS = pairs
    return pairs


def _augment_block(block: np.ndarray, flip_x: bool, sx: float, sy: float,
                   lr_pairs: List[Tuple[int, int]]) -> np.ndarray:
    out = block.copy()
    if sx != 1.0:
        out[:, POS_X_IDX] *= sx
        out[:, VEL_X_IDX] *= sx
    if sy != 1.0:
        out[:, POS_Y_IDX] *= sy
        out[:, VEL_Y_IDX] *= sy
    if flip_x:
        out[:, POS_X_IDX] = -out[:, POS_X_IDX]
        out[:, VEL_X_IDX] = -out[:, VEL_X_IDX]
        if lr_pairs:
            for a, b in lr_pairs:
                for col_a, col_b in (
                    (2 * a, 2 * b), (2 * a + 1, 2 * b + 1),
                    (22 + 2 * a, 22 + 2 * b), (22 + 2 * a + 1, 22 + 2 * b + 1),
                    (44 + 2 * a, 44 + 2 * b), (44 + 2 * a + 1, 44 + 2 * b + 1),
                    (66 + 2 * a, 66 + 2 * b), (66 + 2 * a + 1, 66 + 2 * b + 1),
                ):
                    out[:, [col_a, col_b]] = out[:, [col_b, col_a]]
    return out


def _rotate_block(block: np.ndarray, cos_t: float, sin_t: float) -> np.ndarray:
    """Rotate (x,y) and (vx,vy) pairs; masks are NOT rotated."""
    out = block.copy()
    x = out[:, POS_X_IDX].copy()
    y = out[:, POS_Y_IDX].copy()
    out[:, POS_X_IDX] = cos_t * x - sin_t * y
    out[:, POS_Y_IDX] = sin_t * x + cos_t * y
    vx = out[:, VEL_X_IDX].copy()
    vy = out[:, VEL_Y_IDX].copy()
    out[:, VEL_X_IDX] = cos_t * vx - sin_t * vy
    out[:, VEL_Y_IDX] = sin_t * vx + cos_t * vy
    return out


def _dropout_parts(block: np.ndarray, hide_idx: np.ndarray) -> np.ndarray:
    if len(hide_idx) == 0:
        return block
    out = block.copy()
    for p in hide_idx:
        out[:, [2 * p, 2 * p + 1,
                22 + 2 * p, 22 + 2 * p + 1,
                44 + 2 * p, 44 + 2 * p + 1,
                66 + 2 * p, 66 + 2 * p + 1]] = 0.0
    return out


def apply_augmentations(features: np.ndarray, rng: np.random.Generator,
                        flip_p: float = 0.5,
                        scale_jitter: float = 0.15,
                        rot_prob: float = 0.0,
                        rot_max_deg: float = 0.0,
                        part_dropout_prob: float = 0.0,
                        part_hide_prob: float = 0.0) -> np.ndarray:
    """Augment a [T, 176] = [agent | target] window; same choices applied to both halves."""
    flip_x = rng.random() < flip_p
    sx = float(rng.uniform(1.0 - scale_jitter, 1.0 + scale_jitter)) if scale_jitter > 0 else 1.0
    sy = float(rng.uniform(1.0 - scale_jitter, 1.0 + scale_jitter)) if scale_jitter > 0 else 1.0

    do_rot = rot_prob > 0.0 and rot_max_deg > 0.0 and rng.random() < rot_prob
    if do_rot:
        theta = np.deg2rad(float(rng.uniform(-rot_max_deg, rot_max_deg)))
        cos_t = float(np.cos(theta)); sin_t = float(np.sin(theta))
    else:
        cos_t = 1.0; sin_t = 0.0

    do_drop = (part_dropout_prob > 0.0 and part_hide_prob > 0.0
               and rng.random() < part_dropout_prob)
    if do_drop:
        hide_mask = rng.random(N_PARTS) < part_hide_prob
        hide_idx = np.where(hide_mask)[0]
    else:
        hide_idx = np.array([], dtype=np.int64)

    if not flip_x and sx == 1.0 and sy == 1.0 and not do_rot and len(hide_idx) == 0:
        return features

    lr_pairs = _resolve_lr_pairs()
    agent = features[:, :88]
    target = features[:, 88:]
    agent = _augment_block(agent, flip_x, sx, sy, lr_pairs)
    target = _augment_block(target, flip_x, sx, sy, lr_pairs)
    if do_rot:
        agent = _rotate_block(agent, cos_t, sin_t)
        target = _rotate_block(target, cos_t, sin_t)
    if len(hide_idx) > 0:
        agent = _dropout_parts(agent, hide_idx)
        target = _dropout_parts(target, hide_idx)
    return np.concatenate([agent, target], axis=1).astype(np.float32, copy=False)


def compute_fbeta(tp: float, fp: float, fn: float, beta: float = BETA) -> float:
    b2 = beta ** 2
    denom = (1 + b2) * tp + b2 * fn + fp
    if denom == 0:
        return 0.0
    return (1 + b2) * tp / denom


class BehaviorDataset(Dataset):
    """Lazy mmap-backed dataset; allowed_video_dirs enables video-level splits."""
    def __init__(
        self,
        data_dir: str,
        window_size: int = WINDOW_SIZE,
        stride: int = STRIDE,
        max_videos: int = 0,
        allowed_video_dirs: Optional[Set[str]] = None,
        pair_lookup: Optional[Dict[Tuple[str, int, int], np.ndarray]] = None,
        meta_unk_p: float = 0.0,
        augment: bool = False,
        aug_flip_p: float = 0.5,
        aug_scale_jitter: float = 0.15,
        aug_rot_prob: float = 0.0,
        aug_rot_max_deg: float = 0.0,
        aug_part_dropout_prob: float = 0.0,
        aug_part_hide_prob: float = 0.0,
        time_aug_prob: float = 0.0,
        features_mode: str = "raw",
        embeddings_dir: Optional[str] = None,
        embed_dim: int = 0,
        embed_noise_std: float = 0.0,
        embed_dropout_p: float = 0.0,
    ):
        if features_mode not in FEATURES_MODES:
            raise ValueError(f"features_mode must be one of {FEATURES_MODES}, got {features_mode!r}")
        if features_mode != "raw" and not embeddings_dir:
            raise ValueError(f"features_mode={features_mode!r} requires embeddings_dir.")
        self.window_size = window_size
        self.data_dir = data_dir
        self.pair_lookup = pair_lookup
        self.meta_unk_p = float(meta_unk_p)
        self.augment = bool(augment)
        self.aug_flip_p = float(aug_flip_p)
        self.aug_scale_jitter = float(aug_scale_jitter)
        self.aug_rot_prob = float(aug_rot_prob)
        self.aug_rot_max_deg = float(aug_rot_max_deg)
        self.aug_part_dropout_prob = float(aug_part_dropout_prob)
        self.aug_part_hide_prob = float(aug_part_hide_prob)
        self.time_aug_prob = float(time_aug_prob)
        self.features_mode = features_mode
        self.embeddings_dir = embeddings_dir
        self.embed_dim = int(embed_dim)
        self.input_dim = feature_input_dim(features_mode, self.embed_dim)
        self.embed_noise_std = float(embed_noise_std)
        self.embed_dropout_p = float(embed_dropout_p)
        self._rng = np.random.default_rng()

        action_list_path = os.path.join(data_dir, "action_list.npy")
        self.action_list = np.load(action_list_path, allow_pickle=True).tolist()
        self.num_classes = len(self.action_list)

        index_df = pd.read_csv(os.path.join(data_dir, "index.csv"))
        if max_videos > 0:
            unique_videos = index_df["video_dir"].unique()[:max_videos]
            index_df = index_df[index_df["video_dir"].isin(unique_videos)]
        if allowed_video_dirs is not None:
            index_df = index_df[index_df["video_dir"].isin(allowed_video_dirs)]

        self.pair_keys: List[Tuple[str, str, int]] = []
        self.windows: List[Tuple[int, int, np.ndarray]] = []
        for _, row in index_df.iterrows():
            video_dir = row["video_dir"]
            pair_key = row["pair_key"]
            num_frames = int(row["num_frames"])
            pair_idx = len(self.pair_keys)
            self.pair_keys.append((video_dir, pair_key, num_frames))
            meta_ids = self._resolve_meta_for_seq(video_dir, pair_key)
            for start in range(0, num_frames, stride):
                self.windows.append((pair_idx, start, meta_ids))

        n_vid = index_df["video_dir"].nunique()
        meta_str = (
            "with metadata" if pair_lookup is not None else "without metadata"
        )
        print(
            f"BehaviorDataset: {len(index_df)} sequences ({n_vid} videos), "
            f"{len(self.windows)} windows, {self.num_classes} classes "
            f"[{meta_str}, unk_p={self.meta_unk_p}]"
        )

    def _resolve_meta_for_seq(self, video_dir: str, pair_key: str) -> np.ndarray:
        if self.pair_lookup is None:
            return np.zeros(N_META_SLOTS, dtype=np.int64)
        _, video_id = parse_video_dir(video_dir)
        agent_id, target_id = parse_pair_key(pair_key)
        return resolve_meta_ids(self.pair_lookup, video_id, agent_id, target_id)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pair_idx, start, meta_ids = self.windows[idx]
        video_dir, pair_key, num_frames = self.pair_keys[pair_idx]
        base = os.path.join(self.data_dir, video_dir)

        # Embeddings live in a parallel per-mouse tree (not per-pair); agent/target
        # halves are loaded separately and concat'd along the feature axis.
        raw = None
        if self.features_mode in ("raw", "concat"):
            raw = np.load(
                os.path.join(base, f"{pair_key}_features.npy"), mmap_mode='r'
            )
            num_frames = min(num_frames, len(raw))

        agent_emb = target_emb = None
        if self.features_mode in ("embed", "concat"):
            agent_id, target_id = parse_pair_key(pair_key)
            emb_base = os.path.join(self.embeddings_dir, video_dir)
            agent_emb = np.load(os.path.join(emb_base, f"{agent_id}.npy"), mmap_mode='r')
            target_emb = np.load(os.path.join(emb_base, f"{target_id}.npy"), mmap_mode='r')
            num_frames = min(num_frames, agent_emb.shape[0], target_emb.shape[0])

        labels_path = os.path.join(base, f"{pair_key}_labels.npy")
        if os.path.exists(labels_path):
            labels = np.load(labels_path, mmap_mode='r')
        else:
            labels = np.zeros((num_frames, self.num_classes), dtype=np.float32)

        mask_path = os.path.join(base, f"{pair_key}_loss_mask.npy")
        if os.path.exists(mask_path):
            loss_mask = np.load(mask_path)
        else:
            loss_mask = np.zeros(self.num_classes, dtype=np.float32)

        feat_window = torch.zeros(self.window_size, self.input_dim, dtype=torch.float32)
        label_window = torch.zeros(self.window_size, self.num_classes, dtype=torch.float32)
        padding_mask = torch.ones(self.window_size, dtype=torch.bool)

        # Time aug: every-2nd-frame over a 2× window (~15 fps from 30 fps).
        do_time_aug = (self.augment
                       and self.time_aug_prob > 0.0
                       and self._rng.random() < self.time_aug_prob
                       and start + 2 * self.window_size <= num_frames)
        if do_time_aug:
            sl = slice(start, start + 2 * self.window_size, 2)
            actual_len = self.window_size
        else:
            end = min(start + self.window_size, num_frames)
            sl = slice(start, end)
            actual_len = end - start

        label_slice = np.array(labels[sl])

        # Geometric aug operates on raw 176-d only; in concat mode the embedding half is as-is.
        parts: List[np.ndarray] = []
        if raw is not None:
            raw_slice = np.array(raw[sl], dtype=np.float32)
            if self.augment:
                raw_slice = apply_augmentations(
                    raw_slice, self._rng,
                    flip_p=self.aug_flip_p,
                    scale_jitter=self.aug_scale_jitter,
                    rot_prob=self.aug_rot_prob,
                    rot_max_deg=self.aug_rot_max_deg,
                    part_dropout_prob=self.aug_part_dropout_prob,
                    part_hide_prob=self.aug_part_hide_prob,
                )
            parts.append(raw_slice)
        if agent_emb is not None:
            a = np.asarray(agent_emb[sl], dtype=np.float32)
            t = np.asarray(target_emb[sl], dtype=np.float32)
            if self.augment and self.embed_noise_std > 0.0:
                a = a + self._rng.normal(0.0, self.embed_noise_std, a.shape).astype(np.float32)
                t = t + self._rng.normal(0.0, self.embed_noise_std, t.shape).astype(np.float32)
            if self.augment and self.embed_dropout_p > 0.0:
                # Same channel mask on agent + target keeps relative signal consistent.
                keep = (self._rng.random(a.shape[1]) >= self.embed_dropout_p).astype(np.float32)
                a = a * keep
                t = t * keep
            parts.append(a)
            parts.append(t)
        feat_slice = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=1)

        feat_window[:actual_len] = torch.from_numpy(feat_slice)
        label_window[:actual_len] = torch.from_numpy(label_slice)
        padding_mask[:actual_len] = False

        meta_ids_sample = (
            inject_unk(meta_ids, self.meta_unk_p, self._rng)
            if self.meta_unk_p > 0.0
            else meta_ids
        )

        return {
            "features": feat_window,
            "labels": label_window,
            "loss_mask": torch.from_numpy(loss_mask.astype(np.float32)),
            "padding_mask": padding_mask,
            "meta_ids": torch.from_numpy(np.ascontiguousarray(meta_ids_sample)),
        }


def create_dataloaders(
    data_dir: str,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
    batch_size: int = 32,
    val_split: float = 0.2,
    num_workers: int = 8,
    seed: int = 42,
    max_videos: int = 0,
    pair_lookup: Optional[Dict[Tuple[str, int, int], np.ndarray]] = None,
    meta_unk_p: float = 0.0,
    augment: bool = False,
    aug_flip_p: float = 0.5,
    aug_scale_jitter: float = 0.15,
    aug_rot_prob: float = 0.0,
    aug_rot_max_deg: float = 0.0,
    aug_part_dropout_prob: float = 0.0,
    aug_part_hide_prob: float = 0.0,
    time_aug_prob: float = 0.0,
    features_mode: str = "raw",
    embeddings_dir: Optional[str] = None,
    embed_dim: int = 0,
    embed_noise_std: float = 0.0,
    embed_dropout_p: float = 0.0,
) -> Tuple[DataLoader, DataLoader]:
    """Train/val split is by video_dir (not by overlapping windows)."""
    index_path = os.path.join(data_dir, "index.csv")
    index_df = pd.read_csv(index_path)
    if max_videos > 0:
        keep = index_df["video_dir"].unique()[:max_videos]
        index_df = index_df[index_df["video_dir"].isin(keep)]

    unique_videos = index_df["video_dir"].unique()
    rng = np.random.default_rng(seed)
    shuffled = unique_videos.copy()
    rng.shuffle(shuffled)

    if len(shuffled) == 0:
        train_videos: Set[str] = set()
        val_videos: Set[str] = set()
    elif len(shuffled) == 1 or val_split <= 0:
        train_videos = set(shuffled)
        val_videos = set()
    else:
        # Guarantee at least one train and one val video when len >= 2.
        n_val = max(1, int(round(len(shuffled) * float(val_split))))
        n_val = min(n_val, len(shuffled) - 1)
        val_videos = set(shuffled[:n_val])
        train_videos = set(shuffled[n_val:])

    train_ds = BehaviorDataset(
        data_dir,
        window_size=window_size,
        stride=stride,
        max_videos=0,
        allowed_video_dirs=train_videos,
        pair_lookup=pair_lookup,
        meta_unk_p=meta_unk_p,
        augment=augment,
        aug_flip_p=aug_flip_p,
        aug_scale_jitter=aug_scale_jitter,
        aug_rot_prob=aug_rot_prob,
        aug_rot_max_deg=aug_rot_max_deg,
        aug_part_dropout_prob=aug_part_dropout_prob,
        aug_part_hide_prob=aug_part_hide_prob,
        time_aug_prob=time_aug_prob,
        features_mode=features_mode,
        embeddings_dir=embeddings_dir,
        embed_dim=embed_dim,
        embed_noise_std=embed_noise_std,
        embed_dropout_p=embed_dropout_p,
    )
    val_ds = BehaviorDataset(
        data_dir,
        window_size=window_size,
        stride=stride,
        max_videos=0,
        allowed_video_dirs=val_videos,
        pair_lookup=pair_lookup,
        meta_unk_p=0.0,
        augment=False,
        features_mode=features_mode,
        embeddings_dir=embeddings_dir,
        embed_dim=embed_dim,
    )

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    n_train_w = len(train_ds)
    n_val_w = len(val_ds)
    print(
        f"Video-level split (seed={seed}, val_split={val_split}): "
        f"{len(train_videos)} train videos, {len(val_videos)} val videos | "
        f"{n_train_w} train windows, {n_val_w} val windows"
    )
    if len(val_videos) == 0 and len(shuffled) > 0:
        print("  (no val videos — checkpointing uses train Fb)")

    return train_loader, val_loader


class TemporalCNN(nn.Module):
    """1D temporal conv stack with a residual connection. Shape [B, T, d_model]."""
    def __init__(self, d_model: int, kernel_size: int = 3):
        super().__init__()
        # Dilated stack: receptive field = 1 + 2*(1+2+4+8) = 31 frames (~1s @ 30fps).
        self.convs = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=1, dilation=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=2, dilation=2),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=4, dilation=4),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=8, dilation=8),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.transpose(1, 2)
        x = self.convs(x)
        x = x.transpose(1, 2)
        return x + residual


class MABeTransformer(nn.Module):
    def __init__(self, input_dim: int,
        num_classes: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        window_size: int,
        meta_vocab_sizes: Optional[List[int]] = None,
        meta_slot_to_table: Optional[List[int]] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )

        self.temporal_cnn = TemporalCNN(d_model)

        # Metadata embedding tables applied AFTER the CNN, BEFORE the transformer.
        self.meta_slot_to_table: List[int] = list(meta_slot_to_table or [])
        if meta_vocab_sizes:
            self.meta_embeddings = nn.ModuleList([
                nn.Embedding(int(vsize), d_model) for vsize in meta_vocab_sizes
            ])
            for emb in self.meta_embeddings:
                nn.init.normal_(emb.weight, mean=0.0, std=0.02)
        else:
            self.meta_embeddings = nn.ModuleList()

        self.encoder_layers = nn.ModuleList([
            Encoder(d_model, nhead, dim_feedforward, dropout, max_seq_len=window_size)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

        # +1 for the trailing "none" / background class used by softmax CE
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_classes + 1),
        )

    def _add_meta(self, x: torch.Tensor, meta_ids: Optional[torch.Tensor]) -> torch.Tensor:
        if not self.meta_embeddings or meta_ids is None or not self.meta_slot_to_table:
            return x
        for slot, t_idx in enumerate(self.meta_slot_to_table):
            emb = self.meta_embeddings[t_idx](meta_ids[:, slot])
            x = x + emb.unsqueeze(1)
        return x

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
        meta_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.temporal_cnn(x)
        x = self._add_meta(x, meta_ids)
        # Positional info injected via relative positional bias inside attention.
        for layer in self.encoder_layers:
            x = layer(x, padding_mask)
        x = self.final_norm(x)
        x = self.output_projection(x)
        return x


def masked_cross_entropy_loss(logits, labels, loss_mask, padding_mask):
    """Softmax CE with 'none' class at index C.

    Per frame: target = unique active supervised class, or 'none' if zero;
    frames with >1 active labels are dropped. Unsupervised logits → -inf.
    """
    B, T, C_plus_1 = logits.shape
    C = C_plus_1 - 1
    NONE_IDX = C

    active = labels * loss_mask.unsqueeze(1)
    n_active = active.sum(dim=-1)

    # Mask unsupervised action logits → -inf; "none" is always supervised.
    none_col = torch.ones(B, 1, device=logits.device, dtype=loss_mask.dtype)
    full_loss_mask = torch.cat([loss_mask, none_col], dim=-1)
    keep = full_loss_mask.bool().unsqueeze(1).expand(-1, T, -1)
    masked_logits = logits.masked_fill(~keep, float('-inf'))

    n_active_long = n_active.long()
    target = torch.where(
        n_active_long >= 1,
        active.argmax(dim=-1),
        torch.full_like(n_active_long, NONE_IDX),
    )

    # Loss-eligible frames: not padded AND not multi-active.
    frame_mask = (~padding_mask) & (n_active_long <= 1)

    ce = nn.functional.cross_entropy(
        masked_logits.reshape(-1, C_plus_1),
        target.reshape(-1),
        reduction='none',
    ).reshape(B, T)

    masked_loss = (ce * frame_mask.float()).sum()
    denom = frame_mask.float().sum().clamp(min=1.0)
    return masked_loss / denom


def train_one_epoch(model: MABeTransformer, data_loader, optimizer, device, beta: float = BETA):
    model.train()
    total_loss = 0.0
    total_tp, total_fp, total_fn = 0, 0, 0
    num_batches = len(data_loader)
    if num_batches == 0:
        return 0.0, 0.0
    log_points = {int(num_batches * p) for p in (0.25, 0.5, 0.75)} - {0}

    for batch_idx, batch in enumerate(data_loader):
        x = batch["features"].to(device)
        labels = batch["labels"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        loss_mask = batch["loss_mask"].to(device)
        meta_ids = batch.get("meta_ids")
        if meta_ids is not None:
            meta_ids = meta_ids.to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x, padding_mask, meta_ids)
            loss = masked_cross_entropy_loss(logits, labels, loss_mask, padding_mask)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)

        with torch.no_grad():
            # Predicted class fires iff argmax over (C+1) is not NONE_IDX.
            full_argmax = logits.argmax(dim=-1)
            C = labels.shape[-1]
            fired = (full_argmax != C)
            pred_class = full_argmax.clamp(max=C - 1)
            preds = torch.zeros_like(labels)
            preds.scatter_(-1, pred_class.unsqueeze(-1), 1.0)
            preds = preds * fired.float().unsqueeze(-1)

            frame_mask = (~padding_mask).float().unsqueeze(-1)
            action_mask = loss_mask.unsqueeze(1)
            full_mask = frame_mask * action_mask

            total_tp += ((preds * labels) * full_mask).sum().item()
            total_fp += ((preds * (1 - labels)) * full_mask).sum().item()
            total_fn += (((1 - preds) * labels) * full_mask).sum().item()

        if batch_idx in log_points:
            pct = int(100 * (batch_idx + 1) / num_batches)
            running_fbeta = compute_fbeta(total_tp, total_fp, total_fn, beta)
            running_loss = total_loss / ((batch_idx + 1) * data_loader.batch_size)
            print(f"  [{pct:>3d}%] loss: {running_loss:.4f}  Fb: {running_fbeta:.3f}")

    n_samples = len(data_loader.dataset)
    avg_loss = total_loss / max(n_samples, 1)
    fbeta = compute_fbeta(total_tp, total_fp, total_fn, beta)
    return avg_loss, fbeta


def _overall_fbeta_argmax(pair_data, pair_keys, beta: float) -> float:
    """Competition F-beta with argmax-over-(C+1) predictions; argmax==C → no action."""
    from collections import defaultdict
    from metadata import parse_video_dir

    by_lab_action: Dict[Tuple[str, int], List[int]] = defaultdict(list)
    for pi, (_, _, loss_mask) in enumerate(pair_data):
        if loss_mask is None:
            continue
        lab_id, _ = parse_video_dir(pair_keys[pi][0])
        for a in np.where(loss_mask > 0.5)[0]:
            by_lab_action[(lab_id, int(a))].append(pi)

    lab_action_fb: Dict[str, List[float]] = defaultdict(list)
    for (lab_id, a), pair_indices in by_lab_action.items():
        tp = fp = fn = 0.0
        any_positive = False
        for pi in pair_indices:
            probs_full, labels_arr, _ = pair_data[pi]
            C = labels_arr.shape[1]
            argmax_full = probs_full.argmax(axis=-1)
            preds = argmax_full == a
            lbls = labels_arr[:, a] > 0.5
            if lbls.any():
                any_positive = True
            tp += float((preds & lbls).sum())
            fp += float((preds & ~lbls).sum())
            fn += float((~preds & lbls).sum())
        if any_positive:
            lab_action_fb[lab_id].append(compute_fbeta(tp, fp, fn, beta))

    lab_means = [float(np.mean(fbs)) for fbs in lab_action_fb.values() if fbs]
    if not lab_means:
        return 0.0
    return float(np.mean(lab_means))


@torch.no_grad()
def validate(model: MABeTransformer, data_loader, device, beta: float = BETA):
    """Aggregate per-pair softmax probs across overlapping windows; score by argmax."""
    if len(data_loader.dataset) == 0:
        return 0.0, 0.0

    model.eval()
    val_ds = data_loader.dataset
    pair_keys = val_ds.pair_keys
    num_classes = val_ds.num_classes
    C_plus_1 = num_classes + 1

    # Keep the trailing "none" channel — argmax over (C+1) needs it.
    prob_sum = [np.zeros((nf, C_plus_1), dtype=np.float32) for _, _, nf in pair_keys]
    counts = [np.zeros(nf, dtype=np.int32) for _, _, nf in pair_keys]
    labels_buf = [np.zeros((nf, num_classes), dtype=np.float32) for _, _, nf in pair_keys]
    loss_masks: List[Optional[np.ndarray]] = [None] * len(pair_keys)

    total_loss = 0.0
    n_samples = 0
    counter = 0
    windows = val_ds.windows

    for batch in data_loader:
        x = batch["features"].to(device)
        labels_t = batch["labels"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        loss_mask_t = batch["loss_mask"].to(device)
        meta_ids = batch.get("meta_ids")
        if meta_ids is not None:
            meta_ids = meta_ids.to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x, padding_mask, meta_ids)
            loss = masked_cross_entropy_loss(logits, labels_t, loss_mask_t, padding_mask)
        logits = logits.float()
        total_loss += loss.item() * x.size(0)
        n_samples += x.size(0)

        # Mask unsupervised classes so softmax only spreads over supervised + "none".
        none_col = torch.ones(loss_mask_t.size(0), 1, device=loss_mask_t.device,
                              dtype=loss_mask_t.dtype)
        full_loss_mask = torch.cat([loss_mask_t, none_col], dim=-1)
        keep = full_loss_mask.bool().unsqueeze(1).expand(-1, logits.size(1), -1)
        masked_logits = logits.masked_fill(~keep, float('-inf'))
        probs_full = masked_logits.softmax(dim=-1).cpu().numpy()
        pad_np = padding_mask.cpu().numpy()
        labels_np = labels_t.cpu().numpy()
        mask_np = loss_mask_t.cpu().numpy()

        B = probs_full.shape[0]
        for b in range(B):
            pair_idx, start, _meta = windows[counter + b]
            actual_len = int((~pad_np[b]).sum())
            if actual_len <= 0:
                continue
            end = min(start + actual_len, prob_sum[pair_idx].shape[0])
            actual_len = end - start
            if actual_len <= 0:
                continue
            prob_sum[pair_idx][start:end] += probs_full[b, :actual_len]
            counts[pair_idx][start:end] += 1
            labels_buf[pair_idx][start:end] = labels_np[b, :actual_len]
            if loss_masks[pair_idx] is None:
                loss_masks[pair_idx] = mask_np[b]
        counter += B

    avg_loss = total_loss / max(n_samples, 1)

    pair_data = []
    for pi, (_, _, _) in enumerate(pair_keys):
        c = counts[pi].clip(min=1).astype(np.float32)
        avg_probs = prob_sum[pi] / c[:, None]
        pair_data.append((avg_probs, labels_buf[pi], loss_masks[pi]))

    fb_argmax = _overall_fbeta_argmax(pair_data, pair_keys, beta)
    return avg_loss, fb_argmax


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(
    data_dir: str,
    model_dir: str = ".",
    num_epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-4,
    num_workers: int = 4,
    max_videos: int = 0,
    val_split: float = 0.2,
    seed: int = 42,
    beta: float = BETA,
    train_csv: Optional[str] = None,
    meta_unk_p: float = META_UNK_P_DEFAULT,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
    warmup_epochs: int = 0,
    augment: bool = True,
    aug_flip_p: float = 0.5,
    aug_scale_jitter: float = 0.15,
    aug_rot_prob: float = 0.0,
    aug_rot_max_deg: float = 0.0,
    aug_part_dropout_prob: float = 0.0,
    aug_part_hide_prob: float = 0.0,
    time_aug_prob: float = 0.0,
    features_mode: str = "raw",
    embeddings_dir: Optional[str] = None,
    embed_noise_std: float = 0.0,
    embed_dropout_p: float = 0.0,
):
    device = get_device()
    print(f"Device: {device} | F-beta (β={beta})")

    embed_dim = 0
    if features_mode != "raw":
        if not embeddings_dir:
            raise ValueError(f"--features_mode={features_mode} requires --embeddings_dir")
        embed_dim = read_embedding_dim(embeddings_dir)
        print(f"Features: mode={features_mode}  embeddings_dir={embeddings_dir}  "
              f"embed_dim={embed_dim}")
        if features_mode == "concat":
            print("NOTE: concat mode applies geometric augmentations (flip / "
                  "rotate / scale / part-dropout) to the 176-d raw stream "
                  "ONLY. Embeddings are taken as-is from cache (Path 2). For "
                  "fully-aligned augmentation see "
                  "project_posebert_aug_caches_deferred (Path 1).")
    else:
        print(f"Features: mode=raw (input_dim={NUM_FEATURES})")

    pair_lookup = None
    meta_vocab_sizes: Optional[List[int]] = None
    meta_slot_to_table: Optional[List[int]] = None
    if train_csv and os.path.exists(train_csv):
        print(f"Building metadata vocabs from {train_csv}")
        vocabs = build_vocabs(train_csv)
        meta_vocab_sizes = vocab_sizes_per_table(vocabs)
        meta_slot_to_table = slot_to_table_index()
        pair_lookup = build_pair_lookup(train_csv, vocabs)
        print(
            f"  vocab sizes (lab, strain, sex) = {meta_vocab_sizes}; "
            f"{len(pair_lookup)} (video, agent, target) pairs in lookup"
        )
    else:
        print(f"WARNING: train_csv={train_csv!r} not found; training without metadata.")

    print(f"window_size={window_size}, stride={stride}")
    train_loader, val_loader = create_dataloaders(
        data_dir,
        window_size=window_size,
        stride=stride,
        batch_size=batch_size,
        num_workers=num_workers,
        max_videos=max_videos,
        val_split=val_split,
        seed=seed,
        pair_lookup=pair_lookup,
        meta_unk_p=meta_unk_p if pair_lookup is not None else 0.0,
        augment=augment,
        aug_flip_p=aug_flip_p,
        aug_scale_jitter=aug_scale_jitter,
        aug_rot_prob=aug_rot_prob,
        aug_rot_max_deg=aug_rot_max_deg,
        aug_part_dropout_prob=aug_part_dropout_prob,
        aug_part_hide_prob=aug_part_hide_prob,
        time_aug_prob=time_aug_prob,
        features_mode=features_mode,
        embeddings_dir=embeddings_dir,
        embed_dim=embed_dim,
        embed_noise_std=embed_noise_std,
        embed_dropout_p=embed_dropout_p,
    )
    if augment:
        print(
            f"Train augmentations ON: flip_p={aug_flip_p}, "
            f"scale_jitter=±{aug_scale_jitter:.2f}, "
            f"rot_prob={aug_rot_prob} max±{aug_rot_max_deg:.0f}°, "
            f"part_dropout={aug_part_dropout_prob}/hide={aug_part_hide_prob}, "
            f"time_aug={time_aug_prob}, "
            f"embed_noise_std={embed_noise_std}, embed_dropout_p={embed_dropout_p}"
        )
    else:
        print("Train augmentations OFF")

    num_classes = train_loader.dataset.num_classes
    input_dim = feature_input_dim(features_mode, embed_dim)
    print(f"num_classes={num_classes}, input_dim={input_dim}")

    model = MABeTransformer(
        input_dim=input_dim,
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
    model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    if warmup_epochs > 0 and warmup_epochs < num_epochs:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs - warmup_epochs
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
        )
        print(f"LR schedule: linear warmup {warmup_epochs} epochs → cosine {num_epochs - warmup_epochs} epochs (peak lr={lr:.2e})")
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    import json as _json
    os.makedirs(model_dir, exist_ok=True)
    best_val_fb = 0.0
    best_train_fb = 0.0
    save_path = os.path.join(model_dir, "best_model.pt")
    metrics_path = os.path.join(model_dir, "metrics.jsonl")
    use_val_for_checkpoint = len(val_loader.dataset) > 0

    if use_val_for_checkpoint:
        print(
            "Metrics legend (softmax variant):\n"
            "  Train Fb  = running F-beta during training (argmax over C+1, per-window).\n"
            "  Val Fb    = aggregated val F-beta using per-frame argmax over (C+1).\n"
            "              An action 'fires' iff it wins against the 'none' class.\n"
            "              No thresholds — best model selection uses this metric."
        )

    for epoch in range(num_epochs):
        train_loss, train_fb = train_one_epoch(model, train_loader, optimizer, device, beta=beta)
        if use_val_for_checkpoint:
            val_loss, val_fb = validate(model, val_loader, device, beta=beta)
        else:
            val_loss, val_fb = 0.0, 0.0
        scheduler.step()

        if use_val_for_checkpoint:
            val_str = f"Val loss: {val_loss:.4f} Fb: {val_fb:.3f}"
        else:
            val_str = "Val: (no videos)"
        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train loss {train_loss:.4f} Fb: {train_fb:.3f} | "
            f"{val_str} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        with open(metrics_path, "a") as f:
            f.write(_json.dumps({
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "train_fb": float(train_fb),
                "val_loss": float(val_loss),
                "val_fb": float(val_fb),
                "lr": float(scheduler.get_last_lr()[0]),
            }) + "\n")

        improved = False
        if use_val_for_checkpoint:
            if val_fb > best_val_fb:
                best_val_fb = val_fb
                improved = True
                tag = f"val_Fb={val_fb:.3f}"
        else:
            if train_fb > best_train_fb:
                best_train_fb = train_fb
                improved = True
                tag = f"train_Fb={train_fb:.3f}"

        if improved:
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved best model ({tag})")

    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train MABe behavior model")
    parser.add_argument("data_dir", nargs="?", default="./processed_data")
    parser.add_argument("model_dir", nargs="?", default=".")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_videos", type=int, default=0,
                        help="Train on subset of N videos (0 = all)")
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta", type=float, default=BETA)
    parser.add_argument("--train_csv", type=str, default=DEFAULT_TRAIN_CSV,
                        help="train.csv for metadata; missing file → train without metadata.")
    parser.add_argument("--meta_unk_p", type=float, default=META_UNK_P_DEFAULT,
                        help="Per-slot UNK injection prob to keep UNK embedding trained.")
    parser.add_argument("--window_size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--warmup_epochs", type=int, default=0,
                        help="Linear LR warmup epochs then cosine anneal.")
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--aug_flip_p", type=float, default=0.5)
    parser.add_argument("--aug_scale_jitter", type=float, default=0.15)
    parser.add_argument("--aug_rot_prob", type=float, default=0.0)
    parser.add_argument("--aug_rot_max_deg", type=float, default=0.0)
    parser.add_argument("--aug_part_dropout_prob", type=float, default=0.0)
    parser.add_argument("--aug_part_hide_prob", type=float, default=0.0)
    parser.add_argument("--time_aug_prob", type=float, default=0.0,
                        help="Prob of every-2nd-frame over 2x window (~15 fps).")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--features_mode", choices=FEATURES_MODES, default="raw",
                        help="raw | embed | concat. embed/concat require --embeddings_dir.")
    parser.add_argument("--embeddings_dir", type=str, default=None,
                        help="PoseBERT embeddings cache (required when features_mode != raw).")
    parser.add_argument("--embed_noise_std", type=float, default=0.0)
    parser.add_argument("--embed_dropout_p", type=float, default=0.0)
    args = parser.parse_args()

    import json as _json
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    if args.run_name:
        run_name = args.run_name
    else:
        run_name = (
            f"w{args.window_size}_s{args.stride}_lr{args.lr:.0e}"
            f"_bs{args.batch_size}_e{args.num_epochs}_{timestamp}"
        )
    run_dir = os.path.join(args.model_dir, "runs", run_name)
    os.makedirs(run_dir, exist_ok=True)

    log_path = os.path.join(run_dir, "train.log")
    tee = TeeLogger(log_path)
    sys.stdout = tee
    print(f"Run dir: {run_dir}")
    print(f"Log: {log_path}")
    print(f"Command: {' '.join(sys.argv)}")
    print(f"Started: {datetime.now().isoformat()}")
    print("-" * 60)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        cfg = vars(args).copy()
        cfg["run_name"] = run_name
        cfg["started"] = datetime.now().isoformat()
        _json.dump(cfg, f, indent=2)

    # Update 'latest' symlink at START so it tracks the running model, not the last completed.
    latest = os.path.join(args.model_dir, "latest")
    try:
        if os.path.islink(latest) or os.path.exists(latest):
            os.remove(latest)
        os.symlink(os.path.relpath(run_dir, args.model_dir), latest)
        print(f"Updated symlink: {latest} -> runs/{run_name}")
    except OSError as e:
        print(f"WARN: failed to update 'latest' symlink: {e}")

    try:
        train(
            args.data_dir,
            model_dir=run_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            num_workers=args.num_workers,
            max_videos=args.max_videos,
            val_split=args.val_split,
            seed=args.seed,
            beta=args.beta,
            train_csv=args.train_csv,
            meta_unk_p=args.meta_unk_p,
            window_size=args.window_size,
            stride=args.stride,
            warmup_epochs=args.warmup_epochs,
            augment=not args.no_augment,
            aug_flip_p=args.aug_flip_p,
            aug_scale_jitter=args.aug_scale_jitter,
            aug_rot_prob=args.aug_rot_prob,
            aug_rot_max_deg=args.aug_rot_max_deg,
            aug_part_dropout_prob=args.aug_part_dropout_prob,
            aug_part_hide_prob=args.aug_part_hide_prob,
            time_aug_prob=args.time_aug_prob,
            features_mode=args.features_mode,
            embeddings_dir=args.embeddings_dir,
            embed_noise_std=args.embed_noise_std,
            embed_dropout_p=args.embed_dropout_p,
        )
    finally:
        print("-" * 60)
        print(f"Finished: {datetime.now().isoformat()}")
        tee.close()



