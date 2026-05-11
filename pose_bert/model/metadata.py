"""Metadata conditioning for PoseBERT: five additive embeddings (lab, strain, arena, sex, age)."""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# Order is FIXED — column index into the [B, 5] meta_ids tensor.
# Scope "video": look up by video_id; "mouse": use mouse{mid}_<col>.
METADATA_FIELDS: List[Tuple[str, str, str]] = [
    ("lab_id",     "lab_id",             "video"),
    ("strain",     "mouse{mid}_strain",  "mouse"),
    ("arena_type", "arena_type",         "video"),
    ("sex",        "mouse{mid}_sex",     "mouse"),
    ("age",        "mouse{mid}_age",     "mouse"),
]
N_META_FIELDS = len(METADATA_FIELDS)
UNK_IDX = 0
META_UNK_P_DEFAULT = 0.05


def build_vocabs(csv_path: str) -> Dict[str, Dict[str, int]]:
    """Build {field: {value: idx}} from train.csv; index 0 reserved for <UNK>."""
    df = pd.read_csv(csv_path)
    vocabs: Dict[str, Dict[str, int]] = {}
    for field_name, col_tmpl, scope in METADATA_FIELDS:
        values = set()
        if scope == "video":
            if col_tmpl in df.columns:
                values.update(str(v) for v in df[col_tmpl].dropna().unique())
        else:
            for mid in (1, 2, 3, 4):
                col = col_tmpl.format(mid=mid)
                if col in df.columns:
                    values.update(str(v) for v in df[col].dropna().unique())
        vocab = {"<UNK>": UNK_IDX}
        for v in sorted(values):
            vocab[v] = len(vocab)
        vocabs[field_name] = vocab
    return vocabs


def vocab_sizes(vocabs: Dict[str, Dict[str, int]]) -> List[int]:
    """Return [vocab_size_for_field_i] in METADATA_FIELDS order."""
    return [len(vocabs[name]) for name, _, _ in METADATA_FIELDS]


def build_lookup(csv_path: str, vocabs: Dict[str, Dict[str, int]]) -> Dict[Tuple[str, int], np.ndarray]:
    """{(video_id, mouse_id): np.array([5], int64)} for fast per-track lookup; NaN -> UNK_IDX."""
    df = pd.read_csv(csv_path)
    lookup: Dict[Tuple[str, int], np.ndarray] = {}
    for _, row in df.iterrows():
        video_id = str(row["video_id"])
        for mid in (1, 2, 3, 4):
            ids = np.zeros(N_META_FIELDS, dtype=np.int64)
            for i, (field_name, col_tmpl, scope) in enumerate(METADATA_FIELDS):
                col = col_tmpl if scope == "video" else col_tmpl.format(mid=mid)
                if col not in df.columns:
                    continue
                val = row[col]
                if pd.isna(val):
                    continue
                ids[i] = vocabs[field_name].get(str(val), UNK_IDX)
            lookup[(video_id, mid)] = ids
    return lookup


def inject_unk(
    meta_ids: np.ndarray,
    p: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Randomly replace each field with UNK_IDX with probability p."""
    if p <= 0.0:
        return meta_ids
    drop = rng.random(N_META_FIELDS) < p
    if not drop.any():
        return meta_ids
    out = meta_ids.copy()
    out[drop] = UNK_IDX
    return out
