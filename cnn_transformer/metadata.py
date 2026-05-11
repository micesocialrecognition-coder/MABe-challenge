"""Per-pair metadata vocab/lookup helpers for the MABe challenge.
Example: from metadata import build_vocabs, build_pair_lookup
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


UNK_IDX = 0
META_UNK_P_DEFAULT = 0.05

META_FIELDS: List[Tuple[str, str, str]] = [
    ("lab",    "lab_id",            "video"),
    ("strain", "mouse{mid}_strain", "mouse"),
    ("sex",    "mouse{mid}_sex",    "mouse"),
]

SLOT_FIELD_NAMES: List[str] = ["lab", "strain", "strain", "sex", "sex"]
N_META_SLOTS = len(SLOT_FIELD_NAMES)


def build_vocabs(train_csv: str) -> Dict[str, Dict[str, int]]:
    df = pd.read_csv(train_csv)
    vocabs: Dict[str, Dict[str, int]] = {}
    for field_name, col_tmpl, scope in META_FIELDS:
        values: set = set()
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


def vocab_sizes_per_table(vocabs: Dict[str, Dict[str, int]]) -> List[int]:
    return [len(vocabs[name]) for name, _, _ in META_FIELDS]


def slot_to_table_index() -> List[int]:
    name_to_table = {name: i for i, (name, _, _) in enumerate(META_FIELDS)}
    return [name_to_table[name] for name in SLOT_FIELD_NAMES]


def _per_mouse_lookup(train_csv: str, vocabs: Dict[str, Dict[str, int]]) -> Tuple[Dict[str, int], Dict[Tuple[str, int], Tuple[int, int]]]:
    df = pd.read_csv(train_csv)
    df["video_id"] = df["video_id"].astype(str)

    video_to_lab: Dict[str, int] = {}
    video_mouse_to_strsex: Dict[Tuple[str, int], Tuple[int, int]] = {}

    lab_vocab = vocabs["lab"]
    strain_vocab = vocabs["strain"]
    sex_vocab = vocabs["sex"]

    for _, row in df.iterrows():
        vid = str(row["video_id"])
        lab_val = row.get("lab_id")
        if isinstance(lab_val, float) and np.isnan(lab_val):
            video_to_lab[vid] = UNK_IDX
        else:
            video_to_lab[vid] = lab_vocab.get(str(lab_val), UNK_IDX)

        for mid in (1, 2, 3, 4):
            sc = f"mouse{mid}_strain"
            xc = f"mouse{mid}_sex"
            strain_idx = UNK_IDX
            sex_idx = UNK_IDX
            if sc in df.columns:
                v = row[sc]
                if not (isinstance(v, float) and np.isnan(v)):
                    strain_idx = strain_vocab.get(str(v), UNK_IDX)
            if xc in df.columns:
                v = row[xc]
                if not (isinstance(v, float) and np.isnan(v)):
                    sex_idx = sex_vocab.get(str(v), UNK_IDX)
            video_mouse_to_strsex[(vid, mid)] = (strain_idx, sex_idx)

    return video_to_lab, video_mouse_to_strsex


def build_pair_lookup(train_csv: str, vocabs: Dict[str, Dict[str, int]]) -> Dict[Tuple[str, int, int], np.ndarray]:
    # Slot order matches SLOT_FIELD_NAMES: [lab, agent_strain, target_strain, agent_sex, target_sex].
    video_to_lab, vm = _per_mouse_lookup(train_csv, vocabs)
    pair_lookup: Dict[Tuple[str, int, int], np.ndarray] = {}


    for (vid, mid), _ in vm.items():
        if vid not in video_to_lab:
            continue
        lab_idx = video_to_lab[vid]

    by_video: Dict[str, List[int]] = {}
    for (vid, mid) in vm.keys():
        by_video.setdefault(vid, []).append(mid)

    for vid, mids in by_video.items():
        lab_idx = video_to_lab.get(vid, UNK_IDX)
        for a in mids:
            for t in mids:
                a_strain, a_sex = vm.get((vid, a), (UNK_IDX, UNK_IDX))
                t_strain, t_sex = vm.get((vid, t), (UNK_IDX, UNK_IDX))
                pair_lookup[(vid, a, t)] = np.array(
                    [lab_idx, a_strain, t_strain, a_sex, t_sex],
                    dtype=np.int64,
                )

    return pair_lookup


def resolve_meta_ids(pair_lookup: Dict[Tuple[str, int, int], np.ndarray], video_id: str, agent_id: int, target_id: int) -> np.ndarray:
    key = (str(video_id), int(agent_id), int(target_id))
    hit = pair_lookup.get(key)
    if hit is not None:
        return hit
    return np.zeros(N_META_SLOTS, dtype=np.int64)


def inject_unk(meta_ids: np.ndarray, p: float, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    # Keeps the UNK embedding trained so test-time unseen values hit a meaningful representation.
    if p <= 0.0:
        return meta_ids
    if rng is None:
        rng = np.random.default_rng()
    drop = rng.random(N_META_SLOTS) < p
    if not drop.any():
        return meta_ids
    out = meta_ids.copy()
    out[drop] = UNK_IDX
    return out


def parse_pair_key(pair_key: str) -> Tuple[int, int]:
    a, t = pair_key.split("_")
    return int(a), int(t)


def parse_video_dir(video_dir: str) -> Tuple[str, str]:
    parts = str(video_dir).split("/")
    if len(parts) < 2:
        return ("", str(video_dir))
    return parts[0], parts[1]
