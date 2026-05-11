"""Labels and loss-mask helpers from train.csv + interval annotations."""

from __future__ import annotations

import ast
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_behaviors_cell(cell) -> List[str]:
    """Parse ``train_meta['behaviors_labeled']`` cell into a list of entry strings."""
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    if isinstance(cell, list):
        return [str(x) for x in cell]
    if isinstance(cell, str):
        s = cell.strip()
        if not s:
            return []
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            return []
    return []


def parse_whitelist_entry(entry: str) -> Optional[Tuple[int, int, str]]:
    """Parse ``"mouse1,mouse2,shepherd"`` → ``(1, 2, "shepherd")``. Action may contain commas."""
    if not entry:
        return None
    parts = [p.strip() for p in str(entry).split(",")]
    if len(parts) < 3:
        return None
    a = parts[0]
    t = parts[1]
    action = ",".join(parts[2:]).strip()
    if not action:
        return None

    def _mouse_id(x: str) -> Optional[int]:
        x = x.lower()
        if not x.startswith("mouse"):
            return None
        num = x.replace("mouse", "").strip()
        try:
            return int(num)
        except Exception:
            return None

    agent_id = _mouse_id(a)
    if agent_id is None:
        return None
    if t.strip().lower() == "self":
        target_id = agent_id
    else:
        target_id = _mouse_id(t)
    if target_id is None:
        return None
    return agent_id, target_id, action


def build_action_universe(
    train_meta: pd.DataFrame,
) -> Tuple[
    List[str],
    Dict[Tuple[str, str], Dict[Tuple[int, int], Set[str]]],
    Dict[str, Set[str]],
]:
    """Returns (action_list, per_video_pair_whitelist, lab_action_sets)."""
    all_actions: Set[str] = set()
    per_video_pair_whitelist: Dict[Tuple[str, str], Dict[Tuple[int, int], Set[str]]] = {}
    lab_action_sets: Dict[str, Set[str]] = {}

    train_meta = train_meta.copy()
    train_meta["video_id"] = train_meta["video_id"].astype(str)

    for row in tqdm(train_meta.itertuples(index=False), total=len(train_meta), desc="Scan behaviors_labeled"):
        lab_id = getattr(row, "lab_id")
        video_id = getattr(row, "video_id")
        entries = parse_behaviors_cell(getattr(row, "behaviors_labeled"))
        if not entries:
            per_video_pair_whitelist[(lab_id, video_id)] = {}
            continue

        pair_dict: Dict[Tuple[int, int], Set[str]] = {}
        for e in entries:
            parsed = parse_whitelist_entry(e)
            if parsed is None:
                continue
            agent_id, target_id, action = parsed
            all_actions.add(action)
            pair_dict.setdefault((agent_id, target_id), set()).add(action)
            lab_action_sets.setdefault(lab_id, set()).add(action)

        per_video_pair_whitelist[(lab_id, video_id)] = pair_dict

    action_list = sorted(all_actions)
    return action_list, per_video_pair_whitelist, lab_action_sets


def intervals_to_frame_labels(
    ann_df: pd.DataFrame,
    vf: np.ndarray,
    agent_id: int,
    target_id: int,
    action_to_idx: Dict[str, int],
    action_dim: int,
) -> np.ndarray:
    """Map annotation intervals to ``[T, action_dim]`` float32 multi-hot labels (inclusive)."""
    T = len(vf)
    y = np.zeros((T, action_dim), dtype=np.float32)

    sub = ann_df[(ann_df["agent_id"] == agent_id) & (ann_df["target_id"] == target_id)].copy()
    if len(sub) == 0:
        return y

    for r in sub.itertuples(index=False):
        action = getattr(r, "action")
        if action not in action_to_idx:
            continue
        ai = action_to_idx[action]
        start_frame = int(getattr(r, "start_frame"))
        stop_frame = int(getattr(r, "stop_frame"))

        mask_rows = (vf >= start_frame) & (vf <= stop_frame)
        if mask_rows.any():
            y[mask_rows, ai] = 1.0
    return y


def build_pair_loss_mask(
    pair_whitelist: Dict[Tuple[int, int], Set[str]],
    agent_id: int,
    target_id: int,
    action_to_idx: Dict[str, int],
    action_dim: int,
) -> np.ndarray:
    """``[action_dim]`` mask: 1.0 where the action is whitelisted for this pair."""
    mask = np.zeros((action_dim,), dtype=np.float32)
    allowed_actions = pair_whitelist.get((agent_id, target_id), set())
    for act in allowed_actions:
        if act in action_to_idx:
            mask[action_to_idx[act]] = 1.0
    return mask


def build_lab_loss_mask(
    lab_action_set: Set[str],
    action_to_idx: Dict[str, int],
    action_dim: int,
) -> np.ndarray:
    """``[action_dim]`` mask: 1.0 for any action ever whitelisted in this lab."""
    mask = np.zeros((action_dim,), dtype=np.float32)
    for act in lab_action_set:
        if act in action_to_idx:
            mask[action_to_idx[act]] = 1.0
    return mask


if __name__ == "__main__":
    raise SystemExit(
        "This module provides helpers for labels + loss masks.\n"
        "Use build_npy_dataset.py for end-to-end preprocessing."
    )
