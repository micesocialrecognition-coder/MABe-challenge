"""Build one 88-d per-mouse-per-frame feature vector (176-d for agent-centric directed pairs)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

ALL_PARTS_FULL = sorted(
    set(
        [
            "ear_left",
            "ear_right",
            "hip_left",
            "hip_right",
            "neck",
            "nose",
            "tail_base",
            "body_center",
            "lateral_left",
            "lateral_right",
            "tail_tip",
            "tail_midpoint",
            "spine_1",
            "spine_2",
            "tail_middle_1",
            "tail_middle_2",
            "head",
            "headpiece_bottombackleft",
            "headpiece_bottombackright",
            "headpiece_bottomfrontleft",
            "headpiece_bottomfrontright",
            "headpiece_topbackleft",
            "headpiece_topbackright",
            "headpiece_topfrontleft",
            "headpiece_topfrontright",
        ]
    )
)

DROP_PARTS = [
    "headpiece_bottombackleft",
    "headpiece_bottombackright",
    "headpiece_bottomfrontleft",
    "headpiece_bottomfrontright",
    "headpiece_topbackleft",
    "headpiece_topbackright",
    "headpiece_topfrontleft",
    "headpiece_topfrontright",
    "spine_1",
    "spine_2",
    "tail_middle_1",
    "tail_middle_2",
    "tail_midpoint",
    "head",
]

ALL_PARTS: List[str] = [p for p in ALL_PARTS_FULL if p not in DROP_PARTS]

# V2 adds 4 paws + tail_midpoint, only present in ~90% of pretrain videos.
BODY_PARTS_V1: List[str] = ALL_PARTS
BODY_PARTS_V2: List[str] = ALL_PARTS + [
    "forepaw_left", "forepaw_right",
    "hindpaw_left", "hindpaw_right",
    "tail_midpoint",
]

BODY_PARTS_REGISTRY = {"v1": BODY_PARTS_V1, "v2": BODY_PARTS_V2}


def feature_columns(parts_order: Optional[List[str]] = None) -> List[str]:
    """Fixed 88 column order: coords, position masks, velocities, velocity masks."""
    parts_order = parts_order or ALL_PARTS
    coord_cols: List[str] = [f"{c}_{p}" for p in parts_order for c in ("x", "y")]
    mask_cols: List[str] = [f"m_{c}" for c in coord_cols]
    vel_cols: List[str] = [f"v{c}" for c in coord_cols]
    m_vel_cols: List[str] = [f"m_v{c}" for c in coord_cols]
    cols = coord_cols + mask_cols + vel_cols + m_vel_cols
    if len(cols) != 88:
        raise ValueError(f"Expected 88 feature columns, got {len(cols)}")
    return cols


def pivot_long_to_wide(
    tracking_one_mouse: pd.DataFrame,
    parts_order: List[str],
    video_frames: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Long rows for one mouse → wide table indexed by video_frame with x_/y_ columns in pixels."""
    if video_frames is None:
        video_frames = sorted(tracking_one_mouse["video_frame"].unique())

    sub = tracking_one_mouse.copy()
    wide = sub.pivot_table(index="video_frame", columns="bodypart", values=["x", "y"])
    cols = [(coord, part) for part in parts_order for coord in ("x", "y")]
    wide = wide.reindex(columns=pd.MultiIndex.from_tuples(cols, names=["coord", "bodypart"]))
    wide.columns = [f"{coord}_{part}" for coord, part in wide.columns.to_list()]
    wide = wide.sort_index().reindex(video_frames)
    return wide


def pixels_to_cm(wide_pixels: pd.DataFrame, pix_per_cm: float) -> pd.DataFrame:
    """Divide all coordinate columns by ``pix_per_cm`` (safe fallback if 0/None → 1)."""
    ppcm = float(pix_per_cm) if pix_per_cm is not None and float(pix_per_cm) != 0 else 1.0
    return wide_pixels.astype(np.float32) / ppcm


def position_masks_from_raw(wide_cm_before_fill: pd.DataFrame) -> pd.DataFrame:
    """For each ``x_*`` / ``y_*`` column, add ``m_*`` = 1.0 if finite, else 0.0."""
    raw = wide_cm_before_fill
    mask_df = (~raw.isna()).astype(np.float32)
    mask_df.columns = [f"m_{c}" for c in raw.columns]
    return mask_df


def fill_missing_positions(wide_cm: pd.DataFrame) -> pd.DataFrame:
    """Interpolate → ffill → bfill → 0 (used only for transforms & velocities input)."""
    filled = wide_cm.copy()
    filled = filled.interpolate(limit_direction="both").ffill().bfill().fillna(0.0)
    filled = filled.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return filled


def compute_agent_frame(
    agent_raw_cm: pd.DataFrame,
    agent_filled_cm: pd.DataFrame,
    parts_order: List[str],
    pix_per_cm: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame agent origin (ears midpoint w/ fallbacks) + rotation so heading is along -y."""
    has_ear_l = "x_ear_left" in agent_raw_cm.columns and "y_ear_left" in agent_raw_cm.columns
    has_ear_r = "x_ear_right" in agent_raw_cm.columns and "y_ear_right" in agent_raw_cm.columns
    has_body_center = "x_body_center" in agent_raw_cm.columns and "y_body_center" in agent_raw_cm.columns
    has_neck = "x_neck" in agent_raw_cm.columns and "y_neck" in agent_raw_cm.columns
    has_tail_base = "x_tail_base" in agent_raw_cm.columns and "y_tail_base" in agent_raw_cm.columns
    has_tail_tip = "x_tail_tip" in agent_raw_cm.columns and "y_tail_tip" in agent_raw_cm.columns

    T = len(agent_raw_cm)
    origin_x = np.zeros(T, dtype=np.float32)
    origin_y = np.zeros(T, dtype=np.float32)

    if has_ear_l and has_ear_r:
        ear_l_x = agent_raw_cm["x_ear_left"].to_numpy(dtype=np.float32)
        ear_l_y = agent_raw_cm["y_ear_left"].to_numpy(dtype=np.float32)
        ear_r_x = agent_raw_cm["x_ear_right"].to_numpy(dtype=np.float32)
        ear_r_y = agent_raw_cm["y_ear_right"].to_numpy(dtype=np.float32)
        l_valid = (~np.isnan(ear_l_x)) & (~np.isnan(ear_l_y))
        r_valid = (~np.isnan(ear_r_x)) & (~np.isnan(ear_r_y))
        both = l_valid & r_valid
        left_only = l_valid & ~r_valid
        right_only = r_valid & ~l_valid
        origin_x[both] = 0.5 * (ear_l_x[both] + ear_r_x[both])
        origin_y[both] = 0.5 * (ear_l_y[both] + ear_r_y[both])
        origin_x[left_only] = ear_l_x[left_only]
        origin_y[left_only] = ear_l_y[left_only]
        origin_x[right_only] = ear_r_x[right_only]
        origin_y[right_only] = ear_r_y[right_only]
    elif has_ear_l:
        ear_l_x = agent_raw_cm["x_ear_left"].to_numpy(dtype=np.float32)
        ear_l_y = agent_raw_cm["y_ear_left"].to_numpy(dtype=np.float32)
        l_valid = (~np.isnan(ear_l_x)) & (~np.isnan(ear_l_y))
        origin_x[l_valid] = ear_l_x[l_valid]
        origin_y[l_valid] = ear_l_y[l_valid]
    elif has_ear_r:
        ear_r_x = agent_raw_cm["x_ear_right"].to_numpy(dtype=np.float32)
        ear_r_y = agent_raw_cm["y_ear_right"].to_numpy(dtype=np.float32)
        r_valid = (~np.isnan(ear_r_x)) & (~np.isnan(ear_r_y))
        origin_x[r_valid] = ear_r_x[r_valid]
        origin_y[r_valid] = ear_r_y[r_valid]

    if has_body_center:
        bc_x = agent_raw_cm["x_body_center"].to_numpy(dtype=np.float32)
        bc_y = agent_raw_cm["y_body_center"].to_numpy(dtype=np.float32)
        bc_valid = (~np.isnan(bc_x)) & (~np.isnan(bc_y))
        use = bc_valid & (origin_x == 0.0) & (origin_y == 0.0)
        origin_x[use] = bc_x[use]
        origin_y[use] = bc_y[use]
    elif has_neck:
        neck_x = agent_raw_cm["x_neck"].to_numpy(dtype=np.float32)
        neck_y = agent_raw_cm["y_neck"].to_numpy(dtype=np.float32)
        neck_valid = (~np.isnan(neck_x)) & (~np.isnan(neck_y))
        use = neck_valid & (origin_x == 0.0) & (origin_y == 0.0)
        origin_x[use] = neck_x[use]
        origin_y[use] = neck_y[use]

    cos_a = np.ones(T, dtype=np.float32)
    sin_a = np.zeros(T, dtype=np.float32)
    alpha = np.zeros(T, dtype=np.float32)

    if has_tail_base:
        tb_x = agent_raw_cm["x_tail_base"].to_numpy(dtype=np.float32)
        tb_y = agent_raw_cm["y_tail_base"].to_numpy(dtype=np.float32)
        tb_valid = (~np.isnan(tb_x)) & (~np.isnan(tb_y))

        if has_tail_tip:
            tt_x = agent_raw_cm["x_tail_tip"].to_numpy(dtype=np.float32)
            tt_y = agent_raw_cm["y_tail_tip"].to_numpy(dtype=np.float32)
            tt_valid = (~np.isnan(tt_x)) & (~np.isnan(tt_y))
        else:
            tt_x = tt_y = np.zeros(T, dtype=np.float32)
            tt_valid = np.zeros(T, dtype=bool)

        if "x_neck" in agent_raw_cm.columns and "y_neck" in agent_raw_cm.columns:
            neck_x = agent_raw_cm["x_neck"].to_numpy(dtype=np.float32)
            neck_y = agent_raw_cm["y_neck"].to_numpy(dtype=np.float32)
            neck_valid = (~np.isnan(neck_x)) & (~np.isnan(neck_y))
        else:
            neck_x = neck_y = np.zeros(T, dtype=np.float32)
            neck_valid = np.zeros(T, dtype=bool)

        use_tip = tb_valid & tt_valid
        use_neck = (~tt_valid) & tb_valid & neck_valid
        axes_valid = use_tip | use_neck

        if axes_valid.any():
            h_x = np.where(use_tip, (tt_x - tb_x), np.where(use_neck, (neck_x - tb_x), 0.0))
            h_y = np.where(use_tip, (tt_y - tb_y), np.where(use_neck, (neck_y - tb_y), 0.0))
            norm = np.sqrt(h_x * h_x + h_y * h_y).astype(np.float32)
            good = axes_valid & (norm > 1e-6)
            if good.any():
                u_x = h_x[good] / norm[good]
                u_y = h_y[good] / norm[good]
                u_angle = np.arctan2(u_y, u_x)
                alpha0 = (-np.pi / 2.0) - u_angle
                alpha[good] = alpha0.astype(np.float32)

                tb_filled_x = agent_filled_cm["x_tail_base"].to_numpy(dtype=np.float32)
                tb_filled_y = agent_filled_cm["y_tail_base"].to_numpy(dtype=np.float32)
                tb_rel_x = tb_filled_x - origin_x
                tb_rel_y = tb_filled_y - origin_y
                ca0 = np.cos(alpha[good]).astype(np.float32)
                sa0 = np.sin(alpha[good]).astype(np.float32)
                tb_y_prime = sa0 * tb_rel_x[good] + ca0 * tb_rel_y[good]
                flip = tb_y_prime < 0
                if flip.any():
                    idx = np.where(good)[0][flip]
                    alpha[idx] = (alpha[idx] + np.pi).astype(np.float32)

        cos_a = np.cos(alpha).astype(np.float32)
        sin_a = np.sin(alpha).astype(np.float32)

    return origin_x, origin_y, cos_a, sin_a


def _rotate_xy(x_t: np.ndarray, y_t: np.ndarray, cos_a: np.ndarray, sin_a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x_rot = cos_a * x_t - sin_a * y_t
    y_rot = sin_a * x_t + cos_a * y_t
    return x_rot, y_rot


def to_canonical_coords(
    filled_cm: pd.DataFrame,
    origin_x: np.ndarray,
    origin_y: np.ndarray,
    cos_a: np.ndarray,
    sin_a: np.ndarray,
    parts_order: List[str],
) -> pd.DataFrame:
    """Translate by origin then rotate per frame → DataFrame x_*, y_* in canonical space."""
    x_cols = [f"x_{p}" for p in parts_order]
    y_cols = [f"y_{p}" for p in parts_order]
    x_t = filled_cm[x_cols].to_numpy(dtype=np.float32) - origin_x[:, None]
    y_t = filled_cm[y_cols].to_numpy(dtype=np.float32) - origin_y[:, None]
    x_rot, y_rot = _rotate_xy(x_t, y_t, cos_a[:, None], sin_a[:, None])
    return pd.DataFrame(
        np.concatenate([x_rot, y_rot], axis=1),
        index=filled_cm.index,
        columns=x_cols + y_cols,
    )


def assemble_feature_vector(canonical_xy: pd.DataFrame, pos_masks: pd.DataFrame, fps: float, parts_order: List[str]) -> pd.DataFrame:
    """Combine canonical positions, position masks, velocities, and velocity masks → 88 cols float32."""
    x_cols = [f"x_{p}" for p in parts_order]
    y_cols = [f"y_{p}" for p in parts_order]

    vx = canonical_xy[x_cols].diff().fillna(0.0) * float(fps)
    vy = canonical_xy[y_cols].diff().fillna(0.0) * float(fps)
    vx.columns = [f"vx_{p}" for p in parts_order]
    vy.columns = [f"vy_{p}" for p in parts_order]

    m_x = pos_masks[[f"m_x_{p}" for p in parts_order]]
    m_y = pos_masks[[f"m_y_{p}" for p in parts_order]]
    m_x_prev = m_x.shift(1).fillna(0.0)
    m_y_prev = m_y.shift(1).fillna(0.0)
    m_vx = (m_x * m_x_prev).astype(np.float32)
    m_vy = (m_y * m_y_prev).astype(np.float32)
    m_vx.columns = [f"m_vx_{p}" for p in parts_order]
    m_vy.columns = [f"m_vy_{p}" for p in parts_order]

    vel_masks_df = pd.concat([m_vx, m_vy], axis=1)
    out = pd.concat([canonical_xy[x_cols + y_cols], pos_masks, vx, vy, vel_masks_df], axis=1)
    out = out.reindex(columns=feature_columns(parts_order))
    for c in out.columns:
        out[c] = out[c].astype(np.float32)
    return out


def process_directed_pair_agent_centric(
    tracking_df: pd.DataFrame,
    agent_id: int,
    target_id: int,
    fps: float,
    pix_per_cm: float,
    parts_order: Optional[List[str]] = None,
    drop_parts: Optional[List[str]] = None,
    video_frames: Optional[List[int]] = None,
    bodyparts_already_dropped: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """End-to-end for one directed pair → (agent_features, target_features) each [T, 88].

    ``bodyparts_already_dropped=True`` skips a per-pair copy when iterating many pairs on one video.
    """
    parts_order = parts_order or ALL_PARTS
    drop_parts = drop_parts or DROP_PARTS

    if bodyparts_already_dropped:
        df = tracking_df
    else:
        df = tracking_df.copy()
        df = df[~df["bodypart"].isin(drop_parts)].copy()

    if video_frames is None:
        video_frames = sorted(df["video_frame"].unique())

    ppcm = float(pix_per_cm) if pix_per_cm is not None and float(pix_per_cm) != 0 else 1.0

    agent_wide_px = pivot_long_to_wide(
        df[df["mouse_id"] == agent_id], parts_order=parts_order, video_frames=video_frames
    )
    target_wide_px = pivot_long_to_wide(
        df[df["mouse_id"] == target_id], parts_order=parts_order, video_frames=video_frames
    )

    agent_cm = pixels_to_cm(agent_wide_px, ppcm)
    target_cm = pixels_to_cm(target_wide_px, ppcm)

    agent_masks = position_masks_from_raw(agent_cm)
    target_masks = position_masks_from_raw(target_cm)

    agent_filled = fill_missing_positions(agent_cm)
    target_filled = fill_missing_positions(target_cm)

    ox, oy, cos_a, sin_a = compute_agent_frame(
        agent_raw_cm=agent_cm,
        agent_filled_cm=agent_filled,
        parts_order=parts_order,
        pix_per_cm=ppcm,
    )

    agent_canon = to_canonical_coords(
        agent_filled, ox, oy, cos_a, sin_a, parts_order
    )
    target_canon = to_canonical_coords(
        target_filled, ox, oy, cos_a, sin_a, parts_order
    )

    agent_features = assemble_feature_vector(agent_canon, agent_masks, fps, parts_order)
    target_features = assemble_feature_vector(target_canon, target_masks, fps, parts_order)

    return agent_features, target_features
