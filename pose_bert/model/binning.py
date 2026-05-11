"""Position/velocity binning utilities for PoseBERT pretraining variants."""

from __future__ import annotations

import math
from typing import Tuple

import torch


N_POS_BINS = 28            # 28 x 28 = 784 classes/part; 52/28 ≈ 1.86 cm/bin
POSITION_RANGE_CM = 26.0   # half-width of the dominant 52cm arena (covers 99.4% of videos)

N_VEL_R_BINS = 16
N_VEL_THETA_BINS = 16
N_VEL_BINS = N_VEL_R_BINS * N_VEL_THETA_BINS
MAX_NORM_CM_30FPS = 1.5    # cm/frame at 30fps; covers walking + most running

SIGMA_BINS_DEFAULT = 1.0


def position_to_bin(
    xy: torch.Tensor,
    n_bins: int = N_POS_BINS,
    arena_range_cm: float = POSITION_RANGE_CM,
) -> torch.Tensor:
    """Map (x, y) cm coords to a flat bin index = x_bin * n_bins + y_bin."""
    R = arena_range_cm
    x = xy[..., 0]
    y = xy[..., 1]

    # Map [-R, +R] -> [0, n_bins-1]; clamp keeps out-of-range at the edge.
    x_bin = torch.round((x + R) / (2.0 * R) * (n_bins - 1))
    y_bin = torch.round((y + R) / (2.0 * R) * (n_bins - 1))
    x_bin = x_bin.clamp(0, n_bins - 1).long()
    y_bin = y_bin.clamp(0, n_bins - 1).long()

    return x_bin * n_bins + y_bin


def position_soft_label(
    xy: torch.Tensor,
    n_bins: int = N_POS_BINS,
    arena_range_cm: float = POSITION_RANGE_CM,
    sigma_bins: float = SIGMA_BINS_DEFAULT,
) -> torch.Tensor:
    """2D-Gaussian soft label over the (x_bin, y_bin) grid, flattened to n_bins**2 classes."""
    R = arena_range_cm
    cell_cm = 2.0 * R / (n_bins - 1)
    sigma_cm = sigma_bins * cell_cm

    centers = torch.linspace(-R, R, n_bins, device=xy.device, dtype=xy.dtype)
    cx = centers.view(1, n_bins, 1).expand(1, n_bins, n_bins)
    cy = centers.view(1, 1, n_bins).expand(1, n_bins, n_bins)

    x = xy[..., 0:1].unsqueeze(-1)
    y = xy[..., 1:2].unsqueeze(-1)

    d2 = (x - cx) ** 2 + (y - cy) ** 2

    logits = -0.5 * d2 / (sigma_cm ** 2)
    flat = logits.flatten(start_dim=-2)
    probs = torch.softmax(flat, dim=-1)
    return probs


def decode_position_bin(
    bin_idx: torch.Tensor,
    n_bins: int = N_POS_BINS,
    arena_range_cm: float = POSITION_RANGE_CM,
) -> torch.Tensor:
    """Inverse of position_to_bin: bin index -> bin center (x, y) in cm."""
    R = arena_range_cm
    x_bin = (bin_idx // n_bins).float()
    y_bin = (bin_idx % n_bins).float()
    x = x_bin / (n_bins - 1) * (2.0 * R) - R
    y = y_bin / (n_bins - 1) * (2.0 * R) - R
    return torch.stack([x, y], dim=-1)


def velocity_to_bin(
    dxdy: torch.Tensor,
    n_r_bins: int = N_VEL_R_BINS,
    n_theta_bins: int = N_VEL_THETA_BINS,
    max_norm_cm: float = MAX_NORM_CM_30FPS,
) -> torch.Tensor:
    """Map (dx, dy) cm displacement to a flat polar bin = r_bin * n_theta_bins + theta_bin."""
    dx = dxdy[..., 0]
    dy = dxdy[..., 1]

    r = torch.sqrt(dx * dx + dy * dy)
    r = r.clamp(max=max_norm_cm)
    r_bin = torch.round(r / max_norm_cm * (n_r_bins - 1)).long().clamp(0, n_r_bins - 1)

    theta = torch.atan2(dy, dx)
    theta_norm = (theta + math.pi) / (2 * math.pi)
    theta_bin = torch.round(theta_norm * (n_theta_bins - 1)).long()
    theta_bin = theta_bin.clamp(0, n_theta_bins - 1)

    return r_bin * n_theta_bins + theta_bin


def soft_target_cross_entropy(
    logits: torch.Tensor,
    target_probs: torch.Tensor,
    weight: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Cross-entropy with a target distribution (not one-hot)."""
    log_probs = torch.log_softmax(logits, dim=-1)
    per_elem = -(target_probs * log_probs).sum(dim=-1)
    if weight is not None:
        per_elem = per_elem * weight
        if reduction == "mean":
            denom = weight.sum().clamp(min=1.0)
            return per_elem.sum() / denom
        if reduction == "sum":
            return per_elem.sum()
        return per_elem

    if reduction == "mean":
        return per_elem.mean()
    if reduction == "sum":
        return per_elem.sum()
    return per_elem
