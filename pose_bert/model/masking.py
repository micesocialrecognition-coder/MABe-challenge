"""Temporal span masking for PoseBERT SSL pretraining (pure-numpy, runs in DataLoader workers)."""

import numpy as np
import torch


def generate_span_mask(
    seq_len: int,
    window_size: int,
    mask_ratio: float = 0.15,
    min_span: int = 5,
    max_span: int = 30,
    rng: np.random.RandomState | np.random.Generator | None = None,
) -> torch.BoolTensor:
    """Generate a [window_size] span mask for one sequence (True = masked)."""
    if rng is None:
        rng = np.random

    mask = np.zeros(window_size, dtype=bool)

    if seq_len == 0:
        return torch.from_numpy(mask)

    if seq_len < min_span:
        mask[:seq_len] = True
        return torch.from_numpy(mask)

    effective_max_span = min(max_span, seq_len)
    effective_min_span = min(min_span, effective_max_span)
    target_masked = max(int(seq_len * mask_ratio), effective_min_span)

    masked_count = 0
    max_iters = 1000
    for _ in range(max_iters):
        if masked_count >= target_masked:
            break
        span_len = rng.randint(effective_min_span, effective_max_span + 1)
        start = rng.randint(0, seq_len - span_len + 1)
        mask[start : start + span_len] = True
        masked_count = mask[:seq_len].sum()

    return torch.from_numpy(mask)


def generate_two_disjoint_span_masks(
    seq_len: int,
    window_size: int,
    mask_ratio_pos: float = 0.10,
    mask_ratio_vel: float = 0.10,
    min_span: int = 5,
    max_span: int = 30,
    rng: np.random.RandomState | np.random.Generator | None = None,
) -> tuple[torch.BoolTensor, torch.BoolTensor]:
    """Two non-overlapping span masks (pos, vel) so the two heads can't shortcut."""
    if rng is None:
        rng = np.random

    mask_pos = np.zeros(window_size, dtype=bool)
    mask_vel = np.zeros(window_size, dtype=bool)

    if seq_len == 0 or seq_len < min_span:
        if seq_len > 0:
            mask_pos[:seq_len] = True
        return torch.from_numpy(mask_pos), torch.from_numpy(mask_vel)

    effective_max_span = min(max_span, seq_len)
    effective_min_span = min(min_span, effective_max_span)

    # Position mask: same logic as generate_span_mask.
    target_pos = max(int(seq_len * mask_ratio_pos), effective_min_span)
    placed = 0
    for _ in range(1000):
        if placed >= target_pos:
            break
        span_len = rng.randint(effective_min_span, effective_max_span + 1)
        start = rng.randint(0, seq_len - span_len + 1)
        mask_pos[start : start + span_len] = True
        placed = mask_pos[:seq_len].sum()

    # Velocity mask: reject spans that touch mask_pos.
    target_vel = max(int(seq_len * mask_ratio_vel), effective_min_span)
    placed = 0
    for _ in range(2000):  # higher retry budget because of rejection
        if placed >= target_vel:
            break
        span_len = rng.randint(effective_min_span, effective_max_span + 1)
        start = rng.randint(0, seq_len - span_len + 1)
        if mask_pos[start : start + span_len].any():
            continue
        if mask_vel[start : start + span_len].any():
            continue
        mask_vel[start : start + span_len] = True
        placed = mask_vel[:seq_len].sum()

    return torch.from_numpy(mask_pos), torch.from_numpy(mask_vel)


def generate_two_disjoint_span_masks_batch(
    padding_mask: torch.BoolTensor,
    mask_ratio_pos: float = 0.15,
    mask_ratio_vel: float = 0.15,
    min_span: int = 5,
    max_span: int = 30,
    generator: torch.Generator | None = None,
) -> tuple[torch.BoolTensor, torch.BoolTensor]:
    """Batched generate_two_disjoint_span_masks (used by validate() for determinism)."""
    batch_size, window_size = padding_mask.shape
    if generator is None:
        rng = np.random.RandomState(0)
    else:
        rng = np.random.RandomState(generator.initial_seed() % (2 ** 32))

    masks_pos, masks_vel = [], []
    for i in range(batch_size):
        seq_len = int((~padding_mask[i]).sum().item())
        m_pos, m_vel = generate_two_disjoint_span_masks(
            seq_len=seq_len,
            window_size=window_size,
            mask_ratio_pos=mask_ratio_pos,
            mask_ratio_vel=mask_ratio_vel,
            min_span=min_span,
            max_span=max_span,
            rng=rng,
        )
        masks_pos.append(m_pos)
        masks_vel.append(m_vel)
    return torch.stack(masks_pos), torch.stack(masks_vel)


def generate_span_mask_batch(
    padding_mask: torch.BoolTensor,
    mask_ratio: float = 0.15,
    min_span: int = 5,
    max_span: int = 30,
    generator: torch.Generator | None = None,
) -> torch.BoolTensor:
    """Batched span masks (used by validate(); training masks per-sample in __getitem__)."""
    batch_size, window_size = padding_mask.shape
    if generator is None:
        rng = np.random.RandomState(0)
    else:
        rng = np.random.RandomState(generator.initial_seed() % (2 ** 32))

    masks = []
    for i in range(batch_size):
        seq_len = int((~padding_mask[i]).sum().item())
        masks.append(generate_span_mask(
            seq_len=seq_len,
            window_size=window_size,
            mask_ratio=mask_ratio,
            min_span=min_span,
            max_span=max_span,
            rng=rng,
        ))
    return torch.stack(masks)
