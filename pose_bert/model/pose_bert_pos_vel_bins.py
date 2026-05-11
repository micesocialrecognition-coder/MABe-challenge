"""PoseBERT pos_vel_bins: disjoint pos/vel span masks, two CE heads (16x16 pos + polar vel).

Usage:
    model = PoseBERT(POSEBERT_CONFIG)
    out = model(x, meta_ids, padding_mask, span_mask_pos, span_mask_vel)
"""

import math

import torch
import torch.nn as nn

from pose_bert.model.binning import (
    MAX_NORM_CM_30FPS,
    N_POS_BINS,
    N_VEL_R_BINS,
    N_VEL_THETA_BINS,
    POSITION_RANGE_CM,
    SIGMA_BINS_DEFAULT,
    position_soft_label,
    soft_target_cross_entropy,
    velocity_to_bin,
)


POSEBERT_CONFIG = {
    'input_dim': 33,
    'output_dim': 22,
    'd_model': 256,
    'nhead': 8,
    'num_layers': 6,
    'dim_feedforward': 1024,
    'dropout': 0.1,
    'meta_vocab_sizes': [22, 10, 6, 3, 13],
    'window_size': 128,
    # Total mask budget across both heads is 0.30 (matches mask30 baseline).
    'mask_ratio_pos': 0.15,
    'mask_ratio_vel': 0.15,
    'min_span': 5,
    'max_span': 30,
    'n_pos_bins': N_POS_BINS,
    'position_range_cm': POSITION_RANGE_CM,
    'sigma_bins': SIGMA_BINS_DEFAULT,
    'n_vel_r_bins': N_VEL_R_BINS,
    'n_vel_theta_bins': N_VEL_THETA_BINS,
    'max_norm_cm': MAX_NORM_CM_30FPS,
    'alpha_pos': 1.0,
    'beta_vel': 0.5,
}


class PoseBERT(nn.Module):
    """Two-head pos+vel variant with disjoint span masks."""

    def __init__(self, config: dict):
        super().__init__()

        input_dim = config['input_dim']
        d_model = config['d_model']
        nhead = config['nhead']
        num_layers = config['num_layers']
        dim_feedforward = config['dim_feedforward']
        dropout = config['dropout']
        meta_vocab_sizes = config['meta_vocab_sizes']
        window_size = config['window_size']

        # Body-part count is derived from input_dim. Layout is
        # [2*n_parts coords | n_parts masks] -> input_dim == 3 * n_parts.
        if input_dim % 3 != 0:
            raise ValueError(f"input_dim={input_dim} not divisible by 3 (expected 3*n_parts)")
        self.n_parts = input_dim // 3

        self.d_model = d_model
        self.config = config

        self.n_pos_bins = int(config['n_pos_bins'])
        self.position_range_cm = float(config['position_range_cm'])
        self.sigma_bins = float(config['sigma_bins'])
        self.n_pos_classes = self.n_pos_bins ** 2

        self.n_vel_r_bins = int(config['n_vel_r_bins'])
        self.n_vel_theta_bins = int(config['n_vel_theta_bins'])
        self.max_norm_cm = float(config['max_norm_cm'])
        self.n_vel_classes = self.n_vel_r_bins * self.n_vel_theta_bins

        self.alpha_pos = float(config.get('alpha_pos', 1.0))
        self.beta_vel = float(config.get('beta_vel', 0.5))

        # Encoder body identical to pos_raw.
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )
        position = torch.arange(window_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(math.log(10000.0) / d_model)
        )
        pe = torch.zeros(window_size, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_encoding', pe)
        self.meta_embeddings = nn.ModuleList([
            nn.Embedding(vsize, d_model) for vsize in meta_vocab_sizes
        ])
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.position_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, self.n_parts * self.n_pos_classes),
        )
        self.velocity_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, self.n_parts * self.n_vel_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        meta_ids: torch.Tensor,
        padding_mask: torch.Tensor,
        span_mask_pos: torch.Tensor,
        span_mask_vel: torch.Tensor,
    ) -> dict:
        """Encode at union of masks; pos soft-CE on span_mask_pos, vel hard-CE on disjoint span_mask_vel."""
        B, T, _ = x.shape
        n_parts = self.n_parts
        coord_dim = 2 * n_parts

        # Encoder sees [MASK] at the union of both masks (anything supervised is hidden).
        union_mask = span_mask_pos | span_mask_vel
        h = self.input_projection(x)
        mask_expanded = union_mask.unsqueeze(-1).expand_as(h)
        h = torch.where(mask_expanded, self.mask_token.expand(B, T, -1), h)
        h = h + self.pos_encoding[:T]
        for i, emb in enumerate(self.meta_embeddings):
            h = h + emb(meta_ids[:, i]).unsqueeze(1)
        h = self.transformer_encoder(h, src_key_padding_mask=padding_mask)

        pos_logits = self.position_head(h).view(B, T, n_parts, self.n_pos_classes)
        vel_logits = self.velocity_head(h).view(B, T, n_parts, self.n_vel_classes)

        target_xy = x[:, :, :coord_dim].view(B, T, n_parts, 2)
        obs_mask = x[:, :, coord_dim:]

        target_probs_pos = position_soft_label(
            target_xy,
            n_bins=self.n_pos_bins,
            arena_range_cm=self.position_range_cm,
            sigma_bins=self.sigma_bins,
        )

        valid_frame_pos = (span_mask_pos & ~padding_mask).float()
        weight_pos = valid_frame_pos.unsqueeze(-1) * obs_mask
        num_pos = weight_pos.sum().clamp(min=1.0)

        loss_pos = soft_target_cross_entropy(
            logits=pos_logits,
            target_probs=target_probs_pos,
            weight=weight_pos,
            reduction="mean",
        )

        # Velocity target Δ_t = x_t - x_{t-1}, computed from ORIGINAL (unmasked) x.
        # Pad (x_{-1}) with zeros; t=0 masked out below via has_prev.
        prev_xy = torch.zeros_like(target_xy)
        prev_xy[:, 1:] = target_xy[:, :-1]
        prev_obs = torch.zeros_like(obs_mask)
        prev_obs[:, 1:] = obs_mask[:, :-1]

        delta = target_xy - prev_xy
        vel_target = velocity_to_bin(
            delta,
            n_r_bins=self.n_vel_r_bins,
            n_theta_bins=self.n_vel_theta_bins,
            max_norm_cm=self.max_norm_cm,
        )

        # Validity: t>0, in span_mask_vel, not padded, both x_t and x_{t-1} observed.
        t_index = torch.arange(T, device=x.device).view(1, T, 1)
        has_prev = (t_index > 0).float()
        valid_frame_vel = (span_mask_vel & ~padding_mask).float()
        weight_vel = (
            valid_frame_vel.unsqueeze(-1)
            * has_prev
            * obs_mask
            * prev_obs
        )
        num_vel = weight_vel.sum().clamp(min=1.0)

        log_probs_vel = torch.log_softmax(vel_logits, dim=-1)
        gathered = log_probs_vel.gather(
            dim=-1, index=vel_target.unsqueeze(-1)
        ).squeeze(-1)
        loss_vel = -(gathered * weight_vel).sum() / num_vel

        loss = self.alpha_pos * loss_pos + self.beta_vel * loss_vel

        return {
            'loss': loss,
            'loss_pos': loss_pos.detach(),
            'loss_vel': loss_vel.detach(),
            'predictions': (pos_logits, vel_logits),
            'num_masked_pos': int((span_mask_pos & ~padding_mask).sum().item()),
            'num_masked_vel': int((span_mask_vel & ~padding_mask).sum().item()),
            # for compatibility with train.py logging that reads 'num_masked'
            'num_masked': int(((span_mask_pos | span_mask_vel) & ~padding_mask).sum().item()),
        }

    def extract(self, x: torch.Tensor, meta_ids: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return encoder hidden states for downstream — identical to pos_raw."""
        B, T, _ = x.shape
        h = self.input_projection(x)
        h = h + self.pos_encoding[:T]
        for i, emb in enumerate(self.meta_embeddings):
            h = h + emb(meta_ids[:, i]).unsqueeze(1)
        h = self.transformer_encoder(h, src_key_padding_mask=padding_mask)
        return h


def count_parameters(model: nn.Module) -> int:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total:,} total, {trainable:,} trainable")
    return trainable
