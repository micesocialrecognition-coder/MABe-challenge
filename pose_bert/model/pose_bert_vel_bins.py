"""PoseBERT Velocity-Bins variant: single span mask + hard-target CE on polar Δ_t bins."""

import math

import torch
import torch.nn as nn

from pose_bert.model.binning import (
    MAX_NORM_CM_30FPS,
    N_VEL_R_BINS,
    N_VEL_THETA_BINS,
    velocity_to_bin,
)


POSEBERT_CONFIG = {
    'input_dim': 33,
    'output_dim': 22,           # unused here but kept for backward compat
    'd_model': 256,
    'nhead': 8,
    'num_layers': 6,
    'dim_feedforward': 1024,
    'dropout': 0.1,
    'meta_vocab_sizes': [22, 10, 6, 3, 13],
    'window_size': 128,
    'mask_ratio_single': 0.30,  # matches mask30_* baseline
    'min_span': 5,
    'max_span': 30,
    'n_vel_r_bins': N_VEL_R_BINS,
    'n_vel_theta_bins': N_VEL_THETA_BINS,
    'max_norm_cm': MAX_NORM_CM_30FPS,
}


class PoseBERT(nn.Module):
    """Masked-autoencoder transformer with a velocity-bin classification head."""

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

        self.d_model = d_model
        self.config = config

        # Layout: [2*n_parts coords | n_parts masks] => input_dim == 3 * n_parts.
        if input_dim % 3 != 0:
            raise ValueError(f"input_dim={input_dim} not divisible by 3 (expected 3*n_parts)")
        self.n_parts = input_dim // 3

        self.n_vel_r_bins = int(config['n_vel_r_bins'])
        self.n_vel_theta_bins = int(config['n_vel_theta_bins'])
        self.max_norm_cm = float(config['max_norm_cm'])
        self.n_vel_classes = self.n_vel_r_bins * self.n_vel_theta_bins

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

        # Per-frame logits for n_parts body parts × n_vel_classes.
        self.reconstruction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, self.n_parts * self.n_vel_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        meta_ids: torch.Tensor,
        padding_mask: torch.Tensor,
        span_mask: torch.Tensor,
    ) -> dict:
        B, T, _ = x.shape
        n_parts = self.n_parts
        coord_dim = 2 * n_parts

        h = self.input_projection(x)
        mask_expanded = span_mask.unsqueeze(-1).expand_as(h)
        h = torch.where(mask_expanded, self.mask_token.expand(B, T, -1), h)
        h = h + self.pos_encoding[:T]
        for i, emb in enumerate(self.meta_embeddings):
            h = h + emb(meta_ids[:, i]).unsqueeze(1)
        h = self.transformer_encoder(h, src_key_padding_mask=padding_mask)

        logits = self.reconstruction_head(h)
        logits = logits.view(B, T, n_parts, self.n_vel_classes)

        # Δ_t = x_t - x_{t-1} from the ORIGINAL (unmasked) x; skipped at t=0
        # and at frames where t or t-1 is unobserved per body part.
        target_xy = x[:, :, :coord_dim].view(B, T, n_parts, 2)
        obs_mask = x[:, :, coord_dim:]

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

        t_index = torch.arange(T, device=x.device).view(1, T, 1)
        has_prev = (t_index > 0).float()
        valid_frame = (span_mask & ~padding_mask).float()
        weight = (
            valid_frame.unsqueeze(-1)
            * has_prev
            * obs_mask
            * prev_obs
        )
        num_valid = weight.sum().clamp(min=1.0)

        log_probs = torch.log_softmax(logits, dim=-1)
        gathered = log_probs.gather(
            dim=-1, index=vel_target.unsqueeze(-1)
        ).squeeze(-1)
        loss = -(gathered * weight).sum() / num_valid

        return {
            'loss': loss,
            'predictions': logits,
            'num_masked': int((span_mask & ~padding_mask).sum().item()),
        }

    def extract(
        self,
        x: torch.Tensor,
        meta_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return encoder hidden states for downstream fine-tuning."""
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
