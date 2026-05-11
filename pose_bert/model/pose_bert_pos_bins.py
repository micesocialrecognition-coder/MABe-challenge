"""PoseBERT Position-Bins variant: pos_raw encoder + 2D-Gaussian soft-target CE on binned (x,y)."""

import math

import torch
import torch.nn as nn

from pose_bert.model.binning import (
    N_POS_BINS,
    POSITION_RANGE_CM,
    SIGMA_BINS_DEFAULT,
    position_soft_label,
    soft_target_cross_entropy,
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
    'n_pos_bins': N_POS_BINS,
    'position_range_cm': POSITION_RANGE_CM,
    'sigma_bins': SIGMA_BINS_DEFAULT,
}


class PoseBERT(nn.Module):
    """Masked-autoencoder transformer with a position-bin classification head."""

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

        # Layout: [2*n_parts coords | n_parts masks] => input_dim == 3 * n_parts.
        if input_dim % 3 != 0:
            raise ValueError(f"input_dim={input_dim} not divisible by 3 (expected 3*n_parts)")
        self.n_parts = input_dim // 3

        self.d_model = d_model
        self.config = config

        self.n_pos_bins = int(config['n_pos_bins'])
        self.position_range_cm = float(config['position_range_cm'])
        self.sigma_bins = float(config['sigma_bins'])
        self.n_classes = self.n_pos_bins ** 2

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

        # Per-frame logits for n_parts body parts × n_classes.
        self.reconstruction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, self.n_parts * self.n_classes),
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
        logits = logits.view(B, T, n_parts, self.n_classes)

        target_xy = x[:, :, :coord_dim].view(B, T, n_parts, 2)
        obs_mask = x[:, :, coord_dim:]

        target_probs = position_soft_label(
            target_xy,
            n_bins=self.n_pos_bins,
            arena_range_cm=self.position_range_cm,
            sigma_bins=self.sigma_bins,
        )

        valid_frame = (span_mask & ~padding_mask).float()
        weight = valid_frame.unsqueeze(-1) * obs_mask

        # soft_target_cross_entropy clamps weight.sum() to >= 1.0 internally;
        # skipping a num_valid guard avoids .item() breaking torch.compile.

        loss = soft_target_cross_entropy(
            logits=logits,
            target_probs=target_probs,
            weight=weight,
            reduction="mean",
        )

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
