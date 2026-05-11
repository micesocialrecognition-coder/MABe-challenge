"""PoseBERT: masked-autoencoder transformer for mouse pose SSL pretraining.

Usage: model = PoseBERT(POSEBERT_CONFIG); out = model(x, meta_ids, padding_mask, span_mask)
"""

import math

import torch
import torch.nn as nn


POSEBERT_CONFIG = {
    'input_dim': 33,         # 11 parts × 2 (x,y) + 11 observation masks
    'output_dim': 22,        # 11 parts × 2 (x,y) reconstruction target
    'd_model': 256,
    'nhead': 8,
    'num_layers': 6,
    'dim_feedforward': 1024,
    'dropout': 0.1,
    # [lab, strain, arena_type, sex, age]; idx 0 = UNK. train.py overrides
    # with live vocab sizes before constructing the model.
    'meta_vocab_sizes': [22, 10, 6, 3, 13],
    'window_size': 128,
    'mask_ratio': 0.15,
    'min_span': 5,
    'max_span': 30,
}


class PoseBERT(nn.Module):
    """Masked-autoencoder transformer that reconstructs (x,y) at masked frames."""

    def __init__(self, config: dict):
        super().__init__()

        input_dim = config['input_dim']
        output_dim = config['output_dim']
        d_model = config['d_model']
        nhead = config['nhead']
        num_layers = config['num_layers']
        dim_feedforward = config['dim_feedforward']
        dropout = config['dropout']
        meta_vocab_sizes = config['meta_vocab_sizes']
        window_size = config['window_size']

        self.d_model = d_model
        self.output_dim = output_dim
        self.config = config

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

        self.reconstruction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, output_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        meta_ids: torch.Tensor,
        padding_mask: torch.Tensor,
        span_mask: torch.Tensor,
    ) -> dict:
        """Mask, encode, reconstruct, compute observation-weighted MSE at masked frames."""
        B, T, _ = x.shape

        h = self.input_projection(x)

        mask_expanded = span_mask.unsqueeze(-1).expand_as(h)
        h = torch.where(mask_expanded, self.mask_token.expand(B, T, -1), h)

        h = h + self.pos_encoding[:T]
        for i, emb in enumerate(self.meta_embeddings):
            h = h + emb(meta_ids[:, i]).unsqueeze(1)

        h = self.transformer_encoder(h, src_key_padding_mask=padding_mask)

        predictions = self.reconstruction_head(h)

        target = x[:, :, :22]

        obs_mask_11 = x[:, :, 22:]

        # Each body-part mask covers both its x and y coord.
        obs_mask_22 = obs_mask_11.repeat_interleave(2, dim=-1)

        valid_mask = span_mask & ~padding_mask
        num_masked = valid_mask.sum().item()

        if num_masked == 0:
            return {
                'loss': torch.tensor(0.0, device=x.device, requires_grad=True),
                'predictions': predictions,
                'num_masked': 0,
            }

        sq_error = (predictions - target) ** 2

        sq_error = sq_error * obs_mask_22

        valid_expanded = valid_mask.unsqueeze(-1).expand_as(sq_error)

        masked_sq_error = sq_error * valid_expanded.float()

        # Average only over actually-observed coord slots at masked positions.
        num_observed = (obs_mask_22 * valid_expanded.float()).sum().clamp(min=1.0)

        loss = masked_sq_error.sum() / num_observed

        return {
            'loss': loss,
            'predictions': predictions,
            'num_masked': num_masked,
        }

    def extract(
        self,
        x: torch.Tensor,
        meta_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return encoder hidden states [B, T, d_model] for downstream fine-tuning."""
        B, T, _ = x.shape

        h = self.input_projection(x)

        h = h + self.pos_encoding[:T]
        for i, emb in enumerate(self.meta_embeddings):
            h = h + emb(meta_ids[:, i]).unsqueeze(1)

        h = self.transformer_encoder(h, src_key_padding_mask=padding_mask)

        return h


def count_parameters(model: nn.Module) -> int:
    """Print and return total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total:,} total, {trainable:,} trainable")
    return trainable
