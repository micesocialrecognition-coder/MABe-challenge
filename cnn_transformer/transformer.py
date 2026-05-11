"""Transformer encoder block with learned relative-position attention bias.
Example: from transformer import Encoder
"""
import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, num_heads, d_model, max_seq_len, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.d_k = d_model // num_heads

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.scale = self.d_k ** -0.5

        # Distance ranges from -(max_seq_len-1) to +(max_seq_len-1) -> 2*max_seq_len-1 entries.
        self.rel_pos_bias = nn.Embedding(2 * max_seq_len - 1, num_heads)

        # Orthogonal init maximizes starting diversity across heads.
        nn.init.orthogonal_(self.Wq.weight)
        nn.init.orthogonal_(self.Wk.weight)
        nn.init.orthogonal_(self.Wv.weight)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor=None):

        B, T, _ = x.shape
        Q = self.Wq(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.Wk(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = self.Wv(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        scores = (Q @ K.transpose(-2,-1)) * self.scale

        positions = torch.arange(T, device=x.device)
        relative_dist = positions.unsqueeze(0) - positions.unsqueeze(1)
        relative_dist = relative_dist + self.max_seq_len - 1
        bias = self.rel_pos_bias(relative_dist)
        scores = scores + bias.permute(2, 0, 1).unsqueeze(0)

        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        output = weights @ V
        output = output.transpose(1,2).reshape(B, T, -1)
        output = self.output_projection(output)
        return output


class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout, max_seq_len):
        super().__init__()
        self.attention = Attention(num_heads, d_model, max_seq_len, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor=None):
        residual = x
        norm_x = self.norm1(x)
        attention = self.attention(norm_x, padding_mask)
        x = self.dropout(attention) + residual

        residual = x
        norm_x = self.norm2(x)
        feed_forward = self.feed_forward(norm_x)
        out = self.dropout(feed_forward) + residual
        return out
