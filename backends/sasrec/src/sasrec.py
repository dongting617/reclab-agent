"""A compact causal self-attention sequential recommendation baseline."""

from __future__ import annotations

import torch
from torch import nn


class SASRec(nn.Module):
    def __init__(
        self,
        num_items: int,
        max_length: int = 50,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_items = num_items
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.item_embedding = nn.Embedding(num_items + 1, hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(max_length, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(hidden_size)
        self.item_bias = nn.Parameter(torch.zeros(num_items + 1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()
        nn.init.normal_(self.position_embedding.weight, std=0.02)

    def encode(self, histories: torch.Tensor) -> torch.Tensor:
        if histories.ndim != 2 or histories.shape[1] != self.max_length:
            raise ValueError(f"histories must have shape [batch, {self.max_length}]")
        batch, length = histories.shape
        positions = torch.arange(length, device=histories.device).expand(batch, length)
        x = self.item_embedding(histories) + self.position_embedding(positions)
        x = self.input_dropout(self.input_norm(x))
        causal_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=histories.device),
            diagonal=1,
        )
        padding_mask = histories.eq(0)
        x = self.encoder(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        return self.output_norm(x[:, -1])

    def catalog_logits(self, user_states: torch.Tensor) -> torch.Tensor:
        catalog = self.item_embedding.weight[1:]
        return user_states @ catalog.t() + self.item_bias[1:]
