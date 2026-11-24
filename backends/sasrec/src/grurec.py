"""Small GRU next-item model for checking the existing adapter's portability."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class GRURec(nn.Module):
    def __init__(self, num_items: int, max_length: int, hidden_size: int = 64,
                 num_layers: int = 1, dropout: float = 0.4):
        super().__init__()
        self.max_length = max_length
        self.item_embedding = nn.Embedding(num_items + 1, hidden_size, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.GRU(hidden_size, hidden_size, num_layers=num_layers,
                              batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.output_norm = nn.LayerNorm(hidden_size)
        self.item_bias = nn.Parameter(torch.zeros(num_items + 1))

    def encode(self, histories: torch.Tensor) -> torch.Tensor:
        lengths = histories.ne(0).sum(dim=1)
        if bool((lengths == 0).any()):
            raise ValueError("GRU histories must contain an observed item")
        # The common runtime pads on the left; packed GRU sequences pad on the right.
        positions = (torch.arange(histories.shape[1], device=histories.device)[None, :]
                     + histories.shape[1] - lengths[:, None]) % histories.shape[1]
        compact = histories.gather(1, positions)
        packed = pack_padded_sequence(self.dropout(self.item_embedding(compact)),
                                      lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.encoder(packed)
        return self.output_norm(hidden[-1])

    def catalog_logits(self, user_states: torch.Tensor) -> torch.Tensor:
        return user_states @ self.item_embedding.weight[1:].t() + self.item_bias[1:]
