"""Transformer-encoder sequence classifier for driver fatigue state.

A lightweight encoder over the signal window with learned positional embeddings
and mean-pooling. Self-attention lets it weight the informative parts of the
window (e.g. a run of high PERCLOS) rather than just the final step.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..config import settings
from ..signals import N_CLASSES, N_FEATURES


class TransformerClassifier(nn.Module):
    def __init__(self, n_features: int = N_FEATURES, d_model: int = 64,
                 nhead: int = 4, layers: int = 2, n_classes: int = N_CLASSES,
                 dropout: float = 0.2, max_len: int | None = None):
        super().__init__()
        max_len = max_len or settings.window_len
        self.input = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=128, dropout=dropout,
            batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, T, F]
        h = self.input(x) + self.pos[:, : x.size(1)]
        h = self.encoder(h)
        return self.head(self.norm(h.mean(dim=1)))       # mean-pool over time
