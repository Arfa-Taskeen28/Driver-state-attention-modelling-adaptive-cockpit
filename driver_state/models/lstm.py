"""LSTM sequence classifier (baseline) for driver fatigue state."""
from __future__ import annotations

import torch
import torch.nn as nn

from ..signals import N_CLASSES, N_FEATURES


class LSTMClassifier(nn.Module):
    def __init__(self, n_features: int = N_FEATURES, hidden: int = 64,
                 layers: int = 1, n_classes: int = N_CLASSES, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, T, F]
        out, _ = self.lstm(x)
        return self.head(out[:, -1])                     # state at window end
