from __future__ import annotations

import torch
from torch import nn


class TemporalConvEEGEncoder(nn.Module):
    """Small baseline mapping EEG windows to text-aligned embedding vectors."""

    def __init__(
        self,
        in_channels: int = 128,
        hidden_channels: int = 128,
        embedding_dim: int = 768,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=9, padding=4),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, embedding_dim),
        )

    def forward(self, eeg: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        features = self.encoder(eeg)
        if mask is None:
            pooled = features.mean(dim=-1)
        else:
            weights = mask.to(features.dtype).unsqueeze(1)
            pooled = (features * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)
        return self.head(pooled)


# Backward-compatible name for older imports/configs.
TemporalConvRegressor = TemporalConvEEGEncoder
