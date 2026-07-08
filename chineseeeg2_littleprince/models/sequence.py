from __future__ import annotations

import math

import torch
from torch import nn


def masked_adaptive_avg_pool1d(
    features: torch.Tensor,
    mask: torch.Tensor,
    output_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if features.ndim != 3:
        raise ValueError(f"Expected [batch, channels, time] features, got shape={tuple(features.shape)}")
    if mask.ndim != 2:
        raise ValueError(f"Expected [batch, time] mask, got shape={tuple(mask.shape)}")
    if output_size <= 0:
        raise ValueError(f"output_size must be positive, got {output_size}")

    batch, channels, n_samples = features.shape
    if mask.shape != (batch, n_samples):
        raise ValueError(f"Mask shape {tuple(mask.shape)} does not match features {(batch, n_samples)}")

    pooled = features.new_zeros(batch, channels, output_size)
    pooled_mask = torch.zeros(batch, output_size, dtype=torch.bool, device=features.device)
    mask_float = mask.to(features.dtype)
    for out_idx in range(output_size):
        start = int(math.floor(out_idx * n_samples / output_size))
        stop = int(math.ceil((out_idx + 1) * n_samples / output_size))
        start = min(start, n_samples - 1)
        stop = max(start + 1, min(stop, n_samples))
        weights = mask_float[:, start:stop].unsqueeze(1)
        denom = weights.sum(dim=-1).clamp_min(1.0)
        pooled[:, :, out_idx] = (features[:, :, start:stop] * weights).sum(dim=-1) / denom
        pooled_mask[:, out_idx] = mask[:, start:stop].any(dim=-1)
    return pooled, pooled_mask


class TemporalConvEEGSequenceEncoder(nn.Module):
    """Map EEG windows to ordered feature sequences."""

    def __init__(
        self,
        in_channels: int = 128,
        hidden_channels: int = 128,
        embedding_dim: int = 1024,
        sequence_frames: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        if sequence_frames <= 0:
            raise ValueError(f"sequence_frames must be positive, got {sequence_frames}")
        self.sequence_frames = sequence_frames
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
        self.projection = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, embedding_dim, kernel_size=1),
        )

    def forward(
        self,
        eeg: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_mask: bool = False,
    ):
        features = self.projection(self.encoder(eeg))
        if mask is None:
            pooled = torch.nn.functional.adaptive_avg_pool1d(features, self.sequence_frames)
            sequence_mask = torch.ones(
                eeg.shape[0],
                self.sequence_frames,
                dtype=torch.bool,
                device=eeg.device,
            )
        else:
            sequence_mask_input = mask.to(device=eeg.device, dtype=torch.bool)
            pooled, sequence_mask = masked_adaptive_avg_pool1d(
                features,
                sequence_mask_input,
                self.sequence_frames,
            )
        sequence = pooled.transpose(1, 2).contiguous()
        if return_mask:
            return sequence, sequence_mask
        return sequence
