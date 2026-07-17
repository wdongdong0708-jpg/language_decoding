from __future__ import annotations

import torch
from torch import nn


class SubjectSpecificLinear(nn.Module):
    """Meta-style per-subject linear projection over EEG channels."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_subjects: int,
        init_identity: bool = False,
    ):
        super().__init__()
        if n_subjects <= 0:
            raise ValueError(f"n_subjects must be positive, got {n_subjects}")
        if init_identity and in_channels != out_channels:
            raise ValueError("Identity initialization requires equal input and output channels")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_subjects = n_subjects
        self.weights = nn.Parameter(torch.empty(n_subjects, in_channels, out_channels))
        if init_identity:
            self.weights.data.copy_(torch.eye(in_channels)[None].expand(n_subjects, -1, -1))
        else:
            self.weights.data.normal_()
            self.weights.data *= 1 / in_channels**0.5

    def forward(self, eeg: torch.Tensor, subject_id: torch.Tensor) -> torch.Tensor:
        if eeg.ndim != 3:
            raise ValueError(f"Expected EEG with shape [batch, channels, time], got {eeg.shape}")
        if subject_id.ndim != 1 or subject_id.shape[0] != eeg.shape[0]:
            raise ValueError(
                f"subject_id must have shape [{eeg.shape[0]}], got {tuple(subject_id.shape)}"
            )
        if subject_id.dtype != torch.long:
            raise TypeError(f"subject_id must use torch.long, got {subject_id.dtype}")
        if subject_id.numel() and (
            int(subject_id.min()) < 0 or int(subject_id.max()) >= self.n_subjects
        ):
            raise IndexError(
                f"subject_id values must be in [0, {self.n_subjects - 1}], "
                f"got [{int(subject_id.min())}, {int(subject_id.max())}]"
            )

        weights = self.weights.index_select(0, subject_id)
        return torch.einsum("bct,bcd->bdt", eeg, weights)


class TemporalConvEEGEncoder(nn.Module):
    """Small baseline mapping EEG and text into a shared embedding space."""

    def __init__(
        self,
        in_channels: int = 128,
        hidden_channels: int = 128,
        embedding_dim: int = 768,
        dropout: float = 0.1,
        subject_layers: bool = False,
        n_subjects: int | None = None,
        subject_layers_id: bool = False,
        text_embedding_dim: int | None = None,
        text_projection_hidden_dim: int | None = None,
    ):
        super().__init__()
        self.subject_layers = None
        if subject_layers:
            if n_subjects is None:
                raise ValueError("n_subjects is required when subject_layers=True")
            self.subject_layers = SubjectSpecificLinear(
                in_channels=in_channels,
                out_channels=in_channels,
                n_subjects=n_subjects,
                init_identity=subject_layers_id,
            )
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
        if text_embedding_dim is None:
            text_embedding_dim = embedding_dim
        if text_projection_hidden_dim is None:
            text_projection_hidden_dim = embedding_dim
        if text_embedding_dim <= 0:
            raise ValueError(
                f"text_embedding_dim must be positive, got {text_embedding_dim}"
            )
        if text_projection_hidden_dim <= 0:
            raise ValueError(
                "text_projection_hidden_dim must be positive, "
                f"got {text_projection_hidden_dim}"
            )
        self.text_embedding_dim = text_embedding_dim
        self.text_projection_hidden_dim = text_projection_hidden_dim
        self.text_projection = nn.Sequential(
            nn.Linear(text_embedding_dim, text_projection_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(text_projection_hidden_dim, embedding_dim),
        )

    def project_text_embedding(self, text_embedding: torch.Tensor) -> torch.Tensor:
        """Project frozen text features into the EEG-text similarity space."""

        return self.text_projection(text_embedding)

    def forward(
        self,
        eeg: torch.Tensor,
        mask: torch.Tensor | None = None,
        subject_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.subject_layers is not None:
            if subject_id is None:
                raise ValueError("subject_id is required when subject_layers are enabled")
            eeg = self.subject_layers(eeg, subject_id)
        features = self.encoder(eeg)
        if mask is None:
            pooled = features.mean(dim=-1)
        else:
            weights = mask.to(features.dtype).unsqueeze(1)
            pooled = (features * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)
        return self.head(pooled)


# Backward-compatible name for older imports/configs.
TemporalConvRegressor = TemporalConvEEGEncoder
