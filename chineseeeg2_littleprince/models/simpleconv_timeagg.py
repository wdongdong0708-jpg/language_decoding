from __future__ import annotations

import torch
from torch import nn

from .baseline import SubjectSpecificLinear


class DilatedResidualBlock(nn.Module):
    """Same-length dilated convolution followed by BN, GELU, and a residual add."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd to preserve the temporal length")
        padding = kernel_size // 2 * dilation
        layers: list[nn.Module] = [
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.layers = nn.Sequential(*layers)
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)


class TemporalGLU(nn.Module):
    """Meta-style contextual GLU that preserves channels and temporal length."""

    def __init__(self, channels: int, context: int = 1):
        super().__init__()
        kernel_size = 1 + 2 * context
        self.layers = nn.Sequential(
            nn.Conv1d(
                channels,
                2 * channels,
                kernel_size=kernel_size,
                padding=context,
            ),
            nn.GLU(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class BahdanauAttentionPooling(nn.Module):
    """Masked additive attention over time, matching Meta's query-free use."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.key_projection = nn.Linear(input_size, hidden_size)
        self.score_projection = nn.Linear(hidden_size, 1)

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 3:
            raise ValueError(
                f"Expected features with shape [batch, channels, time], got {features.shape}"
            )
        keys = features.transpose(1, 2)
        scores = self.score_projection(torch.tanh(self.key_projection(keys))).squeeze(-1)

        if mask is not None:
            if mask.shape != scores.shape:
                raise ValueError(f"mask must have shape {tuple(scores.shape)}, got {tuple(mask.shape)}")
            mask = mask.to(device=scores.device, dtype=torch.bool)
            if not torch.all(mask.any(dim=1)):
                raise ValueError("Every sample must contain at least one valid time point")
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        attention = torch.softmax(scores, dim=-1)
        pooled = torch.bmm(attention.unsqueeze(1), keys).squeeze(1)
        return pooled, attention


class SimpleConvTimeAggEEGEncoder(nn.Module):
    """Compact SimpleConvTimeAgg encoder for variable-length EEG windows."""

    def __init__(
        self,
        in_channels: int = 128,
        hidden_channels: int = 128,
        embedding_dim: int = 768,
        depth: int = 5,
        kernel_size: int = 3,
        dilation_growth: int = 2,
        input_dropout: float = 0.1,
        conv_dropout: float = 0.0,
        dropout: float = 0.2,
        glu_every: int = 2,
        glu_context: int = 1,
        attention_hidden_channels: int = 128,
        subject_layers: bool = True,
        n_subjects: int | None = None,
        subject_layers_id: bool = True,
    ):
        super().__init__()
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        if dilation_growth <= 0:
            raise ValueError(f"dilation_growth must be positive, got {dilation_growth}")
        if glu_every < 0:
            raise ValueError(f"glu_every must be non-negative, got {glu_every}")

        self.initial_projection = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        self.subject_layers = None
        if subject_layers:
            if n_subjects is None:
                raise ValueError("n_subjects is required when subject_layers=True")
            self.subject_layers = SubjectSpecificLinear(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                n_subjects=n_subjects,
                init_identity=subject_layers_id,
            )

        self.input_dropout = nn.Dropout(input_dropout)
        self.dilations = tuple(dilation_growth**index for index in range(depth))
        self.blocks = nn.ModuleList(
            [
                DilatedResidualBlock(
                    channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=conv_dropout,
                )
                for dilation in self.dilations
            ]
        )
        self.glus = nn.ModuleDict()
        if glu_every:
            for layer_index in range(glu_every, depth + 1, glu_every):
                self.glus[str(layer_index)] = TemporalGLU(
                    channels=hidden_channels,
                    context=glu_context,
                )

        self.time_aggregation = BahdanauAttentionPooling(
            input_size=hidden_channels,
            hidden_size=attention_hidden_channels,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, embedding_dim),
        )

    def forward(
        self,
        eeg: torch.Tensor,
        mask: torch.Tensor | None = None,
        subject_id: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        valid = None
        if mask is not None:
            if mask.shape != (eeg.shape[0], eeg.shape[-1]):
                raise ValueError(
                    f"mask must have shape {(eeg.shape[0], eeg.shape[-1])}, "
                    f"got {tuple(mask.shape)}"
                )
            valid = mask.to(device=eeg.device, dtype=eeg.dtype).unsqueeze(1)

        x = self.initial_projection(eeg)
        if valid is not None:
            x = x * valid
        if self.subject_layers is not None:
            if subject_id is None:
                raise ValueError("subject_id is required when subject_layers are enabled")
            x = self.subject_layers(x, subject_id)

        x = self.input_dropout(x)
        for layer_index, block in enumerate(self.blocks, start=1):
            x = block(x)
            if valid is not None:
                x = x * valid
            glu = self.glus[str(layer_index)] if str(layer_index) in self.glus else None
            if glu is not None:
                x = glu(x)
                if valid is not None:
                    x = x * valid

        pooled, attention = self.time_aggregation(x, mask)
        output = self.head(pooled)
        if return_attention:
            return output, attention
        return output
