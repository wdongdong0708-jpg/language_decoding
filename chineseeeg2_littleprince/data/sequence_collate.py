from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from chineseeeg2_littleprince.data.collate import _fit_time


def _fit_feature_sequence(sequence: torch.Tensor, target_frames: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    if sequence.ndim != 2:
        raise ValueError(f"Expected [frames, features] speech sequence, got shape={tuple(sequence.shape)}")
    n_frames, feature_dim = sequence.shape
    if n_frames <= 0:
        raise ValueError("Cannot collate an empty speech sequence")

    resized = F.interpolate(
        sequence.T.unsqueeze(0),
        size=target_frames,
        mode="linear",
        align_corners=False,
    ).squeeze(0).T
    mask = torch.ones(target_frames, dtype=torch.bool)
    return resized.contiguous(), mask, min(n_frames, target_frames)


def _pad_feature_sequence(sequence: torch.Tensor, target_frames: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    if sequence.ndim != 2:
        raise ValueError(f"Expected [frames, features] speech sequence, got shape={tuple(sequence.shape)}")
    n_frames, _ = sequence.shape
    if n_frames <= 0:
        raise ValueError("Cannot collate an empty speech sequence")
    if n_frames > target_frames:
        raise ValueError(f"target_frames={target_frames} is shorter than sequence length {n_frames}")

    padded = F.pad(sequence, (0, 0, 0, target_frames - n_frames))
    mask = torch.zeros(target_frames, dtype=torch.bool)
    mask[:n_frames] = True
    return padded.contiguous(), mask, n_frames


def collate_eeg_speech_sequence(
    batch: list[dict[str, Any]],
    max_samples: int | None = None,
    sequence_frames: int | None = 64,
) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    if sequence_frames is not None and sequence_frames <= 0:
        raise ValueError(f"sequence_frames must be positive, got {sequence_frames}")

    target_samples = max_samples or max(int(item["length"]) for item in batch)
    target_speech_frames = sequence_frames or max(int(item["speech_length"]) for item in batch)
    eeg_tensors = []
    eeg_masks = []
    eeg_lengths = []
    speech_tensors = []
    speech_masks = []
    speech_lengths = []
    text_embedding_indices = []
    label_ids = []
    metas = []

    for item in batch:
        eeg, eeg_mask, eeg_valid = _fit_time(item["eeg"], target_samples)
        if sequence_frames is None:
            speech, speech_mask, speech_valid = _pad_feature_sequence(item["speech"], target_speech_frames)
        else:
            speech, speech_mask, speech_valid = _fit_feature_sequence(item["speech"], target_speech_frames)
        eeg_tensors.append(eeg)
        eeg_masks.append(eeg_mask)
        eeg_lengths.append(eeg_valid)
        speech_tensors.append(speech)
        speech_masks.append(speech_mask)
        speech_lengths.append(speech_valid)
        text_embedding_indices.append(item["text_embedding_idx"])
        label_ids.append(item["label_id"])
        metas.append(item["meta"])

    return {
        "eeg": torch.stack(eeg_tensors, dim=0),
        "eeg_mask": torch.stack(eeg_masks, dim=0),
        "mask": torch.stack(eeg_masks, dim=0),
        "eeg_length": torch.tensor(eeg_lengths, dtype=torch.long),
        "length": torch.tensor(eeg_lengths, dtype=torch.long),
        "speech": torch.stack(speech_tensors, dim=0),
        "speech_mask": torch.stack(speech_masks, dim=0),
        "speech_length": torch.tensor(speech_lengths, dtype=torch.long),
        "text_embedding_idx": torch.stack(text_embedding_indices, dim=0),
        "label_id": torch.stack(label_ids, dim=0),
        "meta": metas,
    }
