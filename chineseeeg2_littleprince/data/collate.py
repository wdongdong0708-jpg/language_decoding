from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def _fit_time(eeg: torch.Tensor, target_samples: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    channels, n_samples = eeg.shape
    valid = min(n_samples, target_samples)
    if n_samples > target_samples:
        eeg = eeg[:, :target_samples]
    elif n_samples < target_samples:
        eeg = F.pad(eeg, (0, target_samples - n_samples))
    mask = torch.zeros(target_samples, dtype=torch.bool)
    mask[:valid] = True
    return eeg, mask, valid


def collate_eeg_text(batch: list[dict[str, Any]], max_samples: int | None = None) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty batch")

    target_samples = max_samples or max(int(item["length"]) for item in batch)
    eeg_tensors = []
    masks = []
    lengths = []
    labels = []
    text_embedding_indices = []
    label_ids = []
    target_ids = []
    subject_ids = []
    metas = []

    for item in batch:
        eeg, mask, valid = _fit_time(item["eeg"], target_samples)
        eeg_tensors.append(eeg)
        masks.append(mask)
        lengths.append(valid)
        labels.append(item["label"])
        text_embedding_indices.append(item["text_embedding_idx"])
        label_ids.append(item.get("label_id", item["text_embedding_idx"]))
        target_ids.append(
            item.get("target_id", item.get("label_id", item["text_embedding_idx"]))
        )
        subject_ids.append(item["subject_id"])
        metas.append(item["meta"])

    return {
        "eeg": torch.stack(eeg_tensors, dim=0),
        "label": torch.stack(labels, dim=0),
        "mask": torch.stack(masks, dim=0),
        "length": torch.tensor(lengths, dtype=torch.long),
        "text_embedding_idx": torch.stack(text_embedding_indices, dim=0),
        "label_id": torch.stack(label_ids, dim=0),
        "target_id": torch.stack(target_ids, dim=0),
        "subject_id": torch.stack(subject_ids, dim=0),
        "meta": metas,
    }
