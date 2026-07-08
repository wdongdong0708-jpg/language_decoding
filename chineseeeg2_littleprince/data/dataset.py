from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from chineseeeg2_littleprince.data.manifest import ManifestRecord, load_manifest, validate_manifest
from chineseeeg2_littleprince.io.brainvision import BrainVisionReader


def _normalize_per_channel(eeg: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = eeg.mean(axis=1, keepdims=True)
    std = eeg.std(axis=1, keepdims=True)
    return (eeg - mean) / np.maximum(std, eps)


class EEGTextDataset(Dataset):
    """Line-level EEG-text samples from a manifest CSV."""

    def __init__(
        self,
        manifest_path: str | Path,
        normalize_eeg: bool = True,
        validate: bool = True,
        cache_readers: bool = True,
    ):
        self.manifest_path = Path(manifest_path)
        self.records = load_manifest(self.manifest_path)
        if validate:
            validate_manifest(self.records)

        self.normalize_eeg = normalize_eeg
        self.cache_readers = cache_readers
        self._reader_cache: dict[Path, BrainVisionReader] = {}
        self._embedding_cache: dict[Path, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _reader(self, path: Path) -> BrainVisionReader:
        if not self.cache_readers:
            return BrainVisionReader(path)
        if path not in self._reader_cache:
            self._reader_cache[path] = BrainVisionReader(path)
        return self._reader_cache[path]

    def _embeddings(self, path: Path) -> np.ndarray:
        if path not in self._embedding_cache:
            self._embedding_cache[path] = np.load(path, mmap_mode="r")
        return self._embedding_cache[path]

    def __getitem__(self, index: int) -> dict[str, Any]:
        record: ManifestRecord = self.records[index]
        eeg = self._reader(record.eeg_vhdr_path).read_window(record.start_sample, record.stop_sample)
        if self.normalize_eeg:
            eeg = _normalize_per_channel(eeg)

        label = np.array(
            self._embeddings(record.text_embedding_path)[record.text_embedding_idx],
            dtype=np.float32,
            copy=True,
        )

        return {
            "eeg": torch.from_numpy(np.asarray(eeg, dtype=np.float32)),
            "label": torch.from_numpy(label),
            "length": torch.tensor(eeg.shape[1], dtype=torch.long),
            "text_embedding_idx": torch.tensor(record.text_embedding_idx, dtype=torch.long),
            "label_id": torch.tensor(record.label_id, dtype=torch.long),
            "meta": {
                "subject": record.subject,
                "run": record.run,
                "local_row_idx": record.local_row_idx,
                "global_row_idx": record.global_row_idx,
                "text_embedding_idx": record.text_embedding_idx,
                "label_id": record.label_id,
            },
        }
