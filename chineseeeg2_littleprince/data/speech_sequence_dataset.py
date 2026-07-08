from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from chineseeeg2_littleprince.data.dataset import _normalize_per_channel
from chineseeeg2_littleprince.data.manifest import ManifestRecord, load_manifest, validate_manifest
from chineseeeg2_littleprince.io.brainvision import BrainVisionReader


@dataclass(frozen=True)
class SpeechSequenceManifestRecord:
    eeg: ManifestRecord
    speech_sequence_path: Path
    speech_sequence_idx: int
    speech_frame_start: int
    speech_frame_stop: int
    speech_frame_count: int
    speech_feature_dim: int
    speaker_id: str
    audio_event_idx: int
    audio_file_path: Path
    audio_start_time: float
    audio_stop_time: float
    audio_start_sample: int
    audio_stop_sample: int
    n_audio_samples: int
    text: str = ""

    @property
    def label_id(self) -> int:
        return self.eeg.label_id


def _record_from_row(eeg: ManifestRecord, row: dict[str, str]) -> SpeechSequenceManifestRecord:
    return SpeechSequenceManifestRecord(
        eeg=eeg,
        speech_sequence_path=Path(row["speech_sequence_path"]),
        speech_sequence_idx=int(row["speech_sequence_idx"]),
        speech_frame_start=int(row["speech_frame_start"]),
        speech_frame_stop=int(row["speech_frame_stop"]),
        speech_frame_count=int(row["speech_frame_count"]),
        speech_feature_dim=int(row["speech_feature_dim"]),
        speaker_id=row["speaker_id"],
        audio_event_idx=int(row["audio_event_idx"]),
        audio_file_path=Path(row["audio_file_path"]),
        audio_start_time=float(row["audio_start_time"]),
        audio_stop_time=float(row["audio_stop_time"]),
        audio_start_sample=int(row["audio_start_sample"]),
        audio_stop_sample=int(row["audio_stop_sample"]),
        n_audio_samples=int(row["n_audio_samples"]),
        text=row.get("text", ""),
    )


def load_speech_sequence_manifest(path: str | Path) -> list[SpeechSequenceManifestRecord]:
    manifest_path = Path(path)
    eeg_records = load_manifest(manifest_path)
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != len(eeg_records):
        raise ValueError(f"Speech sequence manifest row mismatch in {manifest_path}")
    return [_record_from_row(eeg, row) for eeg, row in zip(eeg_records, rows)]


def validate_speech_sequence_manifest(records: list[SpeechSequenceManifestRecord]) -> None:
    validate_manifest([record.eeg for record in records])
    for record in records:
        if not record.speech_sequence_path.exists():
            raise FileNotFoundError(record.speech_sequence_path)
        if not record.audio_file_path.exists():
            raise FileNotFoundError(record.audio_file_path)
        if record.speech_frame_stop <= record.speech_frame_start:
            raise ValueError(f"Invalid speech frame window in record {record.eeg.global_row_idx}")
        if record.speech_frame_count != record.speech_frame_stop - record.speech_frame_start:
            raise ValueError(f"speech_frame_count mismatch in record {record.eeg.global_row_idx}")
        if record.speech_feature_dim <= 0:
            raise ValueError(f"Invalid speech_feature_dim in record {record.eeg.global_row_idx}")
        if record.audio_stop_sample <= record.audio_start_sample:
            raise ValueError(f"Invalid audio window in record {record.eeg.global_row_idx}")


class EEGSpeechSequenceDataset(Dataset):
    """Line-level EEG to sentence-level speech feature sequences."""

    def __init__(
        self,
        manifest_path: str | Path,
        normalize_eeg: bool = True,
        validate: bool = True,
        cache_readers: bool = True,
    ):
        self.manifest_path = Path(manifest_path)
        self.records = load_speech_sequence_manifest(self.manifest_path)
        if validate:
            validate_speech_sequence_manifest(self.records)

        self.normalize_eeg = normalize_eeg
        self.cache_readers = cache_readers
        self._reader_cache: dict[Path, BrainVisionReader] = {}
        self._sequence_cache: dict[Path, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _reader(self, path: Path) -> BrainVisionReader:
        if not self.cache_readers:
            return BrainVisionReader(path)
        if path not in self._reader_cache:
            self._reader_cache[path] = BrainVisionReader(path)
        return self._reader_cache[path]

    def _sequences(self, path: Path) -> np.ndarray:
        if path not in self._sequence_cache:
            self._sequence_cache[path] = np.load(path, mmap_mode="r")
        return self._sequence_cache[path]

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        eeg_record = record.eeg
        eeg = self._reader(eeg_record.eeg_vhdr_path).read_window(
            eeg_record.start_sample,
            eeg_record.stop_sample,
        )
        if self.normalize_eeg:
            eeg = _normalize_per_channel(eeg)

        sequence_array = self._sequences(record.speech_sequence_path)
        speech = np.array(
            sequence_array[record.speech_frame_start : record.speech_frame_stop],
            dtype=np.float32,
            copy=True,
        )

        return {
            "eeg": torch.from_numpy(np.asarray(eeg, dtype=np.float32)),
            "speech": torch.from_numpy(speech),
            "length": torch.tensor(eeg.shape[1], dtype=torch.long),
            "speech_length": torch.tensor(speech.shape[0], dtype=torch.long),
            "text_embedding_idx": torch.tensor(eeg_record.text_embedding_idx, dtype=torch.long),
            "label_id": torch.tensor(eeg_record.label_id, dtype=torch.long),
            "meta": {
                "subject": eeg_record.subject,
                "run": eeg_record.run,
                "local_row_idx": eeg_record.local_row_idx,
                "global_row_idx": eeg_record.global_row_idx,
                "text_embedding_idx": eeg_record.text_embedding_idx,
                "label_id": eeg_record.label_id,
                "speaker_id": record.speaker_id,
                "speech_sequence_idx": record.speech_sequence_idx,
                "speech_frame_start": record.speech_frame_start,
                "speech_frame_stop": record.speech_frame_stop,
                "audio_event_idx": record.audio_event_idx,
                "audio_file_path": str(record.audio_file_path),
                "audio_start_time": record.audio_start_time,
                "audio_stop_time": record.audio_stop_time,
                "text": record.text,
            },
        }
