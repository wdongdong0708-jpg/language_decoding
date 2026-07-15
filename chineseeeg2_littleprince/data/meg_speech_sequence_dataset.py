from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from chineseeeg2_littleprince.data.dataset import _normalize_per_channel
from chineseeeg2_littleprince.meg import KitMEGReader


@dataclass(frozen=True)
class MEGSpeechSequenceRecord:
    subject: str
    session: str
    task: str
    story: str
    segment_idx: int
    label_id: int
    start_time: float
    stop_time: float
    sfreq: float
    start_sample: int
    stop_sample: int
    n_samples: int
    meg_con_path: Path
    events_tsv_path: Path
    channels_tsv_path: Path
    speech_sequence_path: Path
    speech_sequence_idx: int
    speech_frame_start: int
    speech_frame_stop: int
    speech_frame_count: int
    speech_feature_dim: int
    audio_file_path: Path
    audio_start_time: float
    audio_stop_time: float
    audio_start_sample: int
    audio_stop_sample: int
    n_audio_samples: int
    meg_window_path: Path | None = None
    meg_window_idx: int | None = None
    meg_window_samples: int | None = None
    text: str = ""


def _record_from_row(row: dict[str, str]) -> MEGSpeechSequenceRecord:
    return MEGSpeechSequenceRecord(
        subject=row["subject"],
        session=row["session"],
        task=row["task"],
        story=row["story"],
        segment_idx=int(row["segment_idx"]),
        label_id=int(row["label_id"]),
        start_time=float(row["start_time"]),
        stop_time=float(row["stop_time"]),
        sfreq=float(row["sfreq"]),
        start_sample=int(row["start_sample"]),
        stop_sample=int(row["stop_sample"]),
        n_samples=int(row["n_samples"]),
        meg_con_path=Path(row["meg_con_path"]),
        events_tsv_path=Path(row["events_tsv_path"]),
        channels_tsv_path=Path(row["channels_tsv_path"]),
        speech_sequence_path=Path(row["speech_sequence_path"]),
        speech_sequence_idx=int(row["speech_sequence_idx"]),
        speech_frame_start=int(row["speech_frame_start"]),
        speech_frame_stop=int(row["speech_frame_stop"]),
        speech_frame_count=int(row["speech_frame_count"]),
        speech_feature_dim=int(row["speech_feature_dim"]),
        audio_file_path=Path(row["audio_file_path"]),
        audio_start_time=float(row["audio_start_time"]),
        audio_stop_time=float(row["audio_stop_time"]),
        audio_start_sample=int(row["audio_start_sample"]),
        audio_stop_sample=int(row["audio_stop_sample"]),
        n_audio_samples=int(row["n_audio_samples"]),
        text=row.get("text", ""),
        meg_window_path=Path(row["meg_window_path"]) if row.get("meg_window_path") else None,
        meg_window_idx=int(row["meg_window_idx"]) if row.get("meg_window_idx") not in (None, "") else None,
        meg_window_samples=int(row["meg_window_samples"]) if row.get("meg_window_samples") not in (None, "") else None,
    )


def load_meg_speech_sequence_manifest(path: str | Path) -> list[MEGSpeechSequenceRecord]:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
        records = [_record_from_row(row) for row in csv.DictReader(f)]
    if not records:
        raise ValueError(f"MEG speech sequence manifest is empty: {manifest_path}")
    return records


def validate_meg_speech_sequence_manifest(records: list[MEGSpeechSequenceRecord]) -> None:
    for record in records:
        if record.stop_sample <= record.start_sample:
            raise ValueError(f"Invalid MEG window in segment {record.segment_idx}")
        if record.n_samples != record.stop_sample - record.start_sample:
            raise ValueError(f"n_samples mismatch in segment {record.segment_idx}")
        if record.speech_frame_stop <= record.speech_frame_start:
            raise ValueError(f"Invalid speech frame window in segment {record.segment_idx}")
        if record.speech_frame_count != record.speech_frame_stop - record.speech_frame_start:
            raise ValueError(f"speech_frame_count mismatch in segment {record.segment_idx}")
        if record.audio_stop_sample <= record.audio_start_sample:
            raise ValueError(f"Invalid audio window in segment {record.segment_idx}")
        paths = [
            record.meg_con_path,
            record.events_tsv_path,
            record.channels_tsv_path,
            record.speech_sequence_path,
            record.audio_file_path,
        ]
        if record.meg_window_path is not None:
            paths.append(record.meg_window_path)
            if record.meg_window_idx is None:
                raise ValueError(f"Missing meg_window_idx in segment {record.segment_idx}")
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(path)


class MEGSpeechSequenceDataset(Dataset):
    """MEG windows paired with speech feature sequences."""

    def __init__(
        self,
        manifest_path: str | Path,
        normalize_meg: bool = True,
        validate: bool = True,
        cache_readers: bool = True,
    ):
        self.manifest_path = Path(manifest_path)
        self.records = load_meg_speech_sequence_manifest(self.manifest_path)
        if validate:
            validate_meg_speech_sequence_manifest(self.records)

        self.normalize_meg = normalize_meg
        self.cache_readers = cache_readers
        self._reader_cache: dict[Path, KitMEGReader] = {}
        self._sequence_cache: dict[Path, np.ndarray] = {}
        self._meg_window_cache: dict[Path, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _reader(self, path: Path) -> KitMEGReader:
        if not self.cache_readers:
            return KitMEGReader(path)
        if path not in self._reader_cache:
            self._reader_cache[path] = KitMEGReader(path)
        return self._reader_cache[path]

    def _sequences(self, path: Path) -> np.ndarray:
        if path not in self._sequence_cache:
            self._sequence_cache[path] = np.load(path, mmap_mode="r")
        return self._sequence_cache[path]

    def _meg_windows(self, path: Path) -> np.ndarray:
        if path not in self._meg_window_cache:
            self._meg_window_cache[path] = np.load(path, mmap_mode="r")
        return self._meg_window_cache[path]

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        if record.meg_window_path is not None and record.meg_window_idx is not None:
            meg = np.array(
                self._meg_windows(record.meg_window_path)[record.meg_window_idx],
                dtype=np.float32,
                copy=True,
            )
        else:
            meg = self._reader(record.meg_con_path).read_window(record.start_sample, record.stop_sample)
        if self.normalize_meg:
            meg = _normalize_per_channel(meg)

        sequence_array = self._sequences(record.speech_sequence_path)
        speech = np.array(
            sequence_array[record.speech_frame_start : record.speech_frame_stop],
            dtype=np.float32,
            copy=True,
        )

        return {
            "eeg": torch.from_numpy(np.asarray(meg, dtype=np.float32)),
            "speech": torch.from_numpy(speech),
            "length": torch.tensor(meg.shape[1], dtype=torch.long),
            "speech_length": torch.tensor(speech.shape[0], dtype=torch.long),
            "text_embedding_idx": torch.tensor(record.speech_sequence_idx, dtype=torch.long),
            "label_id": torch.tensor(record.label_id, dtype=torch.long),
            "meta": {
                "subject": record.subject,
                "session": record.session,
                "task": record.task,
                "story": record.story,
                "segment_idx": record.segment_idx,
                "label_id": record.label_id,
                "text_embedding_idx": record.speech_sequence_idx,
                "speech_sequence_idx": record.speech_sequence_idx,
                "speech_frame_start": record.speech_frame_start,
                "speech_frame_stop": record.speech_frame_stop,
                "audio_file_path": str(record.audio_file_path),
                "audio_start_time": record.audio_start_time,
                "audio_stop_time": record.audio_stop_time,
                "meg_window_path": str(record.meg_window_path) if record.meg_window_path is not None else "",
                "meg_window_idx": record.meg_window_idx if record.meg_window_idx is not None else -1,
                "text": record.text,
            },
        }
