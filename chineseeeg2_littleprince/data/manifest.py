from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ManifestRecord:
    subject: str
    session: str
    task: str
    run: int
    local_row_idx: int
    global_row_idx: int
    text_embedding_idx: int
    start_time: float
    stop_time: float
    sfreq: float
    start_sample: int
    stop_sample: int
    n_samples: int
    eeg_vhdr_path: Path
    events_tsv_path: Path
    text_embedding_path: Path


def _record_from_row(row: dict[str, str]) -> ManifestRecord:
    return ManifestRecord(
        subject=row["subject"],
        session=row["session"],
        task=row["task"],
        run=int(row["run"]),
        local_row_idx=int(row["local_row_idx"]),
        global_row_idx=int(row["global_row_idx"]),
        text_embedding_idx=int(row["text_embedding_idx"]),
        start_time=float(row["start_time"]),
        stop_time=float(row["stop_time"]),
        sfreq=float(row["sfreq"]),
        start_sample=int(row["start_sample"]),
        stop_sample=int(row["stop_sample"]),
        n_samples=int(row["n_samples"]),
        eeg_vhdr_path=Path(row["eeg_vhdr_path"]),
        events_tsv_path=Path(row["events_tsv_path"]),
        text_embedding_path=Path(row["text_embedding_path"]),
    )


def load_manifest(path: str | Path) -> list[ManifestRecord]:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        records = [_record_from_row(row) for row in csv.DictReader(f)]
    if not records:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    return records


def validate_manifest(records: list[ManifestRecord]) -> None:
    for record in records:
        if record.stop_sample <= record.start_sample:
            raise ValueError(f"Invalid sample window in record {record.global_row_idx}")
        if record.n_samples != record.stop_sample - record.start_sample:
            raise ValueError(f"n_samples mismatch in record {record.global_row_idx}")
        if not record.eeg_vhdr_path.exists():
            raise FileNotFoundError(record.eeg_vhdr_path)
        if not record.text_embedding_path.exists():
            raise FileNotFoundError(record.text_embedding_path)
