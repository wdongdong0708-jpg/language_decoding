from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ManifestRecord:
    subject: str
    session: str
    task: str
    run: int
    local_row_idx: int
    global_row_idx: int
    text_embedding_idx: int
    label_id: int
    start_time: float
    stop_time: float
    sfreq: float
    start_sample: int
    stop_sample: int
    n_samples: int
    eeg_vhdr_path: Path
    events_tsv_path: Path
    text_embedding_path: Path
    instance_id: str = ""
    target_uid: str = ""
    target_id: int = -1
    split_group_id: str = ""


def _default_instance_id(row: dict[str, str]) -> str:
    return "/".join(
        [
            row["subject"],
            row["session"],
            row["task"],
            f"run-{int(row['run'])}",
            f"row-{int(row['local_row_idx'])}",
        ]
    )


def _instance_id_from_record(record: ManifestRecord) -> str:
    return "/".join(
        [
            record.subject,
            record.session,
            record.task,
            f"run-{record.run}",
            f"row-{record.local_row_idx}",
        ]
    )


def _record_from_row(row: dict[str, str]) -> ManifestRecord:
    return ManifestRecord(
        subject=row["subject"],
        session=row["session"],
        task=row["task"],
        run=int(row["run"]),
        local_row_idx=int(row["local_row_idx"]),
        global_row_idx=int(row["global_row_idx"]),
        text_embedding_idx=int(row["text_embedding_idx"]),
        label_id=int(row.get("label_id") or row["text_embedding_idx"]),
        start_time=float(row["start_time"]),
        stop_time=float(row["stop_time"]),
        sfreq=float(row["sfreq"]),
        start_sample=int(row["start_sample"]),
        stop_sample=int(row["stop_sample"]),
        n_samples=int(row["n_samples"]),
        eeg_vhdr_path=Path(row["eeg_vhdr_path"]),
        events_tsv_path=Path(row["events_tsv_path"]),
        text_embedding_path=Path(row["text_embedding_path"]),
        instance_id=row.get("instance_id") or _default_instance_id(row),
        target_uid=row.get("target_uid", ""),
        target_id=int(row["target_id"]) if row.get("target_id") not in (None, "") else -1,
        split_group_id=row.get("split_group_id", ""),
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


def canonical_target_uid(embedding: np.ndarray) -> str:
    """Return a stable identity for one exact float32 target embedding.

    P0 deliberately merges exact duplicate targets instead of occurrence-row IDs.
    Approximate semantic matching remains a loss-level experiment rather than part
    of the dataset identity definition.
    """

    vector = np.ascontiguousarray(embedding, dtype=np.float32)
    digest = hashlib.sha256(vector.tobytes()).hexdigest()
    return f"embedding-f32-sha256:{digest}"


def target_id_from_uid(target_uid: str) -> int:
    """Encode a target UID as a stable, signed-int64-safe identifier."""

    digest = hashlib.sha256(target_uid.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def attach_canonical_identities(records: list[ManifestRecord]) -> list[ManifestRecord]:
    """Attach instance, canonical-target, and split-group identities.

    Existing manifest columns take precedence. For legacy manifests, canonical
    targets are inferred from exact embedding contents and also become the split
    group, so identical line targets cannot leak across train/val/test.
    """

    arrays: dict[Path, np.ndarray] = {}
    identity_by_embedding: dict[tuple[Path, int], tuple[str, int]] = {}
    uid_by_target_id: dict[int, str] = {}
    output = []

    for record in records:
        key = (record.text_embedding_path, record.text_embedding_idx)
        if record.target_uid:
            target_uid = record.target_uid
            target_id = record.target_id if record.target_id >= 0 else target_id_from_uid(target_uid)
        else:
            if key not in identity_by_embedding:
                if record.text_embedding_path not in arrays:
                    arrays[record.text_embedding_path] = np.load(record.text_embedding_path, mmap_mode="r")
                embeddings = arrays[record.text_embedding_path]
                if not 0 <= record.text_embedding_idx < len(embeddings):
                    raise IndexError(
                        f"text_embedding_idx={record.text_embedding_idx} is out of bounds for "
                        f"{record.text_embedding_path} with {len(embeddings)} rows"
                    )
                target_uid = canonical_target_uid(embeddings[record.text_embedding_idx])
                identity_by_embedding[key] = (target_uid, target_id_from_uid(target_uid))
            target_uid, target_id = identity_by_embedding[key]

        previous_uid = uid_by_target_id.setdefault(target_id, target_uid)
        if previous_uid != target_uid:
            raise RuntimeError(f"target_id collision between {previous_uid!r} and {target_uid!r}")

        split_group_id = record.split_group_id or target_uid
        output.append(
            replace(
                record,
                instance_id=record.instance_id or _instance_id_from_record(record),
                target_uid=target_uid,
                target_id=target_id,
                split_group_id=split_group_id,
            )
        )

    return output
