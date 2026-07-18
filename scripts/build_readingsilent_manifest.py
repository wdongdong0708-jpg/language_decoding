from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chineseeeg2_littleprince.io.brainvision import parse_vhdr


DATA_ROOT = Path(r"D:\dataset\ChineseEEG")
DEFAULT_EEG_ROOT = DATA_ROOT / "filtered_0.5_30"
OUTPUT_DIR = PROJECT_ROOT / "data" / "manifests"
TASK = "reading"

MANIFEST_FIELDNAMES = [
    "subject",
    "session",
    "task",
    "run",
    "local_row_idx",
    "global_row_idx",
    "text_embedding_idx",
    "label_id",
    "start_time",
    "stop_time",
    "sfreq",
    "start_sample",
    "stop_sample",
    "n_samples",
    "eeg_vhdr_path",
    "events_tsv_path",
    "text_embedding_path",
]

REPORT_FIELDNAMES = [
    "subject",
    "session",
    "run",
    "status",
    "rows_count",
    "rowe_count",
    "expected_count",
    "pre_chapter_rows_count",
    "pre_chapter_rowe_count",
    "reason",
]


@dataclass(frozen=True)
class CorpusSpec:
    key: str
    session: str
    embedding_dir: Path
    label_offset: int


@dataclass(frozen=True)
class EmbeddingRun:
    run: int
    path: Path
    rows: int
    label_start: int


CORPORA = {
    "littleprince": CorpusSpec(
        key="littleprince",
        session="ses-LittlePrince",
        embedding_dir=DATA_ROOT / "text_embeddings" / "LittlePrince_text_embedding",
        label_offset=0,
    ),
    "garnettdream": CorpusSpec(
        key="garnettdream",
        session="ses-GarnettDream",
        embedding_dir=DATA_ROOT / "text_embeddings" / "GarnettDream_text_embedding",
        label_offset=100_000,
    ),
}

RECORDING_PATTERN = re.compile(
    r"(?P<subject>sub-\d+)_(?P<session>ses-[^_]+)_task-reading_"
    r"run-(?P<run>\d+)_events\.tsv"
)
EMBEDDING_PATTERN = re.compile(r"text_embedding_run_(?P<run>\d+)\.npy")


def embedding_catalog(spec: CorpusSpec) -> dict[int, EmbeddingRun]:
    paths = []
    for path in spec.embedding_dir.glob("text_embedding_run_*.npy"):
        match = EMBEDDING_PATTERN.fullmatch(path.name)
        if match:
            paths.append((int(match["run"]), path))
    if not paths:
        raise FileNotFoundError(f"No per-run text embeddings found in {spec.embedding_dir}")

    catalog = {}
    cursor = spec.label_offset
    for run, path in sorted(paths):
        embeddings = np.load(path, mmap_mode="r")
        if embeddings.ndim != 2 or embeddings.shape[1] != 768:
            raise ValueError(f"Expected {path} to have shape [rows, 768], got {embeddings.shape}")
        catalog[run] = EmbeddingRun(
            run=run,
            path=path,
            rows=int(embeddings.shape[0]),
            label_start=cursor,
        )
        cursor += int(embeddings.shape[0])
    return catalog


def read_events(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def brainvision_sample_count(vhdr_path: Path) -> tuple[float, int]:
    info = parse_vhdr(vhdr_path)
    if not info.data_file.exists():
        raise FileNotFoundError(info.data_file)
    value_count = info.data_file.stat().st_size // info.dtype.itemsize
    if value_count % info.n_channels != 0:
        raise ValueError(
            f"{info.data_file} value count is not divisible by {info.n_channels} channels"
        )
    return info.sfreq, value_count // info.n_channels


def report_row(
    subject: str,
    session: str,
    run: int,
    status: str,
    reason: str,
    expected_count: int | str = "",
    rows_count: int | str = "",
    rowe_count: int | str = "",
    pre_rows_count: int | str = "",
    pre_rowe_count: int | str = "",
) -> dict[str, object]:
    return {
        "subject": subject,
        "session": session,
        "run": run,
        "status": status,
        "rows_count": rows_count,
        "rowe_count": rowe_count,
        "expected_count": expected_count,
        "pre_chapter_rows_count": pre_rows_count,
        "pre_chapter_rowe_count": pre_rowe_count,
        "reason": reason,
    }


def build_recording(
    events_path: Path,
    embedding_run: EmbeddingRun | None,
    global_row_idx: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    match = RECORDING_PATTERN.fullmatch(events_path.name)
    if match is None:
        raise ValueError(f"Could not parse ChineseEEG recording name: {events_path}")
    subject = match["subject"]
    session = match["session"]
    run = int(match["run"])

    if embedding_run is None:
        return [], report_row(
            subject,
            session,
            run,
            "skipped",
            "no matching per-run text embedding",
        )

    stem = events_path.name.removesuffix("_events.tsv")
    vhdr_path = events_path.with_name(f"{stem}_eeg.vhdr")
    if not vhdr_path.exists():
        return [], report_row(
            subject,
            session,
            run,
            "skipped",
            f"missing vhdr file: {vhdr_path}",
            expected_count=embedding_run.rows,
        )

    try:
        sfreq, eeg_n_samples = brainvision_sample_count(vhdr_path)
        events = read_events(events_path)
        chapter_indices = [
            index
            for index, event in enumerate(events)
            if (event.get("trial_type") or "").startswith("CH")
        ]
        if not chapter_indices:
            raise ValueError("no chapter marker found")

        first_chapter_idx = chapter_indices[0]
        pre_chapter_events = events[:first_chapter_idx]
        pre_rows_count = sum(
            event.get("trial_type") == "ROWS" for event in pre_chapter_events
        )
        pre_rowe_count = sum(
            event.get("trial_type") == "ROWE" for event in pre_chapter_events
        )
        line_events = [
            event
            for event in events[first_chapter_idx + 1 :]
            if event.get("trial_type") in {"ROWS", "ROWE"}
        ]
        rows = [event for event in line_events if event.get("trial_type") == "ROWS"]
        rowe = [event for event in line_events if event.get("trial_type") == "ROWE"]
        expected_types = [
            event_type
            for _ in range(embedding_run.rows)
            for event_type in ("ROWS", "ROWE")
        ]
        actual_types = [event.get("trial_type") for event in line_events]
        if len(rows) != embedding_run.rows or len(rowe) != embedding_run.rows:
            raise ValueError(
                f"ROWS={len(rows)}, ROWE={len(rowe)}, expected={embedding_run.rows}"
            )
        if actual_types != expected_types:
            raise ValueError("formal ROWS/ROWE markers are not strictly alternating")

        output_rows = []
        previous_stop = -1
        for local_row_idx, (start_event, stop_event) in enumerate(zip(rows, rowe)):
            start_time = float(start_event["onset"])
            stop_time = float(stop_event["onset"])
            start_sample = int(start_event["sample"])
            stop_sample = int(stop_event["sample"])
            if start_sample != round(start_time * sfreq) or stop_sample != round(
                stop_time * sfreq
            ):
                raise ValueError(f"onset/sample mismatch at row {local_row_idx}")
            if (
                start_sample < previous_stop
                or stop_sample <= start_sample
                or stop_sample > eeg_n_samples
            ):
                raise ValueError(
                    f"invalid EEG window at row {local_row_idx}: "
                    f"{start_sample}:{stop_sample} of {eeg_n_samples}"
                )
            output_rows.append(
                {
                    "subject": subject,
                    "session": session,
                    "task": TASK,
                    "run": run,
                    "local_row_idx": local_row_idx,
                    "global_row_idx": global_row_idx + local_row_idx,
                    "text_embedding_idx": local_row_idx,
                    "label_id": embedding_run.label_start + local_row_idx,
                    "start_time": f"{start_time:.8f}",
                    "stop_time": f"{stop_time:.8f}",
                    "sfreq": f"{sfreq:.6f}",
                    "start_sample": start_sample,
                    "stop_sample": stop_sample,
                    "n_samples": stop_sample - start_sample,
                    "eeg_vhdr_path": str(vhdr_path),
                    "events_tsv_path": str(events_path),
                    "text_embedding_path": str(embedding_run.path),
                }
            )
            previous_stop = stop_sample
    except Exception as exc:
        rows_count = len(rows) if "rows" in locals() else ""
        rowe_count = len(rowe) if "rowe" in locals() else ""
        return [], report_row(
            subject,
            session,
            run,
            "skipped",
            str(exc),
            expected_count=embedding_run.rows,
            rows_count=rows_count,
            rowe_count=rowe_count,
            pre_rows_count=locals().get("pre_rows_count", ""),
            pre_rowe_count=locals().get("pre_rowe_count", ""),
        )

    return output_rows, report_row(
        subject,
        session,
        run,
        "ok",
        "",
        expected_count=embedding_run.rows,
        rows_count=len(rows),
        rowe_count=len(rowe),
        pre_rows_count=pre_rows_count,
        pre_rowe_count=pre_rowe_count,
    )


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def default_output_paths(corpus: str) -> tuple[Path, Path]:
    stem = (
        "chineseeeg_readingsilent_all"
        if corpus == "all"
        else f"chineseeeg_readingsilent_{corpus}"
    )
    return (
        OUTPUT_DIR / f"{stem}_clean_manifest.csv",
        OUTPUT_DIR / f"{stem}_alignment_report.csv",
    )


def build_manifest(
    eeg_root: Path,
    corpus: str,
    output_manifest: Path,
    alignment_report: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    selected_specs = list(CORPORA.values()) if corpus == "all" else [CORPORA[corpus]]
    specs_by_session = {spec.session: spec for spec in selected_specs}
    catalogs = {spec.session: embedding_catalog(spec) for spec in selected_specs}

    output_rows = []
    report_rows = []
    global_indices: Counter[tuple[str, str]] = Counter()
    kept_by_subject: Counter[str] = Counter()
    runs_by_subject: Counter[str] = Counter()

    for events_path in sorted(eeg_root.rglob("*_events.tsv")):
        match = RECORDING_PATTERN.fullmatch(events_path.name)
        if match is None or match["session"] not in specs_by_session:
            continue
        subject = match["subject"]
        session = match["session"]
        run = int(match["run"])
        key = (subject, session)
        rows, report = build_recording(
            events_path=events_path,
            embedding_run=catalogs[session].get(run),
            global_row_idx=global_indices[key],
        )
        report_rows.append(report)
        if rows:
            output_rows.extend(rows)
            global_indices[key] += len(rows)
            kept_by_subject[subject] += len(rows)
            runs_by_subject[subject] += 1

    if not output_rows:
        raise RuntimeError(f"No usable ChineseEEG ReadingSilent rows found in {eeg_root}")
    write_csv(output_manifest, output_rows, MANIFEST_FIELDNAMES)
    write_csv(alignment_report, report_rows, REPORT_FIELDNAMES)

    for subject in sorted(kept_by_subject):
        print(
            f"{subject}: usable_runs={runs_by_subject[subject]} "
            f"kept_rows={kept_by_subject[subject]}"
        )
    skipped = sum(row["status"] != "ok" for row in report_rows)
    print(f"readingsilent_rows={len(output_rows)} skipped_recordings={skipped}")
    print(f"wrote_manifest={output_manifest}")
    print(f"wrote_report={alignment_report}")
    return output_rows, report_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a ChineseEEG silent-reading EEG-text manifest from per-run "
            "BrainVision events and per-run text embeddings."
        )
    )
    parser.add_argument("--eeg-root", type=Path, default=DEFAULT_EEG_ROOT)
    parser.add_argument(
        "--corpus",
        choices=["littleprince", "garnettdream", "all"],
        default="littleprince",
    )
    parser.add_argument("--output-manifest", type=Path, default=None)
    parser.add_argument("--alignment-report", type=Path, default=None)
    args = parser.parse_args()

    default_manifest, default_report = default_output_paths(args.corpus)
    build_manifest(
        eeg_root=args.eeg_root,
        corpus=args.corpus,
        output_manifest=args.output_manifest or default_manifest,
        alignment_report=args.alignment_report or default_report,
    )


if __name__ == "__main__":
    main()
