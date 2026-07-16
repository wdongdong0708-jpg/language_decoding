from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chineseeeg2_littleprince.io.brainvision import parse_vhdr


DATA_ROOT = Path(r"D:\dataset\ChineseEEG-2")
DEFAULT_EEG_ROOT = DATA_ROOT / "ReadingAloud" / "derivatives" / "preprocessed"
DEFAULT_REFERENCE_MANIFEST = (
    PROJECT_ROOT / "data" / "manifests" / "littleprince_pl_all_clean_manifest.csv"
)
DEFAULT_OUTPUT_MANIFEST = (
    PROJECT_ROOT / "data" / "manifests" / "littleprince_ra_all_clean_manifest.csv"
)
DEFAULT_ALIGNMENT_REPORT = (
    PROJECT_ROOT / "data" / "manifests" / "littleprince_ra_alignment_report.csv"
)
READING_TASK = "reading"

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
    "reason",
]


@dataclass(frozen=True)
class TargetRow:
    text_embedding_idx: int
    label_id: int
    text_embedding_path: Path


@dataclass(frozen=True)
class RunSpec:
    session: str
    run: int
    targets: tuple[TargetRow, ...]


def run_key(run: int) -> tuple[int, int]:
    text = str(run)
    return int(text[0]), int(text[1:])


def load_reference_run_specs(path: Path) -> list[RunSpec]:
    """Derive the line-to-embedding mapping from a validated PL manifest."""

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reference_rows = list(csv.DictReader(f))
    if not reference_rows:
        raise ValueError(f"Reference manifest is empty: {path}")

    mappings: dict[tuple[str, int], dict[int, TargetRow]] = {}
    session_order: list[str] = []
    for row in reference_rows:
        session = row["session"]
        run = int(row["run"])
        local_row_idx = int(row["local_row_idx"])
        text_embedding_idx = int(row["text_embedding_idx"])
        target = TargetRow(
            text_embedding_idx=text_embedding_idx,
            label_id=int(row.get("label_id") or text_embedding_idx),
            text_embedding_path=Path(row["text_embedding_path"]),
        )
        if session not in session_order:
            session_order.append(session)
        run_mapping = mappings.setdefault((session, run), {})
        previous = run_mapping.setdefault(local_row_idx, target)
        if previous != target:
            raise ValueError(
                "Inconsistent PL reference mapping for "
                f"session={session} run={run} local_row_idx={local_row_idx}: "
                f"{previous} != {target}"
            )

    specs = []
    arrays: dict[Path, np.ndarray] = {}
    for session in session_order:
        session_runs = sorted(
            (run for spec_session, run in mappings if spec_session == session),
            key=run_key,
        )
        for run in session_runs:
            mapping = mappings[(session, run)]
            expected_indices = list(range(len(mapping)))
            if sorted(mapping) != expected_indices:
                raise ValueError(
                    f"Non-contiguous local_row_idx in reference manifest for "
                    f"session={session} run={run}: {sorted(mapping)}"
                )
            targets = tuple(mapping[index] for index in expected_indices)
            for target in targets:
                if target.text_embedding_path not in arrays:
                    arrays[target.text_embedding_path] = np.load(
                        target.text_embedding_path, mmap_mode="r"
                    )
                embeddings = arrays[target.text_embedding_path]
                if embeddings.ndim != 2:
                    raise ValueError(
                        f"Expected a 2D embedding array, got shape={embeddings.shape} "
                        f"for {target.text_embedding_path}"
                    )
                if not 0 <= target.text_embedding_idx < embeddings.shape[0]:
                    raise IndexError(
                        f"text_embedding_idx={target.text_embedding_idx} is out of bounds "
                        f"for {target.text_embedding_path} with {embeddings.shape[0]} rows"
                    )
            specs.append(RunSpec(session=session, run=run, targets=targets))
    return specs


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


def validate_event_pairs(
    events: list[dict[str, str]],
    expected_count: int,
    sfreq: float,
    eeg_n_samples: int,
) -> tuple[bool, str, list[dict[str, str]], list[dict[str, str]]]:
    rows = [event for event in events if event.get("trial_type") == "ROWS"]
    rowe = [event for event in events if event.get("trial_type") == "ROWE"]
    if len(rows) != expected_count or len(rowe) != expected_count:
        return (
            False,
            f"ROWS={len(rows)}, ROWE={len(rowe)}, expected={expected_count}",
            rows,
            rowe,
        )

    line_events = [
        event for event in events if event.get("trial_type") in {"ROWS", "ROWE"}
    ]
    expected_types = [
        event_type
        for _ in range(expected_count)
        for event_type in ("ROWS", "ROWE")
    ]
    actual_types = [event.get("trial_type") for event in line_events]
    if actual_types != expected_types:
        return False, "ROWS/ROWE markers are not strictly alternating", rows, rowe

    previous_stop = -1.0
    for local_row_idx, (start_event, stop_event) in enumerate(zip(rows, rowe)):
        try:
            start_time = float(start_event["onset"])
            stop_time = float(stop_event["onset"])
        except (KeyError, ValueError) as exc:
            return False, f"invalid onset at row {local_row_idx}: {exc}", rows, rowe
        if start_time < previous_stop or stop_time <= start_time:
            return False, f"invalid or overlapping window at row {local_row_idx}", rows, rowe

        start_sample = int(round(start_time * sfreq))
        stop_sample = int(round(stop_time * sfreq))
        if start_sample < 0 or stop_sample <= start_sample or stop_sample > eeg_n_samples:
            return (
                False,
                f"row {local_row_idx}: sample window {start_sample}:{stop_sample} "
                f"outside EEG length {eeg_n_samples}",
                rows,
                rowe,
            )
        previous_stop = stop_time
    return True, "", rows, rowe


def report_row(
    subject: str,
    spec: RunSpec,
    status: str,
    reason: str,
    rows_count: int | str = "",
    rowe_count: int | str = "",
) -> dict[str, object]:
    return {
        "subject": subject,
        "session": spec.session,
        "run": spec.run,
        "status": status,
        "rows_count": rows_count,
        "rowe_count": rowe_count,
        "expected_count": len(spec.targets),
        "reason": reason,
    }


def build_run(
    subject: str,
    eeg_dir: Path,
    spec: RunSpec,
    global_row_idx: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    stem = f"{subject}_{spec.session}_task-{READING_TASK}_run-{spec.run}"
    events_path = eeg_dir / f"{stem}_events.tsv"
    vhdr_path = eeg_dir / f"{stem}_eeg.vhdr"
    if not events_path.exists():
        return [], report_row(subject, spec, "skipped", f"missing events file: {events_path}")
    if not vhdr_path.exists():
        return [], report_row(subject, spec, "skipped", f"missing vhdr file: {vhdr_path}")

    try:
        sfreq, eeg_n_samples = brainvision_sample_count(vhdr_path)
        events = read_events(events_path)
        ok, reason, rows, rowe = validate_event_pairs(
            events, len(spec.targets), sfreq, eeg_n_samples
        )
    except Exception as exc:
        return [], report_row(subject, spec, "skipped", str(exc))
    if not ok:
        return [], report_row(
            subject, spec, "skipped", reason, len(rows), len(rowe)
        )

    output_rows = []
    for local_row_idx, (start_event, stop_event, target) in enumerate(
        zip(rows, rowe, spec.targets)
    ):
        start_time = float(start_event["onset"])
        stop_time = float(stop_event["onset"])
        start_sample = int(round(start_time * sfreq))
        stop_sample = int(round(stop_time * sfreq))
        output_rows.append(
            {
                "subject": subject,
                "session": spec.session,
                "task": READING_TASK,
                "run": spec.run,
                "local_row_idx": local_row_idx,
                "global_row_idx": global_row_idx + local_row_idx,
                "text_embedding_idx": target.text_embedding_idx,
                "label_id": target.label_id,
                "start_time": f"{start_time:.6f}",
                "stop_time": f"{stop_time:.6f}",
                "sfreq": f"{sfreq:.6f}",
                "start_sample": start_sample,
                "stop_sample": stop_sample,
                "n_samples": stop_sample - start_sample,
                "eeg_vhdr_path": str(vhdr_path),
                "events_tsv_path": str(events_path),
                "text_embedding_path": str(target.text_embedding_path),
            }
        )
    return output_rows, report_row(
        subject, spec, "ok", "", len(rows), len(rowe)
    )


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_manifest(
    eeg_root: Path,
    reference_manifest: Path,
    output_manifest: Path,
    alignment_report: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    specs = load_reference_run_specs(reference_manifest)
    subject_dirs = sorted(path for path in eeg_root.glob("sub-*") if path.is_dir())
    if not subject_dirs:
        raise FileNotFoundError(f"No ReadingAloud subject directories found in {eeg_root}")

    all_rows = []
    report_rows = []
    for subject_dir in subject_dirs:
        subject = subject_dir.name
        global_row_idx = 0
        kept_runs = 0
        for spec in specs:
            eeg_dir = subject_dir / spec.session / "eeg"
            rows, report = build_run(subject, eeg_dir, spec, global_row_idx)
            report_rows.append(report)
            if rows:
                all_rows.extend(rows)
                global_row_idx += len(rows)
                kept_runs += 1
        print(
            f"{subject}: usable_runs={kept_runs}/{len(specs)} "
            f"kept_rows={global_row_idx}"
        )

    write_csv(output_manifest, all_rows, MANIFEST_FIELDNAMES)
    write_csv(alignment_report, report_rows, REPORT_FIELDNAMES)
    print(f"readingaloud_rows={len(all_rows)}")
    print(f"wrote_manifest={output_manifest}")
    print(f"wrote_report={alignment_report}")
    return all_rows, report_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a ReadingAloud EEG-text manifest by reusing the validated "
            "line-to-embedding mapping from a PassiveListening manifest."
        )
    )
    parser.add_argument("--eeg-root", type=Path, default=DEFAULT_EEG_ROOT)
    parser.add_argument(
        "--reference-manifest", type=Path, default=DEFAULT_REFERENCE_MANIFEST
    )
    parser.add_argument(
        "--output-manifest", type=Path, default=DEFAULT_OUTPUT_MANIFEST
    )
    parser.add_argument(
        "--alignment-report", type=Path, default=DEFAULT_ALIGNMENT_REPORT
    )
    args = parser.parse_args()
    build_manifest(
        eeg_root=args.eeg_root,
        reference_manifest=args.reference_manifest,
        output_manifest=args.output_manifest,
        alignment_report=args.alignment_report,
    )


if __name__ == "__main__":
    main()
