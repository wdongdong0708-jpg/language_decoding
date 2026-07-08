from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chineseeeg2_littleprince.io.brainvision import parse_vhdr


DATA_ROOT = Path(r"D:\dataset\ChineseEEG-2")
EEG_ROOT = DATA_ROOT / "PassiveListening" / "derivatives" / "preprocessed"
SESSION = "ses-garnettdream"
TASK = "lis"
TEXT_EMBEDDING_PATH = (
    DATA_ROOT
    / "materials&embeddings"
    / "text_embedding"
    / "text_embeddings_garnettdream.npy"
)
OUTPUT_DIR = Path(r"D:\code\chineseeeg2_littleprince_pl\data\manifests")
DEFAULT_LITTLEPRINCE_MANIFEST = OUTPUT_DIR / "littleprince_pl_all_clean_manifest.csv"

TEXT_EMBEDDING_OFFSET = 17
GARNETTDREAM_LABEL_OFFSET = 100_000
LITTLEPRINCE_LABEL_OFFSET = 0

GARNETTDREAM_ALL_CLEAN_MANIFEST = "garnettdream_pl_all_clean_manifest.csv"
COMBINED_ALL_CLEAN_MANIFEST = "pl_littleprince_garnettdream_all_clean_manifest.csv"
ALIGNMENT_REPORT = "garnettdream_alignment_report.csv"

FIELDNAMES = [
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

# This is the GarnettDream layout that matches text_embeddings_garnettdream.npy:
# 17 preface rows + 2147 formal ROWS/ROWE windows = 2164 embedding rows.
EXPECTED_ROWS_9_RUN = {
    11: 232,
    12: 263,
    13: 169,
    14: 341,
    15: 294,
    21: 197,
    22: 282,
    23: 235,
    24: 134,
}


@dataclass
class RunBuildResult:
    rows: list[dict[str, object]]
    report_row: dict[str, object]


def run_key(run: int) -> tuple[int, int]:
    text = str(run)
    return int(text[0]), int(text[1:])


RUN_ORDER = sorted(EXPECTED_ROWS_9_RUN, key=run_key)


def subject_key(path: Path) -> int:
    match = re.fullmatch(r"sub-(\d+)", path.name)
    if not match:
        return 10**9
    return int(match.group(1))


def parse_run(path: Path) -> int:
    match = re.search(r"run-(\d+)", path.name)
    if not match:
        raise ValueError(f"Could not parse run from {path}")
    return int(match.group(1))


def read_events(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def brainvision_sample_count(vhdr_path: Path) -> tuple[float, int]:
    info = parse_vhdr(vhdr_path)
    if not info.data_file.exists():
        raise FileNotFoundError(info.data_file)
    value_count = info.data_file.stat().st_size // info.dtype.itemsize
    if value_count % info.n_channels != 0:
        raise ValueError(f"{info.data_file} sample count is not divisible by {info.n_channels} channels")
    return info.sfreq, value_count // info.n_channels


def run_text_embedding_starts(text_embedding_offset: int) -> dict[int, int]:
    starts = {}
    cursor = text_embedding_offset
    for run in RUN_ORDER:
        starts[run] = cursor
        cursor += EXPECTED_ROWS_9_RUN[run]
    return starts


def report(
    subject: str,
    run: int | str,
    status: str,
    expected: int | str,
    reason: str,
    rows: int | str = "",
    rowe: int | str = "",
) -> dict[str, object]:
    return {
        "subject": subject,
        "session": SESSION,
        "run": run,
        "status": status,
        "rows_count": rows,
        "rowe_count": rowe,
        "expected_count": expected,
        "reason": reason,
    }


def validate_text_embedding(path: Path, text_embedding_offset: int) -> None:
    embeddings = np.load(path, mmap_mode="r")
    if embeddings.ndim != 2:
        raise ValueError(f"Expected a 2D embedding array, got shape={embeddings.shape}")
    expected_rows = text_embedding_offset + sum(EXPECTED_ROWS_9_RUN.values())
    if embeddings.shape[0] != expected_rows:
        raise ValueError(
            f"{path} has {embeddings.shape[0]} rows, but the supported GarnettDream "
            f"9-run layout needs {expected_rows} rows"
        )


def validate_event_pairs(
    rows: list[dict[str, str]],
    rowe: list[dict[str, str]],
    expected: int,
    sfreq: float,
    eeg_n_samples: int,
) -> tuple[bool, str]:
    if len(rows) != expected or len(rowe) != expected:
        return False, f"ROWS={len(rows)}, ROWE={len(rowe)}, expected={expected}"

    previous_start = None
    previous_stop = None
    for local_row_idx, (start_event, stop_event) in enumerate(zip(rows, rowe)):
        try:
            start_time = float(start_event["onset"])
            stop_time = float(stop_event["onset"])
        except (KeyError, ValueError) as exc:
            return False, f"invalid onset at row {local_row_idx}: {exc}"

        if stop_time <= start_time:
            return False, f"row {local_row_idx}: ROWE_onset <= ROWS_onset"
        if previous_start is not None and start_time <= previous_start:
            return False, f"row {local_row_idx}: ROWS onset is not strictly increasing"
        if previous_stop is not None and stop_time <= previous_stop:
            return False, f"row {local_row_idx}: ROWE onset is not strictly increasing"

        start_sample = int(round(start_time * sfreq))
        stop_sample = int(round(stop_time * sfreq))
        if start_sample < 0 or stop_sample <= start_sample or stop_sample > eeg_n_samples:
            return (
                False,
                f"row {local_row_idx}: sample window {start_sample}:{stop_sample} "
                f"outside EEG length {eeg_n_samples}",
            )

        previous_start = start_time
        previous_stop = stop_time

    return True, ""


def build_run(
    subject: str,
    eeg_dir: Path,
    run: int,
    embedding_start: int,
    global_row_idx: int,
    text_embedding_path: Path,
    label_offset: int,
) -> RunBuildResult:
    expected = EXPECTED_ROWS_9_RUN[run]
    events_path = eeg_dir / f"{subject}_{SESSION}_task-{TASK}_run-{run}_events.tsv"
    vhdr_path = eeg_dir / f"{subject}_{SESSION}_task-{TASK}_run-{run}_eeg.vhdr"

    if not events_path.exists():
        return RunBuildResult([], report(subject, run, "skipped", expected, f"missing events file: {events_path}"))
    if not vhdr_path.exists():
        return RunBuildResult([], report(subject, run, "skipped", expected, f"missing vhdr file: {vhdr_path}"))

    try:
        sfreq, eeg_n_samples = brainvision_sample_count(vhdr_path)
        events = read_events(events_path)
        rows = [event for event in events if event.get("trial_type") == "ROWS"]
        rowe = [event for event in events if event.get("trial_type") == "ROWE"]
        ok, reason = validate_event_pairs(rows, rowe, expected, sfreq, eeg_n_samples)
    except Exception as exc:
        return RunBuildResult([], report(subject, run, "skipped", expected, str(exc)))

    if not ok:
        return RunBuildResult([], report(subject, run, "skipped", expected, reason, len(rows), len(rowe)))

    manifest_rows = []
    for local_row_idx, (start_event, stop_event) in enumerate(zip(rows, rowe)):
        start_time = float(start_event["onset"])
        stop_time = float(stop_event["onset"])
        start_sample = int(round(start_time * sfreq))
        stop_sample = int(round(stop_time * sfreq))
        text_embedding_idx = embedding_start + local_row_idx

        manifest_rows.append(
            {
                "subject": subject,
                "session": SESSION,
                "task": TASK,
                "run": run,
                "local_row_idx": local_row_idx,
                "global_row_idx": global_row_idx + local_row_idx,
                "text_embedding_idx": text_embedding_idx,
                "label_id": label_offset + text_embedding_idx,
                "start_time": f"{start_time:.6f}",
                "stop_time": f"{stop_time:.6f}",
                "sfreq": f"{sfreq:.6f}",
                "start_sample": start_sample,
                "stop_sample": stop_sample,
                "n_samples": stop_sample - start_sample,
                "eeg_vhdr_path": str(vhdr_path),
                "events_tsv_path": str(events_path),
                "text_embedding_path": str(text_embedding_path),
            }
        )

    return RunBuildResult(manifest_rows, report(subject, run, "ok", expected, "", len(rows), len(rowe)))


def count_run_events(events_path: Path) -> tuple[int, int]:
    events = read_events(events_path)
    return (
        sum(1 for event in events if event.get("trial_type") == "ROWS"),
        sum(1 for event in events if event.get("trial_type") == "ROWE"),
    )


def unsupported_layout_reports(subject: str, eeg_dir: Path, reason: str) -> list[dict[str, object]]:
    reports = []
    for events_path in sorted(eeg_dir.glob("*_events.tsv"), key=lambda path: run_key(parse_run(path))):
        run = parse_run(events_path)
        try:
            rows_count, rowe_count = count_run_events(events_path)
        except Exception as exc:
            reports.append(report(subject, run, "skipped", "", f"{reason}; also failed to read events: {exc}"))
            continue
        reports.append(report(subject, run, "skipped", "", reason, rows_count, rowe_count))
    if not reports:
        reports.append(report(subject, "-", "skipped", "", reason))
    return reports


def discover_subjects(root: Path) -> list[Path]:
    return sorted([path for path in root.glob("sub-*") if path.is_dir()], key=subject_key)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_subject(subject: str, rows: list[dict[str, object]], skipped_runs: list[int], unsupported: bool) -> str:
    if rows:
        text_indices = [int(row["text_embedding_idx"]) for row in rows]
        label_ids = [int(row["label_id"]) for row in rows]
        sample_lengths = [int(row["n_samples"]) for row in rows]
        text_range = f"{min(text_indices)}..{max(text_indices)}"
        label_range = f"{min(label_ids)}..{max(label_ids)}"
        sample_range = f"{min(sample_lengths)}..{max(sample_lengths)}"
    else:
        text_range = "-"
        label_range = "-"
        sample_range = "-"

    skipped = ",".join(str(run) for run in skipped_runs) if skipped_runs else "-"
    status = "unsupported_layout" if unsupported else "supported_layout"
    ok_runs = len(RUN_ORDER) - len(skipped_runs) if not unsupported else 0
    return (
        f"{subject}: {status} usable_runs={ok_runs}/{len(RUN_ORDER)} "
        f"kept_rows={len(rows)} skipped_runs={skipped} "
        f"text_embedding_idx={text_range} label_id={label_range} n_samples={sample_range}"
    )


def build_subject(
    subject_dir: Path,
    embedding_starts: dict[int, int],
    output_dir: Path,
    text_embedding_path: Path,
    label_offset: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], str]:
    subject = subject_dir.name
    eeg_dir = subject_dir / SESSION / "eeg"
    if not eeg_dir.exists():
        reason = f"missing EEG directory: {eeg_dir}"
        report_rows = [report(subject, run, "skipped", EXPECTED_ROWS_9_RUN[run], reason) for run in RUN_ORDER]
        return [], report_rows, summarize_subject(subject, [], RUN_ORDER, unsupported=False)

    found_runs = {parse_run(path) for path in eeg_dir.glob("*_events.tsv")}
    if found_runs != set(RUN_ORDER):
        reason = (
            "unsupported GarnettDream run layout for current text_embeddings_garnettdream.npy; "
            f"found runs={','.join(str(run) for run in sorted(found_runs, key=run_key))}, "
            f"supported runs={','.join(str(run) for run in RUN_ORDER)}"
        )
        return [], unsupported_layout_reports(subject, eeg_dir, reason), summarize_subject(
            subject,
            [],
            sorted(found_runs, key=run_key),
            unsupported=True,
        )

    subject_rows = []
    report_rows = []
    skipped_runs = []
    global_row_idx = 0
    for run in RUN_ORDER:
        result = build_run(
            subject=subject,
            eeg_dir=eeg_dir,
            run=run,
            embedding_start=embedding_starts[run],
            global_row_idx=global_row_idx,
            text_embedding_path=text_embedding_path,
            label_offset=label_offset,
        )
        report_rows.append(result.report_row)
        if result.rows:
            subject_rows.extend(result.rows)
            global_row_idx += len(result.rows)
        else:
            skipped_runs.append(run)

    if subject_rows:
        output_path = output_dir / f"garnettdream_pl_{subject.replace('-', '')}_manifest.csv"
        write_csv(output_path, subject_rows, FIELDNAMES)

    return subject_rows, report_rows, summarize_subject(subject, subject_rows, skipped_runs, unsupported=False)


def label_offset_for_embedding(path: Path, garnettdream_label_offset: int) -> int:
    name = path.name.lower()
    if name == "text_embeddings_garnettdream.npy":
        return garnettdream_label_offset
    if name == "text_embeddings_littleprince.npy":
        return LITTLEPRINCE_LABEL_OFFSET
    raise ValueError(f"Unknown text embedding path for label_id offset: {path}")


def row_with_label_id(row: dict[str, str], garnettdream_label_offset: int) -> dict[str, object]:
    if row.get("label_id"):
        return dict(row)
    text_embedding_path = Path(row["text_embedding_path"])
    label_offset = label_offset_for_embedding(text_embedding_path, garnettdream_label_offset)
    updated = dict(row)
    updated["label_id"] = label_offset + int(row["text_embedding_idx"])
    return updated


def read_manifest_rows(path: Path, garnettdream_label_offset: int) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [row_with_label_id(row, garnettdream_label_offset) for row in csv.DictReader(f)]


def write_combined_manifest(
    littleprince_manifest: Path,
    garnettdream_rows: list[dict[str, object]],
    output_path: Path,
    garnettdream_label_offset: int,
) -> None:
    if not littleprince_manifest.exists():
        print(f"skip_combined_manifest missing_littleprince_manifest={littleprince_manifest}")
        return

    littleprince_rows = read_manifest_rows(littleprince_manifest, garnettdream_label_offset)
    write_csv(output_path, littleprince_rows + garnettdream_rows, FIELDNAMES)
    print(
        f"wrote_combined_manifest={output_path} "
        f"littleprince_rows={len(littleprince_rows)} garnettdream_rows={len(garnettdream_rows)} "
        f"total_rows={len(littleprince_rows) + len(garnettdream_rows)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eeg-root", type=Path, default=EEG_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--text-embedding-path", type=Path, default=TEXT_EMBEDDING_PATH)
    parser.add_argument("--text-embedding-offset", type=int, default=TEXT_EMBEDDING_OFFSET)
    parser.add_argument("--garnettdream-label-offset", type=int, default=GARNETTDREAM_LABEL_OFFSET)
    parser.add_argument("--littleprince-manifest", type=Path, default=DEFAULT_LITTLEPRINCE_MANIFEST)
    parser.add_argument("--no-combined-manifest", action="store_true")
    args = parser.parse_args()

    if not args.text_embedding_path.exists():
        raise FileNotFoundError(args.text_embedding_path)
    validate_text_embedding(args.text_embedding_path, args.text_embedding_offset)

    embedding_starts = run_text_embedding_starts(args.text_embedding_offset)
    all_rows = []
    all_report_rows = []
    summaries = []

    for subject_dir in discover_subjects(args.eeg_root):
        subject_rows, report_rows, summary = build_subject(
            subject_dir=subject_dir,
            embedding_starts=embedding_starts,
            output_dir=args.output_dir,
            text_embedding_path=args.text_embedding_path,
            label_offset=args.garnettdream_label_offset,
        )
        all_rows.extend(subject_rows)
        all_report_rows.extend(report_rows)
        summaries.append(summary)

    write_csv(args.output_dir / GARNETTDREAM_ALL_CLEAN_MANIFEST, all_rows, FIELDNAMES)
    write_csv(args.output_dir / ALIGNMENT_REPORT, all_report_rows, REPORT_FIELDNAMES)

    expected_total = sum(EXPECTED_ROWS_9_RUN.values())
    subject_row_counts: dict[str, int] = {}
    for row in all_rows:
        subject = str(row["subject"])
        subject_row_counts[subject] = subject_row_counts.get(subject, 0) + 1
    complete_subjects = sorted(subject for subject, count in subject_row_counts.items() if count == expected_total)
    partial_subjects = sorted(subject for subject, count in subject_row_counts.items() if 0 < count < expected_total)

    for summary in summaries:
        print(summary)
    print(f"complete_subjects={','.join(complete_subjects) if complete_subjects else '-'}")
    print(f"partial_subjects={','.join(partial_subjects) if partial_subjects else '-'}")
    print(f"garnettdream_rows={len(all_rows)}")
    print(f"wrote_report={args.output_dir / ALIGNMENT_REPORT}")
    print(f"wrote_garnettdream_manifest={args.output_dir / GARNETTDREAM_ALL_CLEAN_MANIFEST}")

    if not args.no_combined_manifest:
        write_combined_manifest(
            littleprince_manifest=args.littleprince_manifest,
            garnettdream_rows=all_rows,
            output_path=args.output_dir / COMBINED_ALL_CLEAN_MANIFEST,
            garnettdream_label_offset=args.garnettdream_label_offset,
        )


if __name__ == "__main__":
    main()
