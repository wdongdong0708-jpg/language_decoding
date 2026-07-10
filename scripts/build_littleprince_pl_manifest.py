import argparse
import csv
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path


DATA_ROOT = Path(r"D:\dataset\ChineseEEG-2")
EEG_ROOT = DATA_ROOT / "PassiveListening" / "derivatives" / "preprocessed"
SESSION = "ses-littleprince"
TASK = "lis"
TEXT_EMBEDDING_PATH = (
    DATA_ROOT
    / "materials&embeddings"
    / "text_embedding"
    / "text_embeddings_littleprince.npy"
)
OUTPUT_DIR = Path(r"D:\code\chineseeeg2_littleprince_pl\data\manifests")
ALL_CLEAN_MANIFEST = "littleprince_pl_all_clean_manifest.csv"
ALIGNMENT_REPORT = "alignment_report.csv"

FIELDNAMES = [
    "subject",
    "session",
    "task",
    "run",
    "local_row_idx",
    "global_row_idx",
    "text_embedding_idx",
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
    "run",
    "status",
    "rows_count",
    "rowe_count",
    "expected_count",
    "reason",
]

EXPECTED_ROWS = {
    11: 75,
    12: 117,
    13: 85,
    14: 133,
    15: 126,
    16: 42,
    17: 134,
    18: 122,
    19: 78,
    110: 219,
    111: 55,
    112: 28,
    113: 151,
    114: 125,
    21: 150,
    22: 36,
    23: 106,
    24: 26,
    25: 38,
    26: 42,
    27: 259,
    28: 49,
    29: 20,
    210: 123,
    211: 145,
    212: 278,
    213: 75,
}

DTYPE_BYTES_BY_FORMAT = {
    "IEEE_FLOAT_32": 4,
}


@dataclass
class RunBuildResult:
    rows: list[dict[str, object]]
    report_row: dict[str, object]


def run_key(run: int) -> tuple[int, int]:
    text = str(run)
    return int(text[0]), int(text[1:])


RUN_ORDER = sorted(EXPECTED_ROWS, key=run_key)


def subject_key(path: Path) -> int:
    match = re.fullmatch(r"sub-(\d+)", path.stem)
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


def parse_vhdr(path: Path) -> dict[str, str]:
    values = {}
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(";") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def brainvision_info(path: Path) -> tuple[float, int, int]:
    values = parse_vhdr(path)
    binary_format = values.get("BinaryFormat")
    if binary_format not in DTYPE_BYTES_BY_FORMAT:
        raise ValueError(f"Unsupported BrainVision BinaryFormat={binary_format!r}")
    orientation = values.get("DataOrientation")
    if orientation != "MULTIPLEXED":
        raise ValueError(f"Unsupported BrainVision DataOrientation={orientation!r}")

    n_channels = int(values["NumberOfChannels"])
    sfreq = 1_000_000.0 / float(values["SamplingInterval"])
    data_file = path.with_name(values["DataFile"])
    if not data_file.exists():
        raise FileNotFoundError(data_file)

    value_count = data_file.stat().st_size // DTYPE_BYTES_BY_FORMAT[binary_format]
    if value_count % n_channels != 0:
        raise ValueError(f"{data_file} sample count is not divisible by {n_channels} channels")
    return sfreq, n_channels, value_count // n_channels


def run_text_embedding_starts() -> dict[int, int]:
    starts = {}
    cursor = 16
    for run in RUN_ORDER:
        starts[run] = cursor
        cursor += EXPECTED_ROWS[run]
    if cursor != 2853:
        raise ValueError(f"Unexpected final text embedding cursor: {cursor}")
    return starts


def report(subject: str, run: int, status: str, expected: int, reason: str, rows="", rowe="") -> dict[str, object]:
    return {
        "subject": subject,
        "run": run,
        "status": status,
        "rows_count": rows,
        "rowe_count": rowe,
        "expected_count": expected,
        "reason": reason,
    }


def validate_event_pairs(
    subject: str,
    run: int,
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
) -> RunBuildResult:
    expected = EXPECTED_ROWS[run]
    events_path = eeg_dir / f"{subject}_{SESSION}_task-{TASK}_run-{run}_events.tsv"
    vhdr_path = eeg_dir / f"{subject}_{SESSION}_task-{TASK}_run-{run}_eeg.vhdr"

    if not events_path.exists():
        return RunBuildResult([], report(subject, run, "skipped", expected, f"missing events file: {events_path}"))
    if not vhdr_path.exists():
        return RunBuildResult([], report(subject, run, "skipped", expected, f"missing vhdr file: {vhdr_path}"))

    try:
        sfreq, _n_channels, eeg_n_samples = brainvision_info(vhdr_path)
        events = read_events(events_path)
        rows = [e for e in events if e.get("trial_type") == "ROWS"]
        rowe = [e for e in events if e.get("trial_type") == "ROWE"]
        ok, reason = validate_event_pairs(subject, run, rows, rowe, expected, sfreq, eeg_n_samples)
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
        label_idx = embedding_start + local_row_idx

        manifest_rows.append(
            {
                "subject": subject,
                "session": SESSION,
                "task": TASK,
                "run": run,
                "local_row_idx": local_row_idx,
                "global_row_idx": global_row_idx + local_row_idx,
                "text_embedding_idx": label_idx,
                "start_time": f"{start_time:.6f}",
                "stop_time": f"{stop_time:.6f}",
                "sfreq": f"{sfreq:.6f}",
                "start_sample": start_sample,
                "stop_sample": stop_sample,
                "n_samples": stop_sample - start_sample,
                "eeg_vhdr_path": str(vhdr_path),
                "events_tsv_path": str(events_path),
                "text_embedding_path": str(TEXT_EMBEDDING_PATH),
            }
        )

    return RunBuildResult(
        manifest_rows,
        report(subject, run, "ok", expected, "", len(rows), len(rowe)),
    )


def safe_extract(zip_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            member_path = (destination / member.filename).resolve()
            if destination != member_path and destination not in member_path.parents:
                raise ValueError(f"Refusing unsafe zip member {member.filename!r} in {zip_path}")
        zf.extractall(destination)


def ensure_unzipped_subjects(root: Path) -> None:
    for zip_path in sorted(root.glob("sub-*.zip"), key=subject_key):
        subject = zip_path.stem
        subject_dir = root / subject
        if subject_dir.exists():
            continue

        with zipfile.ZipFile(zip_path) as zf:
            names = [name.replace("\\", "/") for name in zf.namelist() if name and not name.endswith("/")]
        has_subject_prefix = any(name.split("/", 1)[0] == subject for name in names)
        extract_to = root if has_subject_prefix else subject_dir
        extract_to.mkdir(parents=True, exist_ok=True)
        safe_extract(zip_path, extract_to)


def discover_subjects(root: Path) -> list[Path]:
    ensure_unzipped_subjects(root)
    return sorted([p for p in root.glob("sub-*") if p.is_dir()], key=subject_key)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_subject(subject: str, rows: list[dict[str, object]], skipped_runs: list[int]) -> str:
    if rows:
        text_indices = [int(row["text_embedding_idx"]) for row in rows]
        sample_lengths = [int(row["n_samples"]) for row in rows]
        idx_range = f"{min(text_indices)}..{max(text_indices)}"
        sample_range = f"{min(sample_lengths)}..{max(sample_lengths)}"
    else:
        idx_range = "-"
        sample_range = "-"

    skipped = ",".join(str(run) for run in skipped_runs) if skipped_runs else "-"
    ok_runs = len(RUN_ORDER) - len(skipped_runs)
    return (
        f"{subject}: usable_runs={ok_runs}/{len(RUN_ORDER)} "
        f"kept_rows={len(rows)} skipped_runs={skipped} "
        f"text_embedding_idx={idx_range} n_samples={sample_range}"
    )


def build_subject(subject_dir: Path, embedding_starts: dict[int, int], output_dir: Path) -> tuple[list[dict[str, object]], list[dict[str, object]], str]:
    subject = subject_dir.name
    eeg_dir = subject_dir / SESSION / "eeg"
    subject_rows = []
    report_rows = []
    skipped_runs = []
    global_row_idx = 0

    if not eeg_dir.exists():
        for run in RUN_ORDER:
            report_rows.append(report(subject, run, "skipped", EXPECTED_ROWS[run], f"missing EEG directory: {eeg_dir}"))
        return [], report_rows, summarize_subject(subject, [], RUN_ORDER)

    for run in RUN_ORDER:
        result = build_run(subject, eeg_dir, run, embedding_starts[run], global_row_idx)
        report_rows.append(result.report_row)
        if result.rows:
            subject_rows.extend(result.rows)
            global_row_idx += len(result.rows)
        else:
            skipped_runs.append(run)

    if subject_rows:
        output_path = output_dir / f"littleprince_pl_{subject.replace('-', '')}_manifest.csv"
        write_csv(output_path, subject_rows, FIELDNAMES)

    return subject_rows, report_rows, summarize_subject(subject, subject_rows, skipped_runs)


def main() -> None:
    global TEXT_EMBEDDING_PATH

    parser = argparse.ArgumentParser()
    parser.add_argument("--eeg-root", type=Path, default=EEG_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--text-embedding-path", type=Path, default=TEXT_EMBEDDING_PATH)
    args = parser.parse_args()

    TEXT_EMBEDDING_PATH = args.text_embedding_path

    if not TEXT_EMBEDDING_PATH.exists():
        raise FileNotFoundError(TEXT_EMBEDDING_PATH)

    embedding_starts = run_text_embedding_starts()
    all_rows = []
    all_report_rows = []
    summaries = []

    for subject_dir in discover_subjects(args.eeg_root):
        subject_rows, report_rows, summary = build_subject(subject_dir, embedding_starts, args.output_dir)
        all_rows.extend(subject_rows)
        all_report_rows.extend(report_rows)
        summaries.append(summary)

    write_csv(args.output_dir / ALL_CLEAN_MANIFEST, all_rows, FIELDNAMES)
    write_csv(args.output_dir / ALIGNMENT_REPORT, all_report_rows, REPORT_FIELDNAMES)

    expected_total = sum(EXPECTED_ROWS.values())
    complete_subjects = sorted(
        {row["subject"] for row in all_rows if sum(1 for item in all_rows if item["subject"] == row["subject"]) == expected_total}
    )
    subject_row_counts = {}
    for row in all_rows:
        subject_row_counts[row["subject"]] = subject_row_counts.get(row["subject"], 0) + 1
    partial_subjects = sorted(
        subject for subject, count in subject_row_counts.items() if 0 < count < expected_total
    )

    for summary in summaries:
        print(summary)
    print(f"complete_subjects={','.join(complete_subjects) if complete_subjects else '-'}")
    print(f"partial_subjects={','.join(partial_subjects) if partial_subjects else '-'}")
    print(f"all_clean_rows={len(all_rows)}")
    print(f"wrote_report={args.output_dir / ALIGNMENT_REPORT}")
    print(f"wrote_all_clean_manifest={args.output_dir / ALL_CLEAN_MANIFEST}")


if __name__ == "__main__":
    main()
