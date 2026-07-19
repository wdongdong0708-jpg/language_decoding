from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chineseeeg2_littleprince.io.brainvision import parse_vhdr


DEFAULT_WINDOW_SECONDS = 0.7


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


def fixed_onset_row(
    row: dict[str, str],
    *,
    window_seconds: float,
    eeg_sfreq: float,
    eeg_n_samples: int,
) -> dict[str, str]:
    """Replace a ROWS-to-ROWE window with a fixed ROWS-anchored EEG window."""

    if window_seconds <= 0:
        raise ValueError(f"window_seconds must be positive, got {window_seconds}")

    manifest_sfreq = float(row["sfreq"])
    if abs(manifest_sfreq - eeg_sfreq) > 1e-6:
        raise ValueError(
            f"Manifest/header sfreq mismatch: {manifest_sfreq} != {eeg_sfreq} "
            f"for {row['eeg_vhdr_path']}"
        )

    start_sample = int(row["start_sample"])
    fixed_samples = int(round(window_seconds * manifest_sfreq))
    if fixed_samples <= 0:
        raise ValueError(
            f"window_seconds={window_seconds} produces no samples at {manifest_sfreq} Hz"
        )
    stop_sample = start_sample + fixed_samples
    if start_sample < 0 or stop_sample > eeg_n_samples:
        raise ValueError(
            f"Fixed window {start_sample}:{stop_sample} is outside EEG length "
            f"{eeg_n_samples} for {row['eeg_vhdr_path']}"
        )

    output = dict(row)
    output["start_time"] = f"{start_sample / manifest_sfreq:.6f}"
    output["stop_time"] = f"{stop_sample / manifest_sfreq:.6f}"
    output["start_sample"] = str(start_sample)
    output["stop_sample"] = str(stop_sample)
    output["n_samples"] = str(fixed_samples)
    return output


def build_fixed_onset_manifest(
    input_manifest: Path,
    output_manifest: Path,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
) -> list[dict[str, str]]:
    with input_manifest.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames or not rows:
        raise ValueError(f"Manifest is empty: {input_manifest}")

    required = {
        "sfreq",
        "start_time",
        "stop_time",
        "start_sample",
        "stop_sample",
        "n_samples",
        "eeg_vhdr_path",
    }
    missing = required - set(fieldnames)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    recording_info: dict[Path, tuple[float, int]] = {}
    output_rows = []
    sample_counts: Counter[int] = Counter()
    for row in rows:
        vhdr_path = Path(row["eeg_vhdr_path"])
        if vhdr_path not in recording_info:
            recording_info[vhdr_path] = brainvision_sample_count(vhdr_path)
        eeg_sfreq, eeg_n_samples = recording_info[vhdr_path]
        output_row = fixed_onset_row(
            row,
            window_seconds=window_seconds,
            eeg_sfreq=eeg_sfreq,
            eeg_n_samples=eeg_n_samples,
        )
        output_rows.append(output_row)
        sample_counts[int(output_row["n_samples"])] += 1

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(
        f"fixed_onset_rows={len(output_rows)} window_seconds={window_seconds:.6f} "
        f"sample_counts={dict(sorted(sample_counts.items()))}"
    )
    print(f"wrote_manifest={output_manifest}")
    return output_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a fixed-duration continuous EEG manifest anchored only at each "
            "existing ROWS/start_sample marker. Original ROWE/stop_sample values "
            "are not used to determine the new window endpoint."
        )
    )
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument(
        "--window-seconds", type=float, default=DEFAULT_WINDOW_SECONDS
    )
    args = parser.parse_args()
    build_fixed_onset_manifest(
        input_manifest=args.input_manifest,
        output_manifest=args.output_manifest,
        window_seconds=args.window_seconds,
    )


if __name__ == "__main__":
    main()
