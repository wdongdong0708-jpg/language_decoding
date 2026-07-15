from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DATA_ROOT = Path(r"D:\experiment\brainmagick\bm\data\gwilliams2022")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "meg" / "gwilliams_preprocessed"


def _split_arg(value: str | None) -> set[str] | None:
    if value is None or value.lower() in {"", "all", "*"}:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def _task_id(path: Path) -> str:
    match = re.search(r"_task-(\d+)_", path.name)
    if not match:
        raise ValueError(f"Could not parse task id from {path}")
    return match.group(1)


def discover_con_files(
    data_root: Path,
    subjects: set[str] | None,
    sessions: set[str] | None,
    tasks: set[str] | None,
) -> list[Path]:
    paths = []
    for con_path in sorted(data_root.glob("sub-*/ses-*/meg/*_meg.con")):
        subject = con_path.parts[-4]
        session = con_path.parts[-3]
        task = _task_id(con_path)
        if subjects is not None and subject not in subjects:
            continue
        if sessions is not None and session not in sessions:
            continue
        if tasks is not None and task not in tasks:
            continue
        paths.append(con_path)
    if not paths:
        raise FileNotFoundError("No Gwilliams .con files matched the selected filters")
    return paths


def output_paths(data_root: Path, output_root: Path, con_path: Path) -> tuple[Path, Path]:
    relative = con_path.relative_to(data_root).with_suffix(".npy")
    data_path = output_root / relative
    meta_path = data_path.with_suffix(".json")
    return data_path, meta_path


def bad_channels_from_tsv(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = csv.DictReader(f, delimiter="\t")
        return [
            row["name"]
            for row in rows
            if row.get("status", "good").lower() != "good"
        ]


def preprocess_one(
    con_path: Path,
    data_root: Path,
    output_root: Path,
    resample_freq: float,
    l_freq: float | None,
    h_freq: float | None,
    notch_freq: float | None,
    interpolate_bads: bool,
    overwrite: bool,
) -> Path:
    try:
        import mne
    except ImportError as exc:
        raise ImportError("Gwilliams preprocessing requires mne.") from exc

    data_path, meta_path = output_paths(data_root, output_root, con_path)
    if data_path.exists() and meta_path.exists() and not overwrite:
        print(f"reuse_preprocessed={data_path}")
        return data_path

    raw = mne.io.read_raw_kit(con_path, preload=True, verbose="ERROR")
    raw.pick_types(
        meg=True,
        ref_meg=False,
        misc=False,
        stim=False,
        eeg=False,
        eog=False,
        ecg=False,
    )

    channels_path = con_path.with_name(con_path.name.replace("_meg.con", "_channels.tsv"))
    bads = [name for name in bad_channels_from_tsv(channels_path) if name in raw.ch_names]
    raw.info["bads"] = sorted(set(raw.info["bads"]) | set(bads))

    if interpolate_bads and raw.info["bads"]:
        raw.interpolate_bads(reset_bads=False, verbose="ERROR")

    if notch_freq is not None and notch_freq > 0:
        raw.notch_filter(freqs=[float(notch_freq)], verbose="ERROR")

    if l_freq is not None or h_freq is not None:
        raw.filter(l_freq=l_freq, h_freq=h_freq, verbose="ERROR")

    if resample_freq and float(resample_freq) > 0 and abs(raw.info["sfreq"] - float(resample_freq)) > 1e-6:
        raw.resample(float(resample_freq), verbose="ERROR")

    data = raw.get_data().astype(np.float32, copy=False)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(data_path, data)

    metadata = {
        "source_con_path": str(con_path),
        "channels_tsv_path": str(channels_path),
        "output_path": str(data_path),
        "sfreq": float(raw.info["sfreq"]),
        "n_channels": int(data.shape[0]),
        "n_samples": int(data.shape[1]),
        "channel_names": list(raw.ch_names),
        "bads": list(raw.info["bads"]),
        "interpolate_bads": bool(interpolate_bads),
        "notch_freq": notch_freq,
        "l_freq": l_freq,
        "h_freq": h_freq,
        "resample_freq": float(resample_freq),
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"wrote_preprocessed={data_path} shape={data.shape}")
    return data_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--subjects", type=str, default="sub-01")
    parser.add_argument("--sessions", type=str, default="ses-0")
    parser.add_argument("--tasks", type=str, default="0")
    parser.add_argument("--resample-freq", type=float, default=250.0)
    parser.add_argument("--l-freq", type=float, default=1.0)
    parser.add_argument("--h-freq", type=float, default=40.0)
    parser.add_argument("--notch-freq", type=float, default=0.0)
    parser.add_argument("--interpolate-bads", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    con_paths = discover_con_files(
        data_root=args.data_root,
        subjects=_split_arg(args.subjects),
        sessions=_split_arg(args.sessions),
        tasks=_split_arg(args.tasks),
    )
    for index, con_path in enumerate(con_paths, start=1):
        print(f"preprocess {index}/{len(con_paths)} {con_path}")
        preprocess_one(
            con_path=con_path,
            data_root=args.data_root,
            output_root=args.output_root,
            resample_freq=args.resample_freq,
            l_freq=args.l_freq,
            h_freq=args.h_freq,
            notch_freq=args.notch_freq if args.notch_freq > 0 else None,
            interpolate_bads=args.interpolate_bads,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
