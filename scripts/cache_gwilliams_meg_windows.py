from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from preprocess_gwilliams_meg import preprocess_one  # noqa: E402


DATA_ROOT = Path(r"D:\experiment\brainmagick\bm\data\gwilliams2022")
DEFAULT_PREPROCESSED_ROOT = PROJECT_ROOT / "data" / "meg" / "gwilliams_preprocessed"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "gwilliams_meg_speech_sequence_manifest.csv"
DEFAULT_WINDOW_OUTPUT = PROJECT_ROOT / "data" / "meg" / "gwilliams_meg_sequence_windows.npy"
DEFAULT_CACHED_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "gwilliams_meg_speech_sequence_cached_manifest.csv"


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def preprocessed_paths(
    data_root: Path,
    preprocessed_root: Path,
    source_con_path: str,
) -> tuple[Path, Path]:
    source = Path(source_con_path)
    relative = source.relative_to(data_root).with_suffix(".npy")
    data_path = preprocessed_root / relative
    return data_path, data_path.with_suffix(".json")


def read_metadata(meta_path: Path) -> dict[str, object]:
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _fit_window(window: np.ndarray, target_samples: int) -> np.ndarray:
    if window.shape[1] == target_samples:
        return window
    if window.shape[1] > target_samples:
        return window[:, :target_samples]
    out = np.zeros((window.shape[0], target_samples), dtype=window.dtype)
    out[:, : window.shape[1]] = window
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--preprocessed-root", type=Path, default=DEFAULT_PREPROCESSED_ROOT)
    parser.add_argument("--window-output", type=Path, default=DEFAULT_WINDOW_OUTPUT)
    parser.add_argument("--cached-manifest-output", type=Path, default=DEFAULT_CACHED_MANIFEST)
    parser.add_argument("--window-samples", type=int, default=None)
    parser.add_argument("--skip-missing-preprocessed", action="store_true")
    parser.add_argument("--preprocess-missing", action="store_true")
    parser.add_argument("--delete-new-preprocessed-after-cache", action="store_true")
    parser.add_argument("--resample-freq", type=float, default=250.0)
    parser.add_argument("--l-freq", type=float, default=1.0)
    parser.add_argument("--h-freq", type=float, default=40.0)
    parser.add_argument("--notch-freq", type=float, default=0.0)
    parser.add_argument("--interpolate-bads", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.window_output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.window_output} already exists. Use --overwrite to rebuild.")

    rows, fieldnames = read_csv(args.manifest)
    if not rows:
        raise ValueError(f"Manifest is empty: {args.manifest}")

    if args.skip_missing_preprocessed:
        original_count = len(rows)
        rows = [
            row
            for row in rows
            if preprocessed_paths(args.data_root, args.preprocessed_root, row["meg_con_path"])[0].exists()
        ]
        print(f"kept_preprocessed_rows={len(rows)}/{original_count}")
        if not rows:
            raise FileNotFoundError("No manifest rows have matching preprocessed MEG arrays")

    first_data_path, first_meta_path = preprocessed_paths(args.data_root, args.preprocessed_root, rows[0]["meg_con_path"])
    created_preprocessed: set[Path] = set()
    if not first_data_path.exists() and args.preprocess_missing:
        preprocess_one(
            con_path=Path(rows[0]["meg_con_path"]),
            data_root=args.data_root,
            output_root=args.preprocessed_root,
            resample_freq=args.resample_freq,
            l_freq=args.l_freq,
            h_freq=args.h_freq,
            notch_freq=args.notch_freq if args.notch_freq > 0 else None,
            interpolate_bads=args.interpolate_bads,
            overwrite=args.overwrite,
        )
        created_preprocessed.add(first_data_path)
    if not first_data_path.exists():
        raise FileNotFoundError(first_data_path)
    first_meta = read_metadata(first_meta_path)
    n_channels = int(first_meta["n_channels"])
    sfreq = float(first_meta["sfreq"])
    target_samples = args.window_samples or int(round((float(rows[0]["stop_time"]) - float(rows[0]["start_time"])) * sfreq))
    if target_samples <= 0:
        raise ValueError(f"Invalid target window samples: {target_samples}")

    args.window_output.parent.mkdir(parents=True, exist_ok=True)
    windows = open_memmap(
        args.window_output,
        mode="w+",
        dtype=np.float32,
        shape=(len(rows), n_channels, target_samples),
    )

    cached_rows = []
    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped_indices[row["meg_con_path"]].append(index)

    cached_by_index: dict[int, dict[str, object]] = {}
    processed = 0
    for group_index, (con_path_text, indices) in enumerate(grouped_indices.items(), start=1):
        data_path, meta_path = preprocessed_paths(args.data_root, args.preprocessed_root, con_path_text)
        if not data_path.exists():
            if not args.preprocess_missing:
                raise FileNotFoundError(data_path)
            print(f"preprocess_missing {group_index}/{len(grouped_indices)} {con_path_text}")
            preprocess_one(
                con_path=Path(con_path_text),
                data_root=args.data_root,
                output_root=args.preprocessed_root,
                resample_freq=args.resample_freq,
                l_freq=args.l_freq,
                h_freq=args.h_freq,
                notch_freq=args.notch_freq if args.notch_freq > 0 else None,
                interpolate_bads=args.interpolate_bads,
                overwrite=args.overwrite,
            )
            created_preprocessed.add(data_path)

        recording = np.load(data_path, mmap_mode="r")
        metadata = read_metadata(meta_path)
        row_sfreq = float(metadata["sfreq"])
        for index in indices:
            row = rows[index]
            start_sample = int(round(float(row["start_time"]) * row_sfreq))
            stop_sample = start_sample + target_samples
            if start_sample < 0 or start_sample >= recording.shape[1]:
                raise ValueError(f"Invalid start sample {start_sample} for {data_path}")
            window = np.asarray(recording[:, start_sample:min(stop_sample, recording.shape[1])], dtype=np.float32)
            windows[index] = _fit_window(window, target_samples)

            updated = dict(row)
            updated["sfreq"] = f"{row_sfreq:.6f}"
            updated["start_sample"] = start_sample
            updated["stop_sample"] = start_sample + target_samples
            updated["n_samples"] = target_samples
            updated["meg_window_path"] = str(args.window_output)
            updated["meg_window_idx"] = index
            updated["meg_window_samples"] = target_samples
            cached_by_index[index] = updated
            processed += 1

            if processed % 100 == 0 or processed == len(rows):
                print(f"cached_windows {processed}/{len(rows)}")

        del recording
        if args.delete_new_preprocessed_after_cache and data_path in created_preprocessed:
            data_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
            print(f"deleted_transient_preprocessed={data_path}")

    windows.flush()
    cached_rows = [cached_by_index[index] for index in range(len(rows))]
    output_fieldnames = list(fieldnames)
    for fieldname in ["meg_window_path", "meg_window_idx", "meg_window_samples"]:
        if fieldname not in output_fieldnames:
            output_fieldnames.append(fieldname)
    write_csv(args.cached_manifest_output, cached_rows, output_fieldnames)
    print(f"wrote_window_cache={args.window_output} shape={(len(rows), n_channels, target_samples)}")
    print(f"wrote_cached_manifest={args.cached_manifest_output} rows={len(cached_rows)}")


if __name__ == "__main__":
    main()
