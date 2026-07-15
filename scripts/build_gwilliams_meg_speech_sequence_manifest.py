from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chineseeeg2_littleprince.audio import OfficialSpeechEmbedder  # noqa: E402


DATA_ROOT = Path(r"D:\experiment\brainmagick\bm\data\gwilliams2022")
DEFAULT_SEGMENTS_OUTPUT = PROJECT_ROOT / "data" / "audio" / "gwilliams_meg_speech_sequence_segments.csv"
DEFAULT_SEQUENCE_OUTPUT = PROJECT_ROOT / "data" / "audio" / "gwilliams_meg_speech_sequence_frames.npy"
DEFAULT_MANIFEST_OUTPUT = PROJECT_ROOT / "data" / "manifests" / "gwilliams_meg_speech_sequence_manifest.csv"

SEGMENT_FIELDNAMES = [
    "speech_sequence_idx",
    "speech_sequence_path",
    "label_id",
    "story",
    "audio_file_path",
    "audio_start_time",
    "audio_stop_time",
    "audio_sample_rate",
    "audio_start_sample",
    "audio_stop_sample",
    "n_audio_samples",
    "speech_frame_start",
    "speech_frame_stop",
    "speech_frame_count",
    "speech_feature_dim",
    "text",
]

MANIFEST_FIELDNAMES = [
    "subject",
    "session",
    "task",
    "story",
    "segment_idx",
    "label_id",
    "start_time",
    "stop_time",
    "sfreq",
    "start_sample",
    "stop_sample",
    "n_samples",
    "meg_con_path",
    "events_tsv_path",
    "channels_tsv_path",
    "speech_sequence_path",
    "speech_sequence_idx",
    "speech_frame_start",
    "speech_frame_stop",
    "speech_frame_count",
    "speech_feature_dim",
    "audio_file_path",
    "audio_start_time",
    "audio_stop_time",
    "audio_start_sample",
    "audio_stop_sample",
    "n_audio_samples",
    "text",
]


@dataclass(frozen=True)
class RecordingFiles:
    subject: str
    session: str
    task: str
    meg_con_path: Path
    meg_json_path: Path
    events_tsv_path: Path
    channels_tsv_path: Path


def _split_arg(value: str | None) -> set[str] | None:
    if value is None or value.lower() in {"", "all", "*"}:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def _task_id(path: Path) -> str:
    match = re.search(r"_task-(\d+)_", path.name)
    if not match:
        raise ValueError(f"Could not parse task id from {path}")
    return match.group(1)


def discover_recordings(
    data_root: Path,
    subjects: set[str] | None,
    sessions: set[str] | None,
    tasks: set[str] | None,
) -> list[RecordingFiles]:
    recordings = []
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
        stem = con_path.name.replace("_meg.con", "")
        recordings.append(
            RecordingFiles(
                subject=subject,
                session=session,
                task=task,
                meg_con_path=con_path,
                meg_json_path=con_path.with_name(stem + "_meg.json"),
                events_tsv_path=con_path.with_name(stem + "_events.tsv"),
                channels_tsv_path=con_path.with_name(stem + "_channels.tsv"),
            )
        )
    if not recordings:
        raise FileNotFoundError("No Gwilliams recordings matched the selected filters")
    return recordings


def read_events(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            info = ast.literal_eval(row["trial_type"])
            info["onset"] = float(row["onset"])
            info["duration"] = float(row["duration"])
            info["sample"] = int(row["sample"])
            rows.append(info)
    return rows


def canonical_audio_path(data_root: Path, sound: str) -> Path:
    raw = sound.replace("\\", "/")
    path = data_root / raw
    if path.exists():
        return path

    candidate = path.with_name(re.sub(r"\.0(?=\.wav$)", "", path.name))
    if candidate.exists():
        return candidate
    raise FileNotFoundError(path)


def words_for_window(
    events: list[dict[str, object]],
    audio_path: Path,
    start_time: float,
    stop_time: float,
    data_root: Path,
) -> str:
    words = []
    for event in events:
        if event.get("kind") != "word":
            continue
        if event.get("condition") != "sentence":
            continue
        try:
            event_audio = canonical_audio_path(data_root, str(event["sound"]))
        except FileNotFoundError:
            continue
        if event_audio != audio_path:
            continue
        event_start = float(event.get("start", math.nan))
        if start_time <= event_start < stop_time:
            word = str(event.get("word", "")).strip()
            if word:
                words.append(word)
    return " ".join(words)


def build_metadata_rows(
    data_root: Path,
    recordings: list[RecordingFiles],
    sequence_output: Path,
    window_seconds: float,
    stride_seconds: float,
    max_segments: int | None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    candidates: dict[tuple[str, str, int, int], dict[str, object]] = {}
    manifest_pending = []

    for recording in recordings:
        with recording.meg_json_path.open("r", encoding="utf-8") as f:
            meg_info = json.load(f)
        sfreq = float(meg_info["SamplingFrequency"])
        recording_samples = int(round(float(meg_info["RecordingDuration"]) * sfreq))
        events = read_events(recording.events_tsv_path)
        sound_events = [event for event in events if event.get("kind") == "sound"]

        for sound_event in sound_events:
            story = str(sound_event["story"])
            audio_path = canonical_audio_path(data_root, str(sound_event["sound"]))
            audio_info = sf.info(str(audio_path))
            audio_duration = float(audio_info.frames / audio_info.samplerate)
            sound_onset = float(sound_event["onset"])

            offset = 0.0
            while offset + window_seconds <= audio_duration + 1e-6:
                audio_start = offset
                audio_stop = min(offset + window_seconds, audio_duration)
                meg_start_time = sound_onset + audio_start
                meg_stop_time = sound_onset + audio_stop
                start_sample = int(round(meg_start_time * sfreq))
                stop_sample = int(round(meg_stop_time * sfreq))
                if start_sample >= 0 and stop_sample <= recording_samples and stop_sample > start_sample:
                    audio_start_sample = int(round(audio_start * audio_info.samplerate))
                    audio_stop_sample = int(round(audio_stop * audio_info.samplerate))
                    key = (
                        story,
                        str(audio_path),
                        audio_start_sample,
                        audio_stop_sample,
                    )
                    if key not in candidates:
                        label_id = len(candidates)
                        candidates[key] = {
                            "speech_sequence_idx": label_id,
                            "speech_sequence_path": str(sequence_output),
                            "label_id": label_id,
                            "story": story,
                            "audio_file_path": str(audio_path),
                            "audio_start_time": f"{audio_start:.6f}",
                            "audio_stop_time": f"{audio_stop:.6f}",
                            "audio_sample_rate": audio_info.samplerate,
                            "audio_start_sample": audio_start_sample,
                            "audio_stop_sample": audio_stop_sample,
                            "n_audio_samples": audio_stop_sample - audio_start_sample,
                            "speech_frame_start": "",
                            "speech_frame_stop": "",
                            "speech_frame_count": "",
                            "speech_feature_dim": "",
                            "text": words_for_window(events, audio_path, audio_start, audio_stop, data_root),
                        }
                    candidate = candidates[key]
                    manifest_pending.append(
                        {
                            "subject": recording.subject,
                            "session": recording.session,
                            "task": recording.task,
                            "story": story,
                            "segment_idx": len(manifest_pending),
                            "label_id": candidate["label_id"],
                            "start_time": f"{meg_start_time:.6f}",
                            "stop_time": f"{meg_stop_time:.6f}",
                            "sfreq": f"{sfreq:.6f}",
                            "start_sample": start_sample,
                            "stop_sample": stop_sample,
                            "n_samples": stop_sample - start_sample,
                            "meg_con_path": str(recording.meg_con_path),
                            "events_tsv_path": str(recording.events_tsv_path),
                            "channels_tsv_path": str(recording.channels_tsv_path),
                            "_candidate_key": key,
                        }
                    )
                offset += stride_seconds
                if max_segments is not None and len(candidates) >= max_segments:
                    break
            if max_segments is not None and len(candidates) >= max_segments:
                break
        if max_segments is not None and len(candidates) >= max_segments:
            break

    segment_rows = sorted(candidates.values(), key=lambda row: int(row["speech_sequence_idx"]))
    manifest_rows = []
    for row in manifest_pending:
        candidate = candidates[row.pop("_candidate_key")]
        merged = dict(row)
        for fieldname in [
            "speech_sequence_path",
            "speech_sequence_idx",
            "speech_frame_start",
            "speech_frame_stop",
            "speech_frame_count",
            "speech_feature_dim",
            "audio_file_path",
            "audio_start_time",
            "audio_stop_time",
            "audio_start_sample",
            "audio_stop_sample",
            "n_audio_samples",
            "text",
        ]:
            merged[fieldname] = candidate[fieldname]
        manifest_rows.append(merged)

    return segment_rows, manifest_rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_sequence_rows(
    segment_rows: list[dict[str, object]],
    sequence_output: Path,
    model_name: str,
    device: str | None,
    overwrite: bool,
) -> list[dict[str, object]]:
    if sequence_output.exists() and not overwrite:
        raise FileExistsError(f"{sequence_output} already exists. Use --overwrite to rebuild.")

    embedder = OfficialSpeechEmbedder(model_name=model_name, device=device)
    sequences = []
    frame_cursor = 0
    updated_rows = []
    for index, row in enumerate(segment_rows, start=1):
        print(
            f"sequence_embedding {index}/{len(segment_rows)} "
            f"{row['story']} {Path(str(row['audio_file_path'])).name} "
            f"{row['audio_start_time']}..{row['audio_stop_time']}"
        )
        sequence = embedder.embed_file_segment_sequence(
            row["audio_file_path"],
            int(row["audio_start_sample"]),
            int(row["audio_stop_sample"]),
        )
        updated = dict(row)
        updated["speech_frame_start"] = frame_cursor
        updated["speech_frame_stop"] = frame_cursor + sequence.shape[0]
        updated["speech_frame_count"] = sequence.shape[0]
        updated["speech_feature_dim"] = sequence.shape[1]
        updated_rows.append(updated)
        sequences.append(sequence.astype(np.float32, copy=False))
        frame_cursor += sequence.shape[0]

    sequence_output.parent.mkdir(parents=True, exist_ok=True)
    frames = np.concatenate(sequences, axis=0)
    np.save(sequence_output, frames)
    print(f"wrote_sequence_frames={sequence_output} shape={frames.shape}")
    return updated_rows


def fill_manifest_frames(
    manifest_rows: list[dict[str, object]],
    segment_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    by_idx = {int(row["speech_sequence_idx"]): row for row in segment_rows}
    output = []
    for row in manifest_rows:
        segment = by_idx[int(row["speech_sequence_idx"])]
        updated = dict(row)
        for fieldname in [
            "speech_frame_start",
            "speech_frame_stop",
            "speech_frame_count",
            "speech_feature_dim",
        ]:
            updated[fieldname] = segment[fieldname]
        output.append(updated)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--subjects", type=str, default="sub-01")
    parser.add_argument("--sessions", type=str, default="ses-0")
    parser.add_argument("--tasks", type=str, default="0")
    parser.add_argument("--window-seconds", type=float, default=3.0)
    parser.add_argument("--stride-seconds", type=float, default=1.5)
    parser.add_argument("--segments-output", type=Path, default=DEFAULT_SEGMENTS_OUTPUT)
    parser.add_argument("--sequence-output", type=Path, default=DEFAULT_SEQUENCE_OUTPUT)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST_OUTPUT)
    parser.add_argument("--model-name", type=str, default="airesearch/wav2vec2-large-xlsr-53-th")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-segments", type=int, default=None)
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    recordings = discover_recordings(
        data_root=args.data_root,
        subjects=_split_arg(args.subjects),
        sessions=_split_arg(args.sessions),
        tasks=_split_arg(args.tasks),
    )
    segment_rows, manifest_rows = build_metadata_rows(
        data_root=args.data_root,
        recordings=recordings,
        sequence_output=args.sequence_output,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
        max_segments=args.max_segments,
    )

    if args.skip_embeddings:
        write_csv(args.segments_output, segment_rows, SEGMENT_FIELDNAMES)
        print(f"wrote_segments_metadata={args.segments_output} rows={len(segment_rows)}")
        print(f"metadata_manifest_rows={len(manifest_rows)}")
        print("skip_embeddings: training manifest was not written because frame offsets are unknown")
        return

    segment_rows = generate_sequence_rows(
        segment_rows=segment_rows,
        sequence_output=args.sequence_output,
        model_name=args.model_name,
        device=args.device,
        overwrite=args.overwrite,
    )
    manifest_rows = fill_manifest_frames(manifest_rows, segment_rows)
    write_csv(args.segments_output, segment_rows, SEGMENT_FIELDNAMES)
    write_csv(args.manifest_output, manifest_rows, MANIFEST_FIELDNAMES)
    print(f"wrote_segments={args.segments_output} rows={len(segment_rows)}")
    print(f"wrote_manifest={args.manifest_output} rows={len(manifest_rows)}")


if __name__ == "__main__":
    main()
