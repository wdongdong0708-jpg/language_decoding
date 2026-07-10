from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chineseeeg2_littleprince.audio import (  # noqa: E402
    AudioSegment,
    AudioTimeline,
    OfficialSpeechEmbedder,
    littleprince_speaker_for_subject,
    read_xlsx_column,
)


DATA_ROOT = Path(r"D:\dataset\ChineseEEG-2")
PROJECT_DATA_DIR = PROJECT_ROOT / "data"

DEFAULT_EEG_MANIFEST = PROJECT_DATA_DIR / "manifests" / "littleprince_pl_all_clean_manifest_with_text.csv"
DEFAULT_AUDIO_ROOT = DATA_ROOT / "materials&embeddings" / "audio"
DEFAULT_XLSX = DEFAULT_AUDIO_ROOT / "littleprince.xlsx"
DEFAULT_SEGMENTS_OUTPUT = PROJECT_DATA_DIR / "audio" / "littleprince_sentence_audio_sequence_segments.csv"
DEFAULT_SEQUENCE_OUTPUT = PROJECT_DATA_DIR / "audio" / "littleprince_sentence_audio_sequence_frames.npy"
DEFAULT_SEQUENCE_MANIFEST_OUTPUT = (
    PROJECT_DATA_DIR / "manifests" / "littleprince_pl_sentence_audio_sequence_manifest.csv"
)
DEFAULT_AUDIO_EVENT_TIME_SCALE = 4.0
DEFAULT_AUDIO_END_TOLERANCE_SECONDS = 0.2

SEQUENCE_FIELDNAMES = [
    "speech_sequence_idx",
    "speech_sequence_path",
    "speaker_id",
    "text_embedding_idx",
    "text",
    "audio_event_idx",
    "audio_event_time_scale",
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
]

SEQUENCE_EXTRA_FIELDNAMES = [
    "speech_sequence_idx",
    "speech_sequence_path",
    "speaker_id",
    "audio_event_idx",
    "audio_event_time_scale",
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
]


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


def littleprince_text_by_embedding_idx(xlsx_path: Path) -> dict[int, str]:
    values = read_xlsx_column(xlsx_path, column="A")
    return {embedding_idx: values[embedding_idx + 2] for embedding_idx in range(len(values) - 2)}


def base_segment_row(
    segment: AudioSegment,
    speech_sequence_idx: int,
    speech_sequence_path: Path,
) -> dict[str, object]:
    return {
        "speech_sequence_idx": speech_sequence_idx,
        "speech_sequence_path": str(speech_sequence_path),
        "speaker_id": segment.speaker_id,
        "text_embedding_idx": segment.text_embedding_idx,
        "text": segment.text,
        "audio_event_idx": segment.audio_event_idx,
        "audio_event_time_scale": segment.event_time_scale,
        "audio_file_path": str(segment.audio_file_path),
        "audio_start_time": f"{segment.audio_start_time:.6f}",
        "audio_stop_time": f"{segment.audio_stop_time:.6f}",
        "audio_sample_rate": segment.audio_sample_rate,
        "audio_start_sample": segment.audio_start_sample,
        "audio_stop_sample": segment.audio_stop_sample,
        "n_audio_samples": segment.n_audio_samples,
        "speech_frame_start": "",
        "speech_frame_stop": "",
        "speech_frame_count": "",
        "speech_feature_dim": "",
    }


def build_base_segment_rows(
    eeg_rows: list[dict[str, str]],
    audio_root: Path,
    xlsx_path: Path,
    sequence_output: Path,
    audio_event_time_scale: float = DEFAULT_AUDIO_EVENT_TIME_SCALE,
    audio_end_tolerance_seconds: float = DEFAULT_AUDIO_END_TOLERANCE_SECONDS,
    max_segments: int | None = None,
) -> list[dict[str, object]]:
    text_by_idx = littleprince_text_by_embedding_idx(xlsx_path)
    required_keys = sorted(
        {
            (littleprince_speaker_for_subject(row["subject"]), int(row["text_embedding_idx"]))
            for row in eeg_rows
        },
        key=lambda item: (item[0], item[1]),
    )
    if max_segments is not None:
        required_keys = required_keys[:max_segments]

    timelines: dict[str, AudioTimeline] = {}
    rows = []
    for speech_sequence_idx, (speaker_id, text_embedding_idx) in enumerate(required_keys):
        if speaker_id not in timelines:
            timelines[speaker_id] = AudioTimeline.from_directory(speaker_id, audio_root / speaker_id)
        segment = timelines[speaker_id].segment_for_text_embedding(
            text_embedding_idx=text_embedding_idx,
            text=text_by_idx.get(text_embedding_idx, ""),
            event_offset=1,
            event_time_scale=audio_event_time_scale,
            end_tolerance_seconds=audio_end_tolerance_seconds,
        )
        rows.append(base_segment_row(segment, speech_sequence_idx, sequence_output))
    return rows


def generate_sequence_rows(
    rows: list[dict[str, object]],
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
    for index, row in enumerate(rows, start=1):
        print(
            f"sequence_embedding {index}/{len(rows)} "
            f"{row['speaker_id']} text_idx={row['text_embedding_idx']}"
        )
        sequence = embedder.embed_file_segment_sequence(
            row["audio_file_path"],
            int(row["audio_start_sample"]),
            int(row["audio_stop_sample"]),
        )
        if sequence.ndim != 2 or sequence.shape[0] == 0:
            raise ValueError(f"Invalid sequence shape for row {index}: {sequence.shape}")
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


def write_sequence_manifest(
    eeg_rows: list[dict[str, str]],
    eeg_fieldnames: list[str],
    segment_rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    segment_by_key = {
        (str(row["speaker_id"]), int(row["text_embedding_idx"])): row
        for row in segment_rows
    }
    speech_manifest_rows = []
    for eeg_row in eeg_rows:
        text_embedding_idx = int(eeg_row["text_embedding_idx"])
        speaker_id = littleprince_speaker_for_subject(eeg_row["subject"])
        segment_row = segment_by_key.get((speaker_id, text_embedding_idx))
        if segment_row is None:
            continue
        merged: dict[str, object] = dict(eeg_row)
        merged["label_id"] = eeg_row.get("label_id") or text_embedding_idx
        merged["text"] = segment_row.get("text", eeg_row.get("text", ""))
        for fieldname in SEQUENCE_EXTRA_FIELDNAMES:
            merged[fieldname] = segment_row[fieldname]
        speech_manifest_rows.append(merged)

    fieldnames = list(eeg_fieldnames)
    for fieldname in ["label_id", "text", *SEQUENCE_EXTRA_FIELDNAMES]:
        if fieldname not in fieldnames:
            fieldnames.append(fieldname)
    write_csv(output_path, speech_manifest_rows, fieldnames)
    print(f"wrote_sequence_manifest={output_path} rows={len(speech_manifest_rows)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eeg-manifest", type=Path, default=DEFAULT_EEG_MANIFEST)
    parser.add_argument("--audio-root", type=Path, default=DEFAULT_AUDIO_ROOT)
    parser.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    parser.add_argument("--segments-output", type=Path, default=DEFAULT_SEGMENTS_OUTPUT)
    parser.add_argument("--sequence-output", type=Path, default=DEFAULT_SEQUENCE_OUTPUT)
    parser.add_argument("--sequence-manifest-output", type=Path, default=DEFAULT_SEQUENCE_MANIFEST_OUTPUT)
    parser.add_argument("--model-name", type=str, default="airesearch/wav2vec2-large-xlsr-53-th")
    parser.add_argument("--audio-event-time-scale", type=float, default=DEFAULT_AUDIO_EVENT_TIME_SCALE)
    parser.add_argument(
        "--audio-end-tolerance-seconds",
        type=float,
        default=DEFAULT_AUDIO_END_TOLERANCE_SECONDS,
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-segments", type=int, default=None)
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    eeg_rows, eeg_fieldnames = read_csv(args.eeg_manifest)
    segment_rows = build_base_segment_rows(
        eeg_rows=eeg_rows,
        audio_root=args.audio_root,
        xlsx_path=args.xlsx,
        sequence_output=args.sequence_output,
        audio_event_time_scale=args.audio_event_time_scale,
        audio_end_tolerance_seconds=args.audio_end_tolerance_seconds,
        max_segments=args.max_segments,
    )

    if args.skip_embeddings:
        write_csv(args.segments_output, segment_rows, SEQUENCE_FIELDNAMES)
        print(f"wrote_sequence_segments_metadata={args.segments_output} rows={len(segment_rows)}")
        print("skip_embeddings: sequence manifest was not written because frame offsets are unknown")
        return

    segment_rows = generate_sequence_rows(
        rows=segment_rows,
        sequence_output=args.sequence_output,
        model_name=args.model_name,
        device=args.device,
        overwrite=args.overwrite,
    )
    write_csv(args.segments_output, segment_rows, SEQUENCE_FIELDNAMES)
    print(f"wrote_sequence_segments={args.segments_output} rows={len(segment_rows)}")
    write_sequence_manifest(
        eeg_rows=eeg_rows,
        eeg_fieldnames=eeg_fieldnames,
        segment_rows=segment_rows,
        output_path=args.sequence_manifest_output,
    )


if __name__ == "__main__":
    main()
