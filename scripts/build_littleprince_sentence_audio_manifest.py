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
DEFAULT_SEGMENTS_OUTPUT = PROJECT_DATA_DIR / "audio" / "littleprince_sentence_audio_segments.csv"
DEFAULT_EMBEDDINGS_OUTPUT = PROJECT_DATA_DIR / "audio" / "littleprince_sentence_audio_embeddings.npy"
DEFAULT_SPEECH_MANIFEST_OUTPUT = (
    PROJECT_DATA_DIR / "manifests" / "littleprince_pl_sentence_audio_manifest.csv"
)

SEGMENT_FIELDNAMES = [
    "speech_embedding_idx",
    "speech_embedding_path",
    "speaker_id",
    "text_embedding_idx",
    "text",
    "audio_event_idx",
    "audio_file_path",
    "audio_start_time",
    "audio_stop_time",
    "audio_sample_rate",
    "audio_start_sample",
    "audio_stop_sample",
    "n_audio_samples",
]

SPEECH_EXTRA_FIELDNAMES = [
    "speech_embedding_idx",
    "speech_embedding_path",
    "speaker_id",
    "audio_event_idx",
    "audio_file_path",
    "audio_start_time",
    "audio_stop_time",
    "audio_sample_rate",
    "audio_start_sample",
    "audio_stop_sample",
    "n_audio_samples",
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


def segment_to_row(
    segment: AudioSegment,
    speech_embedding_idx: int,
    speech_embedding_path: Path,
) -> dict[str, object]:
    return {
        "speech_embedding_idx": speech_embedding_idx,
        "speech_embedding_path": str(speech_embedding_path),
        "speaker_id": segment.speaker_id,
        "text_embedding_idx": segment.text_embedding_idx,
        "text": segment.text,
        "audio_event_idx": segment.audio_event_idx,
        "audio_file_path": str(segment.audio_file_path),
        "audio_start_time": f"{segment.audio_start_time:.6f}",
        "audio_stop_time": f"{segment.audio_stop_time:.6f}",
        "audio_sample_rate": segment.audio_sample_rate,
        "audio_start_sample": segment.audio_start_sample,
        "audio_stop_sample": segment.audio_stop_sample,
        "n_audio_samples": segment.n_audio_samples,
    }


def build_rows(
    eeg_rows: list[dict[str, str]],
    audio_root: Path,
    xlsx_path: Path,
    embeddings_output: Path,
    max_segments: int | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
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
    key_set = set(required_keys)

    timelines: dict[str, AudioTimeline] = {}
    segment_rows = []
    segment_by_key: dict[tuple[str, int], dict[str, object]] = {}
    for speech_embedding_idx, (speaker_id, text_embedding_idx) in enumerate(required_keys):
        if speaker_id not in timelines:
            timelines[speaker_id] = AudioTimeline.from_directory(speaker_id, audio_root / speaker_id)
        segment = timelines[speaker_id].segment_for_text_embedding(
            text_embedding_idx=text_embedding_idx,
            text=text_by_idx.get(text_embedding_idx, ""),
            event_offset=1,
        )
        row = segment_to_row(segment, speech_embedding_idx, embeddings_output)
        segment_rows.append(row)
        segment_by_key[(speaker_id, text_embedding_idx)] = row

    speech_manifest_rows = []
    for eeg_row in eeg_rows:
        text_embedding_idx = int(eeg_row["text_embedding_idx"])
        speaker_id = littleprince_speaker_for_subject(eeg_row["subject"])
        key = (speaker_id, text_embedding_idx)
        if key not in key_set:
            continue
        segment_row = segment_by_key[key]
        merged: dict[str, object] = dict(eeg_row)
        merged["label_id"] = eeg_row.get("label_id") or text_embedding_idx
        merged["text"] = text_by_idx.get(text_embedding_idx, eeg_row.get("text", ""))
        for fieldname in SPEECH_EXTRA_FIELDNAMES:
            merged[fieldname] = segment_row[fieldname]
        speech_manifest_rows.append(merged)

    return segment_rows, speech_manifest_rows


def write_embeddings(
    segment_rows: list[dict[str, object]],
    embeddings_output: Path,
    model_name: str,
    device: str | None,
    overwrite: bool,
) -> None:
    embeddings_output.parent.mkdir(parents=True, exist_ok=True)
    if embeddings_output.exists() and not overwrite:
        existing = np.load(embeddings_output, mmap_mode="r")
        if existing.shape[0] != len(segment_rows):
            raise ValueError(
                f"Existing embedding row count mismatch: {embeddings_output} has "
                f"{existing.shape[0]}, expected {len(segment_rows)}. Use --overwrite to rebuild."
            )
        print(f"reuse_embeddings={embeddings_output} shape={existing.shape}")
        return

    embedder = OfficialSpeechEmbedder(model_name=model_name, device=device)
    vectors = []
    for index, row in enumerate(segment_rows, start=1):
        print(
            f"embedding {index}/{len(segment_rows)} "
            f"{row['speaker_id']} text_idx={row['text_embedding_idx']}"
        )
        vectors.append(
            embedder.embed_file_segment(
                row["audio_file_path"],
                int(row["audio_start_sample"]),
                int(row["audio_stop_sample"]),
            )
        )
    np.save(embeddings_output, np.stack(vectors, axis=0))
    print(f"wrote_embeddings={embeddings_output} shape={(len(vectors), vectors[0].shape[0])}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eeg-manifest", type=Path, default=DEFAULT_EEG_MANIFEST)
    parser.add_argument("--audio-root", type=Path, default=DEFAULT_AUDIO_ROOT)
    parser.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    parser.add_argument("--segments-output", type=Path, default=DEFAULT_SEGMENTS_OUTPUT)
    parser.add_argument("--embeddings-output", type=Path, default=DEFAULT_EMBEDDINGS_OUTPUT)
    parser.add_argument("--speech-manifest-output", type=Path, default=DEFAULT_SPEECH_MANIFEST_OUTPUT)
    parser.add_argument("--model-name", type=str, default="airesearch/wav2vec2-large-xlsr-53-th")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-segments", type=int, default=None)
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    eeg_rows, eeg_fieldnames = read_csv(args.eeg_manifest)
    segment_rows, speech_manifest_rows = build_rows(
        eeg_rows=eeg_rows,
        audio_root=args.audio_root,
        xlsx_path=args.xlsx,
        embeddings_output=args.embeddings_output,
        max_segments=args.max_segments,
    )

    write_csv(args.segments_output, segment_rows, SEGMENT_FIELDNAMES)

    speech_fieldnames = list(eeg_fieldnames)
    for fieldname in ["label_id", "text", *SPEECH_EXTRA_FIELDNAMES]:
        if fieldname not in speech_fieldnames:
            speech_fieldnames.append(fieldname)
    write_csv(args.speech_manifest_output, speech_manifest_rows, speech_fieldnames)

    print(f"wrote_segments={args.segments_output} rows={len(segment_rows)}")
    print(f"wrote_speech_manifest={args.speech_manifest_output} rows={len(speech_manifest_rows)}")

    if args.skip_embeddings:
        print(f"skip_embeddings target={args.embeddings_output}")
    else:
        write_embeddings(
            segment_rows=segment_rows,
            embeddings_output=args.embeddings_output,
            model_name=args.model_name,
            device=args.device,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
