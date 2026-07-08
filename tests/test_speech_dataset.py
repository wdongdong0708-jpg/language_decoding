import csv
from pathlib import Path

import numpy as np

from chineseeeg2_littleprince.data.speech_dataset import EEGSpeechDataset


def test_eeg_speech_dataset_reads_sentence_audio_label(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    source_manifest = project_root / "data" / "manifests" / "littleprince_pl_sub08_manifest.csv"
    with source_manifest.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader)
        fieldnames = list(reader.fieldnames or [])

    embedding_path = tmp_path / "speech.npy"
    np.save(embedding_path, np.ones((1, 1024), dtype=np.float32))

    audio_path = Path(r"D:\dataset\ChineseEEG-2\materials&embeddings\audio\littleprince_m1\audio_1.wav")
    extras = {
        "label_id": row["text_embedding_idx"],
        "text": "1",
        "speech_embedding_path": str(embedding_path),
        "speech_embedding_idx": "0",
        "speaker_id": "littleprince_m1",
        "audio_event_idx": "17",
        "audio_file_path": str(audio_path),
        "audio_start_time": "1.0",
        "audio_stop_time": "1.5",
        "audio_sample_rate": "12000",
        "audio_start_sample": "12000",
        "audio_stop_sample": "18000",
        "n_audio_samples": "6000",
    }
    row.update(extras)
    output_manifest = tmp_path / "speech_manifest.csv"
    output_fieldnames = fieldnames + [name for name in extras if name not in fieldnames]
    with output_manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerow(row)

    dataset = EEGSpeechDataset(output_manifest)
    item = dataset[0]
    assert item["label"].shape == (1024,)
    assert item["meta"]["speaker_id"] == "littleprince_m1"
