import csv
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from chineseeeg2_littleprince.data import EEGSpeechSequenceDataset, collate_eeg_speech_sequence
from chineseeeg2_littleprince.models import TemporalConvEEGSequenceEncoder
from chineseeeg2_littleprince.train_speech_sequence import sequence_similarity_logits


def _write_sequence_manifest(tmp_path: Path) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    source_manifest = project_root / "data" / "manifests" / "littleprince_pl_sub08_manifest.csv"
    with source_manifest.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader)
        fieldnames = list(reader.fieldnames or [])

    sequence_path = tmp_path / "speech_sequence.npy"
    np.save(sequence_path, np.arange(7 * 16, dtype=np.float32).reshape(7, 16))

    audio_path = Path(r"D:\dataset\ChineseEEG-2\materials&embeddings\audio\littleprince_m1\audio_1.wav")
    extras = {
        "label_id": row["text_embedding_idx"],
        "text": "1",
        "speech_sequence_path": str(sequence_path),
        "speech_sequence_idx": "0",
        "speaker_id": "littleprince_m1",
        "audio_event_idx": "17",
        "audio_file_path": str(audio_path),
        "audio_start_time": "1.0",
        "audio_stop_time": "1.5",
        "audio_sample_rate": "12000",
        "audio_start_sample": "12000",
        "audio_stop_sample": "18000",
        "n_audio_samples": "6000",
        "speech_frame_start": "0",
        "speech_frame_stop": "7",
        "speech_frame_count": "7",
        "speech_feature_dim": "16",
    }
    row.update(extras)
    output_manifest = tmp_path / "speech_sequence_manifest.csv"
    output_fieldnames = fieldnames + [name for name in extras if name not in fieldnames]
    with output_manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerow(row)
    return output_manifest


def test_eeg_speech_sequence_dataset_collate_model_and_logits(tmp_path):
    manifest = _write_sequence_manifest(tmp_path)
    dataset = EEGSpeechSequenceDataset(manifest)
    item = dataset[0]
    assert item["speech"].shape == (7, 16)

    loader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=partial(collate_eeg_speech_sequence, max_samples=1300, sequence_frames=8),
    )
    batch = next(iter(loader))
    assert batch["speech"].shape == (1, 8, 16)
    assert batch["speech_mask"].shape == (1, 8)

    model = TemporalConvEEGSequenceEncoder(
        in_channels=128,
        hidden_channels=16,
        embedding_dim=16,
        sequence_frames=8,
        dropout=0.0,
    )
    model.eval()
    with torch.no_grad():
        pred, pred_mask = model(batch["eeg"], batch["eeg_mask"], return_mask=True)
        logits = sequence_similarity_logits(
            pred,
            batch["speech"],
            pred_mask,
            batch["speech_mask"],
            temperature=1.0,
        )
    assert pred.shape == (1, 8, 16)
    assert pred_mask.shape == (1, 8)
    assert logits.shape == (1, 1)
