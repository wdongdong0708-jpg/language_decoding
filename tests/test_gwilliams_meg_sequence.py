import csv
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from chineseeeg2_littleprince.data import MEGSpeechSequenceDataset, collate_eeg_speech_sequence
from chineseeeg2_littleprince.meg import KitMEGReader
from chineseeeg2_littleprince.models import TemporalConvEEGSequenceEncoder


def test_gwilliams_kit_reader_reads_meg_window():
    con_path = Path(
        r"D:\experiment\brainmagick\bm\data\gwilliams2022"
        r"\sub-01\ses-0\meg\sub-01_ses-0_task-0_meg.con"
    )
    reader = KitMEGReader(con_path)
    window = reader.read_window(23506, 24506)
    assert window.shape == (208, 1000)


def test_meg_speech_sequence_dataset_collate_and_model(tmp_path):
    data_root = Path(r"D:\experiment\brainmagick\bm\data\gwilliams2022")
    sequence_path = tmp_path / "speech_sequence.npy"
    np.save(sequence_path, np.ones((12, 16), dtype=np.float32))

    manifest_path = tmp_path / "gwilliams_manifest.csv"
    row = {
        "subject": "sub-01",
        "session": "ses-0",
        "task": "0",
        "story": "lw1",
        "segment_idx": 0,
        "label_id": 0,
        "start_time": "23.506000",
        "stop_time": "26.506000",
        "sfreq": "1000.000000",
        "start_sample": 23506,
        "stop_sample": 26506,
        "n_samples": 3000,
        "meg_con_path": str(data_root / "sub-01/ses-0/meg/sub-01_ses-0_task-0_meg.con"),
        "events_tsv_path": str(data_root / "sub-01/ses-0/meg/sub-01_ses-0_task-0_events.tsv"),
        "channels_tsv_path": str(data_root / "sub-01/ses-0/meg/sub-01_ses-0_task-0_channels.tsv"),
        "speech_sequence_path": str(sequence_path),
        "speech_sequence_idx": 0,
        "speech_frame_start": 0,
        "speech_frame_stop": 12,
        "speech_frame_count": 12,
        "speech_feature_dim": 16,
        "audio_file_path": str(data_root / "stimuli/audio/lw1_0.wav"),
        "audio_start_time": "0.000000",
        "audio_stop_time": "3.000000",
        "audio_start_sample": 0,
        "audio_stop_sample": 66150,
        "n_audio_samples": 66150,
        "text": "Tara stood stock still",
    }
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    dataset = MEGSpeechSequenceDataset(manifest_path)
    batch = next(
        iter(
            DataLoader(
                dataset,
                batch_size=1,
                collate_fn=partial(
                    collate_eeg_speech_sequence,
                    max_samples=3000,
                    sequence_frames=8,
                ),
            )
        )
    )
    assert batch["eeg"].shape == (1, 208, 3000)
    assert batch["speech"].shape == (1, 8, 16)

    model = TemporalConvEEGSequenceEncoder(
        in_channels=208,
        hidden_channels=16,
        embedding_dim=16,
        sequence_frames=8,
        dropout=0.0,
    )
    model.eval()
    with torch.no_grad():
        pred, pred_mask = model(batch["eeg"], batch["eeg_mask"], return_mask=True)
    assert pred.shape == (1, 8, 16)
    assert pred_mask.shape == (1, 8)


def test_meg_speech_sequence_dataset_uses_window_cache(tmp_path):
    data_root = Path(r"D:\experiment\brainmagick\bm\data\gwilliams2022")
    sequence_path = tmp_path / "speech_sequence.npy"
    window_path = tmp_path / "meg_windows.npy"
    np.save(sequence_path, np.ones((12, 16), dtype=np.float32))
    np.save(window_path, np.full((1, 208, 750), 2.0, dtype=np.float32))

    manifest_path = tmp_path / "gwilliams_cached_manifest.csv"
    row = {
        "subject": "sub-01",
        "session": "ses-0",
        "task": "0",
        "story": "lw1",
        "segment_idx": 0,
        "label_id": 0,
        "start_time": "23.506000",
        "stop_time": "26.506000",
        "sfreq": "250.000000",
        "start_sample": 5876,
        "stop_sample": 6626,
        "n_samples": 750,
        "meg_con_path": str(data_root / "sub-01/ses-0/meg/sub-01_ses-0_task-0_meg.con"),
        "events_tsv_path": str(data_root / "sub-01/ses-0/meg/sub-01_ses-0_task-0_events.tsv"),
        "channels_tsv_path": str(data_root / "sub-01/ses-0/meg/sub-01_ses-0_task-0_channels.tsv"),
        "speech_sequence_path": str(sequence_path),
        "speech_sequence_idx": 0,
        "speech_frame_start": 0,
        "speech_frame_stop": 12,
        "speech_frame_count": 12,
        "speech_feature_dim": 16,
        "audio_file_path": str(data_root / "stimuli/audio/lw1_0.wav"),
        "audio_start_time": "0.000000",
        "audio_stop_time": "3.000000",
        "audio_start_sample": 0,
        "audio_stop_sample": 66150,
        "n_audio_samples": 66150,
        "meg_window_path": str(window_path),
        "meg_window_idx": 0,
        "meg_window_samples": 750,
        "text": "Tara stood stock still",
    }
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    dataset = MEGSpeechSequenceDataset(manifest_path, normalize_meg=False)
    item = dataset[0]
    assert item["eeg"].shape == (208, 750)
    assert torch.all(item["eeg"] == 2.0)
