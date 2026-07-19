import torch

from chineseeeg2_littleprince.data import collate_eeg_text


def _item(eeg: torch.Tensor, index: int) -> dict:
    return {
        "eeg": eeg,
        "label": torch.tensor([float(index), 1.0]),
        "length": torch.tensor(eeg.shape[-1]),
        "text_embedding_idx": torch.tensor(index),
        "label_id": torch.tensor(index),
        "target_id": torch.tensor(index),
        "subject_id": torch.tensor(0),
        "meta": {"index": index},
    }


def test_resampled_collate_time_normalizes_complete_windows_and_hides_length():
    short = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    long = torch.arange(16, dtype=torch.float32).reshape(2, 8)

    batch = collate_eeg_text(
        [_item(short, 0), _item(long, 1)],
        max_samples=6,
        time_fit_mode="resample",
    )

    assert batch["eeg"].shape == (2, 2, 6)
    assert batch["mask"].all()
    assert batch["length"].tolist() == [6, 6]
    assert not torch.equal(batch["eeg"][0, :, :4], short)
    assert not torch.equal(batch["eeg"][1, :, :6], long[:, :6])


def test_resampled_collate_requires_an_explicit_common_length():
    item = _item(torch.zeros(2, 4), 0)

    try:
        collate_eeg_text([item], time_fit_mode="resample")
    except ValueError as exc:
        assert "max_samples is required" in str(exc)
    else:
        raise AssertionError("Expected resampling without max_samples to fail")
