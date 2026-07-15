import pytest
import torch

from chineseeeg2_littleprince.models import SubjectSpecificLinear, TemporalConvEEGEncoder


def test_subject_specific_linear_selects_one_channel_matrix_per_subject():
    layer = SubjectSpecificLinear(in_channels=2, out_channels=2, n_subjects=2)
    with torch.no_grad():
        layer.weights[0].copy_(torch.eye(2))
        layer.weights[1].copy_(2 * torch.eye(2))

    eeg = torch.ones(2, 2, 4)
    output = layer(eeg, torch.tensor([0, 1], dtype=torch.long))

    torch.testing.assert_close(output[0], eeg[0])
    torch.testing.assert_close(output[1], 2 * eeg[1])


def test_temporal_encoder_requires_subject_ids_only_when_layer_is_enabled():
    eeg = torch.randn(2, 2, 16)
    mask = torch.ones(2, 16, dtype=torch.bool)
    model = TemporalConvEEGEncoder(
        in_channels=2,
        hidden_channels=4,
        embedding_dim=3,
        dropout=0.0,
        subject_layers=True,
        n_subjects=2,
    )

    output = model(eeg, mask, subject_id=torch.tensor([0, 1], dtype=torch.long))
    assert output.shape == (2, 3)

    with pytest.raises(ValueError, match="subject_id is required"):
        model(eeg, mask)


def test_subject_specific_linear_can_start_from_identity():
    layer = SubjectSpecificLinear(
        in_channels=3,
        out_channels=3,
        n_subjects=2,
        init_identity=True,
    )
    eeg = torch.randn(2, 3, 5)

    output = layer(eeg, torch.tensor([0, 1], dtype=torch.long))

    torch.testing.assert_close(output, eeg)
