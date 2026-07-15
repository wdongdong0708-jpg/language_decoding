import torch
from torch import nn

from chineseeeg2_littleprince.models import (
    SimpleConvTimeAggEEGEncoder,
    build_eeg_encoder,
)


def _small_model(**overrides):
    kwargs = {
        "in_channels": 4,
        "hidden_channels": 8,
        "embedding_dim": 6,
        "depth": 5,
        "kernel_size": 3,
        "dilation_growth": 2,
        "input_dropout": 0.1,
        "conv_dropout": 0.0,
        "dropout": 0.0,
        "glu_every": 2,
        "glu_context": 1,
        "attention_hidden_channels": 5,
        "subject_layers": True,
        "n_subjects": 2,
        "subject_layers_id": True,
    }
    kwargs.update(overrides)
    return SimpleConvTimeAggEEGEncoder(**kwargs)


def test_compact_simpleconv_has_requested_dilations_residual_blocks_and_glus():
    model = _small_model()

    assert model.initial_projection.kernel_size == (1,)
    assert model.dilations == (1, 2, 4, 8, 16)
    assert len(model.blocks) == 5
    assert set(model.glus) == {"2", "4"}
    for block in model.blocks:
        assert any(isinstance(module, nn.BatchNorm1d) for module in block.modules())
        assert any(isinstance(module, nn.GELU) for module in block.modules())


def test_compact_simpleconv_forward_masks_attention_and_preserves_output_shape():
    model = _small_model().eval()
    eeg = torch.randn(2, 4, 32)
    mask = torch.ones(2, 32, dtype=torch.bool)
    mask[1, 24:] = False
    changed_padding = eeg.clone()
    changed_padding[1, :, 24:] = 1_000.0
    subject_id = torch.tensor([0, 1], dtype=torch.long)

    with torch.no_grad():
        output, attention = model(
            eeg,
            mask,
            subject_id=subject_id,
            return_attention=True,
        )
        changed_output = model(changed_padding, mask, subject_id=subject_id)

    assert output.shape == (2, 6)
    assert attention.shape == (2, 32)
    torch.testing.assert_close(attention.sum(dim=1), torch.ones(2))
    torch.testing.assert_close(attention[1, 24:], torch.zeros(8))
    torch.testing.assert_close(output, changed_output)


def test_model_factory_selects_compact_simpleconv():
    model = build_eeg_encoder(
        "simpleconv_timeagg",
        in_channels=4,
        hidden_channels=8,
        embedding_dim=6,
        subject_layers=False,
    )

    assert isinstance(model, SimpleConvTimeAggEEGEncoder)
