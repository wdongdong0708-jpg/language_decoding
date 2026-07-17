import torch
from torch import nn

from chineseeeg2_littleprince.models import TemporalConvEEGEncoder
from chineseeeg2_littleprince.train import (
    eeg_to_text_contrastive_loss,
    text_embedding_for_similarity,
)


def test_temporal_conv_projects_text_to_shared_embedding_dimension():
    model = TemporalConvEEGEncoder(
        in_channels=2,
        hidden_channels=4,
        embedding_dim=3,
        text_embedding_dim=5,
        text_projection_hidden_dim=7,
        dropout=0.0,
    )

    projected = model.project_text_embedding(torch.randn(2, 5))

    assert projected.shape == (2, 3)


def test_text_projection_receives_gradients_from_contrastive_similarity():
    model = TemporalConvEEGEncoder(
        in_channels=2,
        hidden_channels=4,
        embedding_dim=3,
        text_embedding_dim=5,
        dropout=0.0,
    )
    eeg_embedding = torch.randn(3, 3, requires_grad=True)
    text_embedding = torch.randn(3, 5)
    target_id = torch.arange(3)

    projected = text_embedding_for_similarity(model, text_embedding)
    loss, _ = eeg_to_text_contrastive_loss(
        eeg_embedding,
        projected,
        target_id,
        temperature=0.07,
    )
    loss.backward()

    assert model.text_projection[0].weight.grad is not None
    assert model.text_projection[-1].weight.grad is not None


def test_encoder_without_text_projection_keeps_raw_embedding_behavior():
    model = nn.Identity()
    text_embedding = torch.randn(2, 3)

    result = text_embedding_for_similarity(model, text_embedding)

    assert result is text_embedding
