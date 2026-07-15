import torch
import torch.nn.functional as F

from chineseeeg2_littleprince.train_speech import (
    dual_positive_full_retrieval_topk,
    speaker_full_retrieval_topk,
)
from chineseeeg2_littleprince.train_speech_sequence import (
    dual_positive_full_sequence_retrieval_topk,
    speaker_full_sequence_retrieval_topk,
)


def test_vector_speaker_and_dual_positive_protocols_measure_distinct_tasks():
    # Per-speaker candidates deliberately prefer the wrong text for queries 0
    # and 2. The other narrator supplies the correct text target, so dual
    # positives recover every query.
    speech = F.normalize(
        torch.tensor(
            [
                [0.0, 1.0, 0.0],  # female, text 1
                [0.9, 0.0, 0.1],  # female, text 2
                [1.0, 0.0, 0.0],  # male, text 1
                [0.0, 0.9, 0.1],  # male, text 2
            ]
        ),
        dim=-1,
    )
    prediction = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.0, 0.1],
                [0.0, 1.0, 0.0],
                [0.0, 0.9, 0.1],
            ]
        ),
        dim=-1,
    )
    label_id = torch.tensor([1, 2, 1, 2])
    speaker_ids = ["female", "female", "male", "male"]

    assert speaker_full_retrieval_topk(prediction, speech, label_id, speaker_ids, k=1).item() == 0.5
    assert dual_positive_full_retrieval_topk(prediction, speech, label_id, speaker_ids, k=1).item() == 1.0


def test_sequence_speaker_and_dual_positive_protocols_match_vector_semantics():
    speech = F.normalize(
        torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [0.9, 0.0, 0.1],
                [1.0, 0.0, 0.0],
                [0.0, 0.9, 0.1],
            ]
        ),
        dim=-1,
    )
    prediction = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.0, 0.1],
                [0.0, 1.0, 0.0],
                [0.0, 0.9, 0.1],
            ]
        ),
        dim=-1,
    )
    label_id = torch.tensor([1, 2, 1, 2])
    speaker_ids = ["female", "female", "male", "male"]
    mask = torch.ones((4, 1), dtype=torch.bool)

    assert (
        speaker_full_sequence_retrieval_topk(
            prediction.unsqueeze(1),
            speech.unsqueeze(1),
            mask,
            mask,
            label_id,
            speaker_ids,
            k=1,
        ).item()
        == 0.5
    )
    assert (
        dual_positive_full_sequence_retrieval_topk(
            prediction.unsqueeze(1),
            speech.unsqueeze(1),
            mask,
            mask,
            label_id,
            speaker_ids,
            k=1,
        ).item()
        == 1.0
    )
