import torch

from chineseeeg2_littleprince.retrieval import (
    compute_full_retrieval_metrics,
    full_retrieval_topk,
)


def test_canonical_retrieval_deduplicates_targets_and_reports_macro_metrics():
    text = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        dtype=torch.float32,
    )
    prediction = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
        dtype=torch.float32,
    )
    target_id = torch.tensor([1, 1, 1, 2], dtype=torch.long)

    metrics = compute_full_retrieval_metrics(prediction, text, target_id)

    assert metrics["candidate_count"] == 2
    assert metrics["top1"] == 0.75
    assert metrics["macro_top1"] == 0.5
    assert metrics["mean_rank"] == 0.25
    assert metrics["macro_mean_rank"] == 0.5
    assert metrics["macro_median_rank"] == 0.5
    assert metrics["instance_top1"] == 0.5
    assert metrics["instance_mean_rank"] == 0.5
    assert metrics["top10"] == 1.0
    assert full_retrieval_topk(prediction, text, target_id, k=1).item() == 0.75
