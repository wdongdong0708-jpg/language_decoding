from __future__ import annotations

from collections import Counter

import torch
import torch.nn.functional as F


def unique_target_candidates(
    text_embedding: torch.Tensor,
    target_id: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one deterministic text candidate per canonical target."""

    first_by_target: dict[int, int] = {}
    for index, value in enumerate(target_id.detach().cpu().tolist()):
        first_by_target.setdefault(int(value), index)
    ordered_target_ids = sorted(first_by_target)
    candidate_indices = torch.tensor(
        [first_by_target[value] for value in ordered_target_ids],
        dtype=torch.long,
        device=text_embedding.device,
    )
    candidate_target_ids = torch.tensor(
        ordered_target_ids,
        dtype=target_id.dtype,
        device=target_id.device,
    )
    return text_embedding.index_select(0, candidate_indices), candidate_target_ids


def retrieval_ranks(
    eeg_embedding: torch.Tensor,
    text_embedding: torch.Tensor,
    target_id: torch.Tensor,
    chunk_size: int = 1024,
) -> tuple[torch.Tensor, int]:
    """Compute tie-aware zero-based ranks against canonical target candidates."""

    candidates, candidate_target_ids = unique_target_candidates(text_embedding, target_id)
    eeg_embedding = F.normalize(eeg_embedding, dim=-1)
    candidates = F.normalize(candidates, dim=-1)
    candidate_lookup = {
        int(value): index for index, value in enumerate(candidate_target_ids.detach().cpu().tolist())
    }
    true_indices = torch.tensor(
        [candidate_lookup[int(value)] for value in target_id.detach().cpu().tolist()],
        dtype=torch.long,
        device=target_id.device,
    )

    ranks = []
    for start in range(0, eeg_embedding.shape[0], chunk_size):
        stop = min(start + chunk_size, eeg_embedding.shape[0])
        scores = eeg_embedding[start:stop] @ candidates.T
        true_scores = scores.gather(1, true_indices[start:stop, None])
        ranks_greater = (scores > true_scores).sum(dim=1)
        ranks_greater_equal = (scores >= true_scores).sum(dim=1) - 1
        ranks.append((ranks_greater + ranks_greater_equal).to(torch.float32) / 2)
    return torch.cat(ranks), int(candidates.shape[0])


def _macro_mean(values: torch.Tensor, target_id: torch.Tensor) -> torch.Tensor:
    per_target = []
    for value in torch.unique(target_id):
        per_target.append(values[target_id == value].to(torch.float32).mean())
    return torch.stack(per_target).mean()


def _macro_median(values: torch.Tensor, target_id: torch.Tensor) -> torch.Tensor:
    per_target = []
    for value in torch.unique(target_id):
        per_target.append(values[target_id == value].to(torch.float32).median())
    return torch.stack(per_target).mean()


def _aggregate_predictions_by_target(
    prediction: torch.Tensor,
    text_embedding: torch.Tensor,
    target_id: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    predictions = []
    targets = []
    target_ids = []
    for value in torch.unique(target_id, sorted=True):
        indices = torch.where(target_id == value)[0]
        predictions.append(prediction.index_select(0, indices).mean(dim=0))
        targets.append(text_embedding[indices[0]])
        target_ids.append(value)
    return torch.stack(predictions), torch.stack(targets), torch.stack(target_ids)


def _limit_to_frequent_targets(
    prediction: torch.Tensor,
    text_embedding: torch.Tensor,
    target_id: torch.Tensor,
    candidate_limit: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if candidate_limit is None:
        return prediction, text_embedding, target_id
    if candidate_limit <= 0:
        raise ValueError(f"candidate_limit must be positive, got {candidate_limit}")

    counts = Counter(int(value) for value in target_id.detach().cpu().tolist())
    selected = {
        value
        for value, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:candidate_limit]
    }
    mask = torch.tensor(
        [int(value) in selected for value in target_id.detach().cpu().tolist()],
        dtype=torch.bool,
        device=target_id.device,
    )
    return prediction[mask], text_embedding[mask], target_id[mask]


def compute_full_retrieval_metrics(
    eeg_embedding: torch.Tensor,
    text_embedding: torch.Tensor,
    target_id: torch.Tensor,
    candidate_limit: int | None = None,
    chunk_size: int = 1024,
) -> dict[str, float]:
    """Compute Meta-style full-set, macro, rank, and instance-aggregated metrics."""

    eeg_embedding, text_embedding, target_id = _limit_to_frequent_targets(
        eeg_embedding, text_embedding, target_id, candidate_limit
    )
    if eeg_embedding.shape[0] == 0:
        raise ValueError("Cannot evaluate retrieval on an empty set")

    ranks, candidate_count = retrieval_ranks(
        eeg_embedding, text_embedding, target_id, chunk_size=chunk_size
    )
    top1 = ranks < 1
    top10 = ranks < min(10, candidate_count)

    instance_prediction, instance_text, instance_target_id = _aggregate_predictions_by_target(
        eeg_embedding, text_embedding, target_id
    )
    instance_ranks, _ = retrieval_ranks(
        instance_prediction,
        instance_text,
        instance_target_id,
        chunk_size=chunk_size,
    )

    return {
        "top1": float(top1.to(torch.float32).mean()),
        "top10": float(top10.to(torch.float32).mean()),
        "macro_top1": float(_macro_mean(top1, target_id)),
        "macro_top10": float(_macro_mean(top10, target_id)),
        "mean_rank": float(ranks.mean()),
        "median_rank": float(ranks.median()),
        "macro_mean_rank": float(_macro_mean(ranks, target_id)),
        "macro_median_rank": float(_macro_median(ranks, target_id)),
        "instance_top1": float((instance_ranks < 1).to(torch.float32).mean()),
        "instance_top10": float(
            (instance_ranks < min(10, candidate_count)).to(torch.float32).mean()
        ),
        "instance_mean_rank": float(instance_ranks.mean()),
        "instance_median_rank": float(instance_ranks.median()),
        "candidate_count": float(candidate_count),
    }


def full_retrieval_topk(
    eeg_embedding: torch.Tensor,
    text_embedding: torch.Tensor,
    target_id: torch.Tensor,
    k: int,
    chunk_size: int = 1024,
) -> torch.Tensor:
    ranks, candidate_count = retrieval_ranks(
        eeg_embedding, text_embedding, target_id, chunk_size=chunk_size
    )
    return (ranks < min(k, candidate_count)).to(torch.float32).mean()
