from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SphericalKMeansResult:
    labels: np.ndarray
    centers: np.ndarray
    objective: float
    iterations: int
    init_seed: int


def _kmeans_plus_plus(
    x: torch.Tensor,
    k: int,
    weights: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    first = torch.multinomial(weights / weights.sum(), 1, generator=generator)
    selected = [int(first.item())]
    closest_distance = (1.0 - x @ x[first].T).squeeze(1).clamp_min(0.0)

    for _ in range(1, k):
        probabilities = weights * closest_distance
        total = probabilities.sum()
        if float(total) <= 0:
            available = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
            available[selected] = False
            candidates = torch.where(available)[0]
            next_offset = torch.randint(
                len(candidates), (1,), generator=generator, device=x.device
            )
            next_index = candidates[next_offset]
        else:
            next_index = torch.multinomial(
                probabilities / total, 1, generator=generator
            )
        selected.append(int(next_index.item()))
        distance = (1.0 - x @ x[next_index].T).squeeze(1).clamp_min(0.0)
        closest_distance = torch.minimum(closest_distance, distance)

    return x[torch.tensor(selected, dtype=torch.long, device=x.device)].clone()


def spherical_kmeans(
    vectors: np.ndarray,
    k: int,
    *,
    weights: np.ndarray | None = None,
    n_init: int = 8,
    max_iter: int = 100,
    seed: int = 42,
    device: str | torch.device | None = None,
) -> SphericalKMeansResult:
    """Run cosine/spherical K-means and retain the best seeded restart."""

    if vectors.ndim != 2:
        raise ValueError(f"vectors must be 2D, got {vectors.shape}")
    if not 1 <= k <= vectors.shape[0]:
        raise ValueError(f"k must be in [1, {vectors.shape[0]}], got {k}")
    if n_init <= 0 or max_iter <= 0:
        raise ValueError("n_init and max_iter must be positive")

    if device is None or str(device) == "auto":
        resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved_device = torch.device(device)

    x = torch.as_tensor(
        np.ascontiguousarray(vectors, dtype=np.float32), device=resolved_device
    )
    norms = torch.linalg.vector_norm(x, dim=1)
    if torch.any(norms <= 0):
        raise ValueError("Spherical K-means cannot cluster zero vectors")
    x = F.normalize(x, dim=1)

    if weights is None:
        weight_tensor = torch.ones(x.shape[0], dtype=x.dtype, device=x.device)
    else:
        if weights.shape != (vectors.shape[0],):
            raise ValueError(f"weights must have shape {(vectors.shape[0],)}, got {weights.shape}")
        weight_tensor = torch.as_tensor(weights, dtype=x.dtype, device=x.device)
        if torch.any(weight_tensor <= 0):
            raise ValueError("All weights must be positive")

    best: SphericalKMeansResult | None = None
    for restart in range(n_init):
        init_seed = seed + restart
        generator = torch.Generator(device=resolved_device)
        generator.manual_seed(init_seed)
        centers = _kmeans_plus_plus(x, k, weight_tensor, generator)
        previous_labels = None
        iterations = 0

        for iteration in range(1, max_iter + 1):
            similarities = x @ centers.T
            labels = similarities.argmax(dim=1)
            iterations = iteration

            assignments = F.one_hot(labels, num_classes=k).to(x.dtype).T
            weighted_assignments = assignments * weight_tensor.unsqueeze(0)
            sums = weighted_assignments @ x
            cluster_weights = weighted_assignments.sum(dim=1)

            empty = cluster_weights <= 0
            if torch.any(empty):
                best_similarity = similarities.max(dim=1).values
                replacements = torch.argsort(best_similarity)[: int(empty.sum())]
                sums[empty] = x[replacements]

            centers = F.normalize(sums, dim=1)
            if previous_labels is not None and torch.equal(labels, previous_labels):
                break
            previous_labels = labels

        final_similarities = x @ centers.T
        final_labels = final_similarities.argmax(dim=1)
        selected_similarity = final_similarities.gather(1, final_labels[:, None]).squeeze(1)
        objective = float((selected_similarity * weight_tensor).sum().detach().cpu())
        result = SphericalKMeansResult(
            labels=final_labels.detach().cpu().numpy().astype(np.int64, copy=False),
            centers=centers.detach().cpu().numpy().astype(np.float32, copy=False),
            objective=objective,
            iterations=iterations,
            init_seed=init_seed,
        )
        if best is None or result.objective > best.objective:
            best = result

    assert best is not None
    return best
