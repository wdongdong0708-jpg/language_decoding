from __future__ import annotations

import hashlib
import random
from typing import Any


def deterministic_split_name(
    group_id: str,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> str:
    """Assign a group to a stable train/val/test split, following Meta's protocol."""

    if val_fraction < 0 or test_fraction < 0:
        raise ValueError("val_fraction and test_fraction must be non-negative")
    if val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction + test_fraction must be less than 1")
    if not group_id:
        raise ValueError("split_group_id must be non-empty")

    hashed = int(hashlib.sha256(group_id.encode("utf-8")).hexdigest(), 16)
    score = random.Random(hashed + seed).random()
    train_fraction = 1.0 - val_fraction - test_fraction
    if score < train_fraction:
        return "train"
    if score < train_fraction + val_fraction:
        return "val"
    return "test"


def split_indices_by_group(
    records: list[Any],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    """Split records by their stable split_group_id without group leakage."""

    indices = {"train": [], "val": [], "test": []}
    assignment: dict[str, str] = {}
    for index, record in enumerate(records):
        group_id = str(record.split_group_id)
        split = assignment.setdefault(
            group_id,
            deterministic_split_name(group_id, val_fraction, test_fraction, seed),
        )
        indices[split].append(index)
    return indices["train"], indices["val"], indices["test"]
