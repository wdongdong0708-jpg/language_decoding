import torch
from torch import nn

from chineseeeg2_littleprince.train import (
    MultiPositiveTargetBatchSampler,
    UniqueTargetBatchSampler,
    load_transfer_checkpoint,
    retrieval_topk,
)


def test_unique_target_sampler_keeps_every_row_without_batch_false_negatives():
    target_ids = [10, 10, 20, 20, 30]
    sampler = UniqueTargetBatchSampler(
        indices=list(range(len(target_ids))),
        target_ids=target_ids,
        batch_size=3,
        shuffle=False,
        seed=42,
    )

    batches = list(sampler)

    assert sorted(index for batch in batches for index in batch) == list(range(len(target_ids)))
    for batch in batches:
        assert len({target_ids[index] for index in batch}) == len(batch)


def test_batch_retrieval_accepts_any_occurrence_of_the_canonical_target():
    logits = torch.tensor(
        [
            [0.0, 2.0, 1.0],
            [2.0, 0.0, 1.0],
            [0.0, 1.0, 2.0],
        ]
    )
    target_id = torch.tensor([10, 10, 20])

    assert retrieval_topk(logits, target_id, k=1).item() == 1.0


def test_unique_target_sampler_filters_small_batches_and_reports_dropped_rows():
    target_ids = [10, 10, 10, 20, 20, 30]
    sampler = UniqueTargetBatchSampler(
        indices=list(range(len(target_ids))),
        target_ids=target_ids,
        batch_size=3,
        shuffle=False,
        seed=42,
        min_batch_size=2,
    )

    batches = list(sampler)

    assert [len(batch) for batch in batches] == [3, 2]
    assert sampler.retained_samples == 5
    assert sampler.dropped_samples == 1
    for batch in batches:
        assert len({target_ids[index] for index in batch}) == len(batch)


def test_shuffled_unique_target_batches_are_reproducible_but_change_by_epoch():
    target_ids = [value for value in range(8) for _ in range(2)]
    first = UniqueTargetBatchSampler(
        indices=list(range(len(target_ids))),
        target_ids=target_ids,
        batch_size=4,
        shuffle=True,
        seed=42,
    )
    second = UniqueTargetBatchSampler(
        indices=list(range(len(target_ids))),
        target_ids=target_ids,
        batch_size=4,
        shuffle=True,
        seed=42,
    )

    first_epoch = list(first)
    second_epoch = list(first)

    assert first_epoch == list(second)
    assert second_epoch == list(second)
    assert first_epoch != second_epoch


def test_multi_positive_sampler_yields_exact_target_view_groups():
    target_ids = [target for target in range(5) for _ in range(8)]
    sampler = MultiPositiveTargetBatchSampler(
        indices=list(range(len(target_ids))),
        target_ids=target_ids,
        targets_per_batch=2,
        views_per_target=4,
        shuffle=False,
        seed=42,
        view_groups_per_target_per_epoch=2,
        drop_last=True,
    )

    batches = list(sampler)

    assert len(batches) == 4
    assert all(len(batch) == 8 for batch in batches)
    for batch in batches:
        counts = {}
        for index in batch:
            counts[target_ids[index]] = counts.get(target_ids[index], 0) + 1
        assert sorted(counts.values()) == [4, 4]
    assert sampler.samples_per_epoch == 32
    assert sampler.target_groups_per_epoch == 8


def test_multi_positive_sampler_rotates_disjoint_views_for_eight_view_target():
    target_ids = [10] * 8 + [20] * 8
    sampler = MultiPositiveTargetBatchSampler(
        indices=list(range(16)),
        target_ids=target_ids,
        targets_per_batch=2,
        views_per_target=4,
        shuffle=False,
        seed=7,
        view_groups_per_target_per_epoch=2,
        drop_last=True,
    )

    first_round, second_round = list(sampler)

    for target_id in (10, 20):
        first = {index for index in first_round if target_ids[index] == target_id}
        second = {index for index in second_round if target_ids[index] == target_id}
        assert len(first) == len(second) == 4
        assert first.isdisjoint(second)


def test_multi_positive_sampler_rejects_targets_with_too_few_views():
    try:
        MultiPositiveTargetBatchSampler(
            indices=list(range(5)),
            target_ids=[10, 10, 10, 20, 20],
            targets_per_batch=2,
            views_per_target=3,
            shuffle=False,
            seed=42,
        )
    except ValueError as exc:
        assert "fewer than 3 views" in str(exc)
    else:
        raise AssertionError("Expected a target with too few views to fail")


class _TransferModel(nn.Module):
    def __init__(self, n_subjects: int):
        super().__init__()
        self.backbone = nn.Linear(3, 2)
        self.subject_layers = nn.Module()
        self.subject_layers.register_parameter(
            "weights", nn.Parameter(torch.zeros(n_subjects, 2, 2))
        )


def test_transfer_checkpoint_loads_shared_weights_and_keeps_target_subject_layers(tmp_path):
    source = _TransferModel(n_subjects=8)
    target = _TransferModel(n_subjects=4)
    with torch.no_grad():
        source.backbone.weight.fill_(3.0)
        source.backbone.bias.fill_(4.0)
        source.subject_layers.weights.fill_(5.0)
        target.subject_layers.weights.fill_(7.0)
    checkpoint_path = tmp_path / "source.pt"
    torch.save(
        {"epoch": 9, "model_state_dict": source.state_dict()},
        checkpoint_path,
    )

    report = load_transfer_checkpoint(
        target,
        checkpoint_path,
        exclude_prefixes=("subject_layers.",),
    )

    torch.testing.assert_close(target.backbone.weight, source.backbone.weight)
    torch.testing.assert_close(target.backbone.bias, source.backbone.bias)
    torch.testing.assert_close(
        target.subject_layers.weights,
        torch.full_like(target.subject_layers.weights, 7.0),
    )
    assert report["source_epoch"] == 9
    assert report["excluded_keys"] == ["subject_layers.weights"]
    assert report["target_initialized_keys"] == ["subject_layers.weights"]


def test_transfer_checkpoint_rejects_nonexcluded_shape_mismatch(tmp_path):
    source = _TransferModel(n_subjects=8)
    target = _TransferModel(n_subjects=4)
    checkpoint_path = tmp_path / "source.pt"
    state = source.state_dict()
    state["backbone.weight"] = torch.zeros(5, 5)
    torch.save({"model_state_dict": state}, checkpoint_path)

    try:
        load_transfer_checkpoint(
            target,
            checkpoint_path,
            exclude_prefixes=("subject_layers.",),
        )
    except RuntimeError as exc:
        assert "shape_mismatch" in str(exc)
        assert "backbone.weight" in str(exc)
    else:
        raise AssertionError("Expected a non-excluded shape mismatch to fail")


def test_transfer_checkpoint_rejects_unmatched_exclude_prefix(tmp_path):
    source = _TransferModel(n_subjects=4)
    target = _TransferModel(n_subjects=4)
    checkpoint_path = tmp_path / "source.pt"
    torch.save({"model_state_dict": source.state_dict()}, checkpoint_path)

    try:
        load_transfer_checkpoint(
            target,
            checkpoint_path,
            exclude_prefixes=("subject_layer.",),
        )
    except ValueError as exc:
        assert "matched no source keys" in str(exc)
    else:
        raise AssertionError("Expected an unmatched exclude prefix to fail")
