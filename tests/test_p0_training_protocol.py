import torch

from chineseeeg2_littleprince.train import UniqueTargetBatchSampler, retrieval_topk


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
