import csv

import numpy as np

from chineseeeg2_littleprince.data.manifest import attach_canonical_identities, load_manifest
from chineseeeg2_littleprince.data.splitting import deterministic_split_name, split_indices_by_group


def _write_manifest(tmp_path, embedding_path):
    path = tmp_path / "manifest.csv"
    rows = []
    for subject, embedding_idx in [("sub-01", 0), ("sub-02", 1), ("sub-03", 2)]:
        rows.append(
            {
                "subject": subject,
                "session": "ses-test",
                "task": "lis",
                "run": 1,
                "local_row_idx": embedding_idx,
                "global_row_idx": embedding_idx,
                "text_embedding_idx": embedding_idx,
                "label_id": embedding_idx,
                "start_time": 0,
                "stop_time": 1,
                "sfreq": 10,
                "start_sample": 0,
                "stop_sample": 10,
                "n_samples": 10,
                "eeg_vhdr_path": tmp_path / "unused.vhdr",
                "events_tsv_path": tmp_path / "unused.tsv",
                "text_embedding_path": embedding_path,
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_exact_duplicate_embeddings_share_target_and_split_group(tmp_path):
    embedding_path = tmp_path / "embeddings.npy"
    np.save(
        embedding_path,
        np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    records = attach_canonical_identities(load_manifest(_write_manifest(tmp_path, embedding_path)))

    assert records[0].target_uid == records[1].target_uid
    assert records[0].target_id == records[1].target_id
    assert records[0].split_group_id == records[1].split_group_id
    assert records[0].target_id != records[2].target_id
    assert len({record.instance_id for record in records}) == 3


def test_stable_group_split_keeps_duplicates_together_and_is_order_independent(tmp_path):
    embedding_path = tmp_path / "embeddings.npy"
    np.save(
        embedding_path,
        np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    records = attach_canonical_identities(load_manifest(_write_manifest(tmp_path, embedding_path)))
    train, val, test = split_indices_by_group(records, val_fraction=0.1, test_fraction=0.1, seed=42)
    assignment = {
        index: split
        for split, indices in [("train", train), ("val", val), ("test", test)]
        for index in indices
    }

    assert assignment[0] == assignment[1]
    for record in records:
        expected = deterministic_split_name(record.split_group_id, 0.1, 0.1, 42)
        assert assignment[records.index(record)] == expected

    reversed_assignment = {
        record.instance_id: deterministic_split_name(record.split_group_id, 0.1, 0.1, 42)
        for record in reversed(records)
    }
    assert reversed_assignment == {
        record.instance_id: assignment[index] for index, record in enumerate(records)
    }
