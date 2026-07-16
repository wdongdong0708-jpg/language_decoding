from pathlib import Path

from chineseeeg2_littleprince.data.dataset import EEGTextDataset
from chineseeeg2_littleprince.data.manifest import load_manifest, validate_manifest
from chineseeeg2_littleprince.io.brainvision import BrainVisionReader


def test_manifest_and_first_window():
    project_root = Path(__file__).resolve().parents[1]
    manifest = project_root / "data" / "manifests" / "littleprince_pl_sub08_manifest.csv"
    records = load_manifest(manifest)
    validate_manifest(records)
    assert len(records) == 2837
    assert records[0].text_embedding_idx == 16
    assert records[0].label_id == 16
    assert records[-1].text_embedding_idx == 2852

    reader = BrainVisionReader(records[0].eeg_vhdr_path)
    window = reader.read_window(records[0].start_sample, records[0].stop_sample)
    assert window.shape[0] == 128
    assert window.shape[1] == records[0].n_samples


def test_reading_aloud_manifest_and_first_window():
    project_root = Path(__file__).resolve().parents[1]
    manifest = (
        project_root
        / "data"
        / "manifests"
        / "littleprince_ra_all_clean_manifest.csv"
    )
    records = load_manifest(manifest)
    validate_manifest(records)

    assert len(records) == 11150
    assert {record.subject for record in records} == {
        "sub-f1",
        "sub-f2",
        "sub-m1",
        "sub-m2",
    }
    assert {record.task for record in records} == {"reading"}
    assert records[0].text_embedding_idx == 16
    assert records[-1].text_embedding_idx == 2852

    reader = BrainVisionReader(records[0].eeg_vhdr_path)
    window = reader.read_window(records[0].start_sample, records[0].stop_sample)
    assert window.shape == (128, records[0].n_samples)


def test_passive_listening_manifest_can_select_four_subjects():
    project_root = Path(__file__).resolve().parents[1]
    manifest = (
        project_root
        / "data"
        / "manifests"
        / "littleprince_pl_all_clean_manifest.csv"
    )
    dataset = EEGTextDataset(
        manifest,
        subjects=["sub-01", "sub-02", "sub-03", "sub-04"],
    )

    assert len(dataset) == 10390
    assert dataset.subject_to_id == {
        "sub-01": 0,
        "sub-02": 1,
        "sub-03": 2,
        "sub-04": 3,
    }
    assert len({record.target_id for record in dataset.records}) == 2603
