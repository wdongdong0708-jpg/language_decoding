from pathlib import Path

from chineseeeg2_littleprince.data.manifest import load_manifest, validate_manifest
from chineseeeg2_littleprince.io.brainvision import BrainVisionReader


def test_manifest_and_first_window():
    project_root = Path(__file__).resolve().parents[1]
    manifest = project_root / "data" / "manifests" / "littleprince_pl_sub08_manifest.csv"
    records = load_manifest(manifest)
    validate_manifest(records)
    assert len(records) == 2837
    assert records[0].text_embedding_idx == 16
    assert records[-1].text_embedding_idx == 2852

    reader = BrainVisionReader(records[0].eeg_vhdr_path)
    window = reader.read_window(records[0].start_sample, records[0].stop_sample)
    assert window.shape[0] == 128
    assert window.shape[1] == records[0].n_samples
