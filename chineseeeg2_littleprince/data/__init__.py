from .manifest import ManifestRecord, attach_canonical_identities, load_manifest
from .splitting import deterministic_split_name, split_indices_by_group

__all__ = [
    "EEGTextDataset",
    "ManifestRecord",
    "attach_canonical_identities",
    "collate_eeg_text",
    "deterministic_split_name",
    "load_manifest",
    "split_indices_by_group",
]


def __getattr__(name):
    if name == "EEGTextDataset":
        from .dataset import EEGTextDataset

        return EEGTextDataset
    if name == "collate_eeg_text":
        from .collate import collate_eeg_text

        return collate_eeg_text
    raise AttributeError(name)
