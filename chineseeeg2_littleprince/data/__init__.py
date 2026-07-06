from .manifest import ManifestRecord, load_manifest

__all__ = ["EEGTextDataset", "ManifestRecord", "collate_eeg_text", "load_manifest"]


def __getattr__(name):
    if name == "EEGTextDataset":
        from .dataset import EEGTextDataset

        return EEGTextDataset
    if name == "collate_eeg_text":
        from .collate import collate_eeg_text

        return collate_eeg_text
    raise AttributeError(name)
