from .manifest import ManifestRecord, load_manifest

__all__ = [
    "EEGSpeechDataset",
    "EEGSpeechSequenceDataset",
    "EEGTextDataset",
    "ManifestRecord",
    "collate_eeg_speech_sequence",
    "collate_eeg_text",
    "load_manifest",
]


def __getattr__(name):
    if name == "EEGTextDataset":
        from .dataset import EEGTextDataset

        return EEGTextDataset
    if name == "EEGSpeechDataset":
        from .speech_dataset import EEGSpeechDataset

        return EEGSpeechDataset
    if name == "EEGSpeechSequenceDataset":
        from .speech_sequence_dataset import EEGSpeechSequenceDataset

        return EEGSpeechSequenceDataset
    if name == "collate_eeg_speech_sequence":
        from .sequence_collate import collate_eeg_speech_sequence

        return collate_eeg_speech_sequence
    if name == "collate_eeg_text":
        from .collate import collate_eeg_text

        return collate_eeg_text
    raise AttributeError(name)
