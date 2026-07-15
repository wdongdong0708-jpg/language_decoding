from __future__ import annotations

from pathlib import Path

import numpy as np


class KitMEGReader:
    """Lazy KIT MEG window reader for Gwilliams-style BIDS files."""

    def __init__(self, con_path: str | Path, meg_only: bool = True):
        try:
            import mne
        except ImportError as exc:
            raise ImportError("Reading KIT MEG files requires mne.") from exc

        self.con_path = Path(con_path)
        self.raw = mne.io.read_raw_kit(self.con_path, preload=False, verbose="ERROR")
        if meg_only:
            self.raw.pick_types(
                meg=True,
                ref_meg=False,
                misc=False,
                stim=False,
                eeg=False,
                eog=False,
                ecg=False,
            )
        self.sfreq = float(self.raw.info["sfreq"])
        self.n_channels = len(self.raw.ch_names)
        self.n_samples = int(self.raw.n_times)

    def read_window(self, start_sample: int, stop_sample: int) -> np.ndarray:
        if start_sample < 0 or stop_sample <= start_sample or stop_sample > self.n_samples:
            raise ValueError(
                f"Invalid MEG window {start_sample}:{stop_sample} for length {self.n_samples}"
            )
        data = self.raw.get_data(start=int(start_sample), stop=int(stop_sample))
        return np.asarray(data, dtype=np.float32)
