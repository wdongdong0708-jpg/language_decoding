from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


_DTYPE_BY_FORMAT = {
    "IEEE_FLOAT_32": np.dtype("<f4"),
}


@dataclass(frozen=True)
class BrainVisionInfo:
    vhdr_path: Path
    data_file: Path
    marker_file: Path | None
    n_channels: int
    sfreq: float
    dtype: np.dtype
    orientation: str


def _parse_key_values(vhdr_path: Path) -> dict[str, str]:
    text = vhdr_path.read_text(encoding="utf-8", errors="replace")
    values: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(";") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def parse_vhdr(vhdr_path: str | Path) -> BrainVisionInfo:
    path = Path(vhdr_path)
    values = _parse_key_values(path)

    binary_format = values.get("BinaryFormat")
    if binary_format not in _DTYPE_BY_FORMAT:
        raise ValueError(f"Unsupported BrainVision BinaryFormat={binary_format!r} in {path}")

    orientation = values.get("DataOrientation")
    if orientation != "MULTIPLEXED":
        raise ValueError(f"Unsupported BrainVision DataOrientation={orientation!r} in {path}")

    sampling_interval_us = float(values["SamplingInterval"])
    sfreq = 1_000_000.0 / sampling_interval_us
    marker_name = values.get("MarkerFile")

    return BrainVisionInfo(
        vhdr_path=path,
        data_file=path.with_name(values["DataFile"]),
        marker_file=path.with_name(marker_name) if marker_name else None,
        n_channels=int(values["NumberOfChannels"]),
        sfreq=sfreq,
        dtype=_DTYPE_BY_FORMAT[binary_format],
        orientation=orientation,
    )


class BrainVisionReader:
    """Small reader for pybv-style BrainVision EEG files.

    It supports the format used by ChineseEEG-2 derivatives:
    float32, multiplexed, channels interleaved by sample.
    """

    def __init__(self, vhdr_path: str | Path):
        self.info = parse_vhdr(vhdr_path)
        if not self.info.data_file.exists():
            raise FileNotFoundError(self.info.data_file)
        self._flat = np.memmap(self.info.data_file, dtype=self.info.dtype, mode="r")
        if self._flat.size % self.info.n_channels != 0:
            raise ValueError(
                f"{self.info.data_file} has {self._flat.size} values, not divisible by "
                f"{self.info.n_channels} channels"
            )
        self._data = self._flat.reshape((-1, self.info.n_channels))

    @property
    def n_samples(self) -> int:
        return int(self._data.shape[0])

    @property
    def n_channels(self) -> int:
        return self.info.n_channels

    @property
    def sfreq(self) -> float:
        return self.info.sfreq

    def read_window(self, start_sample: int, stop_sample: int) -> np.ndarray:
        if start_sample < 0:
            raise ValueError(f"start_sample must be non-negative, got {start_sample}")
        if stop_sample <= start_sample:
            raise ValueError(f"stop_sample must be greater than start_sample, got {stop_sample}")
        if stop_sample > self.n_samples:
            raise ValueError(f"stop_sample={stop_sample} exceeds n_samples={self.n_samples}")
        return np.asarray(self._data[start_sample:stop_sample, :].T, dtype=np.float32)
