from __future__ import annotations

import math
from pathlib import Path

import numpy as np


class OfficialSpeechEmbedder:
    """Sentence-level audio embeddings following the dataset's official recipe."""

    def __init__(
        self,
        model_name: str = "airesearch/wav2vec2-large-xlsr-53-th",
        device: str | None = None,
    ):
        try:
            import torch
            from transformers import Wav2Vec2Model, Wav2Vec2Processor
        except ImportError as exc:
            raise ImportError(
                "Speech embedding generation needs torch and transformers. "
                "Use the bm5060 conda environment or install the missing packages."
            ) from exc

        self.torch = torch
        self.model_name = model_name
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def embed_file_segment(
        self,
        audio_path: str | Path,
        start_sample: int,
        stop_sample: int,
    ) -> np.ndarray:
        try:
            import soundfile as sf
        except ImportError as exc:
            raise ImportError(
                "Speech embedding generation needs soundfile. "
                "Use the bm5060 conda environment or install soundfile."
            ) from exc

        if stop_sample <= start_sample:
            raise ValueError(f"Invalid audio sample window: {start_sample}:{stop_sample}")

        waveform, sample_rate = sf.read(
            str(audio_path),
            start=int(start_sample),
            stop=int(stop_sample),
            dtype="float32",
            always_2d=False,
        )
        return self.embed_waveform(waveform, sample_rate)

    def embed_waveform(self, waveform, sample_rate: int) -> np.ndarray:
        waveform = self._as_mono_numpy(waveform)

        if waveform.size == 0:
            raise ValueError("Cannot embed an empty audio segment")

        if sample_rate != 16000:
            try:
                from scipy.signal import resample_poly
            except ImportError as exc:
                raise ImportError(
                    "Speech embedding generation needs scipy for sample-rate conversion."
                ) from exc

            common = math.gcd(int(sample_rate), 16000)
            waveform = resample_poly(
                waveform,
                up=16000 // common,
                down=int(sample_rate) // common,
            ).astype(np.float32, copy=False)

        inputs = self.processor(waveform, return_tensors="pt", sampling_rate=16000)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.no_grad():
            outputs = self.model(**inputs)
# embedding池化？
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0).detach().cpu().numpy()
        return np.asarray(embedding, dtype=np.float32)

    def _as_mono_numpy(self, waveform) -> np.ndarray:
        if self.torch.is_tensor(waveform):
            waveform = waveform.detach().cpu().numpy()
        waveform = np.asarray(waveform, dtype=np.float32)

        if waveform.ndim == 1:
            return waveform
        if waveform.ndim != 2:
            raise ValueError(f"Expected mono or stereo waveform, got shape={tuple(waveform.shape)}")

        if waveform.shape[0] <= 8 and waveform.shape[1] > waveform.shape[0]:
            return waveform.mean(axis=0)
        return waveform.mean(axis=1)
