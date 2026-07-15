from .baseline import SubjectSpecificLinear, TemporalConvEEGEncoder, TemporalConvRegressor
from .simpleconv_timeagg import (
    BahdanauAttentionPooling,
    DilatedResidualBlock,
    SimpleConvTimeAggEEGEncoder,
    TemporalGLU,
)


def build_eeg_encoder(name: str = "temporal_conv", **kwargs):
    normalized_name = name.lower().replace("-", "_")
    if normalized_name in {"temporal_conv", "baseline"}:
        return TemporalConvEEGEncoder(**kwargs)
    if normalized_name in {"simpleconv_timeagg", "simpleconv_time_agg"}:
        return SimpleConvTimeAggEEGEncoder(**kwargs)
    raise ValueError(
        f"Unknown EEG encoder {name!r}; expected 'temporal_conv' or 'simpleconv_timeagg'"
    )

__all__ = [
    "BahdanauAttentionPooling",
    "build_eeg_encoder",
    "DilatedResidualBlock",
    "SimpleConvTimeAggEEGEncoder",
    "SubjectSpecificLinear",
    "TemporalConvEEGEncoder",
    "TemporalConvRegressor",
    "TemporalGLU",
]
