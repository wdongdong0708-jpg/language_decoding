from scripts.build_fixed_onset_manifest import fixed_onset_row


def test_fixed_onset_row_ignores_original_rowe_endpoint():
    original = {
        "sfreq": "250.000000",
        "start_time": "4.000000",
        "stop_time": "4.124000",
        "start_sample": "1000",
        "stop_sample": "1031",
        "n_samples": "31",
        "eeg_vhdr_path": "recording.vhdr",
    }

    converted = fixed_onset_row(
        original,
        window_seconds=0.7,
        eeg_sfreq=250.0,
        eeg_n_samples=10_000,
    )

    assert converted["start_sample"] == "1000"
    assert converted["stop_sample"] == "1175"
    assert converted["n_samples"] == "175"
    assert converted["start_time"] == "4.000000"
    assert converted["stop_time"] == "4.700000"
    assert original["stop_sample"] == "1031"


def test_fixed_onset_row_rounds_duration_using_recording_sampling_rate():
    original = {
        "sfreq": "256.000000",
        "start_time": "1.000000",
        "stop_time": "5.000000",
        "start_sample": "256",
        "stop_sample": "1280",
        "n_samples": "1024",
        "eeg_vhdr_path": "recording.vhdr",
    }

    converted = fixed_onset_row(
        original,
        window_seconds=0.7,
        eeg_sfreq=256.0,
        eeg_n_samples=10_000,
    )

    assert converted["stop_sample"] == "435"
    assert converted["n_samples"] == "179"
