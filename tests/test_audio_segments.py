from pathlib import Path

from chineseeeg2_littleprince.audio import AudioTimeline, littleprince_speaker_for_subject, read_xlsx_column


def test_littleprince_audio_timeline_maps_text_idx_to_wav_window():
    data_root = Path(r"D:\dataset\ChineseEEG-2")
    audio_root = data_root / "materials&embeddings" / "audio"
    timeline = AudioTimeline.from_directory("littleprince_f1", audio_root / "littleprince_f1")

    segment = timeline.segment_for_text_embedding(16, text="1")
    assert segment.audio_event_idx == 17
    assert segment.audio_file_path.name == "audio_1.wav"
    assert segment.audio_start_sample >= 0
    assert segment.audio_stop_sample > segment.audio_start_sample
    assert segment.n_audio_samples == segment.audio_stop_sample - segment.audio_start_sample
    assert segment.audio_sample_rate == 12000


def test_littleprince_subject_speaker_rule_and_xlsx_offset():
    data_root = Path(r"D:\dataset\ChineseEEG-2")
    values = read_xlsx_column(data_root / "materials&embeddings" / "audio" / "littleprince.xlsx")

    assert littleprince_speaker_for_subject("sub-01") == "littleprince_f1"
    assert littleprince_speaker_for_subject("sub-08") == "littleprince_m1"
    assert values[16 + 2] == "1"
