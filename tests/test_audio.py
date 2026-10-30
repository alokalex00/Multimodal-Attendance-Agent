"""Unit tests for Audio Module and Acoustic Processing."""

import os
import tempfile
import pytest
import numpy as np

from modules.audio import AudioAnalyzer
from utils.audio_utils import (
    generate_synthetic_classroom_audio,
    load_and_resample_audio,
)


def test_synthetic_audio_generation_and_loading():
    """Test generating synthetic classroom audio and loading as 16kHz mono."""
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "test_audio.wav")
        generate_synthetic_classroom_audio(wav_path, duration_sec=3.0, sr=16000)

        data, sr, meta = load_and_resample_audio(wav_path, target_sr=16000)
        assert meta["valid"] is True
        assert sr == 16000
        assert data is not None
        assert len(data) == pytest.approx(3 * 16000, abs=500)
        assert data.ndim == 1  # Standardized to mono
        assert np.max(np.abs(data)) <= 1.0


def test_audio_analyzer_features():
    """Test chunking, decibel calculation, noise floor, and speech/silence detection."""
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "test_audio.wav")
        generate_synthetic_classroom_audio(wav_path, duration_sec=4.0, sr=16000)

        analyzer = AudioAnalyzer(sample_rate=16000, chunk_duration_sec=1.0)
        res = analyzer.process_video_or_audio(audio_path=wav_path)

        assert res["has_audio"] is True
        assert res["total_chunks"] >= 3
        assert len(res["timeline"]) >= 3
        assert res["noise_floor_db"] < 0
        assert res["speech_ratio"] + res["silence_ratio"] + res["ambient_noise_ratio"] == pytest.approx(1.0, abs=0.05)

        # Inspect chunk properties
        c0 = res["timeline"][0]
        assert "rms_energy" in c0
        assert "decibels" in c0
        assert "state" in c0
        assert c0["state"] in ["Speech", "Silence", "Ambient", "Noise"]


def test_audio_analyzer_missing_input():
    """Test handling when no audio or video is provided."""
    analyzer = AudioAnalyzer()
    res = analyzer.process_video_or_audio(video_path=None, audio_path=None)
    assert res["has_audio"] is False
    assert res["duration_sec"] == 0.0
    assert len(res["timeline"]) == 0


def test_audio_empty_file():
    """Test handling of 0-byte corrupted file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        empty_file = os.path.join(tmpdir, "empty.wav")
        with open(empty_file, "wb") as f:
            pass

        data, sr, meta = load_and_resample_audio(empty_file)
        assert meta["valid"] is False
        assert "empty" in meta["error"].lower()
