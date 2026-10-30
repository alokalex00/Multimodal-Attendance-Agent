"""Unit tests for Engagement Scoring and Signal Fusion."""

import pytest

from modules.engagement import EngagementScoringEngine


def test_engagement_score_calculation():
    """Test rule-based fusion, score ranges, and factor explanations."""
    engine = EngagementScoringEngine(
        weight_presence=0.35,
        weight_visual_activity=0.25,
        weight_audio_speech=0.30,
        weight_baseline=0.10,
        high_threshold=70.0,
        moderate_threshold=40.0,
    )

    # Case 1: High engagement (high presence, active speech, active posture)
    score, level, expl, factors = engine.calculate_window_score(
        presence_rate=0.9,
        visual_activity=0.8,
        audio_speech_conf=0.85,
        noise_penalty=0.0,
    )
    assert 70.0 <= score <= 100.0
    assert level == "High"
    assert "presence" in expl.lower() or "dialogue" in expl.lower()

    # Case 2: Low engagement (low presence, silence, static)
    score_low, level_low, expl_low, _ = engine.calculate_window_score(
        presence_rate=0.1,
        visual_activity=0.1,
        audio_speech_conf=0.0,
        noise_penalty=0.5,
    )
    assert score_low < 40.0
    assert level_low == "Low"


def test_missing_modality_adaptation():
    """Test engine adapts weights when only video or only audio is present."""
    engine = EngagementScoringEngine()

    # Video only (no audio)
    score_v, _, _, factors_v = engine.calculate_window_score(
        presence_rate=0.8,
        visual_activity=0.7,
        audio_speech_conf=0.0,
        has_audio=False,
        has_video=True,
    )
    assert 0 <= score_v <= 100

    # Audio only (no video)
    score_a, _, _, factors_a = engine.calculate_window_score(
        presence_rate=0.0,
        visual_activity=0.0,
        audio_speech_conf=0.8,
        has_audio=True,
        has_video=False,
    )
    assert 0 <= score_a <= 100


def test_timeline_and_period_detection():
    """Test computing full engagement timeline with peak and lull detection."""
    engine = EngagementScoringEngine()

    intervals = [
        {"duration_sec": 1.0, "timestamp_sec": float(i), "window_end_sec": float(i + 1),
         "presence_rate": 0.9, "visual_activity": 0.8, "speech_confidence": 0.9}
        for i in range(5)
    ]
    # Add a lull
    intervals.extend([
        {"duration_sec": 1.0, "timestamp_sec": float(i), "window_end_sec": float(i + 1),
         "presence_rate": 0.1, "visual_activity": 0.1, "speech_confidence": 0.0}
        for i in range(5, 10)
    ])

    res = engine.compute_engagement_timeline(intervals)
    assert res["average_score"] > 0
    assert len(res["timeline"]) == 10
    assert res["high_duration_sec"] >= 4.0
    assert res["low_duration_sec"] >= 4.0
    assert len(res["peaks"]) >= 1
    assert len(res["lulls"]) >= 1
