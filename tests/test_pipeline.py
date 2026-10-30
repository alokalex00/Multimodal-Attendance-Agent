"""End-to-end integration tests for Multimodal Attendance & Engagement Agent."""

import os
import tempfile
import pytest

from modules.multimodal_agent import MultimodalAgent
from utils.audio_utils import generate_synthetic_classroom_audio
from utils.report_utils import (
    generate_attendance_csv,
    generate_engagement_timeline_csv,
    generate_recommendations_csv,
)
from utils.storage import delete_session, list_saved_sessions, load_session, save_session
from utils.video_utils import generate_synthetic_classroom_video


def test_end_to_end_pipeline():
    """Test full multimodal pipeline with synthetic video and audio."""
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "test_class.mp4")
        audio_path = os.path.join(tmpdir, "test_class.wav")

        generate_synthetic_classroom_video(video_path, duration_sec=3.0, fps=10, num_students=2)
        generate_synthetic_classroom_audio(audio_path, duration_sec=3.0, sr=16000)

        agent = MultimodalAgent(present_threshold_pct=60.0, partial_threshold_pct=30.0, sample_fps=5.0)
        res = agent.process_session(video_path=video_path, audio_path=audio_path)

        assert res["success"] is True
        assert res["summary"]["session_duration_sec"] >= 2.5
        assert "attendance" in res
        assert "engagement" in res
        assert "recommendations" in res

        # Verify export utilities
        att_csv = generate_attendance_csv(res["attendance"]["records"])
        assert "Participant ID" in att_csv

        eng_csv = generate_engagement_timeline_csv(res["engagement"]["timeline"])
        assert "Engagement Score" in eng_csv

        recs_csv = generate_recommendations_csv(res["recommendations"])
        assert "Suggestion" in recs_csv

        # Test persistence
        session_id = save_session(res)
        assert session_id is not None

        loaded = load_session(session_id)
        assert loaded is not None
        assert loaded["summary"]["session_duration_sec"] == res["summary"]["session_duration_sec"]

        # Clean up
        delete_session(session_id)
