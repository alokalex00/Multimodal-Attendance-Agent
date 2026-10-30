"""Unit tests for Vision Module and Attendance Logic."""

import os
import tempfile
import pytest
import numpy as np

from modules.vision import AttendanceAnalyzer, ParticipantTrack, VisionAnalyzer
from utils.video_utils import (
    draw_detection_overlay,
    generate_synthetic_classroom_video,
    get_video_metadata,
)


def test_participant_track_lifecycle():
    """Test ParticipantTrack initialization, update, displacement, and occlusion decay."""
    initial_bbox = (100, 150, 60, 80)
    track = ParticipantTrack(track_id=1, initial_bbox=initial_bbox, timestamp=0.0, frame_idx=0)

    assert track.track_id == 1
    assert track.presence_frames == 1
    assert track.consecutive_lost == 0
    assert track.centroid == (130.0, 190.0)

    # Simulate movement update
    next_bbox = (105, 152, 60, 80)
    track.update(next_bbox, timestamp=0.1, frame_idx=1)
    assert track.presence_frames == 2
    assert track.consecutive_lost == 0
    assert len(track.displacements) == 1

    # Simulate occlusion / missed frame
    track.mark_missed()
    assert track.consecutive_lost == 1
    assert track.recent_activity < 0.5


def test_attendance_analyzer():
    """Test attendance presence calculation and classification into Present, Partial, Absent."""
    analyzer = AttendanceAnalyzer(present_threshold_pct=70.0, partial_threshold_pct=35.0)

    sample_tracks = [
        {"track_id": 1, "presence_duration_sec": 8.0, "first_seen_sec": 0.0, "last_seen_sec": 9.0},   # 80% -> Present
        {"track_id": 2, "presence_duration_sec": 5.0, "first_seen_sec": 0.0, "last_seen_sec": 6.0},   # 50% -> Partial
        {"track_id": 3, "presence_duration_sec": 2.0, "first_seen_sec": 0.0, "last_seen_sec": 3.0},   # 20% -> Absent
    ]
    session_duration = 10.0

    res = analyzer.compute_attendance(sample_tracks, session_duration_sec=session_duration)

    assert res["total_participants"] == 3
    assert res["present_count"] == 1
    assert res["partial_count"] == 1
    assert res["absent_count"] == 1
    assert res["records"][0]["attendance_status"] == "Present"
    assert res["records"][1]["attendance_status"] == "Partial"
    assert res["records"][2]["attendance_status"] == "Absent"
    assert res["overall_attendance_rate"] == pytest.approx(50.0, 0.1)


def test_video_metadata_and_synthetic_generation():
    """Test generating synthetic video and verifying metadata properties."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_video_path = os.path.join(tmpdir, "test_classroom.mp4")
        generate_synthetic_classroom_video(test_video_path, duration_sec=2.0, fps=15, num_students=2)

        meta = get_video_metadata(test_video_path)
        assert meta["valid"] is True
        assert meta["duration"] >= 1.5
        assert meta["width"] == 640
        assert meta["height"] == 480


def test_draw_detection_overlay():
    """Test rendering visual bounding boxes and status labels on a frame."""
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    tracked_students = [
        {"track_id": 1, "bbox": (100, 100, 80, 100), "activity_level": 0.8},
        {"track_id": 2, "bbox": (300, 100, 80, 100), "activity_level": 0.2},
    ]
    annotated = draw_detection_overlay(blank_frame, tracked_students)
    assert annotated.shape == (480, 640, 3)
    # Check that overlay pixels were modified
    assert np.any(annotated > 0)


def test_vision_analyzer_nonexistent_video():
    """Test analyzer handles missing video gracefully."""
    analyzer = VisionAnalyzer()
    res = analyzer.analyze_video("nonexistent_video.mp4")
    assert res["valid"] is False
    assert "not found" in res["error"].lower()
