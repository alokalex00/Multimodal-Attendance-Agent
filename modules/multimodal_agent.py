"""Multimodal Agent orchestrating Vision, Audio, Engagement, and Recommendations.

Integrates VisionAnalyzer and AudioAnalyzer, synchronizes temporal signals,
handles missing or single-modality inputs, and executes full end-to-end analysis.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from modules.audio import AudioAnalyzer
from modules.engagement import EngagementScoringEngine
from modules.recommendations import TeachingRecommendationEngine
from modules.vision import AttendanceAnalyzer, VisionAnalyzer


class MultimodalAgent:
    """End-to-end multimodal agent coordinating vision and audio perception engines."""

    def __init__(
        self,
        present_threshold_pct: float = 70.0,
        partial_threshold_pct: float = 35.0,
        sample_fps: float = 10.0,
        window_duration_sec: float = 1.0,
    ):
        """Initialize MultimodalAgent.

        Args:
            present_threshold_pct: Presence percentage for Present status.
            partial_threshold_pct: Presence percentage for Partial status.
            sample_fps: Video frame sampling rate.
            window_duration_sec: Time synchronization slice duration in seconds.
        """
        self.window_duration_sec = window_duration_sec
        self.vision_analyzer = VisionAnalyzer(sample_fps=sample_fps)
        self.attendance_analyzer = AttendanceAnalyzer(
            present_threshold_pct=present_threshold_pct,
            partial_threshold_pct=partial_threshold_pct,
        )
        self.audio_analyzer = AudioAnalyzer(chunk_duration_sec=window_duration_sec)
        self.engagement_engine = EngagementScoringEngine()
        self.recommendation_engine = TeachingRecommendationEngine()

    def process_session(
        self,
        video_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        max_duration: Optional[float] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, Any]:
        """Execute complete multimodal analysis on classroom inputs.

        Args:
            video_path: Optional path to classroom video file.
            audio_path: Optional path to separate audio file.
            max_duration: Optional limit on session duration (seconds).
            progress_callback: Callback(fraction: float, message: str).

        Returns:
            Unified dictionary containing metadata, vision tracking, attendance,
            audio features, engagement timeline, and teaching recommendations.
        """
        start_time = time.time()

        if not video_path and not audio_path:
            return {
                "success": False,
                "error": "No video or audio input provided",
                "summary": {},
            }

        def report_progress(val: float, msg: str) -> None:
            if progress_callback:
                progress_callback(val, msg)

        # -------------------------------------------------------------
        # STEP 1: Vision Perception (if video provided)
        # -------------------------------------------------------------
        has_video = bool(video_path and os.path.exists(video_path))
        vision_result: Dict[str, Any] = {"valid": False, "tracks": [], "frame_timeline": []}

        if has_video:
            report_progress(0.10, "Perceiving visual signals: Face detection and participant tracking...")
            vision_result = self.vision_analyzer.analyze_video(
                video_path=video_path,  # type: ignore
                max_duration=max_duration,
                progress_callback=lambda p, s: report_progress(0.10 + 0.35 * p, s),
            )
        else:
            report_progress(0.10, "No video stream provided; skipping visual processing...")

        # -------------------------------------------------------------
        # STEP 2: Audio Perception (extracted from video or standalone)
        # -------------------------------------------------------------
        report_progress(0.50, "Extracting and analyzing acoustic signals...")
        audio_result = self.audio_analyzer.process_video_or_audio(
            video_path=video_path if has_video else None,
            audio_path=audio_path,
            max_duration=max_duration,
        )
        has_audio = audio_result.get("has_audio", False)

        # -------------------------------------------------------------
        # STEP 3: Temporal Alignment
        # -------------------------------------------------------------
        report_progress(0.70, "Synchronizing multimodal time bases...")

        # Determine effective session duration
        dur_v = vision_result.get("session_duration_sec", 0.0) if has_video else 0.0
        dur_a = audio_result.get("duration_sec", 0.0) if has_audio else 0.0
        effective_duration = max(dur_v, dur_a)
        if effective_duration <= 0.0:
            effective_duration = 1.0

        aligned_intervals = self._align_multimodal_intervals(
            vision_result=vision_result,
            audio_result=audio_result,
            duration_sec=effective_duration,
        )

        # -------------------------------------------------------------
        # STEP 4: Attendance Analytics (Vision-based)
        # -------------------------------------------------------------
        report_progress(0.80, "Calculating attendance presence durations...")
        attendance_result = self.attendance_analyzer.compute_attendance(
            tracks=vision_result.get("tracks", []),
            session_duration_sec=effective_duration,
        )

        # -------------------------------------------------------------
        # STEP 5: Engagement Scoring & Rule-Based Fusion
        # -------------------------------------------------------------
        report_progress(0.88, "Computing fused engagement indicators and timelines...")
        engagement_result = self.engagement_engine.compute_engagement_timeline(
            aligned_intervals=aligned_intervals,
            has_audio=has_audio,
            has_video=has_video,
        )

        # -------------------------------------------------------------
        # STEP 6: Teaching Recommendations
        # -------------------------------------------------------------
        report_progress(0.95, "Synthesizing teaching recommendations...")
        recommendations = self.recommendation_engine.generate_recommendations(
            engagement_timeline=engagement_result.get("timeline", []),
            audio_summary=audio_result,
            attendance_summary=attendance_result,
            session_duration_sec=effective_duration,
        )

        elapsed_sec = time.time() - start_time
        report_progress(1.0, "Multimodal session analysis complete!")

        # Unified session dictionary
        return {
            "success": True,
            "error": None,
            "processing_time_sec": round(elapsed_sec, 2),
            "metadata": {
                "has_video": has_video,
                "has_audio": has_audio,
                "video_path": video_path,
                "audio_path": audio_path,
                "video_filename": os.path.basename(video_path) if video_path else None,
                "audio_filename": os.path.basename(audio_path) if audio_path else None,
                "video_metadata": vision_result.get("metadata", {}),
            },
            "summary": {
                "session_duration_sec": round(effective_duration, 1),
                "total_participants": attendance_result.get("total_participants", 0),
                "present_count": attendance_result.get("present_count", 0),
                "attendance_rate": attendance_result.get("overall_attendance_rate", 0.0),
                "average_engagement_score": engagement_result.get("average_score", 0.0),
                "engagement_level": engagement_result.get("overall_level", "Moderate"),
                "high_engagement_duration_sec": engagement_result.get("high_duration_sec", 0.0),
                "moderate_engagement_duration_sec": engagement_result.get("moderate_duration_sec", 0.0),
                "low_engagement_duration_sec": engagement_result.get("low_duration_sec", 0.0),
            },
            "attendance": attendance_result,
            "vision": {
                "tracks": vision_result.get("tracks", []),
                "keyframes": vision_result.get("keyframes", []),
                "total_detected": vision_result.get("total_participants_detected", 0),
            },
            "audio_summary": {
                "has_audio": has_audio,
                "speech_ratio": audio_result.get("speech_ratio", 0.0),
                "silence_ratio": audio_result.get("silence_ratio", 0.0),
                "noise_floor_db": audio_result.get("noise_floor_db", -60.0),
                "average_decibels": audio_result.get("average_decibels", -60.0),
            },
            "engagement": engagement_result,
            "recommendations": recommendations,
        }

    def _align_multimodal_intervals(
        self,
        vision_result: Dict[str, Any],
        audio_result: Dict[str, Any],
        duration_sec: float,
    ) -> List[Dict[str, Any]]:
        """Slice session into uniform time windows and map vision + audio metrics."""
        step = self.window_duration_sec
        num_windows = max(1, int(math.ceil(duration_sec / step)))

        v_timeline = vision_result.get("frame_timeline", [])
        total_unique_tracks = max(1, vision_result.get("total_participants_detected", 1))

        # Audio chunk lookup by chunk_idx
        a_chunks = audio_result.get("timeline", [])
        a_chunk_map = {c["chunk_idx"]: c for c in a_chunks}

        aligned: List[Dict[str, Any]] = []

        for w_idx in range(num_windows):
            t_start = w_idx * step
            t_end = min(duration_sec, (w_idx + 1) * step)
            dt = max(0.1, t_end - t_start)

            # Find vision frames within [t_start, t_end)
            frames_in_win = [f for f in v_timeline if t_start <= f["timestamp_sec"] < t_end]

            if frames_in_win:
                mean_vis_count = float(np.mean([f["visible_count"] for f in frames_in_win]))
                presence_rate = min(1.0, mean_vis_count / float(total_unique_tracks))
                mean_vis_act = float(np.mean([f["mean_activity"] for f in frames_in_win]))
            else:
                presence_rate = 0.0
                mean_vis_act = 0.0

            # Find matching audio chunk
            a_chunk = a_chunk_map.get(w_idx)
            if a_chunk:
                aud_state = a_chunk["state"]
                aud_speech_conf = a_chunk["speech_confidence"]
                aud_db = a_chunk["decibels"]
            else:
                aud_state = "Silence"
                aud_speech_conf = 0.0
                aud_db = -60.0

            aligned.append(
                {
                    "window_idx": w_idx,
                    "timestamp_sec": float(t_start),
                    "window_end_sec": float(t_end),
                    "duration_sec": float(dt),
                    "presence_rate": float(presence_rate),
                    "visual_activity": float(mean_vis_act),
                    "audio_state": aud_state,
                    "speech_confidence": float(aud_speech_conf),
                    "decibels": float(aud_db),
                }
            )

        return aligned
