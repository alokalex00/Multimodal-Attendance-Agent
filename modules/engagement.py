"""Engagement scoring and multimodal fusion engine.

Implements:
1. EngagementScoringEngine: Combines visual presence, movement activity,
   audio speech activity, and noise penalties into a transparent 0-100 score.
2. Identifies peak engagement periods and lull periods with contextual explanations.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class EngagementScoringEngine:
    """Computes explainable classroom engagement scores from multimodal signals."""

    def __init__(
        self,
        weight_presence: float = 0.35,
        weight_visual_activity: float = 0.25,
        weight_audio_speech: float = 0.30,
        weight_baseline: float = 0.10,
        high_threshold: float = 70.0,
        moderate_threshold: float = 40.0,
    ):
        """Initialize scoring engine with configurable transparent weights.

        Args:
            weight_presence: Weight for participant presence ratio.
            weight_visual_activity: Weight for participant head/body movement activity.
            weight_audio_speech: Weight for active speech / discussion.
            weight_baseline: Baseline attentiveness expectation.
            high_threshold: Minimum score for 'High' engagement (default 70).
            moderate_threshold: Minimum score for 'Moderate' engagement (default 40).
        """
        self.w_presence = weight_presence
        self.w_vis_act = weight_visual_activity
        self.w_audio = weight_audio_speech
        self.w_base = weight_baseline
        self.high_threshold = high_threshold
        self.moderate_threshold = moderate_threshold

    def calculate_window_score(
        self,
        presence_rate: float,
        visual_activity: float,
        audio_speech_conf: float,
        noise_penalty: float = 0.0,
        has_audio: bool = True,
        has_video: bool = True,
    ) -> Tuple[float, str, str, Dict[str, float]]:
        """Calculate score and explanation for an aligned time window.

        Args:
            presence_rate: 0.0 to 1.0 (participants present / expected).
            visual_activity: 0.0 to 1.0 (motion / responsiveness).
            audio_speech_conf: 0.0 to 1.0 (active discussion confidence).
            noise_penalty: 0.0 to 1.0 (subtracted if excessive noise).
            has_audio: Flag if audio modality was available.
            has_video: Flag if video modality was available.

        Returns:
            (score_0_to_100, classification_level, explanation_text, factor_breakdown)
        """
        # Adapt weights if one modality is absent
        if not has_audio and has_video:
            w_pres = 0.55
            w_act = 0.35
            w_aud = 0.0
            w_base = 0.10
        elif not has_video and has_audio:
            w_pres = 0.0
            w_act = 0.0
            w_aud = 0.80
            w_base = 0.20
        else:
            w_pres = self.w_presence
            w_act = self.w_vis_act
            w_aud = self.w_audio
            w_base = self.w_base

        raw_score = (
            (w_pres * presence_rate)
            + (w_act * visual_activity)
            + (w_aud * audio_speech_conf)
            + w_base
            - (0.15 * noise_penalty)
        )

        score_100 = max(0.0, min(100.0, float(raw_score * 100.0)))

        # Classification
        if score_100 >= self.high_threshold:
            level = "High"
        elif score_100 >= self.moderate_threshold:
            level = "Moderate"
        else:
            level = "Low"

        # Generate human-readable explanation
        reasons = []
        if has_video:
            if presence_rate >= 0.8:
                reasons.append("Strong participant visibility")
            elif presence_rate <= 0.3:
                reasons.append("Low participant attendance in frame")

            if visual_activity >= 0.6:
                reasons.append("alert body/head responsiveness")
            elif visual_activity <= 0.2:
                reasons.append("passive/static posture")

        if has_audio:
            if audio_speech_conf >= 0.6:
                reasons.append("active dialogue/discussion")
            elif audio_speech_conf <= 0.1:
                reasons.append("extended lecture silence")
            if noise_penalty > 0.4:
                reasons.append("elevated background disturbance")

        explanation = "; ".join(reasons) if reasons else "Steady baseline classroom environment"

        factors = {
            "visual_presence_rate": float(presence_rate),
            "visual_activity_rate": float(visual_activity),
            "audio_speech_score": float(audio_speech_conf),
            "noise_penalty": float(noise_penalty),
        }

        return score_100, level, explanation, factors

    def compute_engagement_timeline(
        self,
        aligned_intervals: List[Dict[str, Any]],
        has_audio: bool = True,
        has_video: bool = True,
    ) -> Dict[str, Any]:
        """Compute complete engagement timeline, statistics, and critical periods.

        Args:
            aligned_intervals: List of synchronized time windows from MultimodalAgent.
            has_audio: Audio stream availability flag.
            has_video: Video stream availability flag.

        Returns:
            Dictionary containing timeline, average score, peak periods, and lull periods.
        """
        if not aligned_intervals:
            return {
                "average_score": 0.0,
                "overall_level": "Low",
                "timeline": [],
                "high_duration_sec": 0.0,
                "moderate_duration_sec": 0.0,
                "low_duration_sec": 0.0,
                "peaks": [],
                "lulls": [],
            }

        timeline: List[Dict[str, Any]] = []
        scores: List[float] = []

        high_sec = 0.0
        mod_sec = 0.0
        low_sec = 0.0

        for interval in aligned_intervals:
            dt = interval.get("duration_sec", 1.0)
            pres = interval.get("presence_rate", 0.0)
            vis_act = interval.get("visual_activity", 0.0)
            aud_speech = interval.get("speech_confidence", 0.0)
            noise_pen = 0.2 if interval.get("audio_state") == "Noise" else 0.0

            score, level, explanation, factors = self.calculate_window_score(
                presence_rate=pres,
                visual_activity=vis_act,
                audio_speech_conf=aud_speech,
                noise_penalty=noise_pen,
                has_audio=has_audio,
                has_video=has_video,
            )

            scores.append(score)
            if level == "High":
                high_sec += dt
            elif level == "Moderate":
                mod_sec += dt
            else:
                low_sec += dt

            timeline.append(
                {
                    "timestamp_sec": float(interval.get("timestamp_sec", 0.0)),
                    "window_end_sec": float(interval.get("window_end_sec", 0.0)),
                    "engagement_score": float(score),
                    "level": level,
                    "explanation": explanation,
                    "factors": factors,
                }
            )

        avg_score = float(np.mean(scores)) if scores else 0.0
        if avg_score >= self.high_threshold:
            overall_level = "High"
        elif avg_score >= self.moderate_threshold:
            overall_level = "Moderate"
        else:
            overall_level = "Low"

        # Detect peak engagement periods and low-activity lulls
        peaks = self._detect_contiguous_periods(timeline, condition=lambda s: s >= self.high_threshold, min_duration=2.0)
        lulls = self._detect_contiguous_periods(timeline, condition=lambda s: s < self.moderate_threshold, min_duration=3.0)

        return {
            "average_score": round(avg_score, 1),
            "overall_level": overall_level,
            "timeline": timeline,
            "high_duration_sec": round(high_sec, 1),
            "moderate_duration_sec": round(mod_sec, 1),
            "low_duration_sec": round(low_sec, 1),
            "peaks": peaks,
            "lulls": lulls,
        }

    @staticmethod
    def _detect_contiguous_periods(
        timeline: List[Dict[str, Any]],
        condition: Any,
        min_duration: float = 2.0,
    ) -> List[Dict[str, Any]]:
        """Identify contiguous periods satisfying a score condition."""
        periods: List[Dict[str, Any]] = []
        in_period = False
        start_t = 0.0
        period_scores: List[float] = []

        for item in timeline:
            score = item["engagement_score"]
            t = item["timestamp_sec"]
            tend = item["window_end_sec"]

            if condition(score):
                if not in_period:
                    in_period = True
                    start_t = t
                    period_scores = [score]
                else:
                    period_scores.append(score)
            else:
                if in_period:
                    dur = t - start_t
                    if dur >= min_duration:
                        periods.append(
                            {
                                "start_sec": float(start_t),
                                "end_sec": float(t),
                                "duration_sec": round(float(dur), 1),
                                "mean_score": round(float(np.mean(period_scores)), 1),
                            }
                        )
                    in_period = False
                    period_scores = []

        if in_period and timeline:
            dur = timeline[-1]["window_end_sec"] - start_t
            if dur >= min_duration:
                periods.append(
                    {
                        "start_sec": float(start_t),
                        "end_sec": float(timeline[-1]["window_end_sec"]),
                        "duration_sec": round(float(dur), 1),
                        "mean_score": round(float(np.mean(period_scores)), 1),
                    }
                )

        return periods
