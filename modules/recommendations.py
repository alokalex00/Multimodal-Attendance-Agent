"""Teaching recommendation engine for Multimodal Attendance & Engagement Agent.

Analyzes multimodal engagement trends, acoustic conditions, and attendance rates
to generate practical, context-aware pedagogical suggestions with exact timestamps.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class TeachingRecommendationEngine:
    """Derives actionable teaching suggestions based on multimodal session metrics."""

    def __init__(
        self,
        min_silence_duration_sec: float = 25.0,
        engagement_drop_threshold: float = 38.0,
        noise_warning_ratio: float = 0.35,
        low_presence_threshold: float = 50.0,
    ):
        """Initialize recommendation engine.

        Args:
            min_silence_duration_sec: Threshold for identifying extended lecture silence.
            engagement_drop_threshold: Engagement score below which a lull is flagged.
            noise_warning_ratio: Fraction of noisy chunks before alerting on acoustics.
            low_presence_threshold: Attendance percentage threshold for presence alerts.
        """
        self.min_silence_sec = min_silence_duration_sec
        self.drop_threshold = engagement_drop_threshold
        self.noise_ratio_thresh = noise_warning_ratio
        self.low_pres_thresh = low_presence_threshold

    def generate_recommendations(
        self,
        engagement_timeline: List[Dict[str, Any]],
        audio_summary: Dict[str, Any],
        attendance_summary: Dict[str, Any],
        session_duration_sec: float,
    ) -> List[Dict[str, Any]]:
        """Generate categorized, timestamp-aligned teaching suggestions.

        Args:
            engagement_timeline: List of timeline points with engagement scores.
            audio_summary: Aggregated audio metrics dictionary.
            attendance_summary: Aggregated attendance counts and rates.
            session_duration_sec: Session duration in seconds.

        Returns:
            List of structured recommendation dictionaries.
        """
        recs: List[Dict[str, Any]] = []
        rec_id = 1

        # ---------------------------------------------------------
        # 1. Detect Sustained Engagement Drops / Lulls
        # ---------------------------------------------------------
        lull_start: Optional[float] = None
        lull_scores: List[float] = []

        for item in engagement_timeline:
            score = item.get("engagement_score", 50.0)
            t = item.get("timestamp_sec", 0.0)
            tend = item.get("window_end_sec", t + 1.0)

            if score < self.drop_threshold:
                if lull_start is None:
                    lull_start = t
                    lull_scores = [score]
                else:
                    lull_scores.append(score)
            else:
                if lull_start is not None:
                    dur = t - lull_start
                    if dur >= 8.0:
                        recs.append(
                            {
                                "id": f"REC-{rec_id:03d}",
                                "category": "Interaction",
                                "severity": "Alert" if dur > 20.0 else "Suggestion",
                                "title": "Engagement Drop Detected — Re-engage Class",
                                "rationale": (
                                    f"Observed sustained low engagement (average score {sum(lull_scores)/len(lull_scores):.1f}/100) "
                                    f"lasting {dur:.1f} seconds. Recommend breaking the passive rhythm with a quick poll, "
                                    "an open discussion prompt, or a 2-minute think-pair-share exercise."
                                ),
                                "timestamp_start": float(lull_start),
                                "timestamp_end": float(t),
                            }
                        )
                        rec_id += 1
                    lull_start = None
                    lull_scores = []

        if lull_start is not None and engagement_timeline:
            last_t = engagement_timeline[-1].get("window_end_sec", session_duration_sec)
            dur = last_t - lull_start
            if dur >= 8.0:
                recs.append(
                    {
                        "id": f"REC-{rec_id:03d}",
                        "category": "Interaction",
                        "severity": "Alert",
                        "title": "Low Engagement Concluding the Session",
                        "rationale": (
                            f"Engagement remained low for the final {dur:.1f} seconds of class. "
                            "Consider ending lectures with a summary quiz or quick exit-ticket question to re-energize students."
                        ),
                        "timestamp_start": float(lull_start),
                        "timestamp_end": float(last_t),
                    }
                )
                rec_id += 1

        # ---------------------------------------------------------
        # 2. Acoustic Analysis: Excessive Noise or Audio Drop
        # ---------------------------------------------------------
        noise_ratio = audio_summary.get("ambient_noise_ratio", 0.0)
        has_audio = audio_summary.get("has_audio", False)

        if has_audio and noise_ratio >= self.noise_ratio_thresh:
            recs.append(
                {
                    "id": f"REC-{rec_id:03d}",
                    "category": "Acoustics",
                    "severity": "Alert",
                    "title": "Elevated Background Classroom Noise",
                    "rationale": (
                        f"Uncorrelated ambient acoustic noise was detected during {noise_ratio * 100:.1f}% of the session. "
                        "High background noise degrades participant speech intelligibility. "
                        "Check microphone distance, HVAC or window noise, or classroom echo."
                    ),
                    "timestamp_start": 0.0,
                    "timestamp_end": float(session_duration_sec),
                }
            )
            rec_id += 1

        silence_ratio = audio_summary.get("silence_ratio", 0.0)
        if has_audio and silence_ratio >= 0.70 and session_duration_sec > 15.0:
            recs.append(
                {
                    "id": f"REC-{rec_id:03d}",
                    "category": "Pedagogy",
                    "severity": "Suggestion",
                    "title": "Prolonged Silence / One-Way Monologue",
                    "rationale": (
                        f"Speech activity was absent during {silence_ratio * 100:.1f}% of the session. "
                        "If this was not an individual study or quiz period, encourage bilateral student participation, "
                        "Socratic questioning, or cold-calling to stimulate dialogue."
                    ),
                    "timestamp_start": 0.0,
                    "timestamp_end": float(session_duration_sec),
                }
            )
            rec_id += 1

        # ---------------------------------------------------------
        # 3. Attendance & Camera Visibility Check
        # ---------------------------------------------------------
        attendance_rate = attendance_summary.get("overall_attendance_rate", 100.0)
        total_p = attendance_summary.get("total_participants", 0)

        if total_p > 0 and attendance_rate < self.low_pres_thresh:
            recs.append(
                {
                    "id": f"REC-{rec_id:03d}",
                    "category": "Presence",
                    "severity": "Suggestion",
                    "title": "Sub-Optimal Participant Visibility or Camera Angle",
                    "rationale": (
                        f"Overall participant presence rate is {attendance_rate:.1f}%, below the target {self.low_pres_thresh:.0f}%. "
                        "Inspect whether participants are frequently leaving their seats, or if the camera field of view "
                        "has dead angles or poor lighting causing intermittent face occlusion."
                    ),
                    "timestamp_start": 0.0,
                    "timestamp_end": float(session_duration_sec),
                }
            )
            rec_id += 1

        # ---------------------------------------------------------
        # 4. Highlight Peak Engagement Periods (Reinforcement)
        # ---------------------------------------------------------
        high_periods = [item for item in engagement_timeline if item.get("engagement_score", 0.0) >= 75.0]
        if len(high_periods) >= 5:
            first_t = high_periods[0].get("timestamp_sec", 0.0)
            last_t = high_periods[-1].get("window_end_sec", first_t + 1.0)
            recs.append(
                {
                    "id": f"REC-{rec_id:03d}",
                    "category": "Pedagogy",
                    "severity": "Positive",
                    "title": "High Engagement Phase Identified",
                    "rationale": (
                        "Peak attentiveness and active dialogue were sustained during this phase. "
                        "Review the instructional strategies used here (e.g. demonstrations, live problem solving) "
                        "and reproduce them across other lecture segments."
                    ),
                    "timestamp_start": float(first_t),
                    "timestamp_end": float(last_t),
                }
            )
            rec_id += 1

        # Fallback if no specific issues triggered
        if not recs:
            recs.append(
                {
                    "id": "REC-001",
                    "category": "Pacing",
                    "severity": "Positive",
                    "title": "Consistent Session Balance Maintained",
                    "rationale": (
                        "Classroom engagement, attendance visibility, and audio metrics remained balanced "
                        "throughout the recorded session. Continue utilizing balanced lecture pacing."
                    ),
                    "timestamp_start": 0.0,
                    "timestamp_end": float(session_duration_sec),
                }
            )

        return recs
