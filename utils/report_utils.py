"""Report generation utilities for Multimodal Attendance & Engagement Agent.

Generates CSV tables, comprehensive JSON summaries, and formatted text/markdown
reports for export and presentation.
"""

from __future__ import annotations

import io
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def generate_attendance_dataframe(attendance_records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert participant attendance records to a clean formatted DataFrame."""
    if not attendance_records:
        return pd.DataFrame(
            columns=[
                "Participant ID",
                "Presence Duration (s)",
                "Session Duration (s)",
                "Presence Rate (%)",
                "Attendance Status",
                "First Seen (s)",
                "Last Seen (s)",
            ]
        )

    rows = []
    for r in attendance_records:
        rows.append(
            {
                "Participant ID": f"Participant #{r.get('track_id', '?')}",
                "Presence Duration (s)": round(float(r.get("presence_duration_sec", 0.0)), 1),
                "Session Duration (s)": round(float(r.get("session_duration_sec", 0.0)), 1),
                "Presence Rate (%)": round(float(r.get("presence_percentage", 0.0)), 1),
                "Attendance Status": r.get("attendance_status", "Absent"),
                "First Seen (s)": round(float(r.get("first_seen_sec", 0.0)), 1),
                "Last Seen (s)": round(float(r.get("last_seen_sec", 0.0)), 1),
            }
        )

    return pd.DataFrame(rows)


def generate_attendance_csv(attendance_records: List[Dict[str, Any]]) -> str:
    """Generate CSV string from attendance records."""
    df = generate_attendance_dataframe(attendance_records)
    return df.to_csv(index=False)


def generate_engagement_timeline_dataframe(timeline: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert engagement timeline list into a pandas DataFrame."""
    if not timeline:
        return pd.DataFrame(
            columns=[
                "Timestamp (s)",
                "Window End (s)",
                "Engagement Score",
                "Level",
                "Visual Presence",
                "Visual Activity",
                "Audio Speech Score",
                "Audio Noise Penalty",
                "Explanation",
            ]
        )

    rows = []
    for t in timeline:
        factors = t.get("factors", {})
        rows.append(
            {
                "Timestamp (s)": round(float(t.get("timestamp_sec", 0.0)), 1),
                "Window End (s)": round(float(t.get("window_end_sec", 0.0)), 1),
                "Engagement Score": round(float(t.get("engagement_score", 0.0)), 1),
                "Level": t.get("level", "Moderate"),
                "Visual Presence": round(float(factors.get("visual_presence_rate", 0.0)), 2),
                "Visual Activity": round(float(factors.get("visual_activity_rate", 0.0)), 2),
                "Audio Speech Score": round(float(factors.get("audio_speech_score", 0.0)), 2),
                "Audio Noise Penalty": round(float(factors.get("noise_penalty", 0.0)), 2),
                "Explanation": t.get("explanation", ""),
            }
        )

    return pd.DataFrame(rows)


def generate_engagement_timeline_csv(timeline: List[Dict[str, Any]]) -> str:
    """Generate CSV string from engagement timeline."""
    df = generate_engagement_timeline_dataframe(timeline)
    return df.to_csv(index=False)


def generate_recommendations_dataframe(recommendations: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert recommendations list to a formatted DataFrame."""
    if not recommendations:
        return pd.DataFrame(
            columns=["Time Window", "Category", "Severity", "Suggestion", "Contextual Rationale"]
        )

    rows = []
    for rec in recommendations:
        start_t = rec.get("timestamp_start", 0.0)
        end_t = rec.get("timestamp_end", 0.0)
        time_str = f"{start_t:04.1f}s - {end_t:04.1f}s" if end_t > start_t else f"{start_t:04.1f}s"
        rows.append(
            {
                "Time Window": time_str,
                "Category": rec.get("category", "General"),
                "Severity": rec.get("severity", "Info"),
                "Suggestion": rec.get("title", ""),
                "Contextual Rationale": rec.get("rationale", ""),
            }
        )

    return pd.DataFrame(rows)


def generate_recommendations_csv(recommendations: List[Dict[str, Any]]) -> str:
    """Generate CSV string from recommendations."""
    df = generate_recommendations_dataframe(recommendations)
    return df.to_csv(index=False)


def export_full_session_json(
    session_data: Dict[str, Any],
    filepath: Optional[str] = None,
) -> str:
    """Export the complete session analysis to JSON format.

    Args:
        session_data: Complete dictionary containing metadata, attendance,
                      audio, vision, engagement, and recommendations.
        filepath: If provided, writes the JSON to this file path.

    Returns:
        JSON string representation.
    """
    cleaned = _make_json_serializable(session_data)
    json_str = json.dumps(cleaned, indent=2)

    if filepath:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(json_str)

    return json_str


def generate_text_session_summary(session_data: Dict[str, Any]) -> str:
    """Generate a clean markdown summary of the entire session."""
    meta = session_data.get("metadata", {})
    summary = session_data.get("summary", {})
    attendance = session_data.get("attendance", {})
    audio = session_data.get("audio_summary", {})
    recs = session_data.get("recommendations", [])

    lines = [
        "# Multimodal Classroom Attendance & Engagement Report",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Session File:** `{meta.get('video_filename', meta.get('audio_filename', 'Direct Input'))}`",
        f"**Session Duration:** {summary.get('session_duration_sec', 0.0):.1f} seconds",
        "",
        "## 1. Key Metrics Overview",
        f"- **Total Participants Tracked:** {attendance.get('total_participants', 0)}",
        f"- **Present:** {attendance.get('present_count', 0)} | **Partial:** {attendance.get('partial_count', 0)} | **Absent:** {attendance.get('absent_count', 0)}",
        f"- **Attendance Rate:** {attendance.get('overall_attendance_rate', 0.0):.1f}%",
        f"- **Average Engagement Score:** {summary.get('average_engagement_score', 0.0):.1f} / 100",
        f"- **Engagement Level:** {summary.get('engagement_level', 'Moderate')}",
        f"- **Speech Activity Ratio:** {audio.get('speech_ratio', 0.0) * 100:.1f}%",
        f"- **Silence Ratio:** {audio.get('silence_ratio', 0.0) * 100:.1f}%",
        "",
        "## 2. Participant Attendance",
    ]

    records = attendance.get("records", [])
    if records:
        lines.append("| Track ID | Presence (s) | Presence % | Status |")
        lines.append("|---|---|---|---|")
        for r in records:
            lines.append(
                f"| Participant #{r.get('track_id')} | {r.get('presence_duration_sec', 0.0):.1f}s | {r.get('presence_percentage', 0.0):.1f}% | {r.get('attendance_status')} |"
            )
    else:
        lines.append("_No participants detected in this session._")

    lines.extend([
        "",
        "## 3. Top Teaching Recommendations",
    ])

    if recs:
        for idx, rec in enumerate(recs[:5], 1):
            lines.append(f"{idx}. **[{rec.get('severity', 'Suggestion').upper()}] {rec.get('title')}**")
            lines.append(f"   _{rec.get('rationale')}_")
            start = rec.get("timestamp_start", 0.0)
            end = rec.get("timestamp_end", 0.0)
            lines.append(f"   *Applicable window: {start:.1f}s - {end:.1f}s*")
    else:
        lines.append("_Session demonstrated steady engagement without anomalies._")

    lines.extend([
        "",
        "---",
        "*Note: Engagement scores represent rule-based observable activity indicators and are designed as teaching aids, not definitive measures of cognitive attention.*",
    ])

    return "\n".join(lines)


def _make_json_serializable(obj: Any) -> Any:
    """Recursively convert NumPy data types and custom objects to standard JSON types."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        # Avoid serializing massive raw image matrices into JSON text
        if obj.ndim >= 3 and obj.size > 5000:
            return f"<image_array_{obj.shape}>"
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    elif hasattr(obj, "item"):
        try:
            return obj.item()
        except ValueError:
            return str(obj)
    return str(obj)
