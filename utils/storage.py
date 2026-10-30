"""Session storage and persistence for Multimodal Attendance & Engagement Agent.

Saves and indexes classroom analysis sessions in JSON and CSV under `data/reports/`.
"""

from __future__ import annotations

import glob
import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from utils.report_utils import export_full_session_json


def get_reports_dir() -> str:
    """Return absolute path to reports directory, ensuring it exists."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reports_dir = os.path.join(base_dir, "data", "reports")
    os.makedirs(reports_dir, exist_ok=True)
    return reports_dir


def save_session(
    session_data: Dict[str, Any],
    session_id: Optional[str] = None,
) -> str:
    """Save full session data to data/reports/ as JSON and record metadata.

    Args:
        session_data: Complete dictionary from MultimodalAgent.
        session_id: Optional unique identifier. If None, generates a timestamped ID.

    Returns:
        session_id string.
    """
    reports_dir = get_reports_dir()

    if not session_id:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        rand_suffix = uuid.uuid4().hex[:6]
        session_id = f"session_{timestamp_str}_{rand_suffix}"

    filepath = os.path.join(reports_dir, f"{session_id}.json")
    session_data["session_id"] = session_id
    session_data["saved_at"] = datetime.now().isoformat()

    export_full_session_json(session_data, filepath)
    return session_id


def list_saved_sessions() -> List[Dict[str, Any]]:
    """List all previously saved sessions sorted by modification time (most recent first)."""
    reports_dir = get_reports_dir()
    json_files = glob.glob(os.path.join(reports_dir, "session_*.json"))

    sessions = []
    for fp in sorted(json_files, key=os.path.getmtime, reverse=True):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
                summary = data.get("summary", {})
                attendance = data.get("attendance", {})
                meta = data.get("metadata", {})
                sessions.append(
                    {
                        "session_id": data.get("session_id", os.path.splitext(os.path.basename(fp))[0]),
                        "saved_at": data.get("saved_at", datetime.fromtimestamp(os.path.getmtime(fp)).isoformat()),
                        "filename": meta.get("video_filename", meta.get("audio_filename", "Classroom Recording")),
                        "duration_sec": summary.get("session_duration_sec", 0.0),
                        "participants": attendance.get("total_participants", 0),
                        "average_engagement": summary.get("average_engagement_score", 0.0),
                        "filepath": fp,
                    }
                )
        except Exception:
            continue

    return sessions


def load_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Load a session dictionary from file given its ID."""
    reports_dir = get_reports_dir()
    filepath = os.path.join(reports_dir, f"{session_id}.json")
    if not os.path.exists(filepath):
        # Try finding if it's already a full filename
        if os.path.exists(os.path.join(reports_dir, session_id)):
            filepath = os.path.join(reports_dir, session_id)
        else:
            return None

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def delete_session(session_id: str) -> bool:
    """Delete a saved session JSON and associated files."""
    reports_dir = get_reports_dir()
    filepath = os.path.join(reports_dir, f"{session_id}.json")
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            return True
        except OSError:
            return False
    return False
