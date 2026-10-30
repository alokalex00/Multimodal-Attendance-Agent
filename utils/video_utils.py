"""Video utilities for Multimodal Attendance & Engagement Agent.

Handles video loading, frame extraction, metadata probing,
visual overlays for detection & tracking, and synthetic video generation for testing.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np


def get_video_metadata(video_path: str) -> Dict[str, Any]:
    """Extract metadata from a video file using OpenCV.

    Args:
        video_path: Path to the video file.

    Returns:
        Dictionary containing fps, frame_count, duration, width, height, and validity.
    """
    if not os.path.exists(video_path):
        return {
            "valid": False,
            "error": f"File not found: {video_path}",
            "fps": 0.0,
            "frame_count": 0,
            "duration": 0.0,
            "width": 0,
            "height": 0,
        }

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "valid": False,
            "error": "Failed to open video file (unsupported format or corrupted file)",
            "fps": 0.0,
            "frame_count": 0,
            "duration": 0.0,
            "width": 0,
            "height": 0,
        }

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    duration = (frame_count / fps) if fps > 0 else 0.0

    return {
        "valid": True,
        "error": None,
        "fps": float(fps),
        "frame_count": frame_count,
        "duration": float(duration),
        "width": width,
        "height": height,
    }


def iterate_video_frames(
    video_path: str,
    target_fps: Optional[float] = None,
    max_duration: Optional[float] = None,
) -> Generator[Tuple[int, float, np.ndarray], None, None]:
    """Iterate over frames of a video at a targeted sampling rate.

    Args:
        video_path: Path to the video file.
        target_fps: If provided, downsample frames to match target_fps.
        max_duration: Maximum duration in seconds to process.

    Yields:
        (frame_index, timestamp_seconds, frame_bgr)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = 1
    if target_fps and target_fps > 0 and target_fps < source_fps:
        step = max(1, int(round(source_fps / target_fps)))

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            timestamp = frame_idx / source_fps

            if max_duration and timestamp > max_duration:
                break

            if frame_idx % step == 0:
                yield frame_idx, timestamp, frame

            frame_idx += 1
    finally:
        cap.release()


def draw_detection_overlay(
    frame: np.ndarray,
    tracked_participants: List[Dict[str, Any]],
    show_ids: bool = True,
    show_activity: bool = True,
) -> np.ndarray:
    """Draw professional bounding boxes, tracking labels, and activity indicators.

    Args:
        frame: BGR frame to draw on.
        tracked_participants: List of participant track dicts with bbox (x, y, w, h),
                             track_id, status, confidence, activity_level.
        show_ids: Whether to display anonymous participant IDs.
        show_activity: Whether to draw activity badges.

    Returns:
        Decorated BGR frame copy.
    """
    out = frame.copy()
    h, w = out.shape[:2]

    # Color palette (modern cyan, violet, emerald)
    PRIMARY_COLOR = (245, 130, 49)  # Soft blue/cyan BGR: (49, 130, 245) -> (245, 130, 49) in RGB
    ACCENT_COLOR = (200, 80, 160)   # Purple/violet
    TEXT_BG = (24, 24, 32)          # Dark card background
    TEXT_COLOR = (240, 240, 240)    # Crisp light gray

    for p in tracked_participants:
        bbox = p.get("bbox")
        if not bbox:
            continue

        bx, by, bw, bh = bbox
        bx = max(0, min(bx, w - 1))
        by = max(0, min(by, h - 1))
        bw = max(2, min(bw, w - bx))
        bh = max(2, min(bh, h - by))

        track_id = p.get("track_id", 0)
        activity = p.get("activity_level", 0.5)  # 0.0 to 1.0

        # Activity-based border color: emerald for active, amber for moderate, gray for low
        if activity >= 0.6:
            border_color = (60, 210, 120)  # Emerald
            status_text = "Active"
        elif activity >= 0.3:
            border_color = (40, 180, 245)  # Amber / Cyan
            status_text = "Attentive"
        else:
            border_color = (150, 140, 130)  # Muted
            status_text = "Static"

        # Sleek corner accents bounding box
        corner_len = min(20, bw // 3, bh // 3)
        thick = 2

        # Main subtle rectangle
        cv2.rectangle(out, (bx, by), (bx + bw, by + bh), border_color, 1)

        # Thick corner accents
        # Top-left
        cv2.line(out, (bx, by), (bx + corner_len, by), border_color, thick)
        cv2.line(out, (bx, by), (bx, by + corner_len), border_color, thick)
        # Top-right
        cv2.line(out, (bx + bw, by), (bx + bw - corner_len, by), border_color, thick)
        cv2.line(out, (bx + bw, by), (bx + bw, by + corner_len), border_color, thick)
        # Bottom-left
        cv2.line(out, (bx, by + bh), (bx + corner_len, by + bh), border_color, thick)
        cv2.line(out, (bx, by + bh), (bx, by + bh - corner_len), border_color, thick)
        # Bottom-right
        cv2.line(out, (bx + bw, by + bh), (bx + bw - corner_len, by + bh), border_color, thick)
        cv2.line(out, (bx + bw, by + bh), (bx + bw, by + bh - corner_len), border_color, thick)

        # Label tag
        if show_ids:
            label = f"ID #{track_id}"
            if show_activity:
                label += f" | {status_text}"

            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            tag_y1 = max(0, by - th - 8)
            tag_y2 = by
            tag_x1 = bx
            tag_x2 = min(w, bx + tw + 10)

            # Draw tag background
            cv2.rectangle(out, (tag_x1, tag_y1), (tag_x2, tag_y2), TEXT_BG, -1)
            cv2.rectangle(out, (tag_x1, tag_y1), (tag_x2, tag_y2), border_color, 1)
            cv2.putText(
                out,
                label,
                (tag_x1 + 5, tag_y2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                TEXT_COLOR,
                1,
                cv2.LINE_AA,
            )

    return out


def generate_synthetic_classroom_video(
    output_path: str,
    duration_sec: float = 10.0,
    fps: int = 20,
    width: int = 640,
    height: int = 480,
    num_students: int = 3,
) -> str:
    """Generate a realistic synthetic classroom video for testing and demonstration.

    Simulates 3 student avatars seated at classroom desks with head movements,
    nodding, looking at the board, writing, and temporary occlusion.

    Args:
        output_path: Destination path for the MP4 video.
        duration_sec: Duration in seconds.
        fps: Frames per second.
        width: Video width in pixels.
        height: Video height in pixels.
        num_students: Number of simulated participants.

    Returns:
        Absolute path to the created video.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))

    total_frames = int(duration_sec * fps)

    # Base student positions (horizontal spread)
    student_bases = []
    step_x = width // (num_students + 1)
    for i in range(num_students):
        base_x = step_x * (i + 1)
        base_y = height // 2 + 30
        student_bases.append({"x": base_x, "y": base_y, "id": i + 1})

    for f in range(total_frames):
        t = f / fps

        # Background: Modern classroom with dark slate whiteboard and warm room tone
        frame = np.full((height, width, 3), (32, 28, 25), dtype=np.uint8)

        # Classroom board in the background
        cv2.rectangle(frame, (50, 20), (width - 50, 180), (45, 55, 45), -1)
        cv2.rectangle(frame, (48, 18), (width - 48, 182), (80, 95, 80), 2)
        cv2.putText(
            frame,
            "AI & Data Science Lecture",
            (80, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (210, 225, 210),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Session Time: {t:04.1f}s / {duration_sec:04.1f}s",
            (80, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (170, 190, 170),
            1,
            cv2.LINE_AA,
        )

        # Classroom desk line
        cv2.line(frame, (0, height - 80), (width, height - 80), (60, 50, 45), 4)

        # Render each simulated student
        for s in student_bases:
            sid = s["id"]
            # Student 2 temporarily leaves/ducks at t=5s to 7s to simulate occlusion
            if sid == 2 and 5.0 <= t <= 7.0:
                continue

            # Natural subtle motion (nodding, tilting, shifting)
            nod_freq = 0.5 + 0.3 * sid
            shift_x = int(12 * math.sin(2 * math.pi * nod_freq * t))
            shift_y = int(6 * math.cos(2 * math.pi * (nod_freq * 1.5) * t))

            cx = s["x"] + shift_x
            cy = s["y"] + shift_y

            # Body / Torso
            cv2.ellipse(
                frame,
                (cx, cy + 90),
                (55, 70),
                0,
                0,
                360,
                (70 + 20 * sid, 60, 50 + 30 * sid),
                -1,
            )

            # Neck
            cv2.rectangle(frame, (cx - 12, cy + 30), (cx + 12, cy + 50), (180, 190, 215), -1)

            # Head / Face oval (Realistic skin tone)
            skin_color = (175 + 10 * sid, 195 + 10 * sid, 235)  # BGR skin tone
            cv2.ellipse(frame, (cx, cy), (32, 42), 0, 0, 360, skin_color, -1)

            # Hair
            cv2.ellipse(frame, (cx, cy - 20), (34, 25), 0, 180, 360, (25, 25, 30), -1)

            # Eyes (blinking periodically)
            eye_y = cy - 6
            blink = (f % int(fps * 3.5)) < 3
            if blink:
                cv2.line(frame, (cx - 16, eye_y), (cx - 8, eye_y), (40, 35, 35), 2)
                cv2.line(frame, (cx + 8, eye_y), (cx + 16, eye_y), (40, 35, 35), 2)
            else:
                cv2.circle(frame, (cx - 12, eye_y), 4, (245, 245, 250), -1)
                cv2.circle(frame, (cx + 12, eye_y), 4, (245, 245, 250), -1)
                cv2.circle(frame, (cx - 12, eye_y), 2, (30, 25, 25), -1)
                cv2.circle(frame, (cx + 12, eye_y), 2, (30, 25, 25), -1)

            # Eyebrows
            cv2.line(frame, (cx - 18, eye_y - 7), (cx - 6, eye_y - 6), (30, 30, 35), 2)
            cv2.line(frame, (cx + 6, eye_y - 6), (cx + 18, eye_y - 7), (30, 30, 35), 2)

            # Nose
            cv2.line(frame, (cx, eye_y + 2), (cx - 2, eye_y + 12), (150, 165, 205), 2)
            cv2.line(frame, (cx - 2, eye_y + 12), (cx + 3, eye_y + 12), (150, 165, 205), 2)

            # Mouth
            cv2.ellipse(frame, (cx, eye_y + 22), (10, 4), 0, 0, 180, (110, 120, 175), 2)

            # Participant identifier watermark
            cv2.putText(
                frame,
                f"Desk {sid}",
                (cx - 25, cy + 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (140, 140, 150),
                1,
                cv2.LINE_AA,
            )

        out.write(frame)

    out.release()
    return os.path.abspath(output_path)
