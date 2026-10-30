"""Vision module for Multimodal Attendance & Engagement Agent.

Implements:
1. VisionAnalyzer: Frame extraction, face detection via MediaPipe/OpenCV,
   multi-participant tracking with anonymous IDs, activity estimation,
   and occlusion handling.
2. AttendanceAnalyzer: Presence duration calculation, percentage thresholding,
   and Present/Partial/Absent status classification.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from utils.video_utils import draw_detection_overlay, get_video_metadata, iterate_video_frames


class ParticipantTrack:
    """Represents an anonymous tracked participant across video frames."""

    def __init__(self, track_id: int, initial_bbox: Tuple[int, int, int, int], timestamp: float, frame_idx: int):
        self.track_id = track_id
        self.bbox = initial_bbox  # (x, y, w, h)
        self.centroid = self._calc_centroid(initial_bbox)
        self.first_seen_sec = timestamp
        self.last_seen_sec = timestamp
        self.first_seen_frame = frame_idx
        self.last_seen_frame = frame_idx
        self.presence_frames = 1
        self.consecutive_lost = 0
        self.displacements: List[float] = []
        self.recent_activity: float = 0.5  # Normalized 0.0 - 1.0

    @staticmethod
    def _calc_centroid(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x, y, w, h = bbox
        return float(x + w / 2.0), float(y + h / 2.0)

    def update(self, bbox: Tuple[int, int, int, int], timestamp: float, frame_idx: int) -> None:
        """Update track with a newly matched detection."""
        new_centroid = self._calc_centroid(bbox)
        dist = math.hypot(new_centroid[0] - self.centroid[0], new_centroid[1] - self.centroid[1])

        # Smooth bounding box with exponential moving average to reduce jitter
        alpha = 0.7
        x = int(alpha * bbox[0] + (1 - alpha) * self.bbox[0])
        y = int(alpha * bbox[1] + (1 - alpha) * self.bbox[1])
        w = int(alpha * bbox[2] + (1 - alpha) * self.bbox[2])
        h = int(alpha * bbox[3] + (1 - alpha) * self.bbox[3])

        self.bbox = (x, y, w, h)
        self.centroid = new_centroid
        self.last_seen_sec = timestamp
        self.last_seen_frame = frame_idx
        self.presence_frames += 1
        self.consecutive_lost = 0

        # Activity level based on displacement (nodding, turning, taking notes)
        self.displacements.append(dist)
        if len(self.displacements) > 30:
            self.displacements.pop(0)

        # Average pixel movement normalized by head bounding height
        avg_disp = float(np.mean(self.displacements)) if self.displacements else 0.0
        norm_activity = min(1.0, max(0.05, avg_disp / max(10.0, float(h) * 0.2)))
        self.recent_activity = float(0.6 * norm_activity + 0.4 * self.recent_activity)

    def mark_missed(self) -> None:
        """Mark track as missed in current frame."""
        self.consecutive_lost += 1
        # Activity decays when participant is occluded
        self.recent_activity = max(0.0, self.recent_activity * 0.9)


class VisionAnalyzer:
    """Processes video frames, detects faces using MediaPipe, and tracks participants."""

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        max_lost_frames: int = 35,
        max_match_distance: float = 95.0,
        sample_fps: float = 10.0,
    ):
        """Initialize VisionAnalyzer.

        Args:
            min_detection_confidence: Confidence threshold for face detection.
            max_lost_frames: Frames to keep track alive during occlusion/turning away.
            max_match_distance: Pixel threshold for centroid matching.
            sample_fps: Frame processing rate (downsampled from original for high efficiency).
        """
        self.min_confidence = min_detection_confidence
        self.max_lost_frames = max_lost_frames
        self.max_match_distance = max_match_distance
        self.sample_fps = sample_fps
        self._mp_face_detector = None
        self._cv_fallback_detector = None
        self._init_detector()

    def _init_detector(self) -> None:
        """Initialize MediaPipe face detection with OpenCV Haar Cascade fallback."""
        try:
            import mediapipe as mp

            # MediaPipe Solutions API
            mp_face = mp.solutions.face_detection
            self._mp_face_detector = mp_face.FaceDetection(
                min_detection_confidence=self.min_confidence,
                model_selection=1,  # Full range model (better for classroom distance)
            )
        except Exception:
            self._mp_face_detector = None

        # Always initialize OpenCV cascade as reliable fallback
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if os.path.exists(cascade_path):
                self._cv_fallback_detector = cv2.CascadeClassifier(cascade_path)
        except Exception:
            self._cv_fallback_detector = None

    def _detect_faces(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces in a single frame. Returns list of bboxes and confidences."""
        h, w = frame_bgr.shape[:2]
        detections: List[Dict[str, Any]] = []

        # Try MediaPipe first
        if self._mp_face_detector is not None:
            try:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                results = self._mp_face_detector.process(frame_rgb)
                if results and results.detections:
                    for det in results.detections:
                        score = float(det.score[0]) if det.score else 0.8
                        bb = det.location_data.relative_bounding_box
                        bx = int(max(0, bb.xmin * w))
                        by = int(max(0, bb.ymin * h))
                        bw = int(min(w - bx, bb.width * w))
                        bh = int(min(h - by, bb.height * h))
                        if bw > 12 and bh > 12:
                            detections.append({"bbox": (bx, by, bw, bh), "confidence": score})
                    return detections
            except Exception:
                pass

        # Fallback to OpenCV Haar Cascade if MediaPipe returned nothing or failed
        if self._cv_fallback_detector is not None:
            try:
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                faces = self._cv_fallback_detector.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=4, minSize=(25, 25)
                )
                for (x, y, bw, bh) in faces:
                    detections.append({"bbox": (int(x), int(y), int(bw), int(bh)), "confidence": 0.75})
                if detections:
                    return detections
            except Exception:
                pass

        # Fallback for synthetic/simulated classroom demonstration videos
        # Detects synthetic avatar face contours if neither real-face detector triggered
        try:
            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            # Mask for synthetic skin tone range
            lower_skin = np.array([5, 40, 100], dtype=np.uint8)
            upper_skin = np.array([25, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_skin, upper_skin)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 1200 < area < 15000:  # Head oval size range
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    aspect = bh / float(bw)
                    if 1.0 < aspect < 2.0 and y < h * 0.75:  # Realistic head proportions in upper desk area
                        detections.append({"bbox": (int(x), int(y), int(bw), int(bh)), "confidence": 0.88})
        except Exception:
            pass

        return detections

    def analyze_video(
        self,
        video_path: str,
        max_duration: Optional[float] = None,
        progress_callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Analyze classroom video, tracking all participants across frames.

        Args:
            video_path: Path to classroom video.
            max_duration: Optional limit on processing duration.
            progress_callback: Optional callback(fraction_completed, status_text).

        Returns:
            Structured dictionary of tracking metrics, participant presence,
            per-frame timelines, and video metadata.
        """
        meta = get_video_metadata(video_path)
        if not meta["valid"]:
            return {
                "valid": False,
                "error": meta.get("error", "Invalid video file"),
                "tracks": [],
                "frame_timeline": [],
                "metadata": meta,
            }

        total_duration = meta["duration"]
        if max_duration and max_duration < total_duration:
            total_duration = max_duration

        active_tracks: List[ParticipantTrack] = []
        all_tracks: List[ParticipantTrack] = []
        next_track_id = 1

        frame_timeline: List[Dict[str, Any]] = []
        keyframe_thumbnails: List[Dict[str, Any]] = []

        processed_count = 0
        target_frames = int(total_duration * self.sample_fps) if total_duration > 0 else 1

        for frame_idx, timestamp, frame in iterate_video_frames(
            video_path, target_fps=self.sample_fps, max_duration=max_duration
        ):
            processed_count += 1
            if progress_callback and target_frames > 0:
                prog = min(1.0, processed_count / float(target_frames))
                progress_callback(prog, f"Analyzing video frames... ({timestamp:04.1f}s)")

            # Detect faces in current frame
            detections = self._detect_faces(frame)

            # Match detections to existing active tracks (Centroid proximity)
            matched_det_indices = set()
            matched_track_indices = set()

            if active_tracks and detections:
                cost_matrix = np.zeros((len(active_tracks), len(detections)), dtype=np.float32)
                for i, trk in enumerate(active_tracks):
                    for j, det in enumerate(detections):
                        cx, cy = trk.centroid
                        dcx = det["bbox"][0] + det["bbox"][2] / 2.0
                        dcy = det["bbox"][1] + det["bbox"][3] / 2.0
                        dist = math.hypot(cx - dcx, cy - dcy)
                        cost_matrix[i, j] = dist

                # Greedy matching
                while True:
                    min_val = np.min(cost_matrix)
                    if min_val > self.max_match_distance:
                        break
                    r, c = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
                    if min_val == np.inf:
                        break

                    matched_track_indices.add(r)
                    matched_det_indices.add(c)
                    active_tracks[r].update(detections[c]["bbox"], timestamp, frame_idx)

                    # Invalidate row and column
                    cost_matrix[r, :] = np.inf
                    cost_matrix[:, c] = np.inf

            # Update unmatched active tracks
            surviving_active: List[ParticipantTrack] = []
            for i, trk in enumerate(active_tracks):
                if i not in matched_track_indices:
                    trk.mark_missed()
                if trk.consecutive_lost <= self.max_lost_frames:
                    surviving_active.append(trk)

            # Spawn new tracks for unmatched detections
            for j, det in enumerate(detections):
                if j not in matched_det_indices:
                    new_track = ParticipantTrack(next_track_id, det["bbox"], timestamp, frame_idx)
                    next_track_id += 1
                    surviving_active.append(new_track)
                    all_tracks.append(new_track)

            active_tracks = surviving_active

            # Snapshot frame metrics
            visible_participants = [
                {
                    "track_id": trk.track_id,
                    "bbox": trk.bbox,
                    "activity_level": trk.recent_activity,
                }
                for trk in active_tracks
                if trk.consecutive_lost == 0
            ]

            frame_timeline.append(
                {
                    "timestamp_sec": float(timestamp),
                    "frame_idx": frame_idx,
                    "visible_count": len(visible_participants),
                    "participants": visible_participants,
                    "mean_activity": float(np.mean([p["activity_level"] for p in visible_participants]))
                    if visible_participants
                    else 0.0,
                }
            )

            # Capture keyframe thumbnail every ~3-5 seconds
            if len(keyframe_thumbnails) < 6 and (frame_idx % int(self.sample_fps * 4) == 0 or frame_idx == 0):
                overlay = draw_detection_overlay(frame, visible_participants)
                # Resize for lightweight storage
                thumb = cv2.resize(overlay, (320, 240))
                keyframe_thumbnails.append({"timestamp": float(timestamp), "frame_bgr": thumb})

        # Calculate presence duration for all tracks
        # Effective presence duration = presence_frames / sample_fps
        track_summaries: List[Dict[str, Any]] = []
        for trk in all_tracks:
            # Filter spurious one-frame noise detections (must appear in at least 3 samples)
            if trk.presence_frames < 3 and total_duration > 5.0:
                continue

            presence_sec = trk.presence_frames / self.sample_fps
            presence_sec = min(total_duration, presence_sec)
            presence_ratio = (presence_sec / total_duration) if total_duration > 0 else 0.0

            track_summaries.append(
                {
                    "track_id": trk.track_id,
                    "first_seen_sec": float(trk.first_seen_sec),
                    "last_seen_sec": float(trk.last_seen_sec),
                    "presence_duration_sec": float(presence_sec),
                    "presence_percentage": float(presence_ratio * 100.0),
                    "presence_ratio": float(presence_ratio),
                    "total_frames_seen": trk.presence_frames,
                    "average_activity": float(np.mean(trk.displacements)) if trk.displacements else 0.5,
                }
            )

        return {
            "valid": True,
            "error": None,
            "session_duration_sec": float(total_duration),
            "tracks": track_summaries,
            "total_participants_detected": len(track_summaries),
            "frame_timeline": frame_timeline,
            "keyframes": keyframe_thumbnails,
            "metadata": meta,
        }


class AttendanceAnalyzer:
    """Computes participant presence percentage and classifies attendance status."""

    def __init__(
        self,
        present_threshold_pct: float = 70.0,
        partial_threshold_pct: float = 35.0,
    ):
        """Initialize AttendanceAnalyzer.

        Args:
            present_threshold_pct: Threshold percentage for 'Present' (e.g. 70%).
            partial_threshold_pct: Threshold percentage for 'Partial' (e.g. 35%).
                                   Below this is classified as 'Absent'.
        """
        self.present_threshold = present_threshold_pct
        self.partial_threshold = partial_threshold_pct

    def compute_attendance(
        self,
        tracks: List[Dict[str, Any]],
        session_duration_sec: float,
    ) -> Dict[str, Any]:
        """Compute attendance status for all participant tracks.

        Args:
            tracks: List of track summary dictionaries from VisionAnalyzer.
            session_duration_sec: Total duration of classroom session.

        Returns:
            Dictionary with categorized attendance counts and per-participant records.
        """
        records: List[Dict[str, Any]] = []
        present_count = 0
        partial_count = 0
        absent_count = 0

        for trk in sorted(tracks, key=lambda x: x["track_id"]):
            presence_sec = trk.get("presence_duration_sec", 0.0)
            presence_pct = (presence_sec / session_duration_sec * 100.0) if session_duration_sec > 0 else 0.0
            presence_pct = min(100.0, max(0.0, presence_pct))

            if presence_pct >= self.present_threshold:
                status = "Present"
                present_count += 1
            elif presence_pct >= self.partial_threshold:
                status = "Partial"
                partial_count += 1
            else:
                status = "Absent"
                absent_count += 1

            records.append(
                {
                    "track_id": trk["track_id"],
                    "presence_duration_sec": float(presence_sec),
                    "session_duration_sec": float(session_duration_sec),
                    "presence_percentage": float(presence_pct),
                    "attendance_status": status,
                    "first_seen_sec": float(trk.get("first_seen_sec", 0.0)),
                    "last_seen_sec": float(trk.get("last_seen_sec", 0.0)),
                }
            )

        total = len(records)
        overall_rate = (
            sum(r["presence_percentage"] for r in records) / total if total > 0 else 0.0
        )

        return {
            "total_participants": total,
            "present_count": present_count,
            "partial_count": partial_count,
            "absent_count": absent_count,
            "overall_attendance_rate": float(overall_rate),
            "present_threshold_pct": self.present_threshold,
            "partial_threshold_pct": self.partial_threshold,
            "records": records,
        }
