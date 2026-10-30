"""CLI runner for Multimodal Attendance & Engagement Agent.

Allows running end-to-end multimodal analysis from the command line,
generating synthetic demonstration assets, batch processing sessions,
and exporting reports without launching the web dashboard.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
from typing import Optional

from modules.multimodal_agent import MultimodalAgent
from utils.audio_utils import generate_synthetic_classroom_audio
from utils.report_utils import (
    generate_attendance_csv,
    generate_engagement_timeline_csv,
    generate_recommendations_csv,
    generate_text_session_summary,
)
from utils.storage import get_reports_dir, list_saved_sessions, save_session
from utils.video_utils import generate_synthetic_classroom_video


def print_banner() -> None:
    banner = r"""
========================================================================
       MULTIMODAL ATTENDANCE & ENGAGEMENT AGENT
       Classroom Video & Audio Intelligence Engine
========================================================================
"""
    print(banner)


def handle_generate_sample(output_dir: str = "data/sample") -> None:
    """Generate sample synthetic classroom video and audio files."""
    os.makedirs(output_dir, exist_ok=True)
    video_out = os.path.join(output_dir, "sample_classroom.mp4")
    audio_out = os.path.join(output_dir, "sample_classroom_audio.wav")

    print("[*] Generating synthetic classroom test video (640x480, 3 students, 12s)...")
    generate_synthetic_classroom_video(
        output_path=video_out,
        duration_sec=12.0,
        fps=20,
        num_students=3,
    )
    print(f"    [+] Created video: {video_out}")

    print("[*] Generating synthetic classroom test audio (16kHz mono, lecture speech, 12s)...")
    generate_synthetic_classroom_audio(
        output_wav_path=audio_out,
        duration_sec=12.0,
        sr=16000,
    )
    print(f"    [+] Created audio: {audio_out}")
    print("\n[+] Sample generation complete! You can run:")
    print(f"    python main.py --video \"{video_out}\" --audio \"{audio_out}\"")


def run_pipeline(
    video_path: Optional[str] = None,
    audio_path: Optional[str] = None,
    present_threshold: float = 70.0,
    partial_threshold: float = 35.0,
    max_duration: Optional[float] = None,
    save: bool = True,
) -> None:
    """Execute end-to-end processing pipeline on supplied inputs."""
    print("[*] Initializing Multimodal Agent...")
    agent = MultimodalAgent(
        present_threshold_pct=present_threshold,
        partial_threshold_pct=partial_threshold,
        sample_fps=10.0,
        window_duration_sec=1.0,
    )

    def progress(fraction: float, message: str) -> None:
        pct = int(fraction * 100)
        sys.stdout.write(f"\r[{pct:3d}%] {message:<60}")
        sys.stdout.flush()

    print(f"[*] Input Video: {video_path or 'None'}")
    print(f"[*] Input Audio: {audio_path or 'Extracted from video / None'}")
    print("[*] Executing analysis...\n")

    results = agent.process_session(
        video_path=video_path,
        audio_path=audio_path,
        max_duration=max_duration,
        progress_callback=progress,
    )
    print("\n")

    if not results.get("success"):
        print(f"[!] Processing failed: {results.get('error')}")
        return

    # Print summary
    summary = results["summary"]
    att = results["attendance"]
    eng = results["engagement"]
    recs = results["recommendations"]

    print("=" * 65)
    print("                 SESSION RESULTS OVERVIEW                ")
    print("=" * 65)
    print(f"Duration:               {summary['session_duration_sec']:.1f} seconds")
    print(f"Total Participants:     {summary['total_participants']}")
    print(f"Present Participants:   {summary['present_count']} / {summary['total_participants']}")
    print(f"Attendance Rate:        {summary['attendance_rate']:.1f}%")
    print(f"Average Engagement:     {summary['average_engagement_score']:.1f}/100 ({summary['engagement_level']})")
    print(f"High Engagement Time:   {summary['high_engagement_duration_sec']:.1f}s")
    print(f"Moderate Engagement:   {summary['moderate_engagement_duration_sec']:.1f}s")
    print(f"Low Engagement Time:    {summary['low_engagement_duration_sec']:.1f}s")
    print("-" * 65)

    print("\n[+] Participant Attendance Details:")
    for r in att.get("records", []):
        print(
            f"    Participant #{r['track_id']:<2}: {r['presence_duration_sec']:5.1f}s "
            f"({r['presence_percentage']:5.1f}%) -> {r['attendance_status']}"
        )

    print(f"\n[+] Teaching Recommendations ({len(recs)} generated):")
    for r in recs:
        print(f"    * [{r['severity'].upper()}] {r['title']} ({r['timestamp_start']:.1f}s - {r['timestamp_end']:.1f}s)")
        print(f"      Rationale: {r['rationale']}")

    if save:
        session_id = save_session(results)
        reports_dir = get_reports_dir()

        # Also write standalone CSV files for convenient access
        att_csv_path = os.path.join(reports_dir, f"{session_id}_attendance.csv")
        with open(att_csv_path, "w", encoding="utf-8") as f:
            f.write(generate_attendance_csv(att.get("records", [])))

        eng_csv_path = os.path.join(reports_dir, f"{session_id}_engagement.csv")
        with open(eng_csv_path, "w", encoding="utf-8") as f:
            f.write(generate_engagement_timeline_csv(eng.get("timeline", [])))

        recs_csv_path = os.path.join(reports_dir, f"{session_id}_recommendations.csv")
        with open(recs_csv_path, "w", encoding="utf-8") as f:
            f.write(generate_recommendations_csv(recs))

        print(f"\n[+] Full reports successfully persisted to: {reports_dir}")
        print(f"    - JSON Session:        {session_id}.json")
        print(f"    - Attendance CSV:      {session_id}_attendance.csv")
        print(f"    - Engagement CSV:      {session_id}_engagement.csv")
        print(f"    - Recommendations CSV: {session_id}_recommendations.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multimodal Attendance & Engagement Agent - Core CLI Runner",
    )
    parser.add_argument("--video", type=str, help="Path to classroom video file")
    parser.add_argument("--audio", type=str, help="Path to classroom audio file (optional)")
    parser.add_argument("--generate-sample", action="store_true", help="Generate synthetic test classroom video & audio")
    parser.add_argument("--list-sessions", action="store_true", help="List all previously saved sessions")
    parser.add_argument("--present-thresh", type=float, default=70.0, help="Present threshold percentage (default 70.0)")
    parser.add_argument("--partial-thresh", type=float, default=35.0, help="Partial threshold percentage (default 35.0)")
    parser.add_argument("--max-duration", type=float, default=None, help="Max duration in seconds to process")

    args = parser.parse_args()
    print_banner()

    if args.generate_sample:
        handle_generate_sample()
        return

    if args.list_sessions:
        sessions = list_saved_sessions()
        if not sessions:
            print("No saved sessions found in data/reports/.")
        else:
            print(f"Found {len(sessions)} saved session(s):\n")
            for s in sessions:
                print(
                    f"  - [{s['session_id']}] {s['filename']} | "
                    f"Duration: {s['duration_sec']:.1f}s | "
                    f"Avg Engagement: {s['average_engagement']:.1f}/100"
                )
        return

    if not args.video and not args.audio:
        print("[!] No input specified. Generating sample and executing pipeline demonstration...")
        handle_generate_sample()
        video_sample = os.path.join("data", "sample", "sample_classroom.mp4")
        audio_sample = os.path.join("data", "sample", "sample_classroom_audio.wav")
        run_pipeline(
            video_path=video_sample,
            audio_path=audio_sample,
            present_threshold=args.present_thresh,
            partial_threshold=args.partial_thresh,
            max_duration=args.max_duration,
        )
        return

    run_pipeline(
        video_path=args.video,
        audio_path=args.audio,
        present_threshold=args.present_thresh,
        partial_threshold=args.partial_thresh,
        max_duration=args.max_duration,
    )


if __name__ == "__main__":
    main()
