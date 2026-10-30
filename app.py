"""Streamlit Web Application: Multimodal Attendance & Engagement Agent.

A modern, responsive AI analytics dashboard providing visual participant tracking,
presence-based attendance, acoustic speech analysis, fused engagement metrics,
and contextual teaching recommendations.
"""

from __future__ import annotations

import io
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modules.multimodal_agent import MultimodalAgent
from utils.audio_utils import generate_synthetic_classroom_audio
from utils.report_utils import (
    generate_attendance_csv,
    generate_attendance_dataframe,
    generate_engagement_timeline_csv,
    generate_engagement_timeline_dataframe,
    generate_recommendations_csv,
    generate_recommendations_dataframe,
    generate_text_session_summary,
)
from utils.storage import get_reports_dir, list_saved_sessions, load_session, save_session
from utils.video_utils import generate_synthetic_classroom_video

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Multimodal Attendance & Engagement Agent",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for dark slate / navy aesthetic with vibrant cyan/violet accents
CUSTOM_CSS = """
<style>
    /* Global styles */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header Card */
    .main-header {
        background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
        border: 1px solid #312e81;
        padding: 24px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
    }
    .main-header h1 {
        color: #f8fafc;
        font-size: 26px;
        font-weight: 700;
        margin: 0 0 8px 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 14px;
        margin: 0;
    }
    
    /* Metric Cards */
    .metric-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 18px 20px;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .metric-card:hover {
        border-color: #38bdf8;
        transform: translateY(-2px);
    }
    .metric-label {
        font-size: 12px;
        text-transform: uppercase;
        color: #94a3b8;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .metric-val {
        font-size: 26px;
        font-weight: 700;
        color: #f8fafc;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 12px;
        color: #64748b;
        margin-top: 4px;
    }
    
    /* Badges */
    .badge-present {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid #059669;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-partial {
        background-color: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid #d97706;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-absent {
        background-color: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid #dc2626;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
    }
    
    /* Recommendation Card */
    .rec-card {
        background: #111827;
        border-left: 4px solid #38bdf8;
        border-top: 1px solid #1f2937;
        border-right: 1px solid #1f2937;
        border-bottom: 1px solid #1f2937;
        padding: 16px 20px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 14px;
    }
    .rec-card.rec-alert {
        border-left-color: #ef4444;
    }
    .rec-card.rec-suggestion {
        border-left-color: #f59e0b;
    }
    .rec-card.rec-positive {
        border-left-color: #10b981;
    }
    
    /* Disclaimer banner */
    .disclaimer-banner {
        background-color: rgba(30, 41, 59, 0.6);
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 12px;
        color: #94a3b8;
        margin-top: 20px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
if "session_data" not in st.session_state:
    st.session_state["session_data"] = None

if "last_processed_file" not in st.session_state:
    st.session_state["last_processed_file"] = None


# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def get_sample_paths() -> Tuple[str, str]:
    """Ensure synthetic sample files exist and return paths."""
    sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sample")
    os.makedirs(sample_dir, exist_ok=True)
    video_path = os.path.join(sample_dir, "sample_classroom.mp4")
    audio_path = os.path.join(sample_dir, "sample_classroom_audio.wav")

    if not os.path.exists(video_path):
        generate_synthetic_classroom_video(video_path, duration_sec=12.0, fps=20, num_students=3)

    if not os.path.exists(audio_path):
        generate_synthetic_classroom_audio(audio_path, duration_sec=12.0, sr=16000)

    return video_path, audio_path


def format_duration(seconds: float) -> str:
    """Format seconds into MM:SS format."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/artificial-intelligence.png",
        width=48,
    )
    st.markdown("### **Multimodal Agent**")
    st.caption("AI Classroom Analytics & Attendance")

    st.markdown("---")
    menu_selection = st.radio(
        "Navigate",
        [
            "📊 Overview",
            "📤 Upload & Analyze",
            "👥 Attendance Records",
            "📈 Engagement Analytics",
            "💡 Teaching Suggestions",
            "📑 Session Reports",
        ],
        index=0,
    )

    st.markdown("---")

    # Current session quick status
    active_data = st.session_state["session_data"]
    if active_data and active_data.get("success"):
        st.markdown("#### **Active Session**")
        meta = active_data.get("metadata", {})
        summ = active_data.get("summary", {})
        fname = meta.get("video_filename") or meta.get("audio_filename") or "Classroom Session"
        st.write(f"📁 **File:** `{fname}`")
        st.write(f"⏱ **Duration:** `{summ.get('session_duration_sec', 0):.1f}s`")
        st.write(f"👥 **Students:** `{summ.get('total_participants', 0)}`")
        st.write(f"🎯 **Avg Score:** `{summ.get('average_engagement_score', 0):.1f}/100`")

        if st.button("🔄 Clear Active Session", use_container_width=True):
            st.session_state["session_data"] = None
            st.rerun()
    else:
        st.info("No active session loaded. Upload a video or load sample data.")
        if st.button("⚡ Load Demo Session", use_container_width=True, type="primary"):
            v_path, a_path = get_sample_paths()
            agent = MultimodalAgent()
            with st.spinner("Analyzing demo classroom recording..."):
                res = agent.process_session(video_path=v_path, audio_path=a_path)
                save_session(res)
                st.session_state["session_data"] = res
                st.session_state["last_processed_file"] = "sample_classroom.mp4"
            st.success("Demo session loaded successfully!")
            st.rerun()

    st.markdown("---")
    st.caption("Multimodal Attendance & Engagement Agent v1.0.0\nRuns completely locally.")


# -----------------------------------------------------------------------------
# VIEW 1: OVERVIEW
# -----------------------------------------------------------------------------
if menu_selection == "📊 Overview":
    st.markdown(
        """
        <div class="main-header">
            <h1>Multimodal Classroom Attendance & Engagement Dashboard</h1>
            <p>Unified visual participant tracking, presence estimation, acoustic activity analysis, and pedagogical feedback.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    active_data = st.session_state["session_data"]

    if not active_data or not active_data.get("success"):
        st.markdown(
            """
            <div style="background:#111827; border: 1px dashed #374151; padding: 48px; border-radius: 12px; text-align: center;">
                <h3 style="color:#f8fafc; margin-bottom: 8px;">No Classroom Session Analyzed Yet</h3>
                <p style="color:#94a3b8; max-width: 520px; margin: 0 auto 24px auto;">
                    Upload a recorded classroom video (or standalone audio file) to track student attendance,
                    estimate multimodal engagement trends, and receive actionable teaching suggestions.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.write("")
            if st.button("🚀 Load Sample Classroom Session for Instant Demo", use_container_width=True, type="primary"):
                v_path, a_path = get_sample_paths()
                agent = MultimodalAgent()
                with st.spinner("Processing sample classroom video and audio signals..."):
                    res = agent.process_session(video_path=v_path, audio_path=a_path)
                    save_session(res)
                    st.session_state["session_data"] = res
                    st.session_state["last_processed_file"] = "sample_classroom.mp4"
                st.rerun()

    else:
        summ = active_data.get("summary", {})
        att = active_data.get("attendance", {})
        audio = active_data.get("audio_summary", {})
        eng = active_data.get("engagement", {})
        meta = active_data.get("metadata", {})
        vision = active_data.get("vision", {})

        # Top 5 KPI Metric Cards
        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Tracked Participants</div>
                    <div class="metric-val">{att.get('total_participants', 0)}</div>
                    <div class="metric-sub">{att.get('present_count', 0)} Present | {att.get('partial_count', 0)} Partial</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c2:
            att_rate = att.get("overall_attendance_rate", 0.0)
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Attendance Rate</div>
                    <div class="metric-val" style="color: {'#34d399' if att_rate >= 70 else '#fbbf24'};">{att_rate:.1f}%</div>
                    <div class="metric-sub">Threshold: {att.get('present_threshold_pct', 70):.0f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c3:
            avg_score = summ.get("average_engagement_score", 0.0)
            score_color = "#34d399" if avg_score >= 70 else ("#fbbf24" if avg_score >= 40 else "#f87171")
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Avg Engagement</div>
                    <div class="metric-val" style="color: {score_color};">{avg_score:.1f}<span style="font-size:16px;color:#94a3b8;">/100</span></div>
                    <div class="metric-sub">Rating: <b>{summ.get('engagement_level', 'Moderate')}</b></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c4:
            dur = summ.get("session_duration_sec", 0.0)
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Session Duration</div>
                    <div class="metric-val">{dur:.1f}s</div>
                    <div class="metric-sub">{format_duration(dur)} minutes</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c5:
            speech_pct = audio.get("speech_ratio", 0.0) * 100.0
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Speech Activity</div>
                    <div class="metric-val">{speech_pct:.1f}%</div>
                    <div class="metric-sub">Silence: {audio.get('silence_ratio', 0.0)*100:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Main Overview Charts (Engagement Trend & Attendance Breakdown)
        left_col, right_col = st.columns([7, 5])

        with left_col:
            st.markdown("#### **Engagement Dynamics Over Time**")
            timeline = eng.get("timeline", [])
            if timeline:
                df_time = pd.DataFrame(timeline)
                fig = go.Figure()

                # Add engagement score line
                fig.add_trace(
                    go.Scatter(
                        x=df_time["timestamp_sec"],
                        y=df_time["engagement_score"],
                        mode="lines+markers",
                        name="Engagement Score",
                        line=dict(color="#38bdf8", width=3),
                        marker=dict(size=4),
                    )
                )

                # High engagement threshold line
                fig.add_hline(y=70, line_dash="dash", line_color="#10b981", annotation_text="High (70)")
                # Low engagement threshold line
                fig.add_hline(y=40, line_dash="dash", line_color="#ef4444", annotation_text="Low (40)")

                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#111827",
                    plot_bgcolor="#111827",
                    margin=dict(l=40, r=20, t=30, b=40),
                    height=320,
                    xaxis_title="Timeline (Seconds)",
                    yaxis_title="Score (0-100)",
                    yaxis=dict(range=[0, 105]),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No timeline metrics available for this session.")

        with right_col:
            st.markdown("#### **Attendance Status Distribution**")
            p_cnt = att.get("present_count", 0)
            part_cnt = att.get("partial_count", 0)
            abs_cnt = att.get("absent_count", 0)

            fig_pie = go.Figure(
                data=[
                    go.Pie(
                        labels=["Present", "Partial", "Absent"],
                        values=[p_cnt, part_cnt, abs_cnt],
                        hole=0.55,
                        marker=dict(colors=["#10b981", "#f59e0b", "#ef4444"]),
                    )
                ]
            )
            fig_pie.update_layout(
                template="plotly_dark",
                paper_bgcolor="#111827",
                plot_bgcolor="#111827",
                margin=dict(l=20, r=20, t=20, b=20),
                height=320,
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # Visual Keyframes Showcase (MediaPipe Face Detection with anonymous IDs)
        keyframes = vision.get("keyframes", [])
        if keyframes:
            st.markdown("#### **Participant Tracking Snapshots**")
            st.caption("MediaPipe visual tracking overlay with persistent anonymous track IDs.")
            cols = st.columns(len(keyframes))
            for i, kf in enumerate(keyframes):
                with cols[i]:
                    # Convert BGR to RGB for Streamlit image display
                    rgb_frame = kf["frame_bgr"][:, :, ::-1]
                    st.image(
                        rgb_frame,
                        caption=f"t = {kf.get('timestamp', 0):.1f}s",
                        use_container_width=True,
                    )

        # Ethical Transparency Disclaimer
        st.markdown(
            """
            <div class="disclaimer-banner">
                <b>Domain Notice:</b> Engagement scores represent rule-based indicators calculated from observable visual motion,
                facial presence, and acoustic speech patterns. They are designed to support instructional pacing and reflection,
                not as definitive or punitive measures of student comprehension. Anonymous track IDs protect participant privacy.
            </div>
            """,
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------------------
# VIEW 2: UPLOAD & ANALYZE
# -----------------------------------------------------------------------------
elif menu_selection == "📤 Upload & Analyze":
    st.markdown("### 📤 Upload Classroom Session")
    st.write("Upload a classroom video file for simultaneous face tracking, presence calculation, and acoustic speech analysis.")

    col_up, col_conf = st.columns([7, 5])

    with col_up:
        uploaded_video = st.file_uploader(
            "Upload Classroom Video (.mp4, .avi, .mov)",
            type=["mp4", "avi", "mov", "mkv"],
            help="Primary video recording of classroom desks and participants.",
        )

        uploaded_audio = st.file_uploader(
            "Optional: Upload Standalone Classroom Audio (.wav, .mp3)",
            type=["wav", "mp3", "m4a"],
            help="If provided, takes precedence over the video's embedded audio track.",
        )

        st.markdown("---")
        st.write("Or use pre-generated classroom test assets:")
        col_btn1, col_btn2 = st.columns(2)
        use_sample = col_btn1.button("⚡ Use Sample Classroom Data", use_container_width=True)

    with col_conf:
        st.markdown("#### **Analysis Settings**")
        present_thresh = st.slider("Present Threshold (%)", 50, 95, 70, help="Required presence duration to be classified as Present.")
        partial_thresh = st.slider("Partial Threshold (%)", 20, 60, 35, help="Presence duration below which is classified as Absent.")
        sample_fps = st.select_slider("Frame Sampling Rate (FPS)", options=[2.0, 5.0, 10.0, 15.0, 20.0], value=10.0)
        window_duration = st.select_slider("Sliding Window Interval (Seconds)", options=[0.5, 1.0, 2.0, 5.0], value=1.0)
        max_dur = st.number_input("Max Duration to Process (0 = Full Video)", min_value=0, max_value=7200, value=0)
        max_duration_arg = float(max_dur) if max_dur > 0 else None

    # Processing trigger
    start_analysis = st.button("🚀 Start Multimodal Analysis", type="primary", use_container_width=True)

    if start_analysis or use_sample:
        v_path = None
        a_path = None

        if use_sample or (not uploaded_video and not uploaded_audio):
            st.info("Loading synthetic demonstration classroom data...")
            v_path, a_path = get_sample_paths()
            source_label = "sample_classroom.mp4"
        else:
            temp_dir = tempfile.gettempdir()
            if uploaded_video:
                v_path = os.path.join(temp_dir, uploaded_video.name)
                with open(v_path, "wb") as f:
                    f.write(uploaded_video.getbuffer())
                source_label = uploaded_video.name

            if uploaded_audio:
                a_path = os.path.join(temp_dir, uploaded_audio.name)
                with open(a_path, "wb") as f:
                    f.write(uploaded_audio.getbuffer())
                if not uploaded_video:
                    source_label = uploaded_audio.name

        prog_bar = st.progress(0.0)
        status_text = st.empty()

        def on_progress(p: float, msg: str) -> None:
            prog_bar.progress(p)
            status_text.write(f"⏳ **{int(p*100)}%** — {msg}")

        agent = MultimodalAgent(
            present_threshold_pct=float(present_thresh),
            partial_threshold_pct=float(partial_thresh),
            sample_fps=float(sample_fps),
            window_duration_sec=float(window_duration),
        )

        try:
            results = agent.process_session(
                video_path=v_path,
                audio_path=a_path,
                max_duration=max_duration_arg,
                progress_callback=on_progress,
            )

            if results.get("success"):
                session_id = save_session(results)
                st.session_state["session_data"] = results
                st.session_state["last_processed_file"] = source_label
                st.success(f"Analysis complete in {results.get('processing_time_sec', 0):.1f}s! Session saved as `{session_id}`.")
                time.sleep(1.0)
                st.rerun()
            else:
                st.error(f"Analysis failed: {results.get('error')}")

        except Exception as e:
            st.error(f"Unexpected error during processing: {str(e)}")


# -----------------------------------------------------------------------------
# VIEW 3: ATTENDANCE RECORDS
# -----------------------------------------------------------------------------
elif menu_selection == "👥 Attendance Records":
    st.markdown("### 👥 Participant Attendance Records")
    st.write("Presence-based duration tracking. Anonymous Track IDs are assigned consistently across frames.")

    active_data = st.session_state["session_data"]
    if not active_data or not active_data.get("success"):
        st.warning("Please upload a video or load a session first.")
    else:
        att = active_data.get("attendance", {})
        records = att.get("records", [])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Participants", att.get("total_participants", 0))
        c2.metric("Present (≥ threshold)", att.get("present_count", 0))
        c3.metric("Partial", att.get("partial_count", 0))
        c4.metric("Absent (< threshold)", att.get("absent_count", 0))

        st.markdown("---")

        if records:
            df_att = generate_attendance_dataframe(records)

            # Interactive Table with Search / Sorting
            st.dataframe(
                df_att,
                use_container_width=True,
                column_config={
                    "Presence Rate (%)": st.column_config.ProgressColumn(
                        "Presence Rate (%)",
                        min_value=0,
                        max_value=100,
                        format="%.1f%%",
                    )
                },
            )

            # Download CSV
            csv_str = generate_attendance_csv(records)
            st.download_button(
                label="📥 Download Attendance CSV",
                data=csv_str,
                file_name=f"attendance_records_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                type="primary",
            )

            # Chart: Participant Presence Duration
            st.markdown("#### **Participant Presence Comparison**")
            fig_bar = px.bar(
                df_att,
                x="Participant ID",
                y="Presence Duration (s)",
                color="Attendance Status",
                color_discrete_map={"Present": "#10b981", "Partial": "#f59e0b", "Absent": "#ef4444"},
                text="Presence Duration (s)",
                template="plotly_dark",
            )
            fig_bar.update_layout(
                paper_bgcolor="#111827",
                plot_bgcolor="#111827",
                height=340,
                margin=dict(l=30, r=20, t=30, b=40),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        else:
            st.info("No participants were detected in this session video.")


# -----------------------------------------------------------------------------
# VIEW 4: ENGAGEMENT ANALYTICS
# -----------------------------------------------------------------------------
elif menu_selection == "📈 Engagement Analytics":
    st.markdown("### 📈 Multimodal Engagement Analytics")
    st.write("Detailed temporal breakdown of classroom attention combining visual motion, presence, and acoustic speech signals.")

    active_data = st.session_state["session_data"]
    if not active_data or not active_data.get("success"):
        st.warning("Please upload a video or load a session first.")
    else:
        eng = active_data.get("engagement", {})
        timeline = eng.get("timeline", [])

        # Duration Breakdown
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Average Score", f"{eng.get('average_score', 0):.1f}/100", eng.get("overall_level", "Moderate"))
        c2.metric("High Engagement Time", f"{eng.get('high_duration_sec', 0):.1f}s")
        c3.metric("Moderate Time", f"{eng.get('moderate_duration_sec', 0):.1f}s")
        c4.metric("Low Engagement Time", f"{eng.get('low_duration_sec', 0):.1f}s")

        st.markdown("---")

        if timeline:
            df_tl = generate_engagement_timeline_dataframe(timeline)

            # Interactive Multi-modal Time Series Chart
            st.markdown("#### **Fused Multimodal Signal Timeline**")
            fig_sync = go.Figure()

            # Fused Engagement Score
            fig_sync.add_trace(
                go.Scatter(
                    x=df_tl["Timestamp (s)"],
                    y=df_tl["Engagement Score"],
                    mode="lines+markers",
                    name="Fused Engagement Score",
                    line=dict(color="#38bdf8", width=3),
                )
            )

            # Visual Presence
            fig_sync.add_trace(
                go.Scatter(
                    x=df_tl["Timestamp (s)"],
                    y=df_tl["Visual Presence"] * 100,
                    mode="lines",
                    name="Visual Presence %",
                    line=dict(color="#a855f7", width=2, dash="dot"),
                )
            )

            # Visual Movement Activity
            fig_sync.add_trace(
                go.Scatter(
                    x=df_tl["Timestamp (s)"],
                    y=df_tl["Visual Activity"] * 100,
                    mode="lines",
                    name="Visual Movement Activity %",
                    line=dict(color="#10b981", width=1.5),
                )
            )

            # Audio Speech Activity
            fig_sync.add_trace(
                go.Scatter(
                    x=df_tl["Timestamp (s)"],
                    y=df_tl["Audio Speech Score"] * 100,
                    mode="lines",
                    name="Audio Speech Level %",
                    line=dict(color="#f59e0b", width=1.5),
                )
            )

            fig_sync.update_layout(
                template="plotly_dark",
                paper_bgcolor="#111827",
                plot_bgcolor="#111827",
                height=400,
                xaxis_title="Session Timeline (Seconds)",
                yaxis_title="Indicator Level (0-100)",
                yaxis=dict(range=[0, 105]),
                legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
            )
            st.plotly_chart(fig_sync, use_container_width=True)

            # Peaks and Lulls Inspection
            col_p, col_l = st.columns(2)
            with col_p:
                st.markdown("##### 🚀 **Peak Engagement Periods**")
                peaks = eng.get("peaks", [])
                if peaks:
                    for p in peaks:
                        st.success(
                            f"**{p['start_sec']:.1f}s - {p['end_sec']:.1f}s** ({p['duration_sec']:.1f}s duration) | "
                            f"Mean Score: **{p['mean_score']:.1f}/100**"
                        )
                else:
                    st.info("No extended peak periods detected.")

            with col_l:
                st.markdown("##### ⚠️ **Low-Activity / Lull Periods**")
                lulls = eng.get("lulls", [])
                if lulls:
                    for l in lulls:
                        st.error(
                            f"**{l['start_sec']:.1f}s - {l['end_sec']:.1f}s** ({l['duration_sec']:.1f}s duration) | "
                            f"Mean Score: **{l['mean_score']:.1f}/100**"
                        )
                else:
                    st.info("No extended lull periods detected.")

            # Timeline Data Table & Download
            st.markdown("---")
            st.markdown("##### **Detailed Window Breakdown**")
            st.dataframe(df_tl, use_container_width=True)

            csv_eng = generate_engagement_timeline_csv(timeline)
            st.download_button(
                label="📥 Download Engagement Timeline CSV",
                data=csv_eng,
                file_name=f"engagement_timeline_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )


# -----------------------------------------------------------------------------
# VIEW 5: TEACHING RECOMMENDATIONS
# -----------------------------------------------------------------------------
elif menu_selection == "💡 Teaching Suggestions":
    st.markdown("### 💡 Context-Aware Teaching Suggestions")
    st.write("Actionable suggestions derived directly from acoustic patterns, attendance dynamics, and engagement timelines.")

    active_data = st.session_state["session_data"]
    if not active_data or not active_data.get("success"):
        st.warning("Please upload a video or load a session first.")
    else:
        recs = active_data.get("recommendations", [])

        # Filters
        fcol1, fcol2 = st.columns(2)
        cat_filter = fcol1.selectbox("Filter by Category", ["All", "Pedagogy", "Interaction", "Acoustics", "Presence", "Pacing"])
        sev_filter = fcol2.selectbox("Filter by Severity", ["All", "Alert", "Suggestion", "Positive"])

        filtered_recs = []
        for r in recs:
            if cat_filter != "All" and r.get("category") != cat_filter:
                continue
            if sev_filter != "All" and r.get("severity") != sev_filter:
                continue
            filtered_recs.append(r)

        st.markdown(f"**Displaying {len(filtered_recs)} recommendation(s):**")

        for r in filtered_recs:
            sev = r.get("severity", "Suggestion").lower()
            badge_class = f"rec-{sev}"

            start_t = r.get("timestamp_start", 0.0)
            end_t = r.get("timestamp_end", 0.0)
            time_badge = f"{start_t:04.1f}s - {end_t:04.1f}s" if end_t > start_t else f"{start_t:04.1f}s"

            st.markdown(
                f"""
                <div class="rec-card {badge_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-weight: 700; font-size: 16px; color: #f8fafc;">{r.get('title')}</span>
                        <span style="font-size: 11px; background: #1e293b; padding: 2px 8px; border-radius: 4px; color: #94a3b8;">⏱ {time_badge}</span>
                    </div>
                    <div style="color: #cbd5e1; font-size: 13.5px; line-height: 1.5; margin-bottom: 6px;">
                        {r.get('rationale')}
                    </div>
                    <div style="font-size: 11px; color: #64748b; text-transform: uppercase; font-weight: 600;">
                        Category: <span style="color:#38bdf8;">{r.get('category')}</span> | Severity: <span style="color:#f59e0b;">{r.get('severity')}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if recs:
            csv_recs = generate_recommendations_csv(recs)
            st.download_button(
                label="📥 Download Teaching Suggestions CSV",
                data=csv_recs,
                file_name=f"teaching_suggestions_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )


# -----------------------------------------------------------------------------
# VIEW 6: REPORTS & ARCHIVE
# -----------------------------------------------------------------------------
elif menu_selection == "📑 Session Reports":
    st.markdown("### 📑 Reports & Session History")
    st.write("Export complete session reports in JSON, CSV, and Markdown formats, or browse previous sessions.")

    active_data = st.session_state["session_data"]

    t1, t2 = st.tabs(["📄 Current Session Report", "📂 Historical Sessions Archive"])

    with t1:
        if not active_data or not active_data.get("success"):
            st.warning("No active session loaded to export.")
        else:
            summary_md = generate_text_session_summary(active_data)

            st.markdown("#### **Executive Markdown Summary**")
            st.text_area("Summary Markdown", summary_md, height=260)

            # Download buttons
            c1, c2, c3, c4 = st.columns(4)

            # JSON
            json_str = io.StringIO()
            import json
            from utils.report_utils import _make_json_serializable

            json.dump(_make_json_serializable(active_data), json_str, indent=2)
            c1.download_button(
                "📥 Full JSON Report",
                data=json_str.getvalue(),
                file_name=f"classroom_session_{time.strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )

            # Attendance CSV
            att_records = active_data.get("attendance", {}).get("records", [])
            c2.download_button(
                "📥 Attendance CSV",
                data=generate_attendance_csv(att_records),
                file_name="attendance.csv",
                mime="text/csv",
                use_container_width=True,
            )

            # Engagement CSV
            eng_tl = active_data.get("engagement", {}).get("timeline", [])
            c3.download_button(
                "📥 Engagement CSV",
                data=generate_engagement_timeline_csv(eng_tl),
                file_name="engagement.csv",
                mime="text/csv",
                use_container_width=True,
            )

            # Markdown Summary file
            c4.download_button(
                "📥 Markdown Summary",
                data=summary_md,
                file_name="session_summary.md",
                mime="text/markdown",
                use_container_width=True,
            )

    with t2:
        saved = list_saved_sessions()
        if not saved:
            st.info("No saved sessions in `data/reports/` yet.")
        else:
            st.markdown(f"Found **{len(saved)}** saved classroom session(s):")
            for s in saved:
                col_info, col_act = st.columns([8, 2])
                with col_info:
                    st.write(
                        f"**{s['filename']}** (`{s['session_id']}`) — "
                        f"Duration: `{s['duration_sec']:.1f}s` | "
                        f"Students: `{s['participants']}` | "
                        f"Score: `{s['average_engagement']:.1f}/100`"
                    )
                    st.caption(f"Saved at: {s['saved_at']}")
                with col_act:
                    if st.button("Load", key=f"load_{s['session_id']}", use_container_width=True):
                        loaded = load_session(s["session_id"])
                        if loaded:
                            st.session_state["session_data"] = loaded
                            st.success(f"Loaded {s['session_id']}")
                            st.rerun()
                st.markdown("---")
