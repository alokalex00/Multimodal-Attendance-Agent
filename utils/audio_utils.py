"""Audio utilities for Multimodal Attendance & Engagement Agent.

Handles audio extraction from video files, format conversion,
resampling to standard 16 kHz mono, and synthetic audio generation for testing.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Any, Dict, Optional, Tuple

import numpy as np


def extract_audio_from_video(
    video_path: str,
    output_audio_path: Optional[str] = None,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Extract audio track from a video file using available local tools.

    Checks for moviepy, imageio_ffmpeg, or system ffmpeg.
    If no audio stream is present, gracefully returns None.

    Args:
        video_path: Path to the input video file.
        output_audio_path: Optional destination path (.wav). If None, creates a temp file.

    Returns:
        (extracted_audio_path or None, metadata_dict)
    """
    if not os.path.exists(video_path):
        return None, {"success": False, "error": f"Video file not found: {video_path}"}

    if output_audio_path is None:
        temp_dir = tempfile.gettempdir()
        output_audio_path = os.path.join(temp_dir, f"extracted_{os.path.basename(video_path)}.wav")

    # Method 1: Try imageio_ffmpeg (bundled binary)
    try:
        import imageio_ffmpeg

        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_bin,
            "-y",  # Overwrite
            "-i",
            video_path,
            "-vn",  # No video
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            output_audio_path,
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0 and os.path.exists(output_audio_path) and os.path.getsize(output_audio_path) > 100:
            return output_audio_path, {"success": True, "method": "imageio_ffmpeg"}
        # If ffmpeg reported no audio stream:
        if "does not contain any stream" in res.stderr.lower() or "matches no streams" in res.stderr.lower():
            return None, {"success": False, "error": "Video contains no audio stream"}
    except Exception:
        pass

    # Method 2: Try moviepy if available
    try:
        from moviepy.editor import VideoFileClip

        clip = VideoFileClip(video_path)
        if clip.audio is None:
            clip.close()
            return None, {"success": False, "error": "Video contains no audio track"}

        clip.audio.write_audiofile(output_audio_path, fps=16000, nbytes=2, logger=None)
        clip.close()
        if os.path.exists(output_audio_path) and os.path.getsize(output_audio_path) > 100:
            return output_audio_path, {"success": True, "method": "moviepy"}
    except Exception:
        pass

    # Method 3: Try system ffmpeg if installed
    ffmpeg_system = shutil.which("ffmpeg")
    if ffmpeg_system:
        try:
            cmd = [
                ffmpeg_system,
                "-y",
                "-i",
                video_path,
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                output_audio_path,
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.returncode == 0 and os.path.exists(output_audio_path) and os.path.getsize(output_audio_path) > 100:
                return output_audio_path, {"success": True, "method": "system_ffmpeg"}
        except Exception:
            pass

    return None, {
        "success": False,
        "error": "No audio stream found or audio extraction tool unavailable for this format",
    }


def load_and_resample_audio(
    audio_path: str,
    target_sr: int = 16000,
) -> Tuple[Optional[np.ndarray], int, Dict[str, Any]]:
    """Load an audio file, normalize amplitude, convert to mono, and resample.

    Args:
        audio_path: Path to the audio file (.wav, .mp3, etc.).
        target_sr: Standard target sampling rate in Hz (default 16000).

    Returns:
        (waveform_float32, sampling_rate, metadata_dict)
    """
    if not os.path.exists(audio_path):
        return None, target_sr, {"valid": False, "error": f"Audio file not found: {audio_path}"}

    if os.path.getsize(audio_path) == 0:
        return None, target_sr, {"valid": False, "error": "Audio file is empty (0 bytes)"}

    # Attempt 1: soundfile (fast, reliable for WAV/FLAC)
    try:
        import soundfile as sf

        data, sr = sf.read(audio_path, dtype="float32")
        # Multi-channel to mono
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        # Resample if needed
        if sr != target_sr:
            from scipy import signal

            num_samples = int(round(len(data) * float(target_sr) / sr))
            data = signal.resample(data, num_samples).astype(np.float32)

        # Amplitude check
        peak = np.max(np.abs(data)) if len(data) > 0 else 0.0
        if peak > 1.0:
            data = data / peak

        duration = len(data) / target_sr
        return data, target_sr, {
            "valid": True,
            "error": None,
            "duration": float(duration),
            "original_sr": sr,
            "samples": len(data),
            "backend": "soundfile",
        }
    except Exception:
        pass

    # Attempt 2: librosa
    try:
        import librosa

        data, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        peak = np.max(np.abs(data)) if len(data) > 0 else 0.0
        if peak > 1.0:
            data = data / peak

        duration = len(data) / target_sr
        return data, target_sr, {
            "valid": True,
            "error": None,
            "duration": float(duration),
            "original_sr": target_sr,
            "samples": len(data),
            "backend": "librosa",
        }
    except Exception as e:
        return None, target_sr, {
            "valid": False,
            "error": f"Failed to decode audio file: {str(e)}",
        }


def generate_synthetic_classroom_audio(
    output_wav_path: str,
    duration_sec: float = 10.0,
    sr: int = 16000,
) -> str:
    """Synthesize classroom audio for testing and demonstrations.

    Generates realistic speech harmonics (teacher explanation, student question,
    brief silence interval, room tone noise floor) without requiring external audio assets.

    Args:
        output_wav_path: File path to save the generated WAV.
        duration_sec: Length of audio in seconds.
        sr: Sample rate in Hz.

    Returns:
        Absolute path to the created WAV file.
    """
    import scipy.io.wavfile as wav

    os.makedirs(os.path.dirname(os.path.abspath(output_wav_path)), exist_ok=True)
    total_samples = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, total_samples, endpoint=False)

    # Ambient room tone (gentle pink/white noise floor)
    rng = np.random.RandomState(42)
    noise_floor = rng.normal(0, 0.008, total_samples).astype(np.float32)

    speech_signal = np.zeros(total_samples, dtype=np.float32)

    def synthesize_speech_segment(start_t: float, end_t: float, f0: float, amp: float) -> None:
        idx_start = int(start_t * sr)
        idx_end = min(total_samples, int(end_t * sr))
        if idx_start >= idx_end:
            return
        seg_t = t[idx_start:idx_end] - start_t
        # Harmonic series with formant envelope
        voice = (
            0.6 * np.sin(2 * np.pi * f0 * seg_t)
            + 0.3 * np.sin(2 * np.pi * (2 * f0) * seg_t)
            + 0.15 * np.sin(2 * np.pi * (3 * f0) * seg_t)
            + 0.08 * np.sin(2 * np.pi * (4 * f0) * seg_t)
        )
        # Syllable modulation (~4 Hz speech rhythm)
        envelope = (np.sin(2 * np.pi * 3.8 * seg_t) ** 2) * amp
        speech_signal[idx_start:idx_end] += (voice * envelope).astype(np.float32)

    # Segment 1: Teacher lecture (0.5s to 3.5s)
    synthesize_speech_segment(0.5, 3.5, f0=135.0, amp=0.35)

    # Segment 2: Student asking question (3.8s to 5.2s)
    synthesize_speech_segment(3.8, 5.2, f0=190.0, amp=0.28)

    # Segment 3: Silence / pause (5.2s to 7.0s) -> only background noise

    # Segment 4: Teacher answering (7.2s to duration_sec)
    synthesize_speech_segment(7.2, duration_sec, f0=135.0, amp=0.32)

    combined = speech_signal + noise_floor
    # Normalize to -1.0 to 1.0 range
    peak = np.max(np.abs(combined))
    if peak > 0:
        combined = (combined / peak) * 0.85

    # Convert to 16-bit PCM for universal compatibility
    int16_audio = (combined * 32767).astype(np.int16)
    wav.write(output_wav_path, sr, int16_audio)

    return os.path.abspath(output_wav_path)
