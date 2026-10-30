"""Audio module for Multimodal Attendance & Engagement Agent.

Implements:
1. AudioAnalyzer: Extraction from video or standalone audio files,
   resampling to 16 kHz mono, sliding-window chunking, RMS energy,
   decibel estimation, adaptive speech/silence detection, noise floor calculation,
   and speech-to-silence ratios.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils.audio_utils import extract_audio_from_video, load_and_resample_audio


class AudioAnalyzer:
    """Analyzes classroom audio signals for acoustic energy, speech, silence, and noise levels."""

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration_sec: float = 1.0,
        silence_threshold_db: float = -42.0,
        speech_energy_ratio: float = 1.6,
    ):
        """Initialize AudioAnalyzer.

        Args:
            sample_rate: Target sample rate in Hz (standardized to 16000).
            chunk_duration_sec: Fixed window duration for time-synchronized chunking.
            silence_threshold_db: Decibel threshold below which sound is considered silence.
            speech_energy_ratio: Multiplier above estimated noise floor to classify speech activity.
        """
        self.sample_rate = sample_rate
        self.chunk_duration_sec = chunk_duration_sec
        self.silence_threshold_db = silence_threshold_db
        self.speech_energy_ratio = speech_energy_ratio

    def process_video_or_audio(
        self,
        video_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        max_duration: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Process audio from either a video file, separate audio file, or both.

        Args:
            video_path: Path to classroom video (audio extracted automatically).
            audio_path: Path to separate audio file (preferred if explicitly supplied).
            max_duration: Maximum duration in seconds to analyze.

        Returns:
            Dictionary containing audio chunk timeline, metrics, noise floor,
            speech/silence ratios, and status.
        """
        extracted_temp_file = None
        target_audio_file = audio_path

        # If separate audio not provided, extract from video
        if not target_audio_file and video_path:
            extracted_path, extract_meta = extract_audio_from_video(video_path)
            if extracted_path:
                target_audio_file = extracted_path
                extracted_temp_file = extracted_path
            else:
                # Video has no audio track
                return self._create_empty_or_silent_result(
                    reason="No audio track found in the uploaded video",
                    duration_sec=0.0,
                )

        if not target_audio_file:
            return self._create_empty_or_silent_result(
                reason="No audio file or video provided",
                duration_sec=0.0,
            )

        # Load and standardize audio waveform
        waveform, sr, load_meta = load_and_resample_audio(target_audio_file, target_sr=self.sample_rate)

        # Clean up temporary extracted file if created
        if extracted_temp_file and os.path.exists(extracted_temp_file):
            try:
                os.remove(extracted_temp_file)
            except OSError:
                pass

        if not load_meta["valid"] or waveform is None or len(waveform) == 0:
            return self._create_empty_or_silent_result(
                reason=load_meta.get("error", "Failed to load audio"),
                duration_sec=0.0,
            )

        # Truncate if max_duration specified
        if max_duration:
            max_samples = int(max_duration * self.sample_rate)
            if len(waveform) > max_samples:
                waveform = waveform[:max_samples]

        total_samples = len(waveform)
        duration_sec = float(total_samples / self.sample_rate)

        # Compute dynamic noise floor (10th percentile of local energy)
        # to adapt to quiet vs loud classroom recording hardware
        chunk_size = int(self.chunk_duration_sec * self.sample_rate)
        if chunk_size <= 0:
            chunk_size = self.sample_rate

        total_chunks = max(1, int(math.ceil(total_samples / chunk_size)))
        chunk_rms_list: List[float] = []

        for i in range(total_chunks):
            start = i * chunk_size
            end = min(total_samples, (i + 1) * chunk_size)
            chunk = waveform[start:end]
            if len(chunk) == 0:
                continue
            rms = float(np.sqrt(np.mean(chunk**2)))
            chunk_rms_list.append(rms)

        # Robust noise floor estimation: 15th percentile of non-zero chunks
        valid_rms = [r for r in chunk_rms_list if r > 1e-6]
        if valid_rms:
            noise_floor_rms = float(np.percentile(valid_rms, 15))
            noise_floor_db = float(20 * np.log10(max(1e-5, noise_floor_rms)))
        else:
            noise_floor_rms = 0.005
            noise_floor_db = -46.0

        speech_energy_threshold = max(0.015, noise_floor_rms * self.speech_energy_ratio)

        # Detailed chunk analysis
        chunks_timeline: List[Dict[str, Any]] = []
        speech_chunks_count = 0
        silence_chunks_count = 0
        noise_chunks_count = 0

        for i in range(total_chunks):
            start = i * chunk_size
            end = min(total_samples, (i + 1) * chunk_size)
            chunk = waveform[start:end]
            t_start = float(i * self.chunk_duration_sec)
            t_end = min(duration_sec, float((i + 1) * self.chunk_duration_sec))

            if len(chunk) == 0:
                continue

            rms = float(np.sqrt(np.mean(chunk**2)))
            db = float(20 * np.log10(max(1e-5, rms)))
            peak_amp = float(np.max(np.abs(chunk)))

            # Zero-crossing rate (indicates unvoiced consonants or high frequency noise)
            zcr = float(np.mean(np.abs(np.diff(np.signbit(chunk))))) if len(chunk) > 1 else 0.0

            # Classification
            if db < self.silence_threshold_db or rms < (noise_floor_rms * 1.15):
                state = "Silence"
                silence_chunks_count += 1
                speech_confidence = 0.05
            elif rms >= speech_energy_threshold:
                # Differentiate speech vs pure continuous high-frequency noise
                if zcr > 0.45:
                    state = "Noise"
                    noise_chunks_count += 1
                    speech_confidence = 0.25
                else:
                    state = "Speech"
                    speech_chunks_count += 1
                    speech_confidence = min(1.0, (rms / (speech_energy_threshold * 2.0)))
            else:
                state = "Ambient"
                noise_chunks_count += 1
                speech_confidence = 0.20

            chunks_timeline.append(
                {
                    "timestamp_sec": t_start,
                    "window_end_sec": t_end,
                    "chunk_idx": i,
                    "rms_energy": float(rms),
                    "decibels": float(db),
                    "peak_amplitude": float(peak_amp),
                    "state": state,  # Speech, Silence, Ambient, Noise
                    "speech_confidence": float(speech_confidence),
                    "zero_crossing_rate": float(zcr),
                }
            )

        processed_chunks = len(chunks_timeline)
        speech_ratio = (speech_chunks_count / processed_chunks) if processed_chunks > 0 else 0.0
        silence_ratio = (silence_chunks_count / processed_chunks) if processed_chunks > 0 else 0.0
        ambient_noise_ratio = (noise_chunks_count / processed_chunks) if processed_chunks > 0 else 0.0

        avg_db = float(np.mean([c["decibels"] for c in chunks_timeline])) if chunks_timeline else -60.0

        return {
            "has_audio": True,
            "error": None,
            "duration_sec": float(duration_sec),
            "sample_rate": self.sample_rate,
            "total_samples": total_samples,
            "noise_floor_db": float(noise_floor_db),
            "average_decibels": float(avg_db),
            "speech_ratio": float(speech_ratio),
            "silence_ratio": float(silence_ratio),
            "ambient_noise_ratio": float(ambient_noise_ratio),
            "speech_chunks": speech_chunks_count,
            "silence_chunks": silence_chunks_count,
            "noise_chunks": noise_chunks_count,
            "total_chunks": processed_chunks,
            "timeline": chunks_timeline,
        }

    def _create_empty_or_silent_result(self, reason: str, duration_sec: float = 0.0) -> Dict[str, Any]:
        """Return structured fallback result when audio is missing or invalid."""
        return {
            "has_audio": False,
            "error": reason,
            "duration_sec": float(duration_sec),
            "sample_rate": self.sample_rate,
            "total_samples": 0,
            "noise_floor_db": -60.0,
            "average_decibels": -60.0,
            "speech_ratio": 0.0,
            "silence_ratio": 1.0,
            "ambient_noise_ratio": 0.0,
            "speech_chunks": 0,
            "silence_chunks": 0,
            "noise_chunks": 0,
            "total_chunks": 0,
            "timeline": [],
        }
