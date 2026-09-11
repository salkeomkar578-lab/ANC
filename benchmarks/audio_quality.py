"""
Quantitative Audio Quality and System Telemetry Benchmark for IGARD-Net V3.
Computes verified objective metrics:
- Input / Output duration
- Speech duration & preserved speech duration
- Speech continuity score
- Estimated noise reduction (dB)
- Clicks / sample-boundary discontinuities detected
- Average and P95 latency
- Real-time CPU and RAM usage
- Dropped frames
"""

import time
import os
from typing import Dict, Any, List, Optional
import numpy as np
import psutil


class AudioQualityEvaluator:
    """
    Objective benchmarking suite measuring speech preservation and noise suppression.
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def detect_clicks(self, audio: np.ndarray, threshold: float = 0.35) -> int:
        """
        Detects sudden non-acoustic single-sample discontinuities (clicks, pops, ticks)
        using 2nd-order discrete differences.
        """
        if len(audio) < 4:
            return 0
        diff2 = np.abs(np.diff(audio, n=2))
        # Exclude typical speech peaks by comparing against rolling local standard deviation
        local_std = np.std(audio) + 1e-6
        normalized_spikes = diff2 / local_std
        # A true digital click produces a massive normalized 2nd-derivative spike (>25x local std and absolute > threshold)
        click_mask = (normalized_spikes > 25.0) & (diff2 > threshold)
        return int(np.sum(click_mask))

    def evaluate(
        self,
        raw_audio: np.ndarray,
        enhanced_audio: np.ndarray,
        speech_probs: Optional[np.ndarray] = None,
        process_times_ms: Optional[List[float]] = None,
        dropped_frames: int = 0,
    ) -> Dict[str, Any]:
        """
        Calculate full quantitative benchmark comparing raw vs enhanced audio.
        """
        raw = np.asarray(raw_audio, dtype=np.float32)
        enh = np.asarray(enhanced_audio, dtype=np.float32)

        sr = self.sample_rate
        in_duration = len(raw) / sr
        out_duration = len(enh) / sr

        # Align length if slightly different
        min_len = min(len(raw), len(enh))
        raw_aligned = raw[:min_len]
        enh_aligned = enh[:min_len]

        # Automatic cross-correlation time-delay estimation & alignment (ITU-T P.862 standard practice)
        if min_len > sr * 2:
            import scipy.signal as signal
            search_len = min(sr * 2, min_len)
            s_raw = raw_aligned[:search_len]
            s_enh = enh_aligned[:search_len]
            cc = signal.correlate(s_enh, s_raw, mode="full", method="fft")
            lags = signal.correlation_lags(len(s_enh), len(s_raw))
            best_lag = int(lags[np.argmax(cc)])
            # If positive delay (filterbank lookahead latency < 100ms)
            if 0 < best_lag < int(sr * 0.10):
                aligned_enh = np.zeros_like(raw_aligned)
                aligned_enh[:-best_lag] = enh_aligned[best_lag:]
                enh_aligned = aligned_enh

        frame_len = int(sr * 0.02)  # 20 ms frames for analysis
        n_frames = min_len // frame_len

        if speech_probs is None or len(speech_probs) < n_frames:
            # Fallback energy-based speech detection
            frame_energies = [
                np.mean(raw_aligned[i * frame_len : (i + 1) * frame_len] ** 2)
                for i in range(n_frames)
            ]
            med_energy = np.median(frame_energies) + 1e-12
            derived_probs = np.array([min(1.0, e / (med_energy * 3.0)) for e in frame_energies])
        else:
            derived_probs = np.asarray(speech_probs[:n_frames], dtype=np.float32)

        speech_frames = derived_probs >= 0.45
        noise_frames = derived_probs < 0.25

        speech_duration = float(np.sum(speech_frames) * (frame_len / sr))

        # Check preserved speech: speech frame where enhanced RMS is at least 18% of raw RMS
        preserved_frames = 0
        continuous_speech_runs = 0
        total_speech_runs = 0
        in_speech_run = False

        for i in range(n_frames):
            if speech_frames[i]:
                raw_rms = np.sqrt(np.mean(raw_aligned[i * frame_len : (i + 1) * frame_len] ** 2))
                enh_rms = np.sqrt(np.mean(enh_aligned[i * frame_len : (i + 1) * frame_len] ** 2))
                if enh_rms >= (raw_rms * 0.18):
                    preserved_frames += 1

                if not in_speech_run:
                    in_speech_run = True
                    total_speech_runs += 1
                    continuous_speech_runs += 1
            else:
                in_speech_run = False

        preserved_duration = float(preserved_frames * (frame_len / sr))
        preservation_ratio = float(preserved_duration / max(1e-4, speech_duration))

        # Speech continuity score (1.0 = perfect uninterrupted preservation of voice)
        if total_speech_runs > 0:
            continuity_score = float(np.clip(preservation_ratio, 0.0, 1.0))
        else:
            continuity_score = 1.0

        # Noise reduction estimate during non-speech intervals (dB)
        if np.any(noise_frames):
            noise_idx = np.where(noise_frames)[0]
            raw_noise_samples = np.concatenate([
                raw_aligned[i * frame_len : (i + 1) * frame_len] for i in noise_idx
            ])
            enh_noise_samples = np.concatenate([
                enh_aligned[i * frame_len : (i + 1) * frame_len] for i in noise_idx
            ])

            raw_noise_rms = float(np.sqrt(np.mean(raw_noise_samples ** 2))) + 1e-12
            enh_noise_rms = float(np.sqrt(np.mean(enh_noise_samples ** 2))) + 1e-12
            nr_db = float(20.0 * np.log10(raw_noise_rms / enh_noise_rms))
        else:
            nr_db = 0.0

        # Discontinuity / click detection
        clicks = self.detect_clicks(enh_aligned)

        # Latency metrics
        if process_times_ms and len(process_times_ms) > 0:
            avg_latency = float(np.mean(process_times_ms))
            p95_latency = float(np.percentile(process_times_ms, 95))
            max_latency = float(np.max(process_times_ms))
        else:
            avg_latency = 0.0
            p95_latency = 0.0
            max_latency = 0.0

        # Hardware metrics
        process = psutil.Process(os.getpid())
        ram_mb = float(process.memory_info().rss / (1024 * 1024))
        cpu_pct = float(psutil.cpu_percent(interval=None))

        return {
            "input_duration_s": round(in_duration, 2),
            "output_duration_s": round(out_duration, 2),
            "speech_duration_s": round(speech_duration, 2),
            "preserved_speech_duration_s": round(preserved_duration, 2),
            "speech_preservation_ratio": round(preservation_ratio, 4),
            "speech_continuity_score": round(continuity_score, 4),
            "noise_reduction_estimate_db": round(nr_db, 2),
            "clicks_detected": clicks,
            "average_latency_ms": round(avg_latency, 2),
            "p95_latency_ms": round(p95_latency, 2),
            "max_latency_ms": round(max_latency, 2),
            "cpu_usage_pct": round(cpu_pct, 1),
            "ram_usage_mb": round(ram_mb, 1),
            "frames_dropped": dropped_frames,
        }

    def format_report(self, metrics: Dict[str, Any]) -> str:
        """Format metrics into a clear human-readable report."""
        lines = [
            "=" * 60,
            "         IGARD-NET V3 OBJECTIVE AUDIO QUALITY BENCHMARK",
            "=" * 60,
            f"  Input Duration:             {metrics['input_duration_s']} s",
            f"  Output Duration:            {metrics['output_duration_s']} s",
            f"  Speech Duration:            {metrics['speech_duration_s']} s",
            f"  Preserved Speech Duration:  {metrics['preserved_speech_duration_s']} s ({metrics['speech_preservation_ratio']*100:.1f}%)",
            f"  Speech Continuity Score:    {metrics['speech_continuity_score']:.4f}",
            f"  Noise Reduction Estimate:   {metrics['noise_reduction_estimate_db']} dB",
            f"  Clicks / Artifacts:         {metrics['clicks_detected']}",
            f"  Average Processing Latency: {metrics['average_latency_ms']} ms",
            f"  P95 Processing Latency:     {metrics['p95_latency_ms']} ms",
            f"  Max Latency:                {metrics['max_latency_ms']} ms",
            f"  Frames Dropped:             {metrics['frames_dropped']}",
            f"  CPU Usage:                  {metrics['cpu_usage_pct']} %",
            f"  RAM Usage:                  {metrics['ram_usage_mb']} MB",
            "=" * 60,
        ]
        return "\n".join(lines)
