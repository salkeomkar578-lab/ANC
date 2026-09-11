"""
Telemetry Dispatcher and Data Decimator for IGARD-Net.
Decouples audio DSP thread from WebSocket / HTTP dashboard rendering.
Pushes compact packets at a throttled 15-20 FPS into a bounded queue with drop-oldest policy.
"""

import queue
import time
from typing import Dict, Any, Optional
import numpy as np

from core.state import SystemState


def compute_compact_fft(block: np.ndarray, sample_rate: int = 16000, num_bins: int = 64) -> list:
    """Computes a lightweight 64-bin normalized magnitude spectrum [0..1] for UI waterfall."""
    n = len(block)
    if n == 0:
        return [0.0] * num_bins
    
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft(block * win))
    db = 20.0 * np.log10(spec + 1.0e-9)
    db = np.clip(db, -70.0, 0.0)
    norm = (db + 70.0) / 70.0

    # Downsample to num_bins
    n_spec = len(norm)
    if n_spec >= num_bins:
        step = n_spec / num_bins
        bins = [float(np.mean(norm[int(i * step) : int((i + 1) * step)])) for i in range(num_bins)]
    else:
        bins = [float(x) for x in np.interp(np.linspace(0, 1, num_bins), np.linspace(0, 1, n_spec), norm)]
    return bins


def decimate_waveform(block: np.ndarray, target_points: int = 128) -> list:
    """Decimates a block of samples to target_points floats for fast Canvas oscilloscope."""
    n = len(block)
    if n == 0:
        return [0.0] * target_points
    if n == target_points:
        return [float(x) for x in block]
    indices = np.linspace(0, n - 1, target_points).astype(int)
    return [float(block[i]) for i in indices]


class TelemetryDispatcher:
    def __init__(self, state: SystemState, target_fps: int = 20):
        self.state = state
        self.target_fps = target_fps
        self.interval_s = 1.0 / target_fps
        self.last_emit_time = 0.0
        
        # Bounded queue (maxsize=2) with drop-oldest policy
        self._queue = queue.Queue(maxsize=2)

    def push_telemetry(self, pipeline_result: Dict[str, Any]):
        """Non-blocking: called from real-time audio thread."""
        now = time.perf_counter()
        if now - self.last_emit_time < self.interval_s:
            return  # Rate-limit to target FPS

        self.last_emit_time = now

        # Build compact telemetry packet
        raw_audio = pipeline_result.get("raw_audio", np.zeros(128))
        enhanced_audio = pipeline_result.get("enhanced_audio", np.zeros(128))
        output_audio = pipeline_result.get("audio", enhanced_audio)

        packet = {
            "timestamp": time.time(),
            "mode": self.state.mode,
            "autopilot": self.state.autopilot,
            "before_after": self.state.before_after,
            "active_stage": pipeline_result.get("active_stage", 0),
            "stage_status": pipeline_result.get("stage_status", ""),
            "noise_label": pipeline_result.get("noise_label", "standby"),
            "confidence": round(float(pipeline_result.get("confidence", 0.0)), 3),
            "speech_prob": round(float(pipeline_result.get("speech_prob", 0.0)), 3),
            "suppression_strength": round(float(pipeline_result.get("suppression_strength", 0.75)), 3),
            "noise_prob": round(float(pipeline_result.get("noise_prob", 1.0)), 3),
            "estimated_mos": round(float(pipeline_result.get("estimated_mos", self.state.estimated_mos)), 2),
            "is_running": bool(self.state.is_running),
            "shock_score": round(float(pipeline_result.get("shock_score", 0.0)), 3),
            "shock_confirmed": bool(pipeline_result.get("shock_confirmed", False)),
            "used_cleanup_stage": bool(pipeline_result.get("used_cleanup_stage", False)),
            "nlms_step_size": round(float(pipeline_result.get("nlms_step_size", 0.25)), 4),
            "noise_present": bool(pipeline_result.get("noise_present", False)),
            "noise_floor": float(pipeline_result.get("noise_floor", 1.0e-5)),
            "primary_level": round(float(pipeline_result.get("primary_rms", 0.0)), 3),
            "reference_level": round(float(pipeline_result.get("reference_rms", 0.0)), 3),
            "output_level": round(float(pipeline_result.get("output_rms", 0.0)), 3),
            "estimated_snr": round(float(self.state.snr_ema), 2),
            "snr_delta": round(float(pipeline_result.get("snr_delta", 0.0)), 2),
            "blocks_processed": self.state.frames_processed,
            # Compact visualization vectors
            "primary_samples": decimate_waveform(raw_audio, 128),
            "output_samples": decimate_waveform(output_audio, 128),
            "primary_fft": compute_compact_fft(raw_audio, 16000, 64),
            "output_fft": compute_compact_fft(output_audio, 16000, 64),
            # Real measured latency metrics
            "latency_ms": round(self.state.latency_total_ms, 2),
            "avg_latency_ms": round(self.state.avg_latency_ms, 2),
            "p95_latency_ms": round(self.state.p95_latency_ms, 2),
            "max_latency_ms": round(self.state.max_latency_ms, 2),
            "proc_latency_ms": round(float(pipeline_result.get("proc_latency_ms", 0.0)), 2),
            "dropped_frames": self.state.dropped_frames,
            "buffer_underruns": self.state.buffer_underruns,
            "buffer_overruns": self.state.buffer_overruns,
            # Hardware telemetry
            "cpu_percent": round(self.state.cpu_percent, 1),
            "gpu_percent": round(self.state.gpu_percent, 1),
            "ram_mb": round(self.state.ram_mb, 1),
            "active_backend": self.state.active_backend,
            "hardware_profile": self.state.hardware_profile,
            # File progress
            "file_progress": round(self.state.file_progress, 1),
            "current_file": self.state.current_file,
            "processed_file": self.state.processed_file,
        }

        # Put with drop-oldest
        try:
            self._queue.put_nowait(packet)
        except queue.Full:
            try:
                _ = self._queue.get_nowait()
                self._queue.put_nowait(packet)
            except Exception:
                pass

    def get_telemetry(self, timeout: float = 0.05) -> Optional[Dict[str, Any]]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
