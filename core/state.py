"""
Thread-safe System State and Atomic Parameter Bus for IGARD-Net.
Maintains state across real-time audio thread, background worker, and dashboard server.
"""

import threading
import time
from typing import Dict, Any, List
import numpy as np


class SystemState:
    def __init__(self):
        self._lock = threading.Lock()
        
        # Operational modes
        self.mode: str = "idle"              # "idle", "realtime", "file"
        self.is_running: bool = False
        self.autopilot: bool = True          # Automatic processing aggressiveness
        self.before_after: bool = True       # False: Raw Audio, True: Enhanced Audio
        
        # Live Filter & DSP State
        self.mu_current: float = 0.25        # Atomically updated by GQPSO background tuner
        self.noise_label: str = "standby"
        self.confidence: float = 0.0
        self.shock_score: float = 0.0
        self.shock_confirmed: bool = False
        self.noise_present: bool = False
        self.noise_floor: float = 1.0e-5
        self.active_stage: int = 0
        self.stage_status: str = "System ready."
        
        # Audio Levels & Quality
        self.primary_rms: float = 0.0
        self.reference_rms: float = 0.0
        self.output_rms: float = 0.0
        self.snr_delta: float = 0.0
        self.snr_ema: float = 0.0
        
        # Latency & Throughput Metrics (Real measured values)
        self.latency_capture_ms: float = 0.0
        self.latency_process_ms: float = 0.0
        self.latency_queue_ms: float = 0.0
        self.latency_output_ms: float = 0.0
        self.latency_total_ms: float = 0.0
        
        self.recent_latencies: List[float] = []
        self.avg_latency_ms: float = 0.0
        self.p95_latency_ms: float = 0.0
        self.max_latency_ms: float = 0.0
        
        self.frames_processed: int = 0
        self.dropped_frames: int = 0
        self.buffer_underruns: int = 0
        self.buffer_overruns: int = 0
        
        # Hardware Metrics
        self.cpu_percent: float = 0.0
        self.gpu_percent: float = 0.0
        self.ram_mb: float = 0.0
        self.active_backend: str = "CPU"
        self.hardware_profile: str = "CPU_ONLY"
        
        # File Mode State
        self.current_file: str = ""
        self.processed_file: str = ""
        self.file_duration_s: float = 0.0
        self.file_processing_time_s: float = 0.0
        self.file_rtf: float = 0.0
        self.file_progress: float = 0.0
        self.file_status: str = "Idle"
        
        self.last_update_ts: float = time.time()

    def record_latency(self, capture_ms: float, process_ms: float, queue_ms: float, output_ms: float):
        total = capture_ms + process_ms + queue_ms + output_ms
        with self._lock:
            self.latency_capture_ms = capture_ms
            self.latency_process_ms = process_ms
            self.latency_queue_ms = queue_ms
            self.latency_output_ms = output_ms
            self.latency_total_ms = total
            
            self.recent_latencies.append(total)
            if len(self.recent_latencies) > 100:
                self.recent_latencies.pop(0)
            
            if self.recent_latencies:
                self.avg_latency_ms = float(np.mean(self.recent_latencies))
                self.max_latency_ms = float(np.max(self.recent_latencies))
                self.p95_latency_ms = float(np.percentile(self.recent_latencies, 95))
            self.frames_processed += 1

    def update_step_size(self, new_mu: float):
        with self._lock:
            # Bound and smoothly update step size
            self.mu_current = float(np.clip(new_mu, 0.01, 1.8))

    def set_before_after(self, enhanced_enabled: bool):
        with self._lock:
            self.before_after = bool(enhanced_enabled)

    def set_autopilot(self, autopilot_enabled: bool):
        with self._lock:
            self.autopilot = bool(autopilot_enabled)

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "mode": self.mode,
                "is_running": self.is_running,
                "autopilot": self.autopilot,
                "before_after": self.before_after,
                "mu_current": round(self.mu_current, 4),
                "noise_label": self.noise_label,
                "confidence": round(self.confidence, 3),
                "shock_score": round(self.shock_score, 3),
                "shock_confirmed": self.shock_confirmed,
                "noise_present": self.noise_present,
                "noise_floor": self.noise_floor,
                "active_stage": self.active_stage,
                "stage_status": self.stage_status,
                "primary_rms": round(self.primary_rms, 3),
                "reference_rms": round(self.reference_rms, 3),
                "output_rms": round(self.output_rms, 3),
                "snr_delta": round(self.snr_delta, 2),
                "snr_ema": round(self.snr_ema, 2),
                "latency_capture_ms": round(self.latency_capture_ms, 2),
                "latency_process_ms": round(self.latency_process_ms, 2),
                "latency_queue_ms": round(self.latency_queue_ms, 2),
                "latency_output_ms": round(self.latency_output_ms, 2),
                "latency_total_ms": round(self.latency_total_ms, 2),
                "avg_latency_ms": round(self.avg_latency_ms, 2),
                "p95_latency_ms": round(self.p95_latency_ms, 2),
                "max_latency_ms": round(self.max_latency_ms, 2),
                "frames_processed": self.frames_processed,
                "dropped_frames": self.dropped_frames,
                "buffer_underruns": self.buffer_underruns,
                "buffer_overruns": self.buffer_overruns,
                "cpu_percent": round(self.cpu_percent, 1),
                "gpu_percent": round(self.gpu_percent, 1),
                "ram_mb": round(self.ram_mb, 1),
                "active_backend": self.active_backend,
                "hardware_profile": self.hardware_profile,
                "current_file": self.current_file,
                "processed_file": self.processed_file,
                "file_duration_s": round(self.file_duration_s, 2),
                "file_processing_time_s": round(self.file_processing_time_s, 3),
                "file_rtf": round(self.file_rtf, 3),
                "file_progress": round(self.file_progress, 1),
                "file_status": self.file_status,
            }
