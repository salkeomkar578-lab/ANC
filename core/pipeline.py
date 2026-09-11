"""
IGARD-Net V3 — Speech-First Real-Time Noise Suppression Pipeline.
Coordinates Silero VAD (ONNX), native RNNoise, and the Speech Protection Controller.

V3 Processing Pipeline:
Microphone / Audio Buffer
    ↓
Input Normalization
    ↓
16 kHz Mono Stream ──→ Silero VAD ──→ Continuous Speech Probability
                           │
                           ↓
RNNoise (48 kHz C-Engine) ──→ Speech Protection Controller (Collapse Guard & Preservation)
                           │
                           ↓
                     Gain Smoothing & Limiter
                           │
                           ↓
                    Audio Output / Speaker

Decoupled architecture: Audio thread only runs lightweight DSP.
Background tasks handle telemetry, statistics, and objective quality scoring.
"""

import time
from typing import Dict, Any, Tuple, Optional
import numpy as np
import scipy.signal as signal

from core.config import load_config
from core.state import SystemState
from backends.base import ComputeBackend
from backends.backend_manager import BackendManager
from vad.silero_vad import SileroVAD
from processing.rnnoise_processor import RNNoiseProcessor
from processing.speech_protection import SpeechProtectionController, SpeechProtectionGate
from processing.gain_smoother import GainSmoother
from processing.presence_gate import NoisePresenceGate
from processing.noise_classifier import NoiseClassifier
from processing.mos_estimator import MOSEstimator
from sensors.accelerometer import AccelerometerBase, cross_check_confidence
from sensors.mock_sensor import MockAccelerometer


class IgardNetPipeline:
    """
    IGARD-Net V3 Real-Time Speech Enhancement Pipeline.
    Strictly preserves human voice while suppressing environmental noise.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        state: Optional[SystemState] = None,
        backend: Optional[ComputeBackend] = None,
        accelerometer: Optional[AccelerometerBase] = None,
        enable_background_tuner: bool = False,
    ):
        self.config = config if config is not None else load_config()
        self.state = state if state is not None else SystemState()

        # Hardware backend detection
        if backend is None:
            mgr = BackendManager(
                requested_profile=self.config.get("hardware", {}).get("profile", "auto"),
                device_id=self.config.get("hardware", {}).get("cuda_device_id", 0),
            )
            self.backend = mgr.backend
            self.state.active_backend = self.backend.name
            self.state.hardware_profile = mgr.profile
        else:
            self.backend = backend
            self.state.active_backend = backend.name

        self.sample_rate = self.config.get("audio", {}).get("sample_rate", 16000)
        self.frame_size = self.config.get("audio", {}).get("frame_size", 256)

        # 1. Silero VAD (ONNX streaming speech detector)
        self.silero_vad = SileroVAD(force_cpu=True)
        self.speech_detector = self.silero_vad

        # 2. Native RNNoise C Engine
        self.rnnoise = RNNoiseProcessor()

        # 3. Speech Protection Controller
        sp_cfg = self.config.get("speech_protection", {})
        self.speech_protection = SpeechProtectionController(
            sample_rate=self.sample_rate,
            frame_size=self.frame_size,
            high_speech_thresh=sp_cfg.get("high_speech_thresh", 0.75),
            low_speech_thresh=sp_cfg.get("low_speech_thresh", 0.45),
            attack_ms=sp_cfg.get("attack_ms", 12.0),
            release_ms=sp_cfg.get("release_ms", 120.0),
            hold_ms=sp_cfg.get("hold_ms", 150.0),
            default_suppression=sp_cfg.get("default_suppression", 0.75),
        )

        # 4. Gain Smoother and Peak Limiter
        self.gain_smoother = GainSmoother(
            sample_rate=self.sample_rate,
            frame_size=self.frame_size,
            attack_ms=10.0,
            release_ms=80.0,
            max_delta_per_frame=0.08,
            min_gain=0.01,
        )

        # 5. Acoustic Presence & Classifier (for telemetry labels)
        self.presence_gate = NoisePresenceGate()
        self.classifier = NoiseClassifier(sample_rate=self.sample_rate)

        # 6. Accelerometer fusion
        self.accelerometer = accelerometer if accelerometer is not None else MockAccelerometer()
        self.shock_threshold = self.config.get("accelerometer", {}).get("shock_threshold", 0.5)

        # 7. Objective MOS Estimator
        self.mos_estimator = MOSEstimator(sample_rate=self.sample_rate)

        # State tracking
        self._block_count: int = 0
        self._last_out_sample: Optional[float] = None
        self._snr_ema: float = 0.0

    def process_block(
        self,
        primary_block: np.ndarray,
        reference_block: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Process a single audio frame with deterministic low latency (< 1 ms).
        Returns cleaned audio and compact telemetry.
        """
        t_start = time.perf_counter()

        primary = np.asarray(primary_block, dtype=np.float32)
        n_samples = len(primary)
        if reference_block is None:
            reference = np.random.normal(0, 0.01, n_samples).astype(np.float32)
        else:
            reference = np.asarray(reference_block, dtype=np.float32)

        primary_rms = float(np.sqrt(np.mean(primary ** 2))) + 1e-12
        ref_rms = float(np.sqrt(np.mean(reference ** 2))) + 1e-12

        # Sync autopilot setting from system state
        self.speech_protection.autopilot = bool(self.state.autopilot)

        # Stage 1: Silero VAD (16 kHz continuous speech detection)
        if self.sample_rate == 16000:
            speech_prob = self.silero_vad.process_chunk(primary)
        else:
            # Downsample to 16 kHz for VAD
            vad_in = signal.resample_poly(primary, 16000, self.sample_rate)
            speech_prob = self.silero_vad.process_chunk(vad_in)

        # Stage 2: RNNoise Suppression (48 kHz native C engine)
        if self.sample_rate == 48000:
            rnnoise_out, rnnoise_vad = self.rnnoise.process_chunk(primary)
        else:
            # Resample to 48 kHz for RNNoise C engine
            primary_48k = signal.resample_poly(primary, 48000, self.sample_rate)
            rnnoise_out_48k, rnnoise_vad = self.rnnoise.process_chunk(primary_48k)
            # Resample back to pipeline sample rate
            rnnoise_out = signal.resample_poly(rnnoise_out_48k, self.sample_rate, 48000)[:n_samples]

        # Stage 3: Speech Protection Controller (Enforces preservation & collapse guard)
        protected_audio, telem = self.speech_protection.protect(
            raw_frame=primary,
            suppressed_frame=rnnoise_out,
            speech_probability=speech_prob,
            rnnoise_vad=rnnoise_vad,
        )

        # Stage 4: Gain Smoothing & Soft Limiter
        smoothed_audio = self.gain_smoother.smooth_boundary(protected_audio, self._last_out_sample)
        output_audio = GainSmoother.soft_limit(smoothed_audio, ceiling=0.95)

        if len(output_audio) > 0:
            self._last_out_sample = float(output_audio[-1])

        # Stage 5: Before / After Multiplexer
        # If before_after is True, output enhanced audio; else pass raw
        if self.state.before_after:
            final_audio = output_audio
        else:
            final_audio = primary.copy()

        # Telemetry & Status computation
        clean_rms = float(np.sqrt(np.mean(output_audio ** 2))) + 1e-12
        raw_snr = 20.0 * np.log10(primary_rms / (ref_rms + 1e-6))
        enh_snr = 20.0 * np.log10(clean_rms / (ref_rms + 1e-6))
        snr_delta = float(np.clip(enh_snr - raw_snr, -30.0, 30.0))
        self._snr_ema = 0.90 * self._snr_ema + 0.10 * snr_delta

        # Acoustic classification (non-blocking)
        noise_label, raw_confidence = self.classifier.classify(reference)
        shock_score = float(self.accelerometer.read_recent_shock_score())
        confidence = cross_check_confidence(
            acoustic_label=noise_label,
            acoustic_confidence=raw_confidence,
            shock_score=shock_score,
            shock_threshold=self.shock_threshold,
        )
        shock_confirmed = (noise_label == "impulsive") and (shock_score >= self.shock_threshold)

        t_proc_ms = (time.perf_counter() - t_start) * 1000.0
        self.state.record_latency(process_ms=t_proc_ms)

        self._block_count += 1
        self.state.frames_processed = self._block_count
        self.state.speech_prob = float(telem["speech_probability"])
        self.state.noise_confidence = float(telem["suppression_strength"])
        self.state.snr_delta = float(self._snr_ema)

        # Objective MOS calculation
        estimated_mos = self.mos_estimator.estimate_frame_mos(
            primary_block=primary,
            enhanced_block=output_audio,
            speech_prob=float(telem["speech_probability"]),
            snr_delta=float(self._snr_ema),
        )
        self.state.estimated_mos = float(estimated_mos)

        status_text = telem["status"]
        if shock_confirmed:
            status_text = f"⚡ SHOCK CONFIRMED ({confidence*100:.0f}%)"

        return {
            "audio": final_audio,
            "raw_audio": primary,
            "processed_audio": output_audio,
            "enhanced_audio": output_audio,
            "speech_prob": float(telem["speech_probability"]),
            "noise_prob": float(1.0 - telem["speech_probability"]),
            "suppression_strength": float(telem["suppression_strength"]),
            "estimated_mos": float(estimated_mos),
            "noise_label": noise_label,
            "confidence": float(confidence),
            "shock_score": float(shock_score),
            "shock_confirmed": shock_confirmed,
            "used_cleanup_stage": True,
            "active_stage": 3,
            "stage_status": status_text,
            "primary_rms": primary_rms,
            "processed_rms": clean_rms,
            "snr_delta": float(self._snr_ema),
            "processing_time_ms": t_proc_ms,
            "block_count": self._block_count,
        }

    def process_stream(
        self,
        primary: np.ndarray,
        reference: Optional[np.ndarray] = None,
        block_size: Optional[int] = None,
    ) -> Tuple[np.ndarray, list]:
        """
        Batch process an audio stream sequentially in contiguous blocks using
        the EXACT same V3 processing logic as real-time mode.
        """
        bs = block_size if block_size is not None else self.frame_size
        n = len(primary)
        if reference is None:
            reference = np.random.normal(0, 0.01, n).astype(np.float32)

        out_audio = np.empty(n, dtype=np.float32)
        telemetry_log = []

        for idx in range(0, n, bs):
            end = min(idx + bs, n)
            p_chunk = primary[idx:end]
            r_chunk = reference[idx:end]

            if len(p_chunk) < bs:
                pad_len = bs - len(p_chunk)
                p_pad = np.pad(p_chunk, (0, pad_len))
                r_pad = np.pad(r_chunk, (0, pad_len))
                res = self.process_block(p_pad, r_pad)
                out_audio[idx:end] = res["audio"][: end - idx]
            else:
                res = self.process_block(p_chunk, r_chunk)
                out_audio[idx:end] = res["audio"]

            telemetry_log.append({k: v for k, v in res.items() if k not in ("audio", "raw_audio", "processed_audio")})

        return out_audio, telemetry_log

    def reset(self):
        """Reset internal pipeline states."""
        self.silero_vad.reset()
        self.rnnoise.reset()
        self.speech_protection.reset()
        self.gain_smoother.reset(1.0)
        self._last_out_sample = None
        self._block_count = 0
        self._snr_ema = 0.0

    def close(self):
        """Cleanly releases all C/native resources."""
        if hasattr(self, "rnnoise"):
            del self.rnnoise
