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
        track_cfg = self.config.get("speech_tracking", {})
        self.speech_protection = SpeechProtectionController(
            sample_rate=self.sample_rate,
            frame_size=self.frame_size,
            threshold_on=track_cfg.get("threshold_on", 0.60),
            threshold_off=track_cfg.get("threshold_off", 0.40),
            high_speech_thresh=sp_cfg.get("high_speech_thresh", 0.60),
            low_speech_thresh=sp_cfg.get("low_speech_thresh", 0.40),
            attack_ms=track_cfg.get("attack_ms", 10.0),
            release_ms=track_cfg.get("release_ms", 150.0),
            hold_ms=track_cfg.get("hold_ms", 120.0),
            default_suppression=sp_cfg.get("default_suppression", 0.75),
            minimum_voice_gain=sp_cfg.get("minimum_voice_gain", 0.55),
            background_attenuation_floor=sp_cfg.get("background_attenuation_floor", 0.15),
        )

        # 4. Gain Smoother and Peak Limiter
        self.gain_smoother = GainSmoother(
            sample_rate=self.sample_rate,
            frame_size=self.frame_size,
            attack_ms=10.0,
            release_ms=80.0,
            max_delta_per_frame=0.08,
            min_gain=0.15,
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
        self._last_noise_label: str = "ambient"
        self._last_confidence: float = 0.85
        self._last_mos: float = 4.0
        self._heavy_metric_interval: int = 12

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
            vad_in = primary[::3] if self.sample_rate == 48000 else signal.resample_poly(primary, 16000, self.sample_rate)
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

        # Fast speech onset fusion
        effective_sp = max(speech_prob, 0.70 if rnnoise_vad >= 0.80 and speech_prob >= 0.25 else speech_prob)

        # Stage 3: Speech Protection Controller (Enforces preservation & collapse guard)
        protected_audio, telem = self.speech_protection.protect(
            raw_frame=primary,
            suppressed_frame=rnnoise_out,
            speech_probability=effective_sp,
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

        # Downsampled acoustic classification & MOS estimation to keep DSP latency ultra-low (<1.5ms)
        if (self._block_count % self._heavy_metric_interval == 0):
            self._last_noise_label, raw_conf = self.classifier.classify(reference)
            shock_score = float(self.accelerometer.read_recent_shock_score())
            self._last_confidence = float(cross_check_confidence(
                acoustic_label=self._last_noise_label,
                acoustic_confidence=raw_conf,
                shock_score=shock_score,
                shock_threshold=self.shock_threshold,
            ))
            self._last_mos = float(self.mos_estimator.estimate_frame_mos(
                primary_block=primary,
                enhanced_block=output_audio,
                speech_prob=float(telem["speech_probability"]),
                snr_delta=float(self._snr_ema),
            ))
        else:
            shock_score = float(self.accelerometer.read_recent_shock_score())

        noise_label = self._last_noise_label
        confidence = self._last_confidence
        estimated_mos = self._last_mos
        shock_confirmed = (noise_label == "impulsive") and (shock_score >= self.shock_threshold)

        t_proc_ms = (time.perf_counter() - t_start) * 1000.0
        self.state.record_latency(process_ms=t_proc_ms)

        self._block_count += 1
        self.state.frames_processed = self._block_count
        self.state.speech_prob = float(telem["speech_probability"])
        self.state.noise_confidence = float(telem["suppression_strength"])
        self.state.snr_delta = float(self._snr_ema)
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
        progress_callback: Optional[Any] = None,
    ) -> Tuple[np.ndarray, list]:
        """
        High-throughput V4 continuous stream processing.
        Follows Section 20: Unified internal audio flow at 48 kHz with one controlled
        branch for Silero VAD at 16 kHz. Resamples once on input and once on output.
        """
        n_in = len(primary)
        if n_in == 0:
            return np.zeros(0, dtype=np.float32), []

        self.reset()
        from audio.resampler import resample_audio
        orig_sr = self.sample_rate

        if orig_sr != 48000:
            p_48k = resample_audio(primary, orig_sr, 48000).astype(np.float32)
        else:
            p_48k = np.asarray(primary, dtype=np.float32)

        frame_size_48k = 480
        n_frames = len(p_48k) // frame_size_48k
        out_48k = np.empty(len(p_48k), dtype=np.float32)
        telemetry_log = []
        last_out_sample = None

        controller_48k = SpeechProtectionController(
            sample_rate=48000,
            frame_size=frame_size_48k,
            threshold_on=0.60,
            threshold_off=0.40,
            attack_ms=10.0,
            hold_ms=120.0,
            release_ms=150.0,
            default_suppression=0.75,
            minimum_voice_gain=0.55,
            background_attenuation_floor=0.15,
        )
        controller_48k.autopilot = bool(self.state.autopilot)
        smoother_48k = GainSmoother(
            sample_rate=48000,
            frame_size=frame_size_48k,
            attack_ms=10.0,
            release_ms=80.0,
            max_delta_per_frame=0.08,
            min_gain=0.15,
        )

        for i in range(n_frames):
            chunk_48k = p_48k[i * frame_size_48k : (i + 1) * frame_size_48k]

            # Controlled branch for Silero VAD (48k -> 16k: 480 samples -> 160 samples)
            chunk_16k = chunk_48k[::3]
            sp_prob = self.silero_vad.process_chunk(chunk_16k)

            # Native RNNoise suppression (480 samples @ 48 kHz)
            supp_48k, rnn_vad = self.rnnoise.process_frame(chunk_48k)

            # Fast speech onset fusion
            effective_sp = max(sp_prob, 0.70 if rnn_vad >= 0.80 and sp_prob >= 0.25 else sp_prob)

            # Speech Protection Controller
            prot_48k, telem = controller_48k.protect(
                raw_frame=chunk_48k,
                suppressed_frame=supp_48k,
                speech_probability=effective_sp,
                rnnoise_vad=rnn_vad,
            )

            # Gain smoothing & peak limiter
            smooth_48k = smoother_48k.smooth_boundary(prot_48k, last_out_sample)
            final_48k = GainSmoother.soft_limit(smooth_48k, ceiling=0.95)
            last_out_sample = float(final_48k[-1])

            out_48k[i * frame_size_48k : (i + 1) * frame_size_48k] = final_48k
            telemetry_log.append(telem)

            if progress_callback and (i % 25 == 0 or i == n_frames - 1):
                try:
                    progress_callback((i + 1) / n_frames, telem.get("status", "Processing"))
                except Exception:
                    pass

        rem = len(p_48k) % frame_size_48k
        if rem > 0:
            tail = p_48k[n_frames * frame_size_48k :]
            pad_tail = np.zeros(frame_size_48k, dtype=np.float32)
            pad_tail[:rem] = tail
            supp_tail, rnn_vad = self.rnnoise.process_frame(pad_tail)
            out_48k[n_frames * frame_size_48k :] = supp_tail[:rem]

        if orig_sr != 48000:
            out_audio = resample_audio(out_48k, 48000, orig_sr).astype(np.float32)[:n_in]
        else:
            out_audio = out_48k[:n_in]

        return out_audio, telemetry_log

    def reset(self):
        """Reset internal pipeline states."""
        self.silero_vad.reset()
        self.rnnoise.reset()
        self.speech_protection.reset()
        self.gain_smoother.reset(1.0)
        self.presence_gate.recalibrate()
        self._last_out_sample = None
        self._block_count = 0
        self._snr_ema = 0.0
        self._last_noise_label = "ambient"
        self._last_confidence = 0.85
        self._last_mos = 4.0

    def close(self):
        """Cleanly releases all C/native resources."""
        if hasattr(self, "rnnoise"):
            del self.rnnoise
