"""
Core IGARD-Net Signal Chain Pipeline Coordinator.
Reconstructed for low-latency real-time streaming, high-throughput file processing,
and speech-first preservation.

Signal Chain:
  Primary & Reference Mic
       │
  [Stage 0]  NoisePresenceGate         -> Quiet bypass or engage
       │
  [Stage 1]  SpeechDetector            -> Continuous speech_prob & noise_prob (with temporal hold)
       │
  [Stage 2]  NoiseClassifier          -> Acoustic label & confidence
  [Stage 2b] AccelerometerFusion       -> Physical shock cross-modal verification
       │
  [Stage 3]  NLMSFilter (64-tap)       -> Adaptive acoustic cancellation with DTD
  [Stage 3b] Asynchronous GQPSO Tuner  -> Multi-objective background optimization of mu
       │
  [Stage 4]  ConfidenceGate            -> Cleanup vs Fail-safe bypass arbiter
       │
  [Stage 5]  Wiener Soft Masking (OLA) -> Speech-aware spectral cleanup (300-3400 Hz protected)
       │
  [Stage 6]  SpeechProtectionGate      -> Strict speech preservation floor & zero hard mutes
       │
  [Stage 7]  Voice Protection Dynamics -> Compressor & peak limiter
       │
  [B/A Gate] Before / After Selector   -> Instant zero-cost audition toggle
       │
  Cleaned Output Audio Stream
"""

import time
from typing import Dict, Any, Tuple, Optional
import numpy as np

from core.config import load_config
from core.state import SystemState
from backends.base import ComputeBackend
from backends.backend_manager import BackendManager
from processing.presence_gate import NoisePresenceGate
from processing.noise_classifier import NoiseClassifier
from processing.nlms_filter import NLMSFilter
from processing.tuner import QuantumInspiredTuner, BackgroundTunerWorker
from processing.confidence_gate import ConfidenceGate
from processing.wiener_cleanup import ResidualCleanup
from processing.dynamics import VoiceProtectionDynamics
from sensors.accelerometer import AccelerometerBase, cross_check_confidence
from sensors.mock_sensor import MockAccelerometer
from processing.speech_detector import SpeechDetector
from processing.speech_protection import SpeechProtectionGate
from processing.mos_estimator import MOSEstimator


class IgardNetPipeline:
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        state: Optional[SystemState] = None,
        backend: Optional[ComputeBackend] = None,
        accelerometer: Optional[AccelerometerBase] = None,
        enable_background_tuner: bool = True,
    ):
        self.config = config if config is not None else load_config()
        self.state = state if state is not None else SystemState()
        
        # Hardware backend
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

        self.sample_rate = self.config["audio"]["sample_rate"]
        self.frame_size = self.config["audio"]["frame_size"]

        # Stage 0: Presence Gate
        pg_cfg = self.config.get("presence_gate", {})
        self.presence_gate = NoisePresenceGate(
            threshold_margin=pg_cfg.get("threshold_margin", 2.5),
            calibration_blocks=pg_cfg.get("calibration_blocks", 20),
            initial_floor=pg_cfg.get("initial_floor", 1.0e-5),
            hysteresis_db=pg_cfg.get("hysteresis_db", 2.0),
        )

        # Stage 1: Speech Detector (Continuous tracking + temporal hold)
        st_cfg = self.config.get("speech_tracking", {})
        self.speech_detector = SpeechDetector(
            sample_rate=self.sample_rate,
            frame_size=self.frame_size,
            attack_ms=st_cfg.get("attack_ms", 20.0),
            release_ms=st_cfg.get("release_ms", 120.0),
            hold_ms=st_cfg.get("hold_ms", 100.0),
            hysteresis=st_cfg.get("hysteresis", 0.08),
        )

        # Stage 2: Noise Classifier
        self.classifier = NoiseClassifier(sample_rate=self.sample_rate)

        # Stage 2b: Accelerometer
        self.accelerometer = accelerometer if accelerometer is not None else MockAccelerometer()
        self.shock_threshold = self.config.get("accelerometer", {}).get("shock_threshold", 0.5)
        self.agreement_boost = self.config.get("accelerometer", {}).get("agreement_boost", 0.25)

        # Stage 3: NLMS Filter with DTD
        nlms_cfg = self.config.get("nlms", {})
        self.nlms = NLMSFilter(
            num_taps=nlms_cfg.get("taps", 64),
            step_size=nlms_cfg.get("initial_mu", 0.20),
            eps=nlms_cfg.get("epsilon", 1.0e-6),
            leakage=nlms_cfg.get("leakage", 0.9995),
            max_mu=self.config.get("safety_limits", {}).get("max_filter_mu", 0.50),
            backend=self.backend,
        )

        # Stage 3b: GQPSO Tuner (Asynchronous Background Worker)
        gqpso_cfg = self.config.get("gqpso", {})
        self.tuner = QuantumInspiredTuner(
            num_particles=gqpso_cfg.get("num_particles", 8),
            iterations=gqpso_cfg.get("iterations", 5),
            search_min=gqpso_cfg.get("search_min", 0.01),
            search_max=self.config.get("safety_limits", {}).get("max_filter_mu", 0.50),
            mutation_prob=gqpso_cfg.get("mutation_prob", 0.1),
            history_len=gqpso_cfg.get("history_samples", 1024),
            backend=self.backend,
        )
        
        self.enable_background_tuner = enable_background_tuner
        if self.enable_background_tuner and gqpso_cfg.get("enabled", True):
            self.tuner_worker = BackgroundTunerWorker(
                tuner=self.tuner,
                interval_s=gqpso_cfg.get("interval_s", 1.0),
                num_taps=nlms_cfg.get("taps", 64),
            )
            self.tuner_worker.start()
        else:
            self.tuner_worker = None

        # Stage 4: Confidence Gate
        self.confidence_gate = ConfidenceGate(
            threshold=self.config.get("classifier", {}).get("confidence_threshold", 0.60)
        )

        # Stage 5: Wiener Cleanup
        wiener_cfg = self.config.get("wiener", {})
        self.cleanup = ResidualCleanup(
            sample_rate=self.sample_rate,
            gate_strength=wiener_cfg.get("gate_strength", 1.0),
            min_gain=wiener_cfg.get("min_gain", 0.35),
            backend=self.backend,
        )

        # Stage 6: Speech Protection Gate (Mandatory)
        sp_cfg = self.config.get("speech_protection", {})
        self.speech_protection = SpeechProtectionGate(
            sample_rate=self.sample_rate,
            frame_size=self.frame_size,
            probability_threshold=sp_cfg.get("probability_threshold", 0.65),
            minimum_voice_gain=sp_cfg.get("minimum_voice_gain", 0.55),
            uncertain_voice_gain=sp_cfg.get("uncertain_voice_gain", 0.70),
            max_suppression_db=sp_cfg.get("max_suppression_db", 12.0),
            enabled=sp_cfg.get("enabled", True),
        )

        # Stage 7: Voice Protection Dynamics
        vp_cfg = self.config.get("compressor", {})
        self.dynamics = VoiceProtectionDynamics(
            sample_rate=self.sample_rate,
            gain_floor_db=-self.config.get("safety_limits", {}).get("max_suppression_db", 12.0),
            threshold_db=vp_cfg.get("threshold_db", -18.0),
            ratio=vp_cfg.get("ratio", 2.0),
            attack_ms=vp_cfg.get("attack_ms", 5.0),
            release_ms=vp_cfg.get("release_ms", 100.0),
            makeup_gain_db=vp_cfg.get("makeup_gain_db", 0.0),
        )

        # Objective MOS Quality Estimator
        self.mos_estimator = MOSEstimator(sample_rate=self.sample_rate)

        # Rolling history buffers for snapshot extraction
        self._history_len = gqpso_cfg.get("history_samples", 1024)
        self._primary_history = np.zeros(self._history_len, dtype=np.float64)
        self._ref_history = np.zeros(self._history_len, dtype=np.float64)
        self._block_count = 0
        self._snr_ema = 0.0
        self._last_snapshot_time = time.time()
        self._last_out_sample: Optional[float] = None

    def process_block(
        self,
        primary_block: np.ndarray,
        reference_block: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Process a single audio frame with deterministic low latency.
        Returns cleaned audio and compact telemetry.
        """
        t_start = time.perf_counter()
        
        primary_block = np.asarray(primary_block, dtype=np.float64)
        reference_block = np.asarray(reference_block, dtype=np.float64)
        n_samples = len(primary_block)

        primary_rms = float(np.sqrt(np.mean(primary_block ** 2))) + 1.0e-12
        reference_rms = float(np.sqrt(np.mean(reference_block ** 2))) + 1.0e-12

        # Update running history for background optimizer
        if n_samples <= self._history_len:
            self._primary_history[:-n_samples] = self._primary_history[n_samples:]
            self._primary_history[-n_samples:] = primary_block
            self._ref_history[:-n_samples] = self._ref_history[n_samples:]
            self._ref_history[-n_samples:] = reference_block

        # Stage 0: Presence Gate
        noise_present, floor_est, block_energy = self.presence_gate.check(reference_block)

        # Stage 1: Speech Detection
        speech_prob, noise_prob, is_speech = self.speech_detector.detect(primary_block)
        
        # Read atomic mu from background tuner worker
        if self.tuner_worker is not None:
            self.nlms.set_step_size(self.tuner_worker.current_mu)
            self.state.update_step_size(self.tuner_worker.current_mu)

        # Autopilot decision logic:
        # Quiet environment: clean passthrough with minimal modification
        if not noise_present and self.state.autopilot and (speech_prob > 0.35 or block_energy < floor_est * 2.0):
            active_stage = 0
            stage_status = "Quiet — Environment baseline clean. Speech preserved."
            output_audio = primary_block.copy()
            noise_label = "clean_passthrough"
            confidence = 1.0
            shock_score = 0.0
            shock_confirmed = False
            used_cleanup = False
            snr_delta = 0.0
        else:
            # Stage 2: Noise Classifier
            noise_label, raw_confidence = self.classifier.classify(reference_block)

            # Stage 2b: Accelerometer Cross-Check
            shock_score = float(self.accelerometer.read_recent_shock_score())
            confidence = cross_check_confidence(
                acoustic_label=noise_label,
                acoustic_confidence=raw_confidence,
                shock_score=shock_score,
                agreement_boost=self.agreement_boost,
                shock_threshold=self.shock_threshold,
            )
            shock_confirmed = (noise_label == "impulsive") and (shock_score >= self.shock_threshold)

            # Autopilot Aggressiveness:
            # When speech is active, keep suppression moderate; when noise-only, allow full suppression
            if self.state.autopilot:
                if not noise_present:
                    aggressiveness = 0.0
                elif is_speech and confidence >= 0.60:
                    aggressiveness = 0.50
                elif is_speech and confidence < 0.60:
                    aggressiveness = 0.25 # Safe conservative fail-safe
                elif not is_speech and confidence >= 0.70:
                    aggressiveness = 1.00 # Strong stationary noise suppression
                else:
                    aggressiveness = 0.40
            else:
                aggressiveness = 0.75

            # Stage 3: NLMS Adaptive Filter (with speech protection & DTD)
            nlms_out, noise_estimate = self.nlms.process_block(
                primary_block,
                reference_block,
                speech_prob=speech_prob,
            )

            # Periodically submit snapshot to background tuner
            now = time.time()
            if self.tuner_worker is not None and (now - self._last_snapshot_time) >= self.tuner_worker.interval_s:
                self.tuner_worker.submit_snapshot(self._primary_history, self._ref_history)
                self._last_snapshot_time = now

            # Stage 4: Confidence Gate
            run_cleanup = self.confidence_gate.decide(confidence, speech_prob=speech_prob)

            # Stage 5: Speech-Aware 50% OLA Wiener Cleanup
            if run_cleanup and aggressiveness > 0.1:
                if noise_label == "steady" and speech_prob < 0.25:
                    self.cleanup.update_noise_profile(reference_block, speech_prob=speech_prob)

                cleaned = self.cleanup.clean(
                    nlms_out,
                    speech_prob=speech_prob,
                    aggressiveness=aggressiveness,
                )
                used_cleanup = True
                active_stage = 5
            else:
                cleaned = nlms_out
                used_cleanup = False
                active_stage = 4

            # Stage 6: SPEECH PROTECTION GATE (Mandatory)
            protected_speech, applied_gain = self.speech_protection.protect(
                raw_input_frame=primary_block,
                cleaned_frame=cleaned,
                speech_probability=speech_prob,
                noise_probability=noise_prob,
            )

            # Stage 7: Voice Protection Dynamics (Gentle Compressor & Limiter)
            output_audio = self.dynamics.process(protected_speech, primary_block)

            # Build stage status text
            if shock_confirmed:
                stage_status = f"⚡ SHOCK CONFIRMED ({confidence*100:.0f}%) — Cross-modal impact verified."
            elif is_speech:
                stage_status = f"🗣 SPEECH ACTIVE ({speech_prob*100:.0f}%) — Voice protected (gain: {applied_gain:.2f})"
            elif noise_label == "steady":
                stage_status = f"🔊 STEADY NOISE ({confidence*100:.0f}%) — Adaptive filter active."
            else:
                stage_status = f"Filtering ({noise_label}, {confidence*100:.0f}%)"

            # Compute estimated SNR improvement
            clean_rms = float(np.sqrt(np.mean(output_audio ** 2))) + 1.0e-12
            raw_snr = 20.0 * np.log10(primary_rms / (reference_rms + 1.0e-6))
            enh_snr = 20.0 * np.log10(clean_rms / (reference_rms + 1.0e-6))
            snr_delta = float(np.clip(enh_snr - raw_snr, -30.0, 30.0))

        # Boundary cross-fade ramp on enhanced output to eliminate any sample discontinuity (ticking/clicking)
        if self._last_out_sample is not None and len(output_audio) >= 16:
            step = float(self._last_out_sample - output_audio[0])
            if abs(step) > 0.05:
                ramp = np.linspace(1.0, 0.0, 16)
                output_audio[:16] += step * ramp
        if len(output_audio) > 0:
            self._last_out_sample = float(output_audio[-1])

        # Before / After Multiplexer
        if self.state.before_after:
            final_audio = output_audio
        else:
            final_audio = primary_block.copy()

        # Update telemetry metrics
        t_proc_ms = (time.perf_counter() - t_start) * 1000.0
        self.state.record_latency(process_ms=t_proc_ms)
        self._snr_ema = 0.90 * self._snr_ema + 0.10 * snr_delta
        self.state.snr_delta = float(self._snr_ema)
        self.state.noise_confidence = float(confidence)
        self.state.shock_confidence = float(shock_score)
        self.state.current_stage = active_stage
        self.state.speech_prob = float(speech_prob)
        self.state.noise_prob = float(noise_prob)

        # Objective frame MOS calculation
        estimated_mos = self.mos_estimator.estimate_frame_mos(
            primary_block=primary_block,
            enhanced_block=output_audio,
            speech_prob=float(speech_prob),
            snr_delta=float(self._snr_ema),
        )
        self.state.estimated_mos = float(estimated_mos)

        self._block_count += 1
        self.state.frames_processed = self._block_count

        return {
            "audio": final_audio,
            "raw_audio": primary_block,
            "processed_audio": output_audio,
            "enhanced_audio": output_audio,
            "speech_prob": float(speech_prob),
            "noise_prob": float(noise_prob),
            "estimated_mos": float(estimated_mos),
            "noise_label": noise_label,
            "confidence": float(confidence),
            "shock_score": float(shock_score),
            "shock_confirmed": shock_confirmed,
            "used_cleanup_stage": used_cleanup,
            "active_stage": active_stage,
            "stage_status": stage_status,
            "primary_rms": primary_rms,
            "processed_rms": float(np.sqrt(np.mean(output_audio ** 2))),
            "snr_delta": float(self._snr_ema),
            "processing_time_ms": t_proc_ms,
            "block_count": self._block_count,
        }

    def process_stream(
        self,
        primary: np.ndarray,
        reference: np.ndarray,
        block_size: Optional[int] = None,
    ) -> Tuple[np.ndarray, list]:
        """Batch process an audio stream sequentially in contiguous blocks."""
        bs = block_size if block_size is not None else self.frame_size
        n = len(primary)
        assert len(reference) == n, "Primary and Reference streams must have identical length"

        out_audio = np.empty(n, dtype=np.float64)
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

    def close(self):
        """Stops background tuner worker and releases resources."""
        if hasattr(self, 'tuner_worker') and self.tuner_worker is not None:
            self.tuner_worker.stop()
