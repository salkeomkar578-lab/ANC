"""
Core IGARD-Net Signal Chain Pipeline Coordinator.
Reconstructed for low-latency real-time streaming and high-throughput file processing.

Architecture:
  Primary & Reference Mic
       │
  [Stage 0]  NoisePresenceGate         -> Quiet bypass or engage
       │
  [Stage 1]  NoiseClassifier          -> Acoustic label & confidence
  [Stage 1b] AccelerometerFusion       -> Physical shock cross-modal verification
       │
  [Stage 2]  NLMSFilter (64-tap)       -> Adaptive acoustic cancellation
  [Stage 3]  Asynchronous GQPSO Tuner  -> Periodic background optimization of mu
       │
  [Stage 4]  ConfidenceGate            -> Cleanup vs Fail-safe bypass arbiter
       │
  [Stage 5]  Wiener Soft Masking (OLA) -> Residual transient & drone cleanup
       │
  [Stage 6]  Voice Protection Dynamics -> Speech floor, compressor, peak limiter
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

        # Stage 1: Noise Classifier
        self.classifier = NoiseClassifier(sample_rate=self.sample_rate)

        # Stage 1b: Accelerometer
        self.accelerometer = accelerometer if accelerometer is not None else MockAccelerometer()
        self.shock_threshold = self.config.get("accelerometer", {}).get("shock_threshold", 0.5)
        self.agreement_boost = self.config.get("accelerometer", {}).get("agreement_boost", 0.25)

        # Stage 2: NLMS Filter
        nlms_cfg = self.config.get("nlms", {})
        self.nlms = NLMSFilter(
            num_taps=nlms_cfg.get("taps", 64),
            initial_mu=nlms_cfg.get("initial_mu", 0.25),
            eps=nlms_cfg.get("epsilon", 1.0e-6),
            leakage=nlms_cfg.get("leakage", 0.9999),
            adaptation_enabled=nlms_cfg.get("adaptation_enabled", True),
            backend=self.backend,
        )

        # Stage 3: GQPSO Tuner (Asynchronous Background Worker)
        gqpso_cfg = self.config.get("gqpso", {})
        self.tuner = QuantumInspiredTuner(
            num_particles=gqpso_cfg.get("num_particles", 8),
            iterations=gqpso_cfg.get("iterations", 5),
            search_min=gqpso_cfg.get("search_min", 0.01),
            search_max=gqpso_cfg.get("search_max", 1.5),
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
            gate_strength=wiener_cfg.get("gate_strength", 1.4),
            min_gain=wiener_cfg.get("min_gain", 0.08),
            backend=self.backend,
        )

        # Stage 6: Voice Protection Dynamics
        vp_cfg = self.config.get("voice_protection", {})
        self.dynamics = VoiceProtectionDynamics(
            sample_rate=self.sample_rate,
            gain_floor_db=vp_cfg.get("speech_gain_floor_db", -18.0),
            threshold_db=vp_cfg.get("compressor_threshold_db", -6.0),
            ratio=vp_cfg.get("compressor_ratio", 3.0),
            attack_ms=vp_cfg.get("attack_ms", 5.0),
            release_ms=vp_cfg.get("release_ms", 40.0),
            makeup_gain_db=vp_cfg.get("makeup_gain_db", 1.5),
        )

        # Rolling history buffers for snapshot extraction
        self._history_len = gqpso_cfg.get("history_samples", 1024)
        self._primary_history = np.zeros(self._history_len, dtype=np.float64)
        self._ref_history = np.zeros(self._history_len, dtype=np.float64)
        self._block_count = 0
        self._snr_ema = 0.0
        self._last_snapshot_time = time.time()

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

        # Input levels
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
        
        # Read atomic mu from background tuner worker
        if self.tuner_worker is not None:
            self.nlms.set_step_size(self.tuner_worker.current_mu)
            self.state.update_step_size(self.tuner_worker.current_mu)

        # -------------------------------------------------------------
        # AUTOPILOT DECISION LOGIC & EXECUTION PATH
        # -------------------------------------------------------------
        if not noise_present and self.state.autopilot:
            # Quiet environment: clean passthrough with minimal distortion
            active_stage = 0
            stage_status = "Quiet — Environment baseline clean. Passthrough active."
            output_audio = primary_block.copy()
            noise_label = "clean_passthrough"
            confidence = 1.0
            shock_score = 0.0
            shock_confirmed = False
            used_cleanup = False
            snr_delta = 0.0
        else:
            # Stage 1: Noise Classifier
            noise_label, raw_confidence = self.classifier.classify(reference_block)

            # Stage 1b: Accelerometer Cross-Check
            shock_score = float(self.accelerometer.read_recent_shock_score())
            confidence = cross_check_confidence(
                acoustic_label=noise_label,
                acoustic_confidence=raw_confidence,
                shock_score=shock_score,
                agreement_boost=self.agreement_boost,
                shock_threshold=self.shock_threshold,
            )
            shock_confirmed = (noise_label == "impulsive") and (shock_score >= self.shock_threshold)

            # Stage 2: NLMS Adaptive Filter
            nlms_out, noise_estimate = self.nlms.process_block(primary_block, reference_block)

            # Periodically submit snapshot to background tuner
            now = time.time()
            if self.tuner_worker is not None and (now - self._last_snapshot_time) >= self.tuner_worker.interval_s:
                self.tuner_worker.submit_snapshot(self._primary_history, self._ref_history)
                self._last_snapshot_time = now

            # Stage 4: Confidence Gate
            run_cleanup = self.confidence_gate.decide(confidence, autopilot=self.state.autopilot)

            # Stage 5: Wiener Cleanup or Fail-safe Bypass
            if run_cleanup:
                # In steady noise: update background spectral profile
                if noise_label == "steady":
                    self.cleanup.update_noise_profile(reference_block)
                    aggressiveness = 1.2
                elif noise_label == "impulsive" and shock_confirmed:
                    aggressiveness = 1.4
                else:
                    aggressiveness = 0.9

                cleaned = self.cleanup.clean(nlms_out, aggressiveness=aggressiveness)
                used_cleanup = True
                active_stage = 5
            else:
                # Conservative fail-safe bypass (pass NLMS output without spectral slicing)
                cleaned = nlms_out
                used_cleanup = False
                active_stage = 4

            # Stage 6: Voice Protection Dynamics (Compressor, Limiter, Gain Floor)
            output_audio = self.dynamics.process(cleaned, primary_block)

            # Build tactical stage status text
            if shock_confirmed:
                stage_status = f"⚡ SHOCK CONFIRMED ({confidence*100:.0f}%) — Cross-modal impact verified. Full suppression engaged."
            elif noise_label == "impulsive":
                stage_status = f"⚠ IMPULSIVE SOUND ({confidence*100:.0f}%) — No mechanical shock. Conservative fail-safe active."
            elif noise_label == "steady":
                stage_status = f"🔊 STEADY DRONE ({confidence*100:.0f}%) — Adaptive NLMS + OLA Wiener cleanup active."
            else:
                stage_status = f"NOISE PRESENT ({confidence*100:.0f}%) — Adaptive processing engaged."

            # Calculate SNR improvement estimate
            noise_removed = primary_block - output_audio
            sig_pow = float(np.mean(output_audio ** 2)) + 1.0e-12
            noi_pow = float(np.mean(noise_removed ** 2)) + 1.0e-12
            snr_delta = float(np.clip(10.0 * np.log10(sig_pow / noi_pow), -20.0, 25.0))

        # Update running SNR EMA
        self._snr_ema = 0.88 * self._snr_ema + 0.12 * snr_delta
        self._block_count += 1

        # Real processing latency measurement
        t_end = time.perf_counter()
        proc_latency_ms = (t_end - t_start) * 1000.0

        # Before / After selection
        final_audio = output_audio if self.state.before_after else primary_block

        output_rms = float(np.sqrt(np.mean(final_audio ** 2))) + 1.0e-12

        # Update state container
        self.state.noise_label = noise_label
        self.state.confidence = confidence
        self.state.shock_score = shock_score
        self.state.shock_confirmed = shock_confirmed
        self.state.noise_present = noise_present
        self.state.noise_floor = floor_est
        self.state.active_stage = active_stage
        self.state.stage_status = stage_status
        self.state.primary_rms = primary_rms
        self.state.reference_rms = reference_rms
        self.state.output_rms = output_rms
        self.state.snr_delta = snr_delta
        self.state.snr_ema = self._snr_ema

        return {
            "audio": final_audio,
            "raw_audio": primary_block,
            "enhanced_audio": output_audio,
            "noise_label": noise_label,
            "confidence": confidence,
            "shock_score": shock_score,
            "shock_confirmed": shock_confirmed,
            "used_cleanup_stage": used_cleanup,
            "nlms_step_size": self.nlms.step_size,
            "noise_present": noise_present,
            "noise_floor": floor_est,
            "active_stage": active_stage,
            "stage_status": stage_status,
            "primary_rms": primary_rms,
            "reference_rms": reference_rms,
            "output_rms": output_rms,
            "snr_delta": snr_delta,
            "estimated_snr": self._snr_ema,
            "proc_latency_ms": proc_latency_ms,
            "block_count": self._block_count,
        }

    def process_stream(self, primary: np.ndarray, reference: np.ndarray, block_size: Optional[int] = None) -> Tuple[np.ndarray, list]:
        """Convenience method for offline streaming over numpy arrays."""
        assert len(primary) == len(reference), "Primary and reference signals must have identical length."
        bs = block_size if block_size is not None else self.frame_size
        n = len(primary)
        output = np.zeros(n, dtype=np.float64)
        telemetry = []

        for start in range(0, n, bs):
            end = min(start + bs, n)
            p_block = primary[start:end]
            r_block = reference[start:end]
            if len(p_block) < bs:
                pad = bs - len(p_block)
                p_pad = np.pad(p_block, (0, pad))
                r_pad = np.pad(r_block, (0, pad))
                res = self.process_block(p_pad, r_pad)
                output[start:end] = res["audio"][: end - start]
            else:
                res = self.process_block(p_block, r_block)
                output[start:end] = res["audio"]
            telemetry.append({k: v for k, v in res.items() if k != "audio"})

        return output, telemetry

    def close(self):
        """Clean shutdown of background threads."""
        if self.tuner_worker is not None:
            self.tuner_worker.stop()
