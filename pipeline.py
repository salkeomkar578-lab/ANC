"""
The Reconstructed IGARD-Net Speech-First Signal Chain:

  primary mic, reference mic
        |
  [0] NoisePresenceGate         -> engage or clean passthrough? (with hysteresis)
        |
  [1] SpeechDetector            -> continuous speech_prob & noise_prob (with temporal hold)
        |
  [2] NoiseClassifier           -> label, confidence
  [2b] AccelerometerCrossCheck  -> physical shock confirmation
        |
  [3] NLMSFilter                -> nlms_output (with DTD & speech protection)
  [3b] QuantumInspiredTuner     -> multi-objective speech-protective autotuning
        |
  [4] ConfidenceGate            -> speech-first fail-safe decision
        |
  [5] ResidualCleanup           -> 50% OLA Wiener with formant protection (300-3400 Hz)
        |
  [6] SpeechProtectionGate      -> enforce speech gain floor, zero hard mutes
        |
  [7] Dynamics Limiter          -> gentle compressor & peak limiter
        |
     final clean audio
"""

import numpy as np

from nlms_filter import NLMSFilter
from quantum_tuner import QuantumInspiredTuner
from noise_classifier import NoiseClassifier
from confidence_gate import ConfidenceGate
from cleanup_stage import ResidualCleanup
from accel_sensor import MockAccelerometer, cross_check_confidence
from noise_presence import NoisePresenceGate
from speech_detector import SpeechDetector
from speech_protection import SpeechProtectionGate
from spectrogram_helper import compute_fft_magnitudes, compute_rms_level, compute_snr_estimate


class IgardNetPipeline:
    def __init__(
        self,
        sample_rate: int = 16000,
        num_taps: int = 64,
        retune_every_n_blocks: int = 10,
        confidence_threshold: float = 0.60,
        accelerometer=None,
        autopilot: bool = True,
        speech_protection_threshold: float = 0.65,
        minimum_voice_gain: float = 0.55,
        uncertain_voice_gain: float = 0.70,
        max_suppression_db: float = 12.0,
    ):
        self.sample_rate = sample_rate
        self.autopilot = autopilot
        
        # Pipeline Stages
        self.presence_gate = NoisePresenceGate()
        self.speech_detector = SpeechDetector(sample_rate=sample_rate)
        self.classifier = NoiseClassifier(sample_rate=sample_rate)
        self.gate = ConfidenceGate(threshold=confidence_threshold)
        self.nlms = NLMSFilter(num_taps=num_taps, step_size=0.2, max_mu=0.5)
        self.tuner = QuantumInspiredTuner(search_max=0.50)
        self.cleanup = ResidualCleanup(sample_rate=sample_rate, gate_strength=1.0, min_gain=0.35)
        self.speech_protection = SpeechProtectionGate(
            sample_rate=sample_rate,
            probability_threshold=speech_protection_threshold,
            minimum_voice_gain=minimum_voice_gain,
            uncertain_voice_gain=uncertain_voice_gain,
            max_suppression_db=max_suppression_db,
            enabled=True,
        )

        self.accelerometer = accelerometer if accelerometer is not None else MockAccelerometer()
        self.num_taps = num_taps
        self.retune_every_n_blocks = retune_every_n_blocks
        self._block_count = 0

        # Rolling history for background tuner
        self._primary_history = np.zeros(sample_rate // 2)
        self._reference_history = np.zeros(sample_rate // 2)

        self._snr_ema: float = 0.0
        self._last_out_sample = None

    def process_block(self, primary_block, reference_block):
        primary_block = np.asarray(primary_block, dtype=np.float64)
        reference_block = np.asarray(reference_block, dtype=np.float64)

        primary_level = compute_rms_level(primary_block)
        reference_level = compute_rms_level(reference_block)
        primary_fft = compute_fft_magnitudes(primary_block, self.sample_rate)

        # Stage 0: Noise Presence Gate
        noise_present, floor_estimate, block_energy = self.presence_gate.check(reference_block)

        # Stage 1: Speech Detection (continuous probability & temporal hold)
        speech_prob, noise_prob, is_speech = self.speech_detector.detect(primary_block)

        # Autopilot / Quiet Bypass Decision
        # If quiet and no real noise present, pass speech through cleanly
        if not noise_present and (speech_prob > 0.40 or block_energy < floor_estimate * 2.0):
            output_fft = compute_fft_magnitudes(primary_block, self.sample_rate)
            return {
                "audio": primary_block,
                "noise_label": "clean_passthrough",
                "confidence": 1.0,
                "speech_prob": speech_prob,
                "noise_prob": noise_prob,
                "shock_score": 0.0,
                "used_cleanup_stage": False,
                "nlms_step_size": self.nlms.step_size,
                "noise_present": False,
                "noise_floor_estimate": floor_estimate,
                "active_stage": 0,
                "stage_status": "Quiet / Voice Preserved — bypass active, audio untouched",
                "primary_level": primary_level,
                "reference_level": reference_level,
                "primary_fft": primary_fft["magnitudes"],
                "output_fft": output_fft["magnitudes"],
                "primary_samples": primary_block.tolist()[-256:],
                "output_samples": primary_block.tolist()[-256:],
                "estimated_snr": float(self._snr_ema),
                "block_count": self._block_count,
            }

        # Stage 2: Noise Classification
        label, raw_confidence = self.classifier.classify(reference_block)

        # Stage 2b: Cross-modal accelerometer confirmation
        shock_score = self.accelerometer.read_recent_shock_score()
        confidence = cross_check_confidence(label, raw_confidence, shock_score)

        # Autopilot suppression aggressiveness tuning
        if self.autopilot:
            if not noise_present:
                aggressiveness = 0.0
            elif is_speech and confidence >= 0.60:
                # Strong noise + speech -> moderate suppression with voice protection
                aggressiveness = 0.50
            elif is_speech and confidence < 0.60:
                # Uncertain + speech -> safe conservative mode
                aggressiveness = 0.25
            elif not is_speech and confidence >= 0.70:
                # Confident noise without speech -> allow stronger suppression
                aggressiveness = 1.00
            else:
                aggressiveness = 0.40
        else:
            aggressiveness = 0.80

        # Stage 3: NLMS Filter with DTD and Speech Leakage Protection
        nlms_out, noise_est = self.nlms.process_block(
            primary_block,
            reference_block,
            speech_prob=speech_prob,
        )

        # Stage 3b: Multi-Objective GQPSO Auto-Tuning
        self._primary_history = np.roll(self._primary_history, -len(primary_block))
        self._primary_history[-len(primary_block):] = primary_block
        self._reference_history = np.roll(self._reference_history, -len(reference_block))
        self._reference_history[-len(reference_block):] = reference_block

        self._block_count += 1
        retuned = False
        if self._block_count % self.retune_every_n_blocks == 0:
            new_step = self.tuner.retune(
                self._primary_history,
                self._reference_history,
                self.num_taps,
                speech_prob=speech_prob,
            )
            self.nlms.set_step_size(new_step)
            retuned = True

        # Stage 4: Confidence Gate (speech-first fail-safe)
        run_cleanup = self.gate.decide(confidence, speech_prob=speech_prob)

        # Stage 5: Speech-Aware 50% OLA Wiener Cleanup
        if run_cleanup and aggressiveness > 0.1:
            if label == "steady" and speech_prob < 0.25:
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
        # Guarantees no hard mutes and enforces speech preservation floor
        protected_out, applied_gain = self.speech_protection.protect(
            raw_input_frame=primary_block,
            cleaned_frame=cleaned,
            speech_probability=speech_prob,
            noise_probability=noise_prob,
        )

        final_out = protected_out
        if self._last_out_sample is not None and len(final_out) >= 16:
            step = float(self._last_out_sample - final_out[0])
            if abs(step) > 0.05:
                ramp = np.linspace(1.0, 0.0, 16)
                final_out[:16] += step * ramp
        if len(final_out) > 0:
            self._last_out_sample = float(final_out[-1])

        output_fft = compute_fft_magnitudes(final_out, self.sample_rate)
        block_snr = compute_snr_estimate(final_out, primary_block)
        self._snr_ema = 0.85 * self._snr_ema + 0.15 * block_snr

        stage_status = f"Active: Speech Prob {speech_prob*100:.0f}%, Gain {applied_gain:.2f}"
        if is_speech:
            stage_status += " [SPEECH PROTECTED]"

        return {
            "audio": final_out,
            "noise_label": label,
            "confidence": confidence,
            "speech_prob": speech_prob,
            "noise_prob": noise_prob,
            "shock_score": shock_score,
            "used_cleanup_stage": used_cleanup,
            "nlms_step_size": self.nlms.step_size,
            "noise_present": noise_present,
            "noise_floor_estimate": floor_estimate,
            "active_stage": active_stage,
            "stage_status": stage_status,
            "primary_level": primary_level,
            "reference_level": reference_level,
            "primary_fft": primary_fft["magnitudes"],
            "output_fft": output_fft["magnitudes"],
            "primary_samples": primary_block.tolist()[-256:],
            "output_samples": final_out.tolist()[-256:],
            "estimated_snr": float(self._snr_ema),
            "block_count": self._block_count,
            "retuned_this_block": retuned,
        }

    def process_stream(self, primary, reference, block_size=256):
        """Processes continuous stream block by block."""
        assert len(primary) == len(reference)
        n = len(primary)
        output = np.zeros(n, dtype=np.float64)
        telemetry = []

        for start in range(0, n, block_size):
            end = min(start + block_size, n)
            p_block = primary[start:end]
            r_block = reference[start:end]
            if len(p_block) < block_size:
                pad = block_size - len(p_block)
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
        """Releases pipeline resources."""
        pass
