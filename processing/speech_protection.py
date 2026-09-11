"""
IGARD-Net V3 — Speech Protection Controller.
Implements speech-first continuous protection to ensure:
1. Human voice is never muted or chopped (no binary VAD gating).
2. RNNoise performs primary environmental noise suppression.
3. Output energy collapse guard prevents syllable dropout.
4. Adaptive suppression strength with Autopilot and manual override.
5. Continuous temporal smoothing (attack/release, EMA, hysteresis, hangover).
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional


class SpeechProtectionController:
    """
    Supervisor governing downstream noise suppression based on continuous Silero VAD confidence.
    Guarantees natural, continuous human speech preservation over aggressive noise cutting.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        frame_size: int = 480,
        high_speech_thresh: float = 0.75,
        low_speech_thresh: float = 0.45,
        attack_ms: float = 12.0,
        release_ms: float = 120.0,
        hold_ms: float = 150.0,
        default_suppression: float = 0.75,
        probability_threshold: Optional[float] = None,
        minimum_voice_gain: float = 0.55,
        uncertain_voice_gain: float = 0.70,
        max_suppression_db: float = 12.0,
        noise_suppression_floor_db: float = 40.0,
        enabled: bool = True,
        **kwargs,
    ):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.dt = frame_size / sample_rate
        self.dt_ms = self.dt * 1000.0

        self.high_thresh = probability_threshold if probability_threshold is not None else high_speech_thresh
        self.low_thresh = low_speech_thresh
        self.minimum_voice_gain = minimum_voice_gain
        self.uncertain_voice_gain = uncertain_voice_gain
        self.enabled = enabled

        # Temporal smoothing coefficients
        self.alpha_attack = float(np.exp(-self.dt_ms / max(1.0, attack_ms)))
        self.alpha_release = float(np.exp(-self.dt_ms / max(1.0, release_ms)))
        self.hold_frames = int(np.ceil((hold_ms / 1000.0) / self.dt))

        # Persistent controller states
        self._smoothed_speech_prob: float = 0.0
        self._hold_counter: int = 0
        self._suppression_strength: float = default_suppression
        self._target_suppression: float = default_suppression
        self._noise_floor_ema: float = 0.01
        self._last_out_sample: Optional[float] = None
        self._last_gain: float = 1.0

        # Phase-matched delay buffer to align raw voice with RNNoise 20ms filterbank delay
        self._delay_samples = int(self.sample_rate * 0.020)
        self._raw_delay_buffer = np.zeros(self._delay_samples, dtype=np.float32)

        # Autopilot configuration
        self.autopilot: bool = True
        self.manual_suppression: float = default_suppression

    def reset(self) -> None:
        """Reset all internal state variables."""
        self._smoothed_speech_prob = 0.0
        self._hold_counter = 0
        self._suppression_strength = 0.75
        self._target_suppression = 0.75
        self._noise_floor_ema = 0.01
        self._last_out_sample = None
        self._last_gain = 1.0

    def update_speech_probability(self, raw_prob: float) -> float:
        """
        Smooth continuous speech probability with fast attack, minimum hold time
        (hangover), and gentle release.
        """
        raw_prob = float(np.clip(raw_prob, 0.0, 1.0))

        if raw_prob >= self._smoothed_speech_prob:
            # Voice onset / rising confidence: fast attack
            # If strong speech appears after silence, immediately jump to speech threshold
            if raw_prob >= self.high_thresh and self._smoothed_speech_prob < self.high_thresh:
                self._smoothed_speech_prob = max(self._smoothed_speech_prob, raw_prob * 0.82)
            else:
                self._smoothed_speech_prob = (
                    (1.0 - self.alpha_attack) * raw_prob
                    + self.alpha_attack * self._smoothed_speech_prob
                )
            if raw_prob >= self.low_thresh:
                self._hold_counter = self.hold_frames
        else:
            # Falling confidence: evaluate speech hold time (hangover)
            if self._hold_counter > 0:
                self._hold_counter -= 1
                # During hold time, decay very slowly to bridge short consonant/breath pauses
                decay = 0.985
                self._smoothed_speech_prob = max(raw_prob, self._smoothed_speech_prob * decay)
            else:
                # Release decay
                self._smoothed_speech_prob = (
                    (1.0 - self.alpha_release) * raw_prob
                    + self.alpha_release * self._smoothed_speech_prob
                )

        self._smoothed_speech_prob = float(np.clip(self._smoothed_speech_prob, 0.0, 1.0))
        return self._smoothed_speech_prob

    def update_suppression_strength(
        self,
        speech_prob: float,
        noise_rms: float,
        rnnoise_vad: float,
    ) -> float:
        """
        Adapt suppression strength based on environmental noise and speech status.
        Autopilot automatically picks the optimal trade-off:
          - Quiet: light processing (preserves acoustic purity)
          - Moderate: standard suppression
          - Strong engine/wind noise: stronger suppression
          - Speech + high noise: maximum safe suppression with maximum speech protection
        """
        # Track background noise floor during low-speech intervals
        if speech_prob < self.low_thresh:
            self._noise_floor_ema = 0.95 * self._noise_floor_ema + 0.05 * max(1e-4, noise_rms)

        if self.autopilot:
            # Adaptive selection
            if self._noise_floor_ema < 0.015:
                # Quiet room / low noise: gentle suppression
                target = 0.40
            elif self._noise_floor_ema < 0.06:
                # Moderate ambient noise: normal suppression
                target = 0.70
            elif self._noise_floor_ema < 0.15:
                # Strong engine / vehicle noise: high suppression
                target = 0.88
            else:
                # Extreme machinery / wind / broadband noise: maximum safe suppression
                target = 0.96

            # If confident speech is present, slightly temper suppression to protect vocal warmth
            if speech_prob > self.high_thresh:
                target = min(target, 0.85)
        else:
            target = float(np.clip(self.manual_suppression, 0.0, 1.0))

        self._target_suppression = target
        # Smooth suppression strength transitions
        self._suppression_strength = 0.90 * self._suppression_strength + 0.10 * target
        return self._suppression_strength

    def protect(
        self,
        raw_frame: np.ndarray,
        suppressed_frame: np.ndarray,
        speech_probability: float,
        rnnoise_vad: float = 0.0,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply speech-first protection rules to the frame.
        
        Args:
            raw_frame: 1D array of input samples
            suppressed_frame: 1D array of RNNoise-denoised samples
            speech_probability: Raw continuous speech probability from Silero VAD
            rnnoise_vad: Speech confidence reported by RNNoise C engine
            
        Returns:
            Tuple of (protected_output_frame, telemetry_dict)
        """
        raw = np.asarray(raw_frame, dtype=np.float32)
        supp = np.asarray(suppressed_frame, dtype=np.float32)
        n = len(raw)

        # Push current raw into delay buffer and retrieve phase-aligned raw frame
        if len(self._raw_delay_buffer) != self._delay_samples:
            self._raw_delay_buffer = np.zeros(self._delay_samples, dtype=np.float32)
        combined_buf = np.concatenate([self._raw_delay_buffer, raw])
        delayed_raw = combined_buf[:n]
        self._raw_delay_buffer = combined_buf[n:]

        raw_rms = float(np.sqrt(np.mean(delayed_raw ** 2))) + 1e-12
        supp_rms = float(np.sqrt(np.mean(supp ** 2))) + 1e-12

        # 1. Update smoothed continuous speech probability
        smoothed_prob = self.update_speech_probability(speech_probability)

        # 2. Update adaptive suppression strength
        supp_strength = self.update_suppression_strength(
            speech_prob=smoothed_prob,
            noise_rms=raw_rms,
            rnnoise_vad=rnnoise_vad,
        )

        # 3. Categorize into confidence tiers
        if smoothed_prob >= self.high_thresh:
            # Tier 1: Confirmed human speech (>0.75)
            # Maximum speech protection: retain full vocal envelope and blend phase-aligned
            # raw harmonic content (12-18%) to ensure vocal warmth and natural harmonics.
            status_text = "SPEECH ACTIVE"
            raw_mix = 0.15 * smoothed_prob
            processed = (1.0 - raw_mix) * supp + raw_mix * delayed_raw
            protection_active = True
            residual_gain = 1.0

        elif smoothed_prob >= self.low_thresh:
            # Tier 2: Uncertain / transitional speech (0.45 - 0.75)
            # Moderate suppression, strong speech preservation.
            status_text = "SPEECH TRANSITION"
            blend_ratio = (smoothed_prob - self.low_thresh) / (self.high_thresh - self.low_thresh)
            raw_mix = 0.08 * blend_ratio
            processed = (1.0 - raw_mix) * supp + raw_mix * delayed_raw
            protection_active = True
            residual_gain = 1.0

        else:
            # Tier 3: Non-speech / background noise (<0.45)
            # Allow stronger attenuation of residual noise without hard gating.
            status_text = "NOISE SUPPRESSION"
            protection_active = False
            # Scale suppression depth smoothly with suppression_strength
            suppression_factor = supp_strength
            processed = supp
            # Gentle background residual attenuation based on distance below low_thresh
            prob_ratio = smoothed_prob / max(1e-4, self.low_thresh)
            residual_gain = float(np.clip(
                (1.0 - suppression_factor * 0.75) + prob_ratio * (suppression_factor * 0.75),
                0.05,
                1.0,
            ))
            processed = processed * residual_gain

        # 4. HARD SAFETY CONSTRAINT: Output Energy Collapse Guard
        # If speech is detected, the frame MUST NEVER collapse or become excessively quiet.
        energy_guard_triggered = False
        if smoothed_prob >= self.high_thresh and raw_rms > 1e-4:
            curr_rms = float(np.sqrt(np.mean(processed ** 2))) + 1e-12
            target_floor = self.minimum_voice_gain  # e.g. 0.45 - 0.55
            if curr_rms < (raw_rms * target_floor):
                energy_guard_triggered = True
                status_text = "SPEECH RECOVERY (COLLAPSE GUARD)"
                # Calculate rescue blend to ensure voice volume does not collapse
                deficit = (raw_rms * target_floor) - curr_rms
                mix_rescue = float(np.clip(deficit / max(1e-6, raw_rms - curr_rms), 0.20, 0.90))
                processed = (1.0 - mix_rescue) * processed + mix_rescue * delayed_raw
                self._suppression_strength = max(0.30, self._suppression_strength * 0.8)

        elif smoothed_prob >= self.low_thresh and raw_rms > 1e-4:
            curr_rms = float(np.sqrt(np.mean(processed ** 2))) + 1e-12
            target_floor = 0.30
            if curr_rms < (raw_rms * target_floor):
                deficit = (raw_rms * target_floor) - curr_rms
                mix_rescue = float(np.clip(deficit / max(1e-6, raw_rms - curr_rms), 0.15, 0.75))
                processed = (1.0 - mix_rescue) * processed + mix_rescue * delayed_raw

        # 5. Continuous Sample-Boundary Cross-Fade (Anti-Ticking / Anti-Clicking)
        if self._last_out_sample is not None and n >= 16:
            step = float(self._last_out_sample - processed[0])
            if abs(step) > 0.02:
                # Apply 16-sample linear fade ramp to smooth waveform discontinuity
                ramp = np.linspace(1.0, 0.0, 16, dtype=np.float32)
                processed[:16] += step * ramp

        if n > 0:
            self._last_out_sample = float(processed[-1])

        out_rms = float(np.sqrt(np.mean(processed ** 2))) + 1e-12

        telemetry = {
            "speech_probability": smoothed_prob,
            "raw_speech_prob": speech_probability,
            "suppression_strength": supp_strength,
            "autopilot": self.autopilot,
            "status": status_text,
            "raw_rms": raw_rms,
            "enhanced_rms": out_rms,
            "protection_active": protection_active,
            "energy_guard_triggered": energy_guard_triggered,
            "residual_gain": residual_gain,
        }

        return processed, telemetry


from processing.gain_smoother import GainSmoother


class SpeechProtectionGate:
    """Legacy SpeechProtectionGate maintained for backwards-compatibility."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_size: int = 256,
        probability_threshold: float = 0.65,
        minimum_voice_gain: float = 0.55,
        uncertain_voice_gain: float = 0.70,
        max_suppression_db: float = 12.0,
        noise_suppression_floor_db: float = 40.0,
        enabled: bool = True,
        **kwargs,
    ):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.prob_thresh = probability_threshold
        self.min_voice_gain = minimum_voice_gain
        self.uncertain_voice_gain = uncertain_voice_gain
        self.max_suppression_linear = 10.0 ** (-max_suppression_db / 20.0)
        self.noise_floor_linear = 10.0 ** (-noise_suppression_floor_db / 20.0)
        self.enabled = enabled

        self.smoother = GainSmoother(
            sample_rate=sample_rate,
            frame_size=frame_size,
            attack_ms=10.0,
            release_ms=60.0,
            max_delta_per_frame=0.08,
            min_gain=self.noise_floor_linear,
        )

    def reset(self):
        self.smoother.reset(1.0)

    def protect(
        self,
        raw_input_frame: np.ndarray,
        cleaned_frame: np.ndarray,
        speech_probability: float,
        noise_probability: float = 0.0,
    ) -> Tuple[np.ndarray, float]:
        if not self.enabled:
            return cleaned_frame, 1.0

        raw = np.asarray(raw_input_frame, dtype=np.float64)
        clean = np.asarray(cleaned_frame, dtype=np.float64)
        n = len(raw)
        if n == 0:
            return clean, 1.0

        raw_rms = float(np.sqrt(np.mean(raw ** 2))) + 1e-10
        clean_rms = float(np.sqrt(np.mean(clean ** 2))) + 1e-10
        current_attenuation = clean_rms / raw_rms

        if speech_probability >= self.prob_thresh:
            target_floor = self.min_voice_gain
        elif speech_probability > 0.20:
            blend = (speech_probability - 0.20) / (self.prob_thresh - 0.20)
            target_floor = (1.0 - blend) * self.uncertain_voice_gain + blend * self.min_voice_gain
        else:
            blend = max(0.0, speech_probability / 0.20)
            target_floor = (1.0 - blend) * self.noise_floor_linear + blend * self.uncertain_voice_gain

        if current_attenuation < target_floor:
            deficit = target_floor - current_attenuation
            mix_raw = float(np.clip(deficit / max(1e-6, 1.0 - current_attenuation), 0.0, 0.85))
            protected = (1.0 - mix_raw) * clean + mix_raw * raw
            effective_gain = float(target_floor)
        else:
            protected = clean
            effective_gain = float(current_attenuation)

        smoothed_gain = self.smoother.smooth_scalar(effective_gain)
        final_protected = raw * smoothed_gain if speech_probability >= self.prob_thresh else protected
        return final_protected, smoothed_gain

