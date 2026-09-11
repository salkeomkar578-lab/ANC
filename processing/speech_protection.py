"""
IGARD-Net V4 — AI Speech-Preserving Adaptive Noise Suppression.
Speech Protection Controller Module.

Responsibilities:
1. Speech Awareness: Evaluates continuous Silero VAD confidence with stateful hysteresis.
2. Speech Preservation: Protects vocal timbre, formant harmonics, and syllable continuity.
3. Adaptive Suppression: Automatically selects suppression aggressiveness without pumping.
4. Energy Collapse Guard: Prevents RNNoise from dropping vocal power below target floor.
5. Zero Hard-Muting: Strictly forbids zeroing speech or silent regions.
6. Temporal Smoothing: Hysteresis, 120ms speech hold, 150ms release, and boundary cross-fade.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np


class SpeechProtectionController:
    """
    V4 Speech-First Protection Controller.
    Supervises the RNNoise noise suppression stage using continuous Silero VAD signals.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_size: int = 256,
        threshold_on: float = 0.60,
        threshold_off: float = 0.40,
        attack_ms: float = 10.0,
        hold_ms: float = 120.0,
        release_ms: float = 150.0,
        default_suppression: float = 0.75,
        minimum_voice_gain: float = 0.50,
        background_attenuation_floor: float = 0.15,
        enabled: bool = True,
        **kwargs,
    ):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.dt = frame_size / sample_rate
        self.dt_ms = self.dt * 1000.0

        # Speech confidence thresholds with hysteresis
        self.threshold_on = float(kwargs.get("high_speech_thresh", threshold_on))
        self.threshold_off = float(kwargs.get("low_speech_thresh", threshold_off))
        self.minimum_voice_gain = float(kwargs.get("min_voice_gain", minimum_voice_gain))
        self.bg_floor = float(background_attenuation_floor)
        self.enabled = enabled

        # Temporal smoothing coefficients
        self.alpha_attack = float(np.exp(-self.dt_ms / max(1.0, attack_ms)))
        self.alpha_release = float(np.exp(-self.dt_ms / max(1.0, release_ms)))
        self.hold_frames = int(np.ceil((hold_ms / 1000.0) / self.dt))

        # 20ms Phase alignment delay buffer (aligns raw audio with RNNoise 20ms filterbank delay)
        self._delay_samples = int(self.sample_rate * 0.020)
        self._raw_delay_buffer = np.zeros(self._delay_samples, dtype=np.float32)

        # Persistent stateful variables
        self._speech_state: bool = False
        self._smoothed_speech_prob: float = 0.0
        self._hold_counter: int = 0
        self._suppression_strength: float = default_suppression
        self._target_suppression: float = default_suppression
        self._noise_floor_rms: float = 0.02
        self._smoothed_gain: float = 1.0
        self._last_out_sample: Optional[float] = None

        # Autopilot
        self.autopilot: bool = True
        self.manual_suppression: float = default_suppression

    def reset(self) -> None:
        """Reset internal recurrent states between audio streams."""
        self._speech_state = False
        self._smoothed_speech_prob = 0.0
        self._hold_counter = 0
        self._suppression_strength = 0.75
        self._target_suppression = 0.75
        self._noise_floor_rms = 0.02
        self._smoothed_gain = 1.0
        self._last_out_sample = None
        if len(self._raw_delay_buffer) != self._delay_samples:
            self._raw_delay_buffer = np.zeros(self._delay_samples, dtype=np.float32)
        else:
            self._raw_delay_buffer.fill(0.0)

    def update_speech_tracking(self, raw_prob: float, rnnoise_vad: float = 0.0) -> Tuple[float, bool]:
        """
        Stateful speech tracking with hysteresis, attack, hold, and release.
        Returns:
            smoothed_probability: Continuous smoothed speech confidence in [0.0, 1.0]
            is_speech_active: Boolean hysteresis state (True if speaker is speaking or within hold)
        """
        raw_prob = float(np.clip(raw_prob, 0.0, 1.0))

        # Fast onset detection: if RNNoise strongly detects speech and Silero is warming up
        if rnnoise_vad >= 0.75 and raw_prob >= 0.20:
            raw_prob = max(raw_prob, 0.70)

        # Hysteresis state machine
        if self._speech_state:
            # Currently active: keep active unless confidence drops below threshold_off AND hold expires
            if raw_prob < self.threshold_off:
                if self._hold_counter > 0:
                    self._hold_counter -= 1
                else:
                    self._speech_state = False
            else:
                self._hold_counter = self.hold_frames
        else:
            # Currently inactive: transition to active only when confidence exceeds threshold_on
            if raw_prob >= self.threshold_on or (rnnoise_vad >= 0.85 and raw_prob >= 0.30):
                self._speech_state = True
                self._hold_counter = self.hold_frames
                # Snap gain up on speech onset to avoid muting the first syllable
                if self._smoothed_gain < 0.80:
                    self._smoothed_gain = 0.85

        # Asymmetrical ballistics smoothing
        if raw_prob >= self._smoothed_speech_prob:
            # Fast attack (10ms)
            self._smoothed_speech_prob = (
                (1.0 - self.alpha_attack) * raw_prob
                + self.alpha_attack * self._smoothed_speech_prob
            )
        else:
            # Hold hangover or gentle release (150ms)
            if self._hold_counter > 0:
                # During hangover, decay very slowly to bridge word pauses
                self._smoothed_speech_prob = max(0.50, self._smoothed_speech_prob * 0.99)
            else:
                self._smoothed_speech_prob = (
                    (1.0 - self.alpha_release) * raw_prob
                    + self.alpha_release * self._smoothed_speech_prob
                )

        self._smoothed_speech_prob = float(np.clip(self._smoothed_speech_prob, 0.0, 1.0))
        return self._smoothed_speech_prob, self._speech_state

    def update_adaptive_suppression(
        self,
        smoothed_prob: float,
        is_speech_active: bool,
        input_rms: float,
        rnnoise_vad: float = 0.0,
    ) -> float:
        """
        Adaptive Suppression Controller.
        Determines suppression strength based on environmental noise floor,
        speech presence, and acoustic stability.
        """
        # Continuously track ambient noise floor during non-speech intervals
        if not is_speech_active and smoothed_prob < self.threshold_off:
            self._noise_floor_rms = 0.96 * self._noise_floor_rms + 0.04 * max(1e-4, input_rms)

        if self.autopilot:
            # Acoustic scene categorization based on ambient noise floor
            if self._noise_floor_rms < 0.012:
                # Quiet / clean acoustic environment
                target = 0.35
            elif self._noise_floor_rms < 0.045:
                # Moderate office / tactical room background
                target = 0.68
            elif self._noise_floor_rms < 0.12:
                # Strong engine / vehicle / rotor noise
                target = 0.86
            else:
                # Heavy machinery / wind / severe broadband noise
                target = 0.95

            # When human voice is active, cap suppression strength to protect vocal harmonics
            if is_speech_active:
                target = min(target, 0.80)
        else:
            target = float(np.clip(self.manual_suppression, 0.0, 1.0))

        self._target_suppression = target
        # Smooth suppression strength transitions to eliminate pumping
        self._suppression_strength = 0.92 * self._suppression_strength + 0.08 * target
        return self._suppression_strength

    def protect(
        self,
        raw_frame: np.ndarray,
        suppressed_frame: np.ndarray,
        speech_probability: float,
        rnnoise_vad: float = 0.0,
        **kwargs,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Execute speech-first protection and adaptive enhancement.
        
        Args:
            raw_frame: Input audio frame (normalized float32)
            suppressed_frame: RNNoise-cleaned audio frame
            speech_probability: Continuous confidence from Silero VAD
            rnnoise_vad: VAD confidence from RNNoise C engine
            
        Returns:
            Tuple of (protected_audio_frame, telemetry_dict)
        """
        raw = np.asarray(raw_frame, dtype=np.float32)
        supp = np.asarray(suppressed_frame, dtype=np.float32)
        n = len(raw)

        # 1. Align phase with 20ms lookahead delay line
        if len(self._raw_delay_buffer) != self._delay_samples:
            self._raw_delay_buffer = np.zeros(self._delay_samples, dtype=np.float32)
        combined_buf = np.concatenate([self._raw_delay_buffer, raw])
        delayed_raw = combined_buf[:n]
        self._raw_delay_buffer = combined_buf[n:]

        raw_rms = float(np.sqrt(np.mean(delayed_raw ** 2))) + 1e-12
        supp_rms = float(np.sqrt(np.mean(supp ** 2))) + 1e-12

        # 2. Stateful speech tracking with hysteresis
        smoothed_prob, is_speech_active = self.update_speech_tracking(speech_probability, rnnoise_vad=rnnoise_vad)

        # 3. Adaptive suppression control
        supp_strength = self.update_adaptive_suppression(
            smoothed_prob=smoothed_prob,
            is_speech_active=is_speech_active,
            input_rms=raw_rms,
            rnnoise_vad=rnnoise_vad,
        )

        # 4. Processing logic by speech regime
        energy_guard_triggered = False
        if is_speech_active or smoothed_prob >= self.threshold_on:
            # === REGIME 1: CONFIRMED HUMAN SPEECH ===
            # Voice is active: RNNoise removes background noise underneath speech.
            # Blend a small phase-aligned harmonic proportion (8-14%) to restore vocal warmth
            status_text = "SPEECH ACTIVE"
            protection_active = True
            vocal_warmth_mix = float(np.clip(0.12 * smoothed_prob, 0.05, 0.15))
            processed = (1.0 - vocal_warmth_mix) * supp + vocal_warmth_mix * delayed_raw
            target_gain = 1.0

            # --- INTELLIGENT SPEECH ENERGY COLLAPSE GUARD ---
            curr_rms = float(np.sqrt(np.mean(processed ** 2))) + 1e-12
            required_min_rms = raw_rms * self.minimum_voice_gain
            if curr_rms < required_min_rms and raw_rms > 0.0005:
                # RNNoise over-suppressed speech: blend delayed raw to guarantee minimum voice gain
                energy_guard_triggered = True
                status_text = "SPEECH RECOVERY (COLLAPSE GUARD)"
                deficit = required_min_rms - curr_rms
                blend_raw = float(np.clip(deficit / max(1e-4, raw_rms - curr_rms), 0.0, 0.85))
                processed = (1.0 - blend_raw) * processed + blend_raw * delayed_raw
                post_rms = float(np.sqrt(np.mean(processed ** 2))) + 1e-12
                if post_rms < required_min_rms:
                    processed = processed * (required_min_rms / post_rms)

        elif smoothed_prob >= self.threshold_off:
            # === REGIME 2: TRANSITIONAL / UNCERTAIN SPEECH ===
            # Preserve speech cautiously; avoid gating or abrupt gain jumps
            status_text = "SPEECH TRANSITION"
            protection_active = True
            blend = (smoothed_prob - self.threshold_off) / max(1e-4, self.threshold_on - self.threshold_off)
            vocal_warmth_mix = 0.08 * blend
            processed = (1.0 - vocal_warmth_mix) * supp + vocal_warmth_mix * delayed_raw
            target_gain = 1.0

            curr_rms = float(np.sqrt(np.mean(processed ** 2))) + 1e-12
            transitional_min_rms = raw_rms * (self.minimum_voice_gain * 0.85)
            if curr_rms < transitional_min_rms and raw_rms > 0.0005:
                energy_guard_triggered = True
                deficit = transitional_min_rms - curr_rms
                blend_raw = float(np.clip(deficit / max(1e-4, raw_rms - curr_rms), 0.0, 0.75))
                processed = (1.0 - blend_raw) * processed + blend_raw * delayed_raw
                post_rms = float(np.sqrt(np.mean(processed ** 2))) + 1e-12
                if post_rms < transitional_min_rms:
                    processed = processed * (transitional_min_rms / post_rms)

        else:
            # === REGIME 3: NON-SPEECH BACKGROUND NOISE ===
            # Background only: apply RNNoise + gentle adaptive post-attenuation down to safety floor
            status_text = "NOISE SUPPRESSION"
            protection_active = False
            processed = supp

            # Continuous attenuation based on distance below threshold_off
            prob_ratio = smoothed_prob / max(1e-4, self.threshold_off)
            depth = supp_strength * 0.70  # max ~0.66 attenuation
            target_gain = float(np.clip(
                (1.0 - depth) + prob_ratio * depth,
                self.bg_floor,
                1.0,
            ))

        # 5. Continuous gain smoothing to prevent pumping and frame-to-frame jumping
        if target_gain < self._smoothed_gain:
            self._smoothed_gain = (1.0 - self.alpha_release) * target_gain + self.alpha_release * self._smoothed_gain
        else:
            self._smoothed_gain = (1.0 - self.alpha_attack) * target_gain + self.alpha_attack * self._smoothed_gain

        self._smoothed_gain = float(np.clip(self._smoothed_gain, self.bg_floor, 1.0))
        applied_gain = max(0.95, self._smoothed_gain) if is_speech_active else self._smoothed_gain
        processed = processed * applied_gain

        # 6. Acoustic Comfort Floor (Eliminates dead silence & exact zero dropouts)
        p_rms = float(np.sqrt(np.mean(processed ** 2)))
        comfort_target = max(1.5e-4, raw_rms * 0.035)
        if p_rms < comfort_target:
            processed = processed + delayed_raw * 0.035

        # 6. Sample-Boundary Cosine Cross-Fade (Eliminates boundary clicking and ticking)
        if self._last_out_sample is not None and n >= 16:
            step = float(self._last_out_sample - processed[0])
            if abs(step) > 0.015:
                # 16-sample smooth cosine taper
                t = np.linspace(0.0, np.pi / 2.0, 16, dtype=np.float32)
                ramp = np.cos(t) ** 2
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
            "effective_gain": self._smoothed_gain,
        }

        return processed, telemetry


class SpeechProtectionGate:
    """Legacy Speech Protection Gate for backwards-compatibility with unit tests."""

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
        self.prob_thresh = float(probability_threshold)
        self.min_voice_gain = float(minimum_voice_gain)
        self.uncertain_voice_gain = float(uncertain_voice_gain)
        self.max_suppression_linear = 10.0 ** (-max_suppression_db / 20.0)
        self.noise_floor_linear = 10.0 ** (-noise_suppression_floor_db / 20.0)
        self.enabled = enabled
        self._smoothed_gain: float = 1.0

    def protect(
        self,
        raw_frame: np.ndarray,
        cleaned_frame: np.ndarray,
        speech_probability: float,
        noise_probability: float = 0.0,
        **kwargs,
    ) -> Tuple[np.ndarray, float]:
        if not self.enabled:
            return cleaned_frame, 1.0

        raw = np.asarray(raw_frame, dtype=np.float32)
        clean = np.asarray(cleaned_frame, dtype=np.float32)
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
            effective_gain = float(target_floor)
        else:
            effective_gain = float(current_attenuation)

        self._smoothed_gain = 0.90 * self._smoothed_gain + 0.10 * effective_gain
        applied_gain = max(target_floor, self._smoothed_gain) if speech_probability >= 0.20 else self._smoothed_gain
        final_protected = raw * applied_gain if speech_probability >= self.prob_thresh else clean * (applied_gain / max(1e-6, current_attenuation))
        return final_protected, applied_gain
