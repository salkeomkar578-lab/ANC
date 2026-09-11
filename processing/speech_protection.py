"""
Speech Protection Gate for IGARD-Net.
Acts as a fail-safe supervisor positioned after NLMS and Spectral Cleanup.
Monitors the speech probability and the frame-by-frame attenuation.
Ensures human voice is never hard-muted or overly attenuated, maintaining
natural speech continuity, vowel formants, and syllable endings.
"""

from typing import Tuple, Optional
import numpy as np
from processing.gain_smoother import GainSmoother


class SpeechProtectionGate:
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
    ):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.prob_thresh = probability_threshold
        self.min_voice_gain = minimum_voice_gain
        self.uncertain_voice_gain = uncertain_voice_gain
        self.max_suppression_linear = 10.0 ** (-max_suppression_db / 20.0)
        self.noise_floor_linear = 10.0 ** (-noise_suppression_floor_db / 20.0) # ~0.01 (-40 dB)
        self.enabled = enabled

        # Gain smoother dedicated to protection blend
        self.smoother = GainSmoother(
            sample_rate=sample_rate,
            frame_size=frame_size,
            attack_ms=10.0,
            release_ms=60.0,
            max_delta_per_frame=0.08,
            min_gain=self.noise_floor_linear,
        )

        # Compressor / limiter parameters
        self.comp_threshold = 0.50 # -6 dBFS
        self.comp_ratio = 2.0
        self._comp_envelope: float = 0.0

    def reset(self):
        self.smoother.reset(1.0)
        self._comp_envelope = 0.0

    def protect(
        self,
        raw_input_frame: np.ndarray,
        cleaned_frame: np.ndarray,
        speech_probability: float,
        noise_probability: float,
    ) -> Tuple[np.ndarray, float]:
        """
        Enforces speech preservation and prevents excessive voice attenuation,
        while allowing deep cancellation of background noise and chatter when
        human speech is absent.
        Returns:
            protected_frame: 1D numpy array
            applied_gain: float effective gain applied to speech
        """
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

        # Target preservation gain floor based on speech probability:
        if speech_probability >= self.prob_thresh:
            # Confident speech: strict voice preservation floor (never drops below 0.55)
            target_floor = self.min_voice_gain
        elif speech_probability > 0.20:
            # Transitional / soft syllable speech: smoothly blend between uncertain voice gain and min voice gain
            blend = (speech_probability - 0.20) / (self.prob_thresh - 0.20)
            target_floor = (1.0 - blend) * self.uncertain_voice_gain + blend * self.min_voice_gain
        else:
            # Background noise / background chatter without target speech:
            # Allow aggressive background cancellation down to noise_floor_linear (-40 dB),
            # smoothly scaling up only as speech probability approaches 0.20.
            blend = max(0.0, speech_probability / 0.20)
            target_floor = (1.0 - blend) * self.noise_floor_linear + blend * self.uncertain_voice_gain

        # If clean_rms has dropped below target floor, compute blend factor
        if current_attenuation < target_floor:
            deficit = target_floor - current_attenuation
            mix_raw = float(np.clip(deficit / max(1e-6, 1.0 - current_attenuation), 0.0, 0.85))
            blended = (1.0 - mix_raw) * clean + mix_raw * raw
        else:
            blended = clean

        # Ensure output RMS matches smooth target gain
        blended_rms = float(np.sqrt(np.mean(blended ** 2))) + 1e-10
        eff_gain = blended_rms / raw_rms
        smooth_gain = self.smoother.smooth_scalar(eff_gain)
        
        # Strictly enforce target speech preservation floor when speech is active
        if speech_probability >= 0.20:
            smooth_gain = max(target_floor, smooth_gain)

        gain_correction = smooth_gain / eff_gain
        out = blended * gain_correction

        # Gentle limiter to prevent digital clipping
        peak = float(np.max(np.abs(out)))
        if peak > 0.95:
            out = out * (0.95 / peak)

        return out, smooth_gain

