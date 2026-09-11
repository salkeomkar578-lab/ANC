"""
Stage 5 of IGARD-Net: Speech-Aware Overlap-Add (OLA) Wiener Spectral Cleanup.
Protects speech formants (300 Hz - 3400 Hz), preserves vowel energy and consonant
transitions, uses 50% Overlap-Add reconstruction (zero boundary dips), and only
updates noise profiles during confirmed non-speech intervals.
"""

from typing import Optional
import numpy as np
from gain_smoother import GainSmoother


class ResidualCleanup:
    def __init__(
        self,
        sample_rate: int = 16000,
        gate_strength: float = 1.0,
        min_gain: float = 0.35,
    ):
        self.sample_rate = sample_rate
        self.gate_strength = gate_strength
        self.min_gain = min_gain

        self._noise_profile: Optional[np.ndarray] = None
        self._mask_smoother = GainSmoother(
            sample_rate=sample_rate,
            frame_size=256,
            attack_ms=15.0,
            release_ms=80.0,
            max_delta_per_frame=0.06,
            min_gain=min_gain,
        )

        # Stateful 50% Overlap-Add buffers
        self._prev_input: Optional[np.ndarray] = None
        self._overlap_buf: Optional[np.ndarray] = None
        self._window: Optional[np.ndarray] = None
        self._window_len: int = 0
        self._freqs: Optional[np.ndarray] = None
        self._speech_band_mask: Optional[np.ndarray] = None

    def reset(self):
        self._noise_profile = None
        self._prev_input = None
        self._overlap_buf = None
        self._window_len = 0
        self._mask_smoother.reset(1.0)

    def update_noise_profile(self, noise_only_block: np.ndarray, speech_prob: float = 0.0):
        """
        Only updates noise profile during verified quiet or noise-only moments
        to prevent learning speech formants as background noise.
        """
        if speech_prob > 0.30:
            return  # Never update noise profile during speech

        block = np.asarray(noise_only_block, dtype=np.float64)
        n = len(block)
        if n == 0:
            return

        win = np.hanning(n)
        spectrum = np.fft.rfft(block * win)
        power = np.abs(spectrum) ** 2

        if self._noise_profile is None or len(self._noise_profile) != len(power):
            self._noise_profile = power.copy()
        else:
            self._noise_profile = 0.90 * self._noise_profile + 0.10 * power

    def clean(
        self,
        block: np.ndarray,
        speech_prob: float = 0.0,
        aggressiveness: float = 1.0,
    ) -> np.ndarray:
        """
        Speech-aware Wiener spectral cleanup with 50% Overlap-Add.
        """
        block = np.asarray(block, dtype=np.float64)
        n = len(block)
        if n == 0:
            return block

        # Setup 2*N window for continuous 50% OLA
        win_len = 2 * n
        if self._window_len != win_len or self._prev_input is None:
            self._window_len = win_len
            self._window = np.sqrt(np.hanning(win_len)).astype(np.float64)
            self._prev_input = np.zeros(n, dtype=np.float64)
            self._overlap_buf = np.zeros(n, dtype=np.float64)
            self._freqs = np.fft.rfftfreq(win_len, 1.0 / self.sample_rate)
            self._speech_band_mask = (self._freqs >= 300.0) & (self._freqs <= 3400.0)

        # Prepend previous frame to form 2*N analysis window
        frame_2n = np.concatenate((self._prev_input, block))
        self._prev_input = block.copy()

        # Windowed RFFT
        windowed = frame_2n * self._window
        spectrum = np.fft.rfft(windowed)
        power = np.abs(spectrum) ** 2
        phase = np.angle(spectrum)

        # Update noise profile slowly if speech probability is very low
        if speech_prob < 0.20:
            if self._noise_profile is None or len(self._noise_profile) != len(power):
                self._noise_profile = power * 0.5
            else:
                self._noise_profile = 0.92 * self._noise_profile + 0.08 * power

        if self._noise_profile is None:
            # Conservative initial noise power estimate
            noise_power = np.percentile(power, 15) * np.ones_like(power)
        else:
            if len(self._noise_profile) != len(power):
                noise_power = np.interp(
                    np.linspace(0, 1, len(power)),
                    np.linspace(0, 1, len(self._noise_profile)),
                    self._noise_profile,
                )
            else:
                noise_power = self._noise_profile

        # Speech-Adaptive Suppression Control:
        # Lower alpha when speech is present to preserve vowels and consonants
        alpha = self.gate_strength * aggressiveness * max(0.25, 1.0 - 0.75 * speech_prob)

        # Wiener gain: G = max(P_sig - alpha * P_noise, 0) / P_sig
        subtracted = np.maximum(power - alpha * noise_power, 0.0)
        raw_gain = subtracted / (power + 1e-10)

        # Frequency-Dependent Speech Formant Protection:
        # In speech formant bands (300 - 3400 Hz), enforce an elevated gain floor
        formant_floor = np.full_like(raw_gain, self.min_gain)
        if self._speech_band_mask is not None:
            # Raise floor in speech band proportional to speech probability
            formant_boost = self.min_gain + (0.85 - self.min_gain) * speech_prob
            formant_floor[self._speech_band_mask] = np.maximum(
                formant_floor[self._speech_band_mask],
                formant_boost,
            )

        gain_clamped = np.maximum(raw_gain, formant_floor)
        gain_clamped = np.clip(gain_clamped, 0.0, 1.0)

        # 3-tap spectral smoothing across frequency bins
        if len(gain_clamped) >= 3:
            gain_clamped = 0.25 * np.roll(gain_clamped, 1) + 0.5 * gain_clamped + 0.25 * np.roll(gain_clamped, -1)

        # Temporal gain smoothing across frames
        gain_smooth = self._mask_smoother.smooth_spectrum_mask(gain_clamped)

        # Apply gain mask to spectrum
        cleaned_spectrum = spectrum * gain_smooth

        # Inverse RFFT and synthesis window
        rec_2n = np.fft.irfft(cleaned_spectrum, n=win_len)
        rec_win = rec_2n * self._window

        # Overlap-Add
        out_block = rec_win[:n] + self._overlap_buf
        self._overlap_buf = rec_win[n:].copy()

        return out_block
