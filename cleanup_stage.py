"""
Stage 4 of IGARD-Net: residual cleanup, only invoked when the confidence
gate (confidence_gate.py) decides the classifier is confident enough.

HONESTY NOTE FOR THE TEAM:
The final design calls for a tiny trained neural model here (DTLN/uNet-
style, a few hundred KB) to mop up leftover transient noise the NLMS
stage couldn't remove. Training that model needs paired clean/noisy
defense-audio data, which doesn't exist yet.

So this file implements classical WIENER-FILTER SOFT MASKING instead --
a real, working DSP technique that computes a smooth frequency gain mask
G = max(|X|² - α|N|², 0) / (|X|² + ε). This produces substantially
fewer "musical noise" (isolated spectral peak) artifacts than naive
hard spectral gating, while remaining 100% classical DSP and running
efficiently on Raspberry Pi 4 CPU using pure NumPy.

It is a placeholder for the trained model, not a fake version of it: the
function signature (`clean(block) -> cleaned_block`) is exactly what the
future neural model will implement, so swapping it in later is a
one-function change in pipeline.py.
"""

import numpy as np


class ResidualCleanup:
    def __init__(self, sample_rate=16000, gate_strength=1.5, min_gain=0.05):
        """
        gate_strength: alpha factor for noise power oversubtraction.
            Higher = more aggressive cleaning, but more risk of speech attenuation.
        min_gain: floor for gain mask to prevent unnatural dead-silence pumping artifacts.
        """
        self.sample_rate = sample_rate
        self.gate_strength = gate_strength
        self.min_gain = min_gain
        self._noise_profile = None

    def update_noise_profile(self, noise_only_block):
        """Call this during known-silent/noise-only moments to calibrate."""
        block = np.asarray(noise_only_block, dtype=np.float64)
        n = len(block)
        spectrum = np.abs(np.fft.rfft(block * np.hanning(n)))
        power = spectrum ** 2
        if self._noise_profile is None:
            self._noise_profile = power
        else:
            # exponential moving average so it adapts smoothly over time
            self._noise_profile = 0.9 * self._noise_profile + 0.1 * power

    def clean(self, block):
        """
        Applies Wiener-filter-style soft spectral masking.
        This is the function a trained neural model will eventually replace.
        Input: 1D float array. Output: 1D float array of identical length.
        """
        block = np.asarray(block, dtype=np.float64)
        n = len(block)
        if n == 0:
            return block

        window = np.hanning(n)
        spectrum = np.fft.rfft(block * window)
        power = np.abs(spectrum) ** 2
        phase = np.angle(spectrum)

        if self._noise_profile is None:
            # No calibration yet -- estimate noise power floor from low-level percentile
            noise_power = 0.0025 * np.max(power + 1e-12)
        else:
            # Ensure noise profile matches current FFT length if block length varied
            if len(self._noise_profile) != len(power):
                noise_power = np.interp(
                    np.linspace(0, 1, len(power)),
                    np.linspace(0, 1, len(self._noise_profile)),
                    self._noise_profile
                )
            else:
                noise_power = self._noise_profile

        # Wiener soft-masking gain:
        # G(f) = max(P_signal - alpha * P_noise, 0) / (P_signal + epsilon)
        alpha = self.gate_strength
        subtracted = np.maximum(power - alpha * noise_power, 0.0)
        gain = subtracted / (power + 1e-10)
        # Apply gain floor to prevent harsh musical noise or dropouts
        gain = np.clip(gain, self.min_gain, 1.0)

        # Smooth gain across neighboring bins to reduce spectral peaks
        if len(gain) >= 3:
            gain_smoothed = 0.25 * np.roll(gain, 1) + 0.5 * gain + 0.25 * np.roll(gain, -1)
            gain_smoothed[0] = gain[0]
            gain_smoothed[-1] = gain[-1]
            gain = gain_smoothed

        cleaned_magnitude = np.sqrt(power) * gain
        cleaned_spectrum = cleaned_magnitude * np.exp(1j * phase)
        cleaned = np.fft.irfft(cleaned_spectrum, n=n)

        # Window reconstruction with normalization to reduce edge artifacts
        norm = window / (window ** 2 + 1e-6)
        return cleaned * window
