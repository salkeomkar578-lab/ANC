"""
Stage 5: Overlap-Add (OLA) Wiener Soft Spectral Cleanup.
Implements continuous stateful 50% overlap-add STFT soft masking.
Eliminates block-boundary amplitude dipping and suppresses musical noise.
"""

from typing import Optional
import numpy as np
from backends.base import ComputeBackend
from backends.cpu_backend import CPUBackend


class ResidualCleanup:
    def __init__(
        self,
        sample_rate: int = 16000,
        gate_strength: float = 1.4,
        min_gain: float = 0.08,
        backend: Optional[ComputeBackend] = None,
    ):
        self.sample_rate = sample_rate
        self.gate_strength = gate_strength
        self.min_gain = min_gain
        self.backend = backend if backend is not None else CPUBackend()

        self._noise_profile: Optional[np.ndarray] = None
        
        # Stateful Overlap-Add buffers
        self._prev_input: Optional[np.ndarray] = None
        self._overlap_buf: Optional[np.ndarray] = None
        self._window: Optional[np.ndarray] = None
        self._window_len: int = 0

    def update_noise_profile(self, noise_block: np.ndarray):
        """Updates the running spectral noise profile during stationary noise moments."""
        n = len(noise_block)
        win = np.hanning(n)
        _, power, _ = self.backend.compute_rfft_power(noise_block, win)
        
        if self._noise_profile is None or len(self._noise_profile) != len(power):
            self._noise_profile = power.copy()
        else:
            self._noise_profile = 0.85 * self._noise_profile + 0.15 * power

    def clean(self, block: np.ndarray, aggressiveness: float = 1.0) -> np.ndarray:
        """
        Cleans block using 50% Overlap-Add Wiener filtering.
        aggressiveness: scalar [0.0 to 1.5] scaling alpha based on noise severity.
        """
        n = len(block)
        if n == 0:
            return block

        # Setup 2*N window for seamless 50% OLA
        win_len = 2 * n
        if self._window_len != win_len:
            self._window_len = win_len
            # Square-root Hanning for analysis and synthesis (perfect reconstruction property)
            self._window = np.sqrt(np.hanning(win_len)).astype(np.float64)
            self._prev_input = np.zeros(n, dtype=np.float64)
            self._overlap_buf = np.zeros(n, dtype=np.float64)

        # Concatenate previous block and current block
        frame_2n = np.concatenate((self._prev_input, block))
        self._prev_input = block.copy()

        # RFFT and power spectrum
        spectrum, power, _ = self.backend.compute_rfft_power(frame_2n, self._window)

        # Noise power estimation
        if self._noise_profile is None:
            # Low-level percentile noise floor fallback
            noise_power = 0.003 * np.max(power + 1.0e-12)
        else:
            noise_power = self._noise_profile

        # Apply soft spectral gain mask
        alpha = self.gate_strength * aggressiveness
        cleaned_spec = self.backend.apply_spectral_mask(
            spectrum=spectrum,
            power=power,
            noise_power=noise_power,
            alpha=alpha,
            min_gain=self.min_gain,
        )

        # Inverse RFFT and synthesis windowing
        rec_2n = self.backend.compute_irfft(cleaned_spec, n=win_len)
        rec_win = rec_2n * self._window

        # Overlap-Add: combine with previous tail
        out_block = rec_win[:n] + self._overlap_buf
        self._overlap_buf = rec_win[n:].copy()

        return out_block
