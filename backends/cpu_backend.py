"""
CPU Backend for IGARD-Net using optimized NumPy and SciPy.
Designed for low latency on edge processors (e.g. Raspberry Pi 4) and standard CPUs.
"""

from typing import Tuple
import numpy as np
from backends.base import ComputeBackend


class CPUBackend(ComputeBackend):
    def __init__(self):
        self._name = "CPU"

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_cuda(self) -> bool:
        return False

    def nlms_process_block(
        self,
        weights: np.ndarray,
        ref_buffer: np.ndarray,
        primary_block: np.ndarray,
        ref_block: np.ndarray,
        mu: float,
        eps: float,
        leakage: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        High-performance sequential NLMS block processing.
        Uses running energy tracking and in-place array updates to minimize overhead.
        """
        num_taps = len(weights)
        block_len = len(primary_block)
        
        err_out = np.empty(block_len, dtype=np.float64)
        noise_est_out = np.empty(block_len, dtype=np.float64)
        
        concat_ref = np.concatenate((ref_buffer, ref_block))
        w = weights.copy()
        
        # Initial sliding energy for the first window
        init_x = concat_ref[:num_taps]
        running_norm = float(np.dot(init_x, init_x)) + eps

        for k in range(block_len):
            x_k = concat_ref[k : k + num_taps][::-1]
            y_k = float(np.dot(w, x_k))
            e_k = float(primary_block[k] - y_k)
            
            # In-place weight update
            factor = (mu / running_norm) * e_k
            if leakage < 1.0:
                w *= leakage
            w += factor * x_k
            
            err_out[k] = e_k
            noise_est_out[k] = y_k

            # Update running energy for next sample: subtract exiting sample, add entering sample
            if k + num_taps < len(concat_ref):
                old_val = concat_ref[k]
                new_val = concat_ref[k + num_taps]
                running_norm = max(eps, running_norm - old_val * old_val + new_val * new_val)
            
        new_ref_buffer = concat_ref[-num_taps:].copy()
        return err_out, noise_est_out, w, new_ref_buffer

    def compute_rfft_power(self, block: np.ndarray, window: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        windowed = block * window
        spectrum = np.fft.rfft(windowed)
        power = np.abs(spectrum) ** 2
        phase = np.angle(spectrum)
        return spectrum, power, phase

    def compute_irfft(self, spectrum: np.ndarray, n: int) -> np.ndarray:
        return np.fft.irfft(spectrum, n=n)

    def apply_spectral_mask(
        self,
        spectrum: np.ndarray,
        power: np.ndarray,
        noise_power: np.ndarray,
        alpha: float,
        min_gain: float,
    ) -> np.ndarray:
        # Interpolate noise power if sizes differ
        if len(noise_power) != len(power):
            n_p = np.interp(
                np.linspace(0, 1, len(power)),
                np.linspace(0, 1, len(noise_power)),
                noise_power
            )
        else:
            n_p = noise_power

        subtracted = np.maximum(power - alpha * n_p, 0.0)
        gain = subtracted / (power + 1.0e-10)
        gain = np.clip(gain, min_gain, 1.0)
        
        # 3-tap spectral smoothing across adjacent bins to prevent musical noise
        if len(gain) >= 3:
            gain_smoothed = 0.25 * np.roll(gain, 1) + 0.5 * gain + 0.25 * np.roll(gain, -1)
            gain_smoothed[0] = gain[0]
            gain_smoothed[-1] = gain[-1]
            gain = gain_smoothed

        cleaned_spectrum = spectrum * gain
        return cleaned_spectrum
