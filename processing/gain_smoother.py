"""
Gain Smoother for IGARD-Net.
Implements asymmetrical attack/release gain smoothing with hard bounds on
frame-to-frame gain delta. Eliminates pumping, clicking, popping, and sudden
attenuation artifacts.
"""

from typing import Union, Optional
import numpy as np


class GainSmoother:
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_size: int = 256,
        attack_ms: float = 15.0,
        release_ms: float = 80.0,
        max_delta_per_frame: float = 0.05,
        min_gain: float = 0.25,
    ):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.frame_duration_ms = (frame_size / sample_rate) * 1000.0

        # Smoothing coefficients
        self.alpha_attack = float(np.exp(-self.frame_duration_ms / max(1.0, attack_ms)))
        self.alpha_release = float(np.exp(-self.frame_duration_ms / max(1.0, release_ms)))
        self.max_delta = max_delta_per_frame
        self.min_gain = min_gain

        self._current_gain: float = 1.0
        self._current_vector_gain: Union[np.ndarray, None] = None

    def reset(self, initial_gain: float = 1.0):
        self._current_gain = initial_gain
        self._current_vector_gain = None

    def smooth_scalar(self, target_gain: float) -> float:
        """
        Smooths a scalar frame gain target.
        Enforces maximum rate of change and minimum gain floor.
        """
        target = float(np.clip(target_gain, self.min_gain, 1.0))
        
        # Determine whether this is attack (attenuating) or release (recovering)
        if target < self._current_gain:
            # Attenuation (attack towards suppression)
            smoothed = (1.0 - self.alpha_attack) * target + self.alpha_attack * self._current_gain
            # Slew rate limit
            delta = self._current_gain - smoothed
            if delta > self.max_delta:
                smoothed = self._current_gain - self.max_delta
        else:
            # Recovery (release back towards unity gain)
            smoothed = (1.0 - self.alpha_release) * target + self.alpha_release * self._current_gain
            # Slew rate limit
            delta = smoothed - self._current_gain
            if delta > self.max_delta:
                smoothed = self._current_gain + self.max_delta

        self._current_gain = float(np.clip(smoothed, self.min_gain, 1.0))
        return self._current_gain

    def smooth_spectrum_mask(self, target_mask: np.ndarray) -> np.ndarray:
        """
        Smooths a 1D spectral gain mask across time and frequency.
        """
        target_mask = np.clip(target_mask, self.min_gain, 1.0)
        if self._current_vector_gain is None or len(self._current_vector_gain) != len(target_mask):
            self._current_vector_gain = target_mask.copy()
            return self._current_vector_gain

        # Asymmetric time smoothing
        is_attack = target_mask < self._current_vector_gain
        alpha = np.where(is_attack, self.alpha_attack, self.alpha_release)
        
        smoothed = (1.0 - alpha) * target_mask + alpha * self._current_vector_gain
        
        # Bounded rate of change per frame
        delta = np.abs(smoothed - self._current_vector_gain)
        excess = delta > self.max_delta
        if np.any(excess):
            sign = np.sign(smoothed - self._current_vector_gain)
            smoothed[excess] = self._current_vector_gain[excess] + sign[excess] * self.max_delta

        self._current_vector_gain = np.clip(smoothed, self.min_gain, 1.0)
        return self._current_vector_gain

    @staticmethod
    def soft_limit(audio: np.ndarray, ceiling: float = 0.95) -> np.ndarray:
        """
        Soft-knee peak limiter to prevent digital clipping while preserving
        natural audio dynamics and transient punch.
        """
        arr = np.asarray(audio, dtype=np.float32)
        peak = float(np.max(np.abs(arr))) if len(arr) > 0 else 0.0
        if peak <= ceiling:
            return arr

        # Hyperbolic tangent soft compression above ceiling
        knee_start = ceiling * 0.85
        mask = np.abs(arr) > knee_start
        if not np.any(mask):
            return arr

        out = arr.copy()
        exceed = np.abs(arr[mask]) - knee_start
        headroom = ceiling - knee_start
        compressed = knee_start + headroom * np.tanh(exceed / max(1e-6, headroom))
        out[mask] = np.sign(arr[mask]) * compressed
        return out

    def smooth_boundary(self, frame: np.ndarray, last_sample: Optional[float], ramp_len: int = 16) -> np.ndarray:
        """
        Cross-fades frame boundaries to eliminate clicks and ticks.
        """
        if last_sample is None or len(frame) < ramp_len:
            return frame
        step = float(last_sample - frame[0])
        if abs(step) > 0.015:
            ramp = np.linspace(1.0, 0.0, ramp_len, dtype=frame.dtype)
            frame[:ramp_len] += step * ramp
        return frame

