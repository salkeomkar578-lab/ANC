"""
Stage 0: Noise Presence Gate.
Evaluates reference mic energy to bypass downstream processing during quiet conditions.
Prevents speech distortion and saves compute on quiet audio.
"""

from typing import Tuple
import numpy as np


class NoisePresenceGate:
    def __init__(self, threshold_margin: float = 2.5, calibration_blocks: int = 20, initial_floor: float = 1.0e-5, hysteresis_db: float = 2.0):
        self.threshold_margin = threshold_margin
        self.calibration_blocks = calibration_blocks
        self.floor_estimate = initial_floor
        self.hysteresis_linear = 10.0 ** (hysteresis_db / 10.0)
        self._calibration_samples = []
        self._calibrated = False
        self._last_state = False

    def check(self, reference_block: np.ndarray) -> Tuple[bool, float, float]:
        """
        Returns (noise_present: bool, floor_estimate: float, block_energy: float)
        """
        block_energy = float(np.mean(reference_block ** 2)) + 1.0e-12

        if not self._calibrated:
            self._calibration_samples.append(block_energy)
            if len(self._calibration_samples) >= self.calibration_blocks:
                self.floor_estimate = float(np.percentile(self._calibration_samples, 20))
                self._calibrated = True
            else:
                # Startup warm-up: safely assume noise present
                self._last_state = True
                return True, self.floor_estimate, block_energy
        else:
            # Floor can adapt downward to track quieter environments, but cannot artificially rise on loud sustained noise
            if block_energy < self.floor_estimate:
                self.floor_estimate = 0.9 * self.floor_estimate + 0.1 * block_energy

        # Apply hysteresis to avoid chattering
        thresh_on = self.floor_estimate * self.threshold_margin
        thresh_off = thresh_on / self.hysteresis_linear

        if self._last_state:
            noise_present = block_energy > thresh_off
        else:
            noise_present = block_energy > thresh_on

        self._last_state = noise_present
        return noise_present, self.floor_estimate, block_energy

    def recalibrate(self):
        """Operator action: reset baseline calibration."""
        self._calibration_samples.clear()
        self._calibrated = False
        self.floor_estimate = 1.0e-5
        self._last_state = False
