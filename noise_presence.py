"""
Stage 0 of IGARD-Net: Noise-Presence Gate.
Decides whether environmental noise is high enough to warrant active cancellation.
Includes stateful hysteresis to prevent frame-to-frame chattering and sudden bypass clicks.
"""

import numpy as np


class NoisePresenceGate:
    def __init__(
        self,
        threshold_margin: float = 3.0,
        calibration_blocks: int = 20,
        initial_floor: float = 1e-5,
        hysteresis_factor: float = 0.70,
    ):
        self.threshold_margin = threshold_margin
        self.calibration_blocks = calibration_blocks
        self.floor_estimate = initial_floor
        self.hysteresis_factor = hysteresis_factor
        
        self._calibration_samples = []
        self._calibrated = False
        self._current_state = True

    def check(self, reference_block):
        """
        Returns (noise_present: bool, floor_estimate: float, block_energy: float)
        """
        block_energy = float(np.mean(np.asarray(reference_block, dtype=np.float64) ** 2)) + 1e-12

        if not self._calibrated:
            self._calibration_samples.append(block_energy)
            if len(self._calibration_samples) >= self.calibration_blocks:
                self.floor_estimate = float(np.percentile(self._calibration_samples, 20))
                self._calibrated = True
            else:
                return True, self.floor_estimate, block_energy
        else:
            if block_energy < self.floor_estimate:
                self.floor_estimate = block_energy

        # Hysteresis comparison
        high_threshold = self.floor_estimate * self.threshold_margin
        low_threshold = high_threshold * self.hysteresis_factor

        if self._current_state:
            if block_energy < low_threshold:
                self._current_state = False
        else:
            if block_energy > high_threshold:
                self._current_state = True

        return self._current_state, self.floor_estimate, block_energy

    def recalibrate(self):
        self._calibration_samples = []
        self._calibrated = False
        self.floor_estimate = 1e-5
        self._current_state = True
