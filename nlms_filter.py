"""
Stage 2 of IGARD-Net: classical NLMS adaptive filter.

Standard reference-based adaptive noise cancellation:
  primary_mic   = speech + noise
  reference_mic = noise only (roughly correlated with the noise in primary)

The filter learns to predict the noise component in `primary` from
`reference`, and subtracts it out. This is the well-known Widrow (1975)
adaptive noise cancelling structure, using the Normalized LMS update rule
for stability across varying signal power.
"""

import numpy as np


class NLMSFilter:
    def __init__(self, num_taps=64, step_size=0.5, eps=1e-6):
        """
        num_taps  : length of the adaptive FIR filter (longer = models more
                    complex noise, but slower to converge and more compute)
        step_size : mu, the learning rate (0 < mu <= 2 for stability).
                    This is exactly the parameter the quantum-inspired tuner
                    (quantum_tuner.py) adjusts in real time.
        eps       : small constant to avoid divide-by-zero when the
                    reference signal is silent
        """
        self.num_taps = num_taps
        self.step_size = step_size
        self.eps = eps
        self.weights = np.zeros(num_taps, dtype=np.float64)
        self._ref_buffer = np.zeros(num_taps, dtype=np.float64)

    def reset(self):
        self.weights[:] = 0.0
        self._ref_buffer[:] = 0.0

    def process_sample(self, primary_sample, reference_sample):
        """Process one audio sample. Returns (clean_sample, noise_estimate)."""
        # Shift the reference buffer and insert the newest sample
        self._ref_buffer[1:] = self._ref_buffer[:-1]
        self._ref_buffer[0] = reference_sample

        noise_estimate = np.dot(self.weights, self._ref_buffer)
        error = primary_sample - noise_estimate  # this is our "clean" output

        # Normalized LMS weight update
        norm = np.dot(self._ref_buffer, self._ref_buffer) + self.eps
        self.weights += (self.step_size / norm) * error * self._ref_buffer

        return error, noise_estimate

    def process_block(self, primary, reference):
        """
        Process a whole numpy array at once (used for offline testing).
        primary, reference: 1D numpy arrays of the same length.
        Returns: cleaned output array of the same length.
        """
        assert len(primary) == len(reference), "primary and reference must be the same length"
        out = np.zeros_like(primary, dtype=np.float64)
        for i in range(len(primary)):
            out[i], _ = self.process_sample(primary[i], reference[i])
        return out

    def set_step_size(self, new_step_size):
        """Called live by the quantum-inspired tuner (see quantum_tuner.py)."""
        self.step_size = float(np.clip(new_step_size, 1e-4, 2.0))
