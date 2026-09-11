"""
Stage 2: 64-Tap NLMS Adaptive Filter.
Implements Normalized Least-Mean-Squares FIR filter with leakage and dynamic step size.
Integrates with ComputeBackend for accelerated CPU/CUDA execution.
"""

from typing import Tuple, Optional
import numpy as np
from backends.base import ComputeBackend
from backends.cpu_backend import CPUBackend


class NLMSFilter:
    def __init__(
        self,
        num_taps: int = 64,
        initial_mu: float = 0.25,
        eps: float = 1.0e-6,
        leakage: float = 0.9999,
        adaptation_enabled: bool = True,
        backend: Optional[ComputeBackend] = None,
    ):
        self.num_taps = num_taps
        self.step_size = initial_mu
        self.eps = eps
        self.leakage = leakage
        self.adaptation_enabled = adaptation_enabled
        self.backend = backend if backend is not None else CPUBackend()

        self.weights = np.zeros(num_taps, dtype=np.float64)
        self.ref_buffer = np.zeros(num_taps, dtype=np.float64)

    def reset(self):
        self.weights.fill(0.0)
        self.ref_buffer.fill(0.0)

    def set_step_size(self, new_mu: float):
        """Dynamically updated by the GQPSO tuner or autopilot."""
        self.step_size = float(np.clip(new_mu, 0.001, 1.8))

    def process_block(
        self,
        primary_block: np.ndarray,
        reference_block: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Processes a block of audio.
        Returns: (error_output, noise_estimate)
        """
        mu_effective = self.step_size if self.adaptation_enabled else 0.0
        
        err, noise_est, new_w, new_ref = self.backend.nlms_process_block(
            weights=self.weights,
            ref_buffer=self.ref_buffer,
            primary_block=primary_block,
            ref_block=reference_block,
            mu=mu_effective,
            eps=self.eps,
            leakage=self.leakage,
        )

        self.weights = new_w
        self.ref_buffer = new_ref
        return err, noise_est
