"""
Stage 2: 64-Tap Normalized Least Mean Squares (NLMS) Adaptive Filter.
Includes Double-Talk Detection (DTD), speech leakage protection,
leakage regularization, and correlation-weighted adaptation.
"""

from typing import Tuple, Optional
import numpy as np
from backends.base import ComputeBackend
from backends.cpu_backend import CPUBackend


class NLMSFilter:
    def __init__(
        self,
        num_taps: int = 64,
        step_size: float = 0.2,
        initial_mu: Optional[float] = None,
        eps: float = 1e-6,
        leakage: float = 0.9995,
        max_mu: float = 0.5,
        adaptation_enabled: bool = True,
        backend: Optional[ComputeBackend] = None,
    ):
        if initial_mu is not None:
            step_size = initial_mu
        self.num_taps = num_taps
        self.step_size = float(np.clip(step_size, 1e-4, max_mu))
        self.eps = eps
        self.leakage = leakage
        self.max_mu = max_mu
        self.adaptation_enabled = adaptation_enabled
        self.backend = backend if backend is not None else CPUBackend()

        self.weights = np.zeros(num_taps, dtype=np.float64)
        self.ref_buffer = np.zeros(num_taps, dtype=np.float64)
        self.adaptation_frozen: bool = False

    def reset(self):
        self.weights.fill(0.0)
        self.ref_buffer.fill(0.0)
        self.adaptation_frozen = False

    def set_step_size(self, new_mu: float):
        self.step_size = float(np.clip(new_mu, 1e-4, self.max_mu))

    def process_block(
        self,
        primary_block: np.ndarray,
        ref_block: np.ndarray,
        speech_prob: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Processes block with speech-aware step size scaling and DTD.
        """
        primary = np.asarray(primary_block, dtype=np.float64)
        reference = np.asarray(ref_block, dtype=np.float64)
        n = len(primary)

        if n == 0:
            return primary.copy(), np.zeros_like(primary)

        prim_rms = float(np.sqrt(np.mean(primary ** 2))) + 1e-9
        ref_rms = float(np.sqrt(np.mean(reference ** 2))) + 1e-9

        if prim_rms > 0.005 and ref_rms > 0.005:
            cross_corr = float(np.abs(np.mean(primary * reference)) / (prim_rms * ref_rms))
        else:
            cross_corr = 0.0

        is_speech_leakage = (speech_prob > 0.40) and (cross_corr > 0.60)

        if is_speech_leakage or speech_prob > 0.65:
            eff_mu = 0.0
            self.adaptation_frozen = True
        elif speech_prob > 0.25:
            scale = (0.65 - speech_prob) / 0.40
            eff_mu = self.step_size * scale
            self.adaptation_frozen = False
        else:
            eff_mu = self.step_size
            self.adaptation_frozen = False

        # If reference has speech leakage, pass attenuated reference
        ref_input = reference * 0.2 if is_speech_leakage else reference

        err, noise_est, new_w, new_buf = self.backend.nlms_process_block(
            weights=self.weights,
            ref_buffer=self.ref_buffer,
            primary_block=primary,
            ref_block=ref_input,
            mu=eff_mu,
            eps=self.eps,
            leakage=self.leakage,
        )

        np.clip(new_w, -2.0, 2.0, out=new_w)
        self.weights = new_w
        self.ref_buffer = new_buf

        return err, noise_est
