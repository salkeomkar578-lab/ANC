"""
Stage 2 of IGARD-Net: Speech-Preserving NLMS Adaptive Filter.
Includes Double-Talk Detection (DTD), speech leakage protection,
leakage regularization, and correlation-weighted adaptation.

Standard reference-based adaptive noise cancellation:
  primary_mic   = speech + noise
  reference_mic = noise only (environmental reference)

Crucial Safety Rule:
When speech is present, or when the reference channel is contaminated with speech,
adaptation is frozen or attenuated to prevent canceling the speaker's voice.
"""

from typing import Tuple, Optional
import numpy as np


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
    ):
        if initial_mu is not None:
            step_size = initial_mu
        self.num_taps = num_taps
        self.step_size = float(np.clip(step_size, 1e-4, max_mu))
        self.eps = eps
        self.leakage = leakage
        self.max_mu = max_mu
        self.adaptation_enabled = adaptation_enabled

        self.weights = np.zeros(num_taps, dtype=np.float64)
        self._ref_buffer = np.zeros(num_taps, dtype=np.float64)
        self._adaptation_frozen: bool = False

    def reset(self):
        self.weights.fill(0.0)
        self._ref_buffer.fill(0.0)
        self._adaptation_frozen = False

    def set_step_size(self, new_step_size: float):
        self.step_size = float(np.clip(new_step_size, 1e-4, self.max_mu))

    def process_sample(
        self,
        primary_sample: float,
        reference_sample: float,
        speech_prob: float = 0.0,
    ) -> Tuple[float, float]:
        """
        Process one audio sample with speech-aware adaptation control.
        Returns: (error_sample, noise_estimate_sample)
        """
        self._ref_buffer[1:] = self._ref_buffer[:-1]
        self._ref_buffer[0] = reference_sample

        noise_est = float(np.dot(self.weights, self._ref_buffer))
        error = float(primary_sample - noise_est)

        # Double-Talk / Speech Protection Rule:
        # Scale step size down as speech probability increases.
        # Freeze adaptation when speech is strong.
        if speech_prob > 0.65:
            eff_mu = 0.0
            self._adaptation_frozen = True
        elif speech_prob > 0.30:
            scale = (0.65 - speech_prob) / 0.35
            eff_mu = self.step_size * max(0.0, scale)
            self._adaptation_frozen = False
        else:
            eff_mu = self.step_size
            self._adaptation_frozen = False

        if eff_mu > 0.0:
            norm = float(np.dot(self._ref_buffer, self._ref_buffer)) + self.eps
            # Weight update with gentle leakage for stability
            self.weights = self.weights * self.leakage + (eff_mu / norm) * error * self._ref_buffer
            # Clamp weights to prevent divergence
            np.clip(self.weights, -2.0, 2.0, out=self.weights)

        return error, noise_est

    def process_block(
        self,
        primary: np.ndarray,
        reference: np.ndarray,
        speech_prob: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized block processing with reference quality check and DTD.
        Returns: (error_block, noise_estimate_block)
        """
        primary = np.asarray(primary, dtype=np.float64)
        reference = np.asarray(reference, dtype=np.float64)
        n = len(primary)
        assert len(reference) == n, "primary and reference must have the same length"

        if n == 0:
            return primary.copy(), np.zeros_like(primary)

        # 1. Reference Quality & Contamination Check
        prim_rms = float(np.sqrt(np.mean(primary ** 2))) + 1e-9
        ref_rms = float(np.sqrt(np.mean(reference ** 2))) + 1e-9

        # Cross-correlation between primary and reference
        if prim_rms > 0.005 and ref_rms > 0.005:
            cross_corr = float(np.abs(np.mean(primary * reference)) / (prim_rms * ref_rms))
        else:
            cross_corr = 0.0

        # If speech is detected AND reference has strong correlation with primary,
        # it is contaminated with speech leakage (e.g. stereo speech mic).
        # In this case, we MUST prevent NLMS from subtracting the voice!
        is_speech_leakage = (speech_prob > 0.40) and (cross_corr > 0.60)

        # Effective block step size
        if is_speech_leakage or speech_prob > 0.65:
            eff_mu = 0.0
            self._adaptation_frozen = True
        elif speech_prob > 0.25:
            scale = (0.65 - speech_prob) / 0.40
            eff_mu = self.step_size * scale
            self._adaptation_frozen = False
        else:
            eff_mu = self.step_size
            self._adaptation_frozen = False

        # Fast block execution
        out_err = np.empty(n, dtype=np.float64)
        out_noise = np.empty(n, dtype=np.float64)

        concat_ref = np.concatenate((self._ref_buffer, reference))
        w = self.weights.copy()
        num_taps = self.num_taps

        init_x = concat_ref[:num_taps]
        running_norm = float(np.dot(init_x, init_x)) + self.eps

        for k in range(n):
            x_k = concat_ref[k : k + num_taps][::-1]
            y_k = float(np.dot(w, x_k))
            
            # If reference is contaminated with speech, scale down subtraction
            if is_speech_leakage:
                y_k *= 0.2

            e_k = float(primary[k] - y_k)

            if eff_mu > 0.0:
                factor = (eff_mu / running_norm) * e_k
                w = w * self.leakage + factor * x_k

            out_err[k] = e_k
            out_noise[k] = y_k

            if k + num_taps < len(concat_ref):
                old_val = concat_ref[k]
                new_val = concat_ref[k + num_taps]
                running_norm = max(self.eps, running_norm - old_val * old_val + new_val * new_val)

        np.clip(w, -2.0, 2.0, out=w)
        self.weights = w
        self._ref_buffer = concat_ref[-num_taps:].copy()

        return out_err, out_noise
