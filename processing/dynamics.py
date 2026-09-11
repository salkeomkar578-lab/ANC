"""
Stage 6: Voice Protection & Dynamics Safeguards.
Protects human speech intelligibility using:
  - Speech-preservation gain floor (prevents speech vanishing)
  - Soft-knee dynamic range compressor
  - Fast peak limiter (prevents harsh digital clipping)
  - Smooth envelope tracking with attack and release
"""

import numpy as np


class VoiceProtectionDynamics:
    def __init__(
        self,
        sample_rate: int = 16000,
        gain_floor_db: float = -18.0,
        threshold_db: float = -6.0,
        ratio: float = 3.0,
        attack_ms: float = 5.0,
        release_ms: float = 40.0,
        makeup_gain_db: float = 1.5,
    ):
        self.sample_rate = sample_rate
        self.gain_floor_linear = 10.0 ** (gain_floor_db / 20.0)
        self.threshold_linear = 10.0 ** (threshold_db / 20.0)
        self.ratio = ratio
        self.makeup_linear = 10.0 ** (makeup_gain_db / 20.0)

        # Attack and release time constants
        self.alpha_attack = np.exp(-1.0 / (sample_rate * (attack_ms / 1000.0)))
        self.alpha_release = np.exp(-1.0 / (sample_rate * (release_ms / 1000.0)))
        
        self._envelope: float = 0.0

    def process(
        self,
        processed_block: np.ndarray,
        raw_primary_block: np.ndarray,
        speech_prob: float = 0.0,
    ) -> np.ndarray:
        """
        Applies speech floor protection (only when speech is active), compression,
        and peak limiting. When speech is absent, allows full background cancellation.
        """
        n = len(processed_block)
        if n == 0:
            return processed_block

        # 1. Speech Preservation Floor (Active ONLY when speech is present):
        # When speech is detected (speech_prob >= 0.20), ensure the output energy does
        # not drop too far below raw speech energy to protect human vowels & consonants.
        # When speech is absent, DO NOT boost background noise — let it stay cancelled!
        if speech_prob >= 0.20:
            raw_rms = float(np.sqrt(np.mean(raw_primary_block ** 2))) + 1.0e-12
            out_rms = float(np.sqrt(np.mean(processed_block ** 2))) + 1.0e-12
            attenuation = out_rms / raw_rms

            if attenuation < self.gain_floor_linear:
                boost_factor = self.gain_floor_linear / attenuation
                protected_block = processed_block * min(boost_factor, 2.5)
            else:
                protected_block = processed_block
        else:
            protected_block = processed_block

        # 2. Dynamic Compressor / Limiter
        out = np.empty_like(protected_block)
        env = self._envelope
        thresh = self.threshold_linear
        inv_ratio = 1.0 / self.ratio
        makeup = self.makeup_linear

        for i in range(n):
            val = abs(protected_block[i])
            if val > env:
                env = self.alpha_attack * env + (1.0 - self.alpha_attack) * val
            else:
                env = self.alpha_release * env + (1.0 - self.alpha_release) * val

            # Compression curve
            if env > thresh:
                # Gain reduction above threshold
                gain = (thresh + (env - thresh) * inv_ratio) / (env + 1.0e-10)
            else:
                gain = 1.0

            out[i] = protected_block[i] * gain * makeup

        self._envelope = env

        # 3. Transparent soft saturation peak limiter (prevents hard clipping)
        out = np.tanh(out)
        return out
