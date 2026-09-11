"""
Stage 1: Noise Classifier.
Fast rule-based acoustic classifier extracting 8 spectral and temporal features:
  - Energy
  - Zero-crossing rate
  - Spectral centroid
  - Spectral peakiness
  - Crest factor
  - Spectral rolloff
  - Spectral bandwidth
  - Temporal onset strength
Distinguishes between 'impulsive' (gunshot/artillery), 'steady' (engine/rotor), and 'unclassified'.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from speech_detector import SpeechDetector


class NoiseClassifier:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._window_cache: Dict[int, np.ndarray] = {}
        self.speech_detector = SpeechDetector(sample_rate=sample_rate)

    def _get_window(self, n: int) -> np.ndarray:
        if n not in self._window_cache:
            self._window_cache[n] = np.hanning(n)
        return self._window_cache[n]

    def extract_features(self, block: np.ndarray) -> Optional[Dict[str, float]]:
        n = len(block)
        if n == 0:
            return None

        energy = float(np.mean(block ** 2))
        zero_crossings = float(np.mean(np.abs(np.diff(np.sign(block))) > 0))

        window = self._get_window(n)
        spectrum = np.abs(np.fft.rfft(block * window))
        freqs = np.fft.rfftfreq(n, d=1.0 / self.sample_rate)
        spectrum_sum = float(np.sum(spectrum)) + 1.0e-12
        spectral_centroid = float(np.sum(freqs * spectrum) / spectrum_sum)

        power = spectrum ** 2
        total_energy = float(np.sum(power)) + 1.0e-12
        
        # Spectral peakiness using partition instead of full sort for speed
        top_k = max(1, int(len(power) * 0.05))
        if len(power) > top_k:
            partitioned = np.partition(power, -top_k)
            top_energy = float(np.sum(partitioned[-top_k:]))
        else:
            top_energy = total_energy
        spectral_peakiness = top_energy / total_energy

        peak = float(np.max(np.abs(block))) + 1.0e-12
        rms = float(np.sqrt(energy)) + 1.0e-12
        crest_factor = peak / rms

        # Spectral rolloff (85% energy threshold)
        cumulative_energy = np.cumsum(power)
        rolloff_threshold = 0.85 * cumulative_energy[-1]
        rolloff_idx = int(np.searchsorted(cumulative_energy, rolloff_threshold))
        rolloff_idx = min(rolloff_idx, len(freqs) - 1)
        spectral_rolloff = float(freqs[rolloff_idx])

        # Spectral bandwidth
        spectral_bandwidth = float(
            np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * spectrum) / spectrum_sum)
        )

        # Temporal onset strength across sub-frames
        sub_len = max(1, n // 8)
        sub_energies = [float(np.mean(block[i : i + sub_len] ** 2)) for i in range(0, n - sub_len + 1, sub_len)]
        if len(sub_energies) >= 2:
            onset_strength = float(np.max(np.abs(np.diff(sub_energies))))
        else:
            onset_strength = 0.0

        return {
            "energy": energy,
            "zero_crossings": zero_crossings,
            "spectral_centroid": spectral_centroid,
            "spectral_peakiness": spectral_peakiness,
            "crest_factor": crest_factor,
            "spectral_rolloff": spectral_rolloff,
            "spectral_bandwidth": spectral_bandwidth,
            "onset_strength": onset_strength,
        }

    def classify(self, block: np.ndarray) -> Tuple[str, float]:
        feats = self.extract_features(block)
        if feats is None:
            return "unclassified", 0.0

        # Impulsive check: sharp peak, high crest factor, broadband, strong onset
        impulsive_score = 0.0
        if feats["crest_factor"] > 3.0:
            impulsive_score += 0.30
        if feats["spectral_peakiness"] < 0.55:
            impulsive_score += 0.20
        if feats["energy"] > 0.004:
            impulsive_score += 0.10
        if feats["spectral_rolloff"] > self.sample_rate * 0.30:
            impulsive_score += 0.15
        if feats["spectral_bandwidth"] > self.sample_rate * 0.15:
            impulsive_score += 0.15
        if feats["onset_strength"] > 0.01:
            impulsive_score += 0.10

        if impulsive_score >= 0.50:
            confidence = float(np.clip(impulsive_score, 0.50, 0.97))
            return "impulsive", confidence

        # Steady check: tonal, energy concentrated in peaks, low rolloff
        steady_score = 0.0
        if feats["spectral_peakiness"] > 0.80:
            steady_score += 0.35
        if feats["energy"] > 0.001:
            steady_score += 0.15
        if feats["spectral_rolloff"] < self.sample_rate * 0.25:
            steady_score += 0.20
        if feats["spectral_bandwidth"] < self.sample_rate * 0.12:
            steady_score += 0.20
        if feats["onset_strength"] < 0.005:
            steady_score += 0.10

        if steady_score >= 0.50:
            confidence = float(np.clip(steady_score, 0.50, 0.95))
            return "steady", confidence

        return "unclassified", 0.35

    def classify_with_probabilities(self, block: np.ndarray) -> Tuple[str, float, float, float]:
        label, confidence = self.classify(block)
        speech_prob, noise_prob, _ = self.speech_detector.detect(block)
        return label, confidence, speech_prob, noise_prob
