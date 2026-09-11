"""
Stage 1 of IGARD-Net: noise classifier.

HONESTY NOTE FOR THE TEAM:
A trained CNN needs labeled defense-noise audio (gunshot/rotor/vehicle/
artillery clips) which you don't have yet. Rather than fake a "trained"
model with random weights, this ships a working RULE-BASED classifier
using standard audio features (energy, zero-crossing rate, spectral
centroid, spectral flatness, crest factor, spectral rolloff, spectral
bandwidth, onset strength). It runs today, with no training data, and
gives you a real confidence score to feed into the confidence gate.

ENHANCED (Master Build): added spectral rolloff, spectral bandwidth,
and temporal onset detection to the feature set — these give meaningfully
better discrimination between impulsive (gunfire/artillery) and steady
(rotor/engine) noise than the original 5-feature set, while staying
100% classical DSP (numpy-only, no ML framework, Pi 4 compatible).

UPGRADE PATH (do this once you have recorded/collected noise clips):
  1. Record labeled clips per class (impulsive / steady / speech-like)
  2. Extract the same features (or mel-spectrograms) from each clip
  3. Train a small CNN or even a simple sklearn classifier on those features
  4. Replace `_rule_based_classify()` with `self.model.predict(features)`
  Everything else in the pipeline (confidence gate, fail-safe) stays
  identical -- this file is the only one that needs to change.
"""

import numpy as np

from speech_detector import SpeechDetector

LABELS = ["impulsive", "steady", "unclassified"]


class NoiseClassifier:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.model = None  # populate this once a trained model exists
        self.speech_detector = SpeechDetector(sample_rate=sample_rate)

    def extract_features(self, block):
        block = np.asarray(block, dtype=np.float64)
        n = len(block)
        if n == 0:
            return None

        energy = float(np.mean(block ** 2))
        zero_crossings = float(np.mean(np.abs(np.diff(np.sign(block))) > 0))

        spectrum = np.abs(np.fft.rfft(block * np.hanning(n)))
        freqs = np.fft.rfftfreq(n, d=1.0 / self.sample_rate)
        spectrum_sum = np.sum(spectrum) + 1e-12
        spectral_centroid = float(np.sum(freqs * spectrum) / spectrum_sum)

        # Spectral PEAKINESS (not flatness): fraction of total spectral
        # ENERGY held in the top 5% of frequency bins. This is deliberately
        # used instead of the classic geometric-mean spectral flatness
        # measure, which turned out to be badly unstable in testing: a
        # pure tone has almost all bins near zero, so flatness is a ratio
        # dominated by a geometric mean of near-zero numbers -- adding
        # even a tiny, realistic sensor noise floor swung it from ~0.00001
        # to ~0.22 in testing, which would make that threshold useless on
        # any real microphone. Peakiness (an energy-sum ratio) stays
        # stable under a small added noise floor because it's dominated
        # by the arithmetic energy of the few tall peaks, not a geometric
        # mean.
        power = spectrum ** 2
        top_k = max(1, int(len(power) * 0.05))
        top_energy = float(np.sum(np.sort(power)[-top_k:]))
        total_energy = float(np.sum(power)) + 1e-12
        spectral_peakiness = top_energy / total_energy

        peak = float(np.max(np.abs(block)) + 1e-12)
        rms = float(np.sqrt(energy) + 1e-12)
        crest_factor = peak / rms

        # --- ENHANCED FEATURES (Master Build) ---

        # Spectral rolloff: frequency below which 85% of spectral energy
        # is contained. Impulsive sounds have high rolloff (broadband),
        # steady tonal sounds have low rolloff (energy concentrated low).
        cumulative_energy = np.cumsum(power)
        rolloff_threshold = 0.85 * cumulative_energy[-1]
        rolloff_idx = int(np.searchsorted(cumulative_energy, rolloff_threshold))
        rolloff_idx = min(rolloff_idx, len(freqs) - 1)
        spectral_rolloff = float(freqs[rolloff_idx])

        # Spectral bandwidth: weighted standard deviation of frequencies
        # around the centroid. Broad = impulsive, narrow = tonal steady.
        spectral_bandwidth = float(
            np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * spectrum) / spectrum_sum)
        )

        # Temporal onset strength: max absolute difference between
        # consecutive short sub-frames. High onset = sudden transient
        # (gunshot), low onset = sustained (rotor hum).
        sub_frame_len = max(1, n // 8)
        sub_energies = []
        for i in range(0, n - sub_frame_len + 1, sub_frame_len):
            sub = block[i:i + sub_frame_len]
            sub_energies.append(float(np.mean(sub ** 2)))
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

    def _rule_based_classify(self, feats):
        """
        Thresholds below were picked by testing against synthetic tonal
        (steady) and broadband-burst (impulsive) signals -- see
        test_synthetic.py. They are a starting configuration, not a
        final calibration: re-check them once you have real recordings.

        ENHANCED (Master Build): uses 3 additional features (spectral
        rolloff, bandwidth, onset strength) for better discrimination.
        The logic remains rule-based (no trained model) but is more
        robust against edge cases.
        """
        if feats is None:
            return "unclassified", 0.0

        # Impulsive (gunshot/artillery-like): short high-energy burst,
        # broadband (LOW spectral peakiness -- energy spread across many
        # bins, not concentrated), high crest factor (a sharp peak
        # relative to the average level), high spectral rolloff and
        # bandwidth (energy extends to high frequencies), strong onset.
        impulsive_score = 0.0
        if feats["crest_factor"] > 3.0:
            impulsive_score += 0.3
        if feats["spectral_peakiness"] < 0.5:
            impulsive_score += 0.2
        if feats["energy"] > 0.005:
            impulsive_score += 0.1
        if feats["spectral_rolloff"] > self.sample_rate * 0.3:
            impulsive_score += 0.15
        if feats["spectral_bandwidth"] > self.sample_rate * 0.15:
            impulsive_score += 0.15
        if feats["onset_strength"] > 0.01:
            impulsive_score += 0.1

        if impulsive_score >= 0.5:
            confidence = float(np.clip(impulsive_score, 0.5, 0.97))
            return "impulsive", confidence

        # Steady (rotor/vehicle/engine-like): tonal, energy concentrated
        # in a few frequency bins (HIGH spectral peakiness), sustained,
        # low spectral rolloff and bandwidth, low onset strength.
        steady_score = 0.0
        if feats["spectral_peakiness"] > 0.85:
            steady_score += 0.35
        if feats["energy"] > 0.001:
            steady_score += 0.15
        if feats["spectral_rolloff"] < self.sample_rate * 0.25:
            steady_score += 0.2
        if feats["spectral_bandwidth"] < self.sample_rate * 0.1:
            steady_score += 0.2
        if feats["onset_strength"] < 0.005:
            steady_score += 0.1

        if steady_score >= 0.5:
            confidence = float(np.clip(steady_score, 0.5, 0.95))
            return "steady", confidence

        return "unclassified", 0.3

    def classify(self, block):
        """Returns (label, confidence) where confidence is in [0, 1]."""
        if self.model is not None:
            raise NotImplementedError(
                "A trained model was assigned but no inference path is wired up yet. "
                "See the upgrade path note at the top of this file."
            )
        feats = self.extract_features(block)
        return self._rule_based_classify(feats)

    def classify_with_probabilities(self, block):
        """
        Continuous decision model returning:
        (label, confidence, speech_probability, noise_probability)
        """
        label, confidence = self.classify(block)
        speech_prob, noise_prob, _ = self.speech_detector.detect(block)
        return label, confidence, speech_prob, noise_prob
