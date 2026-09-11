"""
Stage 4 gate + Stage 5 fail-safe of IGARD-Net.

This is deliberately the simplest file in the project -- and that is the
point. The single most defensible engineering decision in the whole
system is a plain threshold check: if the classifier isn't confident,
skip the neural/DSP cleanup entirely and pass the NLMS output straight
through, unmodified. No guessing.
"""


class ConfidenceGate:
    def __init__(self, threshold=0.6):
        self.threshold = threshold

    def decide(self, confidence):
        """Returns True if the cleanup stage should run, False = fail-safe bypass."""
        return confidence >= self.threshold
