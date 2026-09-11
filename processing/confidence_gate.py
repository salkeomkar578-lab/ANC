"""
Stage 4: Speech-First Confidence Gate.
Evaluates classifier reliability and determines whether post-NLMS spectral cleanup
should execute, or safely fail-safe to speech preservation bypass.
"""


class ConfidenceGate:
    def __init__(self, threshold: float = 0.60):
        self.threshold = threshold

    def decide(self, confidence: float, speech_prob: float = 0.0, autopilot: bool = True) -> bool:
        if not autopilot:
            return True
        if speech_prob > 0.60 and confidence < self.threshold:
            return False
        return confidence >= self.threshold

    def get_aggressiveness(self, confidence: float, speech_prob: float = 0.0) -> float:
        if confidence < self.threshold:
            return 0.3 * (1.0 - speech_prob)
        return float(max(0.2, (confidence - self.threshold) / (1.0 - self.threshold + 1e-6)) * (1.0 - 0.7 * speech_prob))
