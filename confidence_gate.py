"""
Stage 4 of IGARD-Net: Speech-First Confidence Gate.
Enforces the fail-safe policy:
When classification is uncertain, prioritize speech preservation and prevent
aggressive cleanup.
"""


class ConfidenceGate:
    def __init__(self, threshold: float = 0.60):
        self.threshold = threshold

    def decide(self, confidence: float, speech_prob: float = 0.0, autopilot: bool = True) -> bool:
        """
        Returns True if cleanup should engage; False = fail-safe bypass.
        When autopilot is False, manual override forces cleanup on.
        When speech is active and classifier is uncertain, failsafe to bypass.
        """
        if not autopilot:
            return True
        if speech_prob > 0.60 and confidence < self.threshold:
            return False
        return confidence >= self.threshold

    def get_aggressiveness(self, confidence: float, speech_prob: float = 0.0) -> float:
        """
        Calculates a continuous scalar [0.0 to 1.0] for cleanup aggressiveness.
        Safely attenuates suppression when speech is active or classification is uncertain.
        """
        if confidence < self.threshold:
            # Uncertain -> safe conservative processing
            return 0.3 * (1.0 - speech_prob)
        
        # Confident -> allow controlled cleanup scaled down during speech
        return float(max(0.2, (confidence - self.threshold) / (1.0 - self.threshold + 1e-6)) * (1.0 - 0.7 * speech_prob))
