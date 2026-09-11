"""
Stage 4: Confidence Gate.
Decides whether to engage spectral cleanup or use fail-safe bypass.
Prevents speech destruction when acoustic classification confidence is low.
"""


class ConfidenceGate:
    def __init__(self, threshold: float = 0.60):
        self.threshold = threshold

    def decide(self, confidence: float, autopilot: bool = True) -> bool:
        """
        Returns True if cleanup should run, False for fail-safe bypass.
        """
        if not autopilot:
            return True
        return confidence >= self.threshold

    def set_threshold(self, new_threshold: float):
        self.threshold = float(max(0.1, min(0.95, new_threshold)))
