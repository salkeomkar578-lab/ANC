"""
Accelerometer Sensor Interface and Cross-Modal Confidence Fusion.
Combines acoustic gunfire/shock classification with physical mechanical shock evidence.
"""

from abc import ABC, abstractmethod
import numpy as np


class AccelerometerBase(ABC):
    @abstractmethod
    def read_recent_shock_score(self) -> float:
        """
        Returns shock score in [0.0, 1.0].
        0.0 = no mechanical shock detected.
        1.0 = high-confidence mechanical impact/recoil.
        """
        pass


def cross_check_confidence(
    acoustic_label: str,
    acoustic_confidence: float,
    shock_score: float,
    agreement_boost: float = 0.25,
    shock_threshold: float = 0.5,
) -> float:
    """
    Cross-modal confirmation:
    - If acoustic classifier detects 'impulsive' AND accelerometer confirms shock:
      Boosts confidence (two independent physical modalities agree).
    - If acoustic classifier detects 'impulsive' BUT accelerometer sees NO shock:
      Penalizes confidence (likely acoustic false positive: fireworks, loud shout, backfire).
    - Other labels ('steady', 'quiet') remain unaffected by accelerometer.
    """
    if acoustic_label != "impulsive":
        return acoustic_confidence

    if shock_score >= shock_threshold:
        boosted = acoustic_confidence + agreement_boost * (1.0 - acoustic_confidence)
        return float(np.clip(boosted, 0.0, 0.99))
    else:
        # Penalize acoustic-only false alarm
        penalized = acoustic_confidence * 0.5
        return float(np.clip(penalized, 0.0, 1.0))
