import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import unittest
import numpy as np
from processing.mos_estimator import MOSEstimator


class TestMOSEstimator(unittest.TestCase):
    def setUp(self):
        self.estimator = MOSEstimator(sample_rate=16000)

    def test_clean_speech_high_mos(self):
        t = np.arange(16000) / 16000.0
        # Clean speech harmonic simulation
        clean = (
            0.5 * np.sin(2 * np.pi * 300 * t) +
            0.3 * np.sin(2 * np.pi * 600 * t) +
            0.2 * np.sin(2 * np.pi * 1200 * t)
        )
        res = self.estimator.evaluate_signals(clean, clean)
        self.assertGreaterEqual(res["overall_mos"], 3.8)
        self.assertGreaterEqual(res["speech_preservation_score"], 95.0)
        self.assertIn(res["mos_rating"], ["Good", "Excellent"])

    def test_noisy_input_vs_enhanced(self):
        t = np.arange(16000) / 16000.0
        voice = 0.5 * np.sin(2 * np.pi * 400 * t)
        voice[8000:] = 0.0  # Realistic speech utterance with pause
        noise = np.random.randn(16000) * 0.15
        raw = voice + noise
        enhanced = voice + noise * 0.05  # 26 dB noise reduction

        res = self.estimator.evaluate_signals(raw, enhanced)
        self.assertGreater(res["overall_mos"], res["raw_mos"])
        self.assertGreater(res["mos_gain"], 0.5)
        self.assertGreater(res["snr_improvement_db"], 10.0)

    def test_frame_mos_estimator(self):
        prim = np.random.randn(256) * 0.2
        enh = np.random.randn(256) * 0.05
        mos = self.estimator.estimate_frame_mos(prim, enh, speech_prob=0.8, snr_delta=15.0)
        self.assertGreaterEqual(mos, 1.0)
        self.assertLessEqual(mos, 5.0)


if __name__ == "__main__":
    unittest.main()
