import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import unittest
import numpy as np

from processing.presence_gate import NoisePresenceGate
from processing.noise_classifier import NoiseClassifier
from processing.nlms_filter import NLMSFilter
from processing.tuner import QuantumInspiredTuner
from processing.confidence_gate import ConfidenceGate
from processing.wiener_cleanup import ResidualCleanup
from processing.dynamics import VoiceProtectionDynamics
from sensors.mock_sensor import MockAccelerometer
from sensors.accelerometer import cross_check_confidence
from backends.cpu_backend import CPUBackend


class TestPipelineUnits(unittest.TestCase):
    def setUp(self):
        self.backend = CPUBackend()
        self.sample_rate = 16000

    def test_stage0_presence_gate(self):
        gate = NoisePresenceGate(threshold_margin=2.5, calibration_blocks=5)
        # Feed 5 quiet blocks
        quiet = np.random.randn(256) * 0.001
        for _ in range(5):
            present, floor, _ = gate.check(quiet)
        
        self.assertTrue(gate._calibrated)
        # Check quiet block bypass
        present, _, _ = gate.check(quiet)
        self.assertFalse(present)

        # Check loud noise engages gate
        loud = np.random.randn(256) * 0.2
        present, _, _ = gate.check(loud)
        self.assertTrue(present)

        # Check recalibrate
        gate.recalibrate()
        self.assertFalse(gate._calibrated)

    def test_stage1_noise_classifier(self):
        clf = NoiseClassifier(sample_rate=self.sample_rate)
        
        # Test steady tonal noise (should classify as steady)
        t = np.arange(512) / self.sample_rate
        tonal = np.sin(2 * np.pi * 120 * t) * 0.5 + np.sin(2 * np.pi * 240 * t) * 0.3
        label, conf = clf.classify(tonal)
        self.assertIn(label, ["steady", "unclassified"])
        self.assertGreaterEqual(conf, 0.3)

        # Test impulsive transient burst
        burst = np.zeros(512)
        burst[10:30] = np.random.randn(20) * 2.0
        label, conf = clf.classify(burst)
        self.assertIn(label, ["impulsive", "unclassified"])

    def test_stage1b_accelerometer_cross_check(self):
        accel = MockAccelerometer()
        
        # Case 1: Impulsive + High Shock -> Boosted Confidence
        accel.trigger_shock(0.95)
        shock = accel.read_recent_shock_score()
        conf_in = 0.65
        conf_out = cross_check_confidence("impulsive", conf_in, shock, agreement_boost=0.25)
        self.assertGreater(conf_out, conf_in)

        # Case 2: Impulsive + No Shock -> Penalized Confidence
        shock_none = accel.read_recent_shock_score()
        conf_penalized = cross_check_confidence("impulsive", conf_in, shock_none)
        self.assertLess(conf_penalized, conf_in)

        # Case 3: Steady Noise -> Untouched by accelerometer
        steady_conf = 0.80
        steady_out = cross_check_confidence("steady", steady_conf, shock)
        self.assertEqual(steady_out, steady_conf)

    def test_stage2_nlms_filter(self):
        nlms = NLMSFilter(num_taps=64, initial_mu=0.25, backend=self.backend)
        
        # Correlated noise cancellation test
        t = np.arange(1024) / self.sample_rate
        noise = np.sin(2 * np.pi * 200 * t) * 0.5
        ref = noise + np.random.randn(1024) * 0.01
        prim = noise # only noise
        
        err, _ = nlms.process_block(prim[:512], ref[:512])
        err2, _ = nlms.process_block(prim[512:], ref[512:])
        
        # Residual error should decrease as filter converges
        e1 = np.mean(err ** 2)
        e2 = np.mean(err2 ** 2)
        self.assertLess(e2, e1)

    def test_stage3_tuner_fitness(self):
        tuner = QuantumInspiredTuner(num_particles=4, iterations=3, backend=self.backend)
        prim = np.random.randn(256) * 0.1
        ref = np.random.randn(256) * 0.1
        best_mu = tuner.optimize_step(prim, ref, num_taps=32)
        self.assertGreaterEqual(best_mu, tuner.search_min)
        self.assertLessEqual(best_mu, tuner.search_max)

    def test_stage4_confidence_gate(self):
        gate = ConfidenceGate(threshold=0.60)
        self.assertTrue(gate.decide(0.85, autopilot=True))
        self.assertFalse(gate.decide(0.40, autopilot=True))
        self.assertTrue(gate.decide(0.40, autopilot=False)) # Autopilot off allows bypass override

    def test_stage5_wiener_ola_continuity(self):
        cleanup = ResidualCleanup(sample_rate=self.sample_rate, backend=self.backend)
        noise = np.random.randn(256) * 0.05
        cleanup.update_noise_profile(noise)
        
        block1 = np.sin(np.linspace(0, 10, 256))
        block2 = np.sin(np.linspace(10, 20, 256))
        
        out1 = cleanup.clean(block1)
        out2 = cleanup.clean(block2)
        
        self.assertEqual(len(out1), 256)
        self.assertEqual(len(out2), 256)
        self.assertFalse(np.any(np.isnan(out1)))
        self.assertFalse(np.any(np.isnan(out2)))

    def test_stage6_voice_protection_dynamics(self):
        dyn = VoiceProtectionDynamics(sample_rate=self.sample_rate, gain_floor_db=-18.0)
        raw = np.sin(np.linspace(0, 10, 256)) * 0.8
        # Simulate heavily attenuated block (e.g. oversubtraction)
        over_attenuated = raw * 0.01
        
        protected = dyn.process(over_attenuated, raw)
        # Output should be lifted to satisfy gain floor
        self.assertGreater(np.max(np.abs(protected)), np.max(np.abs(over_attenuated)))
        # Output must not clip
        self.assertLessEqual(np.max(np.abs(protected)), 1.0)


if __name__ == "__main__":
    unittest.main()
