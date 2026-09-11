"""
Unit and integration tests for IGARD-Net Dashboard Telemetry & Infrastructure.
Verifies:
1. compute_fft_magnitudes, compute_rms_level, compute_snr_estimate in spectrogram_helper.py
2. NoisePresenceGate recalibrate() functionality
3. IgardNetPipeline process_block() extended telemetry payload
4. ResidualCleanup Wiener soft-masking output validity
5. Dashboard Flask endpoints (/api/status, /api/recalibrate, /api/devices)
"""

import unittest
import numpy as np
from spectrogram_helper import compute_fft_magnitudes, compute_rms_level, compute_snr_estimate
from noise_presence import NoisePresenceGate
from pipeline import IgardNetPipeline
from cleanup_stage import ResidualCleanup
from dashboard.app import app


class TestDashboardInfrastructure(unittest.TestCase):
    def test_spectrogram_helper_fft(self):
        sr = 16000
        t = np.arange(512) / sr
        sig = np.sin(2 * np.pi * 1000 * t)  # 1 kHz tone
        res = compute_fft_magnitudes(sig, sample_rate=sr, max_bins=64)

        self.assertIn("freqs", res)
        self.assertIn("magnitudes", res)
        self.assertIn("peak_freq", res)
        self.assertIn("peak_magnitude", res)
        self.assertLessEqual(len(res["freqs"]), 64)
        self.assertLessEqual(len(res["magnitudes"]), 64)
        # Peak frequency should be near 1000 Hz
        self.assertAlmostEqual(res["peak_freq"], 1000, delta=250)

    def test_spectrogram_helper_rms_and_snr(self):
        silence = np.zeros(512)
        loud = np.ones(512) * 0.5
        self.assertEqual(compute_rms_level(silence), 0.0)
        self.assertGreater(compute_rms_level(loud), 0.5)

        # SNR estimate
        clean = np.sin(np.linspace(0, 10, 512))
        noisy = clean + 0.1 * np.random.randn(512)
        snr = compute_snr_estimate(clean, noisy)
        self.assertIsInstance(snr, float)

    def test_noise_presence_recalibration(self):
        gate = NoisePresenceGate(calibration_blocks=5, threshold_margin=2.0)
        # Feed high energy during startup
        high_energy = np.ones(512) * 0.8
        for _ in range(5):
            gate.check(high_energy)
        self.assertTrue(gate._calibrated)
        old_floor = gate.floor_estimate

        # Recalibrate
        gate.recalibrate()
        self.assertFalse(gate._calibrated)
        self.assertEqual(len(gate._calibration_samples), 0)

        # Now feed quiet blocks to re-baseline
        quiet = np.ones(512) * 0.001
        for _ in range(5):
            gate.check(quiet)
        self.assertTrue(gate._calibrated)
        self.assertLess(gate.floor_estimate, old_floor)

    def test_pipeline_extended_telemetry(self):
        pipe = IgardNetPipeline(sample_rate=16000)
        primary = np.random.randn(512) * 0.1
        ref = np.random.randn(512) * 0.1

        res = pipe.process_block(primary, ref)
        required_keys = [
            "audio", "noise_label", "confidence", "shock_score",
            "used_cleanup_stage", "nlms_step_size", "noise_present",
            "noise_floor_estimate", "active_stage", "stage_status",
            "primary_level", "reference_level", "primary_fft",
            "output_fft", "primary_samples", "output_samples",
            "estimated_snr", "block_count"
        ]
        for k in required_keys:
            self.assertIn(k, res, f"Missing telemetry key: {k}")

        self.assertIsInstance(res["primary_samples"], list)
        self.assertIsInstance(res["output_samples"], list)
        self.assertIsInstance(res["primary_fft"], list)
        self.assertIsInstance(res["output_fft"], list)

    def test_wiener_cleanup_stage(self):
        cleanup = ResidualCleanup(sample_rate=16000)
        noise = np.random.randn(512) * 0.05
        cleanup.update_noise_profile(noise)
        block = np.sin(np.linspace(0, 20, 512)) + noise
        out = cleanup.clean(block)
        self.assertEqual(len(out), 512)
        self.assertFalse(np.any(np.isnan(out)))

    def test_flask_endpoints(self):
        client = app.test_client()
        index_res = client.get("/")
        self.assertEqual(index_res.status_code, 200)
        self.assertIn(b"IGARD-Net", index_res.data)
        self.assertIn(b"Live Telemetry", index_res.data)

        status_res = client.get("/api/status")
        self.assertEqual(status_res.status_code, 200)
        data = status_res.get_json()
        self.assertIn("mode", data)
        self.assertIn("stage_status", data)

        recal_res = client.post("/api/recalibrate")
        self.assertEqual(recal_res.status_code, 200)
        r_data = recal_res.get_json()
        self.assertEqual(r_data["status"], "ok")

        dev_res = client.get("/api/devices")
        self.assertEqual(dev_res.status_code, 200)
        d_data = dev_res.get_json()
        self.assertIn("devices", d_data)


if __name__ == "__main__":
    unittest.main()
