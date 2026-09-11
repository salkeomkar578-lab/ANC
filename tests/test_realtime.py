import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import unittest
import time
import numpy as np

from core.pipeline import IgardNetPipeline
from core.state import SystemState
from core.telemetry import TelemetryDispatcher
from audio.audio_output import AudioOutputSink
from audio.realtime_engine import RealtimeAudioEngine
from backends.cpu_backend import CPUBackend


class TestRealtimeStreaming(unittest.TestCase):
    def setUp(self):
        self.state = SystemState()
        self.backend = CPUBackend()
        self.pipeline = IgardNetPipeline(state=self.state, backend=self.backend)
        self.telemetry = TelemetryDispatcher(state=self.state, target_fps=20)
        self.sink = AudioOutputSink(sample_rate=16000, channels=1, enable_device_output=False)
        self.engine = RealtimeAudioEngine(
            pipeline=self.pipeline,
            state=self.state,
            telemetry_dispatcher=self.telemetry,
            output_sink=self.sink,
            frame_size=256,
            sample_rate=16000,
        )

    def tearDown(self):
        self.engine.stop()
        self.pipeline.close()

    def test_realtime_latency_bound(self):
        """Verifies that DSP frame processing latency is strictly < 15ms per 256-sample block (<50ms total)."""
        # Warm-up block to exclude ONNX session kernel initialization
        self.pipeline.process_block(np.zeros(256, dtype=np.float32), np.zeros(256, dtype=np.float32))

        latencies = []
        for _ in range(50):
            p = np.random.randn(256) * 0.1
            r = np.random.randn(256) * 0.1
            t0 = time.perf_counter()
            res = self.pipeline.process_block(p, r)
            proc_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(proc_ms)

        avg_lat = float(np.mean(latencies))
        p95_lat = float(np.percentile(latencies, 95))
        
        # Real-time requirement: processing a 16ms block must take far less than 16ms!
        self.assertLess(avg_lat, 8.0, f"Average processing latency {avg_lat:.2f}ms exceeds 8ms budget.")
        self.assertLess(p95_lat, 15.0, f"P95 processing latency {p95_lat:.2f}ms exceeds 15ms limit.")

    def test_before_after_toggle(self):
        """Verifies Before/After toggle switches output instantly without reprocessing."""
        p = np.random.randn(256) * 0.5
        r = np.random.randn(256) * 0.5
        
        # After (Enhanced) Mode
        self.state.set_before_after(True)
        res_enhanced = self.pipeline.process_block(p, r)
        self.assertFalse(np.allclose(res_enhanced["audio"], p))

        # Before (Raw) Mode
        self.state.set_before_after(False)
        res_raw = self.pipeline.process_block(p, r)
        # Raw mode must output primary block exactly
        np.testing.assert_allclose(res_raw["audio"], p)

    def test_streaming_engine_run(self):
        """Verifies engine starts, streams frames, and dispatches telemetry without crashes."""
        self.engine.start()
        time.sleep(0.3)  # Let it stream ~18 frames
        self.assertTrue(self.state.is_running)
        self.assertGreater(self.state.frames_processed, 5)
        
        # Check that telemetry queue receives items
        packet = self.telemetry.get_telemetry(timeout=0.2)
        self.assertIsNotNone(packet)
        self.assertIn("latency_ms", packet)
        self.assertIn("primary_samples", packet)
        self.assertIn("output_samples", packet)
        
        self.engine.stop()
        self.assertFalse(self.state.is_running)


if __name__ == "__main__":
    unittest.main()
