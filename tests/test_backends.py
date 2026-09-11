import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import unittest
import numpy as np

from backends.backend_manager import BackendManager, detect_hardware_profile
from backends.cpu_backend import CPUBackend


class TestBackends(unittest.TestCase):
    def test_hardware_detection(self):
        profile, cuda_available = detect_hardware_profile()
        self.assertIn(profile, ["DESKTOP_NVIDIA", "JETSON_NANO", "CPU_ONLY"])
        self.assertIsInstance(cuda_available, bool)

    def test_cpu_backend_math(self):
        cpu = CPUBackend()
        self.assertEqual(cpu.name, "CPU")
        self.assertFalse(cpu.is_cuda)

        weights = np.zeros(32)
        ref_buf = np.zeros(32)
        prim = np.ones(128)
        ref = np.ones(128) * 0.5

        err, noise_est, new_w, new_ref = cpu.nlms_process_block(
            weights, ref_buf, prim, ref, mu=0.2, eps=1e-6
        )
        self.assertEqual(len(err), 128)
        self.assertEqual(len(new_w), 32)
        self.assertFalse(np.any(np.isnan(err)))

    def test_cuda_backend_if_available(self):
        profile, cuda_available = detect_hardware_profile()
        if not cuda_available:
            self.skipTest("NVIDIA CUDA not available on this machine.")

        from backends.cuda_backend import CUDABackend
        cuda = CUDABackend()
        cpu = CPUBackend()

        self.assertTrue(cuda.is_cuda)
        self.assertIn("CUDA", cuda.name)

        weights = np.zeros(32, dtype=np.float64)
        ref_buf = np.zeros(32, dtype=np.float64)
        prim = np.random.randn(128).astype(np.float64)
        ref = np.random.randn(128).astype(np.float64)

        err_cpu, _, w_cpu, _ = cpu.nlms_process_block(weights, ref_buf, prim, ref, mu=0.2, eps=1e-6)
        err_cuda, _, w_cuda, _ = cuda.nlms_process_block(weights, ref_buf, prim, ref, mu=0.2, eps=1e-6)

        # Numerical equivalence check
        np.testing.assert_allclose(err_cpu, err_cuda, atol=1e-4)
        np.testing.assert_allclose(w_cpu, w_cuda, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
