"""
NVIDIA CUDA Backend for IGARD-Net using PyTorch.
Exploits NVIDIA GPU tensor cores, cuFFT, and parallel operations.
Suitable for NVIDIA Jetson Nano, Jetson Orin, and NVIDIA-equipped workstations/laptops.
"""

from typing import Tuple, Optional
import numpy as np
from backends.base import ComputeBackend

try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False


class CUDABackend(ComputeBackend):
    def __init__(self, device_id: int = 0):
        if not (TORCH_AVAILABLE and CUDA_AVAILABLE):
            raise RuntimeError("PyTorch with CUDA support is not available.")
        self.device = torch.device(f"cuda:{device_id}")
        self._name = f"CUDA:{torch.cuda.get_device_name(self.device)}"
        # Warm up CUDA context
        dummy = torch.zeros(64, device=self.device)
        _ = torch.fft.rfft(dummy)

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_cuda(self) -> bool:
        return True

    def nlms_process_block(
        self,
        weights: np.ndarray,
        ref_buffer: np.ndarray,
        primary_block: np.ndarray,
        ref_block: np.ndarray,
        mu: float,
        eps: float,
        leakage: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Hardware-aware execution:
        Tiny sample-by-sample 64-tap updates execute on host CPU to eliminate
        host-to-device kernel launch latency (saving ~1,280 driver launches per block),
        while CUDA cuFFT is leveraged for parallel spectral operations.
        """
        from backends.cpu_backend import CPUBackend
        if not hasattr(self, "_cpu_fallback"):
            self._cpu_fallback = CPUBackend()
        return self._cpu_fallback.nlms_process_block(
            weights=weights,
            ref_buffer=ref_buffer,
            primary_block=primary_block,
            ref_block=ref_block,
            mu=mu,
            eps=eps,
            leakage=leakage,
        )

    def compute_rfft_power(self, block: np.ndarray, window: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        t_block = torch.tensor(block, dtype=torch.float32, device=self.device)
        t_win = torch.tensor(window, dtype=torch.float32, device=self.device)
        
        windowed = t_block * t_win
        spectrum = torch.fft.rfft(windowed)
        power = torch.abs(spectrum) ** 2
        phase = torch.angle(spectrum)
        
        return (
            spectrum.cpu().numpy(),
            power.cpu().numpy().astype(np.float64),
            phase.cpu().numpy().astype(np.float64),
        )

    def compute_irfft(self, spectrum: np.ndarray, n: int) -> np.ndarray:
        t_spec = torch.tensor(spectrum, dtype=torch.complex64, device=self.device)
        rec = torch.fft.irfft(t_spec, n=n)
        return rec.cpu().numpy().astype(np.float64)

    def apply_spectral_mask(
        self,
        spectrum: np.ndarray,
        power: np.ndarray,
        noise_power: np.ndarray,
        alpha: float,
        min_gain: float,
    ) -> np.ndarray:
        # Move to GPU
        t_spec = torch.tensor(spectrum, dtype=torch.complex64, device=self.device)
        t_p = torch.tensor(power, dtype=torch.float32, device=self.device)
        
        if len(noise_power) != len(power):
            n_p_interp = np.interp(
                np.linspace(0, 1, len(power)),
                np.linspace(0, 1, len(noise_power)),
                noise_power
            )
            t_np = torch.tensor(n_p_interp, dtype=torch.float32, device=self.device)
        else:
            t_np = torch.tensor(noise_power, dtype=torch.float32, device=self.device)

        subtracted = torch.clamp(t_p - alpha * t_np, min=0.0)
        gain = subtracted / (t_p + 1.0e-10)
        gain = torch.clamp(gain, min=min_gain, max=1.0)
        
        # 3-tap spectral smoothing
        if len(gain) >= 3:
            rolled_left = torch.roll(gain, -1)
            rolled_right = torch.roll(gain, 1)
            gain_smoothed = 0.25 * rolled_right + 0.5 * gain + 0.25 * rolled_left
            gain_smoothed[0] = gain[0]
            gain_smoothed[-1] = gain[-1]
            gain = gain_smoothed

        cleaned = t_spec * gain
        return cleaned.cpu().numpy()
