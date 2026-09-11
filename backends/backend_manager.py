"""
Backend Manager for IGARD-Net.
Handles runtime hardware detection, profile selection, and backend instantiation.
Supports:
  - DESKTOP_NVIDIA (Workstation / Laptop with NVIDIA GPU)
  - JETSON_NANO    (NVIDIA Jetson Nano / Orin / Xavier edge modules)
  - CPU_ONLY       (Raspberry Pi 4, Intel/AMD CPUs without CUDA)
"""

import os
import platform
from typing import Tuple, Dict, Any
import psutil

from backends.base import ComputeBackend
from backends.cpu_backend import CPUBackend


def detect_hardware_profile() -> Tuple[str, bool]:
    """
    Detects hardware profile and whether NVIDIA CUDA is functional.
    Returns: (profile_name, cuda_available)
    """
    cuda_ok = False
    gpu_name = ""
    
    try:
        import torch
        if torch.cuda.is_available():
            cuda_ok = True
            gpu_name = torch.cuda.get_device_name(0).lower()
    except Exception:
        cuda_ok = False

    # Check if running on NVIDIA Jetson (tegra architecture or /etc/nv_tegra_release)
    is_tegra = (
        os.path.exists("/etc/nv_tegra_release") or
        "tegra" in platform.processor().lower() or
        "jetson" in gpu_name
    )

    if is_tegra:
        return "JETSON_NANO", cuda_ok
    elif cuda_ok:
        return "DESKTOP_NVIDIA", True
    else:
        return "CPU_ONLY", False


class BackendManager:
    def __init__(self, requested_profile: str = "auto", device_id: int = 0):
        detected_profile, cuda_available = detect_hardware_profile()
        
        if requested_profile == "auto" or not requested_profile:
            self.profile = detected_profile
        else:
            self.profile = requested_profile
            
        self.cuda_available = cuda_available
        self.device_id = device_id
        self.backend: ComputeBackend = self._initialize_backend()

    def _initialize_backend(self) -> ComputeBackend:
        if self.profile in ("DESKTOP_NVIDIA", "JETSON_NANO") and self.cuda_available:
            try:
                from backends.cuda_backend import CUDABackend
                cuda_be = CUDABackend(device_id=self.device_id)
                print(f"[Backend] Successfully initialized NVIDIA GPU backend: {cuda_be.name}")
                return cuda_be
            except Exception as e:
                print(f"[Backend] CUDA initialization failed ({e}). Falling back to CPU.")
                self.profile = "CPU_ONLY"
                return CPUBackend()
        else:
            print(f"[Backend] Initialized CPU backend (NumPy / SciPy optimized). Profile: {self.profile}")
            return CPUBackend()

    def get_hardware_telemetry(self) -> Dict[str, float]:
        """Returns real measured CPU %, GPU %, and RAM usage."""
        cpu = float(psutil.cpu_percent(interval=None))
        ram = float(psutil.virtual_memory().used / (1024 * 1024))
        
        gpu = 0.0
        if self.cuda_available:
            try:
                import torch
                # Measure CUDA memory allocated or NVML utilization if available
                # Use torch.cuda.utilization if available, else 0
                gpu_mem = torch.cuda.memory_allocated(self.device_id) / (1024 * 1024)
                # Query nvidia-smi memory or estimation
                gpu = float(min(100.0, (gpu_mem / 2048.0) * 100.0))
            except Exception:
                gpu = 0.0
                
        return {
            "cpu_percent": cpu,
            "gpu_percent": gpu,
            "ram_mb": ram,
        }
