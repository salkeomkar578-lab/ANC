"""
IGARD-Net Dedicated Entrypoint for NVIDIA Jetson Nano.
Optimized for:
  - 128-core NVIDIA Maxwell GPU (sm_53)
  - Quad-Core ARM Cortex-A57 @ 1.43 GHz
  - 4GB Shared LPDDR4 Unified Memory
  - Sub-20ms real-time audio latency budget
  - Automatic ADXL345 I2C / Mock accelerometer fallback
"""

import os
import sys
import socket
from pathlib import Path

# Ensure root directory is on sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Force hardware profile to JETSON_NANO in environment if not set
os.environ.setdefault("IGARD_HARDWARE_PROFILE", "JETSON_NANO")
# Set thread counts to match Jetson Nano's 4 physical Cortex-A57 cores
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

from core.config import load_config
from backends.backend_manager import BackendManager, detect_hardware_profile
from dashboard.app import app, socketio, backend_mgr, config


def get_local_ip() -> str:
    """Detects local LAN IP for easy remote browser access from laptop/PC to Jetson."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def print_jetson_banner():
    profile, cuda_available = detect_hardware_profile()
    local_ip = get_local_ip()
    port = config["dashboard"].get("port", 5000)

    print("\n" + "=" * 62)
    print("  IGARD-Net — NVIDIA Jetson Nano Edge Tactical Audio AI")
    print("  Reconstructed Low-Latency Speech Preserving Pipeline")
    print("=" * 62)
    print(f"Hardware Target:  NVIDIA Jetson Nano (ARM Cortex-A57 4-Core)")
    print(f"GPU Accelerator:  128 Maxwell CUDA Cores [{'ACTIVE' if cuda_available else 'CPU SIMD FALLBACK'}]")
    print(f"Backend Engine:   {backend_mgr.backend.name}")
    print("Sample Rate:      16 kHz (Frame: 256 samples = 16.0 ms)")
    print("Audio Latency:    < 20 ms Total End-to-End Budget")
    print("Memory Mode:      Edge Optimized (< 300 MB Footprint)")
    print("Sensor Mode:      ADXL345 I2C Auto-Detected / Baseline Safe")
    print("\nOperational Modes:")
    print("  [ 1. LIVE DETECTION ] Continuous real-time stream (<20ms)")
    print("  [ 2. FILE STUDIO ]    WAV processing with ITU-T MOS Quality Analysis")
    print("\nWeb Telemetry & Live Audio Stream:")
    print(f"  Local Access:   http://localhost:{port}")
    print(f"  Network Access: http://{local_ip}:{port}")
    print(f"  Live Audio URL: http://{local_ip}:{port}/api/stream/live_audio")
    print("=" * 62)
    print("Ready. Open browser on any PC/tablet connected to the same network.\n")


def main():
    print_jetson_banner()
    host = "0.0.0.0"
    port = config["dashboard"].get("port", 5000)
    socketio.run(app, host=host, port=port, debug=False, allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    main()
