"""
IGARD-Net Master Entry Point.
Launches the reconstructed low-latency real-time tactical audio system.
"""

import sys
from pathlib import Path

# Ensure igard_net root is in sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import load_config
from backends.backend_manager import BackendManager, detect_hardware_profile
from dashboard.app import app, socketio, backend_mgr, config


def print_startup_banner():
    profile, cuda_available = detect_hardware_profile()
    hw_type = "NVIDIA CUDA Acceleration" if cuda_available else "Standard CPU / Edge ARM"
    backend_name = backend_mgr.backend.name
    
    print("\n" + "=" * 55)
    print("  IGARD-Net — Tactical Audio Noise Cancellation")
    print("  Reconstructed Low-Latency Real-Time Pipeline")
    print("=" * 55)
    print(f"Hardware:       {hw_type} [{profile}]")
    print(f"Backend:        {backend_name}")
    print("Audio Device:   Synthetic Dual-Mic Tactical Simulation (or USB sound cards)")
    print("Sample Rate:    16 kHz (Frame: 256 samples = 16.0 ms)")
    print("\nOperating Modes:")
    print("  [ REAL-TIME MODE ] Continuous streaming (<50 ms latency)")
    print("  [ FILE MODE ]      High-throughput WAV processing (RTF < 0.20)")
    print("\nActive Controls:")
    print("  Autopilot:      [ ON  ] Adaptive Scene Tuning")
    print("  Before / After: [ AFTER (Enhanced Audio) ]")
    print("\nTelemetry & Audio Stream Dashboard:")
    port = config["dashboard"].get("port", 5000)
    print(f"  Web Interface:  http://localhost:{port}")
    print(f"  Live Audio URL: http://localhost:{port}/api/stream/live_audio")
    print("=" * 55)
    print("Status: READY — Streaming active. Open browser to listen live.\n")


def main():
    print_startup_banner()
    host = config["dashboard"].get("host", "0.0.0.0")
    port = config["dashboard"].get("port", 5000)
    socketio.run(app, host=host, port=port, debug=False, allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    main()
