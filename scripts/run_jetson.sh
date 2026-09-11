#!/usr/bin/env bash
# ==============================================================================
# IGARD-Net — NVIDIA Jetson Nano Launcher
# Maximizes hardware clocks, configures CPU affinity, and runs IGARD-Net.
# ==============================================================================

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "[Jetson] Configuring hardware power and clocks..."
# Set 10W MAXN power mode if running with root/sudo
if [ "$EUID" -eq 0 ]; then
    nvpmodel -m 0 || true
    jetson_clocks || true
elif command -v sudo >/dev/null 2>&1; then
    sudo nvpmodel -m 0 2>/dev/null || true
    sudo jetson_clocks 2>/dev/null || true
fi

# Set 4-core threading
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export IGARD_HARDWARE_PROFILE=JETSON_NANO

echo "[Jetson] Launching IGARD-Net tactical audio AI..."
python3 run_jetson.py
