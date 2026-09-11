#!/usr/bin/env bash
# ==============================================================================
# IGARD-Net — NVIDIA Jetson Nano Dependency Installer
# Installs system audio libraries, PyTorch Jetson wheels, and Python packages.
# ==============================================================================

set -e

echo "=== IGARD-Net Jetson Nano Setup ==="

# 1. System audio & I2C packages
echo "[1/3] Installing system audio and I2C dependencies..."
sudo apt-get update
sudo apt-get install -y \
    libportaudio2 \
    libasound2-dev \
    portaudio19-dev \
    libsndfile1 \
    i2c-tools \
    python3-smbus \
    python3-pip \
    python3-dev

# 2. Add current user to i2c & audio groups
echo "[2/3] Configuring permissions..."
sudo usermod -a -G i2c,audio $USER || true

# 3. Python dependencies
echo "[3/3] Installing Python dependencies..."
pip3 install --upgrade pip
pip3 install -r requirements.txt

echo "=== Setup complete! ==="
echo "Run IGARD-Net on Jetson Nano with:"
echo "  ./scripts/run_jetson.sh"
