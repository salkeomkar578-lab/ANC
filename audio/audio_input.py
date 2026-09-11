"""
Audio Input Streamer.
Supports:
  1. Physical dual-microphone hardware capture via sounddevice.
  2. Synthetic Tactical Defense Scenario Audio generator (speech + engine drone + gunfire bursts).
"""

import time
from typing import Tuple, Generator, Optional
import numpy as np

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except (ImportError, OSError):
    SOUNDDEVICE_AVAILABLE = False


def generate_synthetic_tactical_scenario(
    duration_s: float = 12.0,
    sample_rate: int = 16000,
    quiet_lead_in_s: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates high-realism tactical audio:
      - Speech: harmonic speech formant synthesis with realistic speech pauses
      - Noise: low-frequency engine/rotor drone + wind + broadband background
      - Bursts: 4 sharp impulsive shocks (gunfire/blast signatures)
    Returns: (clean_speech, primary_mic, reference_mic)
    """
    n_total = int(sample_rate * duration_s)
    t = np.arange(n_total) / sample_rate

    # 1. Clean speech synthesis (voice simulation with syllabic cadence)
    cadence = 0.5 * (1.0 + np.sin(2.0 * np.pi * 1.5 * t)) * (np.sin(2.0 * np.pi * 0.4 * t) > -0.2)
    voice = (
        0.45 * np.sin(2.0 * np.pi * 190.0 * t) +
        0.30 * np.sin(2.0 * np.pi * 380.0 * t) +
        0.20 * np.sin(2.0 * np.pi * 760.0 * t) +
        0.15 * np.sin(2.0 * np.pi * 1520.0 * t) +
        0.10 * np.sin(2.0 * np.pi * 2280.0 * t)
    ) * cadence

    # 2. Steady engine/rotor drone
    rotor_f0 = 85.0
    drone = (
        0.40 * np.sin(2.0 * np.pi * rotor_f0 * t) +
        0.25 * np.sin(2.0 * np.pi * rotor_f0 * 2 * t) +
        0.18 * np.sin(2.0 * np.pi * rotor_f0 * 3 * t) +
        0.12 * np.sin(2.0 * np.pi * rotor_f0 * 4 * t)
    )
    # Add broadband background rumble
    rng = np.random.default_rng(101)
    broadband = rng.normal(0, 0.08, n_total)
    steady_noise = drone + broadband

    # 3. Impulsive gunfire bursts (sharp transients at specific timestamps)
    burst_noise = steady_noise.copy()
    burst_times = [2.5, 5.0, 7.5, 10.0]
    burst_len = int(sample_rate * 0.035)  # 35 ms burst
    for bt in burst_times:
        start_idx = int(bt * sample_rate)
        if start_idx + burst_len < n_total:
            shock_pulse = rng.normal(0, 1.4, burst_len) * np.hanning(burst_len)
            burst_noise[start_idx : start_idx + burst_len] += shock_pulse

    # 4. Quiet lead-in for presence gate calibration
    lead_samples = int(quiet_lead_in_s * sample_rate)
    clean_speech = np.concatenate((np.zeros(lead_samples), voice))
    noise = np.concatenate((rng.normal(0, 0.002, lead_samples), burst_noise))

    # Primary mic: Speech + Environmental Noise
    primary = clean_speech + noise

    # Reference mic: Environmental Noise + independent acoustic phase/sensor hiss
    reference = noise + rng.normal(0, 0.02, len(noise))

    # Shared normalization
    shared_peak = max(np.max(np.abs(clean_speech)), np.max(np.abs(primary)), np.max(np.abs(reference)), 1.0e-6)
    scale = 0.85 / shared_peak
    
    return clean_speech * scale, primary * scale, reference * scale


class SyntheticAudioStreamer:
    """Streams simulated tactical scenario blocks continuously in a seamless loop."""
    def __init__(self, sample_rate: int = 16000, frame_size: int = 256):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.clean_speech, self.primary, self.reference = generate_synthetic_tactical_scenario(
            duration_s=12.0, sample_rate=sample_rate, quiet_lead_in_s=0.5
        )
        self.total_samples = len(self.primary)
        self.pos = 0

    def read_block(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns: (primary_block, reference_block, clean_ground_truth_block)"""
        end = self.pos + self.frame_size
        if end > self.total_samples:
            # Wrap around loop
            self.pos = int(0.5 * self.sample_rate)  # Skip initial calibration on loop
            end = self.pos + self.frame_size

        p_block = self.primary[self.pos : end]
        r_block = self.reference[self.pos : end]
        c_block = self.clean_speech[self.pos : end]
        self.pos = end
        return p_block, r_block, c_block
