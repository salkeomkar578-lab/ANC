"""
Shared FFT / spectrogram helper for IGARD-Net.

Extracts the windowed-FFT magnitude computation already used in
noise_classifier.py into a reusable function, so both the pipeline
telemetry (for dashboard waveform/spectrogram visuals) and the
classifier use the same math — no duplicated FFT code.

Output is deliberately compact (downsampled to ≤128 frequency bins)
so it can be pushed over WebSocket to a browser every ~32ms without
saturating the Pi 4's network or CPU budget.
"""

import numpy as np


def compute_fft_magnitudes(block, sample_rate=16000, max_bins=128):
    """
    Compute windowed FFT magnitudes suitable for spectrogram display.

    Parameters
    ----------
    block : 1D numpy array
        Audio samples (one processing block, e.g. 512 samples).
    sample_rate : int
        Sample rate in Hz.
    max_bins : int
        Maximum number of frequency bins to return. The full FFT is
        downsampled (averaged) to this many bins to keep WebSocket
        payloads small.

    Returns
    -------
    dict with:
        freqs : 1D array of frequency values (Hz), length ≤ max_bins
        magnitudes : 1D array of magnitude values, same length as freqs
        peak_freq : float, frequency of the strongest bin (Hz)
        peak_magnitude : float, magnitude of the strongest bin
    """
    block = np.asarray(block, dtype=np.float64)
    n = len(block)
    if n == 0:
        return {
            "freqs": [],
            "magnitudes": [],
            "peak_freq": 0.0,
            "peak_magnitude": 0.0,
        }

    # Same windowing as noise_classifier.py — Hanning window + rfft
    window = np.hanning(n)
    spectrum = np.abs(np.fft.rfft(block * window))
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    # Convert to dB scale for better visual dynamic range, clamped to
    # a sane floor so silence doesn't produce -inf
    magnitude_db = 20.0 * np.log10(spectrum + 1e-10)
    magnitude_db = np.clip(magnitude_db, -80.0, 0.0)
    # Normalize to 0..1 range for easy canvas rendering
    magnitude_norm = (magnitude_db + 80.0) / 80.0

    # Downsample to max_bins by averaging adjacent bins — this keeps
    # WebSocket payloads small (128 floats vs 257 for a 512-sample block)
    full_bins = len(freqs)
    if full_bins > max_bins:
        bin_size = full_bins / max_bins
        downsampled_freqs = np.zeros(max_bins)
        downsampled_mags = np.zeros(max_bins)
        for i in range(max_bins):
            start_idx = int(i * bin_size)
            end_idx = int((i + 1) * bin_size)
            end_idx = min(end_idx, full_bins)
            downsampled_freqs[i] = np.mean(freqs[start_idx:end_idx])
            downsampled_mags[i] = np.mean(magnitude_norm[start_idx:end_idx])
        freqs = downsampled_freqs
        magnitude_norm = downsampled_mags
    else:
        freqs = freqs.copy()
        magnitude_norm = magnitude_norm.copy()

    peak_idx = int(np.argmax(magnitude_norm))
    return {
        "freqs": freqs.tolist(),
        "magnitudes": magnitude_norm.tolist(),
        "peak_freq": float(freqs[peak_idx]),
        "peak_magnitude": float(magnitude_norm[peak_idx]),
    }


def compute_rms_level(block):
    """
    Compute RMS level of a block, normalized to 0..1 for mic meter display.
    """
    block = np.asarray(block, dtype=np.float64)
    if len(block) == 0:
        return 0.0
    rms = float(np.sqrt(np.mean(block ** 2)))
    # Map to a 0..1 range assuming audio is normalized to [-1, 1].
    # Use a log scale for perceptually linear meter behavior.
    if rms < 1e-10:
        return 0.0
    db = 20.0 * np.log10(rms + 1e-10)
    # -60 dB = 0.0, 0 dB = 1.0
    level = max(0.0, min(1.0, (db + 60.0) / 60.0))
    return level


def compute_snr_estimate(clean_estimate, noisy_input):
    """
    Estimate SNR improvement from a single block. This is an ESTIMATE,
    not a ground-truth measurement (we don't have the true clean signal
    in real time). Labeled as 'estimated' everywhere it's displayed.
    """
    clean_estimate = np.asarray(clean_estimate, dtype=np.float64)
    noisy_input = np.asarray(noisy_input, dtype=np.float64)

    noise_removed = noisy_input - clean_estimate
    signal_power = float(np.mean(clean_estimate ** 2)) + 1e-12
    noise_power = float(np.mean(noise_removed ** 2)) + 1e-12

    snr_db = 10.0 * np.log10(signal_power / noise_power)
    return float(np.clip(snr_db, -30.0, 30.0))
