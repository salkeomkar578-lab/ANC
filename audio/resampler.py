"""
Audio Resampling Utility.
Fast conversion of arbitrary audio to target sample rate (default 16 kHz).
"""

import numpy as np

try:
    from scipy.signal import resample_poly
    SCIPY_RESAMPLE = True
except ImportError:
    SCIPY_RESAMPLE = False


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    """Resamples 1D or 2D audio array to target_sr."""
    if orig_sr == target_sr:
        return audio.astype(np.float64)

    if len(audio) == 0:
        return audio

    if SCIPY_RESAMPLE:
        import math
        gcd = math.gcd(orig_sr, target_sr)
        up = target_sr // gcd
        down = orig_sr // gcd
        resampled = resample_poly(audio, up, down, axis=0)
        return resampled.astype(np.float64)
    else:
        # Fast linear interpolation fallback
        target_len = int(len(audio) * target_sr / orig_sr)
        if audio.ndim == 1:
            resampled = np.interp(
                np.linspace(0, len(audio), target_len, endpoint=False),
                np.arange(len(audio)),
                audio
            )
        else:
            resampled = np.empty((target_len, audio.shape[1]), dtype=np.float64)
            for ch in range(audio.shape[1]):
                resampled[:, ch] = np.interp(
                    np.linspace(0, len(audio), target_len, endpoint=False),
                    np.arange(len(audio)),
                    audio[:, ch]
                )
        return resampled.astype(np.float64)
