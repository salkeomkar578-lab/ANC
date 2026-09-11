"""
Native C-based RNNoise Noise Suppression Processor for IGARD-Net V3.
Directly interfaces with official rnnoise.dll / librnnoise.so via ctypes.
Processes 48 kHz mono audio in 480-sample (10 ms) frames with persistent GRU state.
"""

import ctypes
import os
from typing import Tuple, Optional, Union
import numpy as np

try:
    from pyrnnoise.rnnoise import lib, FRAME_SIZE, SAMPLE_RATE
except ImportError:
    # Fallback direct ctypes loader
    import platform
    if platform.system() == "Windows":
        lib_name = "rnnoise.dll"
    elif platform.system() == "Darwin":
        lib_name = "librnnoise.dylib"
    else:
        lib_name = "librnnoise.so"
    
    # Try finding in site-packages
    import site
    lib_path = None
    for sp in site.getsitepackages():
        candidate = os.path.join(sp, "pyrnnoise", lib_name)
        if os.path.exists(candidate):
            lib_path = candidate
            break
    if not lib_path or not os.path.exists(lib_path):
        raise OSError(f"Could not locate native RNNoise shared library: {lib_name}")
        
    lib = ctypes.CDLL(lib_path)
    lib.rnnoise_create.argtypes = [ctypes.c_void_p]
    lib.rnnoise_create.restype = ctypes.c_void_p
    lib.rnnoise_destroy.argtypes = [ctypes.c_void_p]
    lib.rnnoise_process_frame.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.rnnoise_process_frame.restype = ctypes.c_float
    lib.rnnoise_get_frame_size.restype = ctypes.c_int
    FRAME_SIZE = lib.rnnoise_get_frame_size()
    SAMPLE_RATE = 48000


class RNNoiseProcessor:
    """
    Native C RNNoise Processor with zero-copy float32 streaming.
    Frame size is 480 samples (10 ms at 48 kHz).
    """

    FRAME_SIZE = 480
    SAMPLE_RATE = 48000

    def __init__(self):
        self._state: Optional[ctypes.c_void_p] = lib.rnnoise_create(None)
        if not self._state:
            raise RuntimeError("Failed to allocate RNNoise C state.")
        self._work_buffer = np.zeros(self.FRAME_SIZE, dtype=np.float32)
        self._ptr = self._work_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self._last_vad: float = 0.0

    def __del__(self):
        if hasattr(self, "_state") and self._state is not None:
            try:
                lib.rnnoise_destroy(self._state)
            except Exception:
                pass
            self._state = None

    def reset(self) -> None:
        """Reset RNNoise internal recurrent states."""
        if self._state is not None:
            lib.rnnoise_destroy(self._state)
        self._state = lib.rnnoise_create(None)
        self._last_vad = 0.0

    def process_frame(self, frame_48k: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Process exactly 480 samples of 48 kHz mono float32 audio.
        Audio range should be normalized [-1.0, 1.0].
        Returns (denoised_frame, rnnoise_vad_probability).
        """
        if len(frame_48k) != self.FRAME_SIZE:
            raise ValueError(
                f"RNNoise frame must be exactly {self.FRAME_SIZE} samples, got {len(frame_48k)}"
            )

        # Scale float [-1.0, 1.0] to RNNoise internal PCM amplitude [-32767.0, 32767.0]
        np.multiply(frame_48k, 32767.0, out=self._work_buffer)

        # In-place C processing (fast zero-copy)
        vad_prob = float(lib.rnnoise_process_frame(self._state, self._ptr, self._ptr))

        # Scale back to normalized float32
        out_frame = self._work_buffer / 32767.0
        self._last_vad = vad_prob
        return out_frame, vad_prob

    def process_chunk(self, audio_48k: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Process an arbitrary-length 48 kHz mono float32 audio chunk.
        Breaks audio into 480-sample frames, padding the tail if necessary,
        and returns exactly len(audio_48k) processed samples.
        """
        arr = np.asarray(audio_48k, dtype=np.float32)
        n = len(arr)
        if n == 0:
            return np.zeros(0, dtype=np.float32), self._last_vad
        if n == self.FRAME_SIZE:
            return self.process_frame(arr)

        out_pieces = []
        last_vad = self._last_vad
        for i in range(0, n, self.FRAME_SIZE):
            sub = arr[i : i + self.FRAME_SIZE]
            sub_len = len(sub)
            if sub_len < self.FRAME_SIZE:
                pad_frame = np.zeros(self.FRAME_SIZE, dtype=np.float32)
                pad_frame[:sub_len] = sub
                denoised, vad = self.process_frame(pad_frame)
                out_pieces.append(denoised[:sub_len])
                last_vad = vad
            else:
                denoised, vad = self.process_frame(sub)
                out_pieces.append(denoised)
                last_vad = vad

        return np.concatenate(out_pieces), last_vad

    @property
    def last_vad_probability(self) -> float:
        """Return the RNNoise internal VAD probability of the last processed frame."""
        return self._last_vad
