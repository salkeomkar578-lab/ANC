"""
Audio Output Subsystem.
Supports:
  1. Local low-latency speaker playback via sounddevice.
  2. Streaming PCM broadcast buffer for real-time browser Web Audio playback.
"""

import threading
from typing import Optional, List
import numpy as np

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except (ImportError, OSError):
    SOUNDDEVICE_AVAILABLE = False


class AudioOutputSink:
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        enable_device_output: bool = False,
        device_index: Optional[int] = None,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.enable_device_output = enable_device_output and SOUNDDEVICE_AVAILABLE
        self.device_index = device_index
        
        self._stream = None
        if self.enable_device_output:
            try:
                self._stream = sd.OutputStream(
                    samplerate=sample_rate,
                    channels=channels,
                    dtype="float32",
                    device=device_index,
                    latency="low",
                )
                self._stream.start()
            except Exception as e:
                print(f"[AudioOutput] Device output init failed ({e}). Playing via web stream only.")
                self.enable_device_output = False

        # Web Audio chunk subscribers (each connected browser client gets PCM chunks)
        self._subscribers: List[threading.Condition] = []
        self._latest_pcm_bytes: bytes = b""
        self._sub_lock = threading.Lock()

    def write(self, audio_block: np.ndarray):
        """Writes audio block to local sound card and browser streaming buffer."""
        # 1. Local sounddevice stream
        if self.enable_device_output and self._stream is not None:
            try:
                arr = audio_block.astype(np.float32)
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                self._stream.write(arr)
            except Exception:
                pass

        # 2. Convert to 16-bit PCM for low-bandwidth, low-latency browser streaming
        clipped = np.clip(audio_block, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype(np.int16).tobytes()

        with self._sub_lock:
            self._latest_pcm_bytes = pcm16

    def get_latest_pcm(self) -> bytes:
        with self._sub_lock:
            return self._latest_pcm_bytes

    def close(self):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
