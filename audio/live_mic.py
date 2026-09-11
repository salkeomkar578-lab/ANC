"""
Live Microphone Input Streamer for IGARD-Net V3.
Captures real-time audio from local microphone input device via sounddevice,
with seamless fallback to synthetic scenario streaming if no hardware device is available.
"""

import threading
import queue
from typing import Optional, Tuple
import numpy as np

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except (ImportError, OSError):
    HAS_SOUNDDEVICE = False

from audio.audio_input import SyntheticAudioStreamer


class LiveMicrophoneStreamer:
    """
    Live audio capture stream feeding frames directly into the V3 audio engine.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_size: int = 256,
        device_index: Optional[int] = None,
    ):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.device_index = device_index
        self._queue = queue.Queue(maxsize=100)
        self._stream: Optional[sd.InputStream] = None
        self._is_active = False
        self.hardware_active = False

        self.synthetic_fallback = SyntheticAudioStreamer(
            sample_rate=sample_rate, frame_size=frame_size
        )

    def start(self) -> bool:
        """Start capturing from physical microphone."""
        if not HAS_SOUNDDEVICE:
            self.hardware_active = False
            return False

        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.frame_size,
                channels=1,
                dtype="float32",
                device=self.device_index,
                callback=self._audio_callback,
            )
            self._stream.start()
            self._is_active = True
            self.hardware_active = True
            return True
        except Exception as e:
            print(f"[LiveMic] Could not open hardware mic ({e}). Using synthetic fallback.")
            self.hardware_active = False
            self._is_active = False
            return False

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            pass
        data = indata[:, 0].copy()
        try:
            self._queue.put_nowait(data)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(data)

    def read_block(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Read the next audio block.
        Returns: (primary_mic, reference_mic, optional_ground_truth)
        """
        if self.hardware_active and self._is_active:
            try:
                chunk = self._queue.get(timeout=0.04)
                # Primary is live mic, reference is synthetic/hiss reference
                ref = np.random.normal(0, 0.01, len(chunk)).astype(np.float32)
                return chunk, ref, None
            except queue.Empty:
                pass

        # Fallback to synthetic tactical generator
        return self.synthetic_fallback.read_block()

    def stop(self):
        """Stop microphone stream."""
        self._is_active = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self.hardware_active = False
