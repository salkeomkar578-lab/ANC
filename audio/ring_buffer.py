"""
High-Performance Audio Ring Buffer with Underrun/Overrun Accounting.
Provides thread-safe FIFO buffering between audio capture, DSP processing, and playback.
"""

import threading
from typing import Tuple
import numpy as np


class AudioRingBuffer:
    def __init__(self, capacity_samples: int = 16000, channels: int = 1, dtype=np.float32):
        self.capacity = capacity_samples
        self.channels = channels
        self.dtype = dtype
        
        self._buffer = np.zeros((capacity_samples, channels), dtype=dtype)
        self._write_pos = 0
        self._read_pos = 0
        self._available = 0
        
        self._lock = threading.Lock()
        self.underruns = 0
        self.overruns = 0

    def write(self, data: np.ndarray) -> int:
        """Writes audio frames into buffer. Drops oldest on overrun."""
        data = np.asarray(data, dtype=self.dtype)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        num_samples = len(data)
        if num_samples == 0:
            return 0

        with self._lock:
            if num_samples > self.capacity:
                # Truncate to capacity
                data = data[-self.capacity:]
                num_samples = self.capacity
                self.overruns += 1

            free_space = self.capacity - self._available
            if num_samples > free_space:
                # Overrun: advance read pointer to drop oldest samples
                dropped = num_samples - free_space
                self._read_pos = (self._read_pos + dropped) % self.capacity
                self._available -= dropped
                self.overruns += 1

            # Circular write
            end1 = min(num_samples, self.capacity - self._write_pos)
            self._buffer[self._write_pos : self._write_pos + end1] = data[:end1]
            
            end2 = num_samples - end1
            if end2 > 0:
                self._buffer[:end2] = data[end1:num_samples]

            self._write_pos = (self._write_pos + num_samples) % self.capacity
            self._available += num_samples
            return num_samples

    def read(self, num_samples: int) -> Tuple[np.ndarray, bool]:
        """
        Reads num_samples. If insufficient data (underrun), pads with zeros.
        Returns: (data, underrun_occurred: bool)
        """
        out = np.zeros((num_samples, self.channels), dtype=self.dtype)
        underrun = False

        with self._lock:
            if self._available < num_samples:
                underrun = True
                self.underruns += 1
                to_read = self._available
            else:
                to_read = num_samples

            if to_read > 0:
                end1 = min(to_read, self.capacity - self._read_pos)
                out[:end1] = self._buffer[self._read_pos : self._read_pos + end1]

                end2 = to_read - end1
                if end2 > 0:
                    out[end1 : end1 + end2] = self._buffer[:end2]

                self._read_pos = (self._read_pos + to_read) % self.capacity
                self._available -= to_read

        return out, underrun

    def available_read(self) -> int:
        with self._lock:
            return self._available

    def clear(self):
        with self._lock:
            self._write_pos = 0
            self._read_pos = 0
            self._available = 0
            self._buffer.fill(0)
