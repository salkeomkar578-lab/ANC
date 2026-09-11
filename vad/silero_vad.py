"""
Silero VAD (ONNX) Streaming Speech Detection Module for IGARD-Net V3.
Provides continuous speech_probability in [0.0, 1.0] rather than binary mute flags.
Optimized for low-latency streaming and edge CPU deployment (Jetson Nano, ARM, x86).
"""

import os
from typing import Optional, Union, Tuple
import numpy as np
import onnxruntime as ort


class SileroVAD:
    """
    Streaming ONNX Silero VAD engine.
    Maintains recurrent hidden states and context across consecutive audio frames.
    Operates natively at 16 kHz with 512-sample (32 ms) analysis chunks.
    """

    SAMPLE_RATE = 16000
    WINDOW_SIZE_SAMPLES = 512  # 32 ms at 16 kHz
    CONTEXT_SIZE = 64

    def __init__(
        self,
        model_path: Optional[str] = None,
        force_cpu: bool = True,
    ):
        if model_path is None or not os.path.exists(model_path):
            # Check local repository path first
            repo_path = os.path.join(os.path.dirname(__file__), "silero_vad.onnx")
            if os.path.exists(repo_path):
                model_path = repo_path
            else:
                # Fallback to silero_vad package path
                try:
                    import silero_vad
                    pkg_path = os.path.join(
                        os.path.dirname(silero_vad.__file__), "data", "silero_vad.onnx"
                    )
                    if os.path.exists(pkg_path):
                        model_path = pkg_path
                except Exception:
                    pass

        if model_path is None or not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Silero VAD ONNX model not found at {model_path}. "
                "Ensure vad/silero_vad.onnx exists."
            )

        self.model_path = model_path
        
        # Configure ONNX Runtime for ultra-low latency single-thread execution
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = ["CPUExecutionProvider"]
        if not force_cpu and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")

        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=opts,
            providers=providers,
        )

        self._sr_tensor = np.array(self.SAMPLE_RATE, dtype=np.int64)
        self._buffer = np.zeros(0, dtype=np.float32)
        self._last_speech_prob: float = 0.0
        self.reset()

    def reset(self) -> None:
        """Reset internal recurrent hidden states and streaming context."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self.CONTEXT_SIZE), dtype=np.float32)
        self._buffer = np.zeros(0, dtype=np.float32)
        self._last_speech_prob = 0.0

    def _infer_512(self, chunk_512: np.ndarray) -> float:
        """
        Execute a single forward pass with exactly 512 samples.
        Updates persistent RNN hidden state and streaming context.
        """
        if chunk_512.ndim == 1:
            chunk_2d = chunk_512.reshape(1, -1).astype(np.float32)
        else:
            chunk_2d = chunk_512.astype(np.float32)

        # Concatenate preceding 64-sample context with current 512 samples -> (1, 576)
        x = np.concatenate([self._context, chunk_2d], axis=1)

        # Run ONNX inference
        ort_inputs = {
            "input": x,
            "state": self._state,
            "sr": self._sr_tensor,
        }
        out, new_state = self.session.run(None, ort_inputs)

        # Update persistent states
        self._state = new_state
        self._context = x[:, -self.CONTEXT_SIZE :]
        
        prob = float(out[0, 0])
        # Clamp to valid probability interval
        prob = max(0.0, min(1.0, prob))
        self._last_speech_prob = prob
        return prob

    def process_chunk(self, audio_16k: np.ndarray) -> float:
        """
        Process an arbitrary-length 16 kHz audio chunk in streaming mode.
        Accumulates samples into internal FIFO buffer.
        Returns the most recent speech_probability in [0.0, 1.0].
        """
        audio = np.asarray(audio_16k, dtype=np.float32)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)  # Downmix to mono if multi-channel

        # Append to streaming buffer
        self._buffer = np.concatenate([self._buffer, audio])

        # Process all available 512-sample chunks
        while len(self._buffer) >= self.WINDOW_SIZE_SAMPLES:
            chunk = self._buffer[: self.WINDOW_SIZE_SAMPLES]
            self._buffer = self._buffer[self.WINDOW_SIZE_SAMPLES :]
            self._infer_512(chunk)

        return self._last_speech_prob

    @property
    def current_speech_prob(self) -> float:
        """Get the most recent continuous speech probability."""
        return self._last_speech_prob
