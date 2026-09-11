"""
Dedicated Real-Time Audio Engine Thread.
Prioritizes: Audio Responsiveness > Speech Quality > Visualization > Optimization.
Operates on small bounded frames (e.g. 256 samples = 16ms @ 16 kHz) with strict latency bounds.
"""

import threading
import time
from typing import Optional, Tuple
import numpy as np

from core.pipeline import IgardNetPipeline
from core.state import SystemState
from audio.audio_input import SyntheticAudioStreamer
from audio.audio_output import AudioOutputSink
from core.telemetry import TelemetryDispatcher


class RealtimeAudioEngine:
    def __init__(
        self,
        pipeline: IgardNetPipeline,
        state: SystemState,
        telemetry_dispatcher: TelemetryDispatcher,
        output_sink: AudioOutputSink,
        frame_size: int = 256,
        sample_rate: int = 16000,
    ):
        self.pipeline = pipeline
        self.state = state
        self.telemetry_dispatcher = telemetry_dispatcher
        self.output_sink = output_sink
        self.frame_size = frame_size
        self.sample_rate = sample_rate
        self.frame_duration_s = frame_size / sample_rate

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._is_running = False

        # Input source (defaults to synthetic tactical scenario generator)
        self.input_streamer = SyntheticAudioStreamer(sample_rate=sample_rate, frame_size=frame_size)

    def start(self):
        if self._is_running:
            return
        self._stop_event.clear()
        self._is_running = True
        self.state.is_running = True
        self.state.mode = "realtime"
        
        self._thread = threading.Thread(target=self._audio_loop, daemon=True, name="Realtime_Audio_Thread")
        self._thread.start()
        print(f"[RealtimeEngine] Started audio thread (Frame: {self.frame_size} samples = {self.frame_duration_s*1000:.1f}ms @ {self.sample_rate}Hz).")

    def stop(self):
        if not self._is_running:
            return
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._is_running = False
        self.state.is_running = False
        self.state.mode = "idle"
        print("[RealtimeEngine] Stopped audio thread.")

    def _audio_loop(self):
        # Precise pace timing
        next_tick = time.perf_counter()

        while not self._stop_event.is_set():
            t0 = time.perf_counter()

            # 1. Capture Frame (synthetic or hardware)
            p_block, r_block, _ = self.input_streamer.read_block()
            t_cap = (time.perf_counter() - t0) * 1000.0

            # 2. Process DSP Frame
            t1 = time.perf_counter()
            result = self.pipeline.process_block(p_block, r_block)
            t_proc = (time.perf_counter() - t1) * 1000.0

            # 3. Output Playback & Web Audio Stream
            t2 = time.perf_counter()
            self.output_sink.write(result["audio"])
            t_out = (time.perf_counter() - t2) * 1000.0

            # 4. Latency Accounting (Real measured metrics)
            frame_buffer_latency_ms = self.frame_duration_s * 1000.0
            total_est_latency_ms = frame_buffer_latency_ms + t_proc + t_out
            self.state.record_latency(
                capture_ms=t_cap,
                process_ms=t_proc,
                queue_ms=frame_buffer_latency_ms,
                output_ms=t_out,
            )

            # 5. Non-blocking Telemetry Dispatch
            self.telemetry_dispatcher.push_telemetry(result)

            # 6. Pace timing (ensure continuous real-time cadence)
            next_tick += self.frame_duration_s
            sleep_time = next_tick - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # Running behind real-time: reset tick and account dropped frame
                if sleep_time < -0.05:
                    next_tick = time.perf_counter()
                    self.state.dropped_frames += 1
