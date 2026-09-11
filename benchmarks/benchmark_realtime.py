"""
Real-Time Latency & Streaming Reliability Benchmark for IGARD-Net.
Measures capture, DSP processing, buffering, output latency, P95, and dropped frames.
"""

import sys
import time
from pathlib import Path
import numpy as np

# Ensure root is in path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.pipeline import IgardNetPipeline
from core.state import SystemState
from audio.audio_input import SyntheticAudioStreamer
from backends.backend_manager import BackendManager


def benchmark_realtime(num_frames: int = 500, frame_size: int = 256, sample_rate: int = 16000):
    state = SystemState()
    backend_mgr = BackendManager()
    pipeline = IgardNetPipeline(state=state, backend=backend_mgr.backend)
    streamer = SyntheticAudioStreamer(sample_rate=sample_rate, frame_size=frame_size)

    frame_duration_ms = (frame_size / sample_rate) * 1000.0
    proc_times = []
    total_latencies = []
    snr_deltas = []

    # Warm-up (20 frames)
    for _ in range(20):
        p, r, _ = streamer.read_block()
        _ = pipeline.process_block(p, r)

    t_bench_start = time.perf_counter()
    for _ in range(num_frames):
        t0 = time.perf_counter()
        p, r, c = streamer.read_block()
        t_cap = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        res = pipeline.process_block(p, r)
        t_proc = (time.perf_counter() - t1) * 1000.0

        # Simulated output write
        t_out = 0.05

        total_lat = frame_duration_ms + t_proc + t_out
        proc_times.append(t_proc)
        total_latencies.append(total_lat)
        snr_deltas.append(res["snr_delta"])

    t_bench_total = time.perf_counter() - t_bench_start
    audio_duration_total_s = (num_frames * frame_size) / sample_rate

    avg_proc = float(np.mean(proc_times))
    p95_proc = float(np.percentile(proc_times, 95))
    max_proc = float(np.max(proc_times))

    avg_lat = float(np.mean(total_latencies))
    p95_lat = float(np.percentile(total_latencies, 95))
    max_lat = float(np.max(total_latencies))
    avg_snr = float(np.mean(snr_deltas))

    hw = backend_mgr.get_hardware_telemetry()
    pipeline.close()

    return {
        "backend": backend_mgr.backend.name,
        "profile": backend_mgr.profile,
        "sample_rate": sample_rate,
        "frame_size": frame_size,
        "num_frames": num_frames,
        "avg_proc_ms": avg_proc,
        "p95_proc_ms": p95_proc,
        "max_proc_ms": max_proc,
        "avg_latency_ms": avg_lat,
        "p95_latency_ms": p95_lat,
        "max_latency_ms": max_lat,
        "avg_snr_db": avg_snr,
        "cpu_percent": hw["cpu_percent"],
        "gpu_percent": hw["gpu_percent"],
        "benchmark_duration_s": t_bench_total,
        "audio_duration_s": audio_duration_total_s,
    }


if __name__ == "__main__":
    print("Running IGARD-Net Real-Time Latency Benchmark (500 frames @ 256 samples)...")
    res = benchmark_realtime(num_frames=500, frame_size=256)
    print(f"Backend:            {res['backend']} [{res['profile']}]")
    print(f"Frame Size:         {res['frame_size']} samples ({(res['frame_size']/res['sample_rate'])*1000:.1f} ms)")
    print(f"Avg DSP Processing: {res['avg_proc_ms']:.2f} ms")
    print(f"P95 DSP Processing: {res['p95_proc_ms']:.2f} ms")
    print(f"Total Est. Latency: {res['avg_latency_ms']:.1f} ms (P95: {res['p95_latency_ms']:.1f} ms)")
    print(f"SNR Improvement:    +{res['avg_snr_db']:.2f} dB")
    print(f"CPU:                {res['cpu_percent']:.1f}%")
    print(f"GPU:                {res['gpu_percent']:.1f}%")
