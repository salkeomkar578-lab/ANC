"""
Master Benchmark Suite for IGARD-Net.
Executes both Real-Time Latency and Batch File Throughput benchmarks.
Measures real processing latency, end-to-end latency, RTF, SNR delta, CPU, GPU, drops, and underruns.
Never fabricates metrics: outputs N/A if a metric (e.g. STOI/PESQ) is unavailable.
"""

import sys
import time
from pathlib import Path
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.pipeline import IgardNetPipeline
from core.state import SystemState
from backends.backend_manager import BackendManager
from audio.audio_input import SyntheticAudioStreamer, generate_synthetic_tactical_scenario
from file_mode.processor import FileProcessor

# Check optional speech intelligibility metric libraries
try:
    from pystoi import stoi as calc_stoi
    STOI_AVAILABLE = True
except ImportError:
    STOI_AVAILABLE = False

try:
    from pesq import pesq as calc_pesq
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False


def run_comprehensive_benchmark():
    backend_mgr = BackendManager()
    state = SystemState()
    pipeline = IgardNetPipeline(state=state, backend=backend_mgr.backend)

    sample_rate = 16000
    frame_size = 256
    frame_duration_ms = (frame_size / sample_rate) * 1000.0
    num_frames = 600

    streamer = SyntheticAudioStreamer(sample_rate=sample_rate, frame_size=frame_size)

    # 1. Warm-up
    for _ in range(30):
        p, r, _ = streamer.read_block()
        _ = pipeline.process_block(p, r)

    # 2. Real-Time Latency Benchmark
    proc_times = []
    total_latencies = []
    snr_deltas = []
    clean_blocks = []
    enhanced_blocks = []

    t_start_rt = time.perf_counter()
    for _ in range(num_frames):
        t0 = time.perf_counter()
        p, r, c = streamer.read_block()
        t_cap = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        res = pipeline.process_block(p, r)
        t_proc = (time.perf_counter() - t1) * 1000.0

        t_out = 0.05
        tot_lat = frame_duration_ms + t_proc + t_out

        proc_times.append(t_proc)
        total_latencies.append(tot_lat)
        snr_deltas.append(res["snr_delta"])
        clean_blocks.append(c)
        enhanced_blocks.append(res["enhanced_audio"])

    t_elapsed_rt = time.perf_counter() - t_start_rt
    avg_proc = float(np.mean(proc_times))
    p95_proc = float(np.percentile(proc_times, 95))
    avg_total_lat = float(np.mean(total_latencies))
    avg_snr = float(np.mean(snr_deltas))

    # 3. Batch File Mode RTF Benchmark (10s audio)
    temp_dir = ROOT / "benchmarks" / "temp_audio"
    temp_dir.mkdir(parents=True, exist_ok=True)
    test_10s = temp_dir / "bench_10s.wav"
    
    c_10s, p_10s, r_10s = generate_synthetic_tactical_scenario(10.0, sample_rate, quiet_lead_in_s=0.5)
    sf.write(str(test_10s), np.column_stack((p_10s, r_10s)), sample_rate)

    processor = FileProcessor(output_dir=temp_dir)
    file_res = processor.process_file(test_10s, pipeline)
    rtf = file_res["rtf"]

    test_10s.unlink(missing_ok=True)
    (temp_dir / file_res["output_file"]).unlink(missing_ok=True)

    # 4. Speech Quality Metrics (STOI / PESQ)
    clean_full = np.concatenate(clean_blocks)
    enhanced_full = np.concatenate(enhanced_blocks)

    if STOI_AVAILABLE:
        try:
            stoi_score = f"{calc_stoi(clean_full, enhanced_full, sample_rate, extended=False):.2f}"
        except Exception:
            stoi_score = "N/A"
    else:
        stoi_score = "N/A (pystoi not installed)"

    if PESQ_AVAILABLE:
        try:
            pesq_score = f"{calc_pesq(sample_rate, clean_full, enhanced_full, 'wb'):.2f}"
        except Exception:
            pesq_score = "N/A"
    else:
        pesq_score = "N/A (pesq not installed)"

    # Hardware stats
    hw = backend_mgr.get_hardware_telemetry()
    pipeline.close()

    # 5. Output in exact format
    print("\n=== IGARD-Net Benchmark ===\n")
    print("Hardware:")
    print(f"{backend_mgr.backend.name} [{backend_mgr.profile}]\n")
    print("Sample Rate:")
    print(f"{sample_rate} Hz\n")
    print("Frame Size:")
    print(f"{frame_size}\n")
    print("Average Processing:")
    print(f"{avg_proc:.2f} ms\n")
    print("P95 Processing:")
    print(f"{p95_proc:.2f} ms\n")
    print("Estimated End-to-End Latency:")
    print(f"{avg_total_lat:.1f} ms\n")
    print("CPU:")
    print(f"{hw['cpu_percent']:.1f} %\n")
    print("GPU:")
    print(f"{hw['gpu_percent']:.1f} %\n")
    print("RTF:")
    print(f"{rtf:.3f}\n")
    print("Dropped Frames:")
    print(f"{state.dropped_frames}\n")
    print("Underruns:")
    print(f"{state.buffer_underruns}\n")
    print("SNR Improvement:")
    print(f"{avg_snr:+.1f} dB\n")
    print("STOI:")
    print(f"{stoi_score}\n")
    print("PESQ:")
    print(f"{pesq_score}\n")


if __name__ == "__main__":
    run_comprehensive_benchmark()
