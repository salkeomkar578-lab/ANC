"""
File Processing Throughput & RTF Benchmark for IGARD-Net.
Benchmarks batch file processing across 10s, 30s, 60s, and 300s (5 min) workloads.
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
from file_mode.processor import FileProcessor
from audio.audio_input import generate_synthetic_tactical_scenario

TEMP_BENCH_DIR = ROOT / "benchmarks" / "temp_audio"
TEMP_BENCH_DIR.mkdir(parents=True, exist_ok=True)


def create_test_wav(duration_s: float, path: Path, sample_rate: int = 16000) -> Path:
    _, prim, ref = generate_synthetic_tactical_scenario(
        duration_s=duration_s, sample_rate=sample_rate, quiet_lead_in_s=0.5
    )
    stereo = np.column_stack((prim, ref))
    sf.write(str(path), (stereo * 32767.0).astype(np.int16), sample_rate, subtype="PCM_16")
    return path


def benchmark_file_throughput(durations=(10.0, 30.0, 60.0, 300.0)):
    backend_mgr = BackendManager()
    state = SystemState()
    pipeline = IgardNetPipeline(state=state, backend=backend_mgr.backend)
    processor = FileProcessor(output_dir=TEMP_BENCH_DIR)

    results = []

    print("=" * 65)
    print("IGARD-Net File Mode Throughput Benchmark")
    print(f"Backend: {backend_mgr.backend.name} | Profile: {backend_mgr.profile}")
    print("=" * 65)

    for dur in durations:
        test_file = TEMP_BENCH_DIR / f"test_{int(dur)}s.wav"
        create_test_wav(dur, test_file)

        res = processor.process_file(test_file, pipeline, state=state)
        res["benchmark_dur_s"] = dur
        results.append(res)

        print(f"File Duration: {dur:5.1f}s | Processing Time: {res['processing_time_s']:6.3f}s | RTF: {res['rtf']:6.3f} | SNR Delta: {res['snr_improvement_db']:+.2f} dB", flush=True)
        
        # Clean up temp test wav
        test_file.unlink(missing_ok=True)
        out_f = TEMP_BENCH_DIR / res["output_file"]
        out_f.unlink(missing_ok=True)

    pipeline.close()
    return results


if __name__ == "__main__":
    benchmark_file_throughput([10.0, 30.0, 60.0])
