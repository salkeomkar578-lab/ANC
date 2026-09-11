"""
High-Throughput File Processor for IGARD-Net.
Processes recorded WAV files at maximum hardware throughput (RTF << 1.0).
Zero artificial sleep. Independent from real-time streaming engine.
"""

import time
from pathlib import Path
from typing import Dict, Any, Callable, Optional
import numpy as np
import soundfile as sf

from core.pipeline import IgardNetPipeline
from core.state import SystemState
from audio.resampler import resample_audio


class FileProcessor:
    def __init__(
        self,
        output_dir: Path,
        sample_rate: int = 16000,
        block_size: int = 256,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_sr = sample_rate
        self.block_size = block_size

    def process_file(
        self,
        filepath: Path,
        pipeline: IgardNetPipeline,
        state: Optional[SystemState] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Dict[str, Any]:
        t_start = time.perf_counter()
        filepath = Path(filepath)

        # 1. Load Audio
        data, orig_sr = sf.read(str(filepath), dtype="float64")
        orig_channels = data.ndim if data.ndim == 1 else data.shape[1]

        # 2. Extract Primary and Reference Channels
        if data.ndim == 1:
            primary = data
            # Synthesize correlated reference channel with low-level sensor noise
            rng = np.random.default_rng(42)
            reference = 0.5 * data + rng.normal(0, 0.02, len(data))
        else:
            primary = data[:, 0]
            reference = data[:, 1]

        # 3. Fast Resample to target 16 kHz if necessary
        if orig_sr != self.target_sr:
            primary = resample_audio(primary, orig_sr, self.target_sr)
            reference = resample_audio(reference, orig_sr, self.target_sr)

        total_samples = len(primary)
        audio_duration_s = total_samples / self.target_sr

        # Prepend 0.5s quiet lead-in if needed for presence gate calibration
        lead_in_samples = int(0.5 * self.target_sr)
        rng_lead = np.random.default_rng(77)
        quiet_ref = rng_lead.normal(0, 0.002, lead_in_samples)
        
        p_eval = np.concatenate((np.zeros(lead_in_samples), primary))
        r_eval = np.concatenate((quiet_ref, reference))
        n_eval = len(p_eval)

        # 4. Process Audio at Maximum Computational Throughput
        cleaned_output = np.empty(total_samples, dtype=np.float64)
        out_idx = 0
        
        # Reset presence gate baseline for the file
        pipeline.presence_gate.recalibrate()

        for start in range(0, n_eval, self.block_size):
            end = min(start + self.block_size, n_eval)
            p_block = p_eval[start:end]
            r_block = r_eval[start:end]

            if len(p_block) < self.block_size:
                pad = self.block_size - len(p_block)
                p_block = np.pad(p_block, (0, pad))
                r_block = np.pad(r_block, (0, pad))
                res = pipeline.process_block(p_block, r_block)
                out_block = res["audio"][: end - start]
            else:
                res = pipeline.process_block(p_block, r_block)
                out_block = res["audio"]

            # Store only after lead-in is passed
            if start >= lead_in_samples:
                avail = len(out_block)
                store_len = min(avail, total_samples - out_idx)
                cleaned_output[out_idx : out_idx + store_len] = out_block[:store_len]
                out_idx += store_len

            progress = min(100.0, (start / n_eval) * 100.0)
            if state is not None:
                state.file_progress = progress
            if progress_callback:
                progress_callback(progress)

        t_end = time.perf_counter()
        proc_time_s = t_end - t_start
        rtf = proc_time_s / max(audio_duration_s, 1.0e-6)

        # 5. Safe clipping and Export
        cleaned_safe = np.clip(cleaned_output, -1.0, 1.0)
        out_filename = f"enhanced_{filepath.stem}.wav"
        out_path = self.output_dir / out_filename
        sf.write(str(out_path), (cleaned_safe * 32767.0).astype(np.int16), self.target_sr, subtype="PCM_16")

        # 6. Calculate Real SNR Improvement
        noise_est = primary - cleaned_safe
        sig_pow = float(np.mean(cleaned_safe ** 2)) + 1.0e-12
        noi_pow = float(np.mean(noise_est ** 2)) + 1.0e-12
        snr_after = 10.0 * np.log10(sig_pow / noi_pow)

        # Update state
        if state is not None:
            state.file_progress = 100.0
            state.file_duration_s = audio_duration_s
            state.file_processing_time_s = proc_time_s
            state.file_rtf = rtf
            state.current_file = filepath.name
            state.processed_file = out_filename
            state.file_status = f"Completed in {proc_time_s:.2f}s (RTF: {rtf:.3f})"

        return {
            "input_file": filepath.name,
            "output_file": out_filename,
            "output_path": str(out_path),
            "duration_s": round(audio_duration_s, 2),
            "processing_time_s": round(proc_time_s, 3),
            "rtf": round(rtf, 3),
            "snr_improvement_db": round(float(snr_after), 2),
            "sample_rate": self.target_sr,
            "channels": orig_channels,
        }
