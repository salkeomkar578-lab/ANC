"""
High-Throughput File Processor for IGARD-Net.
Processes recorded WAV files at maximum hardware throughput (RTF << 1.0).
Zero artificial sleep. Independent from real-time streaming engine.
Includes intelligent channel classification (stereo voice vs dual-mic ANC)
and seamless frame processing without dropped lead-in frames.
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
            # Estimate stationary noise floor from the lowest 10% energy frames
            frame_len = min(len(data), 512)
            n_fr = len(data) // frame_len
            if n_fr > 2:
                fr_rms = [np.sqrt(np.mean(data[i*frame_len:(i+1)*frame_len]**2)) for i in range(n_fr)]
                noise_floor_std = max(1e-4, float(np.percentile(fr_rms, 10)))
            else:
                noise_floor_std = 1e-3
            rng = np.random.default_rng(42)
            reference = rng.normal(0, noise_floor_std, len(data))
        else:
            ch0 = data[:, 0]
            ch1 = data[:, 1]
            # Check cross-correlation between channels
            rms0 = float(np.sqrt(np.mean(ch0 ** 2))) + 1e-10
            rms1 = float(np.sqrt(np.mean(ch1 ** 2))) + 1e-10
            cross_corr = float(np.abs(np.mean(ch0 * ch1)) / (rms0 * rms1))

            if cross_corr > 0.35:
                # Stereo recording containing common speech on both channels.
                # In-phase target voice appears on both channels.
                # Use channel 0 as primary, and the difference (out-of-phase ambient noise) as reference,
                # avoiding cancellation of in-phase target speech!
                primary = ch0
                diff_ref = ch0 - ch1
                # Ensure diff_ref does not contain residual speech
                diff_rms = float(np.sqrt(np.mean(diff_ref ** 2))) + 1e-10
                reference = diff_ref * min(1.0, rms0 / (diff_rms + 1e-6) * 0.5)
            else:
                # True dual-mic setup: channel 0 is primary voice mic, channel 1 is ambient noise mic
                primary = ch0
                reference = ch1

        # 3. Fast Resample to target 16 kHz if necessary
        if orig_sr != self.target_sr:
            primary = resample_audio(primary, orig_sr, self.target_sr)
            reference = resample_audio(reference, orig_sr, self.target_sr)

        total_samples = len(primary)
        audio_duration_s = total_samples / self.target_sr

        # 4. Process Audio at Maximum Computational Throughput
        cleaned_output = np.empty(total_samples, dtype=np.float64)
        
        # Reset presence gate baseline for the file
        pipeline.presence_gate.recalibrate()
        pipeline.speech_detector.reset()
        pipeline.speech_protection.reset()
        pipeline.cleanup.reset()

        for start in range(0, total_samples, self.block_size):
            end = min(start + self.block_size, total_samples)
            p_block = primary[start:end]
            r_block = reference[start:end]

            if len(p_block) < self.block_size:
                pad = self.block_size - len(p_block)
                p_pad = np.pad(p_block, (0, pad))
                r_pad = np.pad(r_block, (0, pad))
                res = pipeline.process_block(p_pad, r_pad)
                cleaned_output[start:end] = res["audio"][: end - start]
            else:
                res = pipeline.process_block(p_block, r_block)
                cleaned_output[start:end] = res["audio"]

            progress = min(100.0, (end / total_samples) * 100.0)
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

        return {
            "status": "success",
            "filename": filepath.name,
            "output_file": str(out_path),
            "output_path": str(out_path),
            "output_filename": out_filename,
            "sample_rate": self.target_sr,
            "original_sr": orig_sr,
            "channels": orig_channels,
            "duration_s": round(audio_duration_s, 2),
            "processing_time_s": round(proc_time_s, 3),
            "rtf": round(rtf, 4),
            "frames_processed": total_samples // self.block_size,
        }
