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
from processing.mos_estimator import MOSEstimator


def decimate_for_ui(sig: np.ndarray, num_pts: int = 300) -> list:
    """Downsamples audio signal to compact list of floats for UI oscilloscope rendering."""
    if len(sig) <= num_pts:
        return [float(x) for x in sig]
    indices = np.linspace(0, len(sig) - 1, num_pts).astype(int)
    return [round(float(sig[i]), 4) for i in indices]


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
        self.mos_estimator = MOSEstimator(sample_rate=self.target_sr)

    def process_file(
        self,
        filepath: Path,
        pipeline: IgardNetPipeline,
        state: Optional[SystemState] = None,
        progress_callback: Optional[Callable[[float, str, int], None]] = None,
    ) -> Dict[str, Any]:
        t_start = time.perf_counter()
        filepath = Path(filepath)

        def report(prog: float, stage_msg: str, stage_idx: int):
            if state is not None:
                state.file_progress = prog
                state.file_status = stage_msg
            if progress_callback:
                try:
                    progress_callback(prog, stage_msg, stage_idx)
                except TypeError:
                    progress_callback(prog)

        # Stage 1: Load Audio & Channel Analysis
        report(5.0, "Ingesting Audio & Analyzing Channel Configuration...", 1)
        from file_mode.uploader import load_audio_universal
        data, orig_sr = load_audio_universal(filepath)
        orig_channels = data.ndim if data.ndim == 1 else data.shape[1]

        # Extract Primary and Reference Channels
        if data.ndim == 1:
            primary = data
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
            rms0 = float(np.sqrt(np.mean(ch0 ** 2))) + 1e-10
            rms1 = float(np.sqrt(np.mean(ch1 ** 2))) + 1e-10
            cross_corr = float(np.abs(np.mean(ch0 * ch1)) / (rms0 * rms1))

            if cross_corr > 0.35:
                # Stereo recording: in-phase speech on both channels.
                # Use ch0 as primary; diff_ref as ambient noise reference to prevent speech cancellation
                primary = ch0
                diff_ref = ch0 - ch1
                diff_rms = float(np.sqrt(np.mean(diff_ref ** 2))) + 1e-10
                reference = diff_ref * min(1.0, rms0 / (diff_rms + 1e-6) * 0.5)
            else:
                # True dual-mic setup: ch0 is primary mic, ch1 is reference mic
                primary = ch0
                reference = ch1

        # Stage 2: Fast Resample to target 16 kHz & AI Feature Extraction
        report(15.0, "AI Voice Identification & Acoustic Resampling (16 kHz)...", 2)
        if orig_sr != self.target_sr:
            primary = resample_audio(primary, orig_sr, self.target_sr)
            reference = resample_audio(reference, orig_sr, self.target_sr)

        total_samples = len(primary)
        audio_duration_s = total_samples / self.target_sr

        # Stage 3 & 4: Process Audio at Maximum Computational Throughput
        def stream_progress(frac: float, msg: str):
            cur_prog = 15.0 + frac * 68.0
            stage_idx = 3 if frac < 0.55 else 4
            stage_msg = f"AI Voice-Preserving Suppression ({int(frac*100)}%): {msg}"
            report(cur_prog, stage_msg, stage_idx)

        cleaned_output, telem_log = pipeline.process_stream(
            primary=primary,
            reference=reference,
            progress_callback=stream_progress,
        )
        speech_frame_count = sum(1 for t in telem_log if t.get("speech_probability", 0.0) >= 0.50)
        total_blocks = max(1, len(telem_log))

        t_end = time.perf_counter()
        proc_time_s = t_end - t_start
        rtf = proc_time_s / max(audio_duration_s, 1.0e-6)

        # Stage 5: Safe Clipping, Export, and Objective MOS Quality Evaluation
        report(85.0, "Exporting 16-bit Master WAV & Normalizing Dynamic Range...", 4)
        cleaned_safe = np.clip(cleaned_output, -1.0, 1.0)
        out_filename = f"enhanced_{filepath.stem}.wav"
        out_path = self.output_dir / out_filename
        sf.write(str(out_path), (cleaned_safe * 32767.0).astype(np.int16), self.target_sr, subtype="PCM_16")

        report(92.0, "Evaluating Objective MOS (ITU-T P.835 / P.800 Standard)...", 5)
        mos_eval = self.mos_estimator.evaluate_signals(
            raw_signal=primary,
            enhanced_signal=cleaned_safe,
            reference_noise=reference,
        )

        # Generate decimated waveforms for immediate visual comparison
        raw_wave = decimate_for_ui(primary, 300)
        enh_wave = decimate_for_ui(cleaned_safe, 300)

        # Export a browser-safe 16-bit PCM WAV of the raw audio into output_dir
        # so any uploaded format (MP3, FLAC, OGG, M4A) can be played directly by the browser!
        raw_out_filename = f"raw_{filepath.stem}.wav"
        raw_out_path = self.output_dir / raw_out_filename
        sf.write(str(raw_out_path), (np.clip(primary, -1.0, 1.0) * 32767.0).astype(np.int16), self.target_sr, subtype="PCM_16")

        report(100.0, "File Processing & Quality Audit Complete!", 5)

        return {
            "status": "success",
            "filename": filepath.name,
            "input_file": filepath.name,
            "raw_audio_file": raw_out_filename,
            "raw_audio_url": f"/api/audio/output/{raw_out_filename}",
            "cleaned_audio_url": f"/api/audio/output/{out_filename}",
            "output_file": out_filename,
            "output_path": str(out_path),
            "output_filename": out_filename,
            "sample_rate": self.target_sr,
            "original_sr": orig_sr,
            "channels": orig_channels,
            "duration_s": round(audio_duration_s, 2),
            "processing_time_s": round(proc_time_s, 3),
            "rtf": round(rtf, 4),
            "frames_processed": total_samples // self.block_size,
            "speech_presence_pct": round((speech_frame_count / max(1, total_blocks)) * 100.0, 1),
            # Visual Waveform Vectors for Dashboard
            "raw_waveform": raw_wave,
            "enhanced_waveform": enh_wave,
            # MOS Solution & Objective Quality Metrics
            "overall_mos": mos_eval["overall_mos"],
            "raw_mos": mos_eval["raw_mos"],
            "mos_gain": mos_eval["mos_gain"],
            "mos_rating": mos_eval["mos_rating"],
            "speech_intelligibility": mos_eval["speech_intelligibility"],
            "noise_suppression": mos_eval["noise_suppression"],
            "speech_preservation_score": mos_eval["speech_preservation_score"],
            "snr_improvement_db": mos_eval["snr_improvement_db"],
        }

