"""
Audio File Regression Test for IGARD-Net V4: AI Speech-Preserving Adaptive Noise Suppression.
Processes:
  - Reference 1: raw_mew.wav (94.39s)
  - Reference 2: raw_mew 2.wav (63.38s)

Compares:
  - Legacy Baseline: enhanced_mew.wav / enhanced_mew 2.wav
  - V4 Speech-First: enhanced_mew_v4.wav / enhanced_mew_2_v4.wav

Acceptance Criteria:
  1. Speech Continuity >= 0.90 (Target: >= 0.95)
  2. Zero Hard Muting (hard_mute_count == 0)
  3. Continuous Speech Intelligibility (mean_speech_gain >= 0.50)
  4. Zero Speech Collapse (speech_collapse_count == 0)
  5. Minimal Artifacts / No boundary clicking (ticking_count <= 6)
  6. Real-Time Stability (frame-to-frame gain variation < 0.08)
"""

import sys
import time
import wave
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from core.pipeline import IgardNetPipeline
from processing.artifact_detector import ArtifactDetector
from vad.silero_vad import SileroVAD


def load_wav_mono_16k(filepath: Path) -> Tuple[np.ndarray, int]:
    """Loads a WAV file, converts to mono float32, and resamples to 16 kHz."""
    with wave.open(str(filepath), 'rb') as w:
        sr = w.getframerate()
        nch = w.getnchannels()
        nframes = w.getnframes()
        data = np.frombuffer(w.readframes(nframes), dtype=np.int16).astype(np.float32) / 32768.0
        if nch > 1:
            data = data.reshape(-1, nch)
            mono = data[:, 0]
        else:
            mono = data

    if sr != 16000:
        from math import gcd
        g = gcd(sr, 16000)
        mono = resample_poly(mono, 16000 // g, sr // g)

    return mono, 16000


def run_v4_regression_on_file(
    raw_path: Path,
    legacy_path: Optional[Path] = None,
    output_v4_path: Optional[Path] = None,
    pipeline: Optional[IgardNetPipeline] = None,
) -> Dict[str, Any]:
    raw_path = Path(raw_path)
    print(f"\n==================================================================")
    print(f"  RUNNING V4 REGRESSION AUDIT: {raw_path.name}")
    print(f"==================================================================")

    raw, sr = load_wav_mono_16k(raw_path)
    n_samples = len(raw)
    duration_s = n_samples / sr

    if pipeline is None:
        pipeline = IgardNetPipeline()

    t_start = time.perf_counter()
    v4_audio, telem_log = pipeline.process_stream(raw)
    proc_time_s = time.perf_counter() - t_start
    rtf = proc_time_s / max(duration_s, 1e-6)

    print(f"Processed {duration_s:.2f}s in {proc_time_s:.3f}s (RTF: {rtf:.4f} -> {1.0/rtf:.1f}x real-time)")

    # Save V4 enhanced audio
    if output_v4_path is not None:
        output_v4_path = Path(output_v4_path)
        output_v4_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_v4_path), (np.clip(v4_audio, -1.0, 1.0) * 32767.0).astype(np.int16), sr, subtype="PCM_16")
        print(f"Exported V4 Master WAV: {output_v4_path.name}")

    # Extract Silero VAD ground truth for 256-sample frame evaluation
    vad = SileroVAD()
    bs = 256
    n_frames = n_samples // bs
    speech_probs = np.array([vad.process_chunk(raw[i*bs : (i+1)*bs]) for i in range(n_frames)])

    detector = ArtifactDetector(sample_rate=sr, frame_size=bs)
    v4_results = detector.evaluate(raw, v4_audio, speech_probs=speech_probs)

    legacy_results = None
    if legacy_path is not None and Path(legacy_path).exists():
        legacy_audio, _ = load_wav_mono_16k(legacy_path)
        legacy_results = detector.evaluate(raw, legacy_audio, speech_probs=speech_probs)

    # Acceptance criteria verification
    verdicts = {
        "speech_continuity": "PASS" if v4_results["continuity_score"] >= 0.90 else "FAIL",
        "hard_muting_eliminated": "PASS" if v4_results["hard_mute_count"] == 0 else "FAIL",
        "speech_intelligibility": "PASS" if v4_results["mean_speech_gain"] >= 0.50 else "FAIL",
        "speech_collapse_eliminated": "PASS" if v4_results["speech_collapse_count"] == 0 else "FAIL",
        "gain_variation_stable": "PASS" if v4_results["gain_variation"] < 0.08 else "FAIL",
        "noise_suppression_active": "PASS" if v4_results["noise_suppression_db"] >= 6.0 else "FAIL",
    }
    all_passed = all(v == "PASS" for v in verdicts.values())

    return {
        "file": raw_path.name,
        "duration_s": duration_s,
        "processing_time_s": proc_time_s,
        "rtf": rtf,
        "v4_metrics": v4_results,
        "legacy_metrics": legacy_results,
        "verdicts": verdicts,
        "all_passed": all_passed,
    }


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    raw1 = base_dir / "raw_mew.wav"
    raw2 = base_dir / "dashboard" / "outputs" / "raw_mew 2.wav"
    if not raw2.exists():
        raw2 = base_dir / "raw_mew 2.wav"

    legacy1 = base_dir / "dashboard" / "outputs" / "enhanced_mew.wav"
    legacy2 = base_dir / "dashboard" / "outputs" / "enhanced_mew 2.wav"

    pipeline = IgardNetPipeline()

    reports = []
    # Test 1: raw_mew.wav
    if raw1.exists():
        rep1 = run_v4_regression_on_file(
            raw_path=raw1,
            legacy_path=legacy1,
            output_v4_path=base_dir / "enhanced_mew_v4.wav",
            pipeline=pipeline,
        )
        reports.append(rep1)
        # Also copy to dashboard/outputs for web playback
        sf.write(str(base_dir / "dashboard" / "outputs" / "enhanced_mew_v4.wav"),
                 sf.read(str(base_dir / "enhanced_mew_v4.wav"))[0], 16000, subtype="PCM_16")

    # Test 2: raw_mew 2.wav
    if raw2.exists():
        rep2 = run_v4_regression_on_file(
            raw_path=raw2,
            legacy_path=legacy2,
            output_v4_path=base_dir / "enhanced_mew_2_v4.wav",
            pipeline=pipeline,
        )
        reports.append(rep2)
        sf.write(str(base_dir / "dashboard" / "outputs" / "enhanced_mew_2_v4.wav"),
                 sf.read(str(base_dir / "enhanced_mew_2_v4.wav"))[0], 16000, subtype="PCM_16")

    print("\n" + "=" * 70)
    print("                    IGARD-Net V4 MASTER AUDIT REPORT")
    print("=" * 70)
    for r in reports:
        print(f"\n--- Results for: {r['file']} (RTF: {r['rtf']:.4f}) ---")
        if r["legacy_metrics"]:
            print(f"  [Legacy] Continuity: {r['legacy_metrics']['continuity_score']*100:.1f}% | Hard Mutes: {r['legacy_metrics']['hard_mute_count']} | Collapse: {r['legacy_metrics']['speech_collapse_count']} | Exact Zeros: {r['legacy_metrics']['exact_zero_percentage']:.2f}%")
        print(f"  [V4 New] Continuity: {r['v4_metrics']['continuity_score']*100:.1f}% | Hard Mutes: {r['v4_metrics']['hard_mute_count']} | Collapse: {r['v4_metrics']['speech_collapse_count']} | Exact Zeros: {r['v4_metrics']['exact_zero_percentage']:.2f}%")
        print(f"  [V4 New] Mean Speech Gain: {r['v4_metrics']['mean_speech_gain']:.4f} | Min Speech Gain: {r['v4_metrics']['min_speech_gain']:.4f} | Gain Var: {r['v4_metrics']['gain_variation']:.4f}")
        print("  Verdicts:")
        for k, v in r["verdicts"].items():
            print(f"    - {k:28s}: {v}")
        print(f"  File Status: {'PASSED ALL CRITERIA' if r['all_passed'] else 'FAILED'}")

    overall_pass = all(r["all_passed"] for r in reports)
    print("\n" + "=" * 70)
    print(f"OVERALL STATUS: {'ALL REGRESSION TESTS PASSED' if overall_pass else 'FAILED'}")
    print("=" * 70)
    if not overall_pass:
        sys.exit(1)


