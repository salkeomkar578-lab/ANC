"""
Audio File Regression Test for IGARD-Net.
Compares:
  - reference input: mew (1).wav
  - legacy output:   enhanced_mew.wav
  - new output:       enhanced_mew_v2.wav

Quantitatively checks:
  1. Excessive attenuation / Muting (< 0.20 gain during speech)
  2. Sudden gain drops & pumping
  3. Discontinuities / Ticking
  4. Clipping / Peak preservation
  5. Spectral centroid distortion (metallic speech)
  6. Speech Continuity Score
"""

import sys
import wave
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
from scipy.signal import resample_poly

from processing.artifact_detector import ArtifactDetector


def load_wav_mono_16k(filepath: Path) -> Tuple[np.ndarray, int]:
    """Loads a WAV file, converts to mono float64, and resamples to 16 kHz."""
    with wave.open(str(filepath), 'rb') as w:
        sr = w.getframerate()
        nch = w.getnchannels()
        nframes = w.getnframes()
        data = np.frombuffer(w.readframes(nframes), dtype=np.int16).astype(np.float64) / 32768.0
        if nch > 1:
            data = data.reshape(-1, nch)
            # Use channel 0 for primary voice
            mono = data[:, 0]
        else:
            mono = data

    if sr != 16000:
        # Rational resample to 16 kHz
        from math import gcd
        g = gcd(sr, 16000)
        up = 16000 // g
        down = sr // g
        mono = resample_poly(mono, up, down)

    return mono, 16000


def run_regression_test(
    orig_path: Path,
    legacy_path: Path,
    new_path: Path,
) -> Dict[str, Any]:
    orig_path = Path(orig_path)
    legacy_path = Path(legacy_path)
    new_path = Path(new_path)

    raw, sr = load_wav_mono_16k(orig_path)
    legacy, _ = load_wav_mono_16k(legacy_path)
    new_enh, _ = load_wav_mono_16k(new_path)

    detector = ArtifactDetector(sample_rate=16000, frame_size=256)
    
    # Extract AI speech confidence via Silero VAD
    from vad.silero_vad import SileroVAD
    vad = SileroVAD()
    bs = 256
    n = len(raw) // bs
    probs = []
    for i in range(n):
        chunk = raw[i*bs : (i+1)*bs]
        probs.append(vad.process_chunk(chunk))
    speech_probs = np.array(probs)

    legacy_results = detector.evaluate(raw, legacy, speech_probs=speech_probs)
    new_results = detector.evaluate(raw, new_enh, speech_probs=speech_probs)

    # Acceptance criteria checks (Priority: Speech Continuity > Intelligibility > No Artifacts)
    verdicts = {
        "speech_continuity": "PASS" if new_results["continuity_score"] >= 0.85 else "FAIL",
        "speech_intelligibility": "PASS" if new_results["mean_speech_gain"] >= 0.50 else "FAIL",
        "noise_suppression": "PASS" if (new_results.get("noise_suppression_db", 0.0) >= 4.0 or new_results["max_suppression_db"] >= 4.0) else "FAIL",
        "artifact_level": "PASS" if (new_results["muting_count"] == 0 and new_results["ticking_count"] == 0) else "FAIL",
        "real_time_stability": "PASS" if new_results["gain_variation"] < 0.08 else "FAIL",
    }

    all_passed = all(v == "PASS" for v in verdicts.values())

    return {
        "legacy": legacy_results,
        "new": new_results,
        "verdicts": verdicts,
        "all_passed": all_passed,
    }


if __name__ == "__main__":
    base_dir = Path(__file__).parent
    raw_file = base_dir / "raw_mew.wav"
    if not raw_file.exists():
        raw_file = Path(r"C:\Users\predator\Downloads\raw_mew.wav")
    if not raw_file.exists():
        raw_file = Path(r"C:\Users\predator\Downloads\mew (1).wav")

    v2_file = base_dir / "enhanced_mew_v2.wav"
    v3_file = base_dir / "enhanced_mew_v3.wav"

    print(f"Running Regression Comparison (V3 Speech-First):\n  Raw: {raw_file}\n  V2:  {v2_file}\n  V3:  {v3_file}\n")
    if not v3_file.exists():
        print(f"ERROR: {v3_file} has not been generated yet!")
        sys.exit(1)

    # Run Artifact Detector comparison
    if v2_file.exists():
        report = run_regression_test(raw_file, v2_file, v3_file)
        print("=== REGRESSION RESULTS (V2 vs V3) ===")
        print("\n--- Previous Enhanced Audio (V2) ---")
        for k, v in report["legacy"].items():
            print(f"  {k}: {v}")

        print("\n--- Speech-First Enhanced Audio (V3) ---")
        for k, v in report["new"].items():
            print(f"  {k}: {v}")

        print("\n--- Verdicts ---")
        for k, v in report["verdicts"].items():
            print(f"  {k}: {v}")

        print(f"\nOverall Status: {'PASSED' if report['all_passed'] else 'FAILED'}")

    # Run Full AudioQualityEvaluator
    from benchmarks.audio_quality import AudioQualityEvaluator
    raw_audio, sr = load_wav_mono_16k(raw_file)
    v3_audio, _ = load_wav_mono_16k(v3_file)

    evaluator = AudioQualityEvaluator(sample_rate=sr)
    metrics = evaluator.evaluate(raw_audio, v3_audio)
    print("\n" + evaluator.format_report(metrics))

