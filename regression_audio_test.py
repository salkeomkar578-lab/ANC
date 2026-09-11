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

from artifact_detector import ArtifactDetector


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
    
    legacy_results = detector.evaluate(raw, legacy)
    new_results = detector.evaluate(raw, new_enh)

    # Acceptance criteria checks
    verdicts = {
        "speech_continuity": "PASS" if new_results["continuity_score"] >= 0.85 else "FAIL",
        "speech_intelligibility": "PASS" if new_results["min_speech_gain"] >= 0.40 else "FAIL",
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
    orig_file = Path(r"C:\Users\predator\Downloads\mew (1).wav")
    if not orig_file.exists():
        orig_file = base_dir / "mew (1).wav"

    legacy_file = Path(r"C:\Users\predator\Downloads\enhanced_mew.wav")
    if not legacy_file.exists():
        legacy_file = base_dir / "enhanced_mew.wav"

    new_file = base_dir / "enhanced_mew_v2.wav"
    if not new_file.exists():
        new_file = Path(r"C:\Users\predator\Downloads\enhanced_mew_v2.wav")

    print(f"Running Regression Comparison:\n  Raw: {orig_file}\n  Legacy: {legacy_file}\n  New: {new_file}\n")
    if not new_file.exists():
        print(f"ERROR: {new_file} has not been generated yet!")
        sys.exit(1)

    report = run_regression_test(orig_file, legacy_file, new_file)
    print("=== REGRESSION RESULTS ===")
    print("\n--- Legacy Enhanced Audio ---")
    for k, v in report["legacy"].items():
        print(f"  {k}: {v}")

    print("\n--- Reconstructed Enhanced Audio (v2) ---")
    for k, v in report["new"].items():
        print(f"  {k}: {v}")

    print("\n--- Verdicts ---")
    for k, v in report["verdicts"].items():
        print(f"  {k}: {v}")

    print(f"\nOverall Status: {'PASSED' if report['all_passed'] else 'FAILED'}")
