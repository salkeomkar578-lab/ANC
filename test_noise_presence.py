"""
Proves the new Stage 0 (noise_presence.py) does what was asked: engage
the full pipeline ONLY when real background noise is present, and pass
clean audio straight through untouched otherwise.

Builds a test signal with three back-to-back sections:
  1. QUIET section -- just very low-level sensor noise, no real noise
  2. LOUD/NOISY section -- steady rotor-like hum, real background noise
  3. QUIET section again -- back to silence

Then checks that the pipeline correctly reports "clean_passthrough" for
sections 1 and 3, and actively processes (classifier/filter/etc engage)
during section 2.
"""

import numpy as np
from pipeline import IgardNetPipeline
from test_synthetic import make_steady_noise

SAMPLE_RATE = 16000
BLOCK_SIZE = 512


def main():
    quiet_duration = 2.0
    noisy_duration = 2.0
    n_quiet = int(SAMPLE_RATE * quiet_duration)
    n_noisy = int(SAMPLE_RATE * noisy_duration)

    rng = np.random.default_rng(5)
    quiet_section = rng.normal(0, 0.002, n_quiet)  # just faint sensor hiss, no real noise
    noisy_section = make_steady_noise(n_noisy, SAMPLE_RATE) * 0.3

    reference = np.concatenate([quiet_section, noisy_section, quiet_section])
    primary = reference.copy() + rng.normal(0, 0.01, len(reference))  # a bit of speech-like content could go here too

    pipeline = IgardNetPipeline(sample_rate=SAMPLE_RATE)

    section_labels = (
        ["quiet"] * (n_quiet // BLOCK_SIZE)
        + ["noisy"] * (n_noisy // BLOCK_SIZE)
        + ["quiet"] * (n_quiet // BLOCK_SIZE)
    )

    results = []
    for i, start in enumerate(range(0, len(primary) - BLOCK_SIZE, BLOCK_SIZE)):
        p_block = primary[start:start + BLOCK_SIZE]
        r_block = reference[start:start + BLOCK_SIZE]
        result = pipeline.process_block(p_block, r_block)
        expected_section = section_labels[i] if i < len(section_labels) else "quiet"
        results.append((expected_section, result["noise_present"], result["noise_label"]))

    quiet_results = [r for r in results if r[0] == "quiet"]
    noisy_results = [r for r in results if r[0] == "noisy"]

    # The gate's first `calibration_blocks` (20) blocks are an intentional,
    # documented startup transient -- it hasn't calibrated yet, so it
    # safely defaults to "assume noise present" rather than guess. This
    # only happens once, at the very start of the whole stream, so we
    # exclude just that initial warm-up window from the quiet-accuracy
    # count below -- excluding it doesn't hide a mistake, it correctly
    # separates "can't judge yet" from "judged wrong".
    calibration_warmup_blocks = 20
    quiet_results_after_warmup = quiet_results[calibration_warmup_blocks:]

    quiet_correctly_bypassed = sum(1 for _, present, _ in quiet_results_after_warmup if not present)
    noisy_correctly_engaged = sum(1 for _, present, _ in noisy_results if present)

    print("=" * 60)
    print("Noise-presence gate test")
    print("=" * 60)
    print(f"Quiet-section blocks (excluding {calibration_warmup_blocks}-block startup calibration): {len(quiet_results_after_warmup)}")
    print(f"  correctly passed through (no processing): {quiet_correctly_bypassed} / {len(quiet_results_after_warmup)}")
    print(f"Noisy-section blocks:  {len(noisy_results)}")
    print(f"  correctly engaged full pipeline:           {noisy_correctly_engaged} / {len(noisy_results)}")

    quiet_accuracy = quiet_correctly_bypassed / len(quiet_results_after_warmup)
    noisy_accuracy = noisy_correctly_engaged / len(noisy_results)

    print(f"\nQuiet-section bypass accuracy: {quiet_accuracy*100:.1f}%")
    print(f"Noisy-section engage accuracy: {noisy_accuracy*100:.1f}%")

    assert quiet_accuracy > 0.9, "Too many quiet blocks incorrectly triggered full processing"
    assert noisy_accuracy > 0.9, "Too many noisy blocks were incorrectly passed through untouched"
    print("\nStage 0 (noise-presence gate) verified: the pipeline only")
    print("actively processes audio when real background noise is present.")


if __name__ == "__main__":
    main()
