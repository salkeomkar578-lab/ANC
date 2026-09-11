"""
End-to-end test using SYNTHETIC audio, so the whole pipeline can be
proven to work today, before real mics or real recordings exist.

Generates:
  - a synthetic "speech-like" signal (formant-ish amplitude-modulated tones
    -- NOT real speech, just a stand-in with speech-like energy structure)
  - synthetic "steady" noise (an engine/rotor-like hum: a fundamental +
    harmonics)
  - synthetic "impulsive" noise bursts (short loud clicks, like gunfire)

Then builds:
  primary_mic   = speech + noise   (what a soldier's mic actually picks up)
  reference_mic = noise only, mixed with a touch of independent sensor
                  noise (a reference mic is never a *perfect* copy of the
                  noise reaching the primary mic -- this keeps the test
                  honest instead of artificially easy)

Runs it through the full IgardNetPipeline and reports:
  - SNR before cleaning
  - SNR after cleaning
  - how often each pipeline stage fired (classifier labels, fail-safe
    bypass rate)

IMPORTANT: this test uses synthetic signals, not real recorded speech or
real defense noise. It proves the pipeline runs correctly end-to-end and
is a development/debugging tool -- it is NOT a substitute for testing on
real recordings once your mics arrive. Do not quote these SNR numbers as
real-world performance results.
"""

import numpy as np
import soundfile as sf
from pipeline import IgardNetPipeline

SAMPLE_RATE = 16000
DURATION_S = 6.0


def make_synthetic_speech(n_samples, sr):
    t = np.arange(n_samples) / sr
    # a handful of "formant-like" tones with a slow amplitude envelope,
    # just to have something with speech-like energy variation over time
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 2.0 * t))
    speech = (
        0.5 * np.sin(2 * np.pi * 200 * t) +
        0.3 * np.sin(2 * np.pi * 700 * t) +
        0.2 * np.sin(2 * np.pi * 1200 * t)
    )
    return 0.3 * envelope * speech


def make_steady_noise(n_samples, sr):
    t = np.arange(n_samples) / sr
    fundamental = 90.0  # rough helicopter-rotor-ish frequency
    noise = (
        0.4 * np.sin(2 * np.pi * fundamental * t) +
        0.25 * np.sin(2 * np.pi * fundamental * 2 * t) +
        0.15 * np.sin(2 * np.pi * fundamental * 3 * t)
    )
    return noise


def add_impulsive_bursts(signal, sr, num_bursts=4, seed=42):
    rng = np.random.default_rng(seed)
    out = signal.copy()
    n = len(signal)
    burst_len = int(0.03 * sr)  # 30ms bursts
    for _ in range(num_bursts):
        start = rng.integers(0, n - burst_len)
        burst = rng.normal(0, 1.5, burst_len) * np.hanning(burst_len)
        out[start:start + burst_len] += burst
    return out


def snr_db(clean, estimate):
    noise = clean - estimate
    signal_power = np.mean(clean ** 2) + 1e-12
    noise_power = np.mean(noise ** 2) + 1e-12
    return 10 * np.log10(signal_power / noise_power)


def main():
    n_samples = int(SAMPLE_RATE * DURATION_S)

    speech = make_synthetic_speech(n_samples, SAMPLE_RATE)
    steady_noise = make_steady_noise(n_samples, SAMPLE_RATE)
    noise = add_impulsive_bursts(steady_noise, SAMPLE_RATE, num_bursts=5)

    # Prepend ~1s of near-silence to both mic channels, matching a
    # realistic deployment: the device is powered on in a relatively
    # quiet moment before entering the noisy environment. The Stage 0
    # noise-presence gate (noise_presence.py) needs this kind of quiet
    # moment to calibrate its baseline -- see that file's docstring for
    # why, and what happens if there isn't one.
    calibration_lead_in = int(SAMPLE_RATE * 1.0)
    rng_lead = np.random.default_rng(99)
    quiet_lead = rng_lead.normal(0, 0.002, calibration_lead_in)

    speech = np.concatenate([np.zeros(calibration_lead_in), speech])
    noise = np.concatenate([quiet_lead, noise])

    primary = speech + noise
    # reference mic: correlated with the noise, plus a bit of independent
    # sensor noise so it's not a cheating perfect copy
    rng = np.random.default_rng(1)
    reference = noise + rng.normal(0, 0.02, len(noise))

    # IMPORTANT: normalize all three signals (speech, primary, reference)
    # by ONE SHARED scale factor, not independently. Normalizing each
    # signal by its own peak would distort their relative proportions
    # (a loud impulsive burst in `primary` would shrink everything else
    # in it relative to standalone `speech`), which corrupts the SNR
    # comparison. A shared scale factor preserves true relative levels.
    shared_peak = max(np.max(np.abs(speech)), np.max(np.abs(primary)), np.max(np.abs(reference)))
    scale = 0.9 / (shared_peak + 1e-9)

    speech_n = speech * scale
    primary_n = primary * scale
    reference_n = reference * scale

    pipeline = IgardNetPipeline(sample_rate=SAMPLE_RATE)
    output, telemetry = pipeline.process_stream(primary_n, reference_n, block_size=512)

    sf.write("synthetic_clean_speech.wav", speech_n, SAMPLE_RATE)
    sf.write("synthetic_noisy_primary.wav", primary_n, SAMPLE_RATE)
    # the pipeline's output is on the same shared scale already, since
    # both its inputs were -- just guard against any small clipping
    output_safe = np.clip(output, -1.0, 1.0)
    sf.write("synthetic_igard_output.wav", output_safe, SAMPLE_RATE)

    before = snr_db(speech_n, primary_n)
    after = snr_db(speech_n, output_safe)

    labels = [t["noise_label"] for t in telemetry]
    used_cleanup = [t["used_cleanup_stage"] for t in telemetry]
    label_counts = {l: labels.count(l) for l in set(labels)}
    bypass_rate = 100.0 * (1 - sum(used_cleanup) / len(used_cleanup))

    print("=" * 60)
    print("IGARD-Net synthetic end-to-end test")
    print("=" * 60)
    print(f"Blocks processed:        {len(telemetry)}")
    print(f"Noise label counts:      {label_counts}")
    print(f"Fail-safe bypass rate:   {bypass_rate:.1f}% of blocks")
    print(f"SNR before cleaning:     {before:.2f} dB")
    print(f"SNR after cleaning:      {after:.2f} dB")
    print(f"SNR improvement:         {after - before:+.2f} dB")
    print()
    print("Wrote: synthetic_clean_speech.wav, synthetic_noisy_primary.wav, synthetic_igard_output.wav")
    print()
    print("NOTE: these numbers are from SYNTHETIC test audio, not real")
    print("recordings -- use them to confirm the pipeline runs correctly,")
    print("not as a claimed real-world performance result.")


if __name__ == "__main__":
    main()
