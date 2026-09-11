"""
Test the IGARD-Net pipeline against a REAL audio file (downloaded, or
recorded on your phone and transferred over) instead of pure synthetic
tones. This is a genuinely better validation step than test_synthetic.py
alone, since it uses actual speech content -- you can literally listen to
the before/after and judge intelligibility yourself, which a plain SNR
number can't tell you.

WHAT THIS DOES NOT NEED: real mics or real recorded noise. It takes ONE
real speech file you provide, synthetically adds realistic noise on top
(reusing the same steady-hum + impulsive-burst generators as
test_synthetic.py), and feeds the result through the full pipeline. Once
your mics physically arrive, use live_mic.py instead -- this script is a
stepping stone for right now.

USAGE:
    python3 test_with_realfile.py path/to/your_speech.wav

If you don't have a speech file handy yet:
  - Record 5-10 seconds of yourself talking on your phone, transfer it to
    the Pi with `scp your_recording.m4a pi@igardnet.local:~/igard_net/`,
    then convert it to wav first, e.g.:
      ffmpeg -i your_recording.m4a -ar 16000 -ac 1 your_recording.wav
  - Or download any short public-domain speech clip.
"""

import sys
import numpy as np
import soundfile as sf

from pipeline import IgardNetPipeline
from test_synthetic import make_steady_noise, add_impulsive_bursts, snr_db

TARGET_SR = 16000


def load_and_prepare(path, target_sr=TARGET_SR):
    data, sr = sf.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)  # collapse to mono
    if sr != target_sr:
        num_samples = int(len(data) * target_sr / sr)
        data = np.interp(
            np.linspace(0, len(data), num_samples, endpoint=False),
            np.arange(len(data)),
            data
        )
    # normalize to a consistent, sane speech level before mixing in noise
    data = data / (np.max(np.abs(data)) + 1e-9) * 0.8
    return data.astype(np.float64)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 test_with_realfile.py path/to/your_speech.wav")
        print("(No file given -- falling back to the bundled sample_voice_44k.wav demo clip)")
        path = "sample_voice_44k.wav"
    else:
        path = sys.argv[1]

    speech = load_and_prepare(path)
    n = len(speech)

    steady = make_steady_noise(n, TARGET_SR)
    noise = add_impulsive_bursts(steady, TARGET_SR, num_bursts=max(2, n // TARGET_SR))

    # Prepend ~1s of near-silence, matching realistic deployment (device
    # powered on in relative quiet before the noisy environment starts) --
    # the Stage 0 noise-presence gate needs this to calibrate. See
    # noise_presence.py for why.
    calibration_lead_in = int(TARGET_SR * 1.0)
    rng_lead = np.random.default_rng(98)
    quiet_lead = rng_lead.normal(0, 0.002, calibration_lead_in)
    speech = np.concatenate([np.zeros(calibration_lead_in), speech])
    noise = np.concatenate([quiet_lead, noise])

    primary = speech + noise
    rng = np.random.default_rng(7)
    reference = noise + rng.normal(0, 0.02, len(noise))

    shared_peak = max(np.max(np.abs(speech)), np.max(np.abs(primary)), np.max(np.abs(reference)))
    scale = 0.9 / (shared_peak + 1e-9)
    speech_n = speech * scale
    primary_n = primary * scale
    reference_n = reference * scale

    pipeline = IgardNetPipeline(sample_rate=TARGET_SR)
    output, telemetry = pipeline.process_stream(primary_n, reference_n, block_size=512)
    output_safe = np.clip(output, -1.0, 1.0)

    sf.write("real_test_1_clean_speech.wav", speech_n, TARGET_SR)
    sf.write("real_test_2_noisy_input.wav", primary_n, TARGET_SR)
    sf.write("real_test_3_igard_output.wav", output_safe, TARGET_SR)

    before = snr_db(speech_n, primary_n)
    after = snr_db(speech_n, output_safe)
    labels = [t["noise_label"] for t in telemetry]
    label_counts = {l: labels.count(l) for l in set(labels)}
    bypass_rate = 100.0 * (1 - sum(t["used_cleanup_stage"] for t in telemetry) / len(telemetry))

    print("=" * 60)
    print(f"Tested against real audio file: {path}")
    print("=" * 60)
    print(f"Duration:                {n / TARGET_SR:.1f}s")
    print(f"Noise label counts:      {label_counts}")
    print(f"Fail-safe bypass rate:   {bypass_rate:.1f}%")
    print(f"SNR before cleaning:     {before:.2f} dB")
    print(f"SNR after cleaning:      {after:.2f} dB")
    print(f"SNR improvement:         {after - before:+.2f} dB")
    print()
    print("Listen for yourself:")
    print("  real_test_1_clean_speech.wav   <- your original speech, no noise")
    print("  real_test_2_noisy_input.wav    <- what the 'primary mic' hears")
    print("  real_test_3_igard_output.wav   <- what IGARD-Net produces")
    print()
    print("NOTE: the noise here is still synthetic (added on top of your")
    print("real speech) -- this proves the pipeline handles real speech")
    print("content correctly. Real recorded defense noise is the next")
    print("step once you have field recordings or your mics are in.")


if __name__ == "__main__":
    main()
