"""
The full IGARD-Net signal chain, block by block:

  primary mic, reference mic
        |
  [0] NoisePresenceGate         -> engage or clean passthrough?
        |
  [1] NoiseClassifier          -> label, confidence
  [1b] AccelerometerCrossCheck  -> adjusted confidence
        |
  [2] NLMSFilter                -> nlms_output   (tuned by [3])
  [3] QuantumInspiredTuner      -> retunes NLMS step size periodically
        |
  [4] ConfidenceGate            -> run cleanup, or bypass?
        |                    \\\\
  [5a] ResidualCleanup       [5b] fail-safe: pass nlms_output through
        |                    /
        final cleaned block
"""

import numpy as np

from nlms_filter import NLMSFilter
from quantum_tuner import QuantumInspiredTuner
from noise_classifier import NoiseClassifier
from confidence_gate import ConfidenceGate
from cleanup_stage import ResidualCleanup
from accel_sensor import MockAccelerometer, cross_check_confidence
from noise_presence import NoisePresenceGate
from spectrogram_helper import compute_fft_magnitudes, compute_rms_level, compute_snr_estimate


class IgardNetPipeline:
    def __init__(self, sample_rate=16000, num_taps=64,
                 retune_every_n_blocks=10, confidence_threshold=0.6,
                 accelerometer=None):
        self.sample_rate = sample_rate
        self.nlms = NLMSFilter(num_taps=num_taps, step_size=0.5)
        self.tuner = QuantumInspiredTuner()
        self.classifier = NoiseClassifier(sample_rate=sample_rate)
        self.gate = ConfidenceGate(threshold=confidence_threshold)
        self.cleanup = ResidualCleanup(sample_rate=sample_rate)
        self.presence_gate = NoisePresenceGate()
        # Cross-modal accelerometer confirmation (see accel_sensor.py).
        # Defaults to a mock backend -- no shock hardware exists yet, so
        # this defaults to "no shocks detected" unless a real sensor or a
        # scripted mock is passed in.
        self.accelerometer = accelerometer if accelerometer is not None else MockAccelerometer()

        self.num_taps = num_taps
        self.retune_every_n_blocks = retune_every_n_blocks
        self._block_count = 0

        # keep a short rolling history for the tuner to evaluate against
        self._primary_history = np.zeros(sample_rate // 2)  # ~0.5s window
        self._reference_history = np.zeros(sample_rate // 2)

        # Running SNR estimate (exponential moving average)
        self._snr_ema = 0.0

    def process_block(self, primary_block, reference_block):
        """
        primary_block, reference_block: 1D numpy float arrays, same length
        (e.g. 512 or 1024 samples -- one "frame" of audio).

        Returns a dict with the cleaned audio plus full telemetry for
        the live dashboard: noise type, confidence, shock score, stage
        status, waveform samples, FFT magnitudes, mic levels, SNR estimate,
        and which stage is currently active.
        """
        primary_block = np.asarray(primary_block, dtype=np.float64)
        reference_block = np.asarray(reference_block, dtype=np.float64)

        # Compute mic levels for meter display
        primary_level = compute_rms_level(primary_block)
        reference_level = compute_rms_level(reference_block)

        # Compute input FFT for "before" spectrogram
        primary_fft = compute_fft_magnitudes(primary_block, self.sample_rate)

        # --- Stage 0: is there actually any real background noise? ---
        noise_present, floor_estimate, block_energy = self.presence_gate.check(reference_block)
        if not noise_present:
            # Clean passthrough: don't run the classifier, filter, tuner,
            # or cleanup at all -- the input is already clean, and
            # running the full pipeline anyway would risk introducing
            # artifacts into good speech for no benefit.
            output_fft = compute_fft_magnitudes(primary_block, self.sample_rate)
            return {
                "audio": primary_block,
                "noise_label": "clean_passthrough",
                "confidence": 1.0,
                "shock_score": 0.0,
                "used_cleanup_stage": False,
                "nlms_step_size": self.nlms.step_size,
                "noise_present": False,
                "noise_floor_estimate": floor_estimate,
                # --- Extended telemetry for dashboard visuals ---
                "active_stage": 0,
                "stage_status": "Quiet — no noise detected, passing speech through untouched",
                "primary_level": primary_level,
                "reference_level": reference_level,
                "primary_fft": primary_fft["magnitudes"],
                "output_fft": output_fft["magnitudes"],
                "primary_samples": primary_block.tolist()[-256:],  # last 256 samples for waveform
                "output_samples": primary_block.tolist()[-256:],
                "estimated_snr": self._snr_ema,
                "block_count": self._block_count,
            }

        # --- Stage 1: classify ---
        label, confidence = self.classifier.classify(reference_block)

        # --- Stage 1b: cross-modal accelerometer confirmation ---
        # Only meaningful when the classifier claims "impulsive" -- this
        # is the false-positive fix for acoustic-only gunshot detection.
        shock_score = self.accelerometer.read_recent_shock_score()
        confidence = cross_check_confidence(label, confidence, shock_score)

        # Build stage status string for the dashboard
        if label == "impulsive":
            if shock_score > 0.5:
                stage_status = (
                    f"⚡ Impulsive noise detected ({confidence*100:.0f}% confidence) "
                    f"— accelerometer confirms real shock — engaging full cleanup"
                )
            else:
                stage_status = (
                    f"⚠ Impulsive sound detected ({confidence*100:.0f}% confidence) "
                    f"— accelerometer found no shock — likely false positive, using fail-safe"
                )
        elif label == "steady":
            stage_status = (
                f"🔊 Steady noise detected ({confidence*100:.0f}% confidence) "
                f"— adaptive filter engaged, quantum tuner optimizing live..."
            )
        else:
            stage_status = (
                f"Noise detected — classifying... ({label}, {confidence*100:.0f}% confidence)"
            )

        # --- Stage 2: NLMS filter ---
        nlms_out = self.nlms.process_block(primary_block, reference_block)

        # --- Stage 3: periodic quantum-inspired retuning ---
        self._primary_history = np.roll(self._primary_history, -len(primary_block))
        self._primary_history[-len(primary_block):] = primary_block
        self._reference_history = np.roll(self._reference_history, -len(reference_block))
        self._reference_history[-len(reference_block):] = reference_block

        self._block_count += 1
        retuned = False
        if self._block_count % self.retune_every_n_blocks == 0:
            new_step = self.tuner.retune(self._primary_history, self._reference_history, self.num_taps)
            self.nlms.set_step_size(new_step)
            retuned = True

        # --- Stage 4: confidence gate ---
        run_cleanup = self.gate.decide(confidence)

        # --- Stage 5: cleanup or fail-safe ---
        if run_cleanup:
            if label == "steady":
                # steady noise -> good moment to (re)calibrate the noise profile
                self.cleanup.update_noise_profile(reference_block)
            final_out = self.cleanup.clean(nlms_out)
            used_neural_cleanup = True
            active_stage = 5
            if label == "steady":
                stage_status += " Cleanup stage ON."
        else:
            final_out = nlms_out  # fail-safe: untouched NLMS output
            used_neural_cleanup = False
            active_stage = 4
            stage_status = (
                f"⚠ Low confidence ({confidence*100:.0f}%) — fail-safe bypass active, "
                f"passing adaptive-filter output through unmodified"
            )

        # Compute output FFT for "after" spectrogram
        output_fft = compute_fft_magnitudes(final_out, self.sample_rate)

        # Running SNR estimate (exponential moving average for smooth display)
        block_snr = compute_snr_estimate(final_out, primary_block)
        self._snr_ema = 0.85 * self._snr_ema + 0.15 * block_snr

        return {
            "audio": final_out,
            "noise_label": label,
            "confidence": confidence,
            "shock_score": shock_score,
            "used_cleanup_stage": used_neural_cleanup,
            "nlms_step_size": self.nlms.step_size,
            "noise_present": True,
            "noise_floor_estimate": floor_estimate,
            # --- Extended telemetry for dashboard visuals ---
            "active_stage": active_stage,
            "stage_status": stage_status,
            "primary_level": primary_level,
            "reference_level": reference_level,
            "primary_fft": primary_fft["magnitudes"],
            "output_fft": output_fft["magnitudes"],
            "primary_samples": primary_block.tolist()[-256:],
            "output_samples": final_out.tolist()[-256:],
            "estimated_snr": float(self._snr_ema),
            "block_count": self._block_count,
            "retuned_this_block": retuned,
        }

    def process_stream(self, primary, reference, block_size=512):
        """Convenience method: process a whole signal at once, block by block."""
        assert len(primary) == len(reference)
        n = len(primary)
        output = np.zeros(n, dtype=np.float64)
        telemetry = []

        for start in range(0, n, block_size):
            end = min(start + block_size, n)
            p_block = primary[start:end]
            r_block = reference[start:end]
            if len(p_block) < block_size:
                # pad the final short block, then trim back down
                pad = block_size - len(p_block)
                p_block = np.pad(p_block, (0, pad))
                r_block = np.pad(r_block, (0, pad))
                result = self.process_block(p_block, r_block)
                output[start:end] = result["audio"][: end - start]
            else:
                result = self.process_block(p_block, r_block)
                output[start:end] = result["audio"]
            telemetry.append({k: v for k, v in result.items() if k != "audio"})

        return output, telemetry
