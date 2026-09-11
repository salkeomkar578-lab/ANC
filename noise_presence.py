"""
Stage 0 of IGARD-Net (new): noise-presence gate.

WHY THIS EXISTS (direct requirement): the system should only actively
process audio when there is REAL background noise to remove. If the
input is already clean speech (quiet environment, no gunfire/rotor/
vehicle noise), running it through classifiers and filters anyway risks
introducing artifacts into perfectly good speech, wastes CPU on a
Raspberry Pi 4 for no benefit, and is dishonest about what the system is
actually doing. This stage decides, before anything else runs, whether
the rest of the pipeline should engage at all.

METHOD: adaptive noise-floor tracking on the reference mic (the mic
dedicated to sensing ambient noise). The floor estimate drops quickly
when a quieter block is seen (tracks the true quiet baseline fast) and
rises slowly on louder blocks (so a single loud moment doesn't
permanently raise what counts as "the floor"). This is a standard,
well-established noise-gate technique, not a novel claim -- it's applied
here specifically to decide whether to engage the ANC pipeline at all.
"""

import numpy as np


class NoisePresenceGate:
    def __init__(self, threshold_margin=3.0, calibration_blocks=20, initial_floor=1e-5):
        """
        threshold_margin: how many times louder than the calibrated floor
            a block must be to count as "real noise present".

        IMPORTANT DESIGN NOTE (fixed after testing found a real bug):
        An earlier version of this gate let the floor estimate slowly
        RISE to match whatever level the last few seconds of audio sat
        at. That's wrong for this use case -- a sustained noise source
        (a helicopter rotor running for an entire flight, not just a
        brief moment) would gradually get "learned" as the new normal
        and silently stop being flagged as noise, which is the opposite
        of what a defence ANC system should do. Verified with
        test_noise_presence.py: continuous rotor noise was incorrectly
        reclassified as "clean" after ~5-6 seconds under the old design.

        THE FIX: calibrate a floor once from the first few blocks (using
        a low percentile, so it isn't skewed by an early burst), then the
        floor is only ever allowed to DECREASE afterward (if the
        environment gets genuinely quieter), never increase. This means
        sustained noise stays correctly flagged as "noise present" for
        as long as it persists, no matter how long that is.
        """
        self.threshold_margin = threshold_margin
        self.calibration_blocks = calibration_blocks
        self.floor_estimate = initial_floor
        self._calibration_samples = []
        self._calibrated = False

    def check(self, reference_block):
        """
        Returns (noise_present: bool, floor_estimate: float, block_energy: float)

        REAL LIMITATION, STATED PLAINLY: this gate needs a genuinely
        quiet moment during its first ~20 blocks (~0.6s) to calibrate
        against -- exactly like noise-cancelling headphones calibrating
        on power-on. If the device is switched on while already in a
        loud environment with no quiet moment at all, it will
        incorrectly calibrate against the noise itself as "the floor" and
        under-detect real noise afterward. Confirmed by direct testing:
        continuous noise present from sample zero, with no quiet lead-in,
        gets miscalibrated as baseline. Mitigation for the real device:
        either power on before entering a noisy area, or add a manual
        "recalibrate" trigger (e.g. a button) the operator can press
        during a momentary lull -- this is not yet implemented.
        """
        block_energy = float(np.mean(np.asarray(reference_block, dtype=np.float64) ** 2)) + 1e-12

        if not self._calibrated:
            self._calibration_samples.append(block_energy)
            if len(self._calibration_samples) >= self.calibration_blocks:
                # Use a low percentile, not the mean/min, so a single very
                # quiet fluke sample (or a burst) during calibration
                # doesn't set an unrealistic floor.
                self.floor_estimate = float(np.percentile(self._calibration_samples, 20))
                self._calibrated = True
            else:
                # Not enough data to calibrate yet -- assume noise is
                # present rather than silently doing nothing during startup.
                return True, self.floor_estimate, block_energy
        else:
            # Floor can only ever decrease (the environment got quieter),
            # never increase (sustained noise must never be "learned away").
            if block_energy < self.floor_estimate:
                self.floor_estimate = block_energy

        noise_present = block_energy > self.floor_estimate * self.threshold_margin
        return noise_present, self.floor_estimate, block_energy

    def recalibrate(self):
        """
        Reset calibration state so the gate re-learns its quiet baseline
        from the next `calibration_blocks` blocks of audio.

        This directly closes the documented startup limitation: if the
        device was powered on in a noisy environment and miscalibrated,
        the operator can press a "Recalibrate" button during a momentary
        lull to get a correct baseline without restarting the whole system.

        Called from the dashboard's recalibration control.
        """
        self._calibration_samples = []
        self._calibrated = False
        self.floor_estimate = 1e-5
