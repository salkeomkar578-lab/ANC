"""
Cross-modal confirmation sensor: a cheap MEMS accelerometer (e.g. ADXL345,
~Rs.100-150, I2C) mounted on the radio unit, used to physically confirm
whether an acoustic "impulsive" classification (gunshot/artillery) is a
real mechanical shock event, not just an acoustically-similar sound like
fireworks or vehicle backfire.

WHY THIS EXISTS (see project research notes):
Published gunshot classifiers -- acoustic-only -- have a documented,
unsolved false-positive problem: they confuse real gunfire with fireworks
and backfire because those sounds are acoustically similar. A real 2014
study (Loeffler, U. Penn, PLOS ONE) showed a cheap wearable accelerometer
distinguishes actual firearm discharge from confusable events at
99.4-99.7% accuracy, because a gunshot's MECHANICAL shock signature is
physically distinct in a way sound alone isn't. This module brings that
same idea into IGARD-Net as a second, independent evidence stream for
the confidence gate.

HONESTY NOTE: no accelerometer hardware exists in this project yet. This
file ships with a MOCK backend (`MockAccelerometer`) that lets the rest
of the pipeline be built, tested, and demoed today. A real backend
(`ADXL345Accelerometer`) is included and written against the correct I2C
register map, but is UNTESTED against real hardware -- verify it the
moment you actually wire up a sensor.
"""

import time
import numpy as np


class AccelerometerBase:
    """Common interface both the mock and the real sensor implement."""

    def read_recent_shock_score(self):
        """
        Returns a float in [0, 1]: how much this looks like a real
        mechanical shock event in the last short window (roughly the
        last ~30-50ms, matching a gunshot's blast+recoil timescale).
        0 = no shock detected, 1 = very confident a real shock occurred.
        """
        raise NotImplementedError


class MockAccelerometer(AccelerometerBase):
    """
    Stand-in for real hardware. By default, returns 0 (no shock) always,
    UNLESS you tell it when a "shock" should be simulated -- this lets
    the rest of the code (confidence gate, pipeline) be built and tested
    against realistic on/off shock events without real hardware.
    """

    def __init__(self):
        self._scripted_shock_times = []  # list of (start_time, duration) tuples
        self._start = time.monotonic()

    def script_shock_at(self, seconds_from_start, duration=0.05):
        """For testing: pretend a real shock happens at this timestamp."""
        self._scripted_shock_times.append((seconds_from_start, duration))

    def read_recent_shock_score(self):
        now = time.monotonic() - self._start
        for start, duration in self._scripted_shock_times:
            if start <= now <= start + duration:
                return 0.95
        return 0.02  # small non-zero baseline, like real sensor noise


class ADXL345Accelerometer(AccelerometerBase):
    """
    Real hardware backend for a common, cheap (~Rs.100-150) I2C
    accelerometer, the ADXL345. UNTESTED against real hardware -- the
    register addresses and math below follow the ADXL345 datasheet, but
    verify this the moment you have the physical sensor wired up.

    Wiring (once you have the part): VCC->3.3V, GND->GND, SDA->Pi SDA
    (pin 3), SCL->Pi SCL (pin 5). Enable I2C first with `sudo raspi-config`
    -> Interface Options -> I2C -> Yes.
    """

    ADXL345_ADDR = 0x53
    REG_POWER_CTL = 0x2D
    REG_DATAX0 = 0x32

    def __init__(self, bus_number=1, shock_g_threshold=3.0, window_seconds=0.05):
        import smbus2  # imported here so the mock path never needs this dependency
        self.bus = smbus2.SMBus(bus_number)
        self.bus.write_byte_data(self.ADXL345_ADDR, self.REG_POWER_CTL, 0x08)  # start measuring
        self.shock_g_threshold = shock_g_threshold
        self.window_seconds = window_seconds
        self._recent_magnitudes = []

    def _read_xyz_g(self):
        data = self.bus.read_i2c_block_data(self.ADXL345_ADDR, self.REG_DATAX0, 6)
        def to_signed16(lo, hi):
            val = (hi << 8) | lo
            return val - 65536 if val > 32767 else val
        x = to_signed16(data[0], data[1]) * 0.0039  # datasheet scale factor, +/-2g range
        y = to_signed16(data[2], data[3]) * 0.0039
        z = to_signed16(data[4], data[5]) * 0.0039
        return x, y, z

    def read_recent_shock_score(self):
        x, y, z = self._read_xyz_g()
        magnitude = float(np.sqrt(x ** 2 + y ** 2 + z ** 2))
        # subtract 1g resting gravity magnitude to get "extra" acceleration
        extra_g = abs(magnitude - 1.0)
        score = float(np.clip(extra_g / self.shock_g_threshold, 0.0, 1.0))
        return score


def cross_check_confidence(acoustic_label, acoustic_confidence, shock_score, agreement_boost=0.25):
    """
    Combine the acoustic classifier's opinion with the accelerometer's
    physical evidence. If the classifier says "impulsive" (a likely
    gunshot/blast) AND the accelerometer agrees (a real shock happened at
    the same time), boost confidence -- two independent evidence streams
    agreeing is much stronger evidence than either alone.

    If the classifier says "impulsive" but the accelerometer saw NO
    shock, that's a strong hint this is a false positive (e.g. fireworks,
    a loud shout, vehicle backfire heard through the mic but with no
    matching physical shock) -- confidence is reduced instead.
    """
    if acoustic_label != "impulsive":
        return acoustic_confidence  # cross-check only matters for impulsive claims

    if shock_score > 0.5:
        boosted = acoustic_confidence + agreement_boost * (1 - acoustic_confidence)
        return float(np.clip(boosted, 0.0, 0.99))
    else:
        penalized = acoustic_confidence * 0.5
        return float(np.clip(penalized, 0.0, 1.0))
