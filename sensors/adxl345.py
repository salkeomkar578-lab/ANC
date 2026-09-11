"""
Hardware Driver for ADXL345 I2C Accelerometer on Linux / Raspberry Pi / Jetson.
"""

import numpy as np
from sensors.accelerometer import AccelerometerBase


class ADXL345Accelerometer(AccelerometerBase):
    ADXL345_ADDR = 0x53
    REG_POWER_CTL = 0x2D
    REG_DATAX0 = 0x32

    def __init__(self, bus_number: int = 1, shock_g_threshold: float = 3.0):
        self.shock_g_threshold = shock_g_threshold
        self.initialized = False
        
        try:
            import smbus2
            self.bus = smbus2.SMBus(bus_number)
            self.bus.write_byte_data(self.ADXL345_ADDR, self.REG_POWER_CTL, 0x08)
            self.initialized = True
        except Exception as e:
            print(f"[Sensor] ADXL345 I2C init failed ({e}). Sensor will return baseline 0.02.")
            self.bus = None

    def _read_xyz_g(self):
        if not self.initialized or self.bus is None:
            return 0.0, 0.0, 1.0

        try:
            data = self.bus.read_i2c_block_data(self.ADXL345_ADDR, self.REG_DATAX0, 6)
            def to_signed16(lo, hi):
                val = (hi << 8) | lo
                return val - 65536 if val > 32767 else val

            x = to_signed16(data[0], data[1]) * 0.0039
            y = to_signed16(data[2], data[3]) * 0.0039
            z = to_signed16(data[4], data[5]) * 0.0039
            return x, y, z
        except Exception:
            return 0.0, 0.0, 1.0

    def read_recent_shock_score(self) -> float:
        x, y, z = self._read_xyz_g()
        magnitude = float(np.sqrt(x ** 2 + y ** 2 + z ** 2))
        extra_g = abs(magnitude - 1.0)
        score = float(np.clip(extra_g / self.shock_g_threshold, 0.0, 1.0))
        return score
