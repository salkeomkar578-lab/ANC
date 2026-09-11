"""
Scriptable Mock Accelerometer for Unit Testing and Demo Simulation.
"""

import time
from typing import List, Tuple
from sensors.accelerometer import AccelerometerBase


class MockAccelerometer(AccelerometerBase):
    def __init__(self):
        self._scripted_shocks: List[Tuple[float, float]] = []
        self._start_time = time.monotonic()
        self._manual_shock: float = 0.0

    def script_shock_at(self, seconds_from_start: float, duration: float = 0.05):
        """Pre-program a simulated shock at a specific timestamp."""
        self._scripted_shocks.append((seconds_from_start, duration))

    def trigger_shock(self, intensity: float = 0.95):
        """Immediately trigger a shock event for interactive demo testing."""
        self._manual_shock = intensity

    def read_recent_shock_score(self) -> float:
        if self._manual_shock > 0.0:
            score = self._manual_shock
            self._manual_shock = 0.0
            return score

        elapsed = time.monotonic() - self._start_time
        for start, dur in self._scripted_shocks:
            if start <= elapsed <= start + dur:
                return 0.95

        # Normal sensor baseline vibration noise
        return 0.02
