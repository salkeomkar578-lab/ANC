"""
Stage 3: Asynchronous Amortized GQPSO Tuner.
Optimizes NLMS step size mu periodically in a background thread.
Never blocks the real-time audio thread. Updates state.mu_current atomically.
"""

import threading
import time
from typing import Optional
import numpy as np

from backends.base import ComputeBackend
from backends.cpu_backend import CPUBackend


class QuantumInspiredTuner:
    def __init__(
        self,
        num_particles: int = 8,
        iterations: int = 5,
        search_min: float = 0.01,
        search_max: float = 1.5,
        beta_max: float = 1.0,
        beta_min: float = 0.3,
        mutation_prob: float = 0.1,
        history_len: int = 1024,
        backend: Optional[ComputeBackend] = None,
    ):
        self.num_particles = num_particles
        self.iterations = iterations
        self.search_min = search_min
        self.search_max = search_max
        self.beta_max = beta_max
        self.beta_min = beta_min
        self.mutation_prob = mutation_prob
        self.history_len = history_len
        self.backend = backend if backend is not None else CPUBackend()

        self._rng = np.random.default_rng(42)
        self.positions = self._rng.uniform(search_min, search_max, size=num_particles)
        self.personal_best = self.positions.copy()
        self.personal_best_fitness = np.full(num_particles, -np.inf)
        self.global_best = float(self.positions[0])
        self.global_best_fitness = -np.inf
        
        self.cached_best_mu: float = 0.25
        self.last_retune_time: float = 0.0

    def _fitness(self, step_size: float, primary_snap: np.ndarray, ref_snap: np.ndarray, num_taps: int) -> float:
        """
        Fast fitness evaluation: negative residual energy after trial filtering.
        Higher fitness (closer to 0) = better noise reduction.
        """
        init_weights = np.zeros(num_taps, dtype=np.float64)
        init_ref = np.zeros(num_taps, dtype=np.float64)
        
        err, _, _, _ = self.backend.nlms_process_block(
            weights=init_weights,
            ref_buffer=init_ref,
            primary_block=primary_snap,
            ref_block=ref_snap,
            mu=step_size,
            eps=1.0e-6,
            leakage=1.0,
        )
        residual_energy = float(np.mean(err ** 2)) + 1.0e-12
        return -residual_energy

    def optimize_step(self, primary_snap: np.ndarray, ref_snap: np.ndarray, num_taps: int = 64) -> float:
        """
        Runs GQPSO swarm search on a snapshot of recent audio.
        Designed to complete in <10 ms on CPU/GPU.
        """
        if len(primary_snap) > self.history_len:
            p_snap = primary_snap[-self.history_len:]
            r_snap = ref_snap[-self.history_len:]
        else:
            p_snap = primary_snap
            r_snap = ref_snap

        for it in range(self.iterations):
            beta = self.beta_max - (self.beta_max - self.beta_min) * (it / max(1, self.iterations - 1))
            mbest = float(np.mean(self.personal_best))

            for i in range(self.num_particles):
                fitness = self._fitness(self.positions[i], p_snap, r_snap, num_taps)
                if fitness > self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness
                    self.personal_best[i] = self.positions[i]
                if fitness > self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best = float(self.positions[i])

            for i in range(self.num_particles):
                phi = self._rng.uniform(0, 1)
                p = phi * self.personal_best[i] + (1 - phi) * self.global_best
                u = self._rng.uniform(1.0e-6, 1.0)
                sign = 1 if self._rng.uniform() < 0.5 else -1
                new_pos = p + sign * beta * abs(mbest - self.positions[i]) * np.log(1.0 / u)

                if self._rng.uniform() < self.mutation_prob:
                    new_pos += self._rng.normal(0, 0.05)

                self.positions[i] = float(np.clip(new_pos, self.search_min, self.search_max))

        self.cached_best_mu = float(self.global_best)
        self.last_retune_time = time.time()
        return self.cached_best_mu


class BackgroundTunerWorker:
    """
    Dedicated worker thread that consumes audio snapshots and updates mu asynchronously.
    Zero interference with real-time audio callback latency.
    """
    def __init__(self, tuner: QuantumInspiredTuner, interval_s: float = 1.0, num_taps: int = 64):
        self.tuner = tuner
        self.interval_s = interval_s
        self.num_taps = num_taps
        
        self._lock = threading.Lock()
        self._latest_primary: Optional[np.ndarray] = None
        self._latest_reference: Optional[np.ndarray] = None
        self._new_data_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        
        self.current_mu: float = 0.25

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="GQPSO_Worker")
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._new_data_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def submit_snapshot(self, primary_window: np.ndarray, reference_window: np.ndarray):
        """Called non-blockingly from pipeline when a snapshot is ready."""
        with self._lock:
            self._latest_primary = primary_window.copy()
            self._latest_reference = reference_window.copy()
        self._new_data_event.set()

    def _worker_loop(self):
        while not self._stop_event.is_set():
            # Wait for interval or new data
            self._new_data_event.wait(timeout=self.interval_s)
            self._new_data_event.clear()
            if self._stop_event.is_set():
                break

            with self._lock:
                p_snap = self._latest_primary
                r_snap = self._latest_reference

            if p_snap is not None and r_snap is not None and len(p_snap) >= 256:
                try:
                    new_mu = self.tuner.optimize_step(p_snap, r_snap, num_taps=self.num_taps)
                    # Exponential smoothing between retunes
                    self.current_mu = 0.7 * self.current_mu + 0.3 * new_mu
                except Exception as e:
                    print(f"[GQPSO Tuner] Background optimization error: {e}")

            time.sleep(self.interval_s)
