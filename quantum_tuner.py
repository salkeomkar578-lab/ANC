"""
Stage 3 of IGARD-Net: quantum-inspired auto-tuner (Gaussian-mutation
Quantum-behaved Particle Swarm Optimization -- GQPSO).

WHY THIS EXISTS:
A fixed NLMS step size is a compromise -- large enough to react to sudden
noise (gunshots), but small enough not to become unstable on steady noise.
Instead of picking one fixed value, a small swarm of "candidate step sizes"
is evaluated against the *actual recent audio*, and the swarm converges
toward whichever value currently gives the lowest residual noise energy.

This is real, published optimization math (not a literal quantum
computer) -- see the GQPSO paper cited in the project report. The swarm
size and iteration count below are deliberately kept small (8 particles,
6 iterations) specifically so this can run in real time on a Raspberry Pi
4's CPU, which is the direct engineering answer to the "too heavy for
embedded devices" gap flagged in the SONIC survey.
"""

import numpy as np
from nlms_filter import NLMSFilter


class QuantumInspiredTuner:
    def __init__(self, num_particles=8, iterations=6,
                 search_min=0.01, search_max=1.5,
                 beta_max=1.0, beta_min=0.3, mutation_prob=0.1):
        self.num_particles = num_particles
        self.iterations = iterations
        self.search_min = search_min
        self.search_max = search_max
        self.beta_max = beta_max
        self.beta_min = beta_min
        self.mutation_prob = mutation_prob

        rng = np.random.default_rng()
        self.positions = rng.uniform(search_min, search_max, size=num_particles)
        self.personal_best = self.positions.copy()
        self.personal_best_fitness = np.full(num_particles, -np.inf)
        self.global_best = self.positions[0]
        self.global_best_fitness = -np.inf
        self._rng = rng

    def _fitness(self, step_size, primary_block, reference_block, num_taps):
        """
        Fitness = negative residual energy after running a scratch NLMS
        filter with this candidate step size over the recent audio block.
        Higher fitness (closer to 0) = less residual noise = better.
        """
        trial_filter = NLMSFilter(num_taps=num_taps, step_size=step_size)
        out = trial_filter.process_block(primary_block, reference_block)
        residual_energy = float(np.mean(out ** 2)) + 1e-12
        return -residual_energy

    def retune(self, primary_block, reference_block, num_taps):
        """
        Run a short GQPSO search using a recent window of audio and return
        the best step size found. Call this periodically (e.g. every
        200-500ms of audio), not on every single sample -- it is too
        expensive to run per-sample.
        """
        for it in range(self.iterations):
            beta = self.beta_max - (self.beta_max - self.beta_min) * (it / max(1, self.iterations - 1))
            mbest = np.mean(self.personal_best)

            for i in range(self.num_particles):
                fitness = self._fitness(self.positions[i], primary_block, reference_block, num_taps)

                if fitness > self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness
                    self.personal_best[i] = self.positions[i]
                if fitness > self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best = self.positions[i]

            for i in range(self.num_particles):
                phi = self._rng.uniform(0, 1)
                p = phi * self.personal_best[i] + (1 - phi) * self.global_best
                u = self._rng.uniform(1e-6, 1.0)
                sign = 1 if self._rng.uniform() < 0.5 else -1
                new_pos = p + sign * beta * abs(mbest - self.positions[i]) * np.log(1.0 / u)

                # Gaussian mutation: occasionally jolt a particle to keep
                # the swarm from collapsing onto one value too early
                if self._rng.uniform() < self.mutation_prob:
                    new_pos += self._rng.normal(0, 0.05)

                self.positions[i] = float(np.clip(new_pos, self.search_min, self.search_max))

        return float(self.global_best)
