"""
Stage 3 of IGARD-Net: Multi-Objective Quantum-Inspired Auto-Tuner (GQPSO).
Optimizes NLMS step size mu by balancing:
  + Speech preservation & intelligibility
  + Environmental noise reduction
  - Heavy penalty for speech attenuation and distortion
  - Hard parameter safety limits (max_mu <= 0.5)
"""

import numpy as np
from nlms_filter import NLMSFilter


class QuantumInspiredTuner:
    def __init__(
        self,
        num_particles: int = 8,
        iterations: int = 6,
        search_min: float = 0.01,
        search_max: float = 0.50,  # Hard safety limit on max mu
        beta_max: float = 1.0,
        beta_min: float = 0.3,
        mutation_prob: float = 0.1,
    ):
        self.num_particles = num_particles
        self.iterations = iterations
        self.search_min = search_min
        self.search_max = search_max
        self.beta_max = beta_max
        self.beta_min = beta_min
        self.mutation_prob = mutation_prob

        rng = np.random.default_rng(42)
        self.positions = rng.uniform(search_min, search_max, size=num_particles)
        self.personal_best = self.positions.copy()
        self.personal_best_fitness = np.full(num_particles, -np.inf)
        self.global_best = float(self.positions[0])
        self.global_best_fitness = -np.inf
        self._rng = rng

    def _multi_objective_fitness(
        self,
        step_size: float,
        primary_block: np.ndarray,
        reference_block: np.ndarray,
        num_taps: int,
        speech_prob: float = 0.0,
    ) -> float:
        """
        Multi-objective cost function:
        score = noise_reduction_score + speech_preservation_score - speech_loss_penalty
        """
        # Hard limit clamp
        step_size = float(np.clip(step_size, self.search_min, self.search_max))
        
        trial_filter = NLMSFilter(num_taps=num_taps, step_size=step_size)
        out, _ = trial_filter.process_block(primary_block, reference_block, speech_prob=speech_prob)
        
        in_energy = float(np.mean(primary_block ** 2)) + 1e-12
        out_energy = float(np.mean(out ** 2)) + 1e-12
        
        # Attenuation ratio
        attenuation_ratio = out_energy / in_energy
        
        # 1. Noise reduction component (desirable during noise)
        noise_reduction_score = -out_energy

        # 2. Speech loss penalty: if speech is probable and energy collapsed, penalize heavily
        speech_loss_penalty = 0.0
        if speech_prob > 0.25:
            # Expected preservation floor
            safe_floor = 0.50
            if attenuation_ratio < safe_floor:
                deficit = safe_floor - attenuation_ratio
                # Severe quadratic penalty on speech loss
                speech_loss_penalty = 10.0 * (deficit ** 2) * in_energy * speech_prob

        # 3. Parameter stability penalty (discourage large erratic step sizes during speech)
        param_penalty = 0.1 * (step_size ** 2) * speech_prob

        fitness = noise_reduction_score - speech_loss_penalty - param_penalty
        return float(fitness)

    def retune(
        self,
        primary_block: np.ndarray,
        reference_block: np.ndarray,
        num_taps: int,
        speech_prob: float = 0.0,
    ) -> float:
        """
        Runs GQPSO search using multi-objective speech-protective fitness.
        """
        # Fast downsample evaluation if block is large to prevent CPU stalls
        if len(primary_block) > 1024:
            eval_p = primary_block[-1024:]
            eval_r = reference_block[-1024:]
        else:
            eval_p = primary_block
            eval_r = reference_block

        for it in range(self.iterations):
            beta = self.beta_max - (self.beta_max - self.beta_min) * (it / max(1, self.iterations - 1))
            mbest = float(np.mean(self.personal_best))

            for i in range(self.num_particles):
                fitness = self._multi_objective_fitness(
                    self.positions[i],
                    eval_p,
                    eval_r,
                    num_taps,
                    speech_prob=speech_prob,
                )

                if fitness > self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness
                    self.personal_best[i] = self.positions[i]
                if fitness > self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best = float(self.positions[i])

            for i in range(self.num_particles):
                phi = self._rng.uniform(0, 1)
                p = phi * self.personal_best[i] + (1 - phi) * self.global_best
                u = self._rng.uniform(1e-6, 1.0)
                sign = 1 if self._rng.uniform() < 0.5 else -1
                new_pos = p + sign * beta * abs(mbest - self.positions[i]) * np.log(1.0 / u)

                if self._rng.uniform() < self.mutation_prob:
                    new_pos += self._rng.normal(0, 0.03)

                self.positions[i] = float(np.clip(new_pos, self.search_min, self.search_max))

        return float(np.clip(self.global_best, self.search_min, self.search_max))
