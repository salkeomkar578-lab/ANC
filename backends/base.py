"""
Base ComputeBackend abstraction for IGARD-Net.
Enables transparent switching between CPU (NumPy/SciPy) and GPU (CUDA/PyTorch).
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import numpy as np


class ComputeBackend(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the backend: 'CPU' or 'CUDA'"""
        pass

    @property
    @abstractmethod
    def is_cuda(self) -> bool:
        """True if running on an NVIDIA GPU via CUDA"""
        pass

    @abstractmethod
    def nlms_process_block(
        self,
        weights: np.ndarray,
        ref_buffer: np.ndarray,
        primary_block: np.ndarray,
        ref_block: np.ndarray,
        mu: float,
        eps: float,
        leakage: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Processes a block of audio through an NLMS adaptive FIR filter.
        Returns: (error_output, noise_estimate, updated_weights, updated_ref_buffer)
        """
        pass

    @abstractmethod
    def compute_rfft_power(self, block: np.ndarray, window: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes real FFT, power spectrum, and phase.
        Returns: (complex_spectrum, power_spectrum, phase)
        """
        pass

    @abstractmethod
    def compute_irfft(self, spectrum: np.ndarray, n: int) -> np.ndarray:
        """Computes inverse real FFT."""
        pass

    @abstractmethod
    def apply_spectral_mask(
        self,
        spectrum: np.ndarray,
        power: np.ndarray,
        noise_power: np.ndarray,
        alpha: float,
        min_gain: float,
    ) -> np.ndarray:
        """
        Applies Wiener-style soft spectral gain mask.
        Returns: cleaned_complex_spectrum
        """
        pass
