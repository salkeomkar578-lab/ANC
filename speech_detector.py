"""
Speech Detector for IGARD-Net.
Provides continuous frame-by-frame speech_probability and noise_probability estimates
with stateful temporal smoothing (attack, release, hold time, and hysteresis).
Guarantees continuous speech tracking so word endings, consonants, and soft syllables
are never abruptly dropped.
"""

from typing import Tuple, Optional
import numpy as np


class SpeechDetector:
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_size: int = 256,
        attack_ms: float = 20.0,
        release_ms: float = 120.0,
        hold_ms: float = 100.0,
        hysteresis: float = 0.08,
    ):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.frame_duration_ms = (frame_size / sample_rate) * 1000.0

        # Temporal smoothing coefficients
        # alpha = exp(-dt / time_constant)
        self.alpha_attack = float(np.exp(-self.frame_duration_ms / max(1.0, attack_ms)))
        self.alpha_release = float(np.exp(-self.frame_duration_ms / max(1.0, release_ms)))
        self.hold_frames = int(np.ceil(hold_ms / max(1.0, self.frame_duration_ms)))
        self.hysteresis = hysteresis

        # Stateful trackers
        self._speech_prob_smooth: float = 0.0
        self._noise_prob_smooth: float = 1.0
        self._hold_counter: int = 0
        self._speech_state_bool: bool = False

        # Noise floor tracking per frequency band (recursive minimum statistics)
        self.n_fft = 256
        self._noise_psd = np.ones(self.n_fft // 2 + 1, dtype=np.float64) * 1e-4
        self._psd_smooth = np.ones(self.n_fft // 2 + 1, dtype=np.float64) * 1e-4
        self._window = np.hanning(frame_size)

        # Precompute speech band bin indices (300 Hz - 3400 Hz)
        freqs = np.fft.rfftfreq(frame_size, 1.0 / sample_rate)
        self.speech_band_mask = (freqs >= 300.0) & (freqs <= 3400.0)
        self.low_band_mask = freqs < 300.0
        self.high_band_mask = freqs > 3400.0

    def reset(self):
        """Reset all temporal state trackers."""
        self._speech_prob_smooth = 0.0
        self._noise_prob_smooth = 1.0
        self._hold_counter = 0
        self._speech_state_bool = False
        self._noise_psd.fill(1e-4)
        self._psd_smooth.fill(1e-4)

    def detect(self, frame: np.ndarray) -> Tuple[float, float, bool]:
        """
        Processes one audio frame.
        Returns:
            speech_prob: float in [0.0, 1.0] (smoothed with hold time)
            noise_prob: float in [0.0, 1.0] (smoothed)
            is_speech: bool (hysteresis-gated speech presence)
        """
        frame = np.asarray(frame, dtype=np.float64)
        n = len(frame)
        if n == 0:
            return 0.0, 1.0, False

        # Compute frame energy & RMS
        energy = float(np.mean(frame ** 2)) + 1e-12
        rms = np.sqrt(energy)

        # FFT Power Spectrum
        if n != len(self._window):
            self._window = np.hanning(n)
        
        spectrum = np.fft.rfft(frame * self._window)
        power = np.abs(spectrum) ** 2

        # 1. Smooth PSD
        self._psd_smooth = 0.8 * self._psd_smooth + 0.2 * power

        # 2. Spectral Flatness (Wiener entropy)
        # Peaky speech spectrum -> low flatness (< 0.2). Broadband noise -> high flatness (> 0.4)
        geom_mean = np.exp(np.mean(np.log(power + 1e-10)))
        arith_mean = np.mean(power) + 1e-10
        flatness = float(np.clip(geom_mean / arith_mean, 0.0, 1.0))

        # 3. Energy concentration in speech formant band (300 Hz - 3400 Hz)
        total_pwr = np.sum(power) + 1e-10
        speech_band_pwr = np.sum(power[self.speech_band_mask]) if np.any(self.speech_band_mask) else 0.0
        speech_band_ratio = float(speech_band_pwr / total_pwr)

        # 4. Sub-band SNR against tracked noise floor
        # Update noise floor estimate when energy is low or during release
        is_quiet = energy < (np.mean(self._noise_psd) * 2.5)
        if is_quiet or self._speech_prob_smooth < 0.2:
            self._noise_psd = 0.95 * self._noise_psd + 0.05 * self._psd_smooth
        else:
            # Slow floor tracking from below
            self._noise_psd = np.minimum(self._noise_psd * 1.002, self._psd_smooth)

        snr_subband = np.maximum(0.0, (self._psd_smooth - self._noise_psd) / (self._noise_psd + 1e-10))
        mean_speech_snr = float(np.mean(snr_subband[self.speech_band_mask])) if np.any(self.speech_band_mask) else 0.0

        # 5. Pitch / Harmonicity via Normalized Autocorrelation
        # Human speech exhibits strong periodicity in 70 - 400 Hz
        min_lag = int(self.sample_rate / 400) # ~40 samples @ 16kHz
        max_lag = int(self.sample_rate / 70)  # ~228 samples @ 16kHz
        
        harmonicity = 0.0
        if n >= max_lag:
            corr = np.correlate(frame - np.mean(frame), frame - np.mean(frame), mode='full')
            mid = len(corr) // 2
            lag_corr = corr[mid + min_lag : mid + max_lag]
            norm_factor = corr[mid] + 1e-10
            max_corr = np.max(lag_corr) if len(lag_corr) > 0 else 0.0
            harmonicity = float(np.clip(max_corr / norm_factor, 0.0, 1.0))

        # Combine physical cues into raw speech probability
        # Speech cues: high speech_band_ratio, low flatness, high harmonicity, positive SNR
        cue_flatness = np.clip((0.45 - flatness) / 0.35, 0.0, 1.0)
        cue_band = np.clip((speech_band_ratio - 0.25) / 0.45, 0.0, 1.0)
        cue_snr = np.clip(mean_speech_snr / 4.0, 0.0, 1.0)
        cue_harm = np.clip(harmonicity / 0.60, 0.0, 1.0)
        cue_energy = np.clip((rms - 0.005) / 0.04, 0.0, 1.0)

        # Weighted raw confidence
        p_raw = (
            0.25 * cue_band +
            0.25 * cue_flatness +
            0.20 * cue_snr +
            0.15 * cue_harm +
            0.15 * cue_energy
        )
        p_raw = float(np.clip(p_raw, 0.0, 1.0))

        # Temporal Smoothing with Attack, Hold, and Release
        if p_raw > self._speech_prob_smooth:
            # Attack: fast transition to speech
            self._speech_prob_smooth = (
                (1.0 - self.alpha_attack) * p_raw + self.alpha_attack * self._speech_prob_smooth
            )
            self._hold_counter = self.hold_frames
        else:
            # In release phase: hold speech state if counter > 0
            if self._hold_counter > 0:
                self._hold_counter -= 1
                # During hold, decay very slowly
                self._speech_prob_smooth = max(self._speech_prob_smooth * 0.98, p_raw)
            else:
                # Release: smooth exponential decay back to noise floor
                self._speech_prob_smooth = (
                    (1.0 - self.alpha_release) * p_raw + self.alpha_release * self._speech_prob_smooth
                )

        self._speech_prob_smooth = float(np.clip(self._speech_prob_smooth, 0.0, 1.0))
        self._noise_prob_smooth = float(np.clip(1.0 - self._speech_prob_smooth, 0.0, 1.0))

        # Hysteresis for boolean speech flag
        high_thresh = 0.50 + self.hysteresis / 2.0
        low_thresh = 0.50 - self.hysteresis / 2.0
        if self._speech_prob_smooth >= high_thresh:
            self._speech_state_bool = True
        elif self._speech_prob_smooth <= low_thresh:
            self._speech_state_bool = False

        return self._speech_prob_smooth, self._noise_prob_smooth, self._speech_state_bool
