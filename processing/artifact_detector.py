"""
Artifact Detector and Diagnostic Evaluator for IGARD-Net V4.
Implements automated detection of:
1. Hard Muting: Exact-zero or near-zero output during significant input activity.
2. Click / Tick Discontinuities: Sudden sample/gain discontinuities across frame boundaries.
3. Gain Pumping: Rapid repetitive gain modulation during speech intervals.
4. Speech Collapse: Strong input speech confidence combined with severely attenuated output.
5. Clipping: Output exceeding linear dynamic range (>= 0.999).
6. Speech Continuity Score: Comprehensive ratio of continuous preserved voice.
"""

from typing import Dict, Any, Optional
import numpy as np
import scipy.signal as signal


class ArtifactDetector:
    def __init__(self, sample_rate: int = 16000, frame_size: int = 256):
        self.sample_rate = sample_rate
        self.frame_size = frame_size

    def evaluate(
        self,
        raw_audio: np.ndarray,
        enhanced_audio: np.ndarray,
        speech_threshold_rms: float = 0.02,
        speech_probs: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive diagnostic audit comparing raw input against enhanced output.
        """
        raw = np.asarray(raw_audio, dtype=np.float64)
        enh = np.asarray(enhanced_audio, dtype=np.float64)
        min_len = min(len(raw), len(enh))
        raw = raw[:min_len]
        enh = enh[:min_len]

        if min_len < self.frame_size:
            return {
                "continuity_score": 1.0,
                "hard_mute_count": 0,
                "muting_count": 0,
                "click_tick_count": 0,
                "ticking_count": 0,
                "pumping_count": 0,
                "speech_collapse_count": 0,
                "clipping_count": 0,
                "exact_zero_percentage": 0.0,
                "min_speech_gain": 1.0,
                "mean_speech_gain": 1.0,
                "gain_variation": 0.0,
                "noise_suppression_db": 0.0,
            }

        # Automatic cross-correlation delay alignment for filterbank latency compensation
        if min_len > self.sample_rate * 2:
            search_len = min(self.sample_rate * 2, min_len)
            s_raw = raw[:search_len]
            s_enh = enh[:search_len]
            cc = signal.correlate(s_enh, s_raw, mode="full", method="fft")
            lags = signal.correlation_lags(len(s_enh), len(s_raw))
            best_lag = int(lags[np.argmax(cc)])
            if 0 < best_lag < int(self.sample_rate * 0.10):
                aligned_enh = np.zeros_like(raw)
                aligned_enh[:-best_lag] = enh[best_lag:]
                enh = aligned_enh

        n_frames = min_len // self.frame_size
        raw_frames = raw[:n_frames * self.frame_size].reshape(n_frames, self.frame_size)
        enh_frames = enh[:n_frames * self.frame_size].reshape(n_frames, self.frame_size)

        raw_rms = np.sqrt(np.mean(raw_frames ** 2, axis=1))
        enh_rms = np.sqrt(np.mean(enh_frames ** 2, axis=1))
        eps = 1e-6
        gains = enh_rms / (raw_rms + eps)

        # Identify speech frames
        if speech_probs is not None and len(speech_probs) >= n_frames:
            speech_mask = speech_probs[:n_frames] >= 0.40
        else:
            speech_mask = raw_rms > speech_threshold_rms

        num_speech_frames = int(np.sum(speech_mask))
        if num_speech_frames == 0:
            speech_mask = raw_rms > (np.mean(raw_rms) * 1.2)
            num_speech_frames = int(np.sum(speech_mask))

        speech_gains = gains[speech_mask] if num_speech_frames > 0 else gains

        # 1. Hard Mute Detection:
        # Input has active energy (raw_rms > 0.01) but enhanced output collapses to exact-zero or near-zero (< 1e-5)
        hard_mute_frames = (raw_rms > 0.01) & (enh_rms < 1e-5)
        hard_mute_count = int(np.sum(hard_mute_frames))

        # Soft Mute / Severe Attenuation during speech (gains < 0.20)
        muted_speech_frames = speech_mask & (gains < 0.20)
        muting_count = int(np.sum(muted_speech_frames))

        # 2. Click / Tick Discontinuity Detection:
        # Check boundary step between consecutive frames
        boundary_diffs = np.abs(enh_frames[1:, 0] - enh_frames[:-1, -1])
        ticking_frames = boundary_diffs > 0.15
        click_tick_count = int(np.sum(ticking_frames))

        # 3. Pumping Detection:
        # Large frame-to-frame gain delta during speech intervals
        if len(speech_gains) > 1:
            gain_diffs = np.abs(np.diff(speech_gains))
            pumping_frames = gain_diffs > 0.20
            pumping_count = int(np.sum(pumping_frames))
            mean_gain_variation = float(np.mean(gain_diffs))
        else:
            pumping_count = 0
            mean_gain_variation = 0.0

        # 4. Speech Collapse:
        # Confident speech frames where voice gain drops below 0.35
        speech_collapse_frames = speech_mask & (gains < 0.35)
        speech_collapse_count = int(np.sum(speech_collapse_frames))

        # 5. Clipping:
        # Output values reaching dynamic range rail (>= 0.999)
        clipping_samples = np.sum(np.abs(enh) >= 0.999)
        clipping_count = int(clipping_samples)

        # Exact zero percentage
        exact_zero_percentage = float(np.sum(np.abs(enh) < 1e-8) / min_len * 100.0)

        # Gain and suppression metrics
        min_speech_gain = float(np.min(speech_gains)) if len(speech_gains) > 0 else 1.0
        mean_speech_gain = float(np.mean(speech_gains)) if len(speech_gains) > 0 else 1.0
        max_suppression_db = float(np.clip(-20.0 * np.log10(max(1e-4, min_speech_gain)), 0.0, 60.0))

        # Noise suppression on non-speech intervals
        noise_mask = ~speech_mask
        if np.any(noise_mask):
            noise_gains = gains[noise_mask]
            min_noise_gain = float(np.min(noise_gains))
            noise_suppression_db = float(np.clip(-20.0 * np.log10(max(1e-4, min_noise_gain)), 0.0, 60.0))
        else:
            noise_suppression_db = max_suppression_db

        # 6. Speech Continuity Score (0.0 to 1.0)
        # Ratio of preserved speech frames without collapse minus variation penalty
        preserved_speech_frames = num_speech_frames - speech_collapse_count
        preservation_ratio = preserved_speech_frames / max(1, num_speech_frames)

        muting_penalty = min(0.35, (muting_count / max(1, num_speech_frames)) * 1.5)
        gain_var_penalty = min(0.15, mean_gain_variation * 0.8)
        click_penalty = min(0.10, (click_tick_count / max(1, n_frames)) * 2.0)

        continuity_score = float(np.clip(
            preservation_ratio - muting_penalty - gain_var_penalty - click_penalty,
            0.0,
            1.0,
        ))

        return {
            "continuity_score": round(continuity_score, 4),
            "hard_mute_count": hard_mute_count,
            "muting_count": muting_count,
            "muting_ratio_percent": round((muting_count / max(1, num_speech_frames)) * 100.0, 2),
            "click_tick_count": click_tick_count,
            "ticking_count": click_tick_count,
            "pumping_count": pumping_count,
            "speech_collapse_count": speech_collapse_count,
            "clipping_count": clipping_count,
            "exact_zero_percentage": round(exact_zero_percentage, 2),
            "min_speech_gain": round(min_speech_gain, 4),
            "mean_speech_gain": round(mean_speech_gain, 4),
            "max_suppression_db": round(max_suppression_db, 2),
            "noise_suppression_db": round(noise_suppression_db, 2),
            "gain_variation": round(mean_gain_variation, 4),
            "peak_raw": round(float(np.max(np.abs(raw))), 4),
            "peak_enh": round(float(np.max(np.abs(enh))), 4),
            "num_speech_frames": num_speech_frames,
            "total_frames": n_frames,
        }
