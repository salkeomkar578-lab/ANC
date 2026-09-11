"""
Artifact Detector and Speech Continuity Evaluator for IGARD-Net.
Performs quantitative analysis on audio signals to detect:
1. Hard Muting (speech collapse)
2. Ticking / Clicking (abrupt amplitude discontinuities)
3. Pumping (unnatural low-frequency gain oscillation)
4. Metallic speech (formant distortion / isolated spectral peaks)
Calculates the unified Speech Continuity Score (0.0 to 1.0).
"""

from typing import Dict, Any, Tuple
import numpy as np


class ArtifactDetector:
    def __init__(self, sample_rate: int = 16000, frame_size: int = 256):
        self.sample_rate = sample_rate
        self.frame_size = frame_size

    def evaluate(
        self,
        raw_audio: np.ndarray,
        enhanced_audio: np.ndarray,
        speech_threshold_rms: float = 0.02,
    ) -> Dict[str, Any]:
        """
        Comprehensive comparison of raw vs enhanced audio.
        Returns dictionary of metrics, artifact counts, and speech continuity score.
        """
        raw = np.asarray(raw_audio, dtype=np.float64)
        enh = np.asarray(enhanced_audio, dtype=np.float64)
        min_len = min(len(raw), len(enh))
        raw = raw[:min_len]
        enh = enh[:min_len]

        if min_len < self.frame_size:
            return {
                "continuity_score": 1.0,
                "muting_count": 0,
                "ticking_count": 0,
                "pumping_count": 0,
                "metallic_count": 0,
                "min_speech_gain": 1.0,
                "max_suppression_db": 0.0,
                "mean_speech_gain": 1.0,
                "gain_variation": 0.0,
                "peak_raw": 0.0,
                "peak_enh": 0.0,
            }

        n_frames = min_len // self.frame_size
        raw_frames = raw[:n_frames * self.frame_size].reshape(n_frames, self.frame_size)
        enh_frames = enh[:n_frames * self.frame_size].reshape(n_frames, self.frame_size)

        raw_rms = np.sqrt(np.mean(raw_frames ** 2, axis=1))
        enh_rms = np.sqrt(np.mean(enh_frames ** 2, axis=1))
        eps = 1e-6
        gains = enh_rms / (raw_rms + eps)

        # Identify speech-active frames in raw audio
        speech_mask = raw_rms > speech_threshold_rms
        num_speech_frames = int(np.sum(speech_mask))

        if num_speech_frames == 0:
            # Fallback if audio has lower overall level
            speech_mask = raw_rms > (np.mean(raw_rms) * 1.2)
            num_speech_frames = int(np.sum(speech_mask))

        speech_gains = gains[speech_mask] if num_speech_frames > 0 else gains

        # 1. Muting Detection: speech present but gain < 0.20
        muted_mask = speech_mask & (gains < 0.20)
        muting_count = int(np.sum(muted_mask))

        # 2. Ticking / Clicking Detection:
        # Detect sample-level discontinuities at frame boundaries and inside frames
        boundary_diffs = np.abs(enh_frames[1:, 0] - enh_frames[:-1, -1])
        intra_diffs = np.abs(np.diff(enh_frames, axis=1))
        max_intra_diffs = np.max(intra_diffs, axis=1)
        # Ticking if boundary jump or intra-frame step exceeds 0.35
        ticking_frames = (boundary_diffs > 0.35)
        ticking_count = int(np.sum(ticking_frames))

        # 3. Pumping Detection:
        # Excessive frame-to-frame gain fluctuations during speech
        if len(speech_gains) > 1:
            gain_diffs = np.abs(np.diff(speech_gains))
            pumping_frames = gain_diffs > 0.30
            pumping_count = int(np.sum(pumping_frames))
            mean_gain_variation = float(np.mean(gain_diffs))
        else:
            pumping_count = 0
            mean_gain_variation = 0.0

        # 4. Metallic Speech Detection:
        # High spectral peakiness / centroid shift in speech frames
        # FFT on raw vs enh frames
        win = np.hanning(self.frame_size)
        raw_spec = np.abs(np.fft.rfft(raw_frames[speech_mask] * win, axis=1)) if num_speech_frames > 0 else np.zeros((1, self.frame_size//2+1))
        enh_spec = np.abs(np.fft.rfft(enh_frames[speech_mask] * win, axis=1)) if num_speech_frames > 0 else np.zeros((1, self.frame_size//2+1))

        freqs = np.fft.rfftfreq(self.frame_size, 1.0 / self.sample_rate)
        raw_centroid = np.sum(raw_spec * freqs, axis=1) / (np.sum(raw_spec, axis=1) + eps)
        enh_centroid = np.sum(enh_spec * freqs, axis=1) / (np.sum(enh_spec, axis=1) + eps)
        centroid_shifts = np.abs(enh_centroid - raw_centroid)
        metallic_frames = centroid_shifts > 1200.0 # severe centroid warping
        metallic_count = int(np.sum(metallic_frames))

        # Metrics
        min_speech_gain = float(np.min(speech_gains)) if len(speech_gains) > 0 else 1.0
        mean_speech_gain = float(np.mean(speech_gains)) if len(speech_gains) > 0 else 1.0
        max_suppression_db = float(np.clip(-20.0 * np.log10(max(1e-4, min_speech_gain)), 0.0, 60.0))

        # Noise suppression evaluated on background / non-speech frames
        noise_mask = ~speech_mask
        if np.any(noise_mask):
            noise_gains = gains[noise_mask]
            min_noise_gain = float(np.min(noise_gains))
            noise_suppression_db = float(np.clip(-20.0 * np.log10(max(1e-4, min_noise_gain)), 0.0, 60.0))
        else:
            noise_suppression_db = max_suppression_db

        # 5. Speech Continuity Score Calculation:
        # continuity_score = 1.0 - gain_variation_penalty - muting_penalty - pumping_penalty
        muting_ratio = muting_count / max(1, num_speech_frames)
        pumping_ratio = pumping_count / max(1, num_speech_frames)
        
        muting_penalty = min(0.40, muting_ratio * 1.5)
        pumping_penalty = min(0.30, pumping_ratio * 1.0)
        gain_var_penalty = min(0.30, mean_gain_variation * 1.2)

        continuity_score = float(np.clip(1.0 - muting_penalty - pumping_penalty - gain_var_penalty, 0.0, 1.0))

        return {
            "continuity_score": round(continuity_score, 4),
            "muting_count": muting_count,
            "muting_ratio_percent": round(muting_ratio * 100.0, 2),
            "ticking_count": ticking_count,
            "pumping_count": pumping_count,
            "metallic_count": metallic_count,
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
