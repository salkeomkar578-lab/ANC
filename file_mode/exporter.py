"""
WAV File Exporter and Metadata Packager for IGARD-Net.
"""

from pathlib import Path
from typing import Dict, Any
import soundfile as sf
import numpy as np


class FileExporter:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_wav(self, audio_data: np.ndarray, filename: str, sample_rate: int = 16000) -> Path:
        out_path = self.output_dir / filename
        clipped = np.clip(audio_data, -1.0, 1.0)
        sf.write(str(out_path), (clipped * 32767.0).astype(np.int16), sample_rate, subtype="PCM_16")
        return out_path
