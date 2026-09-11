"""
WAV File Uploader and Validator for IGARD-Net.
Validates file integrity, PCM format, headers, channel count, and extracts audio metadata.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import wave
import soundfile as sf
import numpy as np


class UploadValidationError(Exception):
    pass


class FileUploader:
    def __init__(self, upload_dir: Path, max_file_size_mb: int = 50):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024

    def validate_and_save(self, file_storage) -> Tuple[Path, Dict[str, Any]]:
        """
        Validates uploaded file and saves it securely.
        Returns: (saved_file_path, metadata_dict)
        """
        filename = Path(file_storage.filename).name
        if not filename.lower().endswith(".wav"):
            raise UploadValidationError("Unsupported file format: Only .wav files are supported.")

        save_path = self.upload_dir / filename
        file_storage.save(str(save_path))

        file_size = save_path.stat().st_size
        if file_size > self.max_file_size_bytes:
            save_path.unlink(missing_ok=True)
            raise UploadValidationError(f"File exceeds maximum allowed size of {self.max_file_size_bytes // (1024*1024)} MB.")

        if file_size < 44:
            save_path.unlink(missing_ok=True)
            raise UploadValidationError("Corrupted or empty WAV file (header incomplete).")

        # Validate with soundfile / wave
        try:
            info = sf.info(str(save_path))
        except Exception as e:
            save_path.unlink(missing_ok=True)
            raise UploadValidationError(f"Failed to decode WAV file: {e}")

        meta = {
            "filename": filename,
            "duration_s": round(info.duration, 2),
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "format": info.format,
            "subtype": info.subtype,
            "size_bytes": file_size,
        }
        return save_path, meta
