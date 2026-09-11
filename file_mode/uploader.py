"""
Universal Audio File Uploader and Validator for IGARD-Net.
Supports WAV, MP3, FLAC, OGG, M4A, AAC, AIFF, and more.
Validates file integrity, extracts audio metadata, and generates a
browser-compatible 16-bit PCM preview for seamless HTML5 audition.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import soundfile as sf
import numpy as np


class UploadValidationError(Exception):
    pass


SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac",
    ".aiff", ".aif", ".opus", ".wma", ".webm", ".caf", ".au"
}


def load_audio_universal(filepath: Path) -> Tuple[np.ndarray, int]:
    """
    Decodes audio using soundfile, with fallback to torchaudio for exotic formats.
    Returns: (audio_data_np, sample_rate)
    """
    try:
        data, sr = sf.read(str(filepath), dtype="float64")
        return data, sr
    except Exception as sf_err:
        try:
            import torchaudio
            tensor, sr = torchaudio.load(str(filepath))
            arr = tensor.numpy().astype(np.float64)
            data = arr.T if arr.ndim == 2 else arr
            return data, sr
        except Exception as ta_err:
            raise UploadValidationError(
                f"Failed to decode audio file '{filepath.name}'. Soundfile error: {sf_err}; Torchaudio error: {ta_err}"
            )


class FileUploader:
    def __init__(self, upload_dir: Path, max_file_size_mb: int = 100):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024

    def validate_and_save(self, file_storage) -> Tuple[Path, Dict[str, Any]]:
        """
        Validates uploaded audio file, saves it, and creates a browser-safe 16-bit PCM preview.
        Returns: (saved_file_path, metadata_dict)
        """
        raw_filename = Path(file_storage.filename).name
        ext = Path(raw_filename).suffix.lower()

        if ext not in SUPPORTED_AUDIO_EXTENSIONS:
            supported_str = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
            raise UploadValidationError(
                f"Unsupported file format '{ext}'. Supported formats: {supported_str}"
            )

        save_path = self.upload_dir / raw_filename
        file_storage.save(str(save_path))

        file_size = save_path.stat().st_size
        if file_size > self.max_file_size_bytes:
            save_path.unlink(missing_ok=True)
            raise UploadValidationError(
                f"File exceeds maximum allowed size of {self.max_file_size_bytes // (1024 * 1024)} MB."
            )

        if file_size < 32:
            save_path.unlink(missing_ok=True)
            raise UploadValidationError("File is empty or corrupted.")

        # Decode audio to verify format and extract metadata
        try:
            data, sr = load_audio_universal(save_path)
        except Exception as e:
            save_path.unlink(missing_ok=True)
            raise UploadValidationError(f"Audio decoding error: {e}")

        channels = 1 if data.ndim == 1 else data.shape[1]
        n_samples = len(data)
        duration_s = round(n_samples / max(1, sr), 2)

        # Generate a standard web-safe 16-bit PCM WAV preview for the raw audio player
        preview_filename = f"raw_preview_{save_path.stem}.wav"
        preview_path = self.upload_dir / preview_filename
        try:
            safe_preview = np.clip(data, -1.0, 1.0)
            sf.write(
                str(preview_path),
                (safe_preview * 32767.0).astype(np.int16),
                sr,
                subtype="PCM_16"
            )
        except Exception:
            # If preview write fails, fallback to using the original save_path
            preview_filename = raw_filename

        meta = {
            "filename": raw_filename,
            "preview_filename": preview_filename,
            "duration_s": duration_s,
            "sample_rate": sr,
            "channels": channels,
            "format": ext.lstrip(".").upper(),
            "size_bytes": file_size,
        }
        return save_path, meta
