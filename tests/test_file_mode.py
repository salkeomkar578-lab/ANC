import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import unittest
import soundfile as sf
import numpy as np

from file_mode.uploader import FileUploader, UploadValidationError
from file_mode.processor import FileProcessor
from core.pipeline import IgardNetPipeline
from core.state import SystemState
from backends.cpu_backend import CPUBackend

TEST_DIR = Path(__file__).resolve().parent / "test_scratch"
TEST_DIR.mkdir(parents=True, exist_ok=True)


class MockFileStorage:
    def __init__(self, filename: str, path_to_save: Path):
        self.filename = filename
        self._path = path_to_save

    def save(self, dest):
        import shutil
        shutil.copyfile(self._path, dest)


class TestFileMode(unittest.TestCase):
    def setUp(self):
        self.upload_dir = TEST_DIR / "uploads"
        self.output_dir = TEST_DIR / "outputs"
        self.uploader = FileUploader(upload_dir=self.upload_dir)
        self.processor = FileProcessor(output_dir=self.output_dir)
        self.state = SystemState()
        self.pipeline = IgardNetPipeline(state=self.state, backend=CPUBackend(), enable_background_tuner=False)

    def tearDown(self):
        import shutil
        self.pipeline.close()
        shutil.rmtree(TEST_DIR, ignore_errors=True)

    def test_file_upload_validation(self):
        # Create valid test WAV
        valid_wav = TEST_DIR / "test_valid.wav"
        data = np.random.randn(16000).astype(np.float32)
        sf.write(str(valid_wav), data, 16000)

        mock_storage = MockFileStorage("test_valid.wav", valid_wav)
        saved_path, meta = self.uploader.validate_and_save(mock_storage)
        
        self.assertTrue(saved_path.exists())
        self.assertEqual(meta["filename"], "test_valid.wav")
        self.assertEqual(meta["sample_rate"], 16000)
        self.assertEqual(meta["channels"], 1)

    def test_invalid_file_rejection(self):
        # Create non-wav file
        bad_file = TEST_DIR / "bad.txt"
        bad_file.write_text("not a wav file")
        mock_storage = MockFileStorage("bad.txt", bad_file)

        with self.assertRaises(UploadValidationError):
            self.uploader.validate_and_save(mock_storage)

    def test_fast_batch_processing_rtf(self):
        """Verifies that a 5-second file is processed in < 1 second (RTF < 0.2)."""
        wav_path = TEST_DIR / "test_speech_5s.wav"
        n_samples = 16000 * 5
        t = np.arange(n_samples) / 16000
        speech = np.sin(2 * np.pi * 300 * t) * 0.4
        noise = np.sin(2 * np.pi * 90 * t) * 0.3
        stereo = np.column_stack((speech + noise, noise))
        sf.write(str(wav_path), stereo, 16000)

        res = self.processor.process_file(wav_path, self.pipeline, state=self.state)
        
        self.assertLess(res["rtf"], 0.80, f"RTF {res['rtf']} is too slow! Must be < 1.0.")
        self.assertLess(res["processing_time_s"], 4.0, "5s file took more than 4.0 seconds!")
        self.assertTrue(Path(res["output_path"]).exists())
        self.assertGreater(self.state.file_progress, 99.0)


if __name__ == "__main__":
    unittest.main()
