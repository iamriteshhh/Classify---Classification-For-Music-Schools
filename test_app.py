import io
import unittest
import numpy as np
import soundfile as sf
import os
import shutil
import tempfile
from classify import create_app
from config import TestingConfig

class TestClassifyPhase1(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="classify_unittest_uploads_")
        
        class CustomTestingConfig(TestingConfig):
            UPLOAD_FOLDER = self.temp_dir

        self.flask_app = create_app(CustomTestingConfig)
        self.app = self.flask_app.test_client()
        self.app.testing = True

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)


    def test_home_page(self):
        """Test that the homepage loads successfully with status code 200."""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"CLASSIFY", response.data)
        self.assertIn(b"Music Classification System for Music Schools", response.data)
        self.assertIn(b"Upload Music File", response.data)

    def test_audio_analysis_pipeline(self):
        """Test uploading a synthetic WAV audio file and checking the analysis response."""
        # Create an in-memory 2-second WAV file
        sr = 22050
        t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
        audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)

        wav_io = io.BytesIO()
        sf.write(wav_io, audio_data, sr, format='WAV')
        wav_io.seek(0)

        data = {
            'audio_file': (wav_io, 'test_unit_audio.wav')
        }

        response = self.app.post('/analyze', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Audio Analysis &amp; Prototype Classification", response.data)
        self.assertIn(b"Extracted Acoustic Properties", response.data)
        self.assertIn(b"PROTOTYPE PREDICTION:", response.data)
        self.assertIn(b"Tempo", response.data)
        self.assertIn(b"RMS Energy", response.data)
        self.assertIn(b"Spectral Centroid", response.data)

    def test_invalid_file_extension(self):
        """Test that uploading a non-audio file redirects and flashes an error."""
        text_file = io.BytesIO(b"This is not audio content.")
        data = {
            'audio_file': (text_file, 'test.txt')
        }
        response = self.app.post('/analyze', data=data, content_type='multipart/form-data', follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Invalid file format", response.data)

if __name__ == '__main__':
    unittest.main()
