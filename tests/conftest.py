"""
conftest.py
===========
Shared pytest fixtures for the CLASSIFY test suite.
"""

import io
import os
import shutil
import tempfile
import numpy as np
import pytest
import soundfile as sf
from classify import create_app
from config import TestingConfig


@pytest.fixture
def app():
    """Create and configure a clean Flask application instance for testing."""
    temp_upload_dir = tempfile.mkdtemp(prefix="classify_test_uploads_")

    class CustomTestingConfig(TestingConfig):
        UPLOAD_FOLDER = temp_upload_dir

    application = create_app(CustomTestingConfig)
    application.testing = True

    yield application

    # Cleanup temporary upload folder after test run
    if os.path.exists(temp_upload_dir):
        shutil.rmtree(temp_upload_dir, ignore_errors=True)


@pytest.fixture
def client(app):
    """Test client for HTTP requests."""
    return app.test_client()


@pytest.fixture
def runner(app):
    """CLI test runner."""
    return app.test_cli_runner()


@pytest.fixture
def create_synthetic_audio(tmp_path):
    """
    Factory fixture to create synthetic audio WAV files for testing.
    """
    def _create(duration=3.0, sr=22050, freq=440.0, with_beats=True):
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        signal = 0.5 * np.sin(2 * np.pi * freq * t)

        if with_beats:
            # Add rhythmic pulses to simulate tempo (~120 BPM)
            beat_interval = int(sr * 0.5)
            for b in range(0, len(signal), beat_interval):
                end_idx = min(b + 500, len(signal))
                signal[b:end_idx] += 0.8 * np.sin(2 * np.pi * 1000 * np.linspace(0, 0.02, end_idx - b))

        # Normalize
        max_val = np.max(np.abs(signal))
        if max_val > 0:
            signal = signal / max_val

        audio_file = tmp_path / "test_tone.wav"
        sf.write(str(audio_file), signal, sr)
        return str(audio_file)

    return _create


@pytest.fixture
def synthetic_wav_bytes():
    """Generates an in-memory 2-second WAV file as BytesIO."""
    sr = 22050
    t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
    audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)

    wav_io = io.BytesIO()
    sf.write(wav_io, audio_data, sr, format="WAV")
    wav_io.seek(0)
    return wav_io
