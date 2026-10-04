"""
test_audio_validation.py
========================
Tests for classify.audio.validation module.
Covers:
- Valid file passing validation
- Missing file
- Empty file (0 bytes)
- Corrupt audio file
- Silent audio signal
- Too-short audio clip (< 1.0s)
- Oversized file check
- Unsupported extension check
"""

import os
import numpy as np
import pytest
import soundfile as sf
from classify.audio.validation import (
    validate_audio_file,
    AudioValidationError,
    UnsupportedAudioFormatError,
    OversizedFileError,
    CorruptAudioError,
    SilentAudioError,
    AudioDurationError,
)


def test_validation_valid_file(create_synthetic_audio):
    """A valid synthetic audio file should pass validation and return audio metadata."""
    audio_path = create_synthetic_audio(duration=2.5, sr=22050, freq=440.0)
    info = validate_audio_file(audio_path, allowed_extensions={"wav"})

    assert info["duration"] == pytest.approx(2.5, rel=0.1)
    assert info["sample_rate"] == 22050
    assert info["channels"] >= 1
    assert info["file_size_kb"] > 0
    assert info["rms_probe"] > 0.001


def test_validation_missing_file():
    """Missing file should raise AudioValidationError."""
    with pytest.raises(AudioValidationError) as exc:
        validate_audio_file("non_existent_file.wav")
    assert "does not exist" in str(exc.value)


def test_validation_empty_file(tmp_path):
    """An empty file (0 bytes) should raise CorruptAudioError."""
    empty_file = tmp_path / "empty.wav"
    empty_file.write_bytes(b"")

    with pytest.raises(CorruptAudioError) as exc:
        validate_audio_file(str(empty_file))
    assert "empty" in str(exc.value)


def test_validation_corrupt_file(tmp_path):
    """Garbage bytes in a .wav file should raise CorruptAudioError."""
    corrupt_file = tmp_path / "corrupt.wav"
    corrupt_file.write_bytes(b"NOT_A_REAL_WAV_HEADER_CORRUPT_BYTES_DATA")

    with pytest.raises(CorruptAudioError):
        validate_audio_file(str(corrupt_file))


def test_validation_silent_audio(tmp_path):
    """A WAV file containing complete silence (all zeros) should raise SilentAudioError."""
    sr = 22050
    silent_data = np.zeros(int(sr * 2.0), dtype=np.float32)
    silent_file = tmp_path / "silent.wav"
    sf.write(str(silent_file), silent_data, sr)

    with pytest.raises(SilentAudioError) as exc:
        validate_audio_file(str(silent_file))
    assert "silent" in str(exc.value).lower()


def test_validation_too_short(tmp_path):
    """An audio clip shorter than min_duration (e.g. 0.4s) should raise AudioDurationError."""
    sr = 22050
    short_data = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.4, int(sr * 0.4)))
    short_file = tmp_path / "short.wav"
    sf.write(str(short_file), short_data, sr)

    with pytest.raises(AudioDurationError) as exc:
        validate_audio_file(str(short_file), min_duration=1.0)
    assert "too short" in str(exc.value).lower()


def test_validation_oversized_file(tmp_path):
    """A file exceeding max_size_bytes should raise OversizedFileError."""
    sr = 22050
    audio_data = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 1.5, int(sr * 1.5)))
    test_file = tmp_path / "oversize_test.wav"
    sf.write(str(test_file), audio_data, sr)

    # Set threshold lower than the file size
    file_size = os.path.getsize(str(test_file))
    with pytest.raises(OversizedFileError) as exc:
        validate_audio_file(str(test_file), max_size_bytes=file_size - 100)
    assert "exceeds maximum allowed limit" in str(exc.value)


def test_validation_unsupported_extension(tmp_path):
    """A file with an extension not in allowed_extensions should raise UnsupportedAudioFormatError."""
    test_file = tmp_path / "track.flac"
    test_file.write_bytes(b"dummy")

    with pytest.raises(UnsupportedAudioFormatError) as exc:
        validate_audio_file(str(test_file), allowed_extensions={"wav", "mp3"})
    assert "Unsupported audio format" in str(exc.value)
