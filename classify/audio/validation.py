"""
validation.py
=============
Input validation for uploaded audio files.
Checks extension, size, decodability, duration boundaries, and silence.
Part of Phase 1 for CLASSIFY.
"""

import os
import soundfile as sf
import numpy as np
import librosa


class AudioValidationError(Exception):
    """Base exception for all audio validation errors."""
    pass


class UnsupportedAudioFormatError(AudioValidationError):
    """Raised when the audio format/extension is not supported."""
    pass


class OversizedFileError(AudioValidationError):
    """Raised when the file exceeds the maximum allowed size."""
    pass


class CorruptAudioError(AudioValidationError):
    """Raised when the audio file cannot be decoded or read."""
    pass


class SilentAudioError(AudioValidationError):
    """Raised when the audio is silent or near-silent."""
    pass


class AudioDurationError(AudioValidationError):
    """Raised when the audio is too short or too long."""
    pass


def validate_audio_file(
    file_path,
    max_size_bytes=25 * 1024 * 1024,
    allowed_extensions=None,
    min_duration=1.0,
    max_duration=600.0,
):
    """
    Validates an audio file on disk for integrity, duration, and acoustic content.

    Parameters:
        file_path (str): Path to audio file.
        max_size_bytes (int): Maximum allowed file size in bytes.
        allowed_extensions (set/list): Allowed file extensions (e.g., {'wav', 'mp3'}).
        min_duration (float): Minimum duration in seconds (default: 1.0s).
        max_duration (float): Maximum allowed duration in seconds (default: 600.0s).

    Returns:
        dict: Metadata with duration, sample_rate, channels, file_size_kb.

    Raises:
        AudioValidationError: Or one of its subclasses if validation fails.
    """
    if not os.path.exists(file_path):
        raise AudioValidationError(f"Audio file does not exist: {file_path}")

    # 1. Check file size
    file_size = os.path.getsize(file_path)
    if file_size == 0:
        raise CorruptAudioError("Uploaded audio file is empty (0 bytes).")
    if max_size_bytes and file_size > max_size_bytes:
        max_mb = round(max_size_bytes / (1024 * 1024), 1)
        actual_mb = round(file_size / (1024 * 1024), 1)
        raise OversizedFileError(
            f"File size ({actual_mb} MB) exceeds maximum allowed limit ({max_mb} MB)."
        )

    # 2. Check extension
    if allowed_extensions:
        ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
        allowed_normalized = {e.lower().lstrip(".") for e in allowed_extensions}
        if ext not in allowed_normalized:
            raise UnsupportedAudioFormatError(
                f"Unsupported audio format '.{ext}'. Allowed formats: {', '.join(sorted(allowed_normalized))}."
            )

    # 3. Check decodability & read basic audio header
    try:
        # soundfile is fast for headers
        info = sf.info(file_path)
        duration = float(info.duration)
        sample_rate = int(info.samplerate)
        channels = int(info.channels)
    except Exception as sf_err:
        # Fallback to librosa if soundfile header reading fails (e.g. some mp3 variants)
        try:
            duration = float(librosa.get_duration(path=file_path))
            sample_rate = 22050
            channels = 1
        except Exception as librosa_err:
            raise CorruptAudioError(
                f"Audio file is corrupt or unreadable: {str(sf_err)} / {str(librosa_err)}"
            )

    # 4. Check duration boundaries
    if duration < min_duration:
        raise AudioDurationError(
            f"Audio clip is too short ({duration:.2f}s). Minimum required duration is {min_duration:.1f}s."
        )
    if duration > max_duration:
        raise AudioDurationError(
            f"Audio clip is too long ({duration:.1f}s). Maximum allowed duration is {max_duration:.1f}s."
        )

    # 5. Check for silence or near-silence
    # Load first 10 seconds to verify acoustic presence
    try:
        y, sr = librosa.load(file_path, sr=22050, duration=10.0, mono=True)
    except Exception as e:
        raise CorruptAudioError(f"Failed to decode audio signal: {str(e)}")

    if len(y) == 0:
        raise CorruptAudioError("Audio signal contains no samples.")

    max_amp = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(y**2)))

    if max_amp < 1e-4 or rms < 1e-4:
        raise SilentAudioError(
            "Audio file contains silent or near-silent audio. Please upload an audio file containing audible music."
        )

    return {
        "duration": round(duration, 2),
        "sample_rate": sample_rate,
        "channels": channels,
        "file_size_kb": round(file_size / 1024, 2),
        "rms_probe": round(rms, 6),
        "peak_amplitude": round(max_amp, 4),
    }
