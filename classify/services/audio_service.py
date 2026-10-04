"""
audio_service.py
================
Audio service wrapping validation and Tier 1 feature extraction.
Part of Phase 2 for CLASSIFY.
"""

from classify.audio.audio_processing import analyze_audio
from classify.audio.validation import validate_audio_file


def process_audio(file_path, allowed_extensions=None, max_size_bytes=None):
    """
    Validates the audio file and extracts comprehensive Tier 1 acoustic features.

    Parameters:
        file_path (str): File path to uploaded audio.
        allowed_extensions (set): Allowed file extensions.
        max_size_bytes (int): Maximum size in bytes.

    Returns:
        tuple: (technical_features_dict, validation_info_dict)
    """
    validation_info = validate_audio_file(
        file_path,
        max_size_bytes=max_size_bytes,
        allowed_extensions=allowed_extensions,
    )
    features = analyze_audio(file_path)
    return features, validation_info
