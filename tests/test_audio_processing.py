"""
test_audio_processing.py
========================
Tests for audio loading and expanded Tier 1 acoustic feature extraction.
Covers:
- All required Tier 1 feature keys
- Full 20 MFCC coefficients (means and variances)
- 12 Chroma pitch classes (means and variances)
- Spectral centroid, bandwidth, rolloff, contrast
- Dynamic measures (RMS mean & var, ZCR mean & var, tempo)
- Onset strength and onset rate
- Mel-spectrogram statistics
"""

import pytest
from audio.audio_processing import analyze_audio


def test_analyze_audio_expanded_tier1_features(create_synthetic_audio):
    """Test full Tier 1 feature extraction on a valid synthetic WAV file."""
    audio_path = create_synthetic_audio(duration=3.0, sr=22050, freq=440.0, with_beats=True)
    features = analyze_audio(audio_path)

    # 1. Essential scalar keys
    core_keys = {
        "file_name",
        "file_size_kb",
        "duration",
        "sample_rate",
        "tempo",
        "rms_energy",
        "rms_mean",
        "rms_var",
        "zero_crossing_rate",
        "zcr_mean",
        "zcr_var",
        "spectral_centroid",
        "spectral_centroid_mean",
        "spectral_centroid_var",
        "spectral_bandwidth_mean",
        "spectral_bandwidth_var",
        "spectral_rolloff_mean",
        "spectral_rolloff_var",
        "spectral_contrast_mean",
        "spectral_contrast_var",
        "mfcc_mean",
        "onset_strength_mean",
        "onset_rate",
        "mel_spectrogram_mean",
    }
    assert core_keys.issubset(features.keys())

    # 2. Check full MFCC vectors (20 means + 20 variances = 40 features)
    assert "mfcc_means" in features
    assert "mfcc_vars" in features
    assert len(features["mfcc_means"]) == 20
    assert len(features["mfcc_vars"]) == 20
    assert all(isinstance(val, float) for val in features["mfcc_means"])
    assert all(isinstance(val, float) for val in features["mfcc_vars"])

    # 3. Check Chroma vectors (12 pitch classes)
    assert "chroma_means" in features
    assert "chroma_vars" in features
    assert len(features["chroma_means"]) == 12
    assert len(features["chroma_vars"]) == 12

    # 4. Check Spectral Contrast frequency bands (7 bands)
    assert "spectral_contrast_bands" in features
    assert len(features["spectral_contrast_bands"]) == 7

    # 5. Check value reasonableness
    assert pytest.approx(features["duration"], 0.2) == 3.0
    assert features["sample_rate"] == 22050
    assert features["tempo"] > 0
    assert features["rms_mean"] > 0
    assert features["rms_var"] >= 0
    assert features["zcr_mean"] >= 0
    assert features["spectral_centroid_mean"] > 0
    assert features["spectral_bandwidth_mean"] > 0
    assert features["onset_strength_mean"] > 0


def test_analyze_audio_missing_file():
    """analyze_audio should raise FileNotFoundError on a non-existent file."""
    with pytest.raises(FileNotFoundError):
        analyze_audio("non_existent_audio_file.wav")
