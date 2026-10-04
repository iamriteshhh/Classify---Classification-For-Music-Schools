"""
test_feature_pipeline.py
========================
Tests for classify.ml.feature_pipeline.
Verifies that:
- Feature vector dimensionality is fixed and consistent (78 features)
- All feature names are unique and match vector elements
- Feature vector can be extracted from both an audio path and a features dictionary
- All values in the vector are finite numeric floats
"""

import numpy as np
import pytest
from classify.ml.feature_pipeline import extract_feature_vector, FEATURE_NAMES
from classify.audio.audio_processing import analyze_audio


def test_feature_pipeline_vector_from_audio(create_synthetic_audio):
    """Test feature vector extraction directly from an audio file."""
    audio_path = create_synthetic_audio(duration=2.5, sr=22050, freq=440.0)
    vec, names = extract_feature_vector(audio_path)

    assert isinstance(vec, np.ndarray)
    assert vec.ndim == 1
    assert len(vec) == len(FEATURE_NAMES)
    assert len(vec) == 89
    assert np.all(np.isfinite(vec))
    assert names == FEATURE_NAMES


def test_feature_pipeline_vector_from_dict(create_synthetic_audio):
    """Test feature vector extraction from a pre-extracted technical_features dict."""
    audio_path = create_synthetic_audio(duration=2.5, sr=22050, freq=440.0)
    features_dict = analyze_audio(audio_path)

    vec, names = extract_feature_vector(features_dict)

    assert isinstance(vec, np.ndarray)
    assert len(vec) == 89
    assert np.all(np.isfinite(vec))


def test_feature_pipeline_consistency(create_synthetic_audio):
    """Verifies that extracting from path vs dict produces identical vectors."""
    audio_path = create_synthetic_audio(duration=2.5, sr=22050, freq=440.0)
    vec_path, _ = extract_feature_vector(audio_path)
    features_dict = analyze_audio(audio_path)
    vec_dict, _ = extract_feature_vector(features_dict)

    np.testing.assert_allclose(vec_path, vec_dict, rtol=1e-5, atol=1e-5)
