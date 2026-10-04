"""
test_audio_segmentation.py
==========================
Tests for multi-window audio segmentation, window placement,
and aggregated 89-dimensional feature extraction.
Part of Phase 3 for CLASSIFY.
"""

import numpy as np
import pytest

from classify.audio.segmentation import (
    compute_segment_windows,
    extract_segmented_features,
)
from classify.ml.feature_pipeline import FEATURE_NAMES


def test_segment_windows_short_audio():
    """Short audio (<= 30s) should yield a single window covering its duration."""
    windows_3s = compute_segment_windows(3.0, num_segments=3, segment_length=30.0)
    assert len(windows_3s) == 1
    assert windows_3s[0] == (0.0, 3.0)

    windows_10s = compute_segment_windows(9.5, num_segments=3, segment_length=30.0)
    assert len(windows_10s) == 1
    assert windows_10s[0] == (0.0, 9.5)


def test_segment_windows_medium_audio():
    """Medium audio (e.g. 45s) should adapt windows gracefully within duration bounds."""
    windows = compute_segment_windows(45.0, num_segments=3, segment_length=30.0)
    assert len(windows) == 3
    for offset, length in windows:
        assert offset >= 0.0
        assert length <= 30.0
        assert offset + length <= 45.01


def test_segment_windows_long_audio():
    """Long audio (e.g. 210s / 3.5 min) should distribute windows around 20%, 50%, and 80%."""
    windows = compute_segment_windows(210.0, num_segments=3, segment_length=30.0)
    assert len(windows) == 3
    for offset, length in windows:
        assert offset >= 0.0
        assert pytest.approx(length, 0.1) == 30.0
        assert offset + length <= 210.0

    # Verify windows are separated and ordered
    assert windows[0][0] < windows[1][0] < windows[2][0]
    # Check centers roughly at 20%, 50%, 80%
    c0 = windows[0][0] + windows[0][1] / 2.0
    c1 = windows[1][0] + windows[1][1] / 2.0
    c2 = windows[2][0] + windows[2][1] / 2.0
    assert pytest.approx(c0, 5.0) == 0.20 * 210.0
    assert pytest.approx(c1, 5.0) == 0.50 * 210.0
    assert pytest.approx(c2, 5.0) == 0.80 * 210.0


def test_segmented_features_vector_contract_and_determinism(create_synthetic_audio):
    """Verifies that segmented extraction returns an 89-length vector and is deterministic."""
    audio_path = create_synthetic_audio(duration=4.0, sr=22050, freq=440.0)

    vec1, names1, segs1 = extract_segmented_features(audio_path, num_segments=3)
    vec2, names2, segs2 = extract_segmented_features(audio_path, num_segments=3)

    assert isinstance(vec1, np.ndarray)
    assert vec1.shape == (89,)
    assert len(names1) == 89
    assert names1 == FEATURE_NAMES

    # Determinism
    np.testing.assert_allclose(vec1, vec2, rtol=1e-5)


def test_segmented_features_median_vs_mean(create_synthetic_audio):
    """Verifies both mean and median aggregations produce valid 89-dim vectors."""
    audio_path = create_synthetic_audio(duration=6.0, sr=22050, freq=330.0)

    vec_mean, _, _ = extract_segmented_features(audio_path, aggregation="mean")
    vec_median, _, _ = extract_segmented_features(audio_path, aggregation="median")

    assert vec_mean.shape == (89,)
    assert vec_median.shape == (89,)
    assert not np.isnan(vec_mean).any()
    assert not np.isnan(vec_median).any()
