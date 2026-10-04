"""Tests for deep audio embedding extraction pipeline."""

import numpy as np
import pytest

from classify.ml.embeddings import AudioEmbeddingExtractor, get_audio_embedding


def test_embedding_dimensions_and_norm():
    """Verify embedding extractor generates 512-dimensional L2-normalized vector."""
    sr = 22050
    # 2 seconds of 440Hz sine wave
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    extractor = AudioEmbeddingExtractor(backend="spectral_projection", embedding_dim=512)
    assert extractor.get_embedding_dimension() == 512

    embedding = extractor.extract(audio, sr=sr)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (512,)
    assert embedding.dtype == np.float32

    # Verify L2 norm is approximately 1.0
    norm = np.linalg.norm(embedding)
    assert abs(norm - 1.0) < 1e-3


def test_get_audio_embedding_convenience_function():
    """Verify global singleton convenience function."""
    sr = 22050
    audio = np.random.uniform(-0.1, 0.1, sr * 1).astype(np.float32)
    embedding = get_audio_embedding(audio, sr=sr)

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (512,)
    assert np.all(np.isfinite(embedding))


def test_empty_audio_handling():
    """Verify empty audio gracefully returns zero vector."""
    extractor = AudioEmbeddingExtractor(embedding_dim=512)
    empty = np.array([], dtype=np.float32)
    embedding = extractor.extract(empty)
    assert embedding.shape == (512,)
    assert np.all(embedding == 0.0)
