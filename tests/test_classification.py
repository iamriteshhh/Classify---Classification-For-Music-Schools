"""
test_classification.py
======================
Tests for real ML genre inference (classify.ml.inference) and explainability.
Verifies real confidence scores, probability distributions, alternatives,
and explainability factors.
Part of Phase 2 for CLASSIFY.
"""

import pytest
from classify.ml.inference import predict_genre
from classifier.prototype_classifier import classify_prototype


def test_real_ml_prediction_schema(create_synthetic_audio):
    """Verifies that predict_genre returns real ML predictions, confidence values, and explanations."""
    audio_path = create_synthetic_audio(duration=2.5, sr=22050, freq=440.0)
    result = predict_genre(audio_path)

    # 1. Output keys
    assert "predicted_genre" in result
    assert "confidence" in result
    assert "confidence_percent" in result
    assert "is_low_confidence" in result
    assert "probabilities" in result
    assert "alternatives" in result
    assert "explanation" in result
    assert "explanation_factors" in result
    assert "model_name" in result
    assert result["is_prototype"] is False

    # 2. Probability & Confidence bounds
    assert 0.0 <= result["confidence"] <= 1.0
    assert isinstance(result["confidence"], float)

    # Total probabilities across all classes must sum to approximately 1.0
    total_prob = sum(result["probabilities"].values())
    assert pytest.approx(total_prob, 0.02) == 1.0

    # 3. Alternatives structure
    assert isinstance(result["alternatives"], list)
    for alt in result["alternatives"]:
        assert "genre" in alt
        assert "confidence" in alt
        assert 0.0 <= alt["confidence"] <= 1.0

    # 4. Explanation structure
    assert len(result["explanation"]) > 20
    assert isinstance(result["explanation_factors"], list)


def test_real_ml_prediction_from_features_dict():
    """Verifies predict_genre works directly with precomputed feature dictionaries."""
    mock_features = {
        "tempo": 138.0,
        "rms_mean": 0.22,
        "rms_var": 0.01,
        "zcr_mean": 0.09,
        "zcr_var": 0.002,
        "spectral_centroid_mean": 2600.0,
        "spectral_centroid_var": 4000.0,
        "spectral_bandwidth_mean": 2100.0,
        "spectral_rolloff_mean": 4800.0,
        "spectral_contrast_mean": 22.0,
        "spectral_contrast_bands": [18.0, 20.0, 22.0, 24.0, 25.0, 26.0, 28.0],
        "mfcc_means": [10.0] * 20,
        "mfcc_vars": [2.0] * 20,
        "chroma_means": [0.5] * 12,
        "chroma_vars": [0.1] * 12,
        "onset_strength_mean": 1.2,
        "onset_rate": 3.5,
        "mel_spectrogram_mean": 0.8,
        "mel_spectrogram_var": 0.2,
    }
    result = predict_genre(mock_features)

    assert isinstance(result["predicted_genre"], str)
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["is_prototype"] is False


def test_prototype_classifier_legacy_reference():
    """Verify that the prototype reference classifier continues to function for historical comparisons."""
    res = classify_prototype({"rms_energy": 0.15, "spectral_centroid": 2400, "tempo": 130})
    assert res["predicted_genre"] == "Rock"
    assert res["is_prototype"] is True


def test_subgenre_and_multilabel_classification_schema(create_synthetic_audio):
    """Verifies that predict_genre returns hierarchical subgenre and multi-label genre tags."""
    audio_path = create_synthetic_audio(duration=2.5, sr=22050, freq=440.0)
    result = predict_genre(audio_path)

    # Subgenre keys
    assert "subgenre" in result
    assert "subgenre_name" in result
    assert "subgenre_slug" in result
    if result["subgenre"] is not None:
        assert "name" in result["subgenre"]
        assert "slug" in result["subgenre"]
        assert "description" in result["subgenre"]
        assert 0.0 <= result["subgenre"]["confidence"] <= 1.0

    # Multi-label tags
    assert "genre_tags" in result
    assert "secondary_genres" in result
    assert len(result["genre_tags"]) >= 1
    primary_tag = result["genre_tags"][0]
    assert primary_tag["is_primary"] is True
    assert primary_tag["genre"] == result["predicted_genre"]


def test_subgenre_classifier_direct_profiles():
    """Verifies direct subgenre classification across parent genres."""
    from classify.ml.subgenre_classifier import predict_subgenre, SUBGENRE_PROFILES

    # 1. Rock with high energy and contrast -> Hard Rock
    hard_rock = predict_subgenre("Rock", feature_dict={"rms_mean": 0.18, "spectral_contrast_mean": 24.0, "tempo": 135.0})
    assert hard_rock["name"] in ["Hard Rock", "Alternative Rock", "Classic Rock"]

    # 2. Pop with fast dance tempo -> Dance Pop
    dance_pop = predict_subgenre("Pop", feature_dict={"tempo": 126.0, "onset_strength_mean": 1.5, "spectral_centroid_mean": 2500.0})
    assert dance_pop["slug"] in ["dance-pop", "synth-pop", "indie-pop"]

    # 3. Jazz with fast tempo and high onset rate -> Bebop
    bebop = predict_subgenre("Jazz", feature_dict={"tempo": 150.0, "onset_rate": 3.8, "chroma_stft_var": 0.08})
    assert bebop["name"] == "Bebop"

    # 4. Metal with high velocity -> Thrash Metal
    thrash = predict_subgenre("Metal", feature_dict={"tempo": 160.0, "onset_rate": 4.0, "spectral_centroid_mean": 2800.0})
    assert thrash["name"] == "Thrash Metal"

    # 5. Disco -> Nu-Disco
    disco = predict_subgenre("Disco")
    assert disco["name"] == "Nu-Disco"

