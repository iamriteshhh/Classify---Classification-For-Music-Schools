"""
test_rnb_jazz_regression.py
===========================
Regression test suite ensuring:
1. R&B is accurately classified and does not sink into Pop/Country.
2. Jazz is accurately classified and does not collapse into Country.
3. Disco and high-energy Dance Pop do not collapse into R&B.
4. Post-prediction confusion guards correctly arbitrate borderline acoustic profiles.
Part of Phase 7 for CLASSIFY.
"""

import os
import pytest
import numpy as np
from classify.ml.inference import predict_genre, apply_confusion_guards
from classify.ml.feature_pipeline import FEATURE_NAMES

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_confusion_guard_disco_vs_rnb_arbitration():
    """Verifies that high-tempo, percussive dance profiles are arbitrated toward Disco."""
    # Synthetic feature vector matching Disco (high tempo, low harmonic ratio, high onset rate)
    fvec = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    fnames = FEATURE_NAMES

    # Set acoustic features
    tempo_idx = fnames.index("tempo")
    harm_idx = fnames.index("harmonic_ratio")
    onset_idx = fnames.index("onset_rate")

    fvec[tempo_idx] = 124.0
    fvec[harm_idx] = 0.38
    fvec[onset_idx] = 3.6

    # Suppose model initially gave R&B 0.35 and Disco 0.25
    initial_probs = {
        "R&B": 0.35,
        "Disco": 0.25,
        "Pop": 0.20,
        "Hip-Hop": 0.20,
    }

    calibrated = apply_confusion_guards(initial_probs, fvec, fnames)

    # After arbitration, Disco must be elevated over R&B
    assert calibrated["Disco"] > calibrated["R&B"]
    assert pytest.approx(sum(calibrated.values()), 0.01) == 1.0


def test_confusion_guard_rnb_ballad_arbitration():
    """Verifies that slow, lush vocal profiles are arbitrated toward R&B rather than Pop/Country."""
    fvec = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    fnames = FEATURE_NAMES

    tempo_idx = fnames.index("tempo")
    harm_idx = fnames.index("harmonic_ratio")
    vocal_idx = fnames.index("vocal_probability")
    bass_idx = fnames.index("bass_spectral_centroid")

    fvec[tempo_idx] = 85.0
    fvec[harm_idx] = 0.58
    fvec[vocal_idx] = 0.65
    fvec[bass_idx] = 140.0  # Warm sub-bass

    # Model was slightly pulled towards Pop
    initial_probs = {
        "Pop": 0.32,
        "Country": 0.15,
        "R&B": 0.28,
        "Soul": 0.10,
        "Jazz": 0.15,
    }

    calibrated = apply_confusion_guards(initial_probs, fvec, fnames)

    assert calibrated["R&B"] > calibrated["Pop"]
    assert pytest.approx(sum(calibrated.values()), 0.01) == 1.0


def test_confusion_guard_jazz_vs_country_arbitration():
    """Verifies that rich chromatic/modal tonnetz variance arbitrates toward Jazz."""
    fvec = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    fnames = FEATURE_NAMES

    harm_idx = fnames.index("harmonic_ratio")
    stability_idx = fnames.index("tempo_stability")
    t0_idx = fnames.index("tonnetz_var_0")
    t1_idx = fnames.index("tonnetz_var_1")

    fvec[harm_idx] = 0.55
    fvec[stability_idx] = 0.06  # Swing rubato variance
    fvec[t0_idx] = 0.008
    fvec[t1_idx] = 0.007  # High tonnetz variance sum >= 0.015

    initial_probs = {
        "Country": 0.30,
        "Jazz": 0.25,
        "Blues": 0.20,
        "Pop": 0.25,
    }

    calibrated = apply_confusion_guards(initial_probs, fvec, fnames)

    assert calibrated["Jazz"] > calibrated["Country"]
    assert pytest.approx(sum(calibrated.values()), 0.01) == 1.0


def test_disco_audio_file_not_classified_as_rnb():
    """Verifies a real Disco track is not classified as R&B."""
    track_path = os.path.join(BASE_DIR, "datasets", "processed", "gtzan_disco_disco.00001.wav")
    if not os.path.exists(track_path):
        pytest.skip("Test track not present on disk.")

    res = predict_genre(track_path)
    assert res["predicted_genre"] != "R&B", f"Disco track incorrectly classified as R&B: {res}"
    assert res["probabilities"].get("R&B", 0.0) < 0.35


def test_jazz_audio_file_not_classified_as_country():
    """Verifies a real Jazz track is not classified as Country."""
    track_path = os.path.join(BASE_DIR, "datasets", "processed", "gtzan_jazz_jazz.00073.wav")
    if not os.path.exists(track_path):
        pytest.skip("Test track not present on disk.")

    res = predict_genre(track_path)
    assert res["predicted_genre"] != "Country", f"Jazz track incorrectly classified as Country: {res}"
    assert res["predicted_genre"] == "Jazz" or res["probabilities"].get("Jazz", 0.0) >= 0.20


def test_country_audio_file_not_classified_as_jazz():
    """Verifies a real Country track is not classified as Jazz."""
    track_path = os.path.join(BASE_DIR, "datasets", "processed", "gtzan_country_country.00030.wav")
    if not os.path.exists(track_path):
        pytest.skip("Test track not present on disk.")

    res = predict_genre(track_path)
    assert res["predicted_genre"] != "Jazz", f"Country track incorrectly classified as Jazz: {res}"
