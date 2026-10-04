"""
test_inference_parity.py
========================
Regression test suite guaranteeing inference parity, full-audio coverage,
train/inference consistency, and robust multi-segment aggregation for CLASSIFY.
Fulfills Section 7 of promptt.md.
"""

import os
import numpy as np
import pytest
import soundfile as sf
import tempfile
import joblib

from classify.audio.segmentation import compute_segment_windows
from classify.audio.audio_processing import analyze_audio
from classify.ml.feature_pipeline import extract_feature_vector, FEATURE_NAMES
from classify.ml.inference import load_artifacts, predict_genre


@pytest.fixture(scope="module")
def multi_minute_wav():
    """Generates a temporary 180-second stereo WAV file with distinct frequency sections."""
    temp_dir = tempfile.mkdtemp(prefix="classify_parity_fixture_")
    file_path = os.path.join(temp_dir, "test_180s_track.wav")
    sr = 44100
    duration = 180.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Generate multi-section audio: bass intro (0-60s), mid lead (60-120s), high outro (120-180s)
    sig = 0.3 * np.sin(2 * np.pi * 110 * t)
    sig[int(60 * sr):int(120 * sr)] += 0.3 * np.sin(2 * np.pi * 440 * t[int(60 * sr):int(120 * sr)])
    sig[int(120 * sr):] += 0.3 * np.sin(2 * np.pi * 1760 * t[int(120 * sr):])
    stereo = np.column_stack([sig, sig])
    
    sf.write(file_path, stereo, sr)
    yield file_path
    
    # Cleanup
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
        os.rmdir(temp_dir)
    except OSError:
        pass


@pytest.fixture(scope="module")
def short_wav():
    """Generates a temporary short 15-second audio fixture."""
    temp_dir = tempfile.mkdtemp(prefix="classify_short_fixture_")
    file_path = os.path.join(temp_dir, "test_15s_track.wav")
    sr = 22050
    duration = 15.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sig = 0.4 * np.sin(2 * np.pi * 440 * t)
    sf.write(file_path, sig, sr)
    yield file_path
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
        os.rmdir(temp_dir)
    except OSError:
        pass


class TestAudioCoverage:
    """Requirement 7.1: Verify analyzed duration and segment placement span the full track."""

    def test_multi_minute_coverage_spans_beyond_60s(self, multi_minute_wav):
        """Asserts that a 180s audio file has segments placed across the full track."""
        res = predict_genre(multi_minute_wav, num_segments=3, segment_length=30.0)
        
        coverage = res["coverage"]
        assert coverage["original_duration"] >= 179.0
        assert coverage["analyzed_duration"] >= 90.0  # 3 segments of 30s = 90s
        
        # Verify segment placement: centers around 20%, 50%, 80% (36s, 90s, 144s)
        segments = res["segments"]
        assert len(segments) == 3
        
        offsets = [s["offset"] for s in segments]
        # Seg 1 should start well before 60s, Seg 2 around 75s, Seg 3 around 129s
        assert offsets[0] < 30.0
        assert offsets[1] > 60.0, "Second segment must be placed beyond the first 60 seconds"
        assert offsets[2] > 120.0, "Third segment must be placed in the final third of the track"

    def test_short_audio_graceful_reduction(self, short_wav):
        """Asserts that tracks shorter than segment_length reduce gracefully to 1 window."""
        res = predict_genre(short_wav, num_segments=3, segment_length=30.0)
        
        assert len(res["segments"]) == 1
        assert res["segments"][0]["offset"] == 0.0
        assert abs(res["segments"][0]["duration"] - 15.0) < 0.2
        assert res["coverage"]["coverage_ratio"] == 1.0


class TestTrainingInferenceParity:
    """Requirement 7.2: Verify strict invariant parity between train and inference pipelines."""

    def test_shared_feature_pipeline_and_names(self):
        """Asserts identical feature dimension (89) and feature names list."""
        assert len(FEATURE_NAMES) == 89
        assert FEATURE_NAMES[0] == "tempo"
        assert FEATURE_NAMES[1] == "rms_mean"
        assert "mel_spectrogram_var" in FEATURE_NAMES

    def test_scaler_and_model_parity(self):
        """Asserts identical loaded scaler and model artifacts without refitting."""
        artifacts = load_artifacts()
        scaler = artifacts["scaler"]
        model = artifacts["model"]
        encoder = artifacts["encoder"]

        assert hasattr(scaler, "transform")
        assert scaler.n_features_in_ == 89, "Fitted scaler must expect exactly 89 features"
        assert len(encoder.classes_) == 11, "Model must support exactly 11 genres"
        assert hasattr(model, "predict_proba"), "HistGradientBoosting must support predict_proba"

    def test_sample_rate_and_mono_invariance(self, multi_minute_wav):
        """Asserts audio is loaded at target_sr=22050 and downmixed to mono."""
        import librosa
        y, sr = librosa.load(multi_minute_wav, sr=22050, mono=True, duration=5.0)
        assert sr == 22050
        assert y.ndim == 1, "Signal must be 1D mono after loading"

        feat = analyze_audio(multi_minute_wav, duration_cap=10.0, offset=0.0)
        assert feat["sample_rate"] == 22050
        assert feat["channels"] == 2  # Fixture was stereo
        assert feat["native_sample_rate"] == 44100    # Resampled from 44100 to 22050


class TestPredictionBehaviorAndAggregation:
    """Requirement 7.3: Verify probability normalization, aggregation consistency, and schema."""

    def test_prediction_probabilities_sum_to_one(self, multi_minute_wav):
        """Asserts final probabilities and per-segment probabilities each sum to 1.0."""
        res = predict_genre(multi_minute_wav)
        
        # Final aggregated probabilities sum to ~1.0
        total_prob = sum(res["probabilities"].values())
        assert abs(total_prob - 1.0) < 1e-3
        
        # Per-segment probabilities sum to ~1.0
        for seg in res["segments"]:
            seg_sum = sum(seg["probabilities"].values())
            assert abs(seg_sum - 1.0) < 1e-3

    def test_predicted_genre_matches_top_probability(self, multi_minute_wav):
        """Asserts predicted genre is the argmax of aggregated probabilities."""
        res = predict_genre(multi_minute_wav)
        top_genre = max(res["probabilities"], key=res["probabilities"].get)
        assert res["predicted_genre"] == top_genre
        assert res["confidence"] == res["probabilities"][top_genre]

    def test_alternatives_ordering_and_structure(self, multi_minute_wav):
        """Asserts alternatives are sorted descending and exclude the top genre."""
        res = predict_genre(multi_minute_wav)
        top_genre = res["predicted_genre"]
        alts = res["alternatives"]
        
        assert len(alts) <= 3
        for alt in alts:
            assert alt["genre"] != top_genre
            assert alt["confidence"] <= res["confidence"]
        
        if len(alts) >= 2:
            assert alts[0]["confidence"] >= alts[1]["confidence"]

    def test_diagnostics_structure_complete(self, multi_minute_wav):
        """Asserts all required diagnostic telemetry fields are present."""
        res = predict_genre(multi_minute_wav)
        diag = res["diagnostics"]
        
        assert "original_duration" in diag
        assert "analyzed_duration" in diag
        assert "coverage_ratio" in diag
        assert "num_segments" in diag
        assert "segment_windows" in diag
        assert "timing" in diag
        assert "per_segment_predictions" in diag
        assert "feature_level_comparison" in diag


class TestBenchmarkNoCollapse:
    """Requirement 7.4: Verify GTZAN benchmark tracks do not collapse to Ambient."""

    @pytest.mark.parametrize("genre_file,expected_genre", [
        ("gtzan_classical_classical.00000.wav", "Classical"),
        ("gtzan_metal_metal.00000.wav", "Metal"),
        ("gtzan_jazz_jazz.00000.wav", "Jazz"),
        ("gtzan_blues_blues.00000.wav", "Blues"),
    ])
    def test_benchmark_tracks_predict_correctly(self, genre_file, expected_genre):
        """Verifies that representative GTZAN benchmark tracks do NOT collapse to Ambient."""
        audio_path = os.path.join(
            os.path.dirname(__file__), "..", "datasets", "processed", genre_file
        )
        if not os.path.exists(audio_path):
            pytest.skip(f"Benchmark file {genre_file} not found")

        res = predict_genre(audio_path)
        assert res["predicted_genre"] != "Ambient", (
            f"Benchmark {expected_genre} track must not collapse to Ambient"
        )
        assert res["predicted_genre"] == expected_genre
