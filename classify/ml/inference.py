"""
inference.py
============
Online genre prediction engine.
Loads trained ML artifacts (Random Forest / scikit-learn) and computes
real probabilistic genre predictions, confidence scores, alternatives,
and feature-importance explanations.
Part of Phase 2 for CLASSIFY.
"""

import os
import joblib
import numpy as np
import time
import librosa
from classify.audio.audio_processing import analyze_audio
from classify.ml.explain import generate_explanation
from classify.ml.feature_pipeline import extract_feature_vector, FEATURE_NAMES
from classify.ml.subgenre_classifier import predict_subgenre

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "model_artifacts")

_MODEL_CACHE = {
    "model": None,
    "scaler": None,
    "encoder": None,
    "report": None,
}


def load_artifacts(artifacts_dir=None):
    """Loads serialized model, scaler, and label encoder once into memory."""
    target_dir = artifacts_dir or ARTIFACTS_DIR
    model_path = os.path.join(target_dir, "genre_classifier.joblib")
    scaler_path = os.path.join(target_dir, "scaler.joblib")
    encoder_path = os.path.join(target_dir, "label_encoder.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model artifact not found at {model_path}. "
            "Run 'python -m classify.ml.train' to generate model artifacts."
        )

    _MODEL_CACHE["model"] = joblib.load(model_path)
    _MODEL_CACHE["scaler"] = joblib.load(scaler_path)
    _MODEL_CACHE["encoder"] = joblib.load(encoder_path)

    return _MODEL_CACHE


def predict_genre(
    audio_source,
    artifacts_dir=None,
    num_segments=3,
    segment_length=30.0,
    aggregation="prediction_average",
):
    """
    Predicts music genre with full-track multi-segment coverage, probabilistic
    aggregation, confidence scoring, transparent explanations, and diagnostic telemetry.

    Parameters:
        audio_source (str or dict): Audio file path (.wav, .mp3) or precomputed features dict.
        artifacts_dir (str, optional): Custom path to model artifacts.
        num_segments (int): Desired number of representative audio segments (default: 3).
        segment_length (float): Window duration in seconds per segment (default: 30.0s).
        aggregation (str): Aggregation method ('prediction_average' or 'feature_average').

    Returns:
        dict: Standard prediction schema with additive diagnostics and segment data:
            - predicted_genre: str
            - confidence: float (0.0 to 1.0)
            - confidence_percent: float
            - is_low_confidence: bool
            - probabilities: dict of genre -> float
            - alternatives: list of {"genre": str, "confidence": float, "percent": float}
            - explanation: str
            - explanation_factors: list of dict
            - model_name: str
            - classification_type: str
            - is_prototype: bool
            - segments: list of per-segment summary dicts
            - diagnostics: detailed telemetry (coverage, windows, per-segment probas, timing)
            - coverage: dict of original_duration, analyzed_duration, coverage_ratio
    """
    if _MODEL_CACHE["model"] is None:
        load_artifacts(artifacts_dir)

    model = _MODEL_CACHE["model"]
    scaler = _MODEL_CACHE["scaler"]
    encoder = _MODEL_CACHE["encoder"]
    classes = encoder.classes_

    t_start = time.perf_counter()

    # Branch A: Full-song multi-segment inference from audio file path
    if isinstance(audio_source, str):
        if not os.path.exists(audio_source):
            raise FileNotFoundError(f"Audio file not found for inference: {audio_source}")

        # 1. Determine actual audio duration
        try:
            full_duration = float(librosa.get_duration(path=audio_source))
        except Exception:
            full_duration = 30.0

        # 2. Compute representative segment windows (e.g. 20%, 50%, 80%)
        from classify.audio.segmentation import compute_segment_windows

        windows = compute_segment_windows(
            full_duration,
            num_segments=num_segments,
            segment_length=segment_length,
        )

        seg_probas = []
        seg_vecs = []
        seg_timings = []
        segment_summaries = []

        # 3. Extract features, scale, and predict for each segment independently
        for idx, (offset, seg_dur) in enumerate(windows):
            t0 = time.perf_counter()
            seg_dict = analyze_audio(audio_source, duration_cap=seg_dur, offset=offset)
            t1 = time.perf_counter()

            svec, feat_names = extract_feature_vector(seg_dict)
            t2 = time.perf_counter()

            scaled_vec = scaler.transform(svec.reshape(1, -1))
            t3 = time.perf_counter()

            if hasattr(model, "predict_proba"):
                sproba = model.predict_proba(scaled_vec)[0]
            else:
                df = model.decision_function(scaled_vec)[0]
                exp_df = np.exp(df - np.max(df))
                sproba = exp_df / np.sum(exp_df)
            t4 = time.perf_counter()

            seg_probas.append(sproba)
            seg_vecs.append(svec)

            seg_rms = float(seg_dict.get("rms_mean", seg_dict.get("rms_energy", 0.0)))
            timing_info = {
                "load_and_extract": round(t1 - t0, 4),
                "vector_assembly": round(t2 - t1, 4),
                "scaling": round(t3 - t2, 4),
                "model_inference": round(t4 - t3, 4),
                "total": round(t4 - t0, 4),
                "rms_energy": round(seg_rms, 6),
            }
            seg_timings.append(timing_info)

            top_seg_idx = int(np.argmax(sproba))
            top_seg_genre = str(classes[top_seg_idx])
            top_seg_conf = round(float(sproba[top_seg_idx]), 4)

            seg_prob_dict = {str(g): round(float(p), 4) for g, p in zip(classes, sproba)}
            segment_summaries.append({
                "segment_index": idx,
                "offset": offset,
                "duration": seg_dur,
                "top_genre": top_seg_genre,
                "confidence": top_seg_conf,
                "probabilities": seg_prob_dict,
                "timing": timing_info,
            })

        stacked_probas = np.array(seg_probas, dtype=np.float64)
        stacked_vecs = np.array(seg_vecs, dtype=np.float32)

        # 4. Aggregation: Task 0.3 RMS-weighted pooling
        rms_energies = np.array([
            float(t["rms_energy"]) for t in seg_timings
        ], dtype=np.float64)
        total_rms = float(np.sum(rms_energies))

        if total_rms > 1e-9:
            weights = rms_energies / total_rms
            avg_proba = np.sum(weights[:, np.newaxis] * stacked_probas, axis=0)
            rep_vector = np.sum(weights[:, np.newaxis] * stacked_vecs, axis=0).astype(np.float32)
        else:
            avg_proba = np.mean(stacked_probas, axis=0)
            rep_vector = np.mean(stacked_vecs, axis=0).astype(np.float32)

        # Re-normalize to guard against float precision drift
        prob_sum = np.sum(avg_proba)
        if prob_sum > 0:
            avg_proba = avg_proba / prob_sum

        # Feature-level aggregation comparison (Approach A diagnostic)
        feat_scaled = scaler.transform(rep_vector.reshape(1, -1))
        if hasattr(model, "predict_proba"):
            feat_proba = model.predict_proba(feat_scaled)[0]
        else:
            df = model.decision_function(feat_scaled)[0]
            exp_df = np.exp(df - np.max(df))
            feat_proba = exp_df / np.sum(exp_df)

        if aggregation == "feature_average":
            final_proba = feat_proba
        else:
            final_proba = avg_proba

        feat_names = FEATURE_NAMES
        total_analyzed = round(float(sum(w[1] for w in windows)), 2)
        coverage_ratio = round(min(1.0, total_analyzed / max(1e-4, full_duration)), 4)

        diagnostics = {
            "original_duration": round(full_duration, 2),
            "analyzed_duration": total_analyzed,
            "coverage_ratio": coverage_ratio,
            "num_segments": len(windows),
            "segment_windows": [{"offset": w[0], "length": w[1]} for w in windows],
            "sample_rate": 22050,
            "channels": 1,
            "feature_dim": 89,
            "model_name": type(model).__name__,
            "scaler_name": type(scaler).__name__,
            "aggregation_method": aggregation,
            "per_segment_predictions": segment_summaries,
            "feature_level_comparison": {
                "predicted_genre": str(classes[int(np.argmax(feat_proba))]),
                "confidence": round(float(np.max(feat_proba)), 4),
                "top_probabilities": sorted(
                    {str(g): round(float(p), 4) for g, p in zip(classes, feat_proba)}.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:3],
            },
            "timing": {
                "total_inference_seconds": round(time.perf_counter() - t_start, 4),
                "load_and_extract_seconds": round(sum(t["load_and_extract"] for t in seg_timings), 4),
                "scaling_seconds": round(sum(t["scaling"] for t in seg_timings), 4),
                "model_inference_seconds": round(sum(t["model_inference"] for t in seg_timings), 4),
            },
        }

    # Branch B: Fallback for precomputed features dict (single window / tests)
    elif isinstance(audio_source, dict):
        rep_vector, feat_names = extract_feature_vector(audio_source)
        scaled_vector = scaler.transform(rep_vector.reshape(1, -1))

        if hasattr(model, "predict_proba"):
            final_proba = model.predict_proba(scaled_vector)[0]
        else:
            df = model.decision_function(scaled_vector)[0]
            exp_df = np.exp(df - np.max(df))
            final_proba = exp_df / np.sum(exp_df)

        dur_val = float(audio_source.get("duration", audio_source.get("full_duration", 60.0)))
        segment_summaries = []
        diagnostics = {
            "input_type": "precomputed_features_dict",
            "original_duration": round(dur_val, 2),
            "analyzed_duration": round(dur_val, 2),
            "coverage_ratio": 1.0,
            "num_segments": 1,
            "aggregation_method": "single_window_direct",
            "model_name": type(model).__name__,
            "scaler_name": type(scaler).__name__,
            "timing": {
                "total_inference_seconds": round(time.perf_counter() - t_start, 4),
            },
        }
    else:
        raise ValueError(f"Expected file path string or features dict, got {type(audio_source)}")

    # Construct final class probability distribution
    prob_dict = {str(genre): round(float(p), 4) for genre, p in zip(classes, final_proba)}

    # Sort genres by probability descending
    sorted_genres = sorted(prob_dict.items(), key=lambda item: item[1], reverse=True)
    top_genre, top_prob = sorted_genres[0]

    # Calculate alternatives (top 3 remaining candidates)
    alternatives = [
        {"genre": genre, "confidence": round(p, 4), "percent": round(p * 100, 1)}
        for genre, p in sorted_genres[1:4]
    ]

    # Task 1.3: Hierarchical Level-2 Subgenre Prediction
    subgenre_info = predict_subgenre(
        parent_genre=top_genre,
        feature_vector=rep_vector,
        feature_names=feat_names,
    )

    # Task 1.4: Multi-label Genre Classification (secondary genres >= 0.25 or significant secondary pull)
    secondary_genres = [
        {"genre": genre, "confidence": round(p, 4), "percent": round(p * 100, 1)}
        for genre, p in sorted_genres[1:]
        if p >= 0.25 or (p >= 0.18 and top_prob < 0.50)
    ]
    genre_tags = [
        {"genre": top_genre, "confidence": round(float(top_prob), 4), "percent": round(float(top_prob * 100), 1), "is_primary": True}
    ] + [
        {"genre": s["genre"], "confidence": s["confidence"], "percent": s["percent"], "is_primary": False}
        for s in secondary_genres
    ]

    # Check for low confidence or close ambiguity
    second_prob = alternatives[0]["confidence"] if alternatives else 0.0
    is_low_confidence = bool(top_prob < 0.40 or (top_prob - second_prob) < 0.10)

    # Generate transparent feature-importance explanation
    explanation_res = generate_explanation(
        feature_vector=rep_vector,
        feature_names=feat_names,
        predicted_genre=top_genre,
        confidence=top_prob,
        model=model,
        scaler=scaler,
        alternatives=alternatives,
        is_low_confidence=is_low_confidence,
    )

    return {
        "predicted_genre": top_genre,
        "confidence": round(float(top_prob), 4),
        "confidence_percent": round(float(top_prob * 100), 1),
        "is_low_confidence": is_low_confidence,
        "probabilities": prob_dict,
        "alternatives": alternatives,
        "subgenre": subgenre_info,
        "subgenre_name": subgenre_info["name"] if subgenre_info else None,
        "subgenre_slug": subgenre_info["slug"] if subgenre_info else None,
        "secondary_genres": secondary_genres,
        "genre_tags": genre_tags,
        "explanation": explanation_res["summary"],
        "explanation_factors": explanation_res["top_factors"],
        "model_name": type(model).__name__,
        "classification_type": "Trained ML Model (Supervised Classical)",
        "is_prototype": False,
        "segments": segment_summaries,
        "diagnostics": diagnostics,
        "coverage": {
            "original_duration": diagnostics.get("original_duration"),
            "analyzed_duration": diagnostics.get("analyzed_duration"),
            "coverage_ratio": diagnostics.get("coverage_ratio"),
        },
    }
