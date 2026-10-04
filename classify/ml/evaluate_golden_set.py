"""
evaluate_golden_set.py
======================
Evaluates Old (single-window 60s) vs New (full-song multi-segment aggregated)
inference on the external Golden Test Set (manifest: datasets/golden_manifest.csv).
Outputs accuracy, macro F1, confusion matrices, timing, and single-class collapse metrics.
Part of the Inference Parity Fix for CLASSIFY.
"""

import csv
import json
import os
import sys
import time
from typing import Any, Dict, List
import numpy as np

# Ensure root workspace is on path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from classify.audio.audio_processing import analyze_audio
from classify.ml.evaluate import evaluate_predictions
from classify.ml.feature_pipeline import extract_feature_vector
from classify.ml.inference import load_artifacts, predict_genre

GOLDEN_MANIFEST = os.path.join(BASE_DIR, "datasets", "golden_manifest.csv")
OUTPUT_REPORT = os.path.join(BASE_DIR, "model_artifacts", "golden_set_evaluation.json")


def run_old_single_window_inference(file_path: str, model, scaler, encoder) -> Dict[str, Any]:
    """Runs legacy single-window (60s cap from 0:00) inference."""
    t0 = time.perf_counter()
    feat_dict = analyze_audio(file_path, duration_cap=60.0, offset=0.0)
    vec, _ = extract_feature_vector(feat_dict)
    scaled_vec = scaler.transform(vec.reshape(1, -1))
    
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(scaled_vec)[0]
    else:
        df = model.decision_function(scaled_vec)[0]
        exp_df = np.exp(df - np.max(df))
        proba = exp_df / np.sum(exp_df)
    t1 = time.perf_counter()

    classes = encoder.classes_
    top_idx = int(np.argmax(proba))
    
    return {
        "predicted_genre": str(classes[top_idx]),
        "confidence": round(float(proba[top_idx]), 4),
        "probabilities": {str(g): round(float(p), 4) for g, p in zip(classes, proba)},
        "latency_seconds": round(t1 - t0, 4),
        "analyzed_duration": min(60.0, float(feat_dict.get("full_duration", 60.0))),
    }


def evaluate_golden_dataset(manifest_path: str = GOLDEN_MANIFEST) -> Dict[str, Any]:
    """Runs comparative evaluation across the Golden Test Set."""
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Golden manifest not found: {manifest_path}")

    artifacts = load_artifacts()
    model = artifacts["model"]
    scaler = artifacts["scaler"]
    encoder = artifacts["encoder"]
    classes = [str(c) for c in encoder.classes_]

    with open(manifest_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        tracks = list(reader)

    print(f"\n=======================================================")
    print(f"  CLASSIFY — GOLDEN TEST SET EVALUATION")
    print(f"  Total evaluation tracks: {len(tracks)}")
    print(f"=======================================================\n")

    y_true = []
    y_pred_old = []
    y_pred_new = []

    old_latencies = []
    new_latencies = []
    old_ambient_probs = []
    new_ambient_probs = []

    track_results = []

    for idx, row in enumerate(tracks):
        rel_path = row["track_path"]
        abs_path = os.path.join(BASE_DIR, rel_path) if not os.path.isabs(rel_path) else rel_path
        true_genre = row["genre"]
        track_id = row["track_id"]

        if not os.path.exists(abs_path):
            print(f"Warning: Audio file missing, skipping: {abs_path}")
            continue

        # 1. Run Old Inference
        res_old = run_old_single_window_inference(abs_path, model, scaler, encoder)
        old_pred = res_old["predicted_genre"]
        old_conf = res_old["confidence"]
        old_latencies.append(res_old["latency_seconds"])
        old_ambient_probs.append(res_old["probabilities"].get("Ambient", 0.0))

        # 2. Run New Inference
        t_start = time.perf_counter()
        res_new = predict_genre(abs_path)
        t_new = time.perf_counter() - t_start
        new_pred = res_new["predicted_genre"]
        new_conf = res_new["confidence"]
        new_latencies.append(t_new)
        new_ambient_probs.append(res_new["probabilities"].get("Ambient", 0.0))

        y_true.append(true_genre)
        y_pred_old.append(old_pred)
        y_pred_new.append(new_pred)

        cov = res_new.get("coverage", {})
        track_results.append({
            "track_id": track_id,
            "filename": os.path.basename(abs_path),
            "true_genre": true_genre,
            "dataset_name": row.get("dataset_name", ""),
            "original_duration": cov.get("original_duration"),
            "old": {
                "predicted": old_pred,
                "confidence": old_conf,
                "latency_s": res_old["latency_seconds"],
                "analyzed_duration_s": res_old["analyzed_duration"],
            },
            "new": {
                "predicted": new_pred,
                "confidence": new_conf,
                "latency_s": round(t_new, 4),
                "analyzed_duration_s": cov.get("analyzed_duration"),
                "coverage_ratio": cov.get("coverage_ratio"),
                "num_segments": res_new.get("diagnostics", {}).get("num_segments"),
            },
        })

        print(
            f"[{idx+1:02d}/{len(tracks)}] {os.path.basename(abs_path)[:30]:30s} | "
            f"True: {true_genre:10s} | "
            f"Old: {old_pred:10s} ({old_conf*100:5.1f}%) | "
            f"New: {new_pred:10s} ({new_conf*100:5.1f}%)"
        )

    # Compute classification metrics
    eval_old = evaluate_predictions(y_true, y_pred_old, class_names=classes)
    eval_new = evaluate_predictions(y_true, y_pred_new, class_names=classes)

    # Calculate collapse statistics
    old_ambient_collapse_count = sum(1 for p in y_pred_old if p == "Ambient")
    new_ambient_collapse_count = sum(1 for p in y_pred_new if p == "Ambient")
    n_total = len(y_true)

    report = {
        "golden_test_set_size": n_total,
        "classes": classes,
        "old_single_window_metrics": {
            "accuracy": eval_old["accuracy"],
            "macro_f1": eval_old["macro_f1"],
            "macro_precision": eval_old["macro_precision"],
            "macro_recall": eval_old["macro_recall"],
            "ambient_prediction_rate": round(old_ambient_collapse_count / n_total, 4),
            "mean_ambient_probability": round(float(np.mean(old_ambient_probs)), 4),
            "mean_latency_seconds": round(float(np.mean(old_latencies)), 4),
            "confusion_matrix": eval_old["confusion_matrix"],
            "per_class": eval_old["per_class"],
        },
        "new_multi_segment_metrics": {
            "accuracy": eval_new["accuracy"],
            "macro_f1": eval_new["macro_f1"],
            "macro_precision": eval_new["macro_precision"],
            "macro_recall": eval_new["macro_recall"],
            "ambient_prediction_rate": round(new_ambient_collapse_count / n_total, 4),
            "mean_ambient_probability": round(float(np.mean(new_ambient_probs)), 4),
            "mean_latency_seconds": round(float(np.mean(new_latencies)), 4),
            "confusion_matrix": eval_new["confusion_matrix"],
            "per_class": eval_new["per_class"],
        },
        "track_comparisons": track_results,
    }

    os.makedirs(os.path.dirname(OUTPUT_REPORT), exist_ok=True)
    with open(OUTPUT_REPORT, mode="w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n-------------------------------------------------------")
    print(f"Old Inference (60s single-window):")
    print(f"   Accuracy:        {eval_old['accuracy']*100:.2f}%")
    print(f"   Macro F1:        {eval_old['macro_f1']:.4f}")
    print(f"   Ambient Rate:    {old_ambient_collapse_count}/{n_total} ({old_ambient_collapse_count/n_total*100:.1f}%)")
    print(f"   Mean Latency:    {np.mean(old_latencies):.2f}s")
    print(f"\nNew Inference (Multi-segment Aggregation):")
    print(f"   Accuracy:        {eval_new['accuracy']*100:.2f}%")
    print(f"   Macro F1:        {eval_new['macro_f1']:.4f}")
    print(f"   Ambient Rate:    {new_ambient_collapse_count}/{n_total} ({new_ambient_collapse_count/n_total*100:.1f}%)")
    print(f"   Mean Latency:    {np.mean(new_latencies):.2f}s")
    print("-------------------------------------------------------")
    print(f"Report saved to: {OUTPUT_REPORT}\n")

    return report


if __name__ == "__main__":
    evaluate_golden_dataset()
