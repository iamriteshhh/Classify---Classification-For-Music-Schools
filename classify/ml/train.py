"""
train.py
========
Offline training script for genre classification models.
Enforces artist-aware stratified train/validation/test splitting, extracts Tier 1
features via feature_pipeline.py with multi-segment sampling, performs controlled
hyperparameter tuning on Train+Val splits, selects candidate by Validation Macro-F1,
refits on Train+Val, evaluates once on held-out Test split, runs error analysis,
and exports production artifacts to model_artifacts/.
Part of Phase 4 for CLASSIFY.
"""

import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, PowerTransformer, StandardScaler
from sklearn.svm import SVC

from classify.ml.dataset import load_manifest, MANIFEST_PATH
from classify.ml.error_analysis import analyze_misclassifications
from classify.ml.evaluate import evaluate_predictions
from classify.ml.feature_pipeline import extract_feature_vector, FEATURE_NAMES
from classify.ml.splitting import artist_aware_stratified_split

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "model_artifacts")
CACHE_DIR = os.path.join(BASE_DIR, "datasets")


def compute_manifest_checksum(manifest_rows: List[Dict[str, Any]]) -> str:
    """Computes SHA-256 hash across manifest paths and checksums for cache invalidation."""
    raw = "".join(f"{r.get('track_path')}:{r.get('checksum')}" for r in manifest_rows)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def extract_or_load_dataset_features(
    manifest_rows: List[Dict[str, Any]],
    cache_dir: Optional[str] = None,
    force_refresh: bool = False,
):
    """
    Extracts 89-dim feature vectors for all tracks in manifest_rows,
    caching to disk for fast iteration and reproducibility.
    """
    cdir = cache_dir or CACHE_DIR
    os.makedirs(cdir, exist_ok=True)
    m_hash = compute_manifest_checksum(manifest_rows)
    cache_file = os.path.join(cdir, f"features_cache_{m_hash}.joblib")

    if not force_refresh and os.path.exists(cache_file):
        print(f"Loading precomputed features from cache: {cache_file}")
        cached = joblib.load(cache_file)
        return cached["X"], cached["y_raw"], cached["feature_names"], m_hash

    print(f"Extracting features for {len(manifest_rows)} audio tracks with multi-segment sampling...")
    t0 = time.time()
    X_list = []
    y_list = []

    for idx, row in enumerate(manifest_rows):
        rel_track = row["track_path"]
        abs_track = os.path.join(BASE_DIR, rel_track) if not os.path.isabs(rel_track) else rel_track

        vec, feat_names = extract_feature_vector(abs_track)
        X_list.append(vec)
        y_list.append(row["genre"])

        if (idx + 1) % 100 == 0 or (idx + 1) == len(manifest_rows):
            print(f"  Processed {idx + 1}/{len(manifest_rows)} tracks ({time.time() - t0:.1f}s)")

    X = np.array(X_list, dtype=np.float32)
    y_raw = np.array(y_list)

    # Save to cache
    joblib.dump(
        {"X": X, "y_raw": y_raw, "feature_names": feat_names, "manifest_hash": m_hash},
        cache_file,
    )
    print(f"Features cached to {cache_file}")
    return X, y_raw, feat_names, m_hash


def tune_and_evaluate_candidates(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    X_val_scaled: np.ndarray,
    y_val: np.ndarray,
    class_names: List[str],
    label_encoder: LabelEncoder,
    random_state: int = 42,
) -> Tuple[Dict[str, Any], str, float, Any, Dict[str, Any]]:
    """
    Evaluates candidate model families with controlled hyperparameter grids
    strictly on train and validation splits (test split is untouched).
    Selects overall winner by Validation Macro-F1.
    """
    candidate_configs = {
        "Random Forest": {
            "model_cls": RandomForestClassifier,
            "grid": [
                {"n_estimators": 100, "max_depth": 10, "min_samples_leaf": 1, "random_state": random_state},
                {"n_estimators": 200, "max_depth": 15, "min_samples_leaf": 1, "random_state": random_state},
                {"n_estimators": 200, "max_depth": 20, "min_samples_leaf": 2, "random_state": random_state},
                {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1, "random_state": random_state},
            ],
        },
        "HistGradientBoosting": {
            "model_cls": HistGradientBoostingClassifier,
            "grid": [
                {"max_iter": 100, "learning_rate": 0.1, "max_depth": 8, "random_state": random_state},
                {"max_iter": 150, "learning_rate": 0.05, "max_depth": 10, "random_state": random_state},
                {"max_iter": 200, "learning_rate": 0.05, "max_depth": None, "random_state": random_state},
            ],
        },
        "Support Vector Machine (RBF)": {
            "model_cls": SVC,
            "grid": [
                {"kernel": "rbf", "C": 4.5, "gamma": 0.007, "probability": True, "random_state": random_state},
                {"kernel": "rbf", "C": 5.0, "gamma": 0.007, "probability": True, "random_state": random_state},
                {"kernel": "rbf", "C": 5.5, "gamma": 0.007, "probability": True, "random_state": random_state},
            ],
        },
        "K-Nearest Neighbors": {
            "model_cls": KNeighborsClassifier,
            "grid": [
                {"n_neighbors": 3, "weights": "uniform"},
                {"n_neighbors": 5, "weights": "distance"},
                {"n_neighbors": 7, "weights": "distance"},
                {"n_neighbors": 11, "weights": "distance"},
            ],
        },
        "Logistic Regression": {
            "model_cls": LogisticRegression,
            "grid": [
                {"C": 0.5, "class_weight": "balanced", "max_iter": 1000, "random_state": random_state},
                {"C": 1.0, "class_weight": "balanced", "max_iter": 1000, "random_state": random_state},
                {"C": 2.0, "class_weight": "balanced", "max_iter": 1000, "random_state": random_state},
            ],
        },
        "Multi-Layer Perceptron (MLP)": {
            "model_cls": MLPClassifier,
            "grid": [
                {"hidden_layer_sizes": (128, 64), "alpha": 0.001, "max_iter": 500, "random_state": random_state},
                {"hidden_layer_sizes": (256, 128), "alpha": 0.01, "max_iter": 500, "random_state": random_state},
                {"hidden_layer_sizes": (128, 128), "alpha": 0.01, "max_iter": 500, "random_state": random_state},
            ],
        },
    }

    model_comparison = {}
    best_candidate_name = None
    best_candidate_macro_f1 = -1.0
    best_candidate_params = {}
    best_candidate_instance = None

    print("\n--- Tuning & Evaluating Candidate Models on Validation Split (Macro F1) ---")

    for family_name, config in candidate_configs.items():
        cls = config["model_cls"]
        best_family_f1 = -1.0
        best_family_eval = None
        best_family_params = None
        best_family_inst = None

        for params in config["grid"]:
            model = cls(**params)
            model.fit(X_train_scaled, y_train)
            y_val_pred = model.predict(X_val_scaled)
            val_eval = evaluate_predictions(
                label_encoder.inverse_transform(y_val),
                label_encoder.inverse_transform(y_val_pred),
                class_names=class_names,
            )

            if val_eval["macro_f1"] > best_family_f1:
                best_family_f1 = val_eval["macro_f1"]
                best_family_eval = val_eval
                best_family_params = params
                best_family_inst = model

        model_comparison[family_name] = {
            "val_macro_f1": best_family_eval["macro_f1"],
            "val_accuracy": best_family_eval["accuracy"],
            "val_precision": best_family_eval["macro_precision"],
            "val_recall": best_family_eval["macro_recall"],
            "best_hyperparameters": {k: (str(v) if not isinstance(v, (int, float, bool)) else v) for k, v in best_family_params.items()},
        }
        print(f"  {family_name:28s} | Best Val Macro F1: {best_family_eval['macro_f1']:.4f} | Accuracy: {best_family_eval['accuracy']:.4f} | Params: {best_family_params}")

        if best_family_f1 > best_candidate_macro_f1:
            best_candidate_macro_f1 = best_family_f1
            best_candidate_name = family_name
            best_candidate_params = best_family_params
            best_candidate_instance = best_family_inst

    print(f"\nWinning Model Family: {best_candidate_name} (Val Macro F1: {best_candidate_macro_f1:.4f})")
    return model_comparison, best_candidate_name, best_candidate_macro_f1, best_candidate_instance, best_candidate_params


def train_and_evaluate(manifest_path=None, artifacts_dir=None, random_state=42, force_refresh_features=False):
    """
    Complete offline training, hyperparameter tuning, evaluation, error analysis,
    and artifact export pipeline.
    """
    m_path = manifest_path or MANIFEST_PATH
    out_dir = artifacts_dir or ARTIFACTS_DIR
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading manifest from: {m_path}")
    manifest_rows = load_manifest(m_path)
    if not manifest_rows:
        raise ValueError("Dataset manifest is empty.")

    # 1. Feature Extraction (with caching)
    X, y_raw, feat_names, m_hash = extract_or_load_dataset_features(
        manifest_rows,
        force_refresh=force_refresh_features,
    )

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    class_names = list(label_encoder.classes_)

    # 2. Artist-Aware Stratified Splitting
    train_rows, val_rows, test_rows, split_report = artist_aware_stratified_split(
        manifest_rows,
        val_size=0.15,
        test_size=0.15,
        min_test_support_per_class=1,
        random_state=random_state,
    )

    train_idx = split_report["train_indices"]
    val_idx = split_report["val_indices"]
    test_idx = split_report["test_indices"]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    train_val_idx = train_idx + val_idx
    X_train_val = X[train_val_idx]
    y_train_val = y[train_val_idx]

    train_artists = split_report["total_artists"]["train"]
    val_artists = split_report["total_artists"]["val"]
    test_artists = split_report["total_artists"]["test"]

    print(
        f"\nArtist-aware stratified split complete:\n"
        f"  Train: {len(X_train)} tracks ({train_artists} artists)\n"
        f"  Val:   {len(X_val)} tracks ({val_artists} artists)\n"
        f"  Test:  {len(X_test)} tracks ({test_artists} artists)\n"
        f"  Total: {len(manifest_rows)} tracks across {len(class_names)} classes"
    )

    # 3. Fit Scaler strictly on Train data
    scaler = Pipeline([
        ("power_transform", PowerTransformer(method="yeo-johnson")),
        ("select_k_best", SelectKBest(f_classif, k=86)),
    ])
    X_train_scaled = scaler.fit_transform(X_train, y_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 4 & 5. Hyperparameter Tuning & Candidate Comparison on Validation Split
    model_comp, winner_name, val_f1, _, winner_params = tune_and_evaluate_candidates(
        X_train_scaled=X_train_scaled,
        y_train=y_train,
        X_val_scaled=X_val_scaled,
        y_val=y_val,
        class_names=class_names,
        label_encoder=label_encoder,
        random_state=random_state,
    )

    # 6. Refit Winning Model on Train + Val, Evaluate on Untouched Held-Out Test Split
    print(f"\nRefitting winning model ({winner_name}) on Train + Val with best params...")
    
    # Re-instantiate model with best params
    model_classes = {
        "Random Forest": RandomForestClassifier,
        "HistGradientBoosting": HistGradientBoostingClassifier,
        "Support Vector Machine (RBF)": SVC,
        "K-Nearest Neighbors": KNeighborsClassifier,
        "Logistic Regression": LogisticRegression,
        "Multi-Layer Perceptron (MLP)": MLPClassifier,
    }
    winning_model = model_classes[winner_name](**winner_params)

    scaler_full = Pipeline([
        ("power_transform", PowerTransformer(method="yeo-johnson")),
        ("select_k_best", SelectKBest(f_classif, k=86)),
    ])
    X_train_val_scaled = scaler_full.fit_transform(X_train_val, y_train_val)
    X_test_final_scaled = scaler_full.transform(X_test)

    winning_model.fit(X_train_val_scaled, y_train_val)
    y_test_pred = winning_model.predict(X_test_final_scaled)

    # Predict test probabilities for error analysis and calibration
    if hasattr(winning_model, "predict_proba"):
        y_test_proba = winning_model.predict_proba(X_test_final_scaled)
    else:
        df = winning_model.decision_function(X_test_final_scaled)
        exp_df = np.exp(df - np.max(df, axis=1, keepdims=True))
        y_test_proba = exp_df / np.sum(exp_df, axis=1, keepdims=True)

    y_test_true_labels = list(label_encoder.inverse_transform(y_test))
    y_test_pred_labels = list(label_encoder.inverse_transform(y_test_pred))

    test_eval = evaluate_predictions(
        y_test_true_labels,
        y_test_pred_labels,
        class_names=class_names,
    )

    print("\n========================================================")
    print("      FINAL TEST SET EVALUATION (UNTOUCHED TEST SPLIT)  ")
    print("========================================================")
    print(f"  Winning Model:      {winner_name}")
    print(f"  Test Macro F1:      {test_eval['macro_f1']:.4f}")
    print(f"  Test Accuracy:      {test_eval['accuracy']:.4f}")
    print(f"  Test Weighted F1:   {test_eval['weighted_f1']:.4f}")
    print(f"  Macro Precision:    {test_eval['macro_precision']:.4f}")
    print(f"  Macro Recall:       {test_eval['macro_recall']:.4f}")

    if test_eval.get("classes_missing_from_eval"):
        print(f"\n[CRITICAL WARNING] Classes missing from final evaluation: {test_eval['classes_missing_from_eval']}")

    # 7. Error Analysis
    error_analysis_path = os.path.join(out_dir, "error_analysis.json")
    error_report = analyze_misclassifications(
        y_true=y_test_true_labels,
        y_pred=y_test_pred_labels,
        y_proba=y_test_proba,
        test_rows=test_rows,
        class_names=class_names,
        output_path=error_analysis_path,
    )
    print(f"\nError analysis recorded {error_report['summary']['total_misclassified']} misclassified tracks:")
    for pair in error_report["summary"]["top_confusion_pairs"][:5]:
        print(f"  {pair['true_genre']} -> {pair['predicted_genre']}: {pair['count']} tracks")

    # 8. Build Evaluation Report
    report = {
        "winning_model": winner_name,
        "hyperparameters": {k: (str(v) if not isinstance(v, (int, float, bool)) else v) for k, v in winner_params.items()},
        "candidate_comparison": model_comp,
        "test_metrics": test_eval,
        "class_labels": class_names,
        "feature_names": feat_names,
        "num_features": len(feat_names),
        "split_summary": split_report,
        "error_summary": error_report["summary"],
    }

    # 9. Save Artifacts
    model_path = os.path.join(out_dir, "genre_classifier.joblib")
    scaler_path = os.path.join(out_dir, "scaler.joblib")
    encoder_path = os.path.join(out_dir, "label_encoder.joblib")
    report_path = os.path.join(out_dir, "evaluation_report.json")

    joblib.dump(winning_model, model_path)
    joblib.dump(scaler_full, scaler_path)
    joblib.dump(label_encoder, encoder_path)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nModel artifacts saved successfully:")
    print(f"  Model:   {model_path}")
    print(f"  Scaler:  {scaler_path}")
    print(f"  Encoder: {encoder_path}")
    print(f"  Report:  {report_path}")
    print(f"  Errors:  {error_analysis_path}")

    return report


if __name__ == "__main__":
    train_and_evaluate()
