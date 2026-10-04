"""
error_analysis.py
=================
Detailed error inspection and confusion pair analysis for CLASSIFY.
Analyzes misclassified test tracks, captures probabilistic confidence margins,
and generates model_artifacts/error_analysis.json.
Part of Phase 4 for CLASSIFY.
"""

import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_ERROR_ANALYSIS_PATH = os.path.join(BASE_DIR, "model_artifacts", "error_analysis.json")


def analyze_misclassifications(
    y_true: List[str],
    y_pred: List[str],
    y_proba: np.ndarray,
    test_rows: List[Dict[str, Any]],
    class_names: List[str],
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Examines all misclassified tracks in the held-out test split.

    Parameters:
        y_true (list of str): Ground truth labels.
        y_pred (list of str): Model predicted labels.
        y_proba (np.ndarray): Predicted probability distribution over classes, shape (N, C).
        test_rows (list of dict): Manifest metadata rows corresponding to test tracks.
        class_names (list of str): Ordered class label names matching y_proba columns.
        output_path (str, optional): Target file path for error_analysis.json.

    Returns:
        dict: Complete error analysis summary.
    """
    out_file = output_path or DEFAULT_ERROR_ANALYSIS_PATH
    misclassified_records = []
    confusion_pairs = Counter()
    per_class_errors = defaultdict(int)

    for i in range(len(y_true)):
        true_lbl = str(y_true[i])
        pred_lbl = str(y_pred[i])

        if true_lbl != pred_lbl:
            confusion_pairs[(true_lbl, pred_lbl)] += 1
            per_class_errors[true_lbl] += 1

            # Top 3 predictions and probabilities
            probs_i = y_proba[i]
            sorted_indices = np.argsort(probs_i)[::-1][:3]
            top_3 = [
                {"genre": class_names[idx], "prob": round(float(probs_i[idx]), 4)}
                for idx in sorted_indices
            ]

            row_meta = test_rows[i] if i < len(test_rows) else {}
            misclassified_records.append({
                "track_id": row_meta.get("track_id", f"test_track_{i}"),
                "track_path": row_meta.get("track_path", ""),
                "true_genre": true_lbl,
                "predicted_genre": pred_lbl,
                "top_3_probs": top_3,
                "artist_id": row_meta.get("artist_id", ""),
                "source_dataset": row_meta.get("source", row_meta.get("dataset_name", "Unknown")),
            })

    # Format top confusion pairs for transparency
    top_confusion_pairs = [
        {"true_genre": pair[0], "predicted_genre": pair[1], "count": count}
        for pair, count in confusion_pairs.most_common(10)
    ]

    report = {
        "summary": {
            "total_test_samples": len(y_true),
            "total_misclassified": len(misclassified_records),
            "error_rate": round(len(misclassified_records) / max(1, len(y_true)), 4),
            "top_confusion_pairs": top_confusion_pairs,
            "errors_by_true_genre": dict(per_class_errors),
        },
        "misclassified_tracks": misclassified_records,
    }

    if out_file:
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return report
