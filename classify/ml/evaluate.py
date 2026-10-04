"""
evaluate.py
===========
Evaluation metrics calculation for genre classification models.
Computes accuracy, macro F1, per-class precision/recall/F1, confusion matrices,
and explicit detection of classes missing from evaluation splits.
Part of Phase 1 for CLASSIFY.
"""

from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)


def evaluate_predictions(
    y_true: Any,
    y_pred: Any,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Computes standard classification evaluation metrics with macro F1 as headline metric.
    Guarantees that all known classes in class_names appear in the report and confusion matrix,
    even if they have zero support in this split.

    Parameters:
        y_true (array-like): Ground truth class labels (string or integer).
        y_pred (array-like): Predicted class labels (string or integer).
        class_names (list of str, optional): Full list of all known classes.

    Returns:
        dict: Evaluation report dictionary with macro F1, accuracy, per-class stats,
              confusion matrix, and classes_missing_from_eval.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    if class_names is not None and len(class_names) > 0:
        active_labels = [str(c) for c in class_names]
        target_names = active_labels
    else:
        # Fallback to present labels
        present_set = sorted(list(set(y_true) | set(y_pred)))
        active_labels = [str(lbl) for lbl in present_set]
        target_names = active_labels

    # Confusion matrix over full known class list
    cm = confusion_matrix(y_true, y_pred, labels=active_labels).tolist()

    # Per-class classification report
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=active_labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    per_class = {}
    classes_missing_from_eval = []

    for name in active_labels:
        if name in report_dict:
            supp = int(report_dict[name]["support"])
            per_class[name] = {
                "precision": round(float(report_dict[name]["precision"]), 4),
                "recall": round(float(report_dict[name]["recall"]), 4),
                "f1_score": round(float(report_dict[name]["f1-score"]), 4),
                "support": supp,
            }
            if supp == 0:
                classes_missing_from_eval.append(name)
        else:
            per_class[name] = {
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "support": 0,
            }
            classes_missing_from_eval.append(name)

    return {
        "macro_f1": round(macro_f1, 4),
        "accuracy": round(acc, 4),
        "weighted_f1": round(weighted_f1, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "classes": active_labels,
        "confusion_matrix": cm,
        "per_class": per_class,
        "classes_missing_from_eval": classes_missing_from_eval,
    }
