"""
classification_service.py
=========================
Classification service wrapping ml/inference.py.
Returns real genre prediction, confidence score, alternatives, and explanations.
Part of Phase 2 for CLASSIFY.
"""

import os
from classify.ml.inference import predict_genre


def classify_audio_features(technical_features, file_path=None):
    """
    Computes trained ML genre prediction from technical features or audio file path.
    When file_path is provided, performs full-song multi-segment analysis.

    Parameters:
        technical_features (dict): Feature dictionary from analyze_audio.
        file_path (str, optional): Absolute path to audio file on disk.

    Returns:
        dict: ML prediction result with predicted_genre, confidence, alternatives,
              segments, diagnostics, etc.
    """
    if file_path and os.path.exists(file_path):
        return predict_genre(file_path)
    return predict_genre(technical_features)

