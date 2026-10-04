"""
prototype_classifier.py
========================
PHASE 1 PROTOTYPE CLASSIFIER

NOTE FOR COLLEGE REVIEW / EVALUATORS:
-------------------------------------
This module implements a transparent, rule-based heuristic classification
mechanism exclusively for demonstrating the Phase 1 application workflow.

DO NOT confuse this with a trained Machine Learning model.

In Phase 2 & Phase 3, this module will be supplemented / replaced by a real
ML pipeline (`ml_classifier.py`) trained on the GTZAN dataset using algorithms
such as Random Forest, Support Vector Machines (SVM), and K-Nearest Neighbors (KNN).
"""

def classify_prototype(features):
    """
    Simulates genre classification using simple rule-based heuristics on basic audio features.
    
    Parameters:
        features (dict): Dictionary of audio features from audio_processing.py
        
    Returns:
        dict: Classification results and demo metadata.
    """
    rms = features.get("rms_energy", 0.0)
    centroid = features.get("spectral_centroid", 0.0)
    tempo = features.get("tempo", 120.0)
    zcr = features.get("zero_crossing_rate", 0.0)

    # Simple heuristic decision tree for demonstration purposes
    if rms >= 0.12 and centroid >= 2200:
        genre = "Rock"
        reasoning = (
            f"High acoustic energy (RMS: {rms}) and bright spectral centroid "
            f"({centroid} Hz) typical of energetic instruments and drums."
        )
    elif tempo >= 115 and rms >= 0.08:
        genre = "Pop"
        reasoning = (
            f"Upbeat tempo ({tempo} BPM) with moderate-to-high dynamic energy "
            f"(RMS: {rms}) matching modern pop arrangements."
        )
    elif rms < 0.06 and centroid < 1800:
        genre = "Classical"
        reasoning = (
            f"Subtle dynamic range (RMS: {rms}) and warm low-to-mid harmonic spectrum "
            f"({centroid} Hz) characteristic of orchestral/acoustic instruments."
        )
    elif 70 <= tempo <= 135 and 1300 <= centroid <= 2500:
        genre = "Jazz"
        reasoning = (
            f"Moderate tempo ({tempo} BPM) and balanced timbral warmth "
            f"({centroid} Hz) aligned with brass/woodwind acoustic signatures."
        )
    else:
        genre = "Acoustic / Ambient"
        reasoning = (
            f"Balanced energy profile (RMS: {rms}, Tempo: {tempo} BPM) "
            f"indicative of softer acoustic or ambient passages."
        )

    return {
        "predicted_genre": genre,
        "is_prototype": True,
        "classification_type": "Rule-Based Heuristic (Proof of Concept)",
        "explanation": reasoning,
        "future_ml_target": "Random Forest / SVM / KNN (Trained on GTZAN in Phase 2)"
    }
