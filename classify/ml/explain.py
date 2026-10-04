"""
explain.py
==========
Explainability engine for ML genre predictions.
Translates model feature importances and extracted acoustic measurements
into transparent, educational explanations.
Part of Phase 2 for CLASSIFY.
"""

import numpy as np


def generate_explanation(
    feature_vector,
    feature_names,
    predicted_genre,
    confidence,
    model=None,
    scaler=None,
    alternatives=None,
    is_low_confidence=False,
):
    """
    Generates a fact-based, transparent explanation of the model's prediction.

    Parameters:
        feature_vector (np.ndarray): 1D array of extracted acoustic features.
        feature_names (list): Names corresponding to feature_vector.
        predicted_genre (str): The top predicted genre label.
        confidence (float): Prediction confidence score (0.0 to 1.0).
        model: Trained scikit-learn estimator (inspects feature_importances_ if available).
        scaler: Fitted StandardScaler used during training.
        alternatives (list): Top alternative candidates with probabilities.
        is_low_confidence (bool): Whether the prediction meets low-confidence criteria.

    Returns:
        dict: {
            "summary": str,
            "top_factors": list of dict (feature, importance, value, description)
        }
    """
    feat_dict = dict(zip(feature_names, feature_vector))

    # Retrieve model feature importances if available
    importances = None
    if model is not None and hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif model is not None and hasattr(model, "coef_"):
        # Linear/logistic model coefficients
        importances = np.mean(np.abs(model.coef_), axis=0)

    top_factors = []
    if importances is not None and len(importances) == len(feature_names):
        # Sort features by importance weight
        top_indices = np.argsort(importances)[::-1][:5]
        for idx in top_indices:
            fname = feature_names[idx]
            val = float(feature_vector[idx])
            weight = float(importances[idx])

            desc = _describe_feature(fname, val)
            top_factors.append({
                "feature": fname,
                "importance_weight": round(weight, 4),
                "measured_value": round(val, 2),
                "description": desc,
            })
    elif scaler is not None and hasattr(scaler, "transform"):
        # Fallback for non-linear models (e.g., RBF SVM, MLP): identify top deviating acoustic drivers
        try:
            scaled_feat = np.asarray(scaler.transform(np.asarray(feature_vector).reshape(1, -1))[0])
            support = None
            if hasattr(scaler, "named_steps"):
                for step in scaler.named_steps.values():
                    if hasattr(step, "get_support"):
                        support = step.get_support(indices=True)
                        break

            top_indices = np.argsort(np.abs(scaled_feat))[::-1][:5]
            for idx in top_indices:
                orig_idx = int(support[idx]) if support is not None and idx < len(support) else idx
                if orig_idx < len(feature_names):
                    fname = feature_names[orig_idx]
                    val = float(feature_vector[orig_idx])
                    desc = _describe_feature(fname, val)
                    top_factors.append({
                        "feature": fname,
                        "importance_weight": round(float(abs(scaled_feat[idx])), 4),
                        "measured_value": round(val, 2),
                        "description": desc,
                    })
        except Exception:
            pass

    # Build human-readable educational explanation text
    tempo = feat_dict.get("tempo", 120.0)
    rms = feat_dict.get("rms_mean", 0.0)
    sc = feat_dict.get("spectral_centroid_mean", 0.0)
    zcr = feat_dict.get("zcr_mean", 0.0)

    factor_phrases = []
    if sc > 2200:
        factor_phrases.append(f"bright high-frequency spectral centroid ({sc:.0f} Hz)")
    elif sc < 1500:
        factor_phrases.append(f"warm, low-to-mid harmonic spectrum ({sc:.0f} Hz)")

    if tempo >= 125:
        factor_phrases.append(f"fast driving tempo ({tempo:.1f} BPM)")
    elif tempo <= 80:
        factor_phrases.append(f"measured, deliberate tempo ({tempo:.1f} BPM)")

    if rms >= 0.15:
        factor_phrases.append(f"high dynamic intensity (RMS: {rms:.3f})")
    elif rms < 0.08:
        factor_phrases.append(f"gentle acoustic dynamics (RMS: {rms:.3f})")

    if not factor_phrases:
        factor_phrases.append(f"balanced acoustic profile (tempo: {tempo:.1f} BPM, centroid: {sc:.0f} Hz)")

    joined_factors = ", ".join(factor_phrases)

    if is_low_confidence and alternatives:
        alt_names = [a.get("genre", "Unknown") for a in alternatives[:2]]
        summary = (
            f"The model predicts '{predicted_genre}' with moderate confidence ({confidence * 100:.1f}%), "
            f"driven by {joined_factors}. However, acoustic overlap was detected with {', '.join(alt_names)}, "
            f"indicating stylistic elements common to multiple genres."
        )
    else:
        summary = (
            f"Predicted as '{predicted_genre}' with {confidence * 100:.1f}% model confidence. "
            f"Primary acoustic indicators include {joined_factors}."
        )

    return {
        "summary": summary,
        "top_factors": top_factors,
    }


def _describe_feature(name, val):
    """Generates concise student-friendly descriptions of audio features."""
    if name == "tempo":
        return f"Rhythmic pace at {val:.1f} beats per minute"
    elif "spectral_centroid" in name:
        return f"Perceived sonic brightness / center of mass ({val:.1f} Hz)"
    elif "spectral_bandwidth" in name:
        return f"Frequency dispersion width ({val:.1f} Hz)"
    elif "rms_mean" in name:
        return f"Average loudness dynamic energy ({val:.4f})"
    elif "zcr_mean" in name:
        return f"Percussive attack rate ({val:.4f})"
    elif "onset_strength" in name:
        return f"Rhythmic note onset prominence ({val:.2f})"
    elif "mfcc" in name:
        return f"Timbral color profile coefficient ({val:.2f})"
    elif "chroma" in name:
        return f"Harmonic pitch-class energy ({val:.3f})"
    return f"Acoustic measurement of {name} ({val:.2f})"
