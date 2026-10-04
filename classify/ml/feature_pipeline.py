"""
feature_pipeline.py
===================
Shared feature-vector construction used by BOTH training and inference.
Guarantees identical feature dimensionality, ordering, and calculations.
Supports multi-window segmented feature aggregation for enhanced accuracy.
Part of Phase 3 for CLASSIFY.
"""

import numpy as np
from classify.audio.audio_processing import analyze_audio

# Standardized list of feature names (89 features total)
FEATURE_NAMES = (
    ["tempo", "rms_mean", "rms_var", "zcr_mean", "zcr_var"]
    + [
        "spectral_centroid_mean",
        "spectral_centroid_var",
        "spectral_bandwidth_mean",
        "spectral_bandwidth_var",
        "spectral_rolloff_mean",
        "spectral_rolloff_var",
        "spectral_contrast_mean",
        "spectral_contrast_var",
    ]
    + [f"spectral_contrast_band_{i}" for i in range(7)]
    + [f"mfcc_mean_{i}" for i in range(20)]
    + [f"mfcc_var_{i}" for i in range(20)]
    + [f"chroma_mean_{i}" for i in range(12)]
    + [f"chroma_var_{i}" for i in range(12)]
    + [
        "onset_strength_mean",
        "onset_strength_var",
        "onset_rate",
        "mel_spectrogram_mean",
        "mel_spectrogram_var",
    ]
)


def extract_feature_vector(audio_source, use_segmentation=True):
    """
    Extracts a standardized 1D 89-feature vector from either an audio file path or
    an existing technical_features dictionary.

    When given a file path, applies multi-window audio segmentation (Phase 3)
    to aggregate representative acoustic windows across the song.

    Parameters:
        audio_source (str or dict): Either file path to audio, or pre-extracted features dict.
        use_segmentation (bool): Whether to apply multi-window segmentation when audio_source is a path.

    Returns:
        tuple: (np.ndarray of shape (89,), list of feature names)
    """
    if isinstance(audio_source, str):
        if use_segmentation:
            from classify.audio.segmentation import extract_segmented_features
            vec, names, _ = extract_segmented_features(audio_source)
            return vec, names
        features = analyze_audio(audio_source)
    elif isinstance(audio_source, dict):
        features = audio_source
    else:
        raise ValueError(f"Expected file path string or features dict, got {type(audio_source)}")

    vector = []

    # 1. Dynamics & Tempo (5)
    vector.append(float(features.get("tempo", 120.0)))
    vector.append(float(features.get("rms_mean", features.get("rms_energy", 0.0))))
    vector.append(float(features.get("rms_var", 0.0)))
    vector.append(float(features.get("zcr_mean", features.get("zero_crossing_rate", 0.0))))
    vector.append(float(features.get("zcr_var", 0.0)))

    # 2. Spectral Descriptors (8)
    vector.append(float(features.get("spectral_centroid_mean", features.get("spectral_centroid", 0.0))))
    vector.append(float(features.get("spectral_centroid_var", 0.0)))
    vector.append(float(features.get("spectral_bandwidth_mean", 0.0)))
    vector.append(float(features.get("spectral_bandwidth_var", 0.0)))
    vector.append(float(features.get("spectral_rolloff_mean", 0.0)))
    vector.append(float(features.get("spectral_rolloff_var", 0.0)))
    # Task 0.4: Revived natural computation (protected by LUFS normalization and 8kHz filter)
    vector.append(float(features.get("spectral_contrast_mean", 0.0)))
    vector.append(float(features.get("spectral_contrast_var", 0.0)))

    # 3. Spectral Contrast sub-bands (7)
    # Bands 0-6 cover all sub-bands; band 6 revived under 8kHz harmonization
    contrast_bands = features.get("spectral_contrast_bands", [0.0] * 7)
    for i in range(7):
        vector.append(float(contrast_bands[i]) if i < len(contrast_bands) else 0.0)

    # 4. MFCC Means (20)
    mfcc_means = features.get("mfcc_means", [features.get("mfcc_mean", 0.0)] * 20)
    for i in range(20):
        vector.append(float(mfcc_means[i]) if i < len(mfcc_means) else 0.0)

    # 5. MFCC Variances (20)
    mfcc_vars = features.get("mfcc_vars", [0.0] * 20)
    for i in range(20):
        vector.append(float(mfcc_vars[i]) if i < len(mfcc_vars) else 0.0)

    # 6. Chroma Means (12)
    chroma_means = features.get("chroma_means", [0.0] * 12)
    for i in range(12):
        vector.append(float(chroma_means[i]) if i < len(chroma_means) else 0.0)

    # 7. Chroma Variances (12)
    chroma_vars = features.get("chroma_vars", [0.0] * 12)
    for i in range(12):
        vector.append(float(chroma_vars[i]) if i < len(chroma_vars) else 0.0)

    # 8. Rhythm & Mel Spectrogram (5)
    vector.append(float(features.get("onset_strength_mean", 0.0)))
    vector.append(float(features.get("onset_strength_var", 0.0)))
    vector.append(float(features.get("onset_rate", 0.0)))
    vector.append(float(features.get("mel_spectrogram_mean", 0.0)))
    # Task 0.4: Revived natural computation (protected by LUFS normalization)
    vector.append(float(features.get("mel_spectrogram_var", 0.0)))

    vector_np = np.array(vector, dtype=np.float32)
    return vector_np, FEATURE_NAMES
