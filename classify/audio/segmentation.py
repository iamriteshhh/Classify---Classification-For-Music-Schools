"""
segmentation.py
===============
Multi-window audio segmentation and feature aggregation for CLASSIFY.
Samples multiple windows per track (e.g., 20%, 50%, 80% positions) and aggregates
them into a consistent 89-dimensional feature vector, guaranteeing train/inference parity.
Part of Phase 3 for CLASSIFY.
"""

import os
from typing import Any, Dict, List, Tuple, Union
import librosa
import numpy as np

from classify.audio.audio_processing import analyze_audio, extract_features_from_signal
from classify.ml.feature_pipeline import extract_feature_vector, FEATURE_NAMES


def compute_segment_windows(
    duration: float,
    num_segments: int = 3,
    segment_length: float = 30.0,
) -> List[Tuple[float, float]]:
    """
    Computes (offset, length) windows across the track duration.

    - For short tracks (duration <= segment_length): returns a single window covering the track.
    - For medium/long tracks: places num_segments windows distributed at approximately
      20%, 50%, and 80% of the duration, each with length up to segment_length.
      Adapts gracefully without exceeding track duration.

    Parameters:
        duration (float): Total audio duration in seconds.
        num_segments (int): Desired number of windows (default: 3).
        segment_length (float): Maximum window duration in seconds (default: 30.0).

    Returns:
        list of (offset, length) tuples.
    """
    if duration <= 0:
        raise ValueError("Audio duration must be greater than 0")

    # If track is shorter than or equal to window length, analyze in a single window
    if duration <= segment_length:
        return [(0.0, float(duration))]

    # Distribute window centers across duration
    if num_segments == 1:
        centers = [0.5 * duration]
    elif num_segments == 2:
        centers = [0.33 * duration, 0.67 * duration]
    else:
        # Default 3 windows at 20%, 50%, 80%
        centers = [
            (0.20 + i * (0.60 / max(1, num_segments - 1))) * duration
            for i in range(num_segments)
        ]

    windows = []
    for c in centers:
        # Center the window of length `segment_length` around `c`
        offset = max(0.0, c - segment_length / 2.0)
        # Ensure window doesn't exceed track duration
        if offset + segment_length > duration:
            offset = max(0.0, duration - segment_length)
        actual_len = min(segment_length, duration - offset)
        windows.append((round(float(offset), 3), round(float(actual_len), 3)))

    return windows


def extract_segmented_features(
    audio_path: str,
    num_segments: int = 3,
    segment_length: float = 30.0,
    aggregation: str = "rms_weighted",
) -> Tuple[np.ndarray, List[str], List[Dict[str, Any]]]:
    """
    Extracts multi-window feature vectors from an audio file and aggregates them.
    Implements Task 0.3 RMS-weighted pooling:
      weight_i = rms_i / sum(rms)
      aggregated_vector = sum(weight_i * feature_vector_i)
    Gives precedence to high-energy choruses and climactic sections, preventing
    quiet intros and fadeouts from dragging songs into Ambient.

    Parameters:
        audio_path (str): Path to audio file.
        num_segments (int): Target number of segments.
        segment_length (float): Duration per window in seconds.
        aggregation (str): Aggregation method ('rms_weighted', 'mean', or 'median').

    Returns:
        tuple: (aggregated_vector, feature_names, segment_features_list)
            - aggregated_vector: 1D np.ndarray of shape (89,)
            - feature_names: list of 89 feature names
            - segment_features_list: list of raw segment feature dicts
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Inspect full duration
    try:
        full_duration = float(librosa.get_duration(path=audio_path))
    except Exception:
        full_duration = 30.0

    windows = compute_segment_windows(full_duration, num_segments=num_segments, segment_length=segment_length)

    segment_vectors = []
    segment_features_list = []

    for offset, seg_dur in windows:
        seg_dict = analyze_audio(audio_path, duration_cap=seg_dur, offset=offset)
        seg_vec, _ = extract_feature_vector(seg_dict)
        segment_vectors.append(seg_vec)
        segment_features_list.append(seg_dict)

    stacked_vectors = np.array(segment_vectors, dtype=np.float32)

    if aggregation == "median":
        aggregated = np.median(stacked_vectors, axis=0)
    elif aggregation == "mean":
        aggregated = np.mean(stacked_vectors, axis=0)
    else:
        # Default RMS-weighted aggregation (Task 0.3)
        rms_energies = np.array([
            float(s.get("rms_mean", s.get("rms_energy", 0.0)))
            for s in segment_features_list
        ], dtype=np.float32)
        total_rms = float(np.sum(rms_energies))
        if total_rms > 1e-9:
            weights = rms_energies / total_rms
            aggregated = np.sum(weights[:, np.newaxis] * stacked_vectors, axis=0)
        else:
            aggregated = np.mean(stacked_vectors, axis=0)

    return aggregated.astype(np.float32), FEATURE_NAMES, segment_features_list

