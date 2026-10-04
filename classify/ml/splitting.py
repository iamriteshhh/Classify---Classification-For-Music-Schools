"""
splitting.py
============
Artist-aware stratified splitting for audio genre classification datasets.
Guarantees zero artist leakage across train, validation, and test splits while
ensuring balanced class representation across splits.
Part of Phase 1 for CLASSIFY.
"""

import logging
import random
from collections import defaultdict
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


def artist_aware_stratified_split(
    manifest_rows: List[Dict[str, Any]],
    val_size: float = 0.15,
    test_size: float = 0.15,
    min_test_support_per_class: int = 1,
    random_state: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Partitions manifest rows into train, validation, and test splits with strict
    artist-level isolation and per-genre stratification.

    Parameters:
        manifest_rows (list of dict): Rows loaded from dataset manifest.
        val_size (float): Proportion of data/artists for validation (default: 0.15).
        test_size (float): Proportion of data/artists for testing (default: 0.15).
        min_test_support_per_class (int): Minimum artists per class for test split if available.
        random_state (int): Seed for deterministic splitting.

    Returns:
        tuple: (train_rows, val_rows, test_rows, split_report)
            - train_rows: list of row dicts assigned to train
            - val_rows: list of row dicts assigned to validation
            - test_rows: list of row dicts assigned to test
            - split_report: dictionary summarizing per-class and total track/artist counts,
                            indices, and any warnings.
    """
    if not manifest_rows:
        raise ValueError("manifest_rows cannot be empty")

    if val_size <= 0 or test_size <= 0 or (val_size + test_size) >= 1.0:
        raise ValueError(f"Invalid split proportions: val_size={val_size}, test_size={test_size}")

    # Group track indices and artists by genre
    genre_artists: Dict[str, List[str]] = defaultdict(list)
    genre_artist_rows: Dict[str, Dict[str, List[Tuple[int, Dict[str, Any]]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for idx, row in enumerate(manifest_rows):
        genre = str(row.get("genre", "Unknown"))
        artist = str(row.get("artist_id", f"unknown_artist_{idx}"))
        genre_artist_rows[genre][artist].append((idx, row))

    rng = random.Random(random_state)
    warnings_list: List[str] = []

    train_artist_set = set()
    val_artist_set = set()
    test_artist_set = set()

    per_class_summary: Dict[str, Dict[str, int]] = {}

    # Sort genres for determinism
    sorted_genres = sorted(genre_artist_rows.keys())

    # Pre-check for any artists appearing across multiple genres
    artist_genres: Dict[str, set] = defaultdict(set)
    for genre, artists_dict in genre_artist_rows.items():
        for artist in artists_dict:
            artist_genres[artist].add(genre)

    # First partition artists per genre
    for genre in sorted_genres:
        artists_dict = genre_artist_rows[genre]
        unique_artists = sorted(artists_dict.keys())
        # Shuffle artists deterministically with genre-derived salt
        genre_seed = rng.randint(0, 10_000_000)
        local_rng = random.Random(genre_seed)
        local_rng.shuffle(unique_artists)

        n_artists = len(unique_artists)

        if n_artists < 3:
            warn_msg = (
                f"Genre '{genre}' has only {n_artists} unique artist(s) (< 3). "
                f"Cannot allocate independent artists to train, validation, and test."
            )
            warnings_list.append(warn_msg)
            logger.warning(warn_msg)

            if n_artists == 1:
                g_train = unique_artists
                g_val: List[str] = []
                g_test: List[str] = []
            elif n_artists == 2:
                # 1 to train, 1 to test (so test has representation if possible)
                g_train = [unique_artists[0]]
                g_val = []
                g_test = [unique_artists[1]]
        else:
            # Calculate target test and val counts based on artist counts
            n_test = max(min_test_support_per_class, int(round(n_artists * test_size)))
            n_val = max(min_test_support_per_class, int(round(n_artists * val_size)))

            # Ensure at least 1 artist remains in train
            if (n_test + n_val) >= n_artists:
                # Scale down if needed while keeping at least 1 in train
                remaining_for_eval = n_artists - 1
                n_test = max(1, remaining_for_eval // 2)
                n_val = max(1, remaining_for_eval - n_test)

            g_test = unique_artists[:n_test]
            g_val = unique_artists[n_test : n_test + n_val]
            g_train = unique_artists[n_test + n_val :]

        # Handle any artist that might have already been assigned in another genre
        for a in g_test:
            if a in train_artist_set or a in val_artist_set:
                continue
            test_artist_set.add(a)

        for a in g_val:
            if a in train_artist_set or a in test_artist_set:
                continue
            val_artist_set.add(a)

        for a in g_train:
            if a in val_artist_set or a in test_artist_set:
                continue
            train_artist_set.add(a)

    # Reconcile all tracks based on artist sets
    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []

    train_rows: List[Dict[str, Any]] = []
    val_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []

    for idx, row in enumerate(manifest_rows):
        artist = str(row.get("artist_id", f"unknown_artist_{idx}"))
        if artist in test_artist_set:
            test_indices.append(idx)
            test_rows.append(row)
        elif artist in val_artist_set:
            val_indices.append(idx)
            val_rows.append(row)
        else:
            # Defaults to train
            train_indices.append(idx)
            train_rows.append(row)
            train_artist_set.add(artist)

    # Verification: Zero artist leakage
    final_train_artists = {str(r.get("artist_id")) for r in train_rows}
    final_val_artists = {str(r.get("artist_id")) for r in val_rows}
    final_test_artists = {str(r.get("artist_id")) for r in test_rows}

    assert len(final_train_artists & final_val_artists) == 0, (
        f"Artist leakage between Train and Validation: {final_train_artists & final_val_artists}"
    )
    assert len(final_train_artists & final_test_artists) == 0, (
        f"Artist leakage between Train and Test: {final_train_artists & final_test_artists}"
    )
    assert len(final_val_artists & final_test_artists) == 0, (
        f"Artist leakage between Validation and Test: {final_val_artists & final_test_artists}"
    )

    # Compute per-class counts
    for genre in sorted_genres:
        c_train_tracks = sum(1 for r in train_rows if str(r.get("genre")) == genre)
        c_val_tracks = sum(1 for r in val_rows if str(r.get("genre")) == genre)
        c_test_tracks = sum(1 for r in test_rows if str(r.get("genre")) == genre)

        c_train_artists = len({str(r.get("artist_id")) for r in train_rows if str(r.get("genre")) == genre})
        c_val_artists = len({str(r.get("artist_id")) for r in val_rows if str(r.get("genre")) == genre})
        c_test_artists = len({str(r.get("artist_id")) for r in test_rows if str(r.get("genre")) == genre})

        per_class_summary[genre] = {
            "train_tracks": c_train_tracks,
            "val_tracks": c_val_tracks,
            "test_tracks": c_test_tracks,
            "train_artists": c_train_artists,
            "val_artists": c_val_artists,
            "test_artists": c_test_artists,
        }

    split_report = {
        "per_class": per_class_summary,
        "total_tracks": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
            "total": len(manifest_rows),
        },
        "total_artists": {
            "train": len(final_train_artists),
            "val": len(final_val_artists),
            "test": len(final_test_artists),
        },
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
        "warnings": warnings_list,
        "zero_artist_leakage_verified": True,
    }

    return train_rows, val_rows, test_rows, split_report
