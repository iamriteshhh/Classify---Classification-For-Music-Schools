"""
test_dataset_validation.py
==========================
Dataset validation suite checking:
- Manifest file existence and required columns
- Non-empty dataset rows
- License and source documentation per track
- No duplicate track paths
- Strict artist-level train/validation/test split with zero artist overlap
Part of Phase 2 for CLASSIFY.
"""

import os
import pytest
from sklearn.model_selection import GroupShuffleSplit
from classify.ml.dataset import load_manifest, MANIFEST_HEADERS, MANIFEST_PATH


def test_manifest_structure_and_completeness():
    """Verify that manifest.csv exists, has required headers, and has complete fields."""
    assert os.path.exists(MANIFEST_PATH), f"Manifest file does not exist at {MANIFEST_PATH}"
    rows = load_manifest(MANIFEST_PATH)
    assert len(rows) > 0, "Manifest has no rows."

    track_paths = set()
    for idx, row in enumerate(rows):
        # Check all headers exist
        for header in MANIFEST_HEADERS:
            assert header in row, f"Missing header '{header}' in row {idx}"
            assert row[header].strip() != "", f"Empty value for '{header}' in row {idx}"

        # Check license is documented
        assert len(row["license"].strip()) > 3, f"Inadequate license documentation in row {idx}"
        assert len(row["source"].strip()) > 0, f"Missing source in row {idx}"

        # Check for unique track paths (no duplicates)
        t_path = row["track_path"]
        assert t_path not in track_paths, f"Duplicate track_path found: {t_path}"
        track_paths.add(t_path)


def test_artist_level_splitting_no_leakage():
    """Verify that artist-based splitting has zero artist overlap across train/val/test sets."""
    rows = load_manifest(MANIFEST_PATH)
    artists = [r["artist_id"] for r in rows]
    genres = [r["genre"] for r in rows]
    indices = list(range(len(rows)))

    # Test artist-level split logic
    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_val_idx, test_idx = next(gss_test.split(indices, genres, groups=artists))

    artists_train_val = [artists[i] for i in train_val_idx]
    artists_test = set(artists[i] for i in test_idx)

    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.18, random_state=42)
    train_sub_idx, val_sub_idx = next(gss_val.split(train_val_idx, [genres[i] for i in train_val_idx], groups=artists_train_val))

    artists_train = set(artists_train_val[i] for i in train_sub_idx)
    artists_val = set(artists_train_val[i] for i in val_sub_idx)

    # CRITICAL CHECK: Zero artist overlap
    assert len(artists_train & artists_val) == 0, f"Artist leakage detected between Train and Val: {artists_train & artists_val}"
    assert len(artists_train & artists_test) == 0, f"Artist leakage detected between Train and Test: {artists_train & artists_test}"
    assert len(artists_val & artists_test) == 0, f"Artist leakage detected between Val and Test: {artists_val & artists_test}"


def test_artist_aware_stratified_split_preserves_classes_and_no_leakage():
    """
    Verify artist_aware_stratified_split ensures:
    1. Zero artist leakage across all splits.
    2. Non-zero test support for every genre that has >= 3 artists.
    3. Proper track and artist accounting in split report.
    """
    from classify.ml.splitting import artist_aware_stratified_split

    rows = load_manifest(MANIFEST_PATH)
    train_rows, val_rows, test_rows, split_report = artist_aware_stratified_split(
        rows, val_size=0.15, test_size=0.15, min_test_support_per_class=1, random_state=42
    )

    # 1. Zero artist leakage
    train_artists = {r["artist_id"] for r in train_rows}
    val_artists = {r["artist_id"] for r in val_rows}
    test_artists = {r["artist_id"] for r in test_rows}

    assert len(train_artists & val_artists) == 0, f"Leakage Train-Val: {train_artists & val_artists}"
    assert len(train_artists & test_artists) == 0, f"Leakage Train-Test: {train_artists & test_artists}"
    assert len(val_artists & test_artists) == 0, f"Leakage Val-Test: {val_artists & test_artists}"

    # 2. Non-zero test support for every genre with >= 3 artists
    from collections import defaultdict
    genre_artist_counts = defaultdict(set)
    for r in rows:
        genre_artist_counts[r["genre"]].add(r["artist_id"])

    test_genres = {r["genre"] for r in test_rows}
    for genre, artists in genre_artist_counts.items():
        if len(artists) >= 3:
            assert genre in test_genres, f"Genre '{genre}' (has {len(artists)} artists) missing from test split!"
            test_support = split_report["per_class"][genre]["test_tracks"]
            assert test_support > 0, f"Genre '{genre}' has 0 test track support!"

    # 3. Total track conservation
    assert len(train_rows) + len(val_rows) + len(test_rows) == len(rows)

