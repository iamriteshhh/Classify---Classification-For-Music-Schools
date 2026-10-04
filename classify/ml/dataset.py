"""
dataset.py
==========
Dataset loading, generation, schema definitions, and validation for CLASSIFY.
Manages datasets/manifest.csv and ensures artist-level grouping to prevent data leakage.
Part of Phase 2 for CLASSIFY.
"""

import csv
import os
import hashlib
import numpy as np
import soundfile as sf

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
MANIFEST_PATH = os.path.join(DATASETS_DIR, "manifest.csv")
PROCESSED_AUDIO_DIR = os.path.join(DATASETS_DIR, "processed")

# Extended manifest schema adhering to Phase 2 requirements
MANIFEST_HEADERS = [
    "track_id",
    "track_path",
    "genre",
    "subgenre",
    "cultural_tag",
    "artist_id",
    "dataset_name",
    "dataset_track_id",
    "checksum",
    "duration",
    "album",
    "year",
    "split",
    "license",
    "source",
]


def load_manifest(manifest_path=None):
    """
    Loads dataset manifest rows into a list of dictionaries.

    Returns:
        list of dict: Rows from manifest.csv.
    """
    path = manifest_path or MANIFEST_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset manifest not found at: {path}")

    rows = []
    with open(path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def generate_baseline_dataset(
    target_dir=None,
    manifest_path=None,
    tracks_per_artist=3,
    sample_rate=22050,
    duration=3.0,
):
    """
    Generates a calibrated synthetic genre audio dataset with distinct acoustic profiles
    and explicit artist IDs to test and train the baseline ML pipeline.
    Preserved for historical and regression reference.
    """
    out_audio_dir = target_dir or PROCESSED_AUDIO_DIR
    out_manifest = manifest_path or MANIFEST_PATH
    os.makedirs(out_audio_dir, exist_ok=True)
    os.makedirs(os.path.dirname(out_manifest), exist_ok=True)

    genre_profiles = {
        "Rock": {
            "tempo": 136,
            "base_freq": 220.0,
            "harmonics": [1.0, 0.8, 0.6, 0.5, 0.4],
            "distortion": 0.35,
            "snare_bright": True,
            "artists": ["artist_rock_01", "artist_rock_02", "artist_rock_03", "artist_rock_04"],
            "subgenres": ["Classic Rock", "Hard Rock"],
        },
        "Pop": {
            "tempo": 120,
            "base_freq": 261.63,
            "harmonics": [1.0, 0.5, 0.25, 0.1],
            "distortion": 0.05,
            "snare_bright": True,
            "artists": ["artist_pop_01", "artist_pop_02", "artist_pop_03", "artist_pop_04"],
            "subgenres": ["Synth Pop", "Dance Pop"],
        },
        "Classical": {
            "tempo": 68,
            "base_freq": 174.61,
            "harmonics": [1.0, 0.7, 0.3, 0.1],
            "distortion": 0.0,
            "snare_bright": False,
            "artists": ["artist_classical_01", "artist_classical_02", "artist_classical_03", "artist_classical_04"],
            "subgenres": ["Symphonic", "Chamber Music"],
        },
        "Jazz": {
            "tempo": 96,
            "base_freq": 196.0,
            "harmonics": [1.0, 0.6, 0.5, 0.3, 0.2],
            "distortion": 0.02,
            "snare_bright": False,
            "artists": ["artist_jazz_01", "artist_jazz_02", "artist_jazz_03", "artist_jazz_04"],
            "subgenres": ["Bebop", "Cool Jazz"],
        },
        "Metal": {
            "tempo": 152,
            "base_freq": 110.0,
            "harmonics": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
            "distortion": 0.75,
            "snare_bright": True,
            "artists": ["artist_metal_01", "artist_metal_02", "artist_metal_03", "artist_metal_04"],
            "subgenres": ["Heavy Metal", "Thrash Metal"],
        },
        "Ambient": {
            "tempo": 58,
            "base_freq": 130.81,
            "harmonics": [1.0, 0.4, 0.2],
            "distortion": 0.0,
            "snare_bright": False,
            "artists": ["artist_ambient_01", "artist_ambient_02", "artist_ambient_03", "artist_ambient_04"],
            "subgenres": ["Drone", "Space Ambient"],
        },
    }

    manifest_rows = []
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)

    for genre, profile in genre_profiles.items():
        artists = profile["artists"]
        harmonics = profile["harmonics"]
        f0 = profile["base_freq"]
        subgenres = profile["subgenres"]

        for artist_idx, artist_id in enumerate(artists):
            for track_num in range(1, tracks_per_artist + 1):
                track_file_name = f"{genre.lower()}_{artist_id}_t{track_num}.wav"
                track_full_path = os.path.join(out_audio_dir, track_file_name)

                # Synthesize deterministic signal
                signal = np.zeros(num_samples)
                for h_idx, h_weight in enumerate(harmonics, start=1):
                    freq = f0 * h_idx
                    if freq < sample_rate / 2:
                        phase = 0.1 * artist_idx + 0.2 * track_num
                        signal += h_weight * np.sin(2 * np.pi * freq * t + phase)

                # Apply non-linear distortion / saturation if specified
                if profile["distortion"] > 0:
                    dist_gain = 1.0 + 3.0 * profile["distortion"]
                    signal = np.tanh(dist_gain * signal)

                # Add rhythmic beat impulses according to genre tempo
                beat_interval = int(sample_rate * (60.0 / profile["tempo"]))
                if profile["snare_bright"]:
                    for b in range(0, len(signal), beat_interval):
                        end_b = min(b + int(sample_rate * 0.03), len(signal))
                        noise = np.random.normal(0, 0.4, end_b - b)
                        signal[b:end_b] += noise

                # Normalize amplitude
                max_amp = np.max(np.abs(signal))
                if max_amp > 0:
                    signal = 0.8 * (signal / max_amp)

                sf.write(track_full_path, signal, sample_rate)

                rel_path = os.path.relpath(track_full_path, BASE_DIR).replace("\\", "/")
                subgenre = subgenres[track_num % len(subgenres)]
                checksum = hashlib.sha256(signal.astype(np.float32).tobytes()).hexdigest()
                track_id = f"synth_{genre.lower()}_{artist_id}_t{track_num}"

                manifest_rows.append({
                    "track_id": track_id,
                    "track_path": rel_path,
                    "genre": genre,
                    "subgenre": subgenre,
                    "cultural_tag": "Global / Western",
                    "artist_id": artist_id,
                    "dataset_name": "CLASSIFY-Synthetic-Engine",
                    "dataset_track_id": f"{genre}_{artist_id}_{track_num}",
                    "checksum": checksum,
                    "duration": str(round(duration, 2)),
                    "album": "Synthetic Audio Collection",
                    "year": "2026",
                    "split": "baseline",
                    "license": "CC-BY-4.0 (Educational Project Synthesized Audio)",
                    "source": "CLASSIFY-Audio-Engine",
                })

    # Write manifest.csv
    with open(out_manifest, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_HEADERS)
        writer.writeheader()
        writer.writerows(manifest_rows)

    return out_manifest, len(manifest_rows)
