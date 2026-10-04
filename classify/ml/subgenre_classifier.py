"""
subgenre_classifier.py
======================
Hierarchical Level-2 subgenre classification engine for CLASSIFY.
Maps predicted parent genres to their specific subgenre traditions
using acoustic signature profiles and musicological rules derived
from the database taxonomy.
Part of Phase 1 for CLASSIFY.
"""

from typing import Any, Dict, List, Optional
import numpy as np


# Subgenre acoustic profile definitions matching the 25 seeded database subgenres
SUBGENRE_PROFILES: Dict[str, List[Dict[str, Any]]] = {
    "Rock": [
        {
            "name": "Hard Rock",
            "slug": "hard-rock",
            "description": "Heavy distortion, aggressive power chords, and anthemic choruses.",
            "rules": lambda f: (
                (1.0 if f.get("rms_mean", 0.0) > 0.12 else 0.0) * 0.4 +
                (1.0 if f.get("spectral_contrast_mean", 0.0) > 22.0 or f.get("spectral_centroid_mean", 0.0) > 2400 else 0.0) * 0.3 +
                (1.0 if f.get("tempo", 120.0) >= 120 else 0.0) * 0.3
            ),
        },
        {
            "name": "Alternative Rock",
            "slug": "alternative-rock",
            "description": "Post-punk derived rock with unconventional guitar textures and expressive lyrics.",
            "rules": lambda f: (
                (1.0 if f.get("rms_var", 0.0) > 0.003 else 0.0) * 0.4 +
                (1.0 if 100 <= f.get("tempo", 120.0) <= 135 else 0.0) * 0.3 +
                (1.0 if f.get("spectral_rolloff_mean", 0.0) > 4000 else 0.0) * 0.3
            ),
        },
        {
            "name": "Classic Rock",
            "slug": "classic-rock",
            "description": "1960s-1970s rock emphasizing driving rhythms, blues riffs, and melodic vocal hooks.",
            "rules": lambda f: (
                (1.0 if 105 <= f.get("tempo", 120.0) <= 130 else 0.0) * 0.4 +
                (1.0 if 1800 <= f.get("spectral_centroid_mean", 0.0) <= 2500 else 0.0) * 0.3 +
                (1.0 if f.get("rms_mean", 0.0) <= 0.13 else 0.0) * 0.3
            ),
        },
    ],
    "Pop": [
        {
            "name": "Dance Pop",
            "slug": "dance-pop",
            "description": "Upbeat pop tailored for nightclub and dance floor energy with 4-on-the-floor kick drums.",
            "rules": lambda f: (
                (1.0 if 118 <= f.get("tempo", 120.0) <= 134 else 0.0) * 0.4 +
                (1.0 if f.get("onset_strength_mean", 0.0) > 1.3 or f.get("rms_mean", 0.0) > 0.12 else 0.0) * 0.3 +
                (1.0 if f.get("spectral_centroid_mean", 0.0) > 2300 else 0.0) * 0.3
            ),
        },
        {
            "name": "Synth Pop",
            "slug": "synth-pop",
            "description": "Pop dominated by analog and digital synthesizer timbres and sequenced arpeggios.",
            "rules": lambda f: (
                (1.0 if f.get("spectral_centroid_mean", 0.0) > 2400 else 0.0) * 0.4 +
                (1.0 if f.get("spectral_rolloff_mean", 0.0) > 4800 else 0.0) * 0.3 +
                (1.0 if 100 <= f.get("tempo", 120.0) <= 140 else 0.0) * 0.3
            ),
        },
        {
            "name": "Indie Pop",
            "slug": "indie-pop",
            "description": "Lighter melodic pop with acoustic guitars and understated DIY production aesthetics.",
            "rules": lambda f: (
                (1.0 if f.get("rms_mean", 0.0) < 0.11 else 0.0) * 0.4 +
                (1.0 if f.get("spectral_centroid_mean", 0.0) < 2200 else 0.0) * 0.3 +
                (1.0 if 85 <= f.get("tempo", 120.0) <= 125 else 0.0) * 0.3
            ),
        },
    ],
    "Classical": [
        {
            "name": "Baroque",
            "slug": "baroque",
            "description": "Polyphonic counterpoint, basso continuo, and ornamental keyboard/string writing.",
            "rules": lambda f: (
                (1.0 if f.get("onset_rate", 0.0) > 2.8 or f.get("tempo", 100.0) > 115 else 0.0) * 0.4 +
                (1.0 if f.get("spectral_centroid_mean", 0.0) > 1700 else 0.0) * 0.3 +
                (1.0 if f.get("zero_crossing_rate_mean", 0.0) > 0.04 else 0.0) * 0.3
            ),
        },
        {
            "name": "Chamber Music",
            "slug": "chamber-music",
            "description": "Intimate acoustic performance for small groups such as string quartets or piano trios.",
            "rules": lambda f: (
                (1.0 if f.get("rms_mean", 0.0) < 0.06 else 0.0) * 0.4 +
                (1.0 if f.get("rms_var", 0.0) < 0.002 else 0.0) * 0.3 +
                (1.0 if f.get("spectral_bandwidth_mean", 0.0) < 2100 else 0.0) * 0.3
            ),
        },
        {
            "name": "Orchestral",
            "slug": "orchestral",
            "description": "Large ensemble symphonic music balancing string, brass, woodwind, and percussion sections.",
            "rules": lambda f: (
                (1.0 if f.get("rms_var", 0.0) >= 0.002 else 0.0) * 0.4 +
                (1.0 if f.get("spectral_bandwidth_mean", 0.0) >= 2000 else 0.0) * 0.3 +
                (1.0 if f.get("rms_mean", 0.0) >= 0.05 else 0.0) * 0.3
            ),
        },
    ],
    "Jazz": [
        {
            "name": "Bebop",
            "slug": "bebop",
            "description": "Fast tempos, complex chord substitutions, and virtuoso soloing pioneered by Charlie Parker.",
            "rules": lambda f: (
                (1.0 if f.get("tempo", 110.0) >= 135 else 0.0) * 0.4 +
                (1.0 if f.get("onset_rate", 0.0) > 3.0 else 0.0) * 0.3 +
                (1.0 if f.get("chroma_stft_var", 0.0) > 0.05 else 0.0) * 0.3
            ),
        },
        {
            "name": "Jazz Fusion",
            "slug": "jazz-fusion",
            "description": "Blending jazz harmony and improvisation with rock rhythms and electric instruments.",
            "rules": lambda f: (
                (1.0 if f.get("rms_mean", 0.0) > 0.09 else 0.0) * 0.4 +
                (1.0 if f.get("spectral_centroid_mean", 0.0) > 2100 else 0.0) * 0.3 +
                (1.0 if 110 <= f.get("tempo", 110.0) <= 145 else 0.0) * 0.3
            ),
        },
        {
            "name": "Cool Jazz",
            "slug": "cool-jazz",
            "description": "Relaxed tempos, lighter tone colors, and classical-influenced arrangements.",
            "rules": lambda f: (
                (1.0 if f.get("tempo", 110.0) < 120 else 0.0) * 0.4 +
                (1.0 if f.get("rms_mean", 0.0) <= 0.09 else 0.0) * 0.3 +
                (1.0 if f.get("spectral_centroid_mean", 0.0) <= 2000 else 0.0) * 0.3
            ),
        },
    ],
    "Metal": [
        {
            "name": "Thrash Metal",
            "slug": "thrash-metal",
            "description": "High-velocity tempo, complex palm-muted riffing, and aggressive vocal delivery.",
            "rules": lambda f: (
                (1.0 if f.get("tempo", 130.0) >= 140 else 0.0) * 0.4 +
                (1.0 if f.get("onset_rate", 0.0) > 3.5 or f.get("onset_strength_mean", 0.0) > 1.4 else 0.0) * 0.3 +
                (1.0 if f.get("spectral_centroid_mean", 0.0) > 2500 else 0.0) * 0.3
            ),
        },
        {
            "name": "Heavy Metal",
            "slug": "heavy-metal",
            "description": "Classic galloping rhythms, dual guitar harmonies, and operatic or gritty vocals.",
            "rules": lambda f: (
                (1.0 if f.get("tempo", 130.0) < 140 else 0.0) * 0.4 +
                (1.0 if f.get("rms_mean", 0.0) > 0.12 else 0.0) * 0.3 +
                (1.0 if f.get("spectral_rolloff_mean", 0.0) > 4200 else 0.0) * 0.3
            ),
        },
    ],
    "Ambient": [
        {
            "name": "Drone",
            "slug": "drone",
            "description": "Sustained continuous tones and microtonal shifts creating deep acoustic resonance.",
            "rules": lambda f: (
                (1.0 if f.get("onset_rate", 0.0) < 1.2 else 0.0) * 0.4 +
                (1.0 if f.get("rms_var", 0.0) < 0.001 else 0.0) * 0.3 +
                (1.0 if f.get("spectral_centroid_mean", 0.0) < 1600 else 0.0) * 0.3
            ),
        },
        {
            "name": "Chillout",
            "slug": "chillout",
            "description": "Ambient with gentle downtempo electronic beats and warm pads.",
            "rules": lambda f: (
                (1.0 if f.get("onset_rate", 0.0) >= 1.2 or 65 <= f.get("tempo", 80.0) <= 105 else 0.0) * 0.5 +
                (1.0 if f.get("spectral_centroid_mean", 0.0) >= 1400 else 0.0) * 0.5
            ),
        },
    ],
    "Blues": [
        {
            "name": "Chicago Blues",
            "slug": "chicago-blues",
            "description": "Electrified urban blues with amplified harmonica, electric guitar, and full rhythm section.",
            "rules": lambda f: (
                (1.0 if f.get("rms_mean", 0.0) > 0.08 else 0.0) * 0.4 +
                (1.0 if f.get("spectral_centroid_mean", 0.0) > 1900 else 0.0) * 0.3 +
                (1.0 if 100 <= f.get("tempo", 100.0) <= 135 else 0.0) * 0.3
            ),
        },
        {
            "name": "Delta Blues",
            "slug": "delta-blues",
            "description": "Acoustic slide guitar, expressive foot-stomping rhythm, and passionate vocal delivery.",
            "rules": lambda f: (
                (1.0 if f.get("rms_mean", 0.0) <= 0.08 else 0.0) * 0.4 +
                (1.0 if f.get("spectral_centroid_mean", 0.0) <= 1900 else 0.0) * 0.3 +
                (1.0 if f.get("tempo", 100.0) < 115 else 0.0) * 0.3
            ),
        },
    ],
    "Country": [
        {
            "name": "Bluegrass",
            "slug": "bluegrass",
            "description": "High-speed acoustic string interplay featuring banjo rolls and fiddle breakdowns.",
            "rules": lambda f: (
                (1.0 if f.get("tempo", 115.0) >= 128 else 0.0) * 0.4 +
                (1.0 if f.get("onset_rate", 0.0) > 3.0 else 0.0) * 0.3 +
                (1.0 if f.get("spectral_centroid_mean", 0.0) > 2000 else 0.0) * 0.3
            ),
        },
        {
            "name": "Traditional Country",
            "slug": "traditional-country",
            "description": "Honky-tonk rhythms, pedal steel melodies, and heartfelt vocal twang.",
            "rules": lambda f: (
                (1.0 if f.get("tempo", 115.0) < 128 else 0.0) * 0.4 +
                (1.0 if f.get("rms_mean", 0.0) >= 0.07 else 0.0) * 0.3 +
                (1.0 if f.get("spectral_bandwidth_mean", 0.0) < 2300 else 0.0) * 0.3
            ),
        },
    ],
    "Hip-Hop": [
        {
            "name": "Trap",
            "slug": "trap",
            "description": "Rapid rolling hi-hats, sub-bass 808 glides, and layered atmospheric brass/synths.",
            "rules": lambda f: (
                (1.0 if f.get("zero_crossing_rate_mean", 0.0) > 0.07 or f.get("tempo", 90.0) > 125 else 0.0) * 0.4 +
                (1.0 if f.get("spectral_rolloff_mean", 0.0) > 4200 else 0.0) * 0.3 +
                (1.0 if f.get("rms_mean", 0.0) > 0.11 else 0.0) * 0.3
            ),
        },
        {
            "name": "Boom Bap",
            "slug": "boom-bap",
            "description": "Hard-hitting acoustic kick and snare sampling with syncopated jazz/soul loops.",
            "rules": lambda f: (
                (1.0 if 80 <= f.get("tempo", 90.0) <= 102 else 0.0) * 0.4 +
                (1.0 if f.get("zero_crossing_rate_mean", 0.0) <= 0.07 else 0.0) * 0.3 +
                (1.0 if f.get("spectral_centroid_mean", 0.0) <= 2200 else 0.0) * 0.3
            ),
        },
    ],
    "Disco": [
        {
            "name": "Nu-Disco",
            "slug": "nu-disco",
            "description": "Modern electronic disco blending classic groove with contemporary synthesizer production.",
            "rules": lambda f: 1.0,
        },
    ],
    "Reggae": [
        {
            "name": "Dub",
            "slug": "dub",
            "description": "Instrumental reggae remixes featuring heavy echo, reverb effects, and dominant bass dropouts.",
            "rules": lambda f: (
                (1.0 if f.get("spectral_centroid_mean", 0.0) < 1800 else 0.0) * 0.4 +
                (1.0 if f.get("rms_var", 0.0) > 0.002 else 0.0) * 0.3 +
                (1.0 if f.get("tempo", 75.0) < 85 else 0.0) * 0.3
            ),
        },
        {
            "name": "Roots Reggae",
            "slug": "roots-reggae",
            "description": "Spiritual, conscious lyrics with heavy one-drop rhythms and organ bubble.",
            "rules": lambda f: (
                (1.0 if f.get("spectral_centroid_mean", 0.0) >= 1800 else 0.0) * 0.4 +
                (1.0 if 70 <= f.get("tempo", 75.0) <= 90 else 0.0) * 0.3 +
                (1.0 if f.get("onset_strength_mean", 0.0) > 1.0 else 0.0) * 0.3
            ),
        },
    ],
}


def predict_subgenre(
    parent_genre: str,
    feature_dict: Optional[Dict[str, Any]] = None,
    feature_vector: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Predicts the specific subgenre under the given parent genre based on acoustic features.

    Parameters:
        parent_genre: Name of the top-predicted genre (e.g., 'Rock', 'Jazz')
        feature_dict: Raw audio features dictionary (tempo, rms, spectral centroid, etc.)
        feature_vector: 1D array of extracted acoustic features
        feature_names: Names corresponding to feature_vector

    Returns:
        dict with subgenre_name, subgenre_slug, confidence, description, or None if no subgenres.
    """
    subgenres = SUBGENRE_PROFILES.get(parent_genre)
    if not subgenres:
        return None

    # Merge vector and dict if provided
    feat: Dict[str, Any] = {}
    if feature_dict:
        feat.update(feature_dict)
    if feature_vector is not None and feature_names is not None:
        for name, val in zip(feature_names, feature_vector):
            if name not in feat:
                feat[name] = float(val)

    # If single subgenre (e.g. Disco -> Nu-Disco), return it directly with high confidence
    if len(subgenres) == 1:
        s = subgenres[0]
        return {
            "name": s["name"],
            "slug": s["slug"],
            "description": s["description"],
            "confidence": 0.85,
        }

    # Evaluate each subgenre candidate
    scores = []
    for s in subgenres:
        rule_fn = s.get("rules")
        score = float(rule_fn(feat)) if callable(rule_fn) else 0.5
        scores.append(score)

    total = sum(scores) + 1e-6
    probas = [s / total for s in scores]

    best_idx = int(np.argmax(probas))
    best_sub = subgenres[best_idx]
    best_conf = round(float(probas[best_idx]), 3)

    return {
        "name": best_sub["name"],
        "slug": best_sub["slug"],
        "description": best_sub["description"],
        "confidence": best_conf,
    }
