"""
teaching_topics_service.py
==========================
Generates evidence-backed music pedagogy and instructional topics for teachers
reviewing student submissions. Correlates predicted/verified musical genres and
Tier 1 acoustic feature measurements with targeted curriculum concepts.
Part of Phase 7 for CLASSIFY.
"""

GENRE_PEDAGOGICAL_GUIDELINES = {
    "Rock": {
        "core_topics": [
            "Syncopated Rhythm & Backbeat Accents (Beats 2 & 4)",
            "Dynamic Range & Distortion Harmonic Control",
            "Drive, Palm Muting & Power Chord Voicing",
            "Riff Formulation & Melodic Phrasing over 4/4 Meters",
        ],
        "listening_focus": "Rhythm section locking between kick drum and bassline; guitar frequency balance in the mid-register (800 Hz - 3 kHz).",
    },
    "Pop": {
        "core_topics": [
            "Melodic Contour & Memorable Hook Construction",
            "Verse-Chorus Structural Tension & Transition Dynamics",
            "Vocal Clarity, Formant Spacing & Centered Intonation",
            "Harmonic Simplicity: Diatonic Cadences (I-V-vi-IV)",
        ],
        "listening_focus": "Spectral balance and high vocal presence without excessive high-frequency sibilance.",
    },
    "Classical": {
        "core_topics": [
            "Rubato & Dynamic Expressivity (Pianissimo to Fortissimo)",
            "Polyphony, Voice Leading & Counterpoint Balance",
            "Harmonic Modulation & Cadential Preparation",
            "Tone Production & Acoustic Resonance (Centroid & Harmonics)",
        ],
        "listening_focus": "Dynamic headroom, natural room acoustics, and pure acoustic instrument timbral separation.",
    },
    "Jazz": {
        "core_topics": [
            "Swing Feel, Rhythmic Displacement & Syncopation",
            "Extended & Altered Chord Voicings (Maj7, Min9, 13b9)",
            "Improvisation Frameworks (Guide Tones, Pentatonics & Modes)",
            "Interactive Accompaniment ('Comping') & Call-and-Response",
        ],
        "listening_focus": "Walking bass consistency, ride cymbal articulation, and tonal shading on modal shifts.",
    },
    "Metal": {
        "core_topics": [
            "High-Tempo Articulation & Double-Kick Synchronization",
            "Intense Palm Muting Dynamics & Low-End Tightness",
            "Modal Minor Exploration (Aeolian, Phrygian & Harmonic Minor)",
            "Extended Meter & Odd Time Signature Execution",
        ],
        "listening_focus": "Transient attack preservation in heavy saturation; scooped vs mid-boosted guitar eq.",
    },
    "Ambient": {
        "core_topics": [
            "Textural Soundscapes & Timbral Evolution over Time",
            "Harmonic Sustained Drones & Extended Reverb Decays",
            "Minimalist Pitch Centers & Micro-Tonal Shifts",
            "Dynamic Subtlety & Ambient Space Management",
        ],
        "listening_focus": "Spectral width, low-frequency rumble control, and slow envelope attack/decay stages.",
    },
    "Blues": {
        "core_topics": [
            "12-Bar Blues Form & Dominant 7th Harmonic Progression",
            "Blue Notes (b3, b5, b7) & Microtonal String Bending",
            "Call-and-Response Phrasing Between Voice and Lead Instrument",
            "Shuffle / Triplet Feel & Rhythmic Grounding",
        ],
        "listening_focus": "Expressive intonation on bends and emotional phrasing around tonic release.",
    },
    "Country": {
        "core_topics": [
            "Narrative Lyric Phrasing & Two-Step / Boom-Chick Rhythm",
            "Pedal Steel, Twang & Acoustic Guitar Fingerpicking",
            "Triadic Harmony & Major Pentatonic Fill Work",
            "Clean Vocal Projection with Natural Vibrato Control",
        ],
        "listening_focus": "Treble twang articulation (2-4 kHz) and acoustic resonance in acoustic instruments.",
    },
    "Hip-Hop": {
        "core_topics": [
            "Sub-Bass (808) Tuning & Kick Transient Layering",
            "Cadence, Flow & Micro-Rhythmic Syllable Placement",
            "Sample Manipulation, Loop Phrasing & Texture Grids",
            "Syncopated High-Hat Rolls & Pocket Timing",
        ],
        "listening_focus": "Low-frequency headroom (<100 Hz) and crisp snare/vocal separation in the stereo field.",
    },
    "Disco": {
        "core_topics": [
            "Four-on-the-Floor Kick Foundation & 16th-Note Hi-Hat Drive",
            "Syncopated Octave Basslines & Groove Pocket",
            "String and Brass Stabs for Dynamic Accentuation",
            "Bright, Punchy Mix Architecture & Polished Energy",
        ],
        "listening_focus": "Steady tempo adherence (typically 115-130 BPM) and rhythmic bass prominence.",
    },
    "Reggae": {
        "core_topics": [
            "One-Drop Drum Feel (Accent on Beat 3) & Rim Shot Articulation",
            "Skank / Chop Accents on the Off-Beats (2 and 4)",
            "Deep Melodic Basslines Anchoring the Harmonic Center",
            "Space, Dub Delays & Echo Placement as Compositional Elements",
        ],
        "listening_focus": "Deep sub-bass fundamentals and rhythmic space without cluttering the off-beat chops.",
    },
}


def get_teaching_topics(genre_name, audio_features=None):
    """
    Returns pedagogical guidance and topic recommendations tailored to
    the genre and acoustic properties of the student's submission.

    Parameters:
        genre_name (str): The primary predicted or selected genre.
        audio_features (dict or AudioFeatures): Extracted acoustic measurements.

    Returns:
        dict: {
            "topics": list of str,
            "listening_focus": str,
            "tempo_note": str or None,
            "spectral_note": str or None,
        }
    """
    matched_genre = "Rock"  # Default fallback
    for g, data in GENRE_PEDAGOGICAL_GUIDELINES.items():
        if g.lower() in (genre_name or "").lower():
            matched_genre = g
            break

    guidelines = GENRE_PEDAGOGICAL_GUIDELINES.get(
        matched_genre, GENRE_PEDAGOGICAL_GUIDELINES["Rock"]
    )
    topics = list(guidelines["core_topics"])
    listening_focus = guidelines["listening_focus"]

    tempo_note = None
    spectral_note = None

    if audio_features:
        tempo = getattr(audio_features, "tempo", None)
        if tempo is None and isinstance(audio_features, dict):
            tempo = audio_features.get("tempo")

        centroid = getattr(audio_features, "spectral_centroid", None)
        if centroid is None and isinstance(audio_features, dict):
            centroid = audio_features.get("spectral_centroid_mean")

        if tempo is not None:
            try:
                bpm = float(tempo)
                if bpm > 145:
                    tempo_note = f"High tempo recorded ({bpm:.1f} BPM). Focus lesson on precision, metronomic pacing, and minimizing tension at speed."
                elif bpm < 75:
                    tempo_note = f"Slow ballad/downtempo pacing ({bpm:.1f} BPM). Focus lesson on sustained breath/bow support and subdividing inner beats."
                else:
                    tempo_note = f"Moderate standard tempo ({bpm:.1f} BPM). Suitable for exploring groove stability and subtle rhythmic micro-timing."
            except (ValueError, TypeError):
                pass

        if centroid is not None:
            try:
                sc = float(centroid)
                if sc > 3000:
                    spectral_note = f"Bright timbral profile ({sc:.0f} Hz). Discuss harmonic overtone control and avoiding harsh top-end frequencies."
                elif sc < 1400:
                    spectral_note = f"Warm, darker timbral profile ({sc:.0f} Hz). Discuss presence, articulation, and cutting through an ensemble mix."
            except (ValueError, TypeError):
                pass

    return {
        "genre": matched_genre,
        "topics": topics,
        "listening_focus": listening_focus,
        "tempo_note": tempo_note,
        "spectral_note": spectral_note,
    }
