"""
test_audio_depth_phase4.py
==========================
Comprehensive tests for Phase 4: Audio Analysis Depth:
- Task 4.1: Time Signature / Meter Detection (4/4, 3/4, 6/8)
- Task 4.2: Chord Progression Detection via Chroma Triad Matching
- Task 4.3: Vocal vs. Instrumental Voice Activity Detection (VAD)
- Database persistence and AudioFeatures model integration
"""

import numpy as np
import pytest
from classify.audio.audio_processing import (
    detect_time_signature,
    detect_chord_progression,
    detect_vocal_presence,
    extract_features_from_signal,
    _CHORD_TEMPLATES,
)
from classify.models import AudioFeatures, Song, Analysis, User, Student
from classify.extensions import db


def test_time_signature_detection_defaults():
    """Verify fallback for empty or insufficient beats signal."""
    res = detect_time_signature(None)
    assert res["time_signature"] == "4/4"
    assert "Common Time" in res["meter_type"]

    # Short synthetic signal with few beats
    sr = 22050
    short_y = np.zeros(sr // 2, dtype=np.float32)
    res_short = detect_time_signature(y=short_y, sr=sr)
    assert res_short["time_signature"] == "4/4"


def test_time_signature_synthetic_meter():
    """Verify meter detection distinguishing triple (3/4) vs quadruple (4/4)."""
    # Create synthetic onset strength envelope with strong lag-3 periodicity (3/4 waltz)
    n_beats = 40
    # Beat pattern: Strong, Weak, Weak (3 beats per measure)
    pattern_34 = np.array([1.0, 0.2, 0.2] * 14)[:n_beats]
    onset_34 = np.zeros(n_beats * 10, dtype=np.float32)
    beat_frames = np.arange(0, n_beats * 10, 10)
    onset_34[beat_frames] = pattern_34

    res_34 = detect_time_signature(onset_env=onset_34, beats=beat_frames)
    assert res_34["time_signature"] == "3/4"
    assert "Triple" in res_34["meter_type"]

    # Beat pattern: Strong, Weak, Medium, Weak (4 beats per measure)
    pattern_44 = np.array([1.0, 0.2, 0.6, 0.2] * 10)[:n_beats]
    onset_44 = np.zeros(n_beats * 10, dtype=np.float32)
    onset_44[beat_frames] = pattern_44

    res_44 = detect_time_signature(onset_env=onset_44, beats=beat_frames)
    assert res_44["time_signature"] == "4/4"
    assert "Common" in res_44["meter_type"]


def test_chord_progression_template_matching():
    """Verify triad detection and sequence extraction."""
    # Synthetic C Major: pitches C (0), E (4), G (7)
    c_maj_chroma = np.zeros((12, 30), dtype=np.float32)
    c_maj_chroma[0, :] = 1.0
    c_maj_chroma[4, :] = 0.8
    c_maj_chroma[7, :] = 0.9

    res_c = detect_chord_progression(chroma=c_maj_chroma)
    assert "C" in res_c["chord_progression"]
    assert res_c["top_chords"][0]["chord"] == "C"

    # Synthetic A Minor: pitches A (9), C (0), E (4)
    a_min_chroma = np.zeros((12, 30), dtype=np.float32)
    a_min_chroma[9, :] = 1.0
    a_min_chroma[0, :] = 0.8
    a_min_chroma[4, :] = 0.9

    res_am = detect_chord_progression(chroma=a_min_chroma)
    assert "Am" in res_am["chord_progression"]

    # Two-chord sequence: C then Am
    c_am_chroma = np.hstack([c_maj_chroma, a_min_chroma])
    res_seq = detect_chord_progression(chroma=c_am_chroma)
    assert "C" in res_seq["chord_progression"]
    assert "Am" in res_seq["chord_progression"]
    assert "→" in res_seq["progression_str"]


def test_vocal_presence_detection():
    """Verify vocal detection on synthetic signals."""
    sr = 22050
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)

    # 1. Pure sine wave (Instrumental pure tone)
    sine_y = 0.5 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    res_inst = detect_vocal_presence(sine_y, sr=sr)
    assert res_inst["vocal_presence"] in ["Vocal", "Instrumental"]
    assert 0.0 <= res_inst["vocal_probability"] <= 1.0
    assert 0.0 <= res_inst["vocal_confidence"] <= 1.0

    # 2. None/empty audio fallback
    res_none = detect_vocal_presence(None)
    assert res_none["vocal_presence"] == "Instrumental"


def test_end_to_end_audio_depth_persistence(app):
    """Test full extraction and database persistence of Phase 4 features."""
    with app.app_context():
        # Setup student and song
        user = User(email="audio_depth_student@music.edu", role="student", email_verified=True)
        user.set_password("DeepAudioPass1!")
        db.session.add(user)
        db.session.commit()

        student = Student(user_id=user.id)
        db.session.add(student)
        db.session.commit()

        song = Song(
            stored_filename="waltz_depth_stored.wav",
            original_filename="waltz_depth.wav",
            student_id=student.id,
            duration=3.0,
            file_size_kb=250.0,
        )
        db.session.add(song)
        db.session.commit()

        analysis = Analysis(song_id=song.id)
        db.session.add(analysis)
        db.session.commit()

        # Generate 3-second audio signal
        sr = 22050
        t = np.linspace(0, 3.0, sr * 3, endpoint=False)
        y = (0.4 * np.sin(2 * np.pi * 261.63 * t) + 0.3 * np.sin(2 * np.pi * 329.63 * t)).astype(np.float32)

        features = extract_features_from_signal(
            y=y,
            sr=sr,
            file_name="waltz_depth.wav",
            file_size_kb=250.0,
            full_duration=3.0,
            native_sr=sr,
            native_channels=1,
            preprocess=False,
        )

        assert "time_signature" in features
        assert "chord_progression" in features
        assert "vocal_presence" in features
        assert "detected_key" in features

        # Persist to database
        af = AudioFeatures(
            analysis_id=analysis.id,
            duration=features["duration"],
            sample_rate=features["sample_rate"],
            tempo=features["tempo"],
            rms_mean=features["rms_mean"],
            zcr_mean=features["zcr_mean"],
            spectral_centroid=features["spectral_centroid"],
            spectral_bandwidth=features["spectral_bandwidth_mean"],
            spectral_rolloff=features["spectral_rolloff_mean"],
            detected_key=features["detected_key"],
            time_signature=features["time_signature"],
            chord_progression=features["progression_str"],
            vocal_presence=features["vocal_presence"],
        )
        db.session.add(af)
        db.session.commit()

        # Query back and verify
        saved_af = AudioFeatures.query.filter_by(analysis_id=analysis.id).first()
        assert saved_af is not None
        assert saved_af.time_signature == features["time_signature"]
        assert saved_af.chord_progression == features["progression_str"]
        assert saved_af.vocal_presence == features["vocal_presence"]
