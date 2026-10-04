"""
analysis_service.py
===================
Orchestrates audio validation, feature extraction, ML classification,
and database persistence.
Part of Phase 2, 3, & 6 for CLASSIFY.
"""

from classify.extensions import db
from classify.models import Song, Analysis, AudioFeatures, Prediction, Genre, Subgenre
from classify.services.audio_service import process_audio
from classify.services.classification_service import classify_audio_features


def analyze_uploaded_audio(
    file_path,
    original_filename=None,
    stored_filename=None,
    user=None,
    allowed_extensions=None,
    max_size_bytes=None,
):
    """
    Complete analysis and persistence workflow:
    1. Validates audio format, size, duration, and silence.
    2. Extracts Tier 1 acoustic features.
    3. Runs ML inference for genre prediction and confidence.
    4. Persists Song, Analysis, AudioFeatures, and Prediction to the database.

    Returns:
        dict: {
            "features": dict,
            "classification": dict,
            "validation": dict,
            "analysis_id": int or None,
            "song_id": int or None,
        }
    """
    features, validation_info = process_audio(
        file_path,
        allowed_extensions=allowed_extensions,
        max_size_bytes=max_size_bytes,
    )
    classification = classify_audio_features(features, file_path=file_path)

    # If full track duration was measured during multi-segment analysis, update features duration
    if classification.get("coverage") and classification["coverage"].get("original_duration"):
        features["duration"] = float(classification["coverage"]["original_duration"])

    # 4. Database Persistence
    analysis_id = None
    song_id = None

    try:
        student_id = None
        if user and getattr(user, "is_authenticated", False) and getattr(user, "student_profile", None):
            student_id = user.student_profile.id

        # Determine filenames
        orig_name = original_filename or features.get("file_name", "unknown.wav")
        stor_name = stored_filename or orig_name

        # Create Song record
        song = Song(
            student_id=student_id,
            stored_filename=stor_name,
            original_filename=orig_name,
            duration=features.get("duration"),
            file_size_kb=features.get("file_size_kb"),
        )
        db.session.add(song)
        db.session.flush()
        song_id = song.id

        # Create Analysis event
        analysis = Analysis(song_id=song.id)
        db.session.add(analysis)
        db.session.flush()
        analysis_id = analysis.id

        # Create AudioFeatures record
        audio_feat = AudioFeatures(
            analysis_id=analysis.id,
            duration=float(features.get("duration", 0.0)),
            sample_rate=int(features.get("sample_rate", 22050)),
            channels=int(features.get("channels", 1)),
            tempo=float(features.get("tempo", 120.0)),
            rms_mean=float(features.get("rms_mean", 0.0)),
            rms_var=float(features.get("rms_var", 0.0)),
            zcr_mean=float(features.get("zcr_mean", 0.0)),
            zcr_var=float(features.get("zcr_var", 0.0)),
            spectral_centroid=float(features.get("spectral_centroid_mean", 0.0)),
            spectral_bandwidth=float(features.get("spectral_bandwidth_mean", 0.0)),
            spectral_rolloff=float(features.get("spectral_rolloff_mean", 0.0)),
            spectral_contrast=float(features.get("spectral_contrast_mean", 0.0)),
            mfcc_vector=features.get("mfcc_means", []),
            chroma_vector=features.get("chroma_means", []),
            onset_rate=float(features.get("onset_rate", 0.0)),
            onset_strength_mean=float(features.get("onset_strength_mean", 0.0)),
            detected_key=features.get("detected_key"),
            time_signature=features.get("time_signature", "4/4"),
            chord_progression=features.get("progression_str"),
            vocal_presence=features.get("vocal_presence"),
        )
        db.session.add(audio_feat)

        # Lookup or create Genre reference
        pred_genre_name = classification.get("predicted_genre", "Unknown")
        genre_record = Genre.query.filter(
            (Genre.name.ilike(pred_genre_name)) | (Genre.slug.ilike(pred_genre_name.lower().replace(" ", "-")))
        ).first()

        if not genre_record:
            genre_record = Genre(
                name=pred_genre_name,
                slug=pred_genre_name.lower().replace(" ", "-"),
                description="Genre identified by ML model.",
            )
            db.session.add(genre_record)
            db.session.flush()

        # Lookup Subgenre reference if available
        subgenre_record = None
        sub_slug = classification.get("subgenre_slug")
        sub_name = classification.get("subgenre_name")
        if sub_slug or sub_name:
            sub_query = Subgenre.query.filter_by(genre_id=genre_record.id)
            if sub_slug:
                subgenre_record = sub_query.filter(Subgenre.slug.ilike(sub_slug)).first()
            if not subgenre_record and sub_name:
                subgenre_record = sub_query.filter(Subgenre.name.ilike(sub_name)).first()

        # Create Prediction record with subgenre_id
        prediction = Prediction(
            analysis_id=analysis.id,
            genre_id=genre_record.id,
            subgenre_id=subgenre_record.id if subgenre_record else None,
            confidence=float(classification.get("confidence", 0.0)),
            probabilities=classification.get("probabilities", {}),
            alternatives=classification.get("alternatives", []),
            explanation=classification.get("explanation", ""),
            is_low_confidence=bool(classification.get("is_low_confidence", False)),
        )
        db.session.add(prediction)
        db.session.commit()

    except Exception:
        db.session.rollback()
        # Fall back gracefully so analysis response can still render even if DB fails
        pass

    return {
        "features": features,
        "classification": classification,
        "validation": validation_info,
        "analysis_id": analysis_id,
        "song_id": song_id,
    }


def cleanup_orphaned_uploads(upload_folder: str = None, max_age_days: int = 30) -> dict:
    """Scans the upload folder and deletes orphaned audio files older than max_age_days.

    A file is considered orphaned if:
    1. It is not referenced in the songs table by stored_filename, AND
    2. Its age exceeds max_age_days.
    """
    import os
    import time
    from flask import current_app

    if upload_folder is None:
        upload_folder = current_app.config.get("UPLOAD_FOLDER", "uploads")

    if not os.path.exists(upload_folder):
        return {"deleted_count": 0, "freed_bytes": 0}

    # Query all active stored filenames from database
    try:
        active_filenames = {s[0] for s in db.session.query(Song.stored_filename).all()}
    except Exception:
        active_filenames = set()

    now = time.time()
    cutoff_seconds = max_age_days * 86400
    deleted_count = 0
    freed_bytes = 0

    for fname in os.listdir(upload_folder):
        fpath = os.path.join(upload_folder, fname)
        if not os.path.isfile(fpath):
            continue

        # If file is not in active database records
        if fname not in active_filenames:
            try:
                mtime = os.path.getmtime(fpath)
                if (now - mtime) >= cutoff_seconds:
                    size = os.path.getsize(fpath)
                    os.remove(fpath)
                    deleted_count += 1
                    freed_bytes += size
            except OSError:
                pass

    return {
        "deleted_count": deleted_count,
        "freed_bytes": freed_bytes,
    }

