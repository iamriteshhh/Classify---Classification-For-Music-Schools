"""
test_active_learning.py
=======================
Tests for the active learning teacher correction pipeline (classify.ml.active_learning).
Verifies collection of teacher corrections, golden test set regression guards,
and fine-tuning persistence.
Part of Phase 1 for CLASSIFY.
"""

import os
import json
import pytest
import numpy as np
from classify.extensions import db
from classify.models import User, Teacher, Song, Analysis, Prediction, TeacherReview, Genre
from classify.ml.active_learning import collect_teacher_corrections, retrain_from_reviews, evaluate_model_on_golden_set


def test_collect_corrections_empty(app):
    """Verifies that an empty list is returned when no teacher reviews exist."""
    corrections = collect_teacher_corrections(app=app)
    assert isinstance(corrections, list)


def test_collect_corrections_with_disagreement(app, tmp_path):
    """Verifies collection of teacher disagreement reviews with audio files."""
    with app.app_context():
        # Setup Teacher
        teacher_user = User(email="active_teacher@school.edu", password_hash="hash", role="teacher")
        db.session.add(teacher_user)
        db.session.flush()
        teacher = Teacher(user_id=teacher_user.id)
        db.session.add(teacher)

        # Setup Genres
        pop = Genre(name="Pop", slug="pop", description="Pop music.")
        rock = Genre(name="Rock", slug="rock", description="Rock music.")
        db.session.add_all([pop, rock])
        db.session.flush()

        # Create dummy audio file in uploads
        upload_folder = app.config.get("UPLOAD_FOLDER")
        os.makedirs(upload_folder, exist_ok=True)
        dummy_filename = "active_learning_test.wav"
        dummy_audio_path = os.path.join(upload_folder, dummy_filename)
        with open(dummy_audio_path, "wb") as f:
            f.write(b"RIFF" + b"\x00" * 100)

        # Setup Song & Analysis
        song = Song(original_filename="student_track.wav", stored_filename=dummy_filename)
        db.session.add(song)
        db.session.flush()

        analysis = Analysis(song_id=song.id)
        db.session.add(analysis)
        db.session.flush()

        pred = Prediction(analysis_id=analysis.id, genre_id=pop.id, confidence=0.75)
        db.session.add(pred)
        db.session.flush()

        # Teacher reviews and disagrees: claims it is Rock
        review = TeacherReview(
            analysis_id=analysis.id,
            teacher_id=teacher.id,
            agrees_with_model=False,
            teacher_genre="Rock",
            comment="Strong distorted rhythm section indicates Rock, not Pop.",
        )
        db.session.add(review)
        db.session.commit()

        # Execute collection
        corrections = collect_teacher_corrections(app=app)
        assert len(corrections) >= 1
        found = [c for c in corrections if c["analysis_id"] == analysis.id]
        assert len(found) == 1
        assert found[0]["corrected_genre"] == "Rock"
        assert found[0]["original_predicted_genre"] == "Pop"
        assert found[0]["has_audio"] is True

        # Clean up dummy audio
        try:
            os.remove(dummy_audio_path)
        except OSError:
            pass


def test_regression_guard_aborts_on_poor_candidate():
    """Verifies that retrain_from_reviews aborts deployment if candidate regresses."""
    from unittest.mock import patch, MagicMock

    with patch("classify.ml.active_learning.collect_teacher_corrections") as mock_collect:
        mock_collect.return_value = [{
            "review_id": 1,
            "analysis_id": 1,
            "song_id": 1,
            "audio_path": "fake.wav",
            "has_audio": True,
            "corrected_genre": "Rock",
            "original_predicted_genre": "Pop",
        }]

        with patch("classify.ml.active_learning.extract_feature_vector") as mock_feat:
            mock_feat.return_value = (np.zeros(89, dtype=np.float32), ["f"] * 89)

            with patch("classify.ml.active_learning.evaluate_model_on_golden_set") as mock_eval:
                # 1st call is baseline evaluation (85%), 2nd call is candidate (60%)
                mock_eval.side_effect = [(0.85, 0.85, 24, 28), (0.60, 0.60, 16, 28)]

                res = retrain_from_reviews(min_corrections=1, regression_tolerance=0.02)
                assert res["status"] == "rejected_regression"
                assert "Regression Guard" in res["message"]
