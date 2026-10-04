"""
test_models.py
==============
Tests for SQLAlchemy models and database relationships.
Verifies:
- User creation with 'student' and 'teacher' roles
- 1:1 Student and Teacher profile relationships with User
- Genre and Subgenre hierarchy and cascading deletes
- Song, Analysis, AudioFeatures, and Prediction 1:1 entity chain
- TeacherReview independent review record persistence
Part of Phase 4 for CLASSIFY.
"""

import pytest
from classify.extensions import db
from classify.models import (
    User,
    Student,
    Teacher,
    Genre,
    Subgenre,
    Song,
    Analysis,
    AudioFeatures,
    Prediction,
    TeacherReview,
)


def test_user_and_role_profiles(app):
    """Test User creation with Student and Teacher profile relationships."""
    with app.app_context():
        # Create student user
        student_user = User(
            email="student@school.edu",
            password_hash="hashed_pw_student",
            role="student",
        )
        db.session.add(student_user)
        db.session.flush()

        student_profile = Student(user_id=student_user.id)
        db.session.add(student_profile)

        # Create teacher user
        teacher_user = User(
            email="teacher@school.edu",
            password_hash="hashed_pw_teacher",
            role="teacher",
        )
        db.session.add(teacher_user)
        db.session.flush()

        teacher_profile = Teacher(user_id=teacher_user.id)
        db.session.add(teacher_profile)
        db.session.commit()

        # Query and assert relationships
        queried_student = User.query.filter_by(email="student@school.edu").first()
        assert queried_student.is_student() is True
        assert queried_student.is_teacher() is False
        assert queried_student.student_profile is not None
        assert queried_student.student_profile.user_id == queried_student.id

        queried_teacher = User.query.filter_by(email="teacher@school.edu").first()
        assert queried_teacher.is_teacher() is True
        assert queried_teacher.teacher_profile is not None


def test_genre_subgenre_hierarchy(app):
    """Test Genre and Subgenre relationships and cascade deletion."""
    with app.app_context():
        rock = Genre(
            name="TestRock",
            slug="test-rock",
            description="Loud dynamic guitar music.",
            history="Started in the 1950s.",
        )
        db.session.add(rock)
        db.session.flush()

        sub1 = Subgenre(genre_id=rock.id, name="TestHardRock", slug="test-hard-rock")
        sub2 = Subgenre(genre_id=rock.id, name="TestClassicRock", slug="test-classic-rock")
        db.session.add_all([sub1, sub2])
        db.session.commit()

        # Verify child relationship
        saved_rock = Genre.query.filter_by(slug="test-rock").first()
        assert len(saved_rock.subgenres) == 2
        sub_names = {s.name for s in saved_rock.subgenres}
        assert "TestHardRock" in sub_names

        # Cascade delete test
        db.session.delete(saved_rock)
        db.session.commit()
        assert Subgenre.query.filter_by(slug="test-hard-rock").first() is None


def test_song_analysis_prediction_chain(app):
    """Test complete Analysis pipeline persistence chain in the database."""
    with app.app_context():
        # Setup Genre
        jazz = Genre(
            name="TestJazz",
            slug="test-jazz",
            description="Improvisational swing music.",
        )
        db.session.add(jazz)
        db.session.flush()

        # Setup Song
        song = Song(
            stored_filename="uuid_test_jazz.wav",
            original_filename="jazz_take1.wav",
            duration=45.2,
            file_size_kb=820.5,
        )
        db.session.add(song)
        db.session.flush()

        # Setup Analysis
        analysis = Analysis(song_id=song.id)
        db.session.add(analysis)
        db.session.flush()

        # Setup AudioFeatures
        features = AudioFeatures(
            analysis_id=analysis.id,
            duration=45.2,
            sample_rate=22050,
            channels=2,
            tempo=96.0,
            rms_mean=0.08,
            rms_var=0.001,
            zcr_mean=0.045,
            zcr_var=0.0002,
            spectral_centroid=1800.0,
            spectral_bandwidth=1900.0,
            spectral_rolloff=3800.0,
            spectral_contrast=20.5,
            mfcc_vector=[-30.0, 15.0, 5.0],
            chroma_vector=[0.1, 0.2, 0.8],
            onset_rate=2.4,
            onset_strength_mean=0.9,
        )
        db.session.add(features)

        # Setup Prediction
        prediction = Prediction(
            analysis_id=analysis.id,
            genre_id=jazz.id,
            confidence=0.912,
            probabilities={"Jazz": 0.912, "Blues": 0.065},
            alternatives=[{"genre": "Blues", "confidence": 0.065}],
            explanation="Swing tempo and warm spectral centroid indicative of Jazz.",
            is_low_confidence=False,
        )
        db.session.add(prediction)
        db.session.commit()

        # Query and assert relationships
        saved_analysis = Analysis.query.filter_by(song_id=song.id).first()
        assert saved_analysis is not None
        assert saved_analysis.song.original_filename == "jazz_take1.wav"
        assert saved_analysis.audio_features.tempo == 96.0
        assert saved_analysis.prediction.confidence == pytest.approx(0.912)
        assert saved_analysis.prediction.genre.name == "TestJazz"


def test_teacher_review_disagreement_persistence(app):
    """Test that a Teacher can record a review that disagrees with the model without altering the prediction."""
    with app.app_context():
        # Setup teacher user & profile
        teacher_user = User(
            email="prof@school.edu", password_hash="pw123", role="teacher"
        )
        db.session.add(teacher_user)
        db.session.flush()
        teacher = Teacher(user_id=teacher_user.id)
        db.session.add(teacher)
        db.session.flush()

        # Setup Genre & Song & Analysis & Prediction
        genre = Genre(name="TestPop", slug="test-pop", description="Pop genre")
        db.session.add(genre)
        db.session.flush()

        song = Song(stored_filename="song_123.wav", original_filename="pop.wav")
        db.session.add(song)
        db.session.flush()

        analysis = Analysis(song_id=song.id)
        db.session.add(analysis)
        db.session.flush()

        prediction = Prediction(
            analysis_id=analysis.id,
            genre_id=genre.id,
            confidence=0.65,
            explanation="Upbeat tempo",
        )
        db.session.add(prediction)
        db.session.flush()

        # Teacher reviews and disagrees: claims it is actually Rock
        review = TeacherReview(
            analysis_id=analysis.id,
            teacher_id=teacher.id,
            agrees_with_model=False,
            teacher_genre="Rock",
            comment="The electric guitar arrangement is closer to Classic Rock.",
        )
        db.session.add(review)
        db.session.commit()

        # Verify Prediction is unchanged while Review preserves disagreement
        saved_analysis = db.session.get(Analysis, analysis.id)
        assert saved_analysis.prediction.genre.name == "TestPop"
        assert saved_analysis.teacher_review is not None
        assert saved_analysis.teacher_review.agrees_with_model is False
        assert saved_analysis.teacher_review.teacher_genre == "Rock"


def test_prediction_subgenre_relationship(app):
    """Test Prediction subgenre_id foreign key and relationship to Subgenre model."""
    with app.app_context():
        rock = Genre(name="RockModelTest", slug="rock-model-test", description="Rock music.")
        db.session.add(rock)
        db.session.flush()

        sub = Subgenre(genre_id=rock.id, name="Hard Rock Test", slug="hard-rock-test")
        db.session.add(sub)
        db.session.flush()

        song = Song(original_filename="test.wav", stored_filename="stored_test.wav")
        db.session.add(song)
        db.session.flush()

        analysis = Analysis(song_id=song.id)
        db.session.add(analysis)
        db.session.flush()

        pred = Prediction(
            analysis_id=analysis.id,
            genre_id=rock.id,
            subgenre_id=sub.id,
            confidence=0.88,
        )
        db.session.add(pred)
        db.session.commit()

        queried_pred = Prediction.query.filter_by(analysis_id=analysis.id).first()
        assert queried_pred.subgenre is not None
        assert queried_pred.subgenre.name == "Hard Rock Test"
        assert queried_pred.subgenre.genre_id == rock.id

