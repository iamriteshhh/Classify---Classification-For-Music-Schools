"""
test_routes_teacher.py
======================
Tests for the teacher dashboard, review workflow, and pedagogical feedback.
Covers:
- Access control: unauthenticated redirected, students get 403, teachers get 200.
- Submissions list rendering and filters (status, genre, student).
- Review submission via POST /teacher/review/<id>.
- Model prediction immutability under teacher disagreement (Ground Rule 6).
- Student dashboard status update from 'Pending Review' to 'Reviewed'.
Part of Phase 7 for CLASSIFY.
"""

from classify.models import User, Student, Teacher, Song, Analysis, Prediction, Genre, TeacherReview, AudioFeatures
from classify.extensions import db


def _setup_teacher_and_student(client, app, synthetic_wav_bytes):
    """Helper to create a student with an analyzed upload, and a registered teacher."""
    with app.app_context():
        # Create student user & profile
        student_user = User(email="student_test@music.edu", role="student")
        student_user.set_password("password123")
        db.session.add(student_user)
        db.session.flush()
        student_profile = Student(user_id=student_user.id)
        db.session.add(student_profile)

        # Create teacher user & profile
        teacher_user = User(email="instructor_test@music.edu", role="teacher")
        teacher_user.set_password("password123")
        db.session.add(teacher_user)
        db.session.flush()
        teacher_profile = Teacher(user_id=teacher_user.id)
        db.session.add(teacher_profile)

        # Create rock genre
        rock = Genre.query.filter_by(name="Rock").first()
        if not rock:
            rock = Genre(name="Rock", slug="rock", description="Rock music")
            db.session.add(rock)
            db.session.flush()

        # Create a song, analysis, prediction
        song = Song(
            student_id=student_profile.id,
            stored_filename="test_song.wav",
            original_filename="student_riff.wav",
            duration=5.0,
            file_size_kb=200.0,
        )
        db.session.add(song)
        db.session.flush()

        analysis = Analysis(song_id=song.id)
        db.session.add(analysis)
        db.session.flush()

        af = AudioFeatures(
            analysis_id=analysis.id,
            duration=5.0,
            sample_rate=22050,
            channels=1,
            tempo=132.0,
            rms_mean=0.15,
            zcr_mean=0.08,
            spectral_centroid=2400.0,
            spectral_bandwidth=2100.0,
            spectral_rolloff=4800.0,
        )
        db.session.add(af)

        pred = Prediction(
            analysis_id=analysis.id,
            genre_id=rock.id,
            confidence=0.88,
            probabilities={"Rock": 0.88, "Metal": 0.12},
            alternatives=[{"genre": "Metal", "probability": 0.12}],
            explanation="Driven by strong backbeat accents.",
        )
        db.session.add(pred)
        db.session.commit()

        analysis_id = analysis.id

    return "student_test@music.edu", "instructor_test@music.edu", analysis_id


def test_teacher_route_access_control(client, app, synthetic_wav_bytes):
    """Test that only teachers can access teacher routes."""
    _, _, analysis_id = _setup_teacher_and_student(client, app, synthetic_wav_bytes)

    # 1. Unauthenticated -> redirect to login
    res = client.get("/teacher/dashboard")
    assert res.status_code == 302
    assert "/auth/login" in res.headers["Location"]

    # 2. Student logged in -> 403 Forbidden
    client.post(
        "/auth/login",
        data={"email": "student_test@music.edu", "password": "password123"},
        follow_redirects=True,
    )
    res_student = client.get("/teacher/dashboard")
    assert res_student.status_code == 403

    res_student_review = client.get(f"/teacher/review/{analysis_id}")
    assert res_student_review.status_code == 403

    client.get("/auth/logout", follow_redirects=True)

    # 3. Teacher logged in -> 200 OK
    client.post(
        "/auth/login",
        data={"email": "instructor_test@music.edu", "password": "password123"},
        follow_redirects=True,
    )
    res_teacher = client.get("/teacher/dashboard")
    assert res_teacher.status_code == 200
    assert b"Instructor Review Dashboard" in res_teacher.data
    assert b"student_riff.wav" in res_teacher.data


def test_teacher_dashboard_filters(client, app, synthetic_wav_bytes):
    """Test teacher dashboard filtering by status and email."""
    _, _, analysis_id = _setup_teacher_and_student(client, app, synthetic_wav_bytes)

    client.post(
        "/auth/login",
        data={"email": "instructor_test@music.edu", "password": "password123"},
        follow_redirects=True,
    )

    # Filter pending -> should find 1 submission
    res_pending = client.get("/teacher/dashboard?status=pending")
    assert res_pending.status_code == 200
    assert b"student_riff.wav" in res_pending.data

    # Filter reviewed -> should find 0 submissions initially
    res_reviewed = client.get("/teacher/dashboard?status=reviewed")
    assert res_reviewed.status_code == 200
    assert b"No submissions found" in res_reviewed.data

    # Filter by non-matching student email -> 0 results
    res_filter_email = client.get("/teacher/dashboard?student=nonexistent@music.edu")
    assert res_filter_email.status_code == 200
    assert b"No submissions found" in res_filter_email.data


def test_teacher_review_submission_and_prediction_immutability(client, app, synthetic_wav_bytes):
    """
    Test submitting a teacher review with disagreement.
    Verifies that:
    1. TeacherReview is created and linked.
    2. Model Prediction remains untouched (immutable).
    3. Status on student dashboard reflects 'Reviewed'.
    """
    _, _, analysis_id = _setup_teacher_and_student(client, app, synthetic_wav_bytes)

    # Log in as teacher
    client.post(
        "/auth/login",
        data={"email": "instructor_test@music.edu", "password": "password123"},
        follow_redirects=True,
    )

    # Inspect review page
    res_get = client.get(f"/teacher/review/{analysis_id}")
    assert res_get.status_code == 200
    assert b"Pedagogical Review" in res_get.data
    assert b"Instructor Evaluation Form" in res_get.data
    assert b"Suggested Instructional Focus" in res_get.data

    # Submit review disagreeing with model: model said Rock, teacher marks Blues
    review_data = {
        "agrees_with_model": "false",
        "teacher_genre": "Blues",
        "comment": "Focus on bending intonation and triplet rhythm on beat 3.",
    }
    res_post = client.post(f"/teacher/review/{analysis_id}", data=review_data, follow_redirects=True)
    assert res_post.status_code == 200
    assert b"Teacher review saved" in res_post.data

    # Verify DB records
    with app.app_context():
        analysis = db.session.get(Analysis, analysis_id)
        assert analysis.teacher_review is not None
        assert analysis.teacher_review.agrees_with_model is False
        assert analysis.teacher_review.teacher_genre == "Blues"
        assert "bending intonation" in analysis.teacher_review.comment

        # MODEL IMMUTABILITY: Ensure Prediction is unchanged
        assert analysis.prediction.genre.name == "Rock"
        assert analysis.prediction.confidence == 0.88

    # Verify student dashboard reflects "Reviewed"
    client.get("/auth/logout", follow_redirects=True)
    client.post(
        "/auth/login",
        data={"email": "student_test@music.edu", "password": "password123"},
        follow_redirects=True,
    )
    res_student_dash = client.get("/student/dashboard")
    assert res_student_dash.status_code == 200
    assert b"Reviewed" in res_student_dash.data
