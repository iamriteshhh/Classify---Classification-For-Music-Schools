"""
test_pedagogical_phase3.py
==========================
Comprehensive tests for Phase 3 Pedagogical UX:
- Musical key / mode detection (Krumhansl-Schmuckler)
- Classrooms and enrollment management
- Homework assignments and student submissions
- Notification generation and retrieval
- PDF report export and authorization guards
"""

import io
import pytest
import numpy as np
from classify.extensions import db
from classify.models import (
    User,
    Student,
    Teacher,
    Classroom,
    Enrollment,
    Assignment,
    Submission,
    Notification,
    Song,
    AudioFeatures,
    Prediction,
    Genre,
    Analysis,
)
from classify.audio.audio_processing import detect_musical_key
from classify.services.pdf_service import generate_analysis_pdf


def test_detect_musical_key_algorithm():
    """Verify Krumhansl-Schmuckler detects major and minor keys on synthetic chroma."""
    # Krumhansl-Kessler C Major profile
    c_maj_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    res_c = detect_musical_key(c_maj_profile)
    assert res_c["detected_key"] == "C Major"
    assert res_c["key_root"] == "C"
    assert res_c["key_mode"] == "Major"

    # Shift profile by 7 semitones (G Major)
    g_maj_profile = np.roll(c_maj_profile, 7)
    res_g = detect_musical_key(g_maj_profile)
    assert res_g["detected_key"] == "G Major"
    assert res_g["key_root"] == "G"

    # A Minor profile (A is index 9: Krumhansl A minor profile)
    k_min = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    a_min_rolled = np.roll(k_min, 9)
    res_a = detect_musical_key(a_min_rolled)
    assert res_a["detected_key"] == "A Minor"
    assert res_a["key_root"] == "A"
    assert res_a["key_mode"] == "Minor"

    # Zero/flat chroma fallback
    zero_chroma = np.zeros(12)
    assert detect_musical_key(zero_chroma)["detected_key"] == "Unknown"


def test_classroom_crud_and_enrollment(app, client):
    """Test teacher creating a classroom and enrolling a student."""
    with app.app_context():
        # Setup teacher user
        teacher_user = User(
            email="prof_phase3@conservatory.edu",
            role="teacher",
            email_verified=True,
        )
        teacher_user.set_password("SecureProf123!")
        db.session.add(teacher_user)
        db.session.commit()

        teacher = Teacher(user_id=teacher_user.id)
        db.session.add(teacher)

        # Setup student user
        student_user = User(
            email="student_phase3@conservatory.edu",
            role="student",
            email_verified=True,
        )
        student_user.set_password("SecureStudent123!")
        db.session.add(student_user)
        db.session.commit()

        student = Student(user_id=student_user.id)
        db.session.add(student)
        db.session.commit()

        # Login as teacher
        client.post("/auth/login", data={"email": "prof_phase3@conservatory.edu", "password": "SecureProf123!"})

        # Create classroom
        resp = client.post(
            "/teacher/classrooms",
            data={"name": "Acoustics & Orchestration", "semester": "Fall 2026"},
            follow_redirects=True,
        )
        assert resp.status_code == 200
        classroom = Classroom.query.filter_by(name="Acoustics & Orchestration").first()
        assert classroom is not None
        assert classroom.teacher_id == teacher.id

        # Enroll student
        resp_enroll = client.post(
            f"/teacher/classrooms/{classroom.id}/enroll",
            data={"email": "student_phase3@conservatory.edu"},
            follow_redirects=True,
        )
        assert resp_enroll.status_code == 200
        enrollment = Enrollment.query.filter_by(classroom_id=classroom.id, student_id=student.id).first()
        assert enrollment is not None

        # Create assignment
        resp_assign = client.post(
            f"/teacher/classrooms/{classroom.id}/assignments",
            data={
                "title": "Bebop Cadence Study",
                "description": "Upload a 12-bar blues in F with ii-V-I substitutions.",
                "genre_focus": "Jazz",
            },
            follow_redirects=True,
        )
        assert resp_assign.status_code == 200
        assignment = Assignment.query.filter_by(title="Bebop Cadence Study").first()
        assert assignment is not None
        assert assignment.classroom_id == classroom.id


def test_student_assignment_submission_and_notifications(app, client):
    """Test student submitting song for assignment and notification generation."""
    with app.app_context():
        # Setup teacher and student
        teacher_user = User(
            email="prof_notify@conservatory.edu",
            role="teacher",
            email_verified=True,
        )
        teacher_user.set_password("Teacher12345!")
        db.session.add(teacher_user)
        db.session.commit()

        teacher = Teacher(user_id=teacher_user.id)
        db.session.add(teacher)

        student_user = User(
            email="pupil_notify@conservatory.edu",
            role="student",
            email_verified=True,
        )
        student_user.set_password("Pupil12345!")
        db.session.add(student_user)
        db.session.commit()

        student = Student(user_id=student_user.id)
        db.session.add(student)
        db.session.commit()

        classroom = Classroom(name="Composition Lab", teacher_id=teacher.id)
        db.session.add(classroom)
        db.session.commit()

        enrollment = Enrollment(classroom_id=classroom.id, student_id=student.id)
        assignment = Assignment(classroom_id=classroom.id, title="Modal Etude")
        db.session.add_all([enrollment, assignment])

        # Create a song owned by student
        song = Song(
            stored_filename="dorian_sketch_stored.wav",
            original_filename="dorian_sketch.wav",
            student_id=student.id,
            duration=45.0,
            file_size_kb=800.0,
        )
        db.session.add(song)
        db.session.commit()

        # Login as student
        client.post("/auth/login", data={"email": "pupil_notify@conservatory.edu", "password": "Pupil12345!"})

        # Submit song
        resp = client.post(
            f"/student/assignments/{assignment.id}/submit",
            data={"song_id": song.id},
            follow_redirects=True,
        )
        assert resp.status_code == 200

        # Verify submission created
        submission = Submission.query.filter_by(assignment_id=assignment.id, student_id=student.id).first()
        assert submission is not None
        assert submission.song_id == song.id

        # Verify teacher got notification
        teacher_notif = Notification.query.filter_by(user_id=teacher_user.id).first()
        assert teacher_notif is not None
        assert "pupil_notify@conservatory.edu" in teacher_notif.message or "Modal Etude" in teacher_notif.message

        # Check student notifications page
        notif_resp = client.get("/student/notifications")
        assert notif_resp.status_code == 200


def test_pdf_report_generation_and_idor_guard(app, client):
    """Test generating PDF report and verifying IDOR authorization rules."""
    with app.app_context():
        # Setup users
        student_user_1 = User(
            email="owner_student@conservatory.edu",
            role="student",
            email_verified=True,
        )
        student_user_1.set_password("Pass12345!")

        student_user_2 = User(
            email="other_student@conservatory.edu",
            role="student",
            email_verified=True,
        )
        student_user_2.set_password("Pass12345!")

        db.session.add_all([student_user_1, student_user_2])
        db.session.commit()

        s1 = Student(user_id=student_user_1.id)
        s2 = Student(user_id=student_user_2.id)
        db.session.add_all([s1, s2])
        db.session.commit()

        genre = Genre.query.first()
        if not genre:
            genre = Genre(name="Classical", slug="classical", description="Art music")
            db.session.add(genre)
            db.session.commit()

        song = Song(
            stored_filename="string_quartet_stored.wav",
            original_filename="string_quartet.wav",
            student_id=s1.id,
            duration=180.5,
            file_size_kb=3500.0,
        )
        db.session.add(song)
        db.session.commit()

        analysis = Analysis(song_id=song.id)
        db.session.add(analysis)
        db.session.commit()

        af = AudioFeatures(
            analysis_id=analysis.id,
            duration=180.5,
            sample_rate=22050,
            tempo=124.0,
            rms_mean=0.15,
            zcr_mean=0.045,
            spectral_centroid=1800.0,
            spectral_bandwidth=2000.0,
            spectral_rolloff=3200.0,
            detected_key="F Major",
        )
        pred = Prediction(
            analysis_id=analysis.id,
            genre_id=genre.id,
            confidence=0.912,
        )
        db.session.add_all([af, pred])
        db.session.commit()

        # Test direct PDF service unit test
        pdf_buf = generate_analysis_pdf(analysis)
        assert isinstance(pdf_buf, io.BytesIO)
        pdf_bytes = pdf_buf.getvalue()
        assert pdf_bytes.startswith(b"%PDF")
        assert len(pdf_bytes) > 1000

        # Test authenticated download as owner
        client.post("/auth/login", data={"email": "owner_student@conservatory.edu", "password": "Pass12345!"})

        resp = client.get(f"/analysis/{analysis.id}/pdf")
        assert resp.status_code == 200
        assert resp.headers["Content-Type"] == "application/pdf"
        assert f"classify_report_{song.original_filename}.pdf" in resp.headers["Content-Disposition"]

        # Logout owner and login other student
        client.get("/auth/logout")
        client.post("/auth/login", data={"email": "other_student@conservatory.edu", "password": "Pass12345!"})

        resp_idor = client.get(f"/analysis/{analysis.id}/pdf")
        assert resp_idor.status_code == 403
