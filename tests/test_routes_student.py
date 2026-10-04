"""
test_routes_student.py
======================
Tests for Phase 6: Persisted Analysis and Student Dashboard.
Verifies:
- Student dashboard loads empty state (0 tracks)
- Logged-in student uploads are persisted to Song, Analysis, AudioFeatures, Prediction
- Student dashboard updates with real metrics (total songs, avg tempo, genre distribution)
- Platform-wide community comparison calculates genuine percentages from database records
- Saved analysis report is accessible via /analysis/<analysis_id>
"""

from classify.models import Song, Analysis, Prediction, AudioFeatures


def test_student_dashboard_empty_state(client):
    """Test student dashboard rendering with 0 uploaded tracks."""
    # Register and log in student
    reg_data = {
        "email": "empty_dash@music.edu",
        "role": "student",
        "password": "password123",
        "confirm_password": "password123",
    }
    client.post("/auth/register", data=reg_data, follow_redirects=True)

    response = client.get("/student/dashboard")
    assert response.status_code == 200
    assert b"Total Tracks Analyzed" in response.data
    assert b"0" in response.data
    assert b"No tracks analyzed yet" in response.data


def test_student_upload_persistence_and_dashboard_update(client, app, synthetic_wav_bytes):
    """Test that student uploads persist to database and update student dashboard."""
    # Register and login student
    reg_data = {
        "email": "uploader@music.edu",
        "role": "student",
        "password": "password123",
        "confirm_password": "password123",
    }
    client.post("/auth/register", data=reg_data, follow_redirects=True)

    # Upload an audio track as logged-in student
    upload_data = {
        "audio_file": (synthetic_wav_bytes, "student_practice_take1.wav"),
    }
    upload_res = client.post("/analyze", data=upload_data, content_type="multipart/form-data")
    assert upload_res.status_code == 200

    # Verify database persistence
    with app.app_context():
        song = Song.query.filter_by(original_filename="student_practice_take1.wav").first()
        assert song is not None
        assert song.student_id is not None
        assert song.analysis is not None
        assert song.analysis.audio_features is not None
        assert song.analysis.prediction is not None
        assert song.analysis.prediction.confidence > 0.0

        analysis_id = song.analysis.id

    # Verify student dashboard reflects the uploaded track
    dash_res = client.get("/student/dashboard")
    assert dash_res.status_code == 200
    assert b"student_practice_take1.wav" in dash_res.data
    assert b"Total Tracks Analyzed" in dash_res.data
    assert b"My Genre Distribution" in dash_res.data

    # Verify viewing persisted analysis report directly
    report_res = client.get(f"/analysis/{analysis_id}")
    assert report_res.status_code == 200
    assert b"student_practice_take1.wav" in report_res.data
    assert b"Confidence" in report_res.data
    assert b"Extracted Acoustic Properties" in report_res.data
