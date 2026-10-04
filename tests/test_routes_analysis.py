"""
test_routes_analysis.py
=======================
Tests for the web application routes in the analysis blueprint.
Evolved from test_app.py to use pytest fixtures and isolated test environment.
"""

import io
import os


def test_homepage_loads(client):
    """Test that the homepage loads successfully with status 200 and expected markup."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"CLASSIFY" in response.data
    assert b"Music Classification System for Music Schools" in response.data
    assert b"Upload Music File" in response.data


def test_audio_analysis_pipeline_success(client, synthetic_wav_bytes):
    """Test uploading a synthetic WAV audio file and verifying the analysis results."""
    data = {
        "audio_file": (synthetic_wav_bytes, "test_unit_audio.wav"),
    }
    response = client.post("/analyze", data=data, content_type="multipart/form-data")

    assert response.status_code == 200
    assert b"Audio Analysis &amp; Prototype Classification" in response.data
    assert b"Extracted Acoustic Properties" in response.data
    assert b"PROTOTYPE PREDICTION:" in response.data
    assert b"Confidence" in response.data
    assert b"Alternative Genre Probabilities" in response.data
    assert b"Why This Classification" in response.data
    assert b"Tempo" in response.data
    assert b"RMS Energy" in response.data
    assert b"Spectral Centroid" in response.data



def test_invalid_file_extension(client):
    """Test that uploading a non-audio file redirects and flashes an error."""
    text_file = io.BytesIO(b"This is not audio content.")
    data = {
        "audio_file": (text_file, "test.txt"),
    }
    response = client.post(
        "/analyze",
        data=data,
        content_type="multipart/form-data",
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert b"Invalid file format" in response.data


def test_no_file_provided(client):
    """Test submitting the analyze route without the audio_file field."""
    response = client.post("/analyze", data={}, follow_redirects=True)
    assert response.status_code == 200
    assert b"No audio file part was provided" in response.data


def test_empty_filename_selection(client):
    """Test submitting with an empty file selector."""
    empty_file = io.BytesIO(b"")
    data = {
        "audio_file": (empty_file, ""),
    }
    response = client.post(
        "/analyze",
        data=data,
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"Please select an audio file" in response.data


def test_uploaded_audio_playback(client, app):
    """Test that uploaded audio file can be retrieved via the /uploads/<filename> route."""
    # Write a test file into the app's upload directory
    upload_dir = app.config["UPLOAD_FOLDER"]
    test_filename = "playback_sample.wav"
    test_filepath = os.path.join(upload_dir, test_filename)
    with open(test_filepath, "wb") as f:
        f.write(b"RIFF dummy wav content")

    response = client.get(f"/uploads/{test_filename}")
    assert response.status_code == 200
    assert response.data == b"RIFF dummy wav content"


def test_all_blueprints_active_and_reachable(client):
    """Verify that all six core blueprint routes are registered, active, and properly gated."""
    assert client.get("/").status_code == 200
    assert client.get("/auth/login").status_code == 200
    assert client.get("/library").status_code == 200
    assert client.get("/explore").status_code == 302
    assert client.get("/student/dashboard").status_code == 302
    assert client.get("/teacher/dashboard").status_code == 302
