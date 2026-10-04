"""
test_devops_and_api_phase5.py
=============================
Tests for Phase 5 DevOps, Infrastructure, and LMS REST API:
- /healthz and /ready endpoints
- REST API Token Authentication (/api/v1/auth/token)
- REST API Genres Taxonomy (/api/v1/genres)
- REST API OpenAPI Specification (/api/v1/openapi.json)
- REST API Audio Analysis & History with Bearer token authentication
"""

import io
import json
import pytest
import soundfile as sf
import numpy as np
from classify.extensions import db
from classify.models import User, Student, Genre


def test_healthz_endpoint(client):
    """Test /healthz liveness probe returns HTTP 200."""
    res = client.get("/healthz")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "ok"
    assert data["service"] == "classify"


def test_ready_endpoint(client):
    """Test /ready readiness probe returns HTTP 200 when DB and model are ready."""
    res = client.get("/ready")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "ready"
    assert data["database"] == "connected"
    assert data["model"] == "loaded"


def test_api_token_auth_flow(app, client):
    """Test obtaining and using a Bearer token via /api/v1/auth/token."""
    with app.app_context():
        user = User(email="lms_student@conservatory.edu", role="student", email_verified=True)
        user.set_password("LmsSecret2026!")
        db.session.add(user)
        db.session.commit()

        student = Student(user_id=user.id)
        db.session.add(student)
        db.session.commit()

    # 1. Invalid credentials
    res_bad = client.post(
        "/api/v1/auth/token",
        json={"email": "lms_student@conservatory.edu", "password": "WrongPassword"},
    )
    assert res_bad.status_code == 401
    assert res_bad.get_json()["success"] is False

    # 2. Valid credentials
    res_good = client.post(
        "/api/v1/auth/token",
        json={"email": "lms_student@conservatory.edu", "password": "LmsSecret2026!"},
    )
    assert res_good.status_code == 200
    data = res_good.get_json()
    assert data["success"] is True
    assert "access_token" in data
    assert data["token_type"] == "Bearer"

    token = data["access_token"]

    # 3. Test accessing protected endpoint without token
    res_unauth = client.get("/api/v1/analyses")
    assert res_unauth.status_code == 401

    # 4. Test accessing protected endpoint with valid Bearer token
    headers = {"Authorization": f"Bearer {token}"}
    res_auth = client.get("/api/v1/analyses", headers=headers)
    assert res_auth.status_code == 200
    list_data = res_auth.get_json()
    assert list_data["success"] is True
    assert "analyses" in list_data


def test_api_genres_and_openapi(client):
    """Test /api/v1/genres and /api/v1/openapi.json."""
    # Genres endpoint
    res_genres = client.get("/api/v1/genres")
    assert res_genres.status_code == 200
    g_data = res_genres.get_json()
    assert g_data["success"] is True
    assert isinstance(g_data["genres"], list)

    # OpenAPI spec endpoint
    res_spec = client.get("/api/v1/openapi.json")
    assert res_spec.status_code == 200
    spec = res_spec.get_json()
    assert spec["openapi"] == "3.0.3"
    assert "/audio/analyze" in spec["paths"]
    assert "/auth/token" in spec["paths"]


def test_api_audio_analyze_endpoint(app, client):
    """Test uploading an audio file via REST API /api/v1/audio/analyze."""
    with app.app_context():
        user = User(email="api_uploader@conservatory.edu", role="student", email_verified=True)
        user.set_password("UploaderPass123!")
        db.session.add(user)
        db.session.commit()

        student = Student(user_id=user.id)
        db.session.add(student)
        db.session.commit()

        genre = Genre.query.first()
        if not genre:
            genre = Genre(name="Classical", slug="classical", description="Art music")
            db.session.add(genre)
            db.session.commit()

    # Get token
    res_token = client.post(
        "/api/v1/auth/token",
        json={"email": "api_uploader@conservatory.edu", "password": "UploaderPass123!"},
    )
    token = res_token.get_json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Generate synthetic WAV
    sr = 22050
    t = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False)
    wav_signal = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    wav_buf = io.BytesIO()
    sf.write(wav_buf, wav_signal, sr, format="WAV")
    wav_buf.seek(0)

    # Upload
    res_upload = client.post(
        "/api/v1/audio/analyze",
        headers=headers,
        data={"audio_file": (wav_buf, "api_etude.wav")},
        content_type="multipart/form-data",
    )
    assert res_upload.status_code == 201
    res_data = res_upload.get_json()
    assert res_data["success"] is True
    assert "analysis_id" in res_data
    assert "prediction" in res_data
    assert "acoustic_features" in res_data
    assert "detected_key" in res_data["acoustic_features"]
    assert "time_signature" in res_data["acoustic_features"]

    # Verify detail retrieval
    analysis_id = res_data["analysis_id"]
    res_detail = client.get(f"/api/v1/analyses/{analysis_id}", headers=headers)
    assert res_detail.status_code == 200
    detail_data = res_detail.get_json()
    assert detail_data["success"] is True
    assert detail_data["id"] == analysis_id
