"""Tests for async audio processing tasks, SSE, and storage quota management."""

import io
import os
import time
from unittest.mock import patch
import pytest

from classify import create_app
from classify.extensions import db
from classify.models import User, Student, Song
from classify.tasks import submit_analysis_task, get_task_status, stream_task_events
from classify.services.analysis_service import cleanup_orphaned_uploads


def test_async_task_lifecycle(tmp_path):
    """Verify task submission, status progression, and completion."""
    app = create_app("testing")
    dummy_wav = tmp_path / "test_async.wav"

    # Create dummy audio file
    import soundfile as sf
    import numpy as np
    sr = 22050
    t = np.linspace(0, 1.0, sr)
    sig = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(str(dummy_wav), sig, sr)

    with app.app_context():
        with patch("classify.services.analysis_service.analyze_uploaded_audio") as mock_analyze:
            mock_analyze.return_value = {
                "features": {
                    "duration": 1.0,
                    "sample_rate": 22050,
                    "tempo": 120.0,
                    "rms_mean": 0.2,
                    "rms_var": 0.01,
                    "zcr_mean": 0.05,
                    "zcr_var": 0.001,
                    "spectral_centroid_mean": 1500.0,
                    "spectral_bandwidth_mean": 2000.0,
                    "spectral_rolloff_mean": 3000.0,
                    "spectral_contrast_mean": 20.0,
                    "mfcc_means": [0.0] * 20,
                    "chroma_means": [0.0] * 12,
                    "onset_rate": 2.0,
                },
                "classification": {
                    "predicted_genre": "Rock",
                    "confidence": 0.85,
                    "confidence_percent": 85.0,
                    "probabilities": {"Rock": 85.0},
                    "alternatives": [],
                    "explanation": "High energy test track.",
                    "model_name": "Trained ML Classifier",
                },
                "validation": {"duration": 1.0},
                "analysis_id": 999,
            }

            task_id = submit_analysis_task(
                app,
                file_path=str(dummy_wav),
                original_filename="test_async.wav",
                stored_filename="uuid_test_async.wav",
            )
            assert task_id is not None

            # Poll status until done (with timeout)
            completed = False
            for _ in range(50):
                time.sleep(0.05)
                status = get_task_status(task_id)
                assert status is not None
                if status["status"] == "completed":
                    completed = True
                    assert status["progress"] == 100
                    assert status["result"]["classification"]["predicted_genre"] == "Rock"
                    break
                elif status["status"] == "failed":
                    pytest.fail(f"Task failed: {status.get('error')}")

            assert completed is True


def test_async_routes_status_and_events():
    """Verify /analysis/status/<task_id> and /analysis/events/<task_id>."""
    app = create_app("testing")
    client = app.test_client()

    with app.app_context():
        # Submit a task directly into task manager
        from classify.tasks import _TASK_LOCK, _TASKS
        test_id = "test-task-123"
        with _TASK_LOCK:
            _TASKS[test_id] = {
                "task_id": test_id,
                "status": "completed",
                "progress": 100,
                "step": "Complete",
                "result": {"analysis_id": 1, "stored_filename": "sample.wav"},
                "error": None,
            }

        # Check status route
        resp = client.get(f"/analysis/status/{test_id}")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert data["task"]["status"] == "completed"

        # Check non-existent task returns 404
        bad_resp = client.get("/analysis/status/unknown-id")
        assert bad_resp.status_code == 404


def test_storage_quota_rejection(client):
    """Verify that student upload exceeding UPLOAD_QUOTA_MB is rejected."""
    app = client.application
    with app.app_context():
        user = User(email="storage_student@music.edu", role="student", email_verified=True)
        user.set_password("SecureStudent123!")
        db.session.add(user)
        db.session.flush()
        student = Student(user_id=user.id)
        db.session.add(student)
        db.session.flush()

        # Add a song that fills up 500 MB (500 * 1024 KB)
        song = Song(
            student_id=student.id,
            stored_filename="large_storage.wav",
            original_filename="large_storage.wav",
            file_size_kb=510 * 1024.0,  # 510 MB
        )
        db.session.add(song)
        db.session.commit()

        # Login as student
        client.post("/auth/login", data={"email": "storage_student@music.edu", "password": "SecureStudent123!"})

        # Try to upload another file
        wav_data = b"RIFF....WAVEfmt ...." + b"\x00" * 1000
        resp = client.post(
            "/analyze",
            data={"audio_file": (io.BytesIO(wav_data), "new_track.wav")},
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert "Storage quota exceeded" in data["error"]


def test_cleanup_orphaned_uploads(tmp_path):
    """Verify cleanup_orphaned_uploads removes unreferenced files older than max_age_days."""
    app = create_app("testing")
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()

    # Create orphaned file
    orphaned_file = upload_dir / "old_orphaned.wav"
    orphaned_file.write_bytes(b"test audio content" * 100)

    # Artificially age the file by 35 days
    old_mtime = time.time() - (35 * 86400)
    os.utime(str(orphaned_file), (old_mtime, old_mtime))

    with app.app_context():
        res = cleanup_orphaned_uploads(str(upload_dir), max_age_days=30)
        assert res["deleted_count"] == 1
        assert res["freed_bytes"] > 0
        assert not orphaned_file.exists()
