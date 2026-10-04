"""
test_security_hardening.py
==========================
Comprehensive tests for application-wide security hardening:
- NIST 800-63B password validation
- Argon2id password hashing and transparent PBKDF2 migration
- Account lockout after 5 consecutive failed login attempts
- Safe redirect validation preventing open redirect vulnerabilities
- Tokenized email verification and password reset flows
- Session invalidation on password reset
- IDOR access controls on student analysis records
- Security headers enforcement
"""

import pytest
from datetime import datetime, timezone, timedelta
from werkzeug.security import generate_password_hash
from classify.extensions import db
from classify.models import User, Student, Song, Analysis, Prediction, Genre
from classify.security.passwords import validate_password_policy, hash_password, verify_and_check_upgrade
from classify.security.tokens import (
    generate_verification_token,
    confirm_verification_token,
    generate_reset_token,
    confirm_reset_token
)
from classify.auth_utils import is_safe_redirect


def test_nist_password_length_requirements(app):
    """Test NIST 800-63B length constraints (10-128 characters)."""
    with app.app_context():
        # Too short (< 10 chars)
        valid, msg = validate_password_policy("Short1!", check_breaches=False)
        assert not valid
        assert "at least 10" in msg

        # Valid length (>= 10 chars)
        valid, msg = validate_password_policy("ValidPassword2026!", check_breaches=False)
        assert valid
        assert msg is None

        # Too long (> 128 chars)
        valid, msg = validate_password_policy("A" * 129, check_breaches=False)
        assert not valid
        assert "must not exceed 128" in msg


def test_argon2id_hashing_and_upgrade(app):
    """Test Argon2id hashing and transparent upgrade from legacy hashes."""
    with app.app_context():
        password = "SecureComplexPassword123"
        hashed = hash_password(password)
        assert hashed.startswith("$argon2id$")

        # Verify Argon2id
        matches, needs_upgrade = verify_and_check_upgrade(hashed, password)
        assert matches
        assert not needs_upgrade

        # Verify legacy PBKDF2 hash gets flagged for upgrade
        legacy_hash = generate_password_hash(password, method="pbkdf2:sha256")
        matches, needs_upgrade = verify_and_check_upgrade(legacy_hash, password)
        assert matches
        assert needs_upgrade


def test_account_lockout_after_failed_attempts(client, app):
    """Test that an account is locked out after 5 consecutive failed login attempts."""
    with app.app_context():
        # Create a test student user
        user = User(
            email="lockout_target@classify.test",
            role="student",
            email_verified=True
        )
        user.set_password("StrongPassword2026!")
        db.session.add(user)
        db.session.flush()
        student = Student(user_id=user.id)
        db.session.add(student)
        db.session.commit()

        # Send 5 incorrect password attempts
        for i in range(5):
            res = client.post("/auth/login", data={
                "email": "lockout_target@classify.test",
                "password": "WrongPassword999"
            }, follow_redirects=True)
            assert res.status_code == 200

        # Query user in DB
        db.session.refresh(user)
        assert user.failed_login_attempts == 5
        assert user.is_locked()
        assert user.locked_until is not None

        # 6th attempt (even with CORRECT password) should be denied due to lockout
        res_locked = client.post("/auth/login", data={
            "email": "lockout_target@classify.test",
            "password": "StrongPassword2026!"
        }, follow_redirects=True)
        assert b"Invalid email or password" in res_locked.data


def test_safe_redirect_validation(app):
    """Test open-redirect guard against malicious schemes, hostnames, and slashes."""
    with app.app_context():
        # Allowed relative paths
        assert is_safe_redirect("/")
        assert is_safe_redirect("/library")
        assert is_safe_redirect("/library/rock")
        assert is_safe_redirect("/explore?g1=rock&g2=pop")

        # Blocked absolute external URLs
        assert not is_safe_redirect("http://evil.com")
        assert not is_safe_redirect("https://evil.com")
        assert not is_safe_redirect("ftp://evil.com")

        # Blocked scheme-relative URLs (open-redirect vectors)
        assert not is_safe_redirect("//evil.com")
        assert not is_safe_redirect("/\\evil.com")
        assert not is_safe_redirect("///evil.com")
        assert not is_safe_redirect("javascript:alert(1)")


def test_tokenized_email_and_password_reset_flow(app):
    """Test generation and verification of time-limited tokens."""
    with app.app_context():
        user = User(
            email="token_user@classify.test",
            role="student"
        )
        user.set_password("TokenPassword2026!")
        db.session.add(user)
        db.session.commit()

        # Email verification token
        verify_token = generate_verification_token(user.email)
        verified_email = confirm_verification_token(verify_token)
        assert verified_email == user.email

        # Password reset token
        reset_token = generate_reset_token(user.email)
        reset_email = confirm_reset_token(reset_token)
        assert reset_email == user.email

        # Invalid token fails safely
        assert confirm_reset_token("invalid.token.here") is None


def test_session_invalidation_on_password_change(client, app):
    """Test that resetting password increments session_version, invalidating prior sessions."""
    with app.app_context():
        user = User(
            email="session_user@classify.test",
            role="student",
            email_verified=True
        )
        user.set_password("InitialPassword2026!")
        db.session.add(user)
        db.session.flush()
        student = Student(user_id=user.id)
        db.session.add(student)
        db.session.commit()
        user_id = user.id
        token = generate_reset_token(user.email)

    # Log in
    res = client.post("/auth/login", data={
        "email": "session_user@classify.test",
        "password": "InitialPassword2026!"
    }, follow_redirects=True)
    assert res.status_code == 200

    # Access student dashboard (should succeed)
    res_dash = client.get("/student/dashboard")
    assert res_dash.status_code == 200

    # Reset password using token
    res_reset = client.post(f"/auth/reset/{token}", data={
        "password": "NewStrongPassword2026!",
        "confirm_password": "NewStrongPassword2026!"
    }, follow_redirects=True)
    assert res_reset.status_code == 200

    # Verify session_version incremented in database
    with app.app_context():
        updated_user = db.session.get(User, user_id)
        assert updated_user.session_version == 2

    # Accessing dashboard with prior session should now redirect to login
    res_dash_after = client.get("/student/dashboard")
    assert res_dash_after.status_code in (302, 401)
    assert "/auth/login" in res_dash_after.headers.get("Location", "")


def test_security_headers_present(client):
    """Verify security headers are applied to HTTP responses."""
    res = client.get("/")
    assert res.headers.get("X-Content-Type-Options") == "nosniff"
    assert res.headers.get("X-Frame-Options") == "DENY"
    assert res.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"
    assert "Content-Security-Policy" in res.headers


def test_idor_prevention_on_student_analysis(client, app):
    """Verify that student A cannot view student B's private analysis."""
    with app.app_context():
        # Create Student A
        user_a = User(email="student_a@classify.test", role="student", email_verified=True)
        user_a.set_password("PasswordA2026!")
        db.session.add(user_a)
        db.session.flush()
        prof_a = Student(user_id=user_a.id)
        db.session.add(prof_a)

        # Create Student B
        user_b = User(email="student_b@classify.test", role="student", email_verified=True)
        user_b.set_password("PasswordB2026!")
        db.session.add(user_b)
        db.session.flush()
        prof_b = Student(user_id=user_b.id)
        db.session.add(prof_b)
        db.session.flush()

        # Create a song & analysis belonging to Student B
        song_b = Song(student_id=prof_b.id, original_filename="b_track.wav", stored_filename="b_track.wav")
        db.session.add(song_b)
        db.session.flush()
        analysis_b = Analysis(song_id=song_b.id)
        db.session.add(analysis_b)
        db.session.commit()
        analysis_b_id = analysis_b.id

    # Log in as Student A
    client.post("/auth/login", data={"email": "student_a@classify.test", "password": "PasswordA2026!"})

    # Student A tries to view Student B's analysis
    res = client.get(f"/analysis/{analysis_b_id}")
    # Must deny access (403 forbidden)
    assert res.status_code == 403
