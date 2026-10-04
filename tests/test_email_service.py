"""Tests for email delivery service and templates."""

from unittest.mock import patch
from classify import create_app
from classify.models import User
from classify.services.email_service import send_password_reset_email, send_verification_email


def test_send_password_reset_email():
    """Verify password reset email generates HTML and plain-text without errors."""
    app = create_app("testing")
    with app.app_context():
        user = User(email="student@music.edu", role="student")
        reset_url = "http://localhost:5000/auth/reset/dummy-token-12345"

        with patch("classify.services.email_service.mail.send") as mock_send:
            # When suppress send is False for testing
            app.config["MAIL_SUPPRESS_SEND"] = False
            success = send_password_reset_email(user, reset_url)
            assert success is True
            assert mock_send.called
            msg = mock_send.call_args[0][0]
            assert "student@music.edu" in msg.recipients
            assert "dummy-token-12345" in msg.body
            assert "dummy-token-12345" in msg.html


def test_send_verification_email():
    """Verify account verification email generates HTML and plain-text without errors."""
    app = create_app("testing")
    with app.app_context():
        user = User(email="newteacher@music.edu", role="teacher")
        verify_url = "http://localhost:5000/auth/verify/verify-token-67890"

        with patch("classify.services.email_service.mail.send") as mock_send:
            app.config["MAIL_SUPPRESS_SEND"] = False
            success = send_verification_email(user, verify_url)
            assert success is True
            assert mock_send.called
            msg = mock_send.call_args[0][0]
            assert "newteacher@music.edu" in msg.recipients
            assert "verify-token-67890" in msg.body
            assert "verify-token-67890" in msg.html


def test_suppressed_email_in_testing():
    """Verify that when MAIL_SUPPRESS_SEND is True, mail.send is not called and returns True."""
    app = create_app("testing")
    with app.app_context():
        user = User(email="student@music.edu", role="student")
        app.config["MAIL_SUPPRESS_SEND"] = True
        with patch("classify.services.email_service.mail.send") as mock_send:
            success = send_password_reset_email(user, "http://localhost:5000/auth/reset/token")
            assert success is True
            assert not mock_send.called
