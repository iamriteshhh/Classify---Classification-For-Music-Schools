"""
tokens.py
=========
Time-limited, cryptographically signed tokens for Email Verification
and Password Reset flows using itsdangerous.
Part of Security Hardening Specification (security.md §3.3, §3.5).
"""

from flask import current_app
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadTimeSignature, BadSignature

EMAIL_VERIFY_SALT = "email-verification-salt"
PASSWORD_RESET_SALT = "password-reset-salt"


def _get_serializer() -> URLSafeTimedSerializer:
    secret_key = current_app.config.get("SECRET_KEY", "fallback-secret-key")
    return URLSafeTimedSerializer(secret_key)


def generate_verification_token(email: str) -> str:
    """Generates a signed, time-limited token for email verification."""
    s = _get_serializer()
    return s.dumps(email, salt=EMAIL_VERIFY_SALT)


def confirm_verification_token(token: str, expiration: int = 86400) -> str | None:
    """
    Confirms an email verification token within the expiration window (default: 24 hours).
    Returns email if valid, or None if expired/tampered.
    """
    s = _get_serializer()
    try:
        email = s.loads(token, salt=EMAIL_VERIFY_SALT, max_age=expiration)
        return email
    except (SignatureExpired, BadTimeSignature, BadSignature):
        return None


def generate_reset_token(email: str) -> str:
    """Generates a signed, time-limited token for password reset."""
    s = _get_serializer()
    return s.dumps(email, salt=PASSWORD_RESET_SALT)


def confirm_reset_token(token: str, expiration: int = 3600) -> str | None:
    """
    Confirms a password reset token within the expiration window (default: 1 hour).
    Returns email if valid, or None if expired/tampered.
    """
    s = _get_serializer()
    try:
        email = s.loads(token, salt=PASSWORD_RESET_SALT, max_age=expiration)
        return email
    except (SignatureExpired, BadTimeSignature, BadSignature):
        return None
