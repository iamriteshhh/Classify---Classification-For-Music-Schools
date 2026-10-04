"""
passwords.py
============
Security utilities for NIST 800-63B compliant password policy,
breached-password checks (Have I Been Pwned k-anonymity API),
and Argon2id password hashing with transparent legacy hash upgrade.
Part of Security Hardening Specification (security.md §2.4, §2.5).
"""

import hashlib
import logging
import requests
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, VerificationError, InvalidHash
from werkzeug.security import check_password_hash as werkzeug_check_password_hash

logger = logging.getLogger(__name__)

# Primary Argon2id password hasher
_hasher = PasswordHasher(
    time_cost=3,
    memory_cost=65536,  # 64 MiB
    parallelism=4,
    hash_len=32,
    salt_len=16,
)

MIN_PASSWORD_LENGTH = 10
MAX_PASSWORD_LENGTH = 128


def hash_password(password: str) -> str:
    """Hashes a password using Argon2id."""
    return _hasher.hash(password)


def verify_and_check_upgrade(stored_hash: str, password: str) -> tuple[bool, bool]:
    """
    Verifies a password against the stored hash.

    Returns:
        tuple[bool, bool]: (is_valid, needs_upgrade)
        If valid and stored using legacy Werkzeug PBKDF2/scrypt, needs_upgrade is True.
    """
    if not stored_hash or not password:
        return False, False

    # Check if this is an Argon2 hash
    if stored_hash.startswith("$argon2"):
        try:
            _hasher.verify(stored_hash, password)
            needs_rehash = _hasher.check_needs_rehash(stored_hash)
            return True, needs_rehash
        except (VerifyMismatchError, VerificationError, InvalidHash):
            return False, False
        except Exception as e:
            logger.error("Unexpected error during Argon2 verification: %s", e)
            return False, False

    # Fallback to Werkzeug for existing PBKDF2/scrypt hashes
    try:
        if werkzeug_check_password_hash(stored_hash, password):
            # Valid password, but stored with legacy algorithm: needs upgrade to Argon2id
            return True, True
        return False, False
    except Exception as e:
        logger.error("Unexpected error during Werkzeug hash check: %s", e)
        return False, False


def check_haveibeenpwned(password: str, timeout: float = 1.5) -> tuple[bool, str | None]:
    """
    Checks if password appears in Have I Been Pwned database using k-anonymity range query.
    Only sends the first 5 characters of the SHA-1 hash (never the plaintext password).

    Returns:
        tuple[bool, str | None]: (is_pwned, warning_or_error_message)
        Soft-fails (returns False, None) if the service is unreachable or errors.
    """
    try:
        sha1_pwd = hashlib.sha1(password.encode("utf-8")).hexdigest().upper()
        prefix = sha1_pwd[:5]
        suffix = sha1_pwd[5:]

        url = f"https://api.pwnedpasswords.com/range/{prefix}"
        headers = {
            "User-Agent": "CLASSIFY-MusicSchool-App/1.0",
            "Add-Padding": "true",
        }
        resp = requests.get(url, headers=headers, timeout=timeout)

        if resp.status_code == 200:
            for line in resp.text.splitlines():
                parts = line.strip().split(":")
                if len(parts) == 2 and parts[0] == suffix:
                    count = int(parts[1])
                    if count > 0:
                        return True, f"This password has appeared in {count:,} known data breaches. Please choose a more unique password."
            return False, None
        else:
            logger.warning("HIBP API returned non-200 status: %d", resp.status_code)
            return False, None
    except Exception as e:
        # Soft-fail: Do not block users if HIBP API is unreachable or offline
        logger.warning("HIBP password check soft-failed: %s", e)
        return False, None


def validate_password_policy(password: str, check_breaches: bool = True) -> tuple[bool, str | None]:
    """
    Validates a password against NIST 800-63B standards:
    - Min length 10 characters
    - Max length 128 characters
    - Breached-password check (soft fail)
    """
    if not password:
        return False, "Password is required."

    if len(password) < MIN_PASSWORD_LENGTH:
        return False, f"Password must be at least {MIN_PASSWORD_LENGTH} characters long."

    if len(password) > MAX_PASSWORD_LENGTH:
        return False, f"Password must not exceed {MAX_PASSWORD_LENGTH} characters."

    from flask import current_app, has_app_context
    if has_app_context() and current_app and current_app.config.get("TESTING"):
        check_breaches = False

    if check_breaches:
        is_pwned, breach_msg = check_haveibeenpwned(password)
        if is_pwned:
            return False, breach_msg

    return True, None
