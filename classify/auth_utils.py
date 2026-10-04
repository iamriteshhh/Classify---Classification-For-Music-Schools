"""
auth_utils.py
=============
Authentication and authorization helpers for CLASSIFY.
Provides user loading for Flask-Login, session version invalidation,
is_safe_redirect URL sanitization, and @role_required / @teacher_required
access control decorators (security.md §2.6, §3.5, §4).
"""

import logging
from functools import wraps
from urllib.parse import urlparse
from flask import abort, redirect, url_for, session
from flask_login import current_user
from classify.extensions import db, login_manager
from classify.models.user import User

logger = logging.getLogger(__name__)


@login_manager.user_loader
def load_user(user_id):
    """
    Flask-Login user loader callback.
    Validates session_version against current User record to enforce
    immediate session revocation on password reset (security.md §3.5).
    """
    try:
        user = db.session.get(User, int(user_id))
        if not user:
            return None

        # Session invalidation check (security.md §3.5)
        current_sess_ver = session.get("session_version")
        expected_ver = user.session_version or 1
        if current_sess_ver != expected_ver:
            return None

        return user
    except (TypeError, ValueError):
        return None


def is_safe_redirect(target: str) -> bool:
    """
    Strictly validates redirect targets to prevent open redirect vulnerabilities.
    Rejects protocol-relative URLs (//example.com), Windows-style backslashes (/\\example.com),
    and absolute URIs with external schemes or netlocs (security.md §2.6).
    """
    if not target or not isinstance(target, str):
        return False
    if target.startswith("//") or target.startswith("/\\"):
        return False
    parsed = urlparse(target)
    return not parsed.netloc and not parsed.scheme and target.startswith("/")


def role_required(required_role):
    """
    Decorator requiring the logged-in user to have a specific role ('student' or 'teacher').
    
    Behavior per security.md §4:
    - If unauthenticated: redirects to login.
    - If authenticated with wrong role: returns HTTP 403 Forbidden.
    - If role matches: proceeds to view function.
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not current_user.is_authenticated:
                return redirect(url_for("auth.login"))
            if current_user.role != required_role:
                abort(403, description=f"Access denied: this area requires '{required_role}' privileges.")
            return f(*args, **kwargs)
        return decorated_function
    return decorator


def teacher_required(f):
    """Convenience decorator requiring teacher role."""
    return role_required("teacher")(f)


def student_required(f):
    """Convenience decorator requiring student role."""
    return role_required("student")(f)
