"""
user.py
=======
User model representing system accounts with role designation ('student' or 'teacher').
Inherits from flask_login.UserMixin for session management.
Part of Phase 4 & Phase 5 for CLASSIFY.
Updated with Security Hardening Specification (security.md §2.1, §2.5, §3.3, §3.5).
"""

from datetime import datetime, timezone, timedelta
from flask_login import UserMixin
from classify.extensions import db
from classify.security.passwords import hash_password, verify_and_check_upgrade


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(20), nullable=False, default="student")  # 'student' | 'teacher'
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    # Security hardening columns (security.md §2.1, §3.3, §3.5)
    failed_login_attempts = db.Column(db.Integer, default=0, nullable=False)
    locked_until = db.Column(db.DateTime, nullable=True)
    email_verified = db.Column(db.Boolean, default=False, nullable=False)
    session_version = db.Column(db.Integer, default=1, nullable=False)

    # 1:1 relationships with role profiles
    student_profile = db.relationship(
        "Student", back_populates="user", uselist=False, cascade="all, delete-orphan"
    )
    teacher_profile = db.relationship(
        "Teacher", back_populates="user", uselist=False, cascade="all, delete-orphan"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.session_version is None:
            self.session_version = 1
        if self.failed_login_attempts is None:
            self.failed_login_attempts = 0

    def set_password(self, password):
        """Hashes and sets the user's password using Argon2id."""
        self.password_hash = hash_password(password)

    def check_password(self, password):
        """
        Verifies the password against stored hash.
        If hash is legacy (PBKDF2/scrypt), transparently upgrades to Argon2id.
        """
        is_valid, needs_upgrade = verify_and_check_upgrade(self.password_hash, password)
        if is_valid and needs_upgrade:
            self.set_password(password)
            try:
                db.session.commit()
            except Exception:
                db.session.rollback()
        return is_valid

    def is_locked(self) -> bool:
        """Returns True if account is currently locked out."""
        if self.locked_until is None:
            return False
        # Normalize to UTC
        now = datetime.now(timezone.utc)
        locked_time = self.locked_until
        if locked_time.tzinfo is None:
            locked_time = locked_time.replace(tzinfo=timezone.utc)
        return now < locked_time

    def record_failed_login(self, max_attempts: int = 5, lockout_minutes: int = 15):
        """Records a failed login attempt and locks account if threshold reached."""
        self.failed_login_attempts = (self.failed_login_attempts or 0) + 1
        if self.failed_login_attempts >= max_attempts:
            self.locked_until = datetime.now(timezone.utc) + timedelta(minutes=lockout_minutes)
        db.session.commit()

    def record_successful_login(self):
        """Resets failed login count and lockout status on successful login."""
        self.failed_login_attempts = 0
        self.locked_until = None
        db.session.commit()

    def increment_session_version(self):
        """Increments session version, invalidating all other active sessions."""
        self.session_version = (self.session_version or 1) + 1
        db.session.commit()

    def is_student(self):
        return self.role == "student"

    def is_teacher(self):
        return self.role == "teacher"

    @property
    def display_name(self):
        """Returns a human-readable display name derived from email."""
        if not self.email:
            return "Student" if self.is_student() else "Teacher"
        local = self.email.split("@")[0]
        cleaned = local.replace(".", " ").replace("_", " ").replace("-", " ").strip()
        return cleaned.title() if cleaned else "Student"

    @property
    def initials(self):
        """Returns 1-2 uppercase letters for user avatar."""
        name = self.display_name
        parts = name.split()
        if len(parts) >= 2:
            return (parts[0][0] + parts[1][0]).upper()
        elif parts and parts[0]:
            return parts[0][:2].upper()
        return "U"

    def __repr__(self):
        return f"<User {self.email} ({self.role})>"
