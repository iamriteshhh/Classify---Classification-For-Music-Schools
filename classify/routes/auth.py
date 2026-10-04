"""
auth.py
=======
Authentication routes for CLASSIFY.
Handles Student and Teacher registration, login, session termination,
password resets, email verification, and safe redirects.
Hardened per Security Specification (security.md §2, §3).
"""

import logging
from flask import (
    Blueprint,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session,
    current_app,
)
from flask_login import login_user, logout_user, login_required, current_user
from email_validator import validate_email, EmailNotValidError

from classify.extensions import db, limiter
from classify.models import User, Student, Teacher
from classify.auth_utils import is_safe_redirect
from classify.security.passwords import validate_password_policy
from classify.security.tokens import (
    generate_verification_token,
    confirm_verification_token,
    generate_reset_token,
    confirm_reset_token,
)
from classify.forms import LoginForm, RegisterForm, RequestResetForm, ResetPasswordForm
from classify.services.email_service import send_verification_email, send_password_reset_email

logger = logging.getLogger("classify.security.audit")

auth_bp = Blueprint("auth", __name__, url_prefix="/auth")


@auth_bp.route("/login", methods=["GET", "POST"])
@limiter.limit("5 per 15 minutes", methods=["POST"])
def login():
    """
    Renders login form and processes authentication.
    Enforces rate limiting, account lockout tracking, generic enumeration-safe
    error responses, safe redirects, and session version tracking.
    """
    if current_user.is_authenticated and request.method == "GET":
        if current_user.is_teacher():
            return redirect(url_for("teacher.dashboard"))
        return redirect(url_for("student.dashboard"))

    form = LoginForm()

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        remember = bool(request.form.get("remember"))

        if not email or not password:
            flash("Please provide both email and password.", "error")
            return render_template("auth/login.html", form=form)

        user = User.query.filter_by(email=email).first()

        # Account lockout check (security.md §2.1)
        if user is not None and user.is_locked():
            logger.warning("AUTH_LOGIN_BLOCKED_LOCKED: email=%s ip=%s", email, request.remote_addr)
            # Generic error prevents enumeration of locked state
            flash("Invalid email or password.", "error")
            return render_template("auth/login.html", form=form)

        # Authentication verification
        if user is None or not user.check_password(password):
            if user is not None:
                user.record_failed_login(max_attempts=5, lockout_minutes=15)
                logger.warning(
                    "AUTH_LOGIN_FAILED: email=%s ip=%s attempts=%d",
                    email,
                    request.remote_addr,
                    user.failed_login_attempts,
                )
            else:
                logger.warning("AUTH_LOGIN_UNKNOWN_USER: email=%s ip=%s", email, request.remote_addr)

            flash("Invalid email or password.", "error")
            return render_template("auth/login.html", form=form)

        # Successful login
        user.record_successful_login()
        session["session_version"] = user.session_version
        login_user(user, remember=remember)
        logger.info("AUTH_LOGIN_SUCCESS: email=%s role=%s ip=%s", email, user.role, request.remote_addr)

        # Redirect to target page if safe, else role dashboard (security.md §2.6)
        next_page = request.args.get("next")
        if next_page and is_safe_redirect(next_page):
            return redirect(next_page)

        if user.is_teacher():
            return redirect(url_for("teacher.dashboard"))
        return redirect(url_for("student.dashboard"))

    return render_template("auth/login.html", form=form)


@auth_bp.route("/register", methods=["GET", "POST"])
@limiter.limit("10 per hour", methods=["POST"])
def register():
    """
    Renders registration form and creates Student/Teacher accounts.
    Enforces NIST 800-63B password requirements, email validation,
    and email verification token generation.
    """
    if current_user.is_authenticated and request.method == "GET":
        if current_user.is_teacher():
            return redirect(url_for("teacher.dashboard"))
        return redirect(url_for("student.dashboard"))

    form = RegisterForm()

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")
        role = request.form.get("role", "student").strip().lower()

        # Validation
        if not email or not password:
            flash("Email and password are required.", "error")
            return render_template("auth/register.html", form=form)

        try:
            valid = validate_email(email, check_deliverability=False)
            email = valid.normalized
        except EmailNotValidError:
            flash("Please enter a valid email address.", "error")
            return render_template("auth/register.html", form=form)

        # NIST 800-63B password policy check (security.md §2.4)
        is_pwd_valid, pwd_err = validate_password_policy(password, check_breaches=True)
        if not is_pwd_valid:
            flash(pwd_err or "Password does not meet security requirements.", "error")
            return render_template("auth/register.html", form=form)

        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return render_template("auth/register.html", form=form)

        if role not in {"student", "teacher"}:
            role = "student"

        # Check existing user
        if User.query.filter_by(email=email).first():
            flash("An account with that email already exists.", "error")
            return render_template("auth/register.html", form=form)

        # Create user and profile
        user = User(email=email, role=role, email_verified=False, session_version=1)
        user.set_password(password)
        db.session.add(user)
        db.session.flush()

        if role == "student":
            student_profile = Student(user_id=user.id)
            db.session.add(student_profile)
        elif role == "teacher":
            teacher_profile = Teacher(user_id=user.id)
            db.session.add(teacher_profile)

        db.session.commit()

        # Generate verification token (security.md §3.3)
        verify_token = generate_verification_token(user.email)
        verify_url = url_for("auth.verify_email", token=verify_token, _external=True)
        send_verification_email(user, verify_url)
        logger.info("AUTH_REGISTER_SUCCESS: email=%s role=%s verify_url=%s", email, role, verify_url)

        session["session_version"] = user.session_version
        login_user(user)
        flash(f"Welcome to CLASSIFY! Account registered as {role.capitalize()}.", "success")

        if role == "teacher":
            return redirect(url_for("teacher.dashboard"))
        return redirect(url_for("student.dashboard"))

    return render_template("auth/register.html", form=form)


@auth_bp.route("/verify/<token>", methods=["GET"])
def verify_email(token):
    """Verifies a user's email address using a signed token."""
    email = confirm_verification_token(token)
    if not email:
        flash("The verification link is invalid or has expired. Please request a new one.", "error")
        return redirect(url_for("auth.login"))

    user = User.query.filter_by(email=email).first()
    if not user:
        flash("User account not found.", "error")
        return redirect(url_for("auth.login"))

    user.email_verified = True
    db.session.commit()
    logger.info("AUTH_EMAIL_VERIFIED: email=%s", email)
    flash("Your email address has been successfully verified!", "success")

    if current_user.is_authenticated:
        if current_user.is_teacher():
            return redirect(url_for("teacher.dashboard"))
        return redirect(url_for("student.dashboard"))
    return redirect(url_for("auth.login"))


@auth_bp.route("/reset-request", methods=["GET", "POST"])
@limiter.limit("5 per hour", methods=["POST"])
def reset_request():
    """
    Renders password reset request form.
    Uses neutral messaging to prevent email enumeration (security.md §3.5).
    """
    form = RequestResetForm()

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        if email:
            user = User.query.filter_by(email=email).first()
            if user:
                token = generate_reset_token(user.email)
                reset_url = url_for("auth.reset_token", token=token, _external=True)
                send_password_reset_email(user, reset_url)
                logger.info("AUTH_PASSWORD_RESET_LINK: email=%s url=%s", email, reset_url)

        # Always return the same neutral confirmation to prevent enumeration
        flash("If an account exists for that email address, a password reset link has been dispatched.", "info")
        return redirect(url_for("auth.login"))

    return render_template("auth/reset_request.html", form=form)


@auth_bp.route("/reset/<token>", methods=["GET", "POST"])
@limiter.limit("5 per hour", methods=["POST"])
def reset_token(token):
    """
    Validates password reset token, updates password with Argon2id,
    and invalidates all existing sessions (security.md §3.5).
    """
    email = confirm_reset_token(token)
    if not email:
        flash("The password reset link is invalid or has expired.", "error")
        return redirect(url_for("auth.reset_request"))

    user = User.query.filter_by(email=email).first()
    if not user:
        flash("User account was not found.", "error")
        return redirect(url_for("auth.reset_request"))

    form = ResetPasswordForm()

    if request.method == "POST":
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        is_pwd_valid, pwd_err = validate_password_policy(password, check_breaches=True)
        if not is_pwd_valid:
            flash(pwd_err or "Password does not meet security requirements.", "error")
            return render_template("auth/reset_token.html", form=form, token=token)

        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return render_template("auth/reset_token.html", form=form, token=token)

        # Set new password and invalidate all existing sessions
        user.set_password(password)
        user.increment_session_version()
        user.record_successful_login()
        logger.info("AUTH_PASSWORD_RESET_SUCCESS: email=%s", email)

        flash("Your password has been reset successfully. Please sign in with your new password.", "success")
        return redirect(url_for("auth.login"))

    return render_template("auth/reset_token.html", form=form, token=token)


@auth_bp.route("/logout", methods=["GET", "POST"])
@login_required
def logout():
    """Logs out the active user and terminates the session."""
    session.pop("session_version", None)
    logout_user()
    flash("You have been signed out.", "info")
    return redirect(url_for("analysis.index"))
