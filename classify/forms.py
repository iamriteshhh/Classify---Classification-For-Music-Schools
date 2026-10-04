"""
forms.py
========
Flask-WTF form classes for Login, Registration, and Password Reset.
Integrates WTForms validators, CSRF protection, and NIST 800-63B password policy.
Part of Security Hardening Specification (security.md §3.4).
"""

from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SelectField, SubmitField
from wtforms.validators import DataRequired, Email, EqualTo, Length, ValidationError
from classify.security.passwords import validate_password_policy


def nist_password_validator(form, field):
    """Custom WTForms validator enforcing NIST 800-63B password policy."""
    password = field.data or ""
    is_valid, error_message = validate_password_policy(password, check_breaches=True)
    if not is_valid:
        raise ValidationError(error_message or "Password does not meet security requirements.")


class LoginForm(FlaskForm):
    """User authentication form."""
    email = StringField("Email Address", validators=[DataRequired(), Email(), Length(max=120)])
    password = PasswordField("Password", validators=[DataRequired()])
    remember = BooleanField("Remember Me")
    submit = SubmitField("Sign In")


class RegisterForm(FlaskForm):
    """User registration form with NIST 800-63B validation."""
    email = StringField("Email Address", validators=[DataRequired(), Email(), Length(max=120)])
    password = PasswordField(
        "Password",
        validators=[
            DataRequired(),
            Length(min=10, max=128, message="Password must be between 10 and 128 characters."),
            nist_password_validator,
        ],
    )
    confirm_password = PasswordField(
        "Confirm Password",
        validators=[
            DataRequired(),
            EqualTo("password", message="Passwords do not match. Please verify your password."),
        ],
    )
    role = SelectField(
        "Account Type",
        choices=[("student", "Student"), ("teacher", "Teacher / Instructor")],
        default="student",
        validators=[DataRequired()],
    )
    submit = SubmitField("Create Account")


class RequestResetForm(FlaskForm):
    """Form requesting a password reset email."""
    email = StringField("Email Address", validators=[DataRequired(), Email(), Length(max=120)])
    submit = SubmitField("Send Password Reset Link")


class ResetPasswordForm(FlaskForm):
    """Form to submit a new password after token validation."""
    password = PasswordField(
        "New Password",
        validators=[
            DataRequired(),
            Length(min=10, max=128, message="Password must be between 10 and 128 characters."),
            nist_password_validator,
        ],
    )
    confirm_password = PasswordField(
        "Confirm New Password",
        validators=[
            DataRequired(),
            EqualTo("password", message="Passwords do not match. Please verify your password."),
        ],
    )
    submit = SubmitField("Reset Password")
