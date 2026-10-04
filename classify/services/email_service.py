"""CLASSIFY - Email Delivery Service.

Handles transactional email dispatch (password resets, account verification)
using Flask-Mail with HTML and plain-text multipart fallbacks.
"""

from __future__ import annotations

import logging
from flask import current_app, render_template
from flask_mail import Message
from classify.extensions import mail

logger = logging.getLogger(__name__)


def send_password_reset_email(user, reset_url: str) -> bool:
    """Dispatches a password reset email to the specified user.

    Returns True if successfully sent or suppressed in testing, False on delivery error.
    """
    subject = "Reset Your CLASSIFY Password"
    sender = current_app.config.get("MAIL_DEFAULT_SENDER", "noreply@classify.music")

    try:
        html_body = render_template("email/reset_password.html", user=user, reset_url=reset_url)
        text_body = render_template("email/reset_password.txt", user=user, reset_url=reset_url)

        msg = Message(
            subject=subject,
            sender=sender,
            recipients=[user.email],
            body=text_body,
            html=html_body,
        )

        if current_app.config.get("MAIL_SUPPRESS_SEND"):
            logger.info("EMAIL_SUPPRESSED: password reset email to %s (url: %s)", user.email, reset_url)
            return True

        mail.send(msg)
        logger.info("EMAIL_SENT: password reset email to %s", user.email)
        return True
    except Exception as e:
        logger.warning("EMAIL_DELIVERY_FAILED: could not send reset email to %s: %s", user.email, e)
        return False


def send_verification_email(user, verify_url: str) -> bool:
    """Dispatches an email verification link to the newly registered user.

    Returns True if successfully sent or suppressed in testing, False on delivery error.
    """
    subject = "Verify Your CLASSIFY Account"
    sender = current_app.config.get("MAIL_DEFAULT_SENDER", "noreply@classify.music")

    try:
        html_body = render_template("email/verify_email.html", user=user, verify_url=verify_url)
        text_body = render_template("email/verify_email.txt", user=user, verify_url=verify_url)

        msg = Message(
            subject=subject,
            sender=sender,
            recipients=[user.email],
            body=text_body,
            html=html_body,
        )

        if current_app.config.get("MAIL_SUPPRESS_SEND"):
            logger.info("EMAIL_SUPPRESSED: verification email to %s (url: %s)", user.email, verify_url)
            return True

        mail.send(msg)
        logger.info("EMAIL_SENT: verification email to %s", user.email)
        return True
    except Exception as e:
        logger.warning("EMAIL_DELIVERY_FAILED: could not send verification email to %s: %s", user.email, e)
        return False
