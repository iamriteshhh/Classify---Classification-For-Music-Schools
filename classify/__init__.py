"""
classify
========
Application package for CLASSIFY — Music Classification System for Music Schools.
Provides the Flask application factory create_app().
Hardened per Security Specification (security.md) and Final Design (final_design.md).
"""

import os
import mimetypes
import click
import logging
from flask import Flask, render_template, request, jsonify
from config import config_by_name, DevelopmentConfig
from classify.extensions import db, migrate, login_manager, csrf, limiter, mail

# Explicitly guarantee proper MIME types on Windows platforms (prevents nosniff CSS blocking)
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("application/javascript", ".js")

logger = logging.getLogger(__name__)


def create_app(config_name_or_class=None):
    """
    Application factory for CLASSIFY.
    
    Parameters:
        config_name_or_class: Config class, config name string ('development', 'testing', 'production'),
                              or None (defaults to FLASK_CONFIG / FLASK_ENV or DevelopmentConfig).
                              
    Returns:
        Flask application instance.
    """
    package_dir = os.path.abspath(os.path.dirname(__file__))
    root_dir = os.path.abspath(os.path.join(package_dir, ".."))

    # Resolve template and static folders (supports both package and root locations)
    template_folder = (
        os.path.join(package_dir, "templates")
        if os.path.exists(os.path.join(package_dir, "templates"))
        else os.path.join(root_dir, "templates")
    )
    static_folder = (
        os.path.join(package_dir, "static")
        if os.path.exists(os.path.join(package_dir, "static"))
        else os.path.join(root_dir, "static")
    )

    app = Flask(
        __name__,
        template_folder=template_folder,
        static_folder=static_folder,
        static_url_path="/static",
    )

    # Resolve configuration
    if config_name_or_class is None:
        env_config = os.environ.get("FLASK_CONFIG") or os.environ.get("FLASK_ENV", "development")
        config_class = config_by_name.get(env_config.lower(), DevelopmentConfig)
    elif isinstance(config_name_or_class, str):
        config_class = config_by_name.get(config_name_or_class.lower(), DevelopmentConfig)
    else:
        config_class = config_name_or_class

    app.config.from_object(config_class)

    # Log boot configuration state (security.md §5.8)
    logger.info("CLASSIFY booting with config: %s (DEBUG=%s, TESTING=%s)", config_class.__name__, app.config.get("DEBUG"), app.config.get("TESTING"))

    # Ensure upload directory exists
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)
    csrf.init_app(app)
    limiter.init_app(app)
    mail.init_app(app)

    # SQLite WAL (Write-Ahead Logging) Mode configuration (Phase 5 Task 5.5)
    with app.app_context():
        try:
            from sqlalchemy import event
            from sqlite3 import Connection as SQLite3Connection

            @event.listens_for(db.engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                if isinstance(dbapi_connection, SQLite3Connection):
                    cursor = dbapi_connection.cursor()
                    cursor.execute("PRAGMA journal_mode=WAL")
                    cursor.execute("PRAGMA synchronous=NORMAL")
                    cursor.close()
        except Exception as e:
            logger.warning("Could not set SQLite WAL pragmas: %s", e)

    # Import auth utilities to register user loader
    from classify import auth_utils  # noqa: F401

    # Import models so SQLAlchemy / Alembic can discover metadata
    with app.app_context():
        from classify import models  # noqa: F401
        db.create_all()

        # Seamless SQLite schema auto-migration for newly added security columns
        try:
            from sqlalchemy import inspect, text
            inspector = inspect(db.engine)
            if "users" in inspector.get_table_names():
                existing_cols = {c["name"] for c in inspector.get_columns("users")}
                with db.engine.connect() as conn:
                    if "failed_login_attempts" not in existing_cols:
                        conn.execute(text("ALTER TABLE users ADD COLUMN failed_login_attempts INTEGER DEFAULT 0 NOT NULL"))
                    if "locked_until" not in existing_cols:
                        conn.execute(text("ALTER TABLE users ADD COLUMN locked_until DATETIME"))
                    if "email_verified" not in existing_cols:
                        conn.execute(text("ALTER TABLE users ADD COLUMN email_verified BOOLEAN DEFAULT 0 NOT NULL"))
                    if "session_version" not in existing_cols:
                        conn.execute(text("ALTER TABLE users ADD COLUMN session_version INTEGER DEFAULT 1 NOT NULL"))
                    conn.commit()

            if "audio_features" in inspector.get_table_names():
                af_cols = {c["name"] for c in inspector.get_columns("audio_features")}
                with db.engine.connect() as conn:
                    if "detected_key" not in af_cols:
                        conn.execute(text("ALTER TABLE audio_features ADD COLUMN detected_key VARCHAR(32)"))
                    if "time_signature" not in af_cols:
                        conn.execute(text("ALTER TABLE audio_features ADD COLUMN time_signature VARCHAR(16) DEFAULT '4/4'"))
                    if "chord_progression" not in af_cols:
                        conn.execute(text("ALTER TABLE audio_features ADD COLUMN chord_progression VARCHAR(120)"))
                    if "vocal_presence" not in af_cols:
                        conn.execute(text("ALTER TABLE audio_features ADD COLUMN vocal_presence VARCHAR(24)"))
                    conn.commit()
        except Exception as e:
            logger.warning("Database schema check notice: %s", e)

    # Security Headers after_request hook (security.md §5.1)
    @app.after_request
    def apply_security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        
        csp_policy = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://unpkg.com; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com data:; "
            "img-src 'self' data:; "
            "media-src 'self' blob:; "
            "connect-src 'self' https://api.pwnedpasswords.com https://unpkg.com;"
        )
        response.headers["Content-Security-Policy"] = csp_policy

        if app.config.get("SESSION_COOKIE_SECURE"):
            response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"

        return response

    # Custom HTTP error pages (security.md §5.8)
    def render_error(code, title, message):
        if request.headers.get("X-Requested-With") == "XMLHttpRequest" or (
            request.accept_mimetypes.accept_json and not request.accept_mimetypes.accept_html
        ):
            return jsonify({"success": False, "error": message, "code": code}), code
        return (
            render_template(
                "errors/error.html",
                error_code=code,
                error_title=title,
                error_message=message,
            ),
            code,
        )

    @app.errorhandler(400)
    def handle_bad_request(e):
        return render_error(400, "Bad Request", "The request could not be processed due to invalid parameters or malformed syntax.")

    @app.errorhandler(403)
    def handle_forbidden(e):
        return render_error(403, "Access Denied", getattr(e, "description", "You do not have permission to view this resource."))

    @app.errorhandler(404)
    def handle_not_found(e):
        return render_error(404, "Page Not Found", getattr(e, "description", "The requested page or resource could not be found."))

    @app.errorhandler(413)
    def handle_payload_too_large(e):
        return render_error(413, "File Too Large", "The uploaded audio file exceeds the maximum permitted upload limit (25 MB).")

    @app.errorhandler(429)
    def handle_rate_limit(e):
        return render_error(429, "Too Many Requests", "Too many requests. Please slow down and try again shortly.")

    @app.errorhandler(500)
    def handle_server_error(e):
        logger.exception("Internal Server Error: %s", e)
        return render_error(500, "Server Error", "An internal error occurred. Please try again later.")

    # Health check & readiness probe endpoints (Phase 5 Task 5.3)
    @app.route("/healthz", methods=["GET"])
    def healthz():
        """Liveness health check endpoint (Phase 5 Task 5.3)."""
        return jsonify({"status": "ok", "service": "classify"}), 200

    @app.route("/ready", methods=["GET"])
    def ready():
        """Readiness check validating database connection and ML model artifact (Phase 5 Task 5.3)."""
        db_healthy = False
        try:
            from sqlalchemy import text
            db.session.execute(text("SELECT 1"))
            db_healthy = True
        except Exception as ex:
            logger.error("Readiness check database failure: %s", ex)

        model_healthy = False
        try:
            from classify.ml.inference import load_artifacts
            artifacts = load_artifacts()
            if artifacts.get("model") is not None:
                model_healthy = True
        except Exception as ex:
            logger.error("Readiness check model failure: %s", ex)

        if db_healthy and model_healthy:
            return jsonify({
                "status": "ready",
                "database": "connected",
                "model": "loaded",
            }), 200

        return jsonify({
            "status": "not_ready",
            "database": "connected" if db_healthy else "disconnected",
            "model": "loaded" if model_healthy else "unavailable",
        }), 503

    # Context processor for global student notifications (Phase 3 Task 3.5)
    @app.context_processor
    def inject_notifications():
        from flask_login import current_user
        from classify.models import Notification
        if current_user and current_user.is_authenticated:
            try:
                unread = Notification.query.filter_by(user_id=current_user.id, is_read=False).count()
                recent = Notification.query.filter_by(user_id=current_user.id).order_by(Notification.created_at.desc()).limit(5).all()
                return {"unread_notifications_count": unread, "recent_notifications": recent}
            except Exception:
                pass
        return {"unread_notifications_count": 0, "recent_notifications": []}

    # Register CLI commands
    @app.cli.command("seed-db")
    def seed_db_command():
        """Seed the database with initial genre taxonomy."""
        from classify.models.seed import seed_genres
        g_count, s_count = seed_genres(app)
        click.echo(f"Seeded {g_count} genres and {s_count} subgenres.")

    @app.cli.command("cleanup-uploads")
    @click.option("--days", default=30, type=int, help="Max age in days for orphaned files.")
    def cleanup_uploads_command(days):
        """Clean up orphaned audio files older than specified days."""
        from classify.services.analysis_service import cleanup_orphaned_uploads
        res = cleanup_orphaned_uploads(app.config["UPLOAD_FOLDER"], max_age_days=days)
        mb = round(res["freed_bytes"] / (1024 * 1024), 2)
        click.echo(f"Cleanup complete: removed {res['deleted_count']} orphaned files ({mb} MB freed).")

    # Register blueprints per structure.md §2
    from .routes import (
        analysis_bp,
        auth_bp,
        student_bp,
        teacher_bp,
        library_bp,
        explore_bp,
        api_bp,
    )

    app.register_blueprint(analysis_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(student_bp)
    app.register_blueprint(teacher_bp)
    app.register_blueprint(library_bp)
    app.register_blueprint(explore_bp)
    app.register_blueprint(api_bp)

    # Exempt REST API from CSRF since it uses Bearer token auth
    csrf.exempt(api_bp)

    # Provide endpoint aliases for backward-compatibility with tests/templates
    if "analysis.index" in app.view_functions and "index" not in app.view_functions:
        app.add_url_rule("/", endpoint="index", view_func=app.view_functions["analysis.index"], methods=["GET"])
    if "analysis.analyze" in app.view_functions and "analyze" not in app.view_functions:
        app.add_url_rule("/analyze", endpoint="analyze", view_func=app.view_functions["analysis.analyze"], methods=["POST"])
    if "analysis.uploaded_file" in app.view_functions and "uploaded_file" not in app.view_functions:
        app.add_url_rule("/uploads/<path:filename>", endpoint="uploaded_file", view_func=app.view_functions["analysis.uploaded_file"], methods=["GET"])

    # Configure ProxyFix for reverse-proxy environments (Cloudflare, Render, Gunicorn, etc.)
    from werkzeug.middleware.proxy_fix import ProxyFix
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

    return app
