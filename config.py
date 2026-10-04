import os
from datetime import timedelta
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    """Base configuration."""
    SECRET_KEY = os.environ.get("SECRET_KEY")
    UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", os.path.join(BASE_DIR, "uploads"))
    MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH", 25 * 1024 * 1024))  # 25 MB default
    
    # Comma-separated or set of allowed extensions
    _raw_extensions = os.environ.get("ALLOWED_AUDIO_EXTENSIONS", "wav,mp3")
    ALLOWED_AUDIO_EXTENSIONS = {ext.strip().lower() for ext in _raw_extensions.split(",") if ext.strip()}
    
    # Database and ML artifacts paths
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL",
        f"sqlite:///{os.path.join(BASE_DIR, 'classify.db')}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    MODEL_ARTIFACT_PATH = os.environ.get(
        "MODEL_ARTIFACT_PATH",
        os.path.join(BASE_DIR, "model_artifacts", "genre_classifier.joblib")
    )
    
    # CSRF Protection
    WTF_CSRF_ENABLED = True
    WTF_CSRF_TIME_LIMIT = None

    # Session & Remember-Me Cookie Hardening (security.md §2.3)
    SESSION_COOKIE_SECURE = os.environ.get("FLASK_ENV") == "production"
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    PERMANENT_SESSION_LIFETIME = timedelta(hours=12)

    REMEMBER_COOKIE_DURATION = timedelta(days=14)
    REMEMBER_COOKIE_SECURE = os.environ.get("FLASK_ENV") == "production"
    REMEMBER_COOKIE_HTTPONLY = True
    REMEMBER_COOKIE_SAMESITE = "Lax"

    # Rate Limiting (security.md §2.1)
    RATELIMIT_DEFAULT = "200 per day; 50 per hour"
    RATELIMIT_STORAGE_URI = os.environ.get("RATELIMIT_STORAGE_URI", "memory://")
    RATELIMIT_STRATEGY = "fixed-window"

    # Storage Quotas (Phase 2 Task 2.3)
    UPLOAD_QUOTA_MB = int(os.environ.get("UPLOAD_QUOTA_MB", 500))

    # Email Delivery Settings (Phase 2 Task 2.2)
    MAIL_SERVER = os.environ.get("MAIL_SERVER", "localhost")
    MAIL_PORT = int(os.environ.get("MAIL_PORT", 25))
    MAIL_USE_TLS = os.environ.get("MAIL_USE_TLS", "false").lower() in ("true", "1", "yes")
    MAIL_USE_SSL = os.environ.get("MAIL_USE_SSL", "false").lower() in ("true", "1", "yes")
    MAIL_USERNAME = os.environ.get("MAIL_USERNAME")
    MAIL_PASSWORD = os.environ.get("MAIL_PASSWORD")
    MAIL_DEFAULT_SENDER = os.environ.get("MAIL_DEFAULT_SENDER", "noreply@classify.music")
    MAIL_SUPPRESS_SEND = os.environ.get("MAIL_SUPPRESS_SEND", "false").lower() in ("true", "1", "yes")

    # Token security salt
    SECURITY_PASSWORD_SALT = os.environ.get("SECURITY_PASSWORD_SALT", "classify-security-salt")


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    TESTING = False
    # Safe fallback only for local development
    SECRET_KEY = os.environ.get("SECRET_KEY", "classify-dev-secret-key-change-in-production")


class TestingConfig(Config):
    """Testing configuration."""
    DEBUG = False
    TESTING = True
    SECRET_KEY = "test-secret-key"
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    WTF_CSRF_ENABLED = False
    RATELIMIT_ENABLED = False
    MAIL_SUPPRESS_SEND = True


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    TESTING = False

    @property
    def SECRET_KEY(self):
        # Enforce fail-closed if SECRET_KEY is not set in production (security.md §5.2)
        key = os.environ.get("SECRET_KEY")
        if not key:
            raise KeyError("SECRET_KEY environment variable must be set in production.")
        return key


config_by_name = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig,
}
