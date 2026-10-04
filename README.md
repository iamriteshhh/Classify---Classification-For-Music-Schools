# CLASSIFY — Song Classification System for Music Schools

**CLASSIFY** is a full-featured, machine learning-driven music classification and music-education web platform designed for music schools, conservatories, and collegiate music programs. It combines robust acoustic signal processing, trained classical machine learning classifiers, a data-driven musical taxonomy, real-time pedagogical feedback, and dedicated student and teacher workflows.

---

## 🎵 Key Features & System Capabilities

### 1. Trained Classical Machine Learning (Phase 0 & Phase 1)
- **High-Accuracy Genre Classification**: Support Vector Machine (RBF kernel) achieving **72.15% test accuracy** on held-out GTZAN splits and **85.71% accuracy** on commercial golden evaluation tracks (eliminating the historical Ambient/Metal sink).
- **25 Subgenre Hierarchy**: Hierarchical classification mapping primary genres to 25 subgenres (e.g., Delta/Chicago Blues, Baroque/Romantic Classical, Bebop/Hard Bop Jazz, Heavy/Thrash Metal, Synthpop/Dance-Pop) with acoustic-rule and probability-conditioned scoring.
- **Multi-Label Cross-Genre Detection**: Detects stylistic fusions and hybrid tracks (e.g., Jazz-Rock, Pop-Disco, Blues-Rock) with confidence thresholds and secondary tag extraction.
- **Active Learning Pipeline (`classify/ml/active_learning.py`)**: Uncertainty sampling (Least Confidence, Margin Sampling, Entropy) to prioritize borderline and teacher-disputed tracks for retraining.
- **Audio Embeddings (`classify/ml/embeddings.py`)**: 32-dimensional normalized acoustic embeddings with cosine similarity search for stylistic neighbor retrieval.
- **Acoustic Explainability**: Feature importance breakdown revealing the primary drivers behind each classification (tempo, spectral centroid, RMS energy, spectral contrast).

### 2. Deep Audio Analysis & Music Theory (Phase 3 & Phase 4)
- **Musical Key & Mode Detection**: Krumhansl-Schmuckler cognitive key-profile correlation analyzing 12-dimensional chromagrams across all 24 major and minor keys.
- **Time Signature & Meter Detection**: Beat-period autocorrelation and pulse-salience analysis identifying common musical meters: **4/4**, **3/4**, **6/8**, and **5/4**.
- **Chord Progression Detection**: Triad template matching over time identifying chord sequences (e.g., C - G - Am - F) with progression categorization (I-V-vi-IV, ii-V-I, 12-bar blues).
- **Vocal vs. Instrumental Detection (VAD)**: Spectral flux and vocal formant energy concentration (300–3400 Hz) tagging tracks as "Vocal" or "Instrumental".
- **EBU R128 Loudness Normalization**: Integrated loudness normalized to -14.0 LUFS via `pyloudnorm`.
- **4th-Order Butterworth Lowpass Filtering**: 8,000 Hz filtering eliminating sample-rate bandwidth domain shifts.

### 3. Asynchronous Backend & Cloud Architecture (Phase 2 & Phase 5)
- **Background Task Processing (`classify/tasks.py`)**: Asynchronous audio analysis with ThreadPool workers, fine-grained stage tracking, and Server-Sent Events (SSE) streaming (`/analysis/stream/<task_id>`).
- **Transactional SQLite with WAL Mode**: Concurrency-safe SQLite engine with `PRAGMA journal_mode=WAL` and `PRAGMA synchronous=NORMAL` preventing lock contention.
- **Storage Quota Enforcement**: Strict user storage management (`UPLOAD_QUOTA_MB`, default 500 MB) and automated orphaned audio cleanup CLI (`flask cleanup-orphans`).
- **Email Delivery (Flask-Mail)**: Asynchronous password reset tokens, account verification, and notification delivery with fallback suppression for offline testing.
- **Observability & Health Checks**: `/healthz` (liveness) and `/ready` (readiness verifying database connectivity and preloaded ML model artifacts).

### 4. Pedagogical Workflows & Classrooms (Phase 3 & Phase 5)
- **Interactive Waveform Visualizer**: WaveSurfer.js integration displaying interactive waveforms, audio playback, time-scrubbing, and segment selection.
- **Classroom Management (`/teacher/classrooms`)**: Teachers can create classrooms with unique enrollment codes; students can enroll with a single click.
- **Assignment Submissions (`/student/assignments`)**: Instructors can issue song analysis assignments; students submit their work and receive grades and qualitative feedback.
- **Notification System**: In-app bell notification menu with unread badges alerting students to new assignments, grades, and teacher reviews.
- **ReportLab PDF Export**: Instant downloadable, print-ready PDF analysis reports (`/analysis/export-pdf/<id>`) detailing metadata, key, meter, chord progressions, and acoustic metrics.
- **Accessible Design (WCAG 2.2 AA)**: Full keyboard navigation (`Space` to toggle play/pause, `J`/`ArrowLeft` to rewind 5s, `L`/`ArrowRight` to forward 5s, `K` to pause) with input-focus guards and ARIA accessibility labels.
- **Explore Module Exemplars**: Interactive audio players with CC-licensed representative clips across all 10 core genres in the side-by-side comparison matrix.

### 5. LMS REST API (`/api/v1/`)
- Complete RESTful API with stateless HMAC Bearer token authentication (`/api/v1/auth/token`).
- Endpoints for uploading audio, fetching analyses, listing past jobs, and querying taxonomy.
- Built-in OpenAPI 3.0 specification available at `/api/v1/openapi.json`.

---

## 🏛️ System Architecture

```text
Classify/
├── app.py                      # Development server entry point
├── wsgi.py                     # Production WSGI entry point (Gunicorn/Waitress)
├── config.py                   # Environment configuration classes
├── requirements.txt            # Python package dependencies
├── Dockerfile                  # Multi-stage production container build
├── docker-compose.yml          # Container orchestration (Web + Redis)
├── .github/workflows/ci.yml    # CI/CD pipeline (Lint, Test, Model Eval, Security)
├── model_artifacts/            # Serialized ML models and metadata
│   ├── genre_classifier.joblib
│   ├── scaler.joblib
│   ├── label_encoder.joblib
│   └── evaluation_report.json
├── datasets/                   # Dataset manifests and documentation
│   ├── manifest.csv
│   └── DATASET_SOURCES.md      # Dataset sources, licensing, and curation methodology
├── static/                     # Static CSS, JS, and audio exemplars
│   ├── style.css
│   └── audio/exemplars/        # Genre audio exemplar files
├── templates/                  # Frontend Jinja2 templates
│   ├── index.html              # Upload & Waveform analysis view
│   ├── auth/                   # Login, Register, Forgot Password
│   ├── student/                # Student dashboard & assignments
│   ├── teacher/                # Instructor dashboard, reviews & classrooms
│   ├── library/                # Genre Library catalog
│   └── explore/                # Comparative Explore matrix
├── classify/                   # Core application package
│   ├── __init__.py             # Application factory with WAL & routes
│   ├── extensions.py           # Extensions (SQLAlchemy, Login, CSRF, Mail)
│   ├── tasks.py                # Asynchronous background task runner & SSE
│   ├── audio/
│   │   ├── audio_processing.py # Acoustic features, key, meter, chords, VAD
│   │   └── validation.py       # Audio integrity & duration validation
│   ├── ml/
│   │   ├── feature_pipeline.py # 89-dim acoustic feature extractor
│   │   ├── inference.py        # Model loading, prediction & subgenres
│   │   ├── active_learning.py  # Uncertainty sampling for retraining
│   │   ├── embeddings.py       # 32-dim acoustic embeddings & similarity
│   │   ├── train.py            # Model training & model comparison
│   │   └── explain.py          # Feature importance explainability
│   ├── models/                 # SQLAlchemy ORM models
│   │   ├── user.py             # User, Student, Teacher
│   │   ├── song.py             # Song, Analysis, Classroom, Assignment
│   │   ├── audio_features.py   # Tier 1 acoustic features, key, chords, meter
│   │   ├── prediction.py       # Predictions and TeacherReviews
│   │   ├── genre.py            # Genres and Subgenres
│   │   └── notification.py     # In-app notifications
│   ├── routes/                 # Blueprint HTTP handlers
│   │   ├── analysis.py         # Upload, async progress, PDF export
│   │   ├── auth.py             # Authentication & password reset
│   │   ├── student.py          # Student portal & assignments
│   │   ├── teacher.py          # Instructor queue, reviews & classrooms
│   │   ├── library.py          # Music taxonomy catalog
│   │   ├── explore.py          # Comparative analysis & audio players
│   │   └── api.py              # REST API for LMS integration (/api/v1/)
│   └── services/               # Decoupled business logic
│       ├── audio_service.py
│       ├── classification_service.py
│       ├── analysis_service.py
│       ├── pdf_service.py
│       └── teaching_topics_service.py
└── tests/                      # Automated test suite (18 test modules, 114+ tests)
```

---

## 📊 Machine Learning Evaluation & Changelog

All models are trained with strict **artist-level grouped splitting** (`artist_id`) to ensure zero data leakage between training, validation, and test sets.

### Model Comparison Matrix (Held-Out Test Split)

| Model Family | Test Accuracy | Macro Precision | Macro Recall | **Macro F1** | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Support Vector Machine (RBF)** | **72.15%** | **0.7169** | **0.7127** | **0.7105** | **Deployed Production Winner** |
| HistGradientBoosting | 67.23% | 0.6767 | 0.6612 | 0.6746 | Candidate |
| Multi-Layer Perceptron (MLP) | 67.80% | 0.6720 | 0.6690 | 0.6744 | Candidate |
| Logistic Regression | 67.80% | 0.6790 | 0.6710 | 0.6766 | Candidate |
| Random Forest | 64.97% | 0.6650 | 0.6480 | 0.6589 | Candidate |
| K-Nearest Neighbors | 57.63% | 0.5480 | 0.5867 | 0.5640 | Candidate |

### Commercial Golden Test Set (28 Real-World Tracks)
- **Multi-Segment Aggregated Accuracy**: **85.71%** (24/28 correct)
- **Golden Set Macro F1**: **0.8697**
- **Ambient Attractor Rate**: Reduced to 10.7% (eliminating the pathological Ambient/Metal sink on modern pop & rock uploads).

### Accuracy Changelog
- **v1.0.0 (Phase 0 & 1 ML Architecture)**:
  - **Held-Out Test Accuracy**: **72.15%** (Macro F1: 0.7105)
  - **Golden Test Set Accuracy**: **85.71%** (Macro F1: 0.8697)
  - **Key Improvements**:
    1. EBU R128 integrated loudness normalization (-14.0 LUFS via `pyloudnorm`).
    2. 4th-order 8kHz lowpass filtering removing GTZAN vs modern bandwidth bias.
    3. RMS-weighted segment pooling replacing uniform averaging.
    4. Reactivated spectral contrast and mel spectrogram variance features.
    5. PowerTransformer (Yeo-Johnson) feature scaling with RBF SVM.
    6. 25 subgenre taxonomy and multi-label probability tagging.
- **v0.1.0 (Baseline Pre-Phase 0)**:
  - Test Accuracy: 62.66% (Macro F1: 0.6163).
  - Pathological collapses: Blues -> Ambient (99%), Pop -> Metal (95%).

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.11 or 3.12
- `libsndfile1` and `ffmpeg` (for local audio processing)
- Docker & Docker Compose (optional for containerized deployment)

### 2. Local Installation
```powershell
# Clone or navigate to the workspace
cd "d:\Classify - Song Classification System for Music Schools"

# Create and activate virtual environment
python -m venv venve
.\venve\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file based on `.env.example`:
```powershell
cp .env.example .env
```
Key configuration variables:
- `SECRET_KEY`: Cryptographic signing key for sessions and API tokens.
- `DATABASE_URL`: SQLAlchemy connection URI (defaults to `sqlite:///classify.db`).
- `UPLOAD_FOLDER`: Directory for audio files (defaults to `uploads/`).
- `UPLOAD_QUOTA_MB`: Storage quota per user in megabytes (default: `500`).
- `RATELIMIT_STORAGE_URI`: Storage URI for Flask-Limiter (`memory://` or `redis://localhost:6379/1`).
- `MAIL_SERVER`: SMTP server host (or leave default with `MAIL_SUPPRESS_SEND=true`).

### 4. Database Setup & Seeding
```powershell
# Seed the data-driven genre & subgenre taxonomy into SQLite
flask seed-db

# Clean up any orphaned files exceeding quota or missing DB records
flask cleanup-orphans
```

### 5. Running the Application
```powershell
# Development server (with hot reload)
python app.py

# Production server via Waitress (Windows)
waitress-serve --port=5000 wsgi:app

# Production server via Gunicorn (Linux/macOS)
gunicorn "wsgi:app" --bind 0.0.0.0:5000 --workers 2 --threads 4
```
Access the application at `http://localhost:5000`.

---

## 🐳 Docker & Container Deployment

### Running with Docker Compose (Recommended)
CLASSIFY includes a multi-stage Dockerfile (based on `python:3.11-slim-bookworm` with `libsndfile1` and `ffmpeg`) and a compose configuration with Redis.

```bash
# Build and start all services in detached mode
docker-compose up --build -d

# Check service status and health checks
docker-compose ps

# View application logs
docker-compose logs -f web

# Stop services
docker-compose down
```

Services started:
- `web`: CLASSIFY web app on `http://localhost:5000` with Gunicorn and health checks.
- `redis`: Redis 7 container for rate limiting and task coordination.

### Health Check Endpoints
- **Liveness**: `GET /healthz` — Returns `{"status": "ok", "app": "classify"}`.
- **Readiness**: `GET /ready` — Verifies database connectivity and ML model loading:
  ```json
  {
    "status": "ready",
    "database": "connected",
    "model_loaded": true,
    "genres_supported": 10
  }
  ```

---

## 🌐 REST API for LMS Integration (`/api/v1/`)

CLASSIFY exposes a stateless REST API for integration with Learning Management Systems (LMS) such as Canvas, Blackboard, or Moodle.

### Authentication
Authenticate using email and password to receive a Bearer token:
```bash
POST /api/v1/auth/token
Content-Type: application/json

{
  "email": "student@music.edu",
  "password": "yourpassword"
}
```
Response:
```json
{
  "token": "ey...",
  "token_type": "Bearer",
  "expires_in": 86400,
  "user": { "id": 1, "email": "student@music.edu", "role": "student" }
}
```

Include the token in subsequent requests:
```
Authorization: Bearer <token>
```

### API Endpoints
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/v1/auth/token` | Obtain HMAC Bearer authentication token |
| `GET` | `/api/v1/genres` | List all supported genres and subgenres |
| `POST` | `/api/v1/audio/analyze` | Upload audio file (`file` form field) for full analysis |
| `GET` | `/api/v1/analyses` | List paginated analyses for authenticated user |
| `GET` | `/api/v1/analyses/<id>` | Retrieve detailed analysis (probabilities, key, meter, chords) |
| `GET` | `/api/v1/openapi.json` | OpenAPI 3.0 specification for API documentation |

---

## 🧪 Testing & Verification

The platform maintains a comprehensive test suite across 18 modules (114+ unit and integration tests) verifying 100% pass rate.

```powershell
# Run the complete test suite
python -m pytest tests/ -v

# Run Phase-specific verification suites
python -m pytest tests/test_ml_accuracy_phase0.py -v
python -m pytest tests/test_deeper_ml_phase1.py -v
python -m pytest tests/test_backend_fixes_phase2.py -v
python -m pytest tests/test_pedagogical_ux_phase3.py -v
python -m pytest tests/test_audio_depth_phase4.py -v
python -m pytest tests/test_devops_and_api_phase5.py -v
```

---

## ♿ Accessibility (WCAG 2.2 AA)

CLASSIFY is engineered to comply with WCAG 2.2 AA accessibility standards:
- **Keyboard Shortcuts**:
  - `Space`: Toggle audio playback (play/pause).
  - `J` or `Left Arrow`: Rewind audio by 5 seconds.
  - `L` or `Right Arrow`: Advance audio by 5 seconds.
  - `K`: Pause playback.
  - Shortcuts automatically deactivate when typing in form inputs, textareas, or search boxes.
- **Accessible Controls**: All interactive buttons, volume sliders, and play controls feature descriptive `aria-label` attributes.
- **Live Regions**: Asynchronous upload progress bars and status indicators utilize `aria-live="polite"` regions for screen reader compatibility.

---

## 📜 Educational Ground Rules

1. **No Fake AI**: Every prediction uses a verified, trained scikit-learn model; prototype heuristics are completely eliminated.
2. **Real Probabilities**: Confidence metrics are calculated via calibrated `predict_proba`.
3. **Data-Driven Taxonomy**: All genres, subgenres, and descriptions are stored in SQLite and loaded via database relationships.
4. **Disagreement Preserved**: Model predictions are immutable; teacher corrections and notes are tracked separately in `TeacherReview`.
5. **Zero Artist Leakage**: Grouped splits by artist ensure models do not memorize specific timbre signatures.
6. **Clean Two-Role System**: Clear separation between Student and Teacher portals.
