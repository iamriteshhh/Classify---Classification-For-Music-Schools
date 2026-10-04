"""
api.py
======
REST API for LMS Integration (Canvas, Blackboard, Moodle) and Third-Party Systems.
Exposes token authentication, audio analysis, analysis history, genre catalog,
and OpenAPI 3.0 specification.
Part of Phase 5 (Task 5.8) for CLASSIFY.
"""

from functools import wraps
import os
import uuid
from flask import Blueprint, request, jsonify, current_app, g
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature
from werkzeug.utils import secure_filename
from classify.extensions import db
from classify.models import User, Analysis, Song, Prediction, AudioFeatures, Genre, Subgenre
from classify.services.analysis_service import analyze_uploaded_audio

api_bp = Blueprint("api", __name__, url_prefix="/api/v1")

TOKEN_SALT = "api-bearer-auth"
DEFAULT_TOKEN_EXPIRY = 86400  # 24 hours


def generate_token(user_id: int, expires_in: int = DEFAULT_TOKEN_EXPIRY) -> str:
    """Generates a cryptographically signed HMAC bearer token for an API user."""
    serializer = URLSafeTimedSerializer(current_app.config["SECRET_KEY"], salt=TOKEN_SALT)
    return serializer.dumps({"sub": user_id})


def verify_token(token: str, max_age: int = DEFAULT_TOKEN_EXPIRY) -> int:
    """Verifies bearer token and returns user_id, or raises ValueError."""
    serializer = URLSafeTimedSerializer(current_app.config["SECRET_KEY"], salt=TOKEN_SALT)
    data = serializer.loads(token, max_age=max_age)
    return data["sub"]


def token_required(f):
    """Decorator to require a valid Bearer token for API endpoints."""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header or not auth_header.startswith("Bearer "):
            return jsonify({
                "success": False,
                "error": "Missing or malformed Authorization header. Expected: Bearer <token>",
            }), 401

        token = auth_header.split(" ", 1)[1].strip()
        try:
            user_id = verify_token(token)
            user = db.session.get(User, user_id)
            if not user:
                return jsonify({"success": False, "error": "User account no longer exists."}), 401
            g.current_api_user = user
        except SignatureExpired:
            return jsonify({"success": False, "error": "Token has expired. Please re-authenticate."}), 401
        except (BadSignature, Exception):
            return jsonify({"success": False, "error": "Invalid token signature."}), 401

        return f(*args, **kwargs)
    return decorated


@api_bp.route("/auth/token", methods=["POST"])
def get_token():
    """Authenticates user credentials and issues a Bearer JWT/token for REST API."""
    data = request.get_json(silent=True) or request.form
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not email or not password:
        return jsonify({"success": False, "error": "Email and password are required."}), 400

    user = User.query.filter_by(email=email).first()
    if not user or not user.check_password(password):
        return jsonify({"success": False, "error": "Invalid email or password."}), 401

    token = generate_token(user.id)
    return jsonify({
        "success": True,
        "access_token": token,
        "token_type": "Bearer",
        "expires_in": DEFAULT_TOKEN_EXPIRY,
        "user": {
            "id": user.id,
            "email": user.email,
            "role": user.role,
        },
    }), 200


@api_bp.route("/genres", methods=["GET"])
def list_genres():
    """Lists all available genres and subgenres in the conservatory taxonomy."""
    genres = Genre.query.order_by(Genre.name).all()
    results = []
    for g_obj in genres:
        subgenres = [{"id": s.id, "name": s.name, "slug": s.slug, "description": s.description} for s in g_obj.subgenres]
        results.append({
            "id": g_obj.id,
            "name": g_obj.name,
            "slug": g_obj.slug,
            "description": g_obj.description,
            "subgenres": subgenres,
        })
    return jsonify({"success": True, "genres": results, "count": len(results)}), 200


@api_bp.route("/audio/analyze", methods=["POST"])
@token_required
def api_analyze_audio():
    """Uploads an audio file and runs classification + acoustic feature extraction."""
    file = request.files.get("audio_file") or request.files.get("audio") or request.files.get("file")
    if not file or not file.filename:
        return jsonify({"success": False, "error": "No audio file provided in request."}), 400

    original_filename = secure_filename(file.filename) or "recording.wav"
    temp_ext = os.path.splitext(original_filename)[1].lower()
    unique_stored_name = f"{uuid.uuid4().hex}_{original_filename}"
    upload_path = os.path.join(current_app.config["UPLOAD_FOLDER"], unique_stored_name)

    file.save(upload_path)

    user = g.current_api_user

    try:
        result = analyze_uploaded_audio(
            file_path=upload_path,
            original_filename=original_filename,
            stored_filename=unique_stored_name,
            user=user,
        )

        analysis_id = result.get("analysis_id")
        song_id = result.get("song_id")
        features = result.get("features", {})
        classification = result.get("classification", {})

        return jsonify({
            "success": True,
            "analysis_id": analysis_id,
            "song": {
                "id": song_id,
                "original_filename": original_filename,
                "duration": features.get("duration"),
            },
            "prediction": {
                "genre": classification.get("predicted_genre", "Unknown"),
                "subgenre": classification.get("subgenre_name"),
                "confidence": classification.get("confidence"),
                "alternatives": classification.get("alternatives", []),
                "explanation": classification.get("explanation"),
            },
            "acoustic_features": {
                "detected_key": features.get("detected_key"),
                "time_signature": features.get("time_signature", "4/4"),
                "chord_progression": features.get("progression_str"),
                "vocal_presence": features.get("vocal_presence"),
                "tempo": features.get("tempo"),
                "rms_energy": features.get("rms_mean"),
                "spectral_centroid": features.get("spectral_centroid"),
                "spectral_bandwidth": features.get("spectral_bandwidth_mean"),
                "spectral_rolloff": features.get("spectral_rolloff_mean"),
            },
        }), 201
    except Exception as e:
        if os.path.exists(upload_path):
            try:
                os.remove(upload_path)
            except Exception:
                pass
        return jsonify({"success": False, "error": f"Analysis failed: {str(e)}"}), 500


@api_bp.route("/analyses", methods=["GET"])
@token_required
def list_analyses():
    """Lists analyses with pagination."""
    user = g.current_api_user
    limit = min(request.args.get("limit", 20, type=int), 100)
    offset = request.args.get("offset", 0, type=int)

    query = Analysis.query.join(Song)
    if user.is_student() and user.student_profile:
        query = query.filter(Song.student_id == user.student_profile.id)

    total = query.count()
    analyses = query.order_by(Analysis.analyzed_at.desc()).offset(offset).limit(limit).all()

    items = []
    for a in analyses:
        pred = a.prediction
        af = a.audio_features
        items.append({
            "id": a.id,
            "filename": a.song.original_filename if a.song else None,
            "analyzed_at": a.analyzed_at.isoformat(),
            "genre": pred.genre.name if pred and pred.genre else None,
            "confidence": pred.confidence if pred else None,
            "detected_key": af.detected_key if af else None,
            "tempo": af.tempo if af else None,
        })

    return jsonify({
        "success": True,
        "total": total,
        "limit": limit,
        "offset": offset,
        "analyses": items,
    }), 200


@api_bp.route("/analyses/<int:analysis_id>", methods=["GET"])
@token_required
def get_analysis_detail(analysis_id):
    """Retrieves full details of an analysis by ID with IDOR protection."""
    analysis = db.session.get(Analysis, analysis_id)
    if not analysis:
        return jsonify({"success": False, "error": "Analysis not found."}), 404

    user = g.current_api_user
    if user.is_student() and user.student_profile:
        if analysis.song and analysis.song.student_id != user.student_profile.id:
            return jsonify({"success": False, "error": "Access denied: unauthorized access."}), 403

    af = analysis.audio_features
    pred = analysis.prediction

    return jsonify({
        "success": True,
        "id": analysis.id,
        "analyzed_at": analysis.analyzed_at.isoformat(),
        "song": {
            "id": analysis.song.id if analysis.song else None,
            "original_filename": analysis.song.original_filename if analysis.song else None,
            "duration": af.duration if af else None,
        },
        "prediction": {
            "genre": pred.genre.name if pred and pred.genre else None,
            "subgenre": pred.subgenre.name if pred and pred.subgenre else None,
            "confidence": pred.confidence if pred else None,
            "explanation": pred.explanation if pred else None,
            "alternatives": pred.alternatives if pred else [],
        },
        "acoustic_features": {
            "detected_key": af.detected_key if af else None,
            "time_signature": af.time_signature if af else None,
            "chord_progression": af.chord_progression if af else None,
            "vocal_presence": af.vocal_presence if af else None,
            "tempo": af.tempo if af else None,
            "rms_mean": af.rms_mean if af else None,
            "zcr_mean": af.zcr_mean if af else None,
            "spectral_centroid": af.spectral_centroid if af else None,
            "spectral_bandwidth": af.spectral_bandwidth if af else None,
            "spectral_rolloff": af.spectral_rolloff if af else None,
        },
    }), 200


@api_bp.route("/openapi.json", methods=["GET"])
def openapi_spec():
    """Returns OpenAPI 3.0.3 specification for LMS / third-party integrations."""
    spec = {
        "openapi": "3.0.3",
        "info": {
            "title": "CLASSIFY Conservatory Music Analysis API",
            "version": "1.0.0",
            "description": "RESTful API for automated musical genre classification, acoustic feature extraction, and LMS integration.",
        },
        "servers": [{"url": "/api/v1", "description": "Current Environment API Base"}],
        "paths": {
            "/auth/token": {
                "post": {
                    "summary": "Obtain Bearer token",
                    "description": "Authenticate using email and password to receive an API access token.",
                    "responses": {"200": {"description": "Token issued successfully"}, "401": {"description": "Invalid credentials"}},
                }
            },
            "/genres": {
                "get": {
                    "summary": "List genre taxonomy",
                    "description": "Retrieve all recognized musical genres and subgenres.",
                    "responses": {"200": {"description": "Genre catalog list"}},
                }
            },
            "/audio/analyze": {
                "post": {
                    "summary": "Upload and classify audio",
                    "description": "Upload a WAV/MP3 track to classify its genre, key, time signature, and chords.",
                    "security": [{"BearerAuth": []}],
                    "responses": {"201": {"description": "Analysis generated"}, "400": {"description": "Missing file"}, "401": {"description": "Unauthorized"}},
                }
            },
            "/analyses": {
                "get": {
                    "summary": "List analyses",
                    "description": "List analysis history for the authenticated user.",
                    "security": [{"BearerAuth": []}],
                    "responses": {"200": {"description": "List of analyses"}},
                }
            },
            "/analyses/{analysis_id}": {
                "get": {
                    "summary": "Get analysis details",
                    "description": "Retrieve comprehensive analysis details, prediction, and features.",
                    "security": [{"BearerAuth": []}],
                    "responses": {"200": {"description": "Analysis details"}, "404": {"description": "Not found"}},
                }
            },
        },
        "components": {
            "securitySchemes": {
                "BearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "bearerFormat": "JWT",
                }
            }
        },
    }
    return jsonify(spec), 200
