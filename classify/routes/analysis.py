"""
analysis.py
===========
Audio upload, feature extraction, and analysis reporting routes.
Routes through the service layer (classify.services.analysis_service)
and integrates the trained ML classification pipeline.
Part of Phase 2 for CLASSIFY.
"""

import os
import uuid
import logging
from flask import (
    Blueprint,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_from_directory,
    current_app,
    jsonify,
)
from flask_login import current_user
from werkzeug.utils import secure_filename

from classify.audio.validation import AudioValidationError
from classify.services.analysis_service import analyze_uploaded_audio

logger = logging.getLogger(__name__)

analysis_bp = Blueprint("analysis", __name__)


def is_ajax_request():
    """Detects AJAX/fetch requests from the client."""
    return request.headers.get("X-Requested-With") == "XMLHttpRequest" or (
        request.accept_mimetypes.accept_json and not request.accept_mimetypes.accept_html
    )


@analysis_bp.context_processor
def inject_sidebar_data():
    """Provides Genre Library and Student/Teacher card data to the interface template."""
    from classify.models import Genre
    try:
        genres = Genre.query.order_by(Genre.name).all()
    except Exception:
        genres = []

    user_stats = None
    if current_user.is_authenticated:
        if current_user.is_student() and current_user.student_profile:
            from classify.models import Analysis, Song
            analyses = (
                Analysis.query.join(Song)
                .filter(Song.student_id == current_user.student_profile.id)
                .all()
            )
            total = len(analyses)
            from collections import Counter
            counts = Counter(
                a.prediction.genre.name
                for a in analyses
                if a.prediction and a.prediction.genre
            )
            most_g = counts.most_common(1)[0][0] if counts else "None yet"
            user_stats = {
                "total_songs": total,
                "most_analyzed_genre": most_g,
                "role": "student",
            }
        elif current_user.is_teacher():
            user_stats = {
                "role": "teacher",
                "title": "Music Instructor",
            }

    return {
        "sidebar_genres": genres,
        "sidebar_user_stats": user_stats,
    }


def allowed_file(filename):
    """Checks if the uploaded file has an allowed audio extension."""
    allowed = current_app.config.get("ALLOWED_AUDIO_EXTENSIONS", {"wav", "mp3"})
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed


@analysis_bp.route("/", methods=["GET"])
def index():
    """Renders the main upload & dashboard page."""
    return render_template("index.html")


@analysis_bp.route("/analyze", methods=["POST"])
def analyze():
    """
    Receives uploaded audio file, validates integrity, extracts Tier 1 features,
    and calculates trained ML classification with confidence and explanations.
    Uses collision-safe UUID-based storage.
    """
    # 1. Check if file is present in request
    if "audio_file" not in request.files:
        if is_ajax_request():
            return jsonify({"success": False, "error": "No audio file part was provided in the request."}), 400
        flash("No audio file part was provided in the request.", "error")
        return redirect(url_for("analysis.index"))

    file = request.files["audio_file"]

    # 2. Check if user submitted an empty file selector
    if file.filename == "":
        if is_ajax_request():
            return jsonify({"success": False, "error": "Please select an audio file before clicking Analyze."}), 400
        flash("Please select an audio file before clicking Analyze.", "error")
        return redirect(url_for("analysis.index"))

    # 3. Check allowed extension (.wav, .mp3)
    if not allowed_file(file.filename):
        if is_ajax_request():
            return jsonify({"success": False, "error": "Invalid file format. Please upload a .wav or .mp3 audio file."}), 400
        flash("Invalid file format. Please upload a .wav or .mp3 audio file.", "error")
        return redirect(url_for("analysis.index"))

    # 3b. Check Student Storage Quota (Phase 2 Task 2.3)
    if current_user.is_authenticated and current_user.is_student() and current_user.student_profile:
        quota_mb = current_app.config.get("UPLOAD_QUOTA_MB", 500)
        from classify.models import Song
        from classify.extensions import db
        used_kb = (
            db.session.query(db.func.sum(Song.file_size_kb))
            .filter(Song.student_id == current_user.student_profile.id)
            .scalar()
            or 0.0
        )
        file.seek(0, os.SEEK_END)
        incoming_bytes = file.tell()
        file.seek(0)
        incoming_kb = incoming_bytes / 1024.0

        if (used_kb + incoming_kb) > (quota_mb * 1024.0):
            used_mb = round(used_kb / 1024.0, 1)
            err_msg = f"Storage quota exceeded ({used_mb} MB used of {quota_mb} MB limit). Please remove older recordings before uploading."
            if is_ajax_request():
                return jsonify({"success": False, "error": err_msg}), 400
            flash(err_msg, "error")
            return redirect(url_for("analysis.index"))

    stored_filename = None
    file_path = None

    try:
        # 4. Save file safely using collision-resistant UUID naming
        original_filename = secure_filename(file.filename)
        stored_filename = f"{uuid.uuid4().hex}_{original_filename}"
        upload_folder = current_app.config["UPLOAD_FOLDER"]
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, stored_filename)
        file.save(file_path)

        # 4b. Check for Async processing request (Phase 2 Task 2.1)
        is_async = (
            request.args.get("async") == "1"
            or request.form.get("async") == "1"
            or request.headers.get("X-Async-Processing") == "1"
        )

        if is_async:
            from classify.tasks import submit_analysis_task
            task_id = submit_analysis_task(
                current_app,
                file_path=file_path,
                original_filename=original_filename,
                stored_filename=stored_filename,
                user=current_user,
                allowed_extensions=current_app.config.get("ALLOWED_AUDIO_EXTENSIONS"),
                max_size_bytes=current_app.config.get("MAX_CONTENT_LENGTH"),
            )
            return jsonify({
                "success": True,
                "async": True,
                "task_id": task_id,
                "status_url": url_for("analysis.task_status", task_id=task_id),
                "events_url": url_for("analysis.task_events", task_id=task_id),
                "audio_url": url_for("analysis.uploaded_file", filename=stored_filename),
            }), 202

        # 5. Execute Synchronous Analysis Workflow via Service Layer with Persistence
        analysis_result = analyze_uploaded_audio(
            file_path=file_path,
            original_filename=original_filename,
            stored_filename=stored_filename,
            user=current_user,
            allowed_extensions=current_app.config.get("ALLOWED_AUDIO_EXTENSIONS"),
            max_size_bytes=current_app.config.get("MAX_CONTENT_LENGTH"),
        )

        features = analysis_result["features"]
        features["file_name"] = original_filename
        classification = analysis_result["classification"]

        result = {
            "features": features,
            "classification": classification,
            "validation": analysis_result["validation"],
            "analysis_id": analysis_result["analysis_id"],
        }

        audio_url = url_for("analysis.uploaded_file", filename=stored_filename)

        if is_ajax_request():
            return jsonify({
                "success": True,
                "result": result,
                "audio_url": audio_url,
            })

        return render_template("index.html", result=result, audio_url=audio_url)

    except AudioValidationError as ave:
        # User-fixable audio validation errors: display clear, actionable message
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass
        if is_ajax_request():
            return jsonify({"success": False, "error": str(ave)}), 400
        flash(str(ave), "error")
        return redirect(url_for("analysis.index"))

    except Exception as e:
        logger.exception("Unexpected error during audio analysis:")
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass
        if is_ajax_request():
            return jsonify({"success": False, "error": f"Error analyzing audio: {str(e)}"}), 500
        flash(f"Error analyzing audio: {str(e)}", "error")
        return redirect(url_for("analysis.index"))


@analysis_bp.route("/analysis/<int:analysis_id>", methods=["GET"])
def view_analysis(analysis_id):
    """Retrieves and displays a previously persisted Music Analysis Report."""
    from classify.models import Analysis
    from classify.extensions import db

    analysis = db.session.get(Analysis, analysis_id)
    if not analysis:
        flash("Requested analysis report was not found.", "error")
        return redirect(url_for("analysis.index"))

    # IDOR authorization guard (security.md §4)
    if analysis.song and analysis.song.student_id:
        if not current_user.is_authenticated:
            return redirect(url_for("auth.login"))
        if current_user.is_student():
            if not current_user.student_profile or analysis.song.student_id != current_user.student_profile.id:
                from flask import abort
                abort(403, description="Access denied: you are not authorized to view another student's submission.")

    # Reconstruct features dict
    af = analysis.audio_features
    pred = analysis.prediction

    features = {
        "file_name": analysis.song.original_filename if analysis.song else "audio_file.wav",
        "duration": af.duration if af else 0.0,
        "sample_rate": af.sample_rate if af else 22050,
        "tempo": af.tempo if af else 120.0,
        "rms_mean": af.rms_mean if af else 0.0,
        "rms_var": af.rms_var if af else 0.0,
        "zcr_mean": af.zcr_mean if af else 0.0,
        "zcr_var": af.zcr_var if af else 0.0,
        "spectral_centroid": af.spectral_centroid if af else 0.0,
        "spectral_bandwidth_mean": af.spectral_bandwidth if af else 0.0,
        "spectral_rolloff_mean": af.spectral_rolloff if af else 0.0,
        "spectral_contrast_mean": af.spectral_contrast if af else 0.0,
        "mfcc_mean": round(float(sum(af.mfcc_vector) / len(af.mfcc_vector)), 2) if af and af.mfcc_vector else 0.0,
        "onset_rate": af.onset_rate if af else 0.0,
        "detected_key": af.detected_key if (af and af.detected_key) else "Unknown",
    }

    classification = {
        "predicted_genre": pred.genre.name if pred and pred.genre else "Unknown",
        "confidence": pred.confidence if pred else 0.0,
        "confidence_percent": round(pred.confidence * 100, 1) if pred else 0.0,
        "is_low_confidence": pred.is_low_confidence if pred else False,
        "probabilities": pred.probabilities if pred else {},
        "alternatives": pred.alternatives if pred else [],
        "explanation": pred.explanation if pred else "",
        "explanation_factors": [],
        "model_name": "Trained ML Classifier",
        "is_prototype": False,
    }

    result = {
        "features": features,
        "classification": classification,
        "analysis_id": analysis.id,
        "teacher_review": analysis.teacher_review,
    }

    audio_url = None
    if analysis.song and analysis.song.stored_filename:
        audio_url = url_for("analysis.uploaded_file", filename=analysis.song.stored_filename)

    return render_template("index.html", result=result, audio_url=audio_url)


@analysis_bp.route("/uploads/<path:filename>", methods=["GET"])
def uploaded_file(filename):
    """Serves the uploaded audio file for playback in the web player."""
    return send_from_directory(current_app.config["UPLOAD_FOLDER"], filename)


@analysis_bp.route("/analysis/status/<task_id>", methods=["GET"])
def task_status(task_id):
    """Retrieves current processing status of an async audio analysis task."""
    from classify.tasks import get_task_status
    status = get_task_status(task_id)
    if not status:
        return jsonify({"success": False, "error": "Task not found"}), 404

    # If task is completed and has stored_filename, augment with audio_url
    if status.get("status") == "completed" and status.get("result"):
        stored_fname = status["result"].get("stored_filename")
        if stored_fname:
            status["result"]["audio_url"] = url_for("analysis.uploaded_file", filename=stored_fname)
    return jsonify({"success": True, "task": status})


@analysis_bp.route("/analysis/events/<task_id>", methods=["GET"])
def task_events(task_id):
    """Server-Sent Events (SSE) stream for real-time progress updates."""
    from flask import Response
    from classify.tasks import stream_task_events
    return Response(stream_task_events(task_id), mimetype="text/event-stream")


@analysis_bp.route("/analysis/<int:analysis_id>/pdf", methods=["GET"])
def export_pdf(analysis_id):
    """Generates and streams an official conservatory PDF report for this analysis (Phase 3 Task 3.6)."""
    from flask import send_file, abort
    from classify.models import Analysis
    from classify.services.pdf_service import generate_analysis_pdf
    from classify.extensions import db

    analysis = db.session.get(Analysis, analysis_id)
    if not analysis:
        flash("Requested analysis report was not found.", "error")
        return redirect(url_for("analysis.index"))

    # IDOR authorization guard (security.md §4)
    if analysis.song and analysis.song.student_id:
        if not current_user.is_authenticated:
            return redirect(url_for("auth.login"))
        if current_user.is_student():
            if not current_user.student_profile or analysis.song.student_id != current_user.student_profile.id:
                abort(403, description="Access denied: you are not authorized to export another student's submission.")

    pdf_buffer = generate_analysis_pdf(analysis)
    track_name = analysis.song.original_filename if analysis.song else f"analysis_{analysis.id}"
    safe_name = "".join(c for c in track_name if c.isalnum() or c in ("-", "_", ".")).rstrip()
    download_filename = f"classify_report_{safe_name}.pdf"

    return send_file(
        pdf_buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=download_filename,
    )



