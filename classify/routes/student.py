"""
student.py
==========
Student dashboard and history routes.
Displays total uploads, average tempo, personal genre distribution,
genuine community comparison, and recent analysis history.
Part of Phase 6 for CLASSIFY.
"""

from collections import Counter
from flask import Blueprint, render_template, request, redirect, url_for, flash, abort, jsonify
from flask_login import login_required, current_user
from classify.auth_utils import role_required
from classify.extensions import db
from classify.models import Analysis, Song, Prediction, Classroom, Enrollment, Assignment, Submission, Notification

student_bp = Blueprint("student", __name__, url_prefix="/student")


@student_bp.route("/dashboard", methods=["GET"])
@login_required
@role_required("student")
def dashboard():
    """Renders the personal student dashboard with cohort context & assignments."""
    student = current_user.student_profile
    if not student:
        abort(403)

    # Fetch all analyses owned by the current student
    analyses = (
        Analysis.query.join(Song)
        .filter(Song.student_id == student.id)
        .order_by(Analysis.analyzed_at.desc())
        .all()
    )

    total_songs = len(analyses)

    # Average tempo calculation
    tempo_sum = sum(
        a.audio_features.tempo for a in analyses if a.audio_features and a.audio_features.tempo
    )
    avg_tempo = round(tempo_sum / total_songs, 1) if total_songs > 0 else 0.0

    # Student genre distribution
    student_genre_counts = Counter(
        a.prediction.genre.name for a in analyses if a.prediction and a.prediction.genre
    )
    most_analyzed_genre = (
        student_genre_counts.most_common(1)[0][0] if student_genre_counts else "None yet"
    )

    student_genres = [
        {
            "genre": genre_name,
            "count": count,
            "percentage": round((count / total_songs) * 100, 1),
        }
        for genre_name, count in student_genre_counts.most_common()
    ]

    # Platform-wide Community Comparison (computed strictly from database records)
    all_predictions = Prediction.query.all()
    total_community = len(all_predictions)
    community_counts = Counter(
        p.genre.name for p in all_predictions if p.genre
    )

    all_genre_names = sorted(set(student_genre_counts.keys()) | set(community_counts.keys()))
    community_comparison = []
    for g_name in all_genre_names:
        s_pct = round((student_genre_counts.get(g_name, 0) / max(total_songs, 1)) * 100, 1) if total_songs > 0 else 0.0
        c_pct = round((community_counts.get(g_name, 0) / max(total_community, 1)) * 100, 1) if total_community > 0 else 0.0
        community_comparison.append({
            "genre": g_name,
            "student_percent": s_pct,
            "community_percent": c_pct,
            "diff": round(s_pct - c_pct, 1),
        })

    # Classroom cohorts and assignments (Phase 3 Task 3.3 & 3.4)
    enrolled_classrooms = [e.classroom for e in student.enrollments if e.classroom]
    classroom_ids = [c.id for c in enrolled_classrooms]

    pending_assignments = (
        Assignment.query.filter(Assignment.classroom_id.in_(classroom_ids))
        .order_by(Assignment.created_at.desc())
        .all()
        if classroom_ids
        else []
    )

    # Submissions lookup dict
    student_submissions = {sub.assignment_id: sub for sub in student.submissions}
    uploaded_songs = student.songs

    return render_template(
        "student/dashboard.html",
        total_songs=total_songs,
        avg_tempo=avg_tempo,
        most_analyzed_genre=most_analyzed_genre,
        student_genres=student_genres,
        community_comparison=community_comparison,
        analyses=analyses,
        classrooms=enrolled_classrooms,
        assignments=pending_assignments,
        submissions=student_submissions,
        uploaded_songs=uploaded_songs,
    )


@student_bp.route("/assignments/<int:assignment_id>/submit", methods=["POST"])
@login_required
@role_required("student")
def submit_assignment(assignment_id):
    """Submits a student's uploaded audio recording against an active assignment."""
    student = current_user.student_profile
    if not student:
        abort(403)

    assignment = db.get_or_404(Assignment, assignment_id)
    song_id = request.form.get("song_id", type=int)
    song = db.session.get(Song, song_id)

    if not song or song.student_id != student.id:
        flash("Please select a valid uploaded song to submit.", "error")
        return redirect(url_for("student.dashboard"))

    # Check if already submitted
    existing = Submission.query.filter_by(assignment_id=assignment.id, student_id=student.id).first()
    if existing:
        existing.song_id = song.id
        flash(f"Updated submission for assignment '{assignment.title}'.", "success")
    else:
        sub = Submission(assignment_id=assignment.id, student_id=student.id, song_id=song.id)
        db.session.add(sub)
        flash(f"Submitted '{song.original_filename}' for assignment '{assignment.title}'.", "success")

    # Notify instructor of submission
    if assignment.classroom and assignment.classroom.teacher and assignment.classroom.teacher.user_id:
        notif = Notification(
            user_id=assignment.classroom.teacher.user_id,
            message=f"{current_user.email} submitted a track for '{assignment.title}'.",
            link=url_for("teacher.view_assignment", assignment_id=assignment.id),
        )
        db.session.add(notif)

    db.session.commit()
    return redirect(url_for("student.dashboard"))


@student_bp.route("/notifications", methods=["GET"])
@login_required
def notifications():
    """Renders student notifications page and marks unread as read (Phase 3 Task 3.5)."""
    user_notifications = (
        Notification.query.filter_by(user_id=current_user.id)
        .order_by(Notification.created_at.desc())
        .all()
    )

    # Mark all as read
    for notif in user_notifications:
        notif.is_read = True
    db.session.commit()

    return render_template("student/notifications.html", notifications=user_notifications)


@student_bp.route("/notifications/<int:notification_id>/read", methods=["POST"])
@login_required
def mark_notification_read(notification_id):
    """Marks a single notification as read."""
    notif = db.get_or_404(Notification, notification_id)
    if notif.user_id != current_user.id:
        abort(403)
    notif.is_read = True
    db.session.commit()
    return jsonify({"success": True})

