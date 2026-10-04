"""
teacher.py
==========
Teacher dashboard and review routes.
Provides submission queues, filtering, pedagogical review interface,
and teaching topic recommendations for instructors.
Part of Phase 7 for CLASSIFY.
"""

from datetime import datetime, timezone
from flask import Blueprint, render_template, request, redirect, url_for, flash, abort
from flask_login import current_user
from classify.auth_utils import role_required
from classify.extensions import db
from classify.models import (
    Analysis,
    Song,
    Student,
    Teacher,
    User,
    Genre,
    TeacherReview,
    Prediction,
    AudioFeatures,
    Classroom,
    Enrollment,
    Assignment,
    Submission,
    Notification,
)
from classify.services.teaching_topics_service import get_teaching_topics

teacher_bp = Blueprint("teacher", __name__, url_prefix="/teacher")


@teacher_bp.route("/dashboard", methods=["GET"])
@role_required("teacher")
def dashboard():
    """
    Teacher dashboard displaying all student submissions with filtering
    by genre, review status (all/pending/reviewed), student email, and classroom cohort.
    """
    selected_status = request.args.get("status", "all").strip().lower()
    selected_genre = request.args.get("genre", "all").strip()
    student_filter = request.args.get("student", "").strip().lower()
    classroom_id = request.args.get("classroom_id", type=int)

    # Query analyses submitted by students
    query = (
        Analysis.query.join(Song, Analysis.song_id == Song.id)
        .join(Student, Song.student_id == Student.id)
        .join(User, Student.user_id == User.id)
        .outerjoin(Prediction, Prediction.analysis_id == Analysis.id)
        .outerjoin(Genre, Prediction.genre_id == Genre.id)
        .outerjoin(TeacherReview, TeacherReview.analysis_id == Analysis.id)
    )

    # Filter by classroom enrollment (Phase 3 Task 3.3)
    if classroom_id:
        query = query.join(Enrollment, Enrollment.student_id == Student.id).filter(
            Enrollment.classroom_id == classroom_id
        )

    # Filter by student email
    if student_filter:
        query = query.filter(User.email.ilike(f"%{student_filter}%"))

    # Filter by genre
    if selected_genre and selected_genre != "all":
        query = query.filter(Genre.name == selected_genre)

    # Filter by review status
    if selected_status == "pending":
        query = query.filter(TeacherReview.id.is_(None))
    elif selected_status == "reviewed":
        query = query.filter(TeacherReview.id.isnot(None))

    submissions = query.order_by(Analysis.analyzed_at.desc()).all()

    # Calculate aggregate metrics across all student analyses
    all_student_analyses = (
        Analysis.query.join(Song, Analysis.song_id == Song.id)
        .filter(Song.student_id.isnot(None))
        .all()
    )
    total_submissions = len(all_student_analyses)
    reviewed_count = sum(1 for a in all_student_analyses if a.teacher_review is not None)
    pending_count = total_submissions - reviewed_count

    # Populate genre filter list from DB
    available_genres = Genre.query.order_by(Genre.name).all()
    teacher = current_user.teacher_profile
    teacher_classrooms = Classroom.query.filter_by(teacher_id=teacher.id).all() if teacher else []

    return render_template(
        "teacher/dashboard.html",
        submissions=submissions,
        total_submissions=total_submissions,
        reviewed_count=reviewed_count,
        pending_count=pending_count,
        selected_status=selected_status,
        selected_genre=selected_genre,
        selected_classroom=classroom_id,
        classrooms=teacher_classrooms,
        student_filter=student_filter,
        available_genres=available_genres,
    )


@teacher_bp.route("/review/<int:analysis_id>", methods=["GET", "POST"])
@role_required("teacher")
def review(analysis_id):
    """
    Teacher review interface:
    - Audio playback and Tier 1 acoustic characteristics.
    - Model prediction, confidence, and alternative candidates.
    - Curated pedagogical teaching recommendations.
    - Review submission (agreement, verified genre correction, notes).
    """
    analysis = db.get_or_404(Analysis, analysis_id)
    song = analysis.song
    prediction = analysis.prediction
    audio_features = analysis.audio_features
    existing_review = analysis.teacher_review

    # Fetch teaching topics based on model prediction and features
    pred_genre_name = prediction.genre.name if (prediction and prediction.genre) else "Rock"
    teaching_guide = get_teaching_topics(pred_genre_name, audio_features)

    # All taxonomy genres from DB for verified selection
    available_genres = Genre.query.order_by(Genre.name).all()

    if request.method == "POST":
        agrees_raw = request.form.get("agrees_with_model", "false")
        agrees_with_model = agrees_raw in {"true", "1", "yes", "on"}
        teacher_genre = request.form.get("teacher_genre", pred_genre_name).strip()
        comment = request.form.get("comment", "").strip()

        teacher_id = current_user.teacher_profile.id if current_user.teacher_profile else None

        if existing_review:
            # Update existing review
            existing_review.agrees_with_model = agrees_with_model
            existing_review.teacher_genre = teacher_genre
            existing_review.comment = comment
            existing_review.reviewed_at = datetime.now(timezone.utc)
            if teacher_id:
                existing_review.teacher_id = teacher_id
        else:
            # Create new review record without altering model prediction
            new_review = TeacherReview(
                analysis_id=analysis.id,
                teacher_id=teacher_id,
                agrees_with_model=agrees_with_model,
                teacher_genre=teacher_genre,
                comment=comment,
            )
            db.session.add(new_review)

        # Notify student of review (Phase 3 Task 3.5)
        if song and song.student and song.student.user_id:
            notif = Notification(
                user_id=song.student.user_id,
                message=f"Your instructor reviewed '{song.original_filename}'.",
                link=url_for("analysis.view_analysis", analysis_id=analysis.id),
            )
            db.session.add(notif)

        db.session.commit()
        flash(f"Teacher review saved for '{song.original_filename}'.", "success")
        return redirect(url_for("teacher.dashboard"))

    return render_template(
        "teacher/review.html",
        analysis=analysis,
        song=song,
        prediction=prediction,
        audio_features=audio_features,
        review=existing_review,
        teaching_guide=teaching_guide,
        available_genres=available_genres,
    )


@teacher_bp.route("/classrooms", methods=["GET", "POST"])
@role_required("teacher")
def classrooms():
    """Classroom management: list cohorts, create new classrooms (Phase 3 Task 3.3)."""
    teacher = current_user.teacher_profile
    if not teacher:
        abort(403)

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        semester = request.form.get("semester", "").strip()
        if name:
            new_classroom = Classroom(name=name, teacher_id=teacher.id, semester=semester)
            db.session.add(new_classroom)
            db.session.commit()
            flash(f"Classroom '{name}' created successfully.", "success")
            return redirect(url_for("teacher.classrooms"))
        flash("Classroom name cannot be empty.", "error")

    all_classrooms = Classroom.query.filter_by(teacher_id=teacher.id).order_by(Classroom.created_at.desc()).all()
    return render_template("teacher/classrooms.html", classrooms=all_classrooms)


@teacher_bp.route("/classrooms/<int:classroom_id>/enroll", methods=["POST"])
@role_required("teacher")
def enroll_student(classroom_id):
    """Enrolls an existing student into the teacher's classroom by email."""
    classroom = db.get_or_404(Classroom, classroom_id)
    if not current_user.teacher_profile or classroom.teacher_id != current_user.teacher_profile.id:
        abort(403)

    email = request.form.get("email", "").strip().lower()
    user = User.query.filter_by(email=email).first()
    if not user or not user.student_profile:
        flash(f"No student account found with email '{email}'.", "error")
        return redirect(url_for("teacher.classrooms"))

    existing = Enrollment.query.filter_by(classroom_id=classroom.id, student_id=user.student_profile.id).first()
    if existing:
        flash(f"Student '{email}' is already enrolled in '{classroom.name}'.", "info")
    else:
        enrollment = Enrollment(classroom_id=classroom.id, student_id=user.student_profile.id)
        db.session.add(enrollment)
        notif = Notification(
            user_id=user.id,
            message=f"You have been enrolled in classroom '{classroom.name}'.",
            link=url_for("student.dashboard"),
        )
        db.session.add(notif)
        db.session.commit()
        flash(f"Student '{email}' enrolled successfully into '{classroom.name}'.", "success")

    return redirect(url_for("teacher.classrooms"))


@teacher_bp.route("/classrooms/<int:classroom_id>/assignments", methods=["POST"])
@role_required("teacher")
def create_assignment(classroom_id):
    """Publishes a new practice assignment to a classroom (Phase 3 Task 3.4)."""
    classroom = db.get_or_404(Classroom, classroom_id)
    if not current_user.teacher_profile or classroom.teacher_id != current_user.teacher_profile.id:
        abort(403)

    title = request.form.get("title", "").strip()
    description = request.form.get("description", "").strip()
    genre_focus = request.form.get("genre_focus", "").strip()
    due_date_str = request.form.get("due_date", "").strip()

    due_date = None
    if due_date_str:
        try:
            due_date = datetime.fromisoformat(due_date_str)
        except ValueError:
            pass

    if title:
        assignment = Assignment(
            classroom_id=classroom.id,
            title=title,
            description=description,
            genre_focus=genre_focus,
            due_date=due_date,
        )
        db.session.add(assignment)
        db.session.flush()

        # Notify enrolled students
        for enrollment in classroom.enrollments:
            if enrollment.student and enrollment.student.user_id:
                notif = Notification(
                    user_id=enrollment.student.user_id,
                    message=f"New assignment in {classroom.name}: '{title}'.",
                    link=url_for("student.dashboard"),
                )
                db.session.add(notif)

        db.session.commit()
        flash(f"Assignment '{title}' published to '{classroom.name}'.", "success")
    else:
        flash("Assignment title is required.", "error")

    return redirect(url_for("teacher.classrooms"))


@teacher_bp.route("/assignments/<int:assignment_id>", methods=["GET"])
@role_required("teacher")
def view_assignment(assignment_id):
    """Inspects submissions for a classroom assignment."""
    assignment = db.get_or_404(Assignment, assignment_id)
    if not current_user.teacher_profile or assignment.classroom.teacher_id != current_user.teacher_profile.id:
        abort(403)
    return render_template("teacher/assignment_detail.html", assignment=assignment)

