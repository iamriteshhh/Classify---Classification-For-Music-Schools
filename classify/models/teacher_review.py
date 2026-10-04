"""
teacher_review.py
=================
TeacherReview model storing instructor feedback on an Analysis.
Preserves disagreement without altering the original Prediction record.
Part of Phase 4 for CLASSIFY.
"""

from datetime import datetime, timezone
from classify.extensions import db


class TeacherReview(db.Model):
    __tablename__ = "teacher_reviews"

    id = db.Column(db.Integer, primary_key=True)
    analysis_id = db.Column(
        db.Integer, db.ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    teacher_id = db.Column(
        db.Integer, db.ForeignKey("teachers.id", ondelete="CASCADE"), nullable=False, index=True
    )
    agrees_with_model = db.Column(db.Boolean, nullable=False, default=True)
    teacher_genre = db.Column(db.String(80), nullable=True)
    comment = db.Column(db.Text, nullable=True)
    reviewed_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    analysis = db.relationship("Analysis", back_populates="teacher_review")
    teacher = db.relationship("Teacher", back_populates="reviews")

    def __repr__(self):
        return (
            f"<TeacherReview id={self.id} analysis_id={self.analysis_id} "
            f"agrees={self.agrees_with_model}>"
        )
