"""Submission model linking student song recordings to assignments."""

from datetime import datetime, timezone
from classify.extensions import db


class Submission(db.Model):
    __tablename__ = "submissions"

    id = db.Column(db.Integer, primary_key=True)
    assignment_id = db.Column(
        db.Integer, db.ForeignKey("assignments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    student_id = db.Column(
        db.Integer, db.ForeignKey("students.id", ondelete="CASCADE"), nullable=False, index=True
    )
    song_id = db.Column(
        db.Integer, db.ForeignKey("songs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    submitted_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    grade = db.Column(db.String(20), nullable=True)
    notes = db.Column(db.Text, nullable=True)

    # Relationships
    assignment = db.relationship("Assignment", back_populates="submissions")
    student = db.relationship("Student", backref=db.backref("submissions", cascade="all, delete-orphan"))
    song = db.relationship("Song")

    def __repr__(self):
        return f"<Submission id={self.id} assignment_id={self.assignment_id} song_id={self.song_id}>"
