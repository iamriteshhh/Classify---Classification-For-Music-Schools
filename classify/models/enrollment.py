"""Enrollment model mapping students to classrooms."""

from datetime import datetime, timezone
from classify.extensions import db


class Enrollment(db.Model):
    __tablename__ = "enrollments"

    id = db.Column(db.Integer, primary_key=True)
    classroom_id = db.Column(
        db.Integer, db.ForeignKey("classrooms.id", ondelete="CASCADE"), nullable=False, index=True
    )
    student_id = db.Column(
        db.Integer, db.ForeignKey("students.id", ondelete="CASCADE"), nullable=False, index=True
    )
    enrolled_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    classroom = db.relationship("Classroom", back_populates="enrollments")
    student = db.relationship("Student", backref=db.backref("enrollments", cascade="all, delete-orphan"))

    def __repr__(self):
        return f"<Enrollment classroom_id={self.classroom_id} student_id={self.student_id}>"
