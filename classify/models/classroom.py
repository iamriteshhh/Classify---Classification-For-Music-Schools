"""Classroom model representing a music school cohort or section."""

from datetime import datetime, timezone
from classify.extensions import db


class Classroom(db.Model):
    __tablename__ = "classrooms"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    teacher_id = db.Column(
        db.Integer, db.ForeignKey("teachers.id", ondelete="CASCADE"), nullable=False, index=True
    )
    semester = db.Column(db.String(50), nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    teacher = db.relationship("Teacher", backref=db.backref("classrooms", cascade="all, delete-orphan"))
    enrollments = db.relationship("Enrollment", back_populates="classroom", cascade="all, delete-orphan")
    assignments = db.relationship("Assignment", back_populates="classroom", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Classroom id={self.id} name={self.name}>"
