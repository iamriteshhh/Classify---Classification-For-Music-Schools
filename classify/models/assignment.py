"""Assignment model for music school homework and repertoire practice."""

from datetime import datetime, timezone
from classify.extensions import db


class Assignment(db.Model):
    __tablename__ = "assignments"

    id = db.Column(db.Integer, primary_key=True)
    classroom_id = db.Column(
        db.Integer, db.ForeignKey("classrooms.id", ondelete="CASCADE"), nullable=False, index=True
    )
    title = db.Column(db.String(150), nullable=False)
    description = db.Column(db.Text, nullable=True)
    genre_focus = db.Column(db.String(100), nullable=True)
    due_date = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    classroom = db.relationship("Classroom", back_populates="assignments")
    submissions = db.relationship("Submission", back_populates="assignment", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Assignment id={self.id} title={self.title}>"
