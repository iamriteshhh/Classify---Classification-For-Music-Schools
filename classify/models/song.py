"""
song.py
=======
Song model representing uploaded audio tracks and file metadata.
Part of Phase 4 for CLASSIFY.
"""

from datetime import datetime, timezone
from classify.extensions import db


class Song(db.Model):
    __tablename__ = "songs"

    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(
        db.Integer, db.ForeignKey("students.id", ondelete="SET NULL"), nullable=True, index=True
    )
    stored_filename = db.Column(db.String(255), nullable=False, unique=True)
    original_filename = db.Column(db.String(255), nullable=False)
    duration = db.Column(db.Float, nullable=True)
    file_size_kb = db.Column(db.Float, nullable=True)
    uploaded_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    student = db.relationship("Student", back_populates="songs")
    analysis = db.relationship(
        "Analysis", back_populates="song", uselist=False, cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Song id={self.id} file={self.original_filename}>"
