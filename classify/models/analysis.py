"""
analysis.py
===========
Analysis model representing an analysis event for an uploaded song.
Owns AudioFeatures, Prediction, and optional TeacherReview.
Part of Phase 4 for CLASSIFY.
"""

from datetime import datetime, timezone
from classify.extensions import db


class Analysis(db.Model):
    __tablename__ = "analyses"

    id = db.Column(db.Integer, primary_key=True)
    song_id = db.Column(
        db.Integer, db.ForeignKey("songs.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    analyzed_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships (1:1 with Song, AudioFeatures, Prediction; 1:0..1 with TeacherReview)
    song = db.relationship("Song", back_populates="analysis")
    audio_features = db.relationship(
        "AudioFeatures", back_populates="analysis", uselist=False, cascade="all, delete-orphan"
    )
    prediction = db.relationship(
        "Prediction", back_populates="analysis", uselist=False, cascade="all, delete-orphan"
    )
    teacher_review = db.relationship(
        "TeacherReview", back_populates="analysis", uselist=False, cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Analysis id={self.id} song_id={self.song_id}>"
