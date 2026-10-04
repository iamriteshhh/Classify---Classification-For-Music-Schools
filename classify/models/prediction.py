"""
prediction.py
=============
Prediction model representing ML classifier output for an Analysis.
References Genre (required) and Subgenre (optional). Stores confidence,
alternatives distribution, and transparent explanation.
Part of Phase 4 for CLASSIFY.
"""

from classify.extensions import db


class Prediction(db.Model):
    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True)
    analysis_id = db.Column(
        db.Integer, db.ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    genre_id = db.Column(
        db.Integer, db.ForeignKey("genres.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    subgenre_id = db.Column(
        db.Integer, db.ForeignKey("subgenres.id", ondelete="SET NULL"), nullable=True, index=True
    )
    cultural_tag = db.Column(db.String(80), nullable=True)
    confidence = db.Column(db.Float, nullable=False)
    probabilities = db.Column(db.JSON, nullable=True)
    alternatives = db.Column(db.JSON, nullable=True)
    explanation = db.Column(db.Text, nullable=True)
    is_low_confidence = db.Column(db.Boolean, default=False)

    # Relationships
    analysis = db.relationship("Analysis", back_populates="prediction")
    genre = db.relationship("Genre", back_populates="predictions")
    subgenre = db.relationship("Subgenre", back_populates="predictions")

    def __repr__(self):
        return f"<Prediction id={self.id} genre_id={self.genre_id} conf={self.confidence}>"
