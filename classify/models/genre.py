"""
genre.py
========
Genre and Subgenre models representing the data-driven taxonomy.
Stored in the database to satisfy Ground Rule 3 (no hard-coded taxonomy in route code).
Part of Phase 4 for CLASSIFY.
"""

from datetime import datetime, timezone
from classify.extensions import db


class Genre(db.Model):
    __tablename__ = "genres"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), unique=True, nullable=False, index=True)
    slug = db.Column(db.String(80), unique=True, nullable=False, index=True)
    description = db.Column(db.Text, nullable=False)
    history = db.Column(db.Text, nullable=True)
    typical_structures = db.Column(db.Text, nullable=True)
    common_instrumentation = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    subgenres = db.relationship(
        "Subgenre", back_populates="genre", cascade="all, delete-orphan", lazy="selectin"
    )
    predictions = db.relationship("Prediction", back_populates="genre")

    def __repr__(self):
        return f"<Genre {self.name}>"


class Subgenre(db.Model):
    __tablename__ = "subgenres"

    id = db.Column(db.Integer, primary_key=True)
    genre_id = db.Column(
        db.Integer, db.ForeignKey("genres.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name = db.Column(db.String(80), nullable=False)
    slug = db.Column(db.String(80), nullable=False)
    description = db.Column(db.Text, nullable=True)

    # Relationships
    genre = db.relationship("Genre", back_populates="subgenres")
    predictions = db.relationship("Prediction", back_populates="subgenre")

    def __repr__(self):
        return f"<Subgenre {self.name} (Genre: {self.genre_id})>"
