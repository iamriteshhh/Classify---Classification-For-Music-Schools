"""
audio_features.py
=================
AudioFeatures model storing Tier 1 physical & acoustic measurements for an Analysis.
Part of Phase 4 for CLASSIFY.
"""

from classify.extensions import db


class AudioFeatures(db.Model):
    __tablename__ = "audio_features"

    id = db.Column(db.Integer, primary_key=True)
    analysis_id = db.Column(
        db.Integer, db.ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False, unique=True
    )

    # Signal & Dynamics
    duration = db.Column(db.Float, nullable=False)
    sample_rate = db.Column(db.Integer, nullable=False)
    channels = db.Column(db.Integer, nullable=False, default=1)
    tempo = db.Column(db.Float, nullable=False)
    rms_mean = db.Column(db.Float, nullable=False)
    rms_var = db.Column(db.Float, nullable=True)
    zcr_mean = db.Column(db.Float, nullable=False)
    zcr_var = db.Column(db.Float, nullable=True)

    # Spectral Descriptors
    spectral_centroid = db.Column(db.Float, nullable=False)
    spectral_bandwidth = db.Column(db.Float, nullable=False)
    spectral_rolloff = db.Column(db.Float, nullable=False)
    spectral_contrast = db.Column(db.Float, nullable=True)

    # Vectors stored as JSON
    mfcc_vector = db.Column(db.JSON, nullable=True)     # 20 MFCC means
    chroma_vector = db.Column(db.JSON, nullable=True)   # 12 Chroma means

    # Rhythm
    onset_rate = db.Column(db.Float, nullable=True)
    onset_strength_mean = db.Column(db.Float, nullable=True)

    # Musical Key (Phase 3 Task 3.2)
    detected_key = db.Column(db.String(32), nullable=True)

    # Audio Analysis Depth (Phase 4 Tasks 4.1, 4.2, 4.3)
    time_signature = db.Column(db.String(16), nullable=True, default="4/4")
    chord_progression = db.Column(db.String(120), nullable=True)
    vocal_presence = db.Column(db.String(24), nullable=True)

    # Relationships
    analysis = db.relationship("Analysis", back_populates="audio_features")

    def __repr__(self):
        return f"<AudioFeatures id={self.id} analysis_id={self.analysis_id} tempo={self.tempo}>"
