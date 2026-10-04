"""
teacher.py
==========
Teacher profile model linked 1:1 with User where role='teacher'.
Part of Phase 4 for CLASSIFY.
"""

from classify.extensions import db


class Teacher(db.Model):
    __tablename__ = "teachers"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(
        db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, unique=True
    )

    # Relationships
    user = db.relationship("User", back_populates="teacher_profile")
    reviews = db.relationship("TeacherReview", back_populates="teacher", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Teacher id={self.id} user_id={self.user_id}>"
