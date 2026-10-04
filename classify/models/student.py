"""
student.py
==========
Student profile model linked 1:1 with User where role='student'.
Part of Phase 4 for CLASSIFY.
"""

from classify.extensions import db


class Student(db.Model):
    __tablename__ = "students"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(
        db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, unique=True
    )

    # Relationships
    user = db.relationship("User", back_populates="student_profile")
    songs = db.relationship("Song", back_populates="student", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Student id={self.id} user_id={self.user_id}>"
