"""
models package
==============
SQLAlchemy ORM definitions for CLASSIFY.
Part of Phase 4 for CLASSIFY.
"""

from .user import User
from .student import Student
from .teacher import Teacher
from .genre import Genre, Subgenre
from .song import Song
from .analysis import Analysis
from .audio_features import AudioFeatures
from .prediction import Prediction
from .teacher_review import TeacherReview
from .classroom import Classroom
from .enrollment import Enrollment
from .assignment import Assignment
from .submission import Submission
from .notification import Notification

__all__ = [
    "User",
    "Student",
    "Teacher",
    "Genre",
    "Subgenre",
    "Song",
    "Analysis",
    "AudioFeatures",
    "Prediction",
    "TeacherReview",
    "Classroom",
    "Enrollment",
    "Assignment",
    "Submission",
    "Notification",
]
