from .auth import auth_bp
from .analysis import analysis_bp
from .student import student_bp
from .teacher import teacher_bp
from .library import library_bp
from .explore import explore_bp
from .api import api_bp

__all__ = [
    "auth_bp",
    "analysis_bp",
    "student_bp",
    "teacher_bp",
    "library_bp",
    "explore_bp",
    "api_bp",
]
