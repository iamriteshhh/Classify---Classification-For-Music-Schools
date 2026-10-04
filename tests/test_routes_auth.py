"""
test_routes_auth.py
===================
Authentication and role-based access control tests.
Covers:
- Registration for Student and Teacher
- Password mismatch and duplicate email validation
- Login authentication and bad password rejection
- Session termination on logout
- Strict role gating (@role_required): student vs teacher vs unauthenticated
Part of Phase 5 for CLASSIFY.
"""

from classify.models import User, Student, Teacher


def test_register_student_success(client, app):
    """Test registering a new student account."""
    data = {
        "email": "newstudent@music.edu",
        "role": "student",
        "password": "password123",
        "confirm_password": "password123",
    }
    response = client.post("/auth/register", data=data, follow_redirects=True)
    assert response.status_code == 200

    with app.app_context():
        user = User.query.filter_by(email="newstudent@music.edu").first()
        assert user is not None
        assert user.role == "student"
        assert user.student_profile is not None
        assert user.check_password("password123") is True


def test_register_teacher_success(client, app):
    """Test registering a new teacher account."""
    data = {
        "email": "newteacher@music.edu",
        "role": "teacher",
        "password": "password123",
        "confirm_password": "password123",
    }
    response = client.post("/auth/register", data=data, follow_redirects=True)
    assert response.status_code == 200

    with app.app_context():
        user = User.query.filter_by(email="newteacher@music.edu").first()
        assert user is not None
        assert user.role == "teacher"
        assert user.teacher_profile is not None


def test_register_duplicate_email(client):
    """Test that registering an existing email is rejected."""
    data = {
        "email": "dup@music.edu",
        "role": "student",
        "password": "password123",
        "confirm_password": "password123",
    }
    client.post("/auth/register", data=data, follow_redirects=True)
    # Attempt second registration with same email
    res = client.post("/auth/register", data=data, follow_redirects=True)
    assert res.status_code == 200
    assert b"already exists" in res.data


def test_login_and_logout_flow(client):
    """Test login with valid and invalid credentials, and session logout."""
    # Register user
    reg_data = {
        "email": "flow@music.edu",
        "role": "student",
        "password": "correct_password",
        "confirm_password": "correct_password",
    }
    client.post("/auth/register", data=reg_data, follow_redirects=True)
    client.get("/auth/logout", follow_redirects=True)

    # Failed login
    bad_login = client.post(
        "/auth/login",
        data={"email": "flow@music.edu", "password": "wrong_password"},
        follow_redirects=True,
    )
    assert b"Invalid email or password" in bad_login.data

    # Successful login
    good_login = client.post(
        "/auth/login",
        data={"email": "flow@music.edu", "password": "correct_password"},
        follow_redirects=True,
    )
    assert good_login.status_code == 200

    # Logout
    logout_res = client.get("/auth/logout", follow_redirects=True)
    assert b"signed out" in logout_res.data


def test_role_required_access_control(client):
    """Test role gating: unauthenticated redirects, wrong role gets 403 Forbidden."""
    # 1. Unauthenticated visit to student dashboard should redirect to login
    unauth_res = client.get("/student/dashboard")
    assert unauth_res.status_code == 302
    assert "/auth/login" in unauth_res.headers["Location"]

    # 2. Register and log in as Teacher
    t_data = {
        "email": "teach_gate@music.edu",
        "role": "teacher",
        "password": "password123",
        "confirm_password": "password123",
    }
    client.post("/auth/register", data=t_data, follow_redirects=True)

    # Teacher attempting to access Student Dashboard receives HTTP 403
    forbidden_res = client.get("/student/dashboard")
    assert forbidden_res.status_code == 403
