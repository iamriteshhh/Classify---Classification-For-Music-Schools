"""
test_routes_explore.py
======================
Tests for the Comparative Musical Exploration routes.
Covers:
- Landing page redirect behavior (/explore).
- Comparative view (/explore/compare/<genre_a>/<genre_b>).
- Dual genre data rendering (acoustic tendencies, structures, instruments).
- 404 response on unknown genre slugs in comparison.
- Identical genre comparison handling.
Part of Phase 9 for CLASSIFY.
"""

from classify.models import Genre
from classify.extensions import db


def _ensure_genres_seeded(app):
    """Helper to ensure seed data is available in test context."""
    with app.app_context():
        if Genre.query.count() == 0:
            from classify.models.seed import seed_genres
            seed_genres(app)


def test_explore_index_redirect(client, app):
    """Test that /explore redirects to comparison view."""
    _ensure_genres_seeded(app)

    # Default redirect
    res = client.get("/explore")
    assert res.status_code == 302
    assert "/explore/compare/" in res.headers["Location"]

    # Parameterized redirect
    res_param = client.get("/explore?g1=jazz&g2=blues")
    assert res_param.status_code == 302
    assert "/explore/compare/jazz/blues" in res_param.headers["Location"]


def test_explore_compare_success(client, app):
    """Test side-by-side comparison between two valid genres."""
    _ensure_genres_seeded(app)

    res = client.get("/explore/compare/rock/classical")
    assert res.status_code == 200
    assert b"Rock" in res.data
    assert b"Classical" in res.data
    assert b"Comparative Musical Exploration" in res.data
    assert b"Acoustic Tendencies" in res.data
    assert b"Structural Framework" in res.data


def test_explore_compare_not_found(client, app):
    """Test that comparison with invalid genre slug returns 404."""
    _ensure_genres_seeded(app)

    res = client.get("/explore/compare/rock/nonexistent-genre-slug")
    assert res.status_code == 404


def test_explore_compare_identical_genre(client, app):
    """Test comparing a genre with itself displays identical notice."""
    _ensure_genres_seeded(app)

    res = client.get("/explore/compare/rock/rock")
    assert res.status_code == 200
    assert b"Notice:</strong> You are viewing the same genre" in res.data
