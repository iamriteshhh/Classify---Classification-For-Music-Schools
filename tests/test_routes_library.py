"""
test_routes_library.py
======================
Tests for the Genre Reference Library routes.
Covers:
- Catalog listing page (/library).
- Keyword search filtering.
- Detailed genre profile view (/library/<genre_slug>).
- Subgenre taxonomy rendering.
- 404 response on unknown genre slug.
Part of Phase 8 for CLASSIFY.
"""

from classify.models import Genre, Subgenre
from classify.extensions import db


def _ensure_genres_seeded(app):
    """Helper to ensure seed data is available in test context."""
    with app.app_context():
        if Genre.query.count() == 0:
            from classify.models.seed import seed_genres
            seed_genres(app)


def test_library_catalog_loads(client, app):
    """Test that /library loads successfully and lists seeded genres."""
    _ensure_genres_seeded(app)

    res = client.get("/library")
    assert res.status_code == 200
    assert b"Genre Reference Library" in res.data
    assert b"Rock" in res.data
    assert b"Classical" in res.data
    assert b"Jazz" in res.data
    assert b"Subgenres" in res.data


def test_library_catalog_search(client, app):
    """Test searching the library catalog by keyword."""
    _ensure_genres_seeded(app)

    # Search for "Jazz"
    res_jazz = client.get("/library?q=Jazz")
    assert res_jazz.status_code == 200
    assert b"Jazz" in res_jazz.data

    # Search for an obscure non-matching term
    res_none = client.get("/library?q=xyznonexistentterm999")
    assert res_none.status_code == 200
    assert b"Rock" not in res_none.data


def test_library_genre_detail_success(client, app):
    """Test that /library/<genre_slug> returns detailed genre profile."""
    _ensure_genres_seeded(app)

    res = client.get("/library/rock")
    assert res.status_code == 200
    assert b"Rock" in res.data
    assert b"Historical Lineage" in res.data
    assert b"Structural Forms" in res.data
    assert b"Core Instrumentation" in res.data
    assert b"Subgenre Taxonomy" in res.data
    assert b"Classic Rock" in res.data


def test_library_genre_detail_not_found(client, app):
    """Test that requesting a non-existent genre returns 404."""
    _ensure_genres_seeded(app)

    res = client.get("/library/unknown-space-genre-404")
    assert res.status_code == 404
