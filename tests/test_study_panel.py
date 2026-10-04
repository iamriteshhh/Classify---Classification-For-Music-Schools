"""
test_study_panel.py
===================
Tests for the Genre Study Panel (final_design.md §3.3):
- Full page two-pane shell render
- AJAX fragment request (X-Requested-With: XMLHttpRequest and ?partial=1)
- Acoustic profile metrics in JetBrains Mono (.mono)
- Teaching topics checklist items
- Structural forms and core instrumentation
- Subgenre taxonomy items
- Comparative analysis links
- 404 response on unknown genre slugs
"""

import pytest
from classify.models import Genre, Subgenre
from classify.extensions import db


def _ensure_genres_seeded(app):
    """Helper to ensure seed data is available in test context."""
    with app.app_context():
        if Genre.query.count() == 0:
            from classify.models.seed import seed_genres
            seed_genres(app)


def test_study_panel_full_page_load(client, app):
    """Test direct full-page load of a genre study view in the two-pane shell."""
    _ensure_genres_seeded(app)

    res = client.get("/library/rock")
    assert res.status_code == 200
    # Two-pane layout shell elements
    assert b"app-shell" in res.data
    assert b"app-shell__sidebar" in res.data
    assert b"study-panel-container" in res.data
    assert b"Rock" in res.data
    assert b"Acoustic Profile &amp; Physical Descriptors" in res.data


def test_study_panel_ajax_partial_request(client, app):
    """Test AJAX request for genre detail returning only the _detail_fragment.html."""
    _ensure_genres_seeded(app)

    # With X-Requested-With: XMLHttpRequest
    res_ajax = client.get("/library/rock", headers={"X-Requested-With": "XMLHttpRequest"})
    assert res_ajax.status_code == 200
    # Should contain fragment root
    assert b'id="study-panel-root"' in res_ajax.data
    # Should NOT contain outer app-shell or <html> document
    assert b"<!DOCTYPE html>" not in res_ajax.data
    assert b'<div class="app-shell">' not in res_ajax.data

    # With ?partial=1 query param
    res_param = client.get("/library/rock?partial=1")
    assert res_param.status_code == 200
    assert b'id="study-panel-root"' in res_param.data
    assert b"<!DOCTYPE html>" not in res_param.data


def test_study_panel_acoustic_profile_metrics(client, app):
    """Verify acoustic properties are formatted with .mono class."""
    _ensure_genres_seeded(app)

    res = client.get("/library/rock", headers={"X-Requested-With": "XMLHttpRequest"})
    assert res.status_code == 200
    html = res.data.decode("utf-8")

    assert "Rhythmic Tempo" in html
    assert "Spectral Brightness" in html
    assert "Dynamic Energy" in html
    assert "Signal Transient Metrics" in html
    assert "class=\"study-panel__stat-value mono\"" in html


def test_study_panel_teaching_topics_and_structure(client, app):
    """Verify teaching topics checklist and structural form sections."""
    _ensure_genres_seeded(app)

    res = client.get("/library/rock", headers={"X-Requested-With": "XMLHttpRequest"})
    assert res.status_code == 200
    html = res.data.decode("utf-8")

    assert "Core Teaching Topics" in html
    assert "Structural Forms &amp; Conventions" in html
    assert "Historical Lineage &amp; Cultural Origins" in html
    assert "Core Instrumentation" in html
    assert "Subgenre Taxonomy" in html


def test_study_panel_comparative_links(client, app):
    """Verify comparative cross-genre links are rendered."""
    _ensure_genres_seeded(app)

    res = client.get("/library/rock", headers={"X-Requested-With": "XMLHttpRequest"})
    assert res.status_code == 200
    html = res.data.decode("utf-8")

    assert "Comparative Analysis &amp; Reference Links" in html
    assert "/explore/compare/rock/" in html


def test_study_panel_unknown_genre_404(client, app):
    """Verify requesting an unknown genre returns a 404 status."""
    _ensure_genres_seeded(app)

    res = client.get("/library/nonexistent-music-slug")
    assert res.status_code == 404
