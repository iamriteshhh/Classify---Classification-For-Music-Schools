"""
test_seed.py
============
Tests for classify.models.seed module.
Verifies that:
- Major genres and subgenres are inserted into the database
- Re-running the seed function is idempotent (no duplicates)
- All seeded genres have valid descriptions and slugs
Part of Phase 4 for CLASSIFY.
"""

from classify.models.genre import Genre, Subgenre
from classify.models.seed import seed_genres


def test_seed_genres_idempotent(app):
    """Test seeding database with genre and subgenre taxonomy."""
    with app.app_context():
        # First seed run
        created_g, created_s = seed_genres(app)
        assert created_g > 0
        assert created_s > 0

        # Query all genres
        genres = Genre.query.all()
        genre_names = {g.name for g in genres}
        expected_core = {"Rock", "Pop", "Classical", "Jazz", "Metal", "Ambient", "Blues", "Country", "Hip-Hop", "Disco", "Reggae"}
        assert expected_core.issubset(genre_names)

        # Verify each genre has description and slug
        for g in genres:
            assert len(g.slug) > 0
            assert len(g.description) > 10

        # Subgenres check
        subgenres = Subgenre.query.all()
        assert len(subgenres) >= 20

        # Second seed run should be idempotent (0 new items added)
        re_g, re_s = seed_genres(app)
        assert re_g == 0
        assert re_s == 0
