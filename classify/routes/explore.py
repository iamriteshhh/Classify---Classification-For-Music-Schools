"""
explore.py
==========
Comparative learning and exploration routes.
Allows students and teachers to compare two musical genres side-by-side,
contrasting historical roots, structural forms, instrumentation, subgenres,
and acoustic feature profiles.
Part of Phase 9 for CLASSIFY.
"""

from flask import Blueprint, render_template, request, redirect, url_for, abort
from classify.models import Genre
from classify.routes.library import ACOUSTIC_PROFILES

explore_bp = Blueprint("explore", __name__, url_prefix="/explore")


@explore_bp.route("", methods=["GET"])
@explore_bp.route("/", methods=["GET"])
def index():
    """
    Landing page for genre comparison.
    If query parameters 'g1' and 'g2' are supplied, redirects to the comparative view.
    Otherwise defaults to comparing Rock vs. Classical.
    """
    g1 = request.args.get("g1", "").strip().lower()
    g2 = request.args.get("g2", "").strip().lower()

    if g1 and g2:
        return redirect(url_for("explore.compare", genre_a=g1, genre_b=g2))
    elif g1:
        # Default second genre to pop if g1 is rock, else rock
        default_other = "pop" if g1 != "pop" else "rock"
        return redirect(url_for("explore.compare", genre_a=g1, genre_b=default_other))

    # Default comparison: rock vs classical
    return redirect(url_for("explore.compare", genre_a="rock", genre_b="classical"))


@explore_bp.route("/compare/<genre_a>/<genre_b>", methods=["GET"])
def compare(genre_a, genre_b):
    """
    Side-by-side comparative analysis of two musical genres.
    Renders acoustic metric comparisons, structural distinctions,
    instrumentation contrasts, and pedagogical distinction insights.
    """
    first_genre = Genre.query.filter_by(slug=genre_a.lower()).first()
    if not first_genre:
        abort(404, description=f"Genre '{genre_a}' not found in the taxonomy database.")

    second_genre = Genre.query.filter_by(slug=genre_b.lower()).first()
    if not second_genre:
        abort(404, description=f"Genre '{genre_b}' not found in the taxonomy database.")

    all_genres = Genre.query.order_by(Genre.name).all()

    profile_a = ACOUSTIC_PROFILES.get(
        first_genre.slug,
        {"bpm_range": "100–140 BPM", "brightness": "Medium (2000 Hz)", "energy": "Medium", "dynamics": "Standard balance"},
    )
    profile_b = ACOUSTIC_PROFILES.get(
        second_genre.slug,
        {"bpm_range": "100–140 BPM", "brightness": "Medium (2000 Hz)", "energy": "Medium", "dynamics": "Standard balance"},
    )

    # Distinctions rationale
    is_identical = first_genre.id == second_genre.id

    return render_template(
        "explore/compare.html",
        genre_a=first_genre,
        genre_b=second_genre,
        profile_a=profile_a,
        profile_b=profile_b,
        all_genres=all_genres,
        is_identical=is_identical,
    )
