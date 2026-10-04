"""
library.py
==========
Genre Library educational reference routes.
Displays data-driven genre profiles, historical context, typical song structures,
instrumentation, and subgenre taxonomy directly from the database.
Enhanced with two-pane full-screen study shell and AJAX progressive enhancement
(final_design.md §3).
"""

from flask import Blueprint, render_template, request, abort
from classify.models import Genre, Subgenre, Prediction, Analysis
from classify.services.teaching_topics_service import get_teaching_topics

library_bp = Blueprint("library", __name__, url_prefix="/library")

# Typical acoustic characteristic ranges across major music genres
ACOUSTIC_PROFILES = {
    "rock": {"bpm_range": "110 – 150 BPM", "brightness": "Medium-High (2000–3500 Hz)", "energy": "High", "dynamics": "Punchy backbeat with mid-range guitar presence"},
    "pop": {"bpm_range": "100 – 130 BPM", "brightness": "High (2500–4000 Hz)", "energy": "High-Medium", "dynamics": "Polished, compressed with centered vocals"},
    "classical": {"bpm_range": "60 – 160 BPM (Variable)", "brightness": "Wide dynamic range (1200–3000 Hz)", "energy": "Low to High (Rubato)", "dynamics": "Uncompressed natural room acoustics with extreme dynamic contrast"},
    "jazz": {"bpm_range": "80 – 220 BPM (Swing)", "brightness": "Warm (1500–2800 Hz)", "energy": "Medium-Low", "dynamics": "Acoustic nuance with ride cymbal and walking bass priority"},
    "metal": {"bpm_range": "120 – 190 BPM", "brightness": "High / Scooped (2500–5000 Hz)", "energy": "Very High", "dynamics": "Heavy distortion compression with aggressive transient attacks"},
    "ambient": {"bpm_range": "50 – 85 BPM", "brightness": "Low-Mid (800–1800 Hz)", "energy": "Very Low", "dynamics": "Expansive reverb decay with sustained drones and minimal percussion"},
    "blues": {"bpm_range": "70 – 130 BPM (Shuffle)", "brightness": "Warm-Mid (1400–2600 Hz)", "energy": "Medium", "dynamics": "Microtonal expressive string bends with 12-bar cyclic pulse"},
    "country": {"bpm_range": "85 – 130 BPM", "brightness": "Bright-Clean (2200–3800 Hz)", "energy": "Medium", "dynamics": "Acoustic guitar and twang transient clarity"},
    "hip-hop": {"bpm_range": "75 – 100 BPM", "brightness": "Deep Lows + Crisp Highs (1200–3200 Hz)", "energy": "High-Medium", "dynamics": "Sub-bass 808 foundation with sharp snare transients"},
    "disco": {"bpm_range": "115 – 130 BPM", "brightness": "Bright (2400–4000 Hz)", "energy": "Very High", "dynamics": "Four-on-the-floor kick drive with continuous syncopated groove"},
    "reggae": {"bpm_range": "65 – 90 BPM", "brightness": "Warm Lows (1000–2200 Hz)", "energy": "Medium-Low", "dynamics": "Heavy sub-bass with off-beat rim shots and chop chords"},
}


@library_bp.route("", methods=["GET"])
@library_bp.route("/", methods=["GET"])
def index():
    """
    Catalog overview displaying all genres in the database taxonomy.
    Renders the persistent two-pane shell container with empty study state.
    """
    search_query = request.args.get("q", "").strip()

    query = Genre.query.order_by(Genre.name)
    if search_query:
        query = query.filter(
            (Genre.name.ilike(f"%{search_query}%"))
            | (Genre.description.ilike(f"%{search_query}%"))
        )

    genres = query.all()
    all_genres = Genre.query.order_by(Genre.name).all()
    total_subgenres = Subgenre.query.count()

    return render_template(
        "library/index.html",
        genres=genres,
        all_genres=all_genres,
        search_query=search_query,
        total_subgenres=total_subgenres,
        active_genre=None,
    )


@library_bp.route("/<genre_slug>", methods=["GET"])
def detail(genre_slug):
    """
    Detailed educational study view for a specific genre.
    Supports progressive enhancement: returns fragment template on AJAX
    (X-Requested-With or ?partial=1) or the full server-rendered shell on direct load.
    """
    genre = Genre.query.filter_by(slug=genre_slug.lower()).first()
    if not genre:
        abort(404, description=f"Genre '{genre_slug}' was not found in the library.")

    # Fetch platform submissions count for this genre
    submission_count = (
        Prediction.query.filter_by(genre_id=genre.id)
        .join(Analysis, Prediction.analysis_id == Analysis.id)
        .count()
    )

    # Fetch acoustic profile reference
    acoustic_info = ACOUSTIC_PROFILES.get(
        genre.slug,
        {
            "bpm_range": "90 – 140 BPM",
            "brightness": "Medium (1500–3000 Hz)",
            "energy": "Medium",
            "dynamics": "Balanced acoustic and dynamic profile",
        },
    )

    # Fetch pedagogical teaching topics and listening focus (final_design.md §3.3)
    teaching_guide = get_teaching_topics(genre.name)

    # Fetch other genres for comparative cross-links
    other_genres = Genre.query.filter(Genre.id != genre.id).order_by(Genre.name).all()
    all_genres = Genre.query.order_by(Genre.name).all()

    # Progressive enhancement branch (final_design.md §3.5)
    is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.args.get("partial") == "1"
    if is_ajax:
        return render_template(
            "library/_detail_fragment.html",
            genre=genre,
            subgenres=genre.subgenres,
            submission_count=submission_count,
            acoustic_info=acoustic_info,
            other_genres=other_genres,
            teaching_guide=teaching_guide,
        )

    return render_template(
        "library/detail.html",
        genre=genre,
        subgenres=genre.subgenres,
        submission_count=submission_count,
        acoustic_info=acoustic_info,
        other_genres=other_genres,
        teaching_guide=teaching_guide,
        all_genres=all_genres,
        active_genre=genre,
    )
