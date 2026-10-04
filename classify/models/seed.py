"""
seed.py
=======
Seeds the SQLite database with the core data-driven genre and subgenre taxonomy.
Fulfills Ground Rule 3: Taxonomy lives in the database, never hard-coded in Python or routes.
Part of Phase 4 for CLASSIFY.
"""

from classify.extensions import db
from classify.models.genre import Genre, Subgenre

GENRE_SEED_DATA = [
    {
        "name": "Rock",
        "slug": "rock",
        "description": "A genre of popular music characterized by a prominent rhythm section, electric guitars, and strong vocal melodies with dynamic drive.",
        "history": "Originating in the United States in the late 1940s and early 1950s from rhythm and blues, country, and gospel, rock rapidly evolved across Britain and the world throughout the 1960s and 1970s.",
        "typical_structures": "Verse-Chorus form (AABA or ABABCB), guitar solo section, 4/4 time signature with backbeat on beats 2 and 4.",
        "common_instrumentation": "Electric guitar, bass guitar, acoustic drums, vocals, keyboards/synthesizer.",
        "subgenres": [
            {"name": "Classic Rock", "slug": "classic-rock", "description": "1960s-1970s rock emphasizing driving rhythms, blues riffs, and melodic vocal hooks."},
            {"name": "Hard Rock", "slug": "hard-rock", "description": "Heavy distortion, aggressive power chords, and anthemic choruses."},
            {"name": "Alternative Rock", "slug": "alternative-rock", "description": "Post-punk derived rock with unconventional guitar textures and expressive lyrics."},
        ],
    },
    {
        "name": "Pop",
        "slug": "pop",
        "description": "Popular music characterized by catchy melodic hooks, regular rhythmic grooves, accessible song structures, and modern production.",
        "history": "Emerged in its modern form during the mid-1950s as softer rock and roll and continued evolving through dance, electronic, and global influences.",
        "typical_structures": "Intro-Verse-Pre-Chorus-Chorus-Verse-Chorus-Bridge-Chorus-Outro with emphasis on immediate hook recognition.",
        "common_instrumentation": "Synthesizers, electronic drum programming, electric bass/synth-bass, multi-tracked lead and backing vocals.",
        "subgenres": [
            {"name": "Dance Pop", "slug": "dance-pop", "description": "Upbeat pop tailored for nightclub and dance floor energy with 4-on-the-floor kick drums."},
            {"name": "Synth Pop", "slug": "synth-pop", "description": "Pop dominated by analog and digital synthesizer timbres and sequenced arpeggios."},
            {"name": "Indie Pop", "slug": "indie-pop", "description": "Lighter melodic pop with acoustic guitars and understated DIY production aesthetics."},
        ],
    },
    {
        "name": "Classical",
        "slug": "classical",
        "description": "Art music rooted in Western European tradition, emphasizing complex harmonic progressions, rich dynamic nuance, and acoustic orchestration.",
        "history": "Spans medieval chant through the Baroque, Classical, Romantic, and Modern eras, codified by composers like Bach, Mozart, and Beethoven.",
        "typical_structures": "Sonata-allegro form, Theme and Variations, Rondo, Concerto movement structures.",
        "common_instrumentation": "Violin, viola, cello, double bass, flute, oboe, clarinet, bassoon, French horn, trumpet, timpani, piano.",
        "subgenres": [
            {"name": "Orchestral", "slug": "orchestral", "description": "Large ensemble symphonic music balancing string, brass, woodwind, and percussion sections."},
            {"name": "Chamber Music", "slug": "chamber-music", "description": "Intimate acoustic performance for small groups such as string quartets or piano trios."},
            {"name": "Baroque", "slug": "baroque", "description": "Polyphonic counterpoint, basso continuo, and ornamental keyboard/string writing."},
        ],
    },
    {
        "name": "Jazz",
        "slug": "jazz",
        "description": "A sophisticated musical style characterized by swing rhythms, complex polyrhythms, syncopation, extended chords, and improvisation.",
        "history": "Born in African American communities in New Orleans during the late 19th and early 20th centuries, fusing blues, ragtime, and brass band music.",
        "typical_structures": "Head-Solos-Head structure over 32-bar AABA or 12-bar blues progressions.",
        "common_instrumentation": "Saxophone, trumpet, trombone, upright bass, piano, drum kit (with ride cymbal and brushes), hollow-body guitar.",
        "subgenres": [
            {"name": "Bebop", "slug": "bebop", "description": "Fast tempos, complex chord substitutions, and virtuoso soloing pioneered by Charlie Parker and Dizzy Gillespie."},
            {"name": "Cool Jazz", "slug": "cool-jazz", "description": "Relaxed tempos, lighter tone colors, and classical-influenced arrangements."},
            {"name": "Jazz Fusion", "slug": "jazz-fusion", "description": "Blending jazz harmony and improvisation with rock rhythms and electric instruments."},
        ],
    },
    {
        "name": "Metal",
        "slug": "metal",
        "description": "A genre of heavy music defined by distorted electric guitars, emphatic rhythms, dense bass and drum sound, and vigorous vocal styles.",
        "history": "Developed in the late 1960s and early 1970s primarily in the UK and US, sparked by Black Sabbath, Deep Purple, and Led Zeppelin.",
        "typical_structures": "Riff-centric verse-chorus arrangements with breakdown sections and technical guitar solos.",
        "common_instrumentation": "Down-tuned electric guitars, double-kick bass drums, high-output electric bass, harsh or soaring vocals.",
        "subgenres": [
            {"name": "Heavy Metal", "slug": "heavy-metal", "description": "Classic galloping rhythms, dual guitar harmonies, and operatic or gritty vocals."},
            {"name": "Thrash Metal", "slug": "thrash-metal", "description": "High-velocity tempo, complex palm-muted riffing, and aggressive vocal delivery."},
        ],
    },
    {
        "name": "Ambient",
        "slug": "ambient",
        "description": "A genre of music that emphasizes tone and atmosphere over traditional musical structure or rhythm.",
        "history": "Pioneered in the 1970s by Brian Eno, drawing from experimental avant-garde, minimalism, and synthesizer soundscapes.",
        "typical_structures": "Free-form non-rhythmic soundscapes, slow dynamic fades, and repeating textural cycles.",
        "common_instrumentation": "Synthesizers, tape loops, digital reverbs/delays, field recordings, treated guitars and pianos.",
        "subgenres": [
            {"name": "Drone", "slug": "drone", "description": "Sustained continuous tones and microtonal shifts creating deep acoustic resonance."},
            {"name": "Chillout", "slug": "chillout", "description": "Ambient with gentle downtempo electronic beats and warm pads."},
        ],
    },
    {
        "name": "Blues",
        "slug": "blues",
        "description": "A foundational African American musical form characterized by call-and-response, bent notes, blue notes, and emotive storytelling.",
        "history": "Originated in the Deep South of the United States around the 1860s, deeply influencing jazz, rock, and country music.",
        "typical_structures": "12-bar blues progression using dominant 7th chords with I-IV-V harmonic movement.",
        "common_instrumentation": "Acoustic or electric guitar, harmonica, piano, bass, drums, expressive vocals.",
        "subgenres": [
            {"name": "Delta Blues", "slug": "delta-blues", "description": "Acoustic slide guitar, expressive foot-stomping rhythm, and passionate vocal delivery."},
            {"name": "Chicago Blues", "slug": "chicago-blues", "description": "Electrified urban blues with amplified harmonica, electric guitar, and full rhythm section."},
        ],
    },
    {
        "name": "Country",
        "slug": "country",
        "description": "A genre of American popular music characterized by ballads and dance tunes with simple form, folk lyrics, and string accompaniment.",
        "history": "Originated in the southern and western regions of the United States in the early 1920s from roots in folk, spirituals, and blues.",
        "typical_structures": "Verse-chorus storytelling form with narrative story arc and memorable melodic hooks.",
        "common_instrumentation": "Acoustic guitar, pedal steel guitar, fiddle, banjo, mandolin, upright or electric bass.",
        "subgenres": [
            {"name": "Traditional Country", "slug": "traditional-country", "description": "Honky-tonk rhythms, pedal steel melodies, and heartfelt vocal twang."},
            {"name": "Bluegrass", "slug": "bluegrass", "description": "High-speed acoustic string interplay featuring banjo rolls and fiddle breakdowns."},
        ],
    },
    {
        "name": "Hip-Hop",
        "slug": "hip-hop",
        "description": "A cultural and musical movement defined by rhythmic vocal delivery (rapping/MCing) over syncopated beats and sampled grooves.",
        "history": "Born in the Bronx, New York in August 1973 through block parties hosted by DJ Kool Herc, Grandmaster Flash, and Afrika Bambaataa.",
        "typical_structures": "16-bar verses separated by 8-bar chorus hooks over looped beat patterns.",
        "common_instrumentation": "Drum machines (TR-808), samplers, turntables, synthesizer basslines, vocals.",
        "subgenres": [
            {"name": "Boom Bap", "slug": "boom-bap", "description": "Hard-hitting acoustic kick and snare sampling with syncopated jazz/soul loops."},
            {"name": "Trap", "slug": "trap", "description": "Rapid rolling hi-hats, sub-bass 808 glides, and layered atmospheric brass/synths."},
        ],
    },
    {
        "name": "Disco",
        "slug": "disco",
        "description": "An energetic dance music genre characterized by four-on-the-floor beats, syncopated basslines, string sections, and rhythm guitars.",
        "history": "Flourished in urban nightlife and discotheques during the 1970s, establishing the foundation for modern electronic club music.",
        "typical_structures": "Extended dance grooves with verse-chorus builds, orchestral breaks, and rhythm drops.",
        "common_instrumentation": "Electric bass, rhythm guitar (chic scratch style), brass, lush strings, drum kit with open hi-hat on upbeats.",
        "subgenres": [
            {"name": "Nu-Disco", "slug": "nu-disco", "description": "Modern electronic disco blending classic groove with contemporary synthesizer production."},
        ],
    },
    {
        "name": "Reggae",
        "slug": "reggae",
        "description": "A music genre that originated in Jamaica, characterized by offbeat rhythmic accents, prominent basslines, and socially conscious lyrics.",
        "history": "Developed in Jamaica in the late 1960s from ska and rocksteady, popularized internationally by Bob Marley and the Wailers.",
        "typical_structures": "Emphasis on the offbeat (skank) with one-drop or steppers drum rhythm and heavy sub-bass.",
        "common_instrumentation": "Bass guitar, drums, electric guitar, organ/piano (skank), brass section, percussion.",
        "subgenres": [
            {"name": "Roots Reggae", "slug": "roots-reggae", "description": "Spiritual, conscious lyrics with heavy one-drop rhythms and organ bubble."},
            {"name": "Dub", "slug": "dub", "description": "Instrumental reggae remixes featuring heavy echo, reverb effects, and dominant bass dropouts."},
        ],
    },
]


def seed_genres(app=None):
    """
    Inserts or updates the core genre and subgenre taxonomy in the database.
    Idempotent: skips existing genres and subgenres without duplication.
    """
    created_genres = 0
    created_subgenres = 0

    for item in GENRE_SEED_DATA:
        genre = Genre.query.filter_by(slug=item["slug"]).first()
        if not genre:
            genre = Genre(
                name=item["name"],
                slug=item["slug"],
                description=item["description"],
                history=item["history"],
                typical_structures=item["typical_structures"],
                common_instrumentation=item["common_instrumentation"],
            )
            db.session.add(genre)
            db.session.flush()
            created_genres += 1

        for sub_item in item.get("subgenres", []):
            subgenre = Subgenre.query.filter_by(
                genre_id=genre.id, slug=sub_item["slug"]
            ).first()
            if not subgenre:
                sub = Subgenre(
                    genre_id=genre.id,
                    name=sub_item["name"],
                    slug=sub_item["slug"],
                    description=sub_item["description"],
                )
                db.session.add(sub)
                created_subgenres += 1

    db.session.commit()
    return created_genres, created_subgenres
