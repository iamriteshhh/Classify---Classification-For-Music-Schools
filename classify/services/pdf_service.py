"""CLASSIFY - PDF Report Generation Service.

Generates official conservatory accreditation and pedagogical jury reports
for audio analysis submissions using ReportLab.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def generate_analysis_pdf(analysis: Any) -> io.BytesIO:
    """Generates an official PDF report for an Analysis record.

    Returns:
        io.BytesIO: PDF buffer ready for streaming.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0f172a"),
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#64748b"),
    )
    section_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=12,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155"),
    )

    story = []

    # 1. Header Banner
    story.append(Paragraph("CLASSIFY &bull; Music School Acoustic Analysis Report", title_style))
    story.append(
        Paragraph(
            f"Generated: {datetime.now(timezone.utc).strftime('%B %d, %Y at %H:%M UTC')} &bull; Analysis ID: #{analysis.id}",
            subtitle_style,
        )
    )
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563eb"), spaceAfter=14))

    # 2. Track & Submission Overview
    song = analysis.song
    student_email = song.student.user.email if (song and song.student and song.student.user) else "Anonymous Student"
    track_title = song.original_filename if song else "Audio Submission"
    duration = f"{song.duration:.2f} s" if (song and song.duration) else "N/A"

    meta_data = [
        [Paragraph("<b>Track Title:</b>", body_style), Paragraph(track_title, body_style),
         Paragraph("<b>Student:</b>", body_style), Paragraph(student_email, body_style)],
        [Paragraph("<b>Recorded At:</b>", body_style), Paragraph(analysis.analyzed_at.strftime('%Y-%m-%d %H:%M'), body_style),
         Paragraph("<b>Duration:</b>", body_style), Paragraph(duration, body_style)],
    ]
    meta_table = Table(meta_data, colWidths=[80, 180, 70, 190])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("PADDING", (0, 0), (-1, -1), 6),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 14))

    # 3. Machine Learning Classification
    pred = analysis.prediction
    genre_name = pred.genre.name if (pred and pred.genre) else "Unknown"
    subgenre_name = pred.subgenre.name if (pred and pred.subgenre) else "None"
    conf_pct = f"{pred.confidence * 100:.1f}%" if pred else "N/A"
    explanation = pred.explanation if pred else "No model explanation available."

    story.append(Paragraph("Machine Learning Genre Prediction", section_style))
    ml_data = [
        [Paragraph("<b>Primary Genre:</b>", body_style), Paragraph(f"<b>{genre_name}</b>", body_style),
         Paragraph("<b>Subgenre:</b>", body_style), Paragraph(subgenre_name, body_style)],
        [Paragraph("<b>Confidence:</b>", body_style), Paragraph(conf_pct, body_style),
         Paragraph("<b>Model Architecture:</b>", body_style), Paragraph("Trained Classical ML Classifier", body_style)],
    ]
    ml_table = Table(ml_data, colWidths=[90, 170, 110, 150])
    ml_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f0fdf4")),
        ("PADDING", (0, 0), (-1, -1), 6),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#bbf7d0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(ml_table)
    story.append(Spacer(1, 6))

    exp_para = Paragraph(f"<b>Acoustic Rationale:</b> {explanation}", body_style)
    story.append(exp_para)
    story.append(Spacer(1, 14))

    # 4. Tier 1 Acoustic Measurements
    af = analysis.audio_features
    story.append(Paragraph("Tier 1 Acoustic Measurements", section_style))
    if af:
        key_val = af.detected_key or "N/A"
        time_sig_val = af.time_signature or "4/4"
        chords_val = af.chord_progression or "None detected"
        vocal_val = af.vocal_presence or "Instrumental"
        tempo_val = f"{af.tempo:.1f} BPM" if af.tempo else "N/A"
        rms_val = f"{af.rms_mean:.4f} (var: {af.rms_var or 0:.4f})"
        zcr_val = f"{af.zcr_mean:.4f}"
        sc_val = f"{af.spectral_centroid:.1f} Hz"
        sb_val = f"{af.spectral_bandwidth:.1f} Hz"
        sr_val = f"{af.spectral_rolloff:.1f} Hz"
        contrast_val = f"{af.spectral_contrast or 0:.1f} dB"

        ac_data = [
            [Paragraph("<b>Detected Key:</b>", body_style), Paragraph(key_val, body_style),
             Paragraph("<b>Time Signature:</b>", body_style), Paragraph(time_sig_val, body_style)],
            [Paragraph("<b>Chord Progression:</b>", body_style), Paragraph(chords_val, body_style),
             Paragraph("<b>Vocal Presence:</b>", body_style), Paragraph(vocal_val, body_style)],
            [Paragraph("<b>Tempo:</b>", body_style), Paragraph(tempo_val, body_style),
             Paragraph("<b>RMS Energy:</b>", body_style), Paragraph(rms_val, body_style)],
            [Paragraph("<b>Spectral Centroid:</b>", body_style), Paragraph(sc_val, body_style),
             Paragraph("<b>Spectral Bandwidth:</b>", body_style), Paragraph(sb_val, body_style)],
            [Paragraph("<b>Spectral Rolloff:</b>", body_style), Paragraph(sr_val, body_style),
             Paragraph("<b>Zero Crossing Rate:</b>", body_style), Paragraph(zcr_val, body_style)],
        ]
        ac_table = Table(ac_data, colWidths=[120, 140, 110, 150])
        ac_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("PADDING", (0, 0), (-1, -1), 5),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(ac_table)
    story.append(Spacer(1, 14))

    # 5. Instructor Pedagogical Review
    review = analysis.teacher_review
    story.append(Paragraph("Instructor Pedagogical Review", section_style))
    if review:
        status_text = "Agreed with Model" if review.agrees_with_model else f"Corrected to {review.teacher_genre}"
        reviewed_at_str = review.reviewed_at.strftime('%Y-%m-%d %H:%M') if review.reviewed_at else "Recorded"
        rev_data = [
            [Paragraph("<b>Instructor Assessment:</b>", body_style), Paragraph(status_text, body_style),
             Paragraph("<b>Review Date:</b>", body_style), Paragraph(reviewed_at_str, body_style)],
        ]
        rev_table = Table(rev_data, colWidths=[120, 140, 90, 170])
        rev_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eff6ff")),
            ("PADDING", (0, 0), (-1, -1), 6),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#bfdbfe")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(rev_table)
        story.append(Spacer(1, 6))
        comment_text = review.comment or "No written comments recorded."
        story.append(Paragraph(f"<b>Instructor Notes:</b> {comment_text}", body_style))
    else:
        story.append(Paragraph("<i>This submission has not yet received an instructor pedagogical review.</i>", body_style))

    story.append(Spacer(1, 24))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd5e1"), spaceAfter=8))
    story.append(Paragraph("CLASSIFY &bull; Music School Song Classification &amp; Pedagogical Evaluation System", subtitle_style))

    doc.build(story)
    buffer.seek(0)
    return buffer
