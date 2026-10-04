"""
dataset_quality.py
==================
Audio quality assurance, decodability verification, duplicate detection,
and quality reporting for CLASSIFY dataset curation.
Part of Phase 2 for CLASSIFY.
"""

import hashlib
import json
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_QUALITY_REPORT_PATH = os.path.join(BASE_DIR, "datasets", "quality_report.json")


def compute_audio_checksum(y: np.ndarray) -> str:
    """
    Computes a SHA-256 checksum of raw decoded audio sample values.
    Catches exact content duplicates across re-encoded file formats.
    """
    # Use float32 representation for deterministic checksum
    y_contiguous = np.ascontiguousarray(y.astype(np.float32))
    return hashlib.sha256(y_contiguous.tobytes()).hexdigest()


def verify_audio_file(
    file_path: str,
    min_duration: float = 5.0,
    rms_floor: float = 0.0005,
    target_sr: int = 22050,
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """
    Validates that an audio file:
    1. Exists on disk.
    2. Decodes cleanly without corruption.
    3. Has valid duration (>= min_duration).
    4. Has audible content above the RMS silence floor.
    5. Yields a consistent audio sample checksum.

    Returns:
        tuple: (is_valid, rejection_reason, metadata_dict)
    """
    if not os.path.exists(file_path):
        return False, f"File does not exist: {file_path}", {}

    # Check file size > 0
    size_bytes = os.path.getsize(file_path)
    if size_bytes == 0:
        return False, "File is zero bytes (empty)", {}

    # Attempt soundfile/librosa decode
    try:
        y, sr = librosa.load(file_path, sr=target_sr, mono=True)
    except Exception as e:
        return False, f"Decoding error: {type(e).__name__}: {str(e)}", {}

    duration = float(librosa.get_duration(y=y, sr=sr))

    # Duration bound check
    if duration < min_duration:
        return False, f"Duration {duration:.2f}s is below minimum threshold of {min_duration}s", {}

    # RMS silence floor check
    rms = float(np.sqrt(np.mean(y ** 2)))
    if rms < rms_floor:
        return False, f"Audio is silent or near-silent (RMS {rms:.6f} < floor {rms_floor:.6f})", {}

    checksum = compute_audio_checksum(y)

    meta = {
        "duration": round(duration, 2),
        "sample_rate": sr,
        "rms": round(rms, 6),
        "checksum": checksum,
        "size_bytes": size_bytes,
    }

    return True, None, meta


def run_quality_audit(
    manifest_rows: List[Dict[str, Any]],
    output_report_path: Optional[str] = None,
    base_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Runs quality verification on all candidate audio files in manifest_rows.
    Flags exact content duplicates, corruption, silence, or duration defects.
    Logs every rejection with explicit reasons.

    Writes quality report to datasets/quality_report.json.
    """
    root_dir = base_dir or BASE_DIR
    out_path = output_report_path or DEFAULT_QUALITY_REPORT_PATH

    accepted_rows: List[Dict[str, Any]] = []
    rejected_rows: List[Dict[str, Any]] = []

    seen_checksums: Dict[str, str] = {}  # checksum -> original track_path
    genre_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"accepted": 0, "rejected": 0})
    source_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"accepted": 0, "rejected": 0})

    for row in manifest_rows:
        rel_path = row.get("track_path", "")
        abs_path = os.path.join(root_dir, rel_path) if not os.path.isabs(rel_path) else rel_path
        genre = str(row.get("genre", "Unknown"))
        source = str(row.get("source", "Unknown"))

        is_valid, reason, meta = verify_audio_file(abs_path)

        if not is_valid:
            rejection = {
                "track_path": rel_path,
                "genre": genre,
                "source": source,
                "reason": reason,
            }
            rejected_rows.append(rejection)
            genre_counts[genre]["rejected"] += 1
            source_counts[source]["rejected"] += 1
            logger.warning(f"Rejected {rel_path}: {reason}")
            continue

        # Check for duplicate audio checksum
        checksum = meta["checksum"]
        if checksum in seen_checksums:
            dupe_reason = f"Exact audio duplicate of previously registered track: {seen_checksums[checksum]}"
            rejection = {
                "track_path": rel_path,
                "genre": genre,
                "source": source,
                "reason": dupe_reason,
                "checksum": checksum,
            }
            rejected_rows.append(rejection)
            genre_counts[genre]["rejected"] += 1
            source_counts[source]["rejected"] += 1
            logger.warning(f"Rejected duplicate {rel_path}: {dupe_reason}")
            continue

        seen_checksums[checksum] = rel_path
        row_with_meta = dict(row)
        row_with_meta["checksum"] = checksum
        row_with_meta["duration"] = meta["duration"]
        accepted_rows.append(row_with_meta)
        genre_counts[genre]["accepted"] += 1
        source_counts[source]["accepted"] += 1

    report = {
        "summary": {
            "total_candidates_examined": len(manifest_rows),
            "total_accepted": len(accepted_rows),
            "total_rejected": len(rejected_rows),
            "acceptance_rate": round(len(accepted_rows) / max(1, len(manifest_rows)), 4),
        },
        "by_genre": dict(genre_counts),
        "by_source": dict(source_counts),
        "rejections": rejected_rows,
    }

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Quality report written to {out_path}")

    return report
