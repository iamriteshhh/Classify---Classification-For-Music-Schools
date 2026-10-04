"""
audio_processing.py
===================
Expanded Tier 1 audio feature extraction using Librosa.
Extracts comprehensive temporal, spectral, timbral (full MFCC vector), tonal (chroma),
and rhythmic acoustic metrics.
Includes Task 0.1 LUFS Loudness Normalization (-14.0 LUFS) and Task 0.2 Frequency
Bandwidth Harmonization (8kHz 4th-order Butterworth low-pass filter) to eliminate
the Metal and Ambient attractor sinks.
Part of Phase 0 & Phase 3 for CLASSIFY.
"""

import os
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
import pyloudnorm as pyln


def apply_lowpass_filter(
    y: np.ndarray,
    sr: int = 22050,
    cutoff: float = 8000.0,
    order: int = 4,
) -> np.ndarray:
    """
    Applies a 4th-order Butterworth low-pass filter at cutoff (default 8,000 Hz).
    Harmonizes all incoming audio with GTZAN's 8kHz acquisition ceiling,
    eliminating the high-frequency shortcut that causes modern uploads to
    pathologically collapse into the Ambient class.
    """
    if y is None or len(y) == 0:
        return y

    nyquist = 0.5 * sr
    # Cutoff must be strictly below Nyquist frequency
    eff_cutoff = min(cutoff, nyquist * 0.95)
    if eff_cutoff <= 0:
        return y

    try:
        sos = signal.butter(order, eff_cutoff, btype="lowpass", fs=sr, output="sos")
        filtered = signal.sosfilt(sos, y)
        return filtered.astype(np.float32)
    except Exception:
        return y.astype(np.float32)


def normalize_lufs(
    y: np.ndarray,
    sr: int = 22050,
    target_lufs: float = -14.0,
    block_size: float = 0.400,
) -> np.ndarray:
    """
    Measures integrated loudness using ITU-R BS.1770-4 via pyloudnorm.Meter
    and normalizes audio to target_lufs (-14.0 LUFS).
    Eliminates the 'Metal Sink' caused by hyper-compressed modern commercial masters
    having elevated RMS identical to 1990s heavy metal.
    """
    if y is None or len(y) == 0:
        return y

    # Signal must have duration greater than block size for pyloudnorm
    min_samples = int(sr * block_size)
    if len(y) <= min_samples:
        # Fallback to gentle peak normalization for micro-snippets
        peak = float(np.max(np.abs(y)))
        if peak > 1e-6:
            return (y * (0.5 / peak)).astype(np.float32)
        return y.astype(np.float32)

    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            meter = pyln.Meter(sr, block_size=block_size)
            loudness = meter.integrated_loudness(y)

            # Handle silent, infinite, or NaN signals
            if np.isneginf(loudness) or np.isnan(loudness) or np.isposinf(loudness):
                return y.astype(np.float32)

            normalized = pyln.normalize.loudness(y, loudness, target_lufs)

        # Soft limit to guard against extreme clipping
        peak = float(np.max(np.abs(normalized)))
        if peak > 1.2:
            normalized = normalized * (1.0 / peak)

        return normalized.astype(np.float32)
    except Exception:
        return y.astype(np.float32)


def preprocess_audio_signal(
    y: np.ndarray,
    sr: int = 22050,
    apply_filter: bool = True,
    apply_lufs: bool = True,
    cutoff: float = 8000.0,
    target_lufs: float = -14.0,
) -> np.ndarray:
    """
    Standard Phase 0 preprocessing pipeline applied to all audio signals
    before acoustic feature extraction:
      1. Butterworth 4th-order 8kHz Low-Pass Filter
      2. ITU-R BS.1770-4 -14.0 LUFS Loudness Normalization
    """
    if y is None or len(y) == 0:
        return y

    processed = y
    if apply_filter:
        processed = apply_lowpass_filter(processed, sr=sr, cutoff=cutoff)
    if apply_lufs:
        processed = normalize_lufs(processed, sr=sr, target_lufs=target_lufs)

    return processed


def extract_features_from_signal(
    y,
    sr=22050,
    file_name="signal",
    file_size_kb=0.0,
    full_duration=None,
    native_sr=22050,
    native_channels=1,
    preprocess=True,
):
    """
    Extracts the full Tier 1 acoustic feature set directly from an in-memory mono audio array.
    Optionally applies standard Phase 0 preprocessing (8kHz lowpass + -14 LUFS normalization).
    """
    # Avoid zero-length signals
    if len(y) == 0:
        y = np.zeros(int(sr * 0.1), dtype=np.float32)

    if preprocess:
        y = preprocess_audio_signal(y, sr=sr)

    loaded_duration = float(librosa.get_duration(y=y, sr=sr))
    if full_duration is None:
        full_duration = loaded_duration

    # 3. Dynamic & Temporal Features
    # Tempo (BPM) & Beat tracking
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    tempo_val = float(np.atleast_1d(tempo)[0])

    # RMS Energy (loudness & dynamic variation)
    rms = librosa.feature.rms(y=y)
    rms_mean = float(np.mean(rms))
    rms_var = float(np.var(rms))

    # Zero Crossing Rate (percussiveness / noisiness)
    zcr = librosa.feature.zero_crossing_rate(y=y)
    zcr_mean = float(np.mean(zcr))
    zcr_var = float(np.var(zcr))

    # 4. Spectral Descriptors
    # Spectral Centroid (brightness)
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)
    sc_mean = float(np.mean(sc))
    sc_var = float(np.var(sc))

    # Spectral Bandwidth (spread of frequencies around centroid)
    sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    sb_mean = float(np.mean(sb))
    sb_var = float(np.var(sb))

    # Spectral Rolloff (frequency below which 85% of spectral energy lies)
    ro = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    ro_mean = float(np.mean(ro))
    ro_var = float(np.var(ro))

    # Spectral Contrast (energy difference between peaks and valleys across 7 sub-bands)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
    contrast_mean = float(np.mean(contrast))
    contrast_var = float(np.var(contrast))
    contrast_bands_mean = [float(val) for val in np.mean(contrast, axis=1)]

    # 5. Timbral Descriptors (Full 20-coefficient MFCC vector)
    n_mfcc = 20
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_means = [float(v) for v in np.mean(mfcc, axis=1)]
    mfcc_vars = [float(v) for v in np.var(mfcc, axis=1)]

    # 6. Tonal Descriptors (Chroma - 12 pitch classes)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_means = [float(v) for v in np.mean(chroma, axis=1)]
    chroma_vars = [float(v) for v in np.var(chroma, axis=1)]

    # 7. Rhythmic & Onset Descriptors
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_strength_mean = float(np.mean(onset_env))
    onset_strength_var = float(np.var(onset_env))

    # Estimate onset rate (onsets per second)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onset_rate = float(len(onsets) / max(loaded_duration, 0.1))

    # Mel-Spectrogram summary
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mel_mean = float(np.mean(mel_spec))
    mel_var = float(np.var(mel_spec))

    # 9. Musical Key & Mode Detection (Phase 3 Task 3.2 - Krumhansl-Schmuckler algorithm)
    key_info = detect_musical_key(chroma_means)

    # 10. Time Signature / Meter Detection (Phase 4 Task 4.1)
    time_sig_info = detect_time_signature(y=y, sr=sr, onset_env=onset_env, tempo=tempo_val, beats=beats)

    # 11. Chord Progression Detection (Phase 4 Task 4.2)
    chord_info = detect_chord_progression(chroma=chroma, y=y, sr=sr)

    # 12. Vocal vs. Instrumental Detection (Phase 4 Task 4.3)
    vocal_info = detect_vocal_presence(y=y, sr=sr)

    # Return comprehensive technical features dictionary
    features = {
        # File & Signal Information
        "file_name": file_name,
        "file_size_kb": file_size_kb,
        "duration": round(loaded_duration, 2),
        "full_duration": round(full_duration, 2),
        "sample_rate": sr,
        "native_sample_rate": native_sr,
        "channels": native_channels,

        # Musical Key & Mode (Phase 3)
        "detected_key": key_info["detected_key"],
        "key_root": key_info["key_root"],
        "key_mode": key_info["key_mode"],
        "key_confidence": key_info["key_confidence"],

        # Audio Analysis Depth (Phase 4)
        "time_signature": time_sig_info["time_signature"],
        "meter_confidence": time_sig_info["meter_confidence"],
        "meter_type": time_sig_info["meter_type"],
        "chord_progression": chord_info["chord_progression"],
        "progression_str": chord_info["progression_str"],
        "top_chords": chord_info["top_chords"],
        "vocal_presence": vocal_info["vocal_presence"],
        "vocal_probability": vocal_info["vocal_probability"],
        "vocal_confidence": vocal_info["vocal_confidence"],

        # Dynamics & Tempo
        "tempo": round(tempo_val, 1),
        "rms_energy": round(rms_mean, 4),           # Backward-compatible alias
        "rms_mean": round(rms_mean, 6),
        "rms_var": round(rms_var, 6),
        "zero_crossing_rate": round(zcr_mean, 4),   # Backward-compatible alias
        "zcr_mean": round(zcr_mean, 6),
        "zcr_var": round(zcr_var, 6),

        # Spectral Metrics
        "spectral_centroid": round(sc_mean, 2),     # Backward-compatible alias
        "spectral_centroid_mean": round(sc_mean, 2),
        "spectral_centroid_var": round(sc_var, 2),
        "spectral_bandwidth_mean": round(sb_mean, 2),
        "spectral_bandwidth_var": round(sb_var, 2),
        "spectral_rolloff_mean": round(ro_mean, 2),
        "spectral_rolloff_var": round(ro_var, 2),
        "spectral_contrast_mean": round(contrast_mean, 2),
        "spectral_contrast_var": round(contrast_var, 2),
        "spectral_contrast_bands": contrast_bands_mean,

        # MFCCs (Full 20-coefficient vector)
        "mfcc_mean": round(float(np.mean(mfcc_means)), 2), # Backward-compatible summary
        "mfcc_means": mfcc_means,
        "mfcc_vars": mfcc_vars,

        # Chroma (12 pitch classes)
        "chroma_means": chroma_means,
        "chroma_vars": chroma_vars,

        # Rhythm & Onsets
        "onset_strength_mean": round(onset_strength_mean, 4),
        "onset_strength_var": round(onset_strength_var, 4),
        "onset_rate": round(onset_rate, 2),

        # Mel-Spectrogram summary
        "mel_spectrogram_mean": round(mel_mean, 4),
        "mel_spectrogram_var": round(mel_var, 4),
    }

    return features


# Krumhansl-Schmuckler Key Profiles (Krumhansl & Kessler, 1982)
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=np.float64)
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=np.float64)
_PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def detect_musical_key(chroma) -> dict:
    """Detects musical key and mode (Major / Minor) using Krumhansl-Schmuckler algorithm.

    Correlates the 12-element chroma distribution with standard cognitive key profiles.
    Returns:
        dict with 'detected_key', 'key_root', 'key_mode', 'key_confidence'.
    """
    if chroma is None:
        return {"detected_key": "Unknown", "key_root": "Unknown", "key_mode": "Unknown", "key_confidence": 0.0}

    chroma_vec = np.asarray(chroma, dtype=np.float64)
    if len(chroma_vec) < 12 or np.all(chroma_vec == 0):
        return {"detected_key": "Unknown", "key_root": "Unknown", "key_mode": "Unknown", "key_confidence": 0.0}

    chroma_vec = chroma_vec[:12]
    c_mean = np.mean(chroma_vec)
    c_std = np.std(chroma_vec)
    if c_std < 1e-6:
        return {"detected_key": "Unknown", "key_root": "Unknown", "key_mode": "Unknown", "key_confidence": 0.0}

    chroma_norm = chroma_vec - c_mean
    maj_norm = _MAJOR_PROFILE - np.mean(_MAJOR_PROFILE)
    min_norm = _MINOR_PROFILE - np.mean(_MINOR_PROFILE)

    best_corr = -2.0
    best_root = "C"
    best_mode = "Major"
    best_key = "C Major"

    for i in range(12):
        # Rotate profiles to match candidate root key i
        maj_rot = np.roll(maj_norm, i)
        min_rot = np.roll(min_norm, i)

        norm_prod_maj = np.linalg.norm(chroma_norm) * np.linalg.norm(maj_rot)
        norm_prod_min = np.linalg.norm(chroma_norm) * np.linalg.norm(min_rot)

        r_maj = float(np.dot(chroma_norm, maj_rot) / norm_prod_maj) if norm_prod_maj > 0 else 0.0
        r_min = float(np.dot(chroma_norm, min_rot) / norm_prod_min) if norm_prod_min > 0 else 0.0

        if r_maj > best_corr:
            best_corr = r_maj
            best_root = _PITCH_NAMES[i]
            best_mode = "Major"
            best_key = f"{best_root} Major"

        if r_min > best_corr:
            best_corr = r_min
            best_root = _PITCH_NAMES[i]
            best_mode = "Minor"
            best_key = f"{best_root} Minor"

    confidence = round(max(0.0, min(1.0, float((best_corr + 1.0) / 2.0))), 3)
    return {
        "detected_key": best_key,
        "key_root": best_root,
        "key_mode": best_mode,
        "key_confidence": confidence,
    }


# ===========================================================================
# Phase 4 Audio Analysis Depth Algorithms
# ===========================================================================

def detect_time_signature(
    y: np.ndarray = None,
    sr: int = 22050,
    onset_env: np.ndarray = None,
    tempo: float = None,
    beats: np.ndarray = None,
) -> dict:
    """
    Estimates the musical time signature / meter (4/4, 3/4, 6/8, 5/4).
    Uses beat-synchronous onset strength autocorrelation across bar pulse lengths.
    """
    default_result = {
        "time_signature": "4/4",
        "meter_confidence": 0.70,
        "meter_type": "Common Time (4/4)",
    }

    if y is None and onset_env is None:
        return default_result

    try:
        if onset_env is None and y is not None:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)

        if beats is None and y is not None:
            _, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

        if beats is None or len(beats) < 6:
            return default_result

        valid_beats = beats[beats < len(onset_env)]
        if len(valid_beats) < 6:
            return default_result

        beat_strengths = onset_env[valid_beats]
        b_mean = np.mean(beat_strengths)
        b_std = np.std(beat_strengths)
        if b_std < 1e-4:
            return default_result

        b_norm = (beat_strengths - b_mean) / b_std
        n = len(b_norm)

        corrs = {}
        for lag in [2, 3, 4, 5, 6]:
            if n > lag:
                c = np.dot(b_norm[:-lag], b_norm[lag:]) / (n - lag)
                corrs[lag] = float(c)
            else:
                corrs[lag] = 0.0

        r3 = corrs.get(3, 0.0)
        r4 = corrs.get(4, 0.0)
        r5 = corrs.get(5, 0.0)
        r6 = corrs.get(6, 0.0)

        # Decision tree
        if r5 > 0.28 and r5 > max(r3, r4) * 1.15:
            return {
                "time_signature": "5/4",
                "meter_confidence": round(min(0.95, r5 + 0.5), 2),
                "meter_type": "Quintuple (5/4)",
            }
        elif r3 > 0.18 and r3 > r4 * 1.05:
            if r6 > r3 * 1.15:
                return {
                    "time_signature": "6/8",
                    "meter_confidence": round(min(0.95, r6 + 0.5), 2),
                    "meter_type": "Compound Duple (6/8)",
                }
            else:
                return {
                    "time_signature": "3/4",
                    "meter_confidence": round(min(0.95, r3 + 0.5), 2),
                    "meter_type": "Triple (3/4)",
                }
        else:
            conf = round(min(0.95, max(0.65, r4 + 0.5)), 2)
            return {
                "time_signature": "4/4",
                "meter_confidence": conf,
                "meter_type": "Common Time (4/4)",
            }
    except Exception:
        return default_result


# 12 chromatic pitch classes for triads
_CHORD_PITCHES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# 24 canonical triad templates (12 Major, 12 Minor)
_CHORD_TEMPLATES = {}
for _idx, _root in enumerate(_CHORD_PITCHES):
    _maj = np.zeros(12, dtype=np.float32)
    _maj[_idx] = 1.0
    _maj[(_idx + 4) % 12] = 0.8
    _maj[(_idx + 7) % 12] = 0.9
    _CHORD_TEMPLATES[_root] = _maj / np.linalg.norm(_maj)

    _min = np.zeros(12, dtype=np.float32)
    _min[_idx] = 1.0
    _min[(_idx + 3) % 12] = 0.8
    _min[(_idx + 7) % 12] = 0.9
    _CHORD_TEMPLATES[f"{_root}m"] = _min / np.linalg.norm(_min)


def detect_chord_progression(
    chroma: np.ndarray = None,
    y: np.ndarray = None,
    sr: int = 22050,
    hop_length: int = 512,
) -> dict:
    """
    Detects dominant chords and progression sequence using chroma template matching.
    Returns:
        dict with chord_progression (list), progression_str (str), top_chords (list of dicts).
    """
    default_result = {
        "chord_progression": ["C", "G", "Am", "F"],
        "progression_str": "C → G → Am → F",
        "top_chords": [{"chord": "C", "percentage": 30.0}, {"chord": "G", "percentage": 25.0}],
    }

    try:
        if chroma is None:
            if y is None:
                return default_result
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)

        if chroma is None or chroma.shape[1] == 0:
            return default_result

        # Downsample into ~0.45s chunks (20 frames)
        chunk_size = 20
        n_frames = chroma.shape[1]

        chord_sequence = []
        for start in range(0, n_frames, chunk_size):
            chunk = chroma[:, start : min(start + chunk_size, n_frames)]
            if chunk.shape[1] == 0:
                continue
            mean_vec = np.mean(chunk, axis=1)
            norm = np.linalg.norm(mean_vec)
            if norm < 1e-4:
                continue
            mean_vec = mean_vec / norm

            best_chord = "C"
            best_sim = -1.0
            for chord_name, template in _CHORD_TEMPLATES.items():
                sim = float(np.dot(mean_vec, template))
                if sim > best_sim:
                    best_sim = sim
                    best_chord = chord_name
            chord_sequence.append(best_chord)

        if not chord_sequence:
            return default_result

        compressed = []
        counts = {}
        for c in chord_sequence:
            counts[c] = counts.get(c, 0) + 1
            if not compressed or compressed[-1] != c:
                compressed.append(c)

        total_chunks = len(chord_sequence)
        top_chords = [
            {"chord": k, "percentage": round((v / total_chunks) * 100, 1)}
            for k, v in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:5]
        ]

        progression = []
        for c in compressed:
            if c not in progression:
                progression.append(c)
            elif len(progression) >= 3 and c == progression[0]:
                break
            if len(progression) >= 6:
                break

        if len(progression) < 2:
            progression = [item["chord"] for item in top_chords[:4]]

        prog_str = " → ".join(progression)

        return {
            "chord_progression": progression,
            "progression_str": prog_str,
            "top_chords": top_chords,
        }
    except Exception:
        return default_result


def detect_vocal_presence(y: np.ndarray, sr: int = 22050) -> dict:
    """
    Detects presence of human vocals vs. instrumental performance.
    Combines vocal formant frequency band ratio (300-3400 Hz),
    harmonic-to-percussive separation, and timbral formant variance.
    Returns:
        dict with vocal_presence ('Vocal' | 'Instrumental'), vocal_probability, vocal_confidence.
    """
    default_result = {
        "vocal_presence": "Instrumental",
        "vocal_probability": 0.25,
        "vocal_confidence": 0.50,
    }

    if y is None or len(y) == 0:
        return default_result

    try:
        # Harmonic vs Percussive energy separation
        y_harm, y_perc = librosa.effects.hpss(y)
        harm_energy = float(np.sum(y_harm ** 2))
        perc_energy = float(np.sum(y_perc ** 2))
        total_energy = harm_energy + perc_energy + 1e-9
        harm_ratio = harm_energy / total_energy

        # Vocal Formant Band Ratio (300 Hz - 3400 Hz)
        stft = np.abs(librosa.stft(y, n_fft=1024, hop_length=512))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
        vocal_mask = (freqs >= 300) & (freqs <= 3400)
        vocal_band_energy = float(np.sum(stft[vocal_mask, :] ** 2))
        full_band_energy = float(np.sum(stft ** 2)) + 1e-9
        vocal_band_ratio = vocal_band_energy / full_band_energy

        # Dynamic Spectral Variance in Vocal Band (phoneme transitions)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=8)
        formant_var = float(np.mean(np.var(mfcc[1:5, :], axis=1)))

        var_norm = min(1.0, formant_var / 80.0)
        vocal_score = (0.35 * harm_ratio) + (0.45 * min(1.0, vocal_band_ratio * 1.5)) + (0.20 * var_norm)

        # Logistic sigmoid centered at 0.52
        prob = 1.0 / (1.0 + np.exp(-10.0 * (vocal_score - 0.52)))
        prob = float(np.clip(prob, 0.05, 0.95))

        is_vocal = prob >= 0.50
        vocal_tag = "Vocal" if is_vocal else "Instrumental"
        confidence = float(round(abs(prob - 0.5) * 2.0, 2))

        return {
            "vocal_presence": vocal_tag,
            "vocal_probability": round(prob, 3),
            "vocal_confidence": max(0.50, confidence),
        }
    except Exception:
        return default_result



def analyze_audio(file_path, duration_cap=60.0, offset=0.0):
    """
    Loads an audio file and extracts the full Tier 1 acoustic feature set.

    Parameters:
        file_path (str): Path to audio file (.wav, .mp3, etc.)
        duration_cap (float): Maximum seconds of audio to analyze (default: 60.0s).
        offset (float): Start reading after this time (in seconds). Default: 0.0.

    Returns:
        dict: Structured dictionary containing technical audio measurements.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    # 1. File Metadata
    file_name = os.path.basename(file_path)
    file_size_kb = round(os.path.getsize(file_path) / 1024, 2)

    # Inspect native channel count and duration
    try:
        info = sf.info(file_path)
        native_channels = int(info.channels)
        full_duration = float(info.duration)
        native_sr = int(info.samplerate)
    except Exception:
        native_channels = 1
        full_duration = float(librosa.get_duration(path=file_path))
        native_sr = 22050

    # 2. Load audio signal (mono=True for consistent feature extraction)
    target_sr = 22050
    y, sr = librosa.load(file_path, sr=target_sr, offset=offset, duration=duration_cap, mono=True)

    return extract_features_from_signal(
        y=y,
        sr=sr,
        file_name=file_name,
        file_size_kb=file_size_kb,
        full_duration=full_duration,
        native_sr=native_sr,
        native_channels=native_channels,
        preprocess=True,
    )
