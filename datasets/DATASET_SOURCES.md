# CLASSIFY — Dataset Sources, Licensing, and Curation Methodology

## 1. Executive Summary

This document specifies the legitimate music audio datasets, licenses, fault-filtering exclusions, and artist-grouping methodologies utilized in the CLASSIFY ML Accuracy Upgrade.

To meet the product's 11-genre taxonomy requirements without fabricating audio or violating copyright, the dataset integrates:
1. **GTZAN Music Genre Dataset** (10 genres: Blues, Classical, Country, Disco, Hip-Hop, Jazz, Metal, Pop, Reggae, Rock), filtered using the published Kereliuk & Sturm fault-filter list to eliminate duplicates, corrupted files, and documented mislabelings.
2. **Free Music Archive (FMA)** (Ambient genre), filtered to legitimate Creative Commons-licensed audio with true per-track artist and album metadata.
3. **Legacy Synthetic Signal Retention**: The original 72 synthetic sine/noise tracks from Phase 1 (`CLASSIFY-Audio-Engine`) are preserved in `model_artifacts/baseline_v0_synthetic_72/` strictly for historical reference and regression testing, but are excluded from the trained production manifest.

---

## 2. GTZAN Dataset (10 Genres)

### Origin & Citation
- **Original Authors**: George Tzanetakis and Perry Cook (2002), *Musical Genre Classification of Audio Signals*, IEEE Transactions on Speech and Audio Processing.
- **Marsyas Repository**: `http://marsyas.info/downloads/datasets.html`
- **Mirror Distribution**: Hugging Face `m-a-p/GTZAN` (`https://huggingface.co/datasets/m-a-p/GTZAN`)
- **License**: Educational / Academic Research Use (Public Domain & Fair Use Research Benchmark terms per Marsyas project distribution).

### Controlled Vocabulary Label Mapping
GTZAN's 10 genre directory names map 1:1 to CLASSIFY's pedagogical categories:
- `blues` $\rightarrow$ **Blues**
- `classical` $\rightarrow$ **Classical**
- `country` $\rightarrow$ **Country**
- `disco` $\rightarrow$ **Disco**
- `hiphop` $\rightarrow$ **Hip-Hop**
- `jazz` $\rightarrow$ **Jazz**
- `metal` $\rightarrow$ **Metal**
- `pop` $\rightarrow$ **Pop**
- `reggae` $\rightarrow$ **Reggae**
- `rock` $\rightarrow$ **Rock**

### Fault-Filtered Exclusion & Cleaning
As documented by Bob L. Sturm (2013, 2014) and Corey Kereliuk et al. (2015, *Deep Learning and Music Adversaries*), the raw 1,000-track GTZAN dataset contains ~5% exact duplicates, distorted files, and mislabeled excerpts.
- **Source of Filtering**: Published Kereliuk & Sturm partition lists (`train_filtered.txt`, `valid_filtered.txt`, `test_filtered.txt`), mirrored in `jongpillee/music_dataset_split` and `m-a-p/GTZAN`.
- **Exclusion Count**: 71 problematic / corrupted / duplicate tracks are strictly excluded.
- **Usable Total**: 929 clean audio tracks across the 10 genres (~93 tracks per genre).

### Artist ID Proxy & Documented Limitation
Because the original GTZAN collection did not package verified per-track artist metadata, artist isolation for GTZAN tracks is derived via the fault-filtered artist-cluster partitions (Sturm & Kereliuk artist groupings) as the **documented artist proxy** (`artist_id = gtzan_proxy_{genre}_{cluster_id}`). This prevents track excerpts from the same recording session or artist from crossing train/validation/test split boundaries.

---

## 3. Free Music Archive (FMA) — Ambient Class

### Origin & Citation
- **Authors**: Michaël Defferrard, Kirell Benzi, Pierre Vandergheynst, Xavier Bresson (ISMIR 2017), *FMA: A Dataset For Music Analysis*, arXiv:1612.01840.
- **Repository**: `https://github.com/mdeff/fma` / Switch.ch mirror (`https://os.unil.cloud.switch.ch/fma/`) / Hugging Face.
- **Metadata**: Extracted from `fma_metadata/tracks.csv` and `fma_metadata/genres.csv`.

### Selection Criteria
- Genre Filter: Tracks tagged with Genre ID `107` (`Ambient`) in `track.genres_all`.
- Subset: Curated 30-second clips matching the duration profile of the GTZAN benchmark.
- Licensing: Verified Creative Commons licenses per track (e.g. CC-BY, CC-BY-SA, CC-BY-NC).
- Diversity: Curated across 40+ distinct independent artists with verified `artist_id` metadata. No artist proxies required.

---

## 4. Dataset Quality Verification Standards

Every candidate file is vetted through [`classify/ml/dataset_quality.py`](file:///d:/Classify%20-%20Song%20Classification%20System%20for%20Music%20Schools/classify/ml/dataset_quality.py):
1. **Decodability**: Verified through `soundfile` and `librosa.load(sr=22050, mono=True)`.
2. **Duration**: $\ge 5.0$s duration floor; tracks under 5 seconds are rejected.
3. **Audio Integrity**: RMS energy floor of $0.0005$ to reject silence and corrupted muted signals.
4. **Duplicate Detection**: SHA-256 hash computed over decoded float32 audio arrays to eliminate re-encoded duplicate tracks.
5. **Quality Logging**: Full accounts of accepted/rejected candidate files are recorded in `datasets/quality_report.json`.

---

## 5. Phase 1 Expansion Strategy: MTG-Jamendo & FMA Medium

To eliminate the GTZAN domain shift and achieve $\ge 80\%$ multi-environment accuracy, Phase 1 establishes an augmentation pipeline incorporating contemporary digital masters (2010–2024).

### 5.1 MTG-Jamendo Dataset
- **Authors**: Dmitry Bogdanov et al. (ISMIR 2019), *The MTG-Jamendo Dataset for Music Auto-Tagging*.
- **Repository**: `https://github.com/MTG/mtg-jamendo-dataset`
- **Audio Scope**: 55,000 full audio tracks in 320kbps MP3, all licensed under Creative Commons.
- **Taxonomy Alignment**:
  - `genre---blues` $\rightarrow$ **Blues**
  - `genre---classical` $\rightarrow$ **Classical**
  - `genre---country` $\rightarrow$ **Country**
  - `genre---disco` $\rightarrow$ **Disco**
  - `genre---hiphop` $\rightarrow$ **Hip-Hop**
  - `genre---jazz` $\rightarrow$ **Jazz**
  - `genre---metal`, `genre---heavymetal` $\rightarrow$ **Metal**
  - `genre---pop` $\rightarrow$ **Pop**
  - `genre---reggae` $\rightarrow$ **Reggae**
  - `genre---rock` $\rightarrow$ **Rock**
  - `genre---ambient` $\rightarrow$ **Ambient**

### 5.2 FMA Medium Subset
- **Authors**: Michaël Defferrard et al. (ISMIR 2017).
- **Scope**: 25,000 tracks (30s clips), 16 root genres with granular subgenre trees.
- **Licensing**: Verified Creative Commons (CC-BY, CC-BY-NC, CC0).
- **Subgenre Utility**: Directly provides training labels for hierarchical subgenres (e.g. Bebop, Synth-pop, Black Metal, Bluegrass, Afrobeat).

### 5.3 Ingestion & Harmonization Pipeline
All newly ingested tracks from modern repositories MUST pass the Phase 0 Harmonization Protocol before training feature extraction:
1. **Loudness Normalization**: Target integrated loudness strictly normalized to **-14.0 LUFS** via `pyloudnorm`.
2. **Bandwidth Harmonization**: 4th-order Butterworth low-pass filter at 8,000 Hz applied to align modern 20kHz masters with the GTZAN acoustic baseline.
3. **RMS-Weighted Energy Pooling**: 3-window segmentation (20%, 50%, 80%) aggregated with energy weighting.
4. **Target Volume**: 200+ tracks per genre to support deep embedding fine-tuning.

---

## 6. 12-Class Expansion & Modern Balance Top-Up (Solutions Architecture)

To resolve P-001 (R&B absence), P-002 (Reggae/Hip-Hop confusion), and P-003 (Jazz domain shift), the production manifest was expanded from 1,019 tracks to 1,508 tracks incorporating real Creative Commons audio from FMA and MTG-Jamendo:

### 6.1 Genre Breakdown (1,508 Tracks, 618 Artists)
| Genre | Total Tracks | Sources Breakdown | Distinct Artists |
|:------|:-------------|:------------------|:-----------------|
| **R&B** | 154 | 94 FMA Soul-RnB (`CC-BY`, `CC-BY-NC`), 60 MTG-Jamendo (`CC-BY`, `CC-BY-SA`, `CC-BY-NC-SA`) | 119 |
| **Reggae** | 151 | 86 GTZAN, 65 MTG-Jamendo | 73 |
| **Jazz** | 151 | 86 GTZAN, 65 MTG-Jamendo | 73 |
| **Country** | 138 | 98 GTZAN, 40 MTG-Jamendo | 47 |
| **Hip-Hop** | 137 | 92 GTZAN, 45 MTG-Jamendo | 52 |
| **Rock** | 135 | 100 GTZAN, 35 MTG-Jamendo | 42 |
| **Disco** | 133 | 93 GTZAN, 40 MTG-Jamendo | 45 |
| **Pop** | 129 | 84 GTZAN, 45 MTG-Jamendo | 49 |
| **Blues** | 100 | 100 GTZAN | 30 |
| **Classical** | 99 | 99 GTZAN | 30 |
| **Metal** | 91 | 91 GTZAN | 30 |
| **Ambient** | 90 | 90 FMA | 28 |
| **TOTAL** | **1,508** | **929 GTZAN, 184 FMA, 395 MTG-Jamendo** | **618** |

### 6.2 Curation & Integrity Constraints
- **Artist Cap**: Maximum 3 tracks per artist for newly ingested modern tracks.
- **Zero Leakage**: All 618 artists partitioned strictly into either train, val, or test splits with 0 cross-split artist overlap.
- **Legacy Invariant**: 100% of the 1,019 original tracks retain their frozen split assignments (684 train / 177 val / 158 test).
- **Licensing Compliance**: All added tracks are licensed under open Creative Commons terms (CC-BY, CC-BY-NC, CC-BY-SA, CC0, LAL) and documented in `datasets/manifest.csv`.
- **Quality Audit**: 1,508 / 1,508 tracks passed automated quality vetting (`classify/ml/dataset_quality.py`) with 0 corrupted files or sub-threshold clips.

