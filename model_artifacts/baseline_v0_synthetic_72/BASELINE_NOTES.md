# Baseline v0 (72 Synthetic Tracks) Notes

## Overview
This snapshot preserves the initial state of the CLASSIFY ML model and evaluation artifacts prior to the ML Accuracy Upgrade, retained solely as a historical reproducibility reference.

## Dataset Characteristics
- **Dataset Size**: 72 tracks across 6 classes (`Ambient`, `Classical`, `Jazz`, `Metal`, `Pop`, `Rock`).
- **Nature of Audio**: 100% synthetic audio generated via sine-wave harmonic stacks and synthetic noise bursts (`CLASSIFY-Audio-Engine`), 3.0 seconds duration each.
- **Artists**: 4 synthetic artist IDs per genre (24 artists total, 3 tracks per artist).
- **Limitation**: The model has never seen real, recorded music.

## Evaluation Split Bug & The "100% Accuracy" Artifact
The reported 100% test accuracy in `evaluation_report.json` was an artifact of a structural splitting flaw:
- `classify/ml/train.py` utilized a single global `GroupShuffleSplit(test_size=0.15)` across all 6 classes pooled together.
- Because there were only 4 artists per class across 24 total artists, the global group split randomly selected 4 artists for the test fold that belonged to only 3 classes:
  - `Ambient`: 3 tracks (1 artist)
  - `Jazz`: 3 tracks (1 artist)
  - `Pop`: 6 tracks (2 artists)
  - `Classical`: 0 tracks (0 artists)
  - `Metal`: 0 tracks (0 artists)
  - `Rock`: 0 tracks (0 artists)
- Furthermore, `classify/ml/evaluate.py` computed the confusion matrix only across classes present in `y_true | y_pred` (`labels = sorted(list(set(y_true) | set(y_pred)))`), silently omitting `Classical`, `Metal`, and `Rock`.
- Resulting confusion matrix:
  ```json
  [
    [3, 0, 0],
    [0, 3, 0],
    [0, 0, 6]
  ]
  ```
- **Conclusion**: The 100% accuracy was arithmetically correct on the broken test split, but completely invalid as a measure of 6-class generalization.

## Retained Artifacts
- `genre_classifier.joblib`: Trained Random Forest model (un-tuned, `n_estimators=100`, `max_depth=10`).
- `scaler.joblib`: StandardScaler fit on train+val.
- `label_encoder.joblib`: LabelEncoder for the 6 synthetic classes.
- `evaluation_report.json`: Original evaluation report.
