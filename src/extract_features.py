#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_features.py

Extracts EEG features from preprocessed .npz files and builds the
feature matrix CSV used as input for classification.

For each recording, per-channel temporal and spectral features are computed.
An intra-subject reactivity measure (d_*) is also derived for each feature:
    d_feature = feature_emotion - mean_feature_across_emotions (per subject)

Input:
    Directory tree:  BASE_IN/{HC,AD}/{condition}/subject_N.npz
    (output of preprocess_eeg.py)

Output:
    OUT_CSV  — feature matrix CSV (features_ml_full.csv)
    emotion_mapping.json — integer encoding of emotion labels

Usage:
    Set BASE_IN and OUT_CSV in the CONFIG section, then:
        python src/extract_features.py
"""

import os, glob, re, json
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import skew, kurtosis

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_IN = "data/processed"                              # TO BE SET BY USER
OUT_CSV = "data/features_ml_full.csv"                  # TO BE SET BY USER
# ─────────────────────────────────────────────────────────────────────────────

FS_FALLBACK = 128.0

BANDS = {
    "delta": (1,  4),
    "theta": (4,  8),
    "alpha": (8,  13),
    "beta":  (13, 30),
    "gamma": (30, 40),
}
TOTAL_BAND = (1, 45)

CH_NAMES = ["AF3","F7","F3","FC5","T7","P7","O1","O2","P8","T8","FC6","F4","F8","AF4"]
FRONTAL  = ["AF3","AF4","F3","F4","F7","F8","FC5","FC6"]


def bandpower(f, psd, band):
    lo, hi = band
    mask = (f >= lo) & (f < hi)
    if not np.any(mask):
        return np.nan
    return np.sum(psd[mask]) * np.mean(np.diff(f))


def spectral_entropy(psd):
    p = psd / np.sum(psd)
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


def temporal_entropy(x, bins=100):
    hist, _ = np.histogram(x, bins=bins, density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))


def get_subject_emotion(path):
    parts = os.path.normpath(path).split(os.sep)
    group   = parts[-3]
    emotion = parts[-2]
    sid     = re.search(r"subject_(\d+)\.npz", os.path.basename(path)).group(1)
    return group, emotion, sid


def extract_features(npz_path):
    """
    Extracts per-channel temporal and spectral features from one .npz file.
    Returns a dict (one row of the feature matrix).
    """
    z     = np.load(npz_path, allow_pickle=True)
    data  = z["data"]
    fs    = float(z["fs"]) if "fs" in z else FS_FALLBACK
    group, emotion, sid = get_subject_emotion(npz_path)

    row = {
        "file":       npz_path,
        "subject_id": sid,
        "group":      1 if group == "AD" else 0,
        "emotion":    emotion,
    }

    for ci, ch in enumerate(CH_NAMES):
        x = np.nan_to_num(data[ci])
        f, psd    = welch(x, fs=fs, nperseg=int(fs * 2))
        total_pw  = bandpower(f, psd, TOTAL_BAND)

        # Temporal features
        row[f"mean_{ch}"] = np.mean(x)
        row[f"std_{ch}"]  = np.std(x)
        row[f"rms_{ch}"]  = np.sqrt(np.mean(x ** 2))
        row[f"skew_{ch}"] = skew(x)
        row[f"kurt_{ch}"] = kurtosis(x)

        # Spectral power per band
        for b, br in BANDS.items():
            bp = bandpower(f, psd, br)
            row[f"abs_{b}_{ch}"] = bp
            row[f"rel_{b}_{ch}"] = bp / total_pw if total_pw and total_pw > 0 else np.nan

        # Entropy
        row[f"entropy_spec_{ch}"] = spectral_entropy(psd)
        row[f"entropy_time_{ch}"] = temporal_entropy(x)

    # Frontal aggregates (mean across frontal channels)
    for b in BANDS:
        row[f"rel_{b}_frontal"] = np.nanmean([row[f"rel_{b}_{c}"] for c in FRONTAL])
        row[f"abs_{b}_frontal"] = np.nanmean([row[f"abs_{b}_{c}"] for c in FRONTAL])

    # Frontal alpha asymmetry index
    try:
        row["FAI_alpha"] = np.log(row["abs_alpha_F4"]) - np.log(row["abs_alpha_F3"])
    except Exception:
        row["FAI_alpha"] = np.nan

    # Cross-band ratios (frontal)
    row["ratio_theta_alpha"] = row["abs_theta_frontal"] / row["abs_alpha_frontal"]
    row["ratio_beta_alpha"]  = row["abs_beta_frontal"]  / row["abs_alpha_frontal"]

    return row


def main():
    files = sorted(
        glob.glob(os.path.join(BASE_IN, "HC", "*", "*.npz")) +
        glob.glob(os.path.join(BASE_IN, "AD", "*", "*.npz"))
    )
    if not files:
        print("No .npz files found in", BASE_IN)
        return

    rows = []
    for f in files:
        try:
            rows.append(extract_features(f))
        except Exception as e:
            print(f"[SKIP] {os.path.basename(f)}: {e}")

    df = pd.DataFrame(rows)

    # Integer encoding of emotion labels
    emo2id = {e: i for i, e in enumerate(sorted(df["emotion"].unique()))}
    df["emotion_id"] = df["emotion"].map(emo2id)

    feat_cols = [c for c in df.columns
                 if c not in {"file", "subject_id", "group", "emotion", "emotion_id"}]

    # Intra-subject reactivity: d_feature = feature - subject mean across conditions
    subj_mean = df.groupby("subject_id")[feat_cols].mean()
    for c in feat_cols:
        df[f"d_{c}"] = np.nan
    for i, r in df.iterrows():
        sid = r["subject_id"]
        if sid in subj_mean.index:
            for c in feat_cols:
                df.at[i, f"d_{c}"] = r[c] - subj_mean.loc[sid, c]

    os.makedirs(os.path.dirname(OUT_CSV) or ".", exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    mapping_path = os.path.join(os.path.dirname(OUT_CSV) or ".", "emotion_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump(emo2id, f, indent=2)

    print(f"Output CSV : {OUT_CSV}  ({df.shape[0]} rows)")
    print(f"Features   : {len(feat_cols)} (+ {len(feat_cols)} reactivity d_* columns)")
    print(f"Emotion map: {mapping_path}")


if __name__ == "__main__":
    main()
