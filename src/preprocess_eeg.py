#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess_eeg.py

Preprocesses 14-channel EPOC EEG recordings (.mat files) and saves
cleaned signals as .npz files.

Input directory structure:
    BASE_IN/{HC,AD}/{condition}/subject_N.mat

Output mirrors input structure under BASE_OUT.
"""

import os, re, glob
import numpy as np
from scipy.io import loadmat
import mne

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_IN  = "data/raw"       # root folder containing HC/ and AD/ subfolders
BASE_OUT = "data/processed" # output folder for .npz files

FS_DEFAULT       = 128.0   # sampling frequency in Hz
UNITS_ARE_UV     = True    # True if signals are in microvolts
MAX_BAD_CHANNELS = 3       # skip file if more than this many bad channels
MAX_NAN_FRAC     = 0.20    # channel is bad if NaN fraction exceeds this
BAD_STD_THRESH   = 1e-6    # channel is flat if std (µV) is below this
BP_LO, BP_HI     = 0.5, 45.0
NOTCH_HZ         = 50.0

CH_NAMES = ["AF3","F7","F3","FC5","T7","P7","O1","O2","P8","T8","FC6","F4","F8","AF4"]
# ─────────────────────────────────────────────────────────────────────────────


def parse_meta(path):
    parts = os.path.normpath(path).split(os.sep)
    group, condition = None, None
    for i, p in enumerate(parts):
        if p in ("HC", "AD") and i + 1 < len(parts):
            group, condition = p, parts[i + 1]
            break
    m = re.match(r"subject_(\d+)\.mat", os.path.basename(path))
    return group, condition, (m.group(1) if m else None)


def load_mat(path):
    try:
        d = loadmat(path, squeeze_me=True, struct_as_record=False)
        return {k: v for k, v in d.items() if not k.startswith("__")}
    except Exception:
        import h5py
        out = {}
        with h5py.File(path, "r") as f:
            def rd(o):
                if isinstance(o, h5py.Dataset): return o[()]
                if isinstance(o, h5py.Group):   return {k: rd(o[k]) for k in o}
                return o
            for k in f: out[k] = rd(f[k])
        return out


def pick_array(d):
    pick = None
    for k, v in d.items():
        a = np.array(v)
        if a.ndim == 2 and np.issubdtype(a.dtype, np.number):
            if pick is None or a.size > pick[1].size:
                pick = (k, a)
    return pick


def ensure_14xT(a):
    A = np.array(a)
    if A.shape[0] == 14: return A
    if A.shape[1] == 14: return A.T
    if 16 in A.shape:
        B = A if A.shape[0] in (14, 16) else A.T
        if B.shape[0] == 16: return B[:14, :]
    raise RuntimeError(f"Unexpected shape: {A.shape}")


def bad_channels(X_uv):
    bad = np.zeros(X_uv.shape[0], dtype=bool)
    for i, xi in enumerate(X_uv):
        if np.isnan(xi).mean() > MAX_NAN_FRAC or np.nanstd(xi) < BAD_STD_THRESH:
            bad[i] = True
    return bad


def process_one(path):
    group, condition, sid = parse_meta(path)
    sel = pick_array(load_mat(path))
    if sel is None:
        raise RuntimeError("No 2D array found.")
    key, arr = sel

    X = ensure_14xT(arr).astype(np.float64)
    bad = bad_channels(X)
    bad_chs = [CH_NAMES[i] for i, b in enumerate(bad) if b]

    if bad.sum() > MAX_BAD_CHANNELS:
        print(f"[SKIP] {os.path.basename(path)} — {bad.sum()} bad channels")
        return None

    scale = 1e-6 if UNITS_ARE_UV else 1.0
    info = mne.create_info(CH_NAMES, FS_DEFAULT, ch_types="eeg")
    raw  = mne.io.RawArray(X * scale, info, verbose=False)
    raw.set_montage(mne.channels.make_standard_montage("standard_1020"),
                    on_missing="warn", verbose=False)
    raw.info["bads"] = bad_chs
    raw.interpolate_bads(reset_bads=True, mode="accurate", verbose=False)
    raw.notch_filter([NOTCH_HZ], picks="eeg", verbose=False)
    raw.filter(BP_LO, BP_HI, picks="eeg", verbose=False)
    raw.set_eeg_reference("average", verbose=False)

    data = raw.get_data(picks="eeg")
    if UNITS_ARE_UV:
        data = data / scale
    data = data.astype(np.float32)

    rel = os.path.relpath(os.path.dirname(path), BASE_IN)
    out_dir = os.path.join(BASE_OUT, rel)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.basename(path).replace(".mat", ".npz"))

    np.savez_compressed(
        out_path,
        data=data, fs=float(FS_DEFAULT), ch_names=np.array(CH_NAMES, dtype=object),
        group=group, condition=condition, subject_id=sid,
        units=("microvolt" if UNITS_ARE_UV else "volt"),
        bad_channels_initial=np.array(bad_chs, dtype=object),
        reref="average", interp_method="spherical_spline",
        filters={"bandpass": [BP_LO, BP_HI], "notch": NOTCH_HZ},
    )
    print(f"[OK] {os.path.relpath(out_path, BASE_OUT)}")
    return out_path


def main():
    mats = sorted(
        glob.glob(os.path.join(BASE_IN, "HC", "*", "*.mat")) +
        glob.glob(os.path.join(BASE_IN, "AD", "*", "*.mat"))
    )
    if not mats:
        print("No .mat files found in", BASE_IN); return

    os.makedirs(BASE_OUT, exist_ok=True)
    ok = sk = 0
    for p in mats:
        try:
            ok += 1 if process_one(p) else 0
            sk += 0 if process_one(p) else 1
        except Exception as e:
            print(f"[ERR] {os.path.basename(p)}: {e}"); sk += 1

    print(f"\nDone — processed: {ok}, skipped/errors: {sk}")


if __name__ == "__main__":
    main()
