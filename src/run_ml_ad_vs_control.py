#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_ml_ad_vs_control.py

AD vs. HC classification from a pre-computed EEG feature matrix (CSV).
Runs 5 classifiers under subject-stratified GroupKFold cross-validation
across multiple feature subset strategies and comparison definitions.

Input:
    CSV_PATH  — feature matrix with columns: file, subject_id, group, emotion,
                plus any number of numeric feature columns.
                group values: 1/"AD"/"alzheimer" → AD; 0/"HC" → HC

Output:
    OUT_DIR/  — per-fold metrics, feature importances, JSON summaries,
                and aggregated SUMMARY_*.csv files.
"""

import os, re, json, itertools
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# ── CONFIG ────────────────────────────────────────────────────────────────────
CSV_PATH  = "data/features_ml_full.csv"  # path to input feature matrix
OUT_DIR   = "results/ml_output"          # root output directory

N_SPLITS     = 5
RANDOM_STATE = 42
FEATURE_REGEX_KEEP = None  # optional regex to filter feature columns; None = all

BANDS = ["delta", "theta", "alpha", "beta", "gamma"]

FRONTAL_CHANNELS = {"AF3","F7","F3","FC5","FC6","F4","F8","AF4"}

EMOTIONS     = ["affection","amusement","anger","fear","sadness","alzheimer","neutral"]
VALENCE_HIGH = {"affection", "amusement"}
VALENCE_LOW  = {"anger", "fear", "sadness"}
AROUSAL_HIGH = {"amusement", "anger", "fear"}
AROUSAL_LOW  = {"affection", "sadness"}
# ─────────────────────────────────────────────────────────────────────────────

MODEL_SPECS = {
    "LR_L1": Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler()),
        ("clf", LogisticRegression(penalty="l1", solver="liblinear", C=1.0,
                                   max_iter=4000, class_weight="balanced",
                                   random_state=RANDOM_STATE))
    ]),
    "SVM_RBF": Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler()),
        ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
                    probability=True, random_state=RANDOM_STATE))
    ]),
    "RF": Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(n_estimators=600, min_samples_leaf=2,
                                       n_jobs=-1, class_weight="balanced_subsample",
                                       random_state=RANDOM_STATE))
    ]),
    "XGB": Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                               subsample=0.7, colsample_bytree=0.7, min_child_weight=5,
                               reg_lambda=1.0, objective="binary:logistic",
                               eval_metric="logloss", n_jobs=-1, tree_method="hist",
                               importance_type="gain", random_state=RANDOM_STATE))
    ]),
    "CAT": Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", CatBoostClassifier(iterations=600, depth=5, learning_rate=0.05,
                                   l2_leaf_reg=3.0, loss_function="Logloss",
                                   verbose=0, random_seed=RANDOM_STATE))
    ]),
}


# ── DATA LOADING ──────────────────────────────────────────────────────────────

def load_data(path):
    df = pd.read_csv(path)
    required = {"file", "subject_id", "group", "emotion"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["emotion"] = df["emotion"].apply(lambda e: str(e).strip().lower().replace(" ", "_"))
    df["group"]   = df["group"].apply(lambda g: 1 if str(g).lower() in {"1","ad","alzheimer"} else 0)

    if "emotion_id" not in df.columns:
        df["emotion_id"] = pd.factorize(df["emotion"])[0]

    df["valence_bin"] = pd.Series(pd.NA, index=df.index, dtype="string")
    df.loc[df["emotion"].isin(VALENCE_HIGH), "valence_bin"] = "high"
    df.loc[df["emotion"].isin(VALENCE_LOW),  "valence_bin"] = "low"

    df["arousal_bin"] = pd.Series(pd.NA, index=df.index, dtype="string")
    df.loc[df["emotion"].isin(AROUSAL_HIGH), "arousal_bin"] = "high"
    df.loc[df["emotion"].isin(AROUSAL_LOW),  "arousal_bin"] = "low"

    meta = {"file","subject_id","group","emotion","emotion_id","valence_bin","arousal_bin"}
    feat_cols = [c for c in df.columns
                 if c not in meta and np.issubdtype(df[c].dtype, np.number) and df[c].notna().any()]
    return df, feat_cols


def filter_features(cols, pattern):
    if pattern is None: return cols
    rgx = re.compile(pattern)
    return [c for c in cols if rgx.search(c)]


# ── EVALUATION ────────────────────────────────────────────────────────────────

def get_proba(clf, X):
    if hasattr(clf, "predict_proba"):
        p = clf.predict_proba(X)
        if p.shape[1] == 2: return p[:, 1]
    if hasattr(clf, "decision_function"):
        s = clf.decision_function(X)
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn) if mx > mn else np.full_like(s, 0.5)
    return clf.predict(X).astype(float)


def feature_importance(model_name, pipe, feat_names, X_tr, y_tr, X_te, y_te):
    clf = pipe.named_steps["clf"]
    if model_name == "LR_L1":
        coef = getattr(clf, "coef_", None)
        imp = pd.Series(0.0, index=feat_names) if coef is None \
              else pd.Series(np.abs(coef.ravel()), index=feat_names)
    elif model_name == "SVM_RBF":
        Xte = pd.DataFrame(X_te, columns=feat_names) if not isinstance(X_te, pd.DataFrame) else X_te
        try:
            perm = permutation_importance(pipe, Xte, y_te, n_repeats=5,
                                          random_state=RANDOM_STATE, n_jobs=-1, scoring="f1_macro")
            imp = pd.Series(perm.importances_mean, index=feat_names)
        except Exception:
            imp = pd.Series(0.0, index=feat_names)
    elif model_name in ("RF", "XGB"):
        fi = getattr(clf, "feature_importances_", None)
        imp = pd.Series(fi if fi is not None else np.zeros(len(feat_names)), index=feat_names)
    elif model_name == "CAT":
        try:   imp = pd.Series(clf.get_feature_importance(), index=feat_names)
        except: imp = pd.Series(0.0, index=feat_names)
    else:
        imp = pd.Series(0.0, index=feat_names)
    return imp


def eval_gkf(X, y, groups, model_name, pipe, out_dir, label, feat_names):
    os.makedirs(out_dir, exist_ok=True)
    accs, f1s, aucs, per_fold, fi_list = [], [], [], [], []

    for fold, (tr, te) in enumerate(GroupKFold(n_splits=N_SPLITS).split(X, y, groups), 1):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        pipe.fit(X.iloc[tr], y[tr])
        yp  = pipe.predict(X.iloc[te])
        acc = accuracy_score(y[te], yp)
        f1  = f1_score(y[te], yp, average="macro")

        Xte = pipe.named_steps["imp"].transform(X.iloc[te])
        if "sc" in dict(pipe.named_steps):
            Xte = pipe.named_steps["sc"].transform(Xte)
        try:    auc = roc_auc_score(y[te], get_proba(pipe.named_steps["clf"], Xte))
        except: auc = np.nan

        accs.append(acc); f1s.append(f1); aucs.append(auc)
        per_fold.append({"fold": fold, "ACC": acc, "F1": f1, "AUC": auc})

        fi = feature_importance(model_name, pipe, feat_names,
                                 X.iloc[tr], y[tr], X.iloc[te], y[te])
        fi.sort_values(ascending=False).reset_index() \
          .rename(columns={"index":"feature", 0:"importance"}) \
          .to_csv(os.path.join(out_dir, f"{label}__{model_name}__FI_fold{fold}.csv"), index=False)
        fi_list.append(fi)

    pd.DataFrame(per_fold).to_csv(
        os.path.join(out_dir, f"{label}__{model_name}__folds.csv"), index=False)

    if fi_list:
        fi_mean = pd.concat(fi_list, axis=1).mean(axis=1).sort_values(ascending=False)
        fi_mean.reset_index().rename(columns={"index":"feature", 0:"importance"}) \
               .to_csv(os.path.join(out_dir, f"{label}__{model_name}__FI_mean.csv"), index=False)

    summary = {
        "comparison": label, "model": model_name,
        "ACC_mean":  float(np.nanmean(accs))        if accs       else np.nan,
        "ACC_std":   float(np.nanstd(accs, ddof=1)) if len(accs)>1 else np.nan,
        "F1_mean":   float(np.nanmean(f1s))         if f1s        else np.nan,
        "F1_std":    float(np.nanstd(f1s,  ddof=1)) if len(f1s)>1  else np.nan,
        "AUC_mean":  float(np.nanmean(aucs))        if aucs       else np.nan,
        "AUC_std":   float(np.nanstd(aucs, ddof=1)) if len(aucs)>1 else np.nan,
        "n_folds_valid": len(accs), "n_rows": len(X), "n_features": X.shape[1],
    }
    with open(os.path.join(out_dir, f"{label}__{model_name}__summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# ── COMPARISON RUNNERS ────────────────────────────────────────────────────────

def run_setting(df, feat_cols, label, out_dir):
    X, y, g = df[feat_cols].copy(), df["group"].values, df["subject_id"].values
    return [eval_gkf(X, y, g, mn, pipe, out_dir, label, feat_cols)
            for mn, pipe in MODEL_SPECS.items()]


def run_within_emotion_pairs(df, feat_cols, group_val, emotions, out_base):
    gname = "AD" if group_val == 1 else "HC"
    out_g = os.path.join(out_base, gname); os.makedirs(out_g, exist_ok=True)
    emos  = sorted(e for e in emotions if e in df[df["group"]==group_val]["emotion"].unique())
    results = []
    for e1, e2 in itertools.combinations(emos, 2):
        dfg = df[(df["group"]==group_val) & df["emotion"].isin([e1,e2])].copy()
        ea, eb = sorted([e1, e2])
        dfg["y_bin"] = (dfg["emotion"] == ea).astype(int)
        if dfg["y_bin"].nunique() < 2: continue
        out_p = os.path.join(out_g, f"{ea}_vs_{eb}"); os.makedirs(out_p, exist_ok=True)
        for mn, pipe in MODEL_SPECS.items():
            s = eval_gkf(dfg[feat_cols].copy(), dfg["y_bin"].values, dfg["subject_id"].values,
                         mn, pipe, out_p, f"{gname}__{ea}_vs_{eb}", feat_cols)
            s.update({"group": gname, "emotion_pos": ea, "emotion_neg": eb})
            results.append(s)
    if results:
        pd.DataFrame(results).sort_values(["comparison","F1_mean"], ascending=[True,False]) \
          .to_csv(os.path.join(out_g, f"SUMMARY_WITHIN_{gname}.csv"), index=False)
    return results


def run_intergroup_bin(df, feat_cols, bin_col, bin_val, out_base, tag):
    dff = df[df[bin_col]==bin_val].copy()
    if len(dff)==0 or dff["group"].nunique()<2: return []
    out = os.path.join(out_base, f"{bin_col}_{bin_val}"); os.makedirs(out, exist_ok=True)
    return run_setting(dff, feat_cols, f"{tag}__AD_vs_HC__{bin_col}_{bin_val}", out)


def run_within_bin(df, feat_cols, group_val, bin_col, out_base, tag):
    gname = "AD" if group_val==1 else "HC"
    dfg = df[(df["group"]==group_val) & df[bin_col].isin(["high","low"])].copy()
    if len(dfg)==0 or dfg[bin_col].nunique()<2: return []
    dfg["y_bin"] = (dfg[bin_col]=="high").astype(int)
    if dfg["y_bin"].nunique()<2: return []
    out = os.path.join(out_base, gname, f"{bin_col}_high_vs_low"); os.makedirs(out, exist_ok=True)
    results = []
    for mn, pipe in MODEL_SPECS.items():
        s = eval_gkf(dfg[feat_cols].copy(), dfg["y_bin"].values, dfg["subject_id"].values,
                     mn, pipe, out, f"{tag}__{gname}__{bin_col}_high_vs_low", feat_cols)
        results.append(s)
    pd.DataFrame(results).sort_values("F1_mean", ascending=False) \
      .to_csv(os.path.join(out, f"SUMMARY_{tag}__{gname}__{bin_col}_high_vs_low.csv"), index=False)
    return results


def run_full_suite(df, feat_cols, emotions, out_root, tag):
    res = []
    res += run_setting(df.copy(), feat_cols, f"{tag}__AD_vs_HC_GLOBAL",
                       os.path.join(out_root, "GLOBAL"))
    out_emo = os.path.join(out_root, "PER_EMOTION"); os.makedirs(out_emo, exist_ok=True)
    for emo in emotions:
        dfe = df[df["emotion"]==emo].copy()
        if dfe["group"].nunique()<2: continue
        out_e = os.path.join(out_emo, emo); os.makedirs(out_e, exist_ok=True)
        res += run_setting(dfe, feat_cols, f"{tag}__AD_vs_HC_{emo}", out_e)
    out_w = os.path.join(out_root, "WITHIN_GROUP"); os.makedirs(out_w, exist_ok=True)
    res += run_within_emotion_pairs(df, feat_cols, 0, emotions, out_w)
    res += run_within_emotion_pairs(df, feat_cols, 1, emotions, out_w)
    out_b = os.path.join(out_root, "VALENCE_AROUSAL"); os.makedirs(out_b, exist_ok=True)
    for bc, bv in [("valence_bin","high"),("valence_bin","low"),
                   ("arousal_bin","high"),("arousal_bin","low")]:
        res += run_intergroup_bin(df, feat_cols, bc, bv, out_b, tag)
    for gv in (0, 1):
        for bc in ("valence_bin", "arousal_bin"):
            res += run_within_bin(df, feat_cols, gv, bc, out_b, tag)
    if res:
        pd.DataFrame(res).sort_values(["comparison","F1_mean"], ascending=[True,False]) \
          .to_csv(os.path.join(out_root, f"SUMMARY_{tag}.csv"), index=False)
    return res


# ── FEATURE SELECTORS ─────────────────────────────────────────────────────────

def _dedup(lst):
    seen = set(); return [x for x in lst if not (x in seen or seen.add(x))]

def band_power_features(cols, band):
    band = band.lower()
    pats = [rf"(^|_)abs_{band}_", rf"(^|_)rel_{band}_",
            rf"(^|_)d_abs_{band}_", rf"(^|_)d_rel_{band}_"]
    if band == "alpha":
        pats += [r"(^|_)FAI_alpha($|_)", r"(^|_)d_FAI_alpha($|_)",
                 r"(^|_)ratio_theta_alpha($|_)", r"(^|_)d_ratio_theta_alpha($|_)",
                 r"(^|_)ratio_beta_alpha($|_)",  r"(^|_)d_ratio_beta_alpha($|_)"]
    elif band == "theta":
        pats += [r"(^|_)ratio_theta_alpha($|_)", r"(^|_)d_ratio_theta_alpha($|_)"]
    elif band == "beta":
        pats += [r"(^|_)ratio_beta_alpha($|_)", r"(^|_)d_ratio_beta_alpha($|_)"]
    rgx = re.compile("|".join(pats))
    return _dedup([c for c in cols if rgx.search(c)])


def band_all_features(cols, band):
    band = band.lower()
    rgx  = re.compile(rf"(_{band}_)|(^{band}_)|(^d_{band}_)")
    out  = [c for c in cols if rgx.search(c)]
    extras = {"alpha": ["FAI_alpha","d_FAI_alpha","ratio_theta_alpha",
                        "d_ratio_theta_alpha","ratio_beta_alpha","d_ratio_beta_alpha"],
              "theta": ["ratio_theta_alpha","d_ratio_theta_alpha"],
              "beta":  ["ratio_beta_alpha","d_ratio_beta_alpha"]}.get(band, [])
    return _dedup(out + [e for e in extras if e in cols])


def frontal_features(cols):
    def is_frontal(c):
        if "_frontal" in c: return True
        if c in {"FAI_alpha","ratio_theta_alpha","ratio_beta_alpha",
                 "d_FAI_alpha","d_ratio_theta_alpha","d_ratio_beta_alpha"}: return True
        return c.split("_")[-1] in FRONTAL_CHANNELS
    return _dedup([c for c in cols if is_frontal(c)])


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df, all_feats = load_data(CSV_PATH)
    all_feats = filter_features(all_feats, FEATURE_REGEX_KEEP)
    emotions  = [e for e in EMOTIONS if e in df["emotion"].unique()]
    print(f"Emotions: {emotions}")
    print(f"Features: {len(all_feats)}")

    master = []

    # Block 1 — all features
    print("\n[1/4] ORIGINAL")
    r1 = run_full_suite(df, all_feats, emotions, os.path.join(OUT_DIR, "ORIGINAL"), "ORIGINAL")
    master += r1

    # Block 2 — band power features
    print("\n[2/4] BY_BAND_POWER")
    out2 = os.path.join(OUT_DIR, "BY_BAND_POWER"); os.makedirs(out2, exist_ok=True)
    r2 = []
    for band in BANDS:
        feats = band_power_features(all_feats, band)
        if not feats: continue
        r2 += run_full_suite(df, feats, emotions, os.path.join(out2, band), f"BAND_POWER_{band}")
    if r2:
        pd.DataFrame(r2).sort_values(["comparison","F1_mean"], ascending=[True,False]) \
          .to_csv(os.path.join(out2, "SUMMARY_ALL_BANDS_POWER.csv"), index=False)
    master += r2

    # Block 3 — all features per band
    print("\n[3/4] BY_BAND_ALL")
    out3 = os.path.join(OUT_DIR, "BY_BAND_ALL"); os.makedirs(out3, exist_ok=True)
    r3 = []
    for band in BANDS:
        feats = band_all_features(all_feats, band)
        if not feats: continue
        r3 += run_full_suite(df, feats, emotions, os.path.join(out3, band), f"BAND_ALL_{band}")
    if r3:
        pd.DataFrame(r3).sort_values(["comparison","F1_mean"], ascending=[True,False]) \
          .to_csv(os.path.join(out3, "SUMMARY_ALL_BANDS_ALL.csv"), index=False)
    master += r3

    # Block 4 — frontal channels only
    print("\n[4/4] FRONTAL_ONLY")
    feats_front = frontal_features(all_feats)
    r4 = []
    if feats_front:
        out4 = os.path.join(OUT_DIR, "FRONTAL_ONLY")
        r4 = run_full_suite(df, feats_front, emotions, out4, "FRONTAL_ONLY")
    master += r4

    # Master summary
    if master:
        mp = os.path.join(OUT_DIR, "SUMMARY_MASTER.csv")
        pd.DataFrame(master).sort_values(["comparison","F1_mean"], ascending=[True,False]) \
          .to_csv(mp, index=False)
        print(f"\nMaster summary: {mp}")

    print("Done.")


if __name__ == "__main__":
    main()
