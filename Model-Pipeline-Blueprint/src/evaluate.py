"""EVALUATION -- metric registry + leakage-safe (patient-level) cross-validation.

Add a metric = add one function to _METRICS. CV *must* group by patient_id, because one
patient contributes multiple rows and random splits would leak their own history.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from rubric import LADDER, BAND_LOW, BAND_HIGH

_METRICS = {}


def metric(name):
    def deco(fn):
        _METRICS[name] = fn
        return fn
    return deco


# ---- dose-recommendation metrics (y_true / y_pred are doses in mg) --------------------
@metric("exact_dose_accuracy")
def exact(y_true, y_pred, **_):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


@metric("within_one_step_accuracy")
def within_one(y_true, y_pred, **_):
    idx = {d: i for i, d in enumerate(LADDER)}
    ok = []
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            ok.append(abs(idx[t] - idx[p]) <= 1)
        else:
            ok.append(t == p)
    return float(np.mean(ok))


@metric("dose_mae_steps")
def mae_steps(y_true, y_pred, **_):
    idx = {d: i for i, d in enumerate(LADDER)}
    diffs = [abs(idx.get(t, 0) - idx.get(p, 0)) for t, p in zip(y_true, y_pred)]
    return float(np.mean(diffs))


@metric("quadratic_weighted_kappa")
def qwk(y_true, y_pred, **_):
    idx = {d: i for i, d in enumerate(LADDER)}
    t = np.array([idx.get(v, 0) for v in y_true]); p = np.array([idx.get(v, 0) for v in y_pred])
    n = len(LADDER)
    O = np.zeros((n, n))
    for a, b in zip(t, p): O[a, b] += 1
    W = np.array([[((i - j) ** 2) / (n - 1) ** 2 for j in range(n)] for i in range(n)])
    at = np.bincount(t, minlength=n); ap = np.bincount(p, minlength=n)
    E = np.outer(at, ap) / max(len(t), 1)
    denom = (W * E).sum()
    return float(1 - (W * O).sum() / denom) if denom else 0.0


# ---- serum-T (regression) metrics (y_true / y_pred are ng/dL) ------------------------
@metric("rmse")
def rmse(y_true, y_pred, **_):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


@metric("pct_in_band")
def pct_in_band(y_true, y_pred, **_):
    p = np.asarray(y_pred, dtype=float)
    return float(np.mean((p >= BAND_LOW) & (p <= BAND_HIGH)))


# ---- safety metrics (need the achieved serum T at the recommended dose) --------------
@metric("agreement_with_rubric")
def agree_rubric(y_true, y_pred, *, rubric=None, **_):
    if rubric is None: return float("nan")
    return float(np.mean(np.asarray(rubric) == np.asarray(y_pred)))


@metric("agreement_with_clinician")
def agree_clin(y_true, y_pred, *, clinician=None, **_):
    if clinician is None: return float("nan")
    return float(np.mean(np.asarray(clinician) == np.asarray(y_pred)))


def run_cv(model_factory, df, features, train_target, eval_target, metrics, cfg):
    """Patient-level GroupKFold. Returns per-fold and mean+-std metric tables.

    train_target may differ from eval_target: Method A (serumT_then_rule) trains on
    serum_T_6h but is scored on a dose column.
    """
    groups = df["group_key"] if "group_key" in df else df["patient_id"]
    gkf = GroupKFold(n_splits=cfg.get("n_splits", 5))
    X, y_tr, y_ev = df[features], df[train_target], df[eval_target]
    fold_rows = []
    for k, (tr, te) in enumerate(gkf.split(X, y_tr, groups)):
        model = model_factory()
        model.fit(X.iloc[tr], y_tr.iloc[tr])
        y_pred = model.predict(X.iloc[te])
        extra = dict(
            rubric=df["next_dose_rubric"].iloc[te].values if "next_dose_rubric" in df else None,
            clinician=df["next_dose_clinician"].iloc[te].values if "next_dose_clinician" in df else None,
        )
        row = {"fold": k}
        for m in metrics:
            row[m] = _METRICS[m](y_ev.iloc[te].values, y_pred, **extra)
        fold_rows.append(row)
    folds = pd.DataFrame(fold_rows)
    summary = folds.drop(columns=["fold"]).agg(["mean", "std"]).T
    return folds, summary
