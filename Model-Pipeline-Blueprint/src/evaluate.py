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


def run_outcome_eval(model_factory, df, cfg):
    """Logically-strong evaluation for the outcome-T model. Two honest tests, both on
    patients held out entirely (GroupKFold by subject -- no leakage).

    TEST 1 -- Outcome-T prediction accuracy (the model's core skill).
      Predict outcome_T for held-out (state, new_dose) rows and compare to the OBSERVED
      outcome_T. Fully honest: we score against a value that was actually measured.
      Metrics: RMSE, MAE, R^2, and % within +/-100 ng/dL.

    TEST 2 -- Inverse recommendation accuracy (does the recommender back out reality?).
      For each held-out row we KNOW the patient took dose `new_dose` and reached
      `outcome_T`. Set desired_T := that achieved outcome_T and ask the recommender which
      dose to give. If it returns the dose actually taken (exact) or within one ladder
      step, it is correct -- because that dose is, by observation, the one that produced
      that T for this patient. No counterfactual is assumed, so the test is honest.
      Baseline for context: 'keep the current dose'.
    """
    group_col = cfg.get("group_key", "subject_id")
    groups = df[group_col] if group_col in df.columns else df["subject_id"]
    gkf = GroupKFold(n_splits=cfg.get("n_splits", 5))
    y = df["outcome_T"]
    # Dose set is data-driven (trial doses differ from the commercial ladder).
    doses = sorted(df["new_dose"].unique().tolist())
    idx = {d: i for i, d in enumerate(doses)}
    rows = []
    for k, (tr, te) in enumerate(gkf.split(df, y, groups)):
        model = model_factory().fit(df.iloc[tr], y.iloc[tr])
        te_df = df.iloc[te]
        yt = te_df["outcome_T"].values
        pred = model.predict(te_df)

        ss_res = float(np.sum((yt - pred) ** 2))
        ss_tot = float(np.sum((yt - yt.mean()) ** 2))

        rec = model.recommend(te_df, yt)                 # desired_T = achieved T
        taken = te_df["new_dose"].values
        keep = te_df["current_dose"].values              # naive 'keep current dose' baseline

        def within1(mask):
            a, b = rec[mask], taken[mask]
            if len(a) == 0:
                return float("nan")
            return float(np.mean([abs(idx[x] - idx[y]) <= 1 for x, y in zip(a, b)]))

        # Most rows are stable-dose (new==current), where 'keep dose' is trivially right.
        # The dose-CHANGE rows are the informative ones -> report them separately.
        is_switch = (te_df["is_switch"].values == 1) if "is_switch" in te_df else (taken != keep)

        rows.append({
            "fold": k,
            "outcomeT_rmse": float(np.sqrt(np.mean((yt - pred) ** 2))),
            "outcomeT_mae": float(np.mean(np.abs(yt - pred))),
            "outcomeT_r2": float(1 - ss_res / ss_tot) if ss_tot else 0.0,
            "outcomeT_within_100ngdl": float(np.mean(np.abs(yt - pred) <= 100)),
            "inverse_within_1_overall": within1(np.ones(len(rec), bool)),
            "inverse_within_1_SWITCH": within1(is_switch),      # the decisions that matter
            "inverse_within_1_stable": within1(~is_switch),
            "baseline_keepdose_exact": float(np.mean(keep == taken)),
        })
    folds = pd.DataFrame(rows)
    summary = folds.drop(columns=["fold"]).agg(["mean", "std"]).T
    return folds, summary


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
