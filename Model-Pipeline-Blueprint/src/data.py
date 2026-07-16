"""DATA layer -> produces the Analytics Base Table (ABT).

The ABT is one flat, tidy table: features as columns, one row per (patient x visit).
It is the single hand-off to every model, so switching models never touches this file.

Two paths:
  * build_abt_from_trial(): real loader stub. TODO: map the CDISC/transformed sheets
    (pkdata, adtespar, Consolidated T/SHBG, adex, addosad, ...) into the ABT columns.
  * synthetic_abt(): a runnable stand-in with a plausible dose-response so the whole
    pipeline executes end-to-end today. Swap in the real loader when the mapping is done.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from rubric import LADDER, next_dose_from_rubric

ABT_FEATURES = ["current_T", "current_dose", "target_T", "bmi", "age"]
ABT_TARGETS = ["serum_T_6h", "next_dose_rubric", "next_dose_clinician"]


# --------------------------------------------------------------------------- real loader
def build_abt_from_trial(raw_sources: list[str]) -> pd.DataFrame:
    """TODO: assemble the ABT from the trial workbooks.

    Planned mapping (to implement once column semantics are confirmed):
      current_T          <- 6h post-dose serum T at the current visit's dose
                            (Consolidated T / 'Hour 4 T' / Serum T sheets, LBTPT ~ 6h)
      current_dose       <- adex / addosad current dose
      target_T           <- fixed band midpoint or configured target
      bmi, age           <- addosad (BMIBL) / demographics
      serum_T_6h         <- same source as current_T at the *evaluated* dose
      next_dose_clinician<- next visit's dose from dosing history
      next_dose_rubric   <- next_dose_from_rubric(current_T, current_dose)
      group_key          <- USUBJID
    """
    raise NotImplementedError(
        "Real ABT loader not yet wired. Run with data.source='synthetic' for now."
    )


# --------------------------------------------------------------------------- synthetic
def synthetic_abt(n_patients: int = 300, visits: int = 3, seed: int = 42) -> pd.DataFrame:
    """Plausible synthetic ABT so the pipeline runs before the real loader is wired.

    Encodes a simple, monotone dose-response: higher dose -> higher serum T, damped by
    BMI and age, with patient-level random sensitivity -- enough to exercise every metric.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_patients):
        sensitivity = rng.normal(1.0, 0.25)          # patient-specific responsiveness
        age = int(rng.integers(18, 75))
        bmi = float(np.clip(rng.normal(28, 5), 18, 40))
        dose = int(rng.choice(LADDER))
        for v in range(visits):
            # 6h serum T as a function of dose + covariates + noise.
            base = 1.9 * dose * sensitivity
            serum = base * (1 - 0.010 * (bmi - 25)) * (1 - 0.004 * (age - 40))
            serum = float(max(50, serum + rng.normal(0, 90)))
            nd_rubric = next_dose_from_rubric(serum, dose)
            # Clinician mostly follows rubric, occasionally deviates (real-world noise).
            nd_clin = nd_rubric if rng.random() > 0.15 else int(rng.choice(LADDER))
            rows.append(dict(
                patient_id=f"P{p:04d}", visit=v,
                current_T=serum, current_dose=dose,
                target_T=697.5,                       # band midpoint
                bmi=round(bmi, 1), age=age,
                serum_T_6h=serum,
                next_dose_rubric=nd_rubric,
                next_dose_clinician=nd_clin,
            ))
            # advance to next visit's dose along the (rubric) trajectory
            dose = nd_rubric if nd_rubric in LADDER else dose
    return pd.DataFrame(rows)


def build_abt(cfg: dict) -> pd.DataFrame:
    """Dispatch on config: real trial loader or synthetic fallback, then apply outliers."""
    source = cfg.get("source", "synthetic")
    if source == "trial":
        df = build_abt_from_trial(cfg["raw_sources"])
    else:
        df = synthetic_abt(seed=cfg.get("seed", 42))
    df = _handle_outliers(df, cfg.get("outliers", {}))
    return df


def _handle_outliers(df: pd.DataFrame, oc: dict) -> pd.DataFrame:
    policy = oc.get("policy", "none")
    if policy in ("none", "drop_data_errors_only"):
        return df
    lo, hi = oc.get("lower_pct", 2), oc.get("upper_pct", 98)
    cols = [c for c in oc.get("apply_to", []) if c in df.columns]
    if policy == "winsorize":
        for c in cols:
            lo_v, hi_v = np.percentile(df[c], [lo, hi])
            df[c] = df[c].clip(lo_v, hi_v)
    elif policy == "drop":
        mask = np.ones(len(df), dtype=bool)
        for c in cols:
            lo_v, hi_v = np.percentile(df[c], [lo, hi])
            mask &= df[c].between(lo_v, hi_v)
        df = df[mask].reset_index(drop=True)
    return df
