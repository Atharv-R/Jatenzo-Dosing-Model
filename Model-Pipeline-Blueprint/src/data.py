"""DATA layer -> produces the Analytics Base Table (ABT).

The ABT is one flat, tidy table: features as columns, one row per (subject x dose-interval).
It is the single hand-off to every model, so switching models never touches this file.

For the outcome-T model (the primary approach) each row is one observation of the
"function" the data encodes:
    (age, bmi, current_T, current_dose, new_dose)  -->  outcome_T
i.e. a patient in some state is given a dose and we observe the resulting T.

Three sources (set via config `data.source`):
  * "file"     -> read a cleaned CSV you drop at `data.abt_path` (see data/README.md).
  * "trial"    -> build_abt_from_trial(): real loader stub, TODO map the workbooks.
  * "synthetic"-> runnable stand-in so the pipeline executes today.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from rubric import LADDER, next_dose_from_rubric

# Columns every ABT must contain for the outcome-T model.
REQUIRED = ["subject_id", "age", "bmi", "current_T", "current_dose", "new_dose", "outcome_T"]


# --------------------------------------------------------------------------- real loader
def build_abt_from_trial(raw_sources: list[str]) -> pd.DataFrame:
    """TODO: assemble the ABT from the trial workbooks / cleaned sheet.

    Mapping from the transformed sheet (SUBJECT / DOSE / T-before / T-after):
      subject_id   <- SUBJECT
      age, bmi     <- AGE, BMI
      current_T    <- READING (BEFORE)          # T entering the interval
      new_dose     <- DOSE                       # dose applied during the interval
      current_dose <- DOSE of the *previous* interval (0 if treatment-naive)
      outcome_T    <- READING (AFTER)            # T achieved at new_dose
    Confirm the T-reading timepoint (label decision = 6h post-dose) and map DOSE units
    to the ladder (158/198/237/316/396) -- see DATA-SPEC.md.
    """
    raise NotImplementedError(
        "Real ABT loader not yet wired. Use data.source='file' with a cleaned CSV, "
        "or data.source='synthetic' to exercise the pipeline."
    )


# --------------------------------------------------------------------------- synthetic
def synthetic_abt(n_patients: int = 300, visits: int = 4, seed: int = 42) -> pd.DataFrame:
    """Plausible synthetic ABT encoding (state, new_dose) -> outcome_T.

    Higher dose -> higher T, damped by BMI and age, partly persistent from current_T,
    with patient-specific sensitivity and noise -- enough to exercise every metric.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_patients):
        sens = rng.normal(1.0, 0.25)
        age = int(rng.integers(18, 75))
        bmi = float(np.clip(rng.normal(28, 5), 18, 40))
        current_dose = 0                                  # treatment-naive at entry
        current_T = float(np.clip(rng.normal(300, 60), 80, 550))
        for v in range(visits):
            new_dose = int(rng.choice(LADDER))
            base = 1.9 * new_dose * sens * (1 - 0.010 * (bmi - 25)) * (1 - 0.004 * (age - 40))
            outcome_T = float(max(50, 0.15 * current_T + base + rng.normal(0, 90)))
            rows.append(dict(
                subject_id=f"P{p:04d}", visit=v,
                age=age, bmi=round(bmi, 1),
                current_T=round(current_T, 1), current_dose=current_dose,
                new_dose=new_dose, outcome_T=round(outcome_T, 1),
                desired_T=697.5,                          # band midpoint default
                next_dose_rubric=next_dose_from_rubric(outcome_T, new_dose),
            ))
            current_dose, current_T = new_dose, outcome_T  # advance state
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- dispatch
def build_abt(cfg: dict) -> pd.DataFrame:
    source = cfg.get("source", "synthetic")
    if source == "file":
        df = pd.read_csv(cfg["abt_path"])
    elif source == "trial":
        df = build_abt_from_trial(cfg["raw_sources"])
    else:
        df = synthetic_abt(seed=cfg.get("seed", 42))

    if "subject_id" not in df and "patient_id" in df:
        df["subject_id"] = df["patient_id"]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"ABT is missing required columns: {missing}. See data/README.md.")
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
