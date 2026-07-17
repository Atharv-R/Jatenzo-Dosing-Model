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


# ------------------------------------------------------------------- column normalization
# Map the many possible source names (the colleague's sheet has evolved) to canonical ABT
# names. Handles CURRENT_DOSE_JAT/NEW_DOSE_JAT and the older CURRENT_DOSE/NEW_DOSE.
ALIASES = {
    "subject_id": ["subject_id", "SUBJECT", "SUBJID", "subject"],
    "age": ["age", "AGE"],
    "bmi": ["bmi", "BMI"],
    "current_dose": ["current_dose", "CURRENT_DOSE_JAT", "CURRENT_DOSE"],
    "new_dose": ["new_dose", "NEW_DOSE_JAT", "NEW_DOSE"],
    "current_T": ["current_T", "CURRENT_T"],
    "outcome_T": ["outcome_T", "OUTCOME_T"],
    "delta_t": ["delta_t", "DELTA_T"],
    "delta_t_win": ["delta_t_win", "DELTA_T_WIN"],
    "interval_days": ["interval_days", "INTERVAL_DAYS"],
    "is_switch": ["is_switch", "IS_SWITCH"],
    "pair": ["pair", "PAIR"],
}


def _num(x):
    """Parse a number, treating accounting-style '(1429)' as -1429 and ',' as thousands."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        s = x.strip().replace(",", "")
        if s in ("", "-", "NA", "nan", "None"):
            return np.nan
        neg = s.startswith("(") and s.endswith(")")
        if neg:
            s = s[1:-1]
        try:
            v = float(s)
        except ValueError:
            return np.nan
        return -v if neg else v
    return float(x)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    ren = {}
    for canon, aliases in ALIASES.items():
        for a in aliases:
            if a in df.columns and canon not in df.columns:
                ren[a] = canon
                break
    df = df.rename(columns=ren)
    for c in ["delta_t", "delta_t_win", "current_T", "outcome_T",
              "current_dose", "new_dose", "bmi"]:
        if c in df.columns and df[c].dtype == object:
            df[c] = df[c].map(_num)
    return df


def build_abt_from_ml_sheet(path: str, sheet: str = "DatasetML") -> pd.DataFrame:
    """Load the cleaned 'DatasetML' sheet and normalize to ABT columns."""
    return _normalize_columns(pd.read_excel(path, sheet_name=sheet))


# --------------------------------------------------------------------------- dispatch
def build_abt(cfg: dict) -> pd.DataFrame:
    source = cfg.get("source", "synthetic")
    if source == "file":
        df = pd.read_csv(cfg["abt_path"])
    elif source == "excel":
        df = build_abt_from_ml_sheet(cfg["abt_path"], cfg.get("sheet", "DatasetML"))
    elif source == "trial":
        df = build_abt_from_trial(cfg["raw_sources"])
    else:
        df = synthetic_abt(seed=cfg.get("seed", 42))

    df = _normalize_columns(df)                      # robust to source column naming
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
