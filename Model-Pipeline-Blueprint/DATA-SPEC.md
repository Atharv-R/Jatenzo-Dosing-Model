# Shared ABT Data Spec

Goal: both of us clean the trial data into the **same Analytics Base Table (ABT)** so our
models are directly comparable. One row = one **(subject × dose-interval)** observation.
This maps almost directly onto the transformed sheet (SUBJECT / DOSE / T-before / T-after / CHANGE).

## Columns to produce

| ABT column | Source column (screenshot) | Notes |
|---|---|---|
| `subject_id` | SUBJECT | keep — used for **patient-level CV grouping** (no random splits) |
| `age` | AGE | |
| `bmi` | BMI | |
| `race`, `ethnicity` | RACE, ETHNIC | optional categoricals (CatBoost-friendly) |
| `dose` | DOSE | ⚠️ **confirm units** and map to label ladder (158/198/237/316/396) |
| `t_before` | READING (BEFORE) | baseline T for the interval — **keep as a feature** |
| `t_after` | READING (AFTER) | on-treatment T (candidate target) |
| `delta_t` | CHANGE | = t_after − t_before |
| `day_before`,`day_after`,`day_diff` | DAY (BEFORE/AFTER/DIFFERENCE) | interval timing |

## Targets (pick per experiment; pipeline supports all)

- **T-after (absolute)** — recommended primary. Model `t_after ~ dose + t_before + covariates`.
- **delta_t (ΔT)** — the colleague's framing; equivalent **only if `t_before` is a feature**.
- **next_dose_rubric** — decision benchmark (compute via FDA Table 1 when a 6h reading exists).

## Open questions to resolve before modeling (these change the design)

1. **Timepoint of the T reading** — is READING (AFTER) drawn **6h after the morning dose**
   (the label's decision variable), or trough / Cavg / mixed? Everything keys off this.
2. **DOSE encoding** — observed 0/150/200 do **not** match the label ladder
   (158/198/237/316/396). What scale is this, and how do we map to the recommendation ladder?
3. **Interval consistency** — CHANGE spans different windows (30d vs 90d). Filter to
   steady-state intervals or normalize per-day; don't mix silently.
4. **Regression to mean** — extreme `t_before` reverts (e.g. 1030→446 at same dose).
   Always include `t_before` as a predictor; consider it explicitly in interpretation.
5. **Missing exposure drivers** — **SHBG** and **dietary fat** drive oral-TU variability and
   likely explain the large same-dose swings. Available in the source workbooks? Add if so.

## CV rule (non-negotiable)

Subject IDs repeat → **group by `subject_id`** in cross-validation, or a subject's own
history leaks into its test score. The pipeline enforces this (GroupKFold).
