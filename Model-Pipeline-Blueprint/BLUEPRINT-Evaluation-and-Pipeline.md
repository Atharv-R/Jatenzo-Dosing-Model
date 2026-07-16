# Jatenzo Dosing Model — Evaluation & Pipeline Blueprint

Scope: how models are **evaluated**, how the **data should be shaped**, and the
**modular Python pipeline** that lets us swap models/configs/data/metrics quickly.
Written to satisfy leadership's Friday asks and to keep both modeling directions
(serum-T method vs direct-dose method) on the table.

---

## Part 1 — What we treat as the "true value" (regression target)

This is the most important decision, so it comes first. There are three candidate
targets, and they answer **different questions**. My recommendation is to support all
three in the pipeline but designate a **primary** one.

### The three candidate targets

| Target | What it is | What a model trained on it learns | Noise / issues |
|---|---|---|---|
| **T1. Serum T (continuous)** | Measured on-treatment testosterone (6h post-dose and/or Cavg24) | The drug's dose–response for this patient | Cleanest signal; directly measured |
| **T2. Rubric dose** | Next dose from the FDA lookup table | To reproduce a *known deterministic rule* | Zero noise, but can be trivial (see below) |
| **T3. Clinician dose** | Dose actually given at the next visit in the trial | Real-world prescribing behavior | Noisy: deviations, missed visits, judgment |

### The trap to avoid (circularity / leakage)

The FDA rubric's input is the **6h post-dose serum T + current dose**. If our feature
"Current T" **is** that 6h post-dose value, then the rubric dose (T2) is a *deterministic
function of the features we already have* — a model would score ~100% by memorizing the
lookup table, and the evaluation would be meaningless. ML only adds value when there is a
**gap between the inputs we have and the input the rubric needs.**

So the meaning of "Current T" decides everything:

- If **Current T = the 6h on-treatment level** → the rubric is directly computable; ML is
  pointless for T2. The real ML task is **T1: predict that serum T** at doses not yet tried.
- If **Current T = a baseline / convenient-time level** → predicting the rubric outcome (T2)
  or the clinician dose (T3) from non-rubric inputs is a genuine, non-trivial ML problem.

**We must pin down what "Current T" is before finalizing the target.** (Open question below.)

### Recommendation

- **Primary target: T1 — measured serum T (continuous regression).** It is the cleanest,
  most informative, directly measured, and it powers the safe "predict-then-apply-rubric"
  design. Every model can be scored on how well it predicts the level that the clinical
  decision actually depends on.
- **Primary *decision* benchmark: T2 — rubric dose**, used as the **ground-truth action**
  when scoring dose recommendations, because the rubric is the clinically sanctioned policy
  (noise-free "correct answer").
- **Reality-check target: T3 — clinician dose**, used as a *secondary* metric to measure
  real-world agreement and to surface where trial practice diverged from the rubric.

In short: **regress over serum T; judge dose recommendations against the rubric; sanity-check
against what clinicians actually did.** This lets a single dataset serve both modeling
directions and gives leadership three complementary views instead of one brittle number.

---

## Part 2 — Outlier policy (the "permission to remove 2%" question)

Short answer: **yes to handling outliers, but winsorize rather than delete, be specific
about which variable, and never let outlier removal hide a safety case.**

Reasoning:

- **"Top/bottom 2%" of *what*?** Of a feature (e.g. BMI), of serum T, or of the dose?
  Removing the extremes of **dose or serum T** deletes exactly the ceiling/floor and
  extreme-responder patients — the ones a dosing model most needs to handle safely. That's
  the opposite of what we want.
- **At n ≈ 300 patients, 2% top + 2% bottom ≈ 4% ≈ ~12 patients.** Non-trivial loss.
- **Winsorize (cap at the 2nd/98th percentile) instead of drop** for continuous *features*.
  It tames leverage from extreme values without throwing away observations or their labels.
- **Only hard-delete documented data-quality errors** (impossible values, assay failures,
  protocol violations) — not statistically-extreme-but-valid patients.
- **Keep excluded/extreme patients in the *test* evaluation** for safety metrics, even if
  they're down-weighted or excluded in *training*. The model must be *measured* on them.
- Make it a **config switch** with a sensitivity analysis: run with `none`, `winsorize(2,98)`,
  and `drop(2,98)` and report how conclusions change.

Recommended default: `winsorize` continuous features at (2, 98); `drop` only flagged data
errors; report a with/without sensitivity check.

---

## Part 3 — The five input variables (and a flag)

Committed baseline feature set (matches the existing calculator, per leadership):

1. **Current T** — *needs definition:* baseline vs 6h on-treatment level (see Part 1).
2. **Current dose** — current mg BID (0/158/198/237/316/396).
3. **Desired dose** — *flag:* listed as "desired dose," but as a **model input predicting a
   dose** this is likely meant to be **desired/target testosterone**, matching the
   calculator's "Target T." Predicting a dose *from* a desired dose is circular. Treating it
   as **target T** unless told otherwise.
4. **BMI**
5. **Age**

**Design note:** these five exclude the covariates that most drive oral-TU exposure (SHBG,
dietary fat, baseline T). We honor the 5-variable baseline as **one feature config**, and
keep the data layer able to add columns so the extended/serum-T direction stays a
one-line switch — not a rebuild.

---

## Part 4 — Data shape (leadership ask #1)

**Analytics Base Table (ABT): one flat, tidy table — features as columns, one row per
observation — so any model can consume it and we can switch models freely.**

- **Row granularity:** one row per **(patient × titration decision point / visit)**. This
  yields more rows than patients (each visit is a decision), which helps at small n — but
  see the CV note.
- **Column groups:**

| Group | Columns (examples) |
|---|---|
| **Keys / meta** | `usubjid`, `visit`, `study_id` |
| **Group key (for CV)** | `patient_id` (= usubjid) — used to prevent leakage |
| **Baseline features (the 5)** | `current_T`, `current_dose`, `target_T`, `bmi`, `age` |
| **Extended features (optional, off by default)** | `shbg`, `dietary_fat`, `baseline_T`, `race`, `hct` |
| **Targets** | `serum_T_6h` (T1), `next_dose_rubric` (T2), `next_dose_clinician` (T3) |
| **Safety / flags** | `at_ladder_ceiling`, `at_ladder_floor`, `ood_flag`, `outlier_flag` |

- **One physical file** (`abt.parquet` or `.csv`) is the single hand-off between the
  data layer and every model. Change the data → regenerate this one table; nothing
  downstream changes.
- **CV caveat:** because one patient contributes multiple rows, splits **must group by
  `patient_id`** (GroupKFold) — never random row splits — or the model leaks a patient's
  own history into its test score.

---

## Part 5 — Modular pipeline (leadership ask #2)

Goal: change the model, its hyperparameters, the data-cleaning, or the metrics **by editing
one config file**, not the code. Plain Python + scikit-learn conventions; no heavy framework
required (Hydra optional later).

### Directory layout

```
Model-Pipeline-Blueprint/
  config.example.yaml         # one experiment = one config file
  src/
    data.py                   # load raw sheets -> clean -> feature build -> ABT
    models.py                 # model registry: name -> estimator (common interface)
    evaluate.py               # metric registry + patient-level CV runner
    guardrails.py             # ladder clamp, one-step limit, OOD defer (Part 6)
    run.py                    # orchestrator: read config -> build -> train -> eval -> log
  configs/                    # saved experiment configs
  results/                    # metrics.json, predictions.csv, plots per run
```

### The three swap points (each isolated to one module)

1. **Swap the data** → edit `data.py` cleaning/feature functions or the `data:` config
   block. Outputs the ABT. Nothing else changes.
2. **Swap the model / hyperparameters** → change the `model:` block in the config. `models.py`
   is a **registry** mapping a name to an estimator with a **common interface**
   (`fit(X, y)`, `predict(X)`, optional `predict_proba`). To add a model, register one class.
3. **Swap the evaluation** → change the `evaluation:` block. `evaluate.py` is a **metric
   registry**; add a metric = add one function.

### Common model interface (the key to swappability)

Every model — linear regression, ordinal logistic, random forest, gradient boosting
(LightGBM/XGBoost), the rubric baseline, the serum-T-then-rule wrapper — exposes the same
methods, so `run.py` never needs to know which model it's running. New model = new registry
entry, zero orchestrator changes.

### Candidate models to register (for "begin testing" on Friday)

- **Baselines:** the rubric itself; the existing linear calculator; "always 237 / no change."
- **Direct-dose (Method B):** linear/logistic regression, **ordinal logistic**, random
  forest, **monotone-constrained gradient boosting**.
- **Serum-T (Method A):** GBM/linear regression on `serum_T_6h`, wrapped by the rubric.

### Metrics to register

- **Continuous (serum T / Method A):** RMSE, MAE, R², calibration, **% predictions in-band**.
- **Dose (classification/ordinal):** exact accuracy, **within-one-step accuracy**, MAE in
  ladder steps, **quadratic-weighted kappa**, confusion matrix.
- **Safety (both):** rate of recs that would push serum T **>970 (overshoot)** or
  **<425 (undershoot)**; **off-ladder / >1-step** rate; **agreement with rubric**;
  **agreement with clinician**.
- **Protocol:** patient-level **GroupKFold**, report mean ± std across folds, always against
  the three baselines above.

### On "I can also design this, or we both do it and compare"
This blueprint is the design + concrete interfaces. Next step is a runnable skeleton
(`run.py` + registries + `config.example.yaml`). Happy to scaffold it, or to build it in
parallel with your version and diff the two — your call.

---

## Part 6 — The "four unknown scenarios" + guardrails (my interpretation)

Leadership's line — *"our model needs to succeed across four unknown scenarios (we need
guardrails to address what those are)"* — most plausibly means: **the model must behave
safely in input regimes that the ~300-patient trial under-represents**, and each such regime
needs a **deterministic guardrail wrapping the ML output.** This is exactly the Layer-2
safety layer from the first blueprint. Below is a concrete proposed set of four —
**to confirm with leadership**, not assumed final.

| # | Scenario (under-represented / risky regime) | Guardrail |
|---|---|---|
| **1** | **Initiation / cold start** — new patient, no current dose | Default to label start **237 mg**; or Method-A predict from baseline covariates, never a blind jump |
| **2** | **Ladder boundaries** — already at 396 but still low, or at 158 but still high | Enforce ladder limits (158–396); apply the **discontinue** rule at floor; never recommend beyond the ladder |
| **3** | **Out-of-distribution inputs** — age/BMI/T outside training range | OOD detector → **clamp inputs + widen uncertainty + defer to rubric**, and flag for clinician review |
| **4** | **Poor absorbers / non-responders / extreme SHBG** — predicted vs actual serum T diverge | If the predictive interval **crosses a band edge**, recommend **re-measure at 7 days** (the label's recheck) instead of a confident dose change |

Common principle: the ML model **proposes**, the guardrail layer **disposes** — every
recommendation is clamped to the label ladder, limited to one step, and downgraded to
"defer / re-measure" when confidence is low or inputs are off-distribution. I'll bring this
table to Friday as a strawman so leadership can name the four they actually mean.

---

## Part 7 — Path to Friday

1. **Confirm two open questions** (below) — they finalize the target and features.
2. **Build the ABT** from the trial sheets (shared, direction-agnostic).
3. **Scaffold the pipeline** (`run.py` + registries + `config.example.yaml`).
4. **Register 2–3 models + baselines**, run patient-level CV, produce a first metrics table.
5. **Bring:** this blueprint, the ABT schema, a runnable pipeline, an initial results table,
   and the guardrails strawman.

### Open questions that change the specifics
- **Q1. What is "Current T"** — the 6h on-treatment level (→ serum-T direction, rubric is
  computable) or a baseline level (→ direct-dose ML is the real task)? This sets the target.
- **Q2. "Desired dose"** — confirm it means **target testosterone** (as assumed), not a dose.
