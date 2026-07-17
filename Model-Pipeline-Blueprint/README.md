# Jatenzo Dosing Model — Pipeline

Modular, config-driven pipeline for testing dosing models. Change the model, its
hyperparameters, the data cleaning, or the metrics by editing **one YAML file** — the code
never changes. See `BLUEPRINT-Evaluation-and-Pipeline.md` for the full design rationale.

## Setup

Requires **Python 3.10+, 64-bit** (the notebooks and `xgboost`/`lightgbm`/`catboost`
wheels do not ship 32-bit builds). Verify your interpreter before installing:

```bash
python3 -c "import platform, struct; print(platform.python_version(), struct.calcsize('P')*8, platform.machine())"
# expect e.g. "3.11.6 64 arm64" or "3.11.6 64 x86_64" — if it prints "32", you have
# a 32-bit Python install and must switch to a 64-bit one (see below)
```

Use a virtual environment so the project's dependencies don't collide with anything else
on your machine:

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

If VS Code / Jupyter is picking the wrong interpreter (e.g. a stray 32-bit or system
Python instead of `.venv`), select it explicitly: Command Palette →
**Python: Select Interpreter** → choose `.venv`, then re-select the kernel at the
top-right of the notebook.

## Primary model: outcome-T (predict final T, then pick the closest dose)

This is the lead approach. The model predicts **outcome (final) T** from
`[age, bmi, current_T, current_dose, new_dose]`, then to recommend it sweeps all five
ladder doses and returns the one whose predicted outcome T is closest to the desired T.

### 1. Where the cleaned data goes
Drop the cleaned table at **`data/abt.csv`** (columns in `data/README.md`), then set in
your config: `data.source: file` and `data.abt_path: data/abt.csv`. Until then,
`data.source: synthetic` runs on a built-in stand-in so everything is testable.

### 2. How to run
```bash
pip install -r requirements.txt
python src/run.py --config config.outcome_t.yaml
```

### 3. How it's evaluated (two honest, patient-grouped tests)
Cross-validation holds out whole **subjects** (GroupKFold), so no patient's own history
leaks into its score.

- **TEST 1 — outcome-T prediction** (the model's core skill): predict outcome T for
  held-out rows, compare to the **actually measured** T. Metrics: `rmse`, `mae`, `r2`,
  `within_100ngdl`. Fully honest — scored against a real observation.
- **TEST 2 — inverse recommendation** (does the recommender back out reality?): for a
  held-out row we know the patient took `new_dose` and reached `outcome_T`. Set the
  desired T to that achieved T and ask the recommender for a dose; correct if it returns
  the dose actually taken (`inverse_exact_dose`) or within one step
  (`inverse_within_one_step`). No counterfactual is assumed. `baseline_keepdose_exact`
  is the naive "just keep the current dose" baseline for context.

### Swapping the engine (HistGBM / LightGBM / CatBoost)
Same model, same recommendation logic and eval — only the underlying regressor changes.
Set `model.engine` in the config: `histgbm` (default, no install), `lightgbm`, `catboost`,
or `xgboost`. Ready-to-run configs: `configs/outcome_t_lightgbm.yaml`,
`configs/outcome_t_catboost.yaml`, `configs/outcome_t_xgboost.yaml`.
Optional physiology guardrail via `hyperparameters.monotone_constraints`
(e.g. `{new_dose: 1, current_T: 1}` — higher dose/current T never lowers predicted outcome T).

Synthetic-data comparison (5-fold, patient-grouped) — illustrative until real data lands:

| engine | outcome-T R² | inverse within-1-step |
|---|---|---|
| histgbm | ~0.63 | ~0.84 |
| lightgbm | ~0.63 | ~0.84 |
| catboost | ~0.65 | ~0.86 |
| xgboost | ~0.62 | ~0.85 |

## Notebooks (for a non-technical audience)

`notebooks/` has three friendly, standalone notebooks — one per engine
(`01_XGBoost_model.ipynb`, `02_LightGBM_model.ipynb`, `03_CatBoost_model.ipynb`). Each
walks through load → train → "how good is it" → recommend a dose → guardrails, with plain
explanations and charts, and comes with outputs already embedded. They read `data/abt.csv`
if present, otherwise a built-in sample. Open in Jupyter and run top to bottom.

### 4. Where to see results
Printed to the console, and saved to `results/<experiment_name>_summary.csv`,
`_summary.json`, and `_folds.csv`. For `config.outcome_t.yaml` that's
`results/outcome_t_gbm_v1_*`.

Latest synthetic run: R² ≈ 0.62, inverse within-one-step ≈ 0.84 vs keep-dose baseline
≈ 0.15 — the recommender clearly beats the naive baseline. Numbers are illustrative until
`data/abt.csv` is the real cleaned data.

---

## Other models / general runner

```bash
python src/run.py --config config.example.yaml    # dose-target models, synthetic by default
```

Set `data.source: synthetic` (runnable today), `file`, or `trial` (once the loader in
`src/data.py` is wired to the trial workbooks).

## Layout

| File | Role | Swap point |
|---|---|---|
| `src/data.py` | Build the Analytics Base Table (ABT) | **change data / cleaning here** |
| `src/models.py` | Model registry (common `fit`/`predict`) | **add/select model here** |
| `src/evaluate.py` | Metric registry + patient-level GroupKFold | **change metrics here** |
| `src/rubric.py` | FDA Table 1 titration logic (ground truth) | fixed clinical rule |
| `src/guardrails.py` | Deterministic safety layer (4 scenarios) | fixed safety layer |
| `src/run.py` | Orchestrator (read config → train → eval → log) | never changes |
| `config.example.yaml` | One experiment = one config | — |

## Registered models

- **Baselines:** `rubric`, `existing_linear_calculator`, `always_237`
- **Direct-dose (Method B):** `ordinal_logistic`, `random_forest`, `gbm_monotone`
- **Serum-T (Method A):** `serumT_then_rule`

`MODEL_KIND` in `models.py` tells the harness how to target/score each: `dose` (train+score
on a dose column), `dose_via_serum` (train on serum T, score on dose — Method A), `serum`
(regression on serum T).

## Note on evaluating Method A fairly

Method A (`serumT_then_rule`) jumps directly to the dose that puts predicted serum T
in-band; it does **not** make the rubric's one-step move. So scoring it by exact-match to
`next_dose_rubric` understates it — Method A should be judged on **serum-T RMSE** and
**% achieved in-band**, not one-step agreement. The harness supports both; pick metrics
that match the model's objective.

## Status snapshot (synthetic data, 300 patients, 5-fold patient-level CV)

Plumbing verified end-to-end; numbers are illustrative until the real ABT is wired.

| Model | Exact dose acc. | Within-one-step |
|---|---|---|
| rubric | ~0.99 | 1.00 |
| gbm_monotone | ~0.97 | 1.00 |
| ordinal_logistic | ~0.97 | 1.00 |
| existing_linear_calculator | ~0.84 | 1.00 |
| always_237 | ~0.25 | 0.72 |

## Next

1. Wire `build_abt_from_trial()` in `src/data.py` to the trial workbooks.
2. Re-run all configs; compare against the three baselines.
3. Confirm the four guardrail scenarios with leadership.
