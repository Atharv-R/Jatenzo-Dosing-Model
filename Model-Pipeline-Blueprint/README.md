# Jatenzo Dosing Model — Pipeline

Modular, config-driven pipeline for testing dosing models. Change the model, its
hyperparameters, the data cleaning, or the metrics by editing **one YAML file** — the code
never changes. See `BLUEPRINT-Evaluation-and-Pipeline.md` for the full design rationale.

## Run it

```bash
pip install -r requirements.txt
python src/run.py --config config.example.yaml    # runs on synthetic data by default
```

Set `data.source: synthetic` (runnable today) or `data.source: trial` (once the real ABT
loader in `src/data.py` is wired to the trial workbooks).

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
