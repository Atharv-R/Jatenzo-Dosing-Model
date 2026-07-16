# Jatenzo Dosing Model — Design Blueprint

**Goal:** Recommend a Jatenzo (oral testosterone undecanoate) dose for a hypogonadal
patient, grounded in the drug's real behavior and the FDA titration label, using the
pivotal-trial clinical dataset.

This document lays out two candidate architectures, argues why the **serum-T (exposure)
method is the more robust one**, explains how the **direct recommended-dose method**
falls short and how to harden it if we go that way, and defines a shared blueprint so we
can commit to a direction late.

---

## 0. Ground truth: how Jatenzo dosing actually works

Any model must respect the label, because the label *is* the clinical decision rule.

- Fixed dose ladder, twice daily: **158 → 198 → 237 → 316 → 396 mg**. Start at 237.
- Decision variable: **serum total testosterone drawn 6 hours after the morning dose**,
  measured ≥7 days after any dose change.
- Titration rule (FDA Table 1):

| 6h serum T | Action |
|---|---|
| < 425 ng/dL | step **up** one level |
| 425–970 ng/dL | **no change** (target band) |
| > 970 ng/dL | step **down** one level; if already at 158 → discontinue |

Two facts drive the whole design:
1. The decision is a **fixed, auditable rule** on a **single measured quantity** (6h serum T).
2. The only genuinely patient-variable, hard-to-predict part is **what serum T a given
   patient will produce at a given dose** — this is a pharmacokinetics (PK) problem, and
   for oral TU it is strongly influenced by **SHBG and dietary fat/food**, less so by age/BMI.

---

## 1. Method A — Predict serum T, then apply the label rule (RECOMMENDED)

### Concept
A **two-stage** system:

- **Stage 1 (learned): an exposure model.** Predict on-treatment serum testosterone
  (at the label's 6h timepoint, and/or the trial's 24h average `Cavg24`) as a function of
  **dose + patient covariates**. This is a dose–exposure / PK model.
- **Stage 2 (fixed): the titration rule.** Feed the predicted serum T into the FDA Table 1
  logic to output the dose recommendation. Deterministic, transparent, safe.

```
covariates + candidate dose ──▶ [Exposure model] ──▶ predicted 6h serum T (+ uncertainty)
                                                              │
                                                              ▼
                                                   [FDA Table 1 rule] ──▶ dose recommendation
```

### Why this is the more robust method

1. **It mirrors the drug's causal chain.** dose → absorption (food/SHBG dependent) →
   serum T → clinical decision. The model only learns the part that's actually uncertain
   (exposure) and leaves the regulated decision to a fixed rule.
2. **It reasons counterfactually — which is the whole point of a recommender.** To pick a
   *new* dose you must estimate serum T at doses the patient has **not yet tried**. Because
   the exposure model is *parameterized by dose*, it can predict along the dose axis and
   answer "what would happen at 316 vs 396?" A direct dose model (Method B) has no such
   handle — it only interpolates observed input→dose pairs.
3. **Safety by construction.** Every recommendation comes out of the label band. The learned
   component cannot invent an off-label jump; the worst it can do is mis-predict serum T,
   which we surface with an uncertainty interval.
4. **Interpretable / defensible.** You can show a clinician "predicted 6h serum T = 510 ng/dL
   (95% PI 430–600) → within band → no change." That is auditable to regulators and trusted
   by prescribers.
5. **The data gives you the exact target.** The dataset directly measures dose→serum T at
   multiple timepoints (`Hour 4 T`, 6h reads, `Peak T`, `Cavg24`, and derived PK params in
   `adtespar`). The label-relevant outcome is literally a column, not a proxy.
6. **Personalization = the real clinical win.** Predict where a *new* patient will land
   *before* titrating, so they start near their eventual maintenance dose → fewer clinic
   visits and blood draws.

### How Stage 1 can be built (two options, can escalate)

- **(A1) ML regression** — gradient-boosted trees / mixed-effects ML predicting the summary
  exposure (6h serum T and/or `Cavg24`) from dose + covariates. Fast, simple, good enough
  for a recommender. Start here.
- **(A2) Population PK (NLME / compartmental)** — the gold standard for this data shape.
  Note the `pkdata- working` sheet is already in **NONMEM format** (`NMID, EVID, AMT, II,
  SS, MDV, ACTUALTI` columns), i.e. the sponsor modeled it this way. A popPK model estimates
  absorption/clearance and covariate effects (SHBG, food/fat) and yields a full
  concentration–time curve per patient. More work, most mechanistic. Optional Phase-2+.

### Recommendation loop
For a patient's covariates, predict 6h serum T at each ladder dose; choose the lowest dose
whose predicted level lands in 425–970 (or, from a known current dose, apply the one-step
rule). Carry a predictive interval to flag overshoot risk (>970) or undershoot (<425).

### Weaknesses to manage
- Needs the label covariates measured (SHBG, dietary fat) to be accurate; if a deployment
  setting lacks SHBG, accuracy drops.
- Two-stage error propagation — mitigated by reporting Stage-1 uncertainty.

---

## 2. Method B — Directly predict the recommended dose (the existing calculator's approach)

### Concept
A **single model** maps `(age, BMI, current T, target T, current dose) → recommended dose`.
This is what `trail-rho-three.vercel.app` does (linear regression onto the dose ladder).

### Why it's less robust

1. **No causal grounding.** It learns a correlation between inputs and the dose that was
   eventually assigned, not the biology — so it can't explain *why*, and generalizes poorly
   off-distribution.
2. **Counterfactual blindness.** It never estimates "serum T at dose X," so it can't truly
   reason about untried doses; it interpolates historical input→dose pairs.
3. **It can contradict the label.** Continuous linear output can land off-ladder or
   disagree with the titration band, requiring post-hoc clamping that papers over the model.
4. **The training target is itself rule-derived.** "Recommended dose" = (titration rule ∘ PK).
   Learning that composite end-to-end from ~300 patients is far less sample-efficient than
   learning PK and bolting on the *known* rule (Method A).
5. **Conceptual mismatch in its inputs.** A free "target testosterone" (300–1000) doesn't
   exist in the label — the target band is fixed at 425–970. And "current T" (baseline)
   isn't the label's decision variable; the **on-treatment 6h serum T** is.
6. **Extrapolation risk.** Linear regression extrapolates unbounded outside the training
   range (age/BMI extremes) and returns nonsense at the edges (empty inputs → 0 mg).

### If we build it anyway — how to make it much better

- **Reframe as ordinal classification** over the 5 ladder levels (not continuous
  regression) → respects discreteness and ordering. Use ordinal logistic / ordered boosting.
- **Constrain the output space:** snap to the ladder **and** limit to **≤1 step from the
  current dose per visit**, mirroring the label's one-step titration.
- **Fix the features:** drop the free "target T"; replace generic "current T" with the
  **on-treatment 6h serum T**; add **SHBG, baseline T, dietary fat**. Keep age/BMI as weak
  covariates.
- **Enforce monotonicity:** recommended dose must be **monotonically non-increasing in
  observed serum T** (higher serum → step down). Monotone-constrained GBMs (e.g. LightGBM
  monotone constraints) enforce this, killing a whole class of unsafe outputs.
- **Quantify uncertainty and defer:** when confidence is low, fall back to the label rule.
- **Define the target correctly:** train against the label-rule output *or* the actual
  next-visit dose from the dosing-history domain (`adex`/`addosad`), and validate on
  **held-out patients**, not rows.

Note the punchline: every improvement above nudges Method B toward "respect the rule + use
the real decision variable" — i.e. toward Method A. That's the tell that A is the cleaner
formulation.

---

## 3. Head-to-head

| Dimension | A: Serum-T + rule | B: Direct dose |
|---|---|---|
| Grounded in drug mechanism | ✅ Yes (PK) | ❌ Correlational |
| Reasons about untried doses | ✅ Yes | ❌ No |
| Cannot violate label | ✅ By construction | ⚠️ Needs clamping |
| Interpretable to clinician | ✅ Predicted serum T + rule | ⚠️ Black box |
| Sample efficiency (~300 pts) | ✅ Learns only PK | ❌ Learns PK∘rule |
| Uses the label's real variable (6h T) | ✅ | ❌ (uses baseline) |
| Build complexity | Medium | Low |
| Personalization / shorten titration | ✅ Strong | ⚠️ Weak |

---

## 4. Unified blueprint (build once, choose the direction late)

Design so both methods share infrastructure and differ only in the decision layer.

**Layer 0 — Data layer (shared).** Harmonize the sheets into one tidy
per-subject-per-visit table keyed on `USUBJID`:
- Covariates: age, BMI, race, **SHBG**, baseline T, **dietary fat/meal**.
- Treatment: current dose, dose history (`adex`, `addosad`).
- Outcomes: serum T at 4h / 6h / peak, `Cavg24`, `Cmax24`, `AUC24` (`adtespar`, `pkdata`).
- Safety: hematocrit, blood pressure, AEs (`adae`, `TEAE`).

**Layer 1 — Exposure model (Method A core).** dose + covariates → predicted serum T
(6h and `Cavg24`) with uncertainty. Start GBM; optionally popPK later.

**Layer 2 — Decision layer (swappable).**
- **Path A:** apply FDA Table 1 to Layer-1 predictions.
- **Path B:** constrained ordinal model (ladder + one-step + monotonic) → dose.
- Both emit the **same interface:**
  `{recommended_dose, predicted_serum_T, rationale, confidence, safety_flags}`.

**Layer 3 — Evaluation harness (shared).** Patient-level cross-validation. Metrics:
- Path A: serum-T RMSE + calibration, % predicted-in-band.
- Path B: exact-dose accuracy, within-one-step accuracy, ordinal loss.
- Both: agreement with label rule, agreement with **actual trial clinician decisions**,
  and **safety-violation rate** (recs that would push >970 or <425).

**Layer 4 — Interface.** Rebuild the calculator UI backed by whichever decision path,
now displaying predicted serum T + rationale, not just a number.

### Phasing
1. **Phase 1 — Data layer + EDA.** Confirm dose–response; quantify SHBG/food effects; sanity-check against the label bands.
2. **Phase 2 — Layer-1 exposure model.** GBM on 6h T / `Cavg24` with patient-level CV.
3. **Phase 3 — Decision layer, both paths + eval harness.** Direct comparison on the same folds.
4. **Phase 4 — Choose, then wrap in UI.** Benchmark against the existing calculator.

The key property: **Phases 1–3 are identical regardless of direction.** The A-vs-B choice
is isolated to Layer 2, so we don't have to commit until we've seen both on the same data.
