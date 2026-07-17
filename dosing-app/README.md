# JATENZO Dosing Tool (static app)

A single self-contained `index.html` — no server, no build step, no internet needed. Open it
in any browser, or deploy the folder as a static site (Vercel/Netlify/GitHub Pages: just point
them at this folder; `index.html` is the entry point). Same architecture as the original tool.

## Inputs (match the reference tool)

- **Testosterone units:** ng/dL or nmol/L (toggle)
- **Age:** 18–72, step 2
- **BMI:** 18–40, step 2
- **Current testosterone:** 100–1000 ng/dL, step 100
- **Target testosterone:** 300–1000 ng/dL, step 100
- **Current JATENZO dose:** naive / 158 / 198 / 237 / 316 / 396 mg
- Option: *limit change to one dose step* (titration guideline)

## How it works

The app is **not** running the model live in the browser. At build time the production
model's predicted final testosterone was computed for every grid combination
(28 ages × 12 BMIs × 10 current-T × 5 doses = 16,800 values) and baked into the page. The
browser just: looks up predicted T at each of the five marketed doses, picks the dose whose
predicted T is closest to the target, then applies the directional guardrails (naive → model
start; maintain → keep; raise → step up; lower → step down; optional one-step cap).

The lookup uses the **CatBoost** model with monotone constraints (higher dose / higher current
T never lowers predicted T), so recommendations are clinically sensible.

## Rebuilding

Regenerate `index.html` after the model or data changes by re-running the build script
(`build_app.py`) against `Model-Pipeline-Blueprint/data/abt.csv`. Change the engine, grid, or
dose set there.

## Caveat

Research prototype. Predicted testosterone is a modeled estimate with real uncertainty
(held-out R²-delta ≈ 0.67; exact-dose match is modest). Not medical advice — a clinician must
confirm every dose.
