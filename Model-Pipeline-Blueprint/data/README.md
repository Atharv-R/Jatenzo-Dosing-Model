# Where the cleaned data goes

Drop the cleaned Analytics Base Table here as **`abt.csv`**, then set `data.source: file`
and `data.abt_path: data/abt.csv` in your config (already the default in
`config.outcome_t.yaml`).

## Required columns (exact names)

One row = one **(subject × dose-interval)** observation.

| Column | Meaning |
|---|---|
| `subject_id` | patient ID (repeats across rows; used for leakage-safe CV grouping) |
| `age` | years |
| `bmi` | kg/m² |
| `current_T` | T entering the interval (= READING BEFORE) |
| `current_dose` | dose the patient was on before this interval (0 if treatment-naive) |
| `new_dose` | dose applied during this interval (mg; must map to the ladder) |
| `outcome_T` | T achieved at `new_dose` (= READING AFTER) — the model's target |

Optional but recommended: `desired_T`, `next_dose_rubric`, `race`, `ethnicity`, `shbg`, `dietary_fat`.

If any required column is missing, the pipeline stops with a clear message.
Until the real `abt.csv` exists, run with `data.source: synthetic` to exercise everything.

Note: any `abt.csv` currently in this folder is a **synthetic sample** showing the exact
expected format — replace it with the real cleaned data before drawing conclusions.
