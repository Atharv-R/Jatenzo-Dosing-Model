"""Metrics report in the colleague's two-table format.

    python src/report.py --config config.outcome_t.yaml

Table A - Statistical accuracy (predicting testosterone):
    n | R2 (final T) | R2 (delta T) | RMSE | MAE | MSE | % within 150 ng/dL
  across three regimes: In-sample (full fit), Out-of-sample (CV out-of-fold),
  Out-of-sample (held-out patients).

Table B - Dose-recommendation accuracy (inverse-recommendation test):
    n | Exact % | Within-1-step % | n(clean) | Exact %(clean) | Within-1 %(clean)
  for the production model (full data) and held-out patients. 'Clean subset' = rows whose
  actual next dose is a real marketed rung (no snapping).

Why two R2s: raw OUTCOME_T variance is dominated by regression-to-the-mean noise, so
R2(final T) can look weak even with real skill. R2(delta T) scores predicting the CHANGE
and is the more honest read. RMSE/MAE/MSE/%-within are identical either way (current_T
cancels in the residual).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd, yaml
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as data_mod
import models as models_mod

MARKETED = [158, 198, 237, 316, 396]     # real Jatenzo rungs (474 is an extrapolated step)


def _stat_row(y_true, y_pred, current_T):
    y_true, y_pred, cT = map(lambda a: np.asarray(a, float), (y_true, y_pred, current_T))
    resid = y_pred - y_true
    sse = float(np.sum(resid ** 2)); n = len(y_true)
    ss_final = float(np.sum((y_true - y_true.mean()) ** 2))
    d_true = y_true - cT
    ss_delta = float(np.sum((d_true - d_true.mean()) ** 2))
    return {
        "n": n,
        "R2 (final T)": 1 - sse / ss_final if ss_final else float("nan"),
        "R2 (delta T)": 1 - sse / ss_delta if ss_delta else float("nan"),
        "RMSE": np.sqrt(sse / n), "MAE": float(np.mean(np.abs(resid))),
        "MSE": sse / n, "% within 150": 100 * float(np.mean(np.abs(resid) <= 150)),
    }


def _rec_row(model, df, ladder):
    idx = {d: i for i, d in enumerate(ladder)}
    rec = model.recommend(df, df["outcome_T"].values)
    taken = df["new_dose"].values
    exact = 100 * float(np.mean(rec == taken))
    within1 = 100 * float(np.mean([abs(idx[a] - idx[b]) <= 1 for a, b in zip(rec, taken)]))
    clean = np.isin(taken, MARKETED)
    rc, tc = rec[clean], taken[clean]
    ex_c = 100 * float(np.mean(rc == tc)) if clean.sum() else float("nan")
    w1_c = 100 * float(np.mean([abs(idx[a] - idx[b]) <= 1 for a, b in zip(rc, tc)])) if clean.sum() else float("nan")
    return {"n": len(df), "Exact %": exact, "Within 1 step %": within1,
            "n (clean)": int(clean.sum()), "Exact % (clean)": ex_c, "Within 1 step % (clean)": w1_c}


def _behavior_from(rec, taken, cur, ladder):
    """Dosing-behavior metrics from recommendation/actual/current dose arrays.

    over/under/exact/mis-dose are vs the ACTUAL dose (accuracy, needs real data).
    '% >1 rung from current' is vs the patient's CURRENT dose (guideline: avoid multi-rung
    moves) and is a pure behavioral property of the recommender.
    """
    idx = {d: i for i, d in enumerate(ladder)}
    ri = np.array([idx[x] for x in rec]); ti = np.array([idx[x] for x in taken])
    over = 100 * float(np.mean(ri > ti)); under = 100 * float(np.mean(ri < ti))
    exact = 100 * float(np.mean(ri == ti))
    mask = np.array([c in idx for c in cur])
    multi = float("nan")
    if mask.sum():
        ci = np.array([idx[c] for c in np.asarray(cur)[mask]])
        multi = 100 * float(np.mean(np.abs(ri[mask] - ci) > 1))
    return {"n": len(rec), "% over": over, "% under": under, "% exact": exact,
            "% mis-dosed": over + under, "% >1 rung from current": multi}


def _behavior(model, df, ladder):
    rec = model.recommend(df, df["outcome_T"].values)
    return _behavior_from(rec, df["new_dose"].values, df["current_dose"].values, ladder)


def stress_test(cfg, n=1_000_000, seed=0):
    """Behavioral stress test over synthetic input combinations. VALID for behavioral
    metrics only (e.g. how often the recommender would move >1 rung from current) -- these
    depend only on the model's decision function, not on unknown ground truth. NOT valid
    for accuracy (over/under/mis-dose) since mock patients have no observed outcome.

    Inputs are sampled from the REAL data's ranges so the combinations stay plausible.
    """
    df = data_mod.build_abt({**cfg["data"], "seed": cfg.get("experiment", {}).get("seed", 42)})
    model = models_mod.build_model(cfg["model"]).fit(df, df["outcome_T"])
    ladder = model.ladder
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "age": rng.integers(int(df.age.min()), int(df.age.max()) + 1, n),
        "bmi": rng.uniform(df.bmi.min(), df.bmi.max(), n),
        "current_T": rng.uniform(df.current_T.quantile(.02), df.current_T.quantile(.98), n),
        "current_dose": rng.choice(ladder, n),
        "new_dose": 0,
    })
    desired = rng.uniform(300, 900, n)                      # plausible target band
    rec = model.recommend(X, desired)
    idx = {d: i for i, d in enumerate(ladder)}
    ri = np.array([idx[x] for x in rec]); ci = np.array([idx[c] for c in X.current_dose])
    jump = np.abs(ri - ci)
    return {"combinations": n, "% >1 rung from current": 100 * float(np.mean(jump > 1)),
            "% up >1 rung": 100 * float(np.mean((ri - ci) > 1)),
            "% down >1 rung": 100 * float(np.mean((ci - ri) > 1)),
            "% within 1 rung": 100 * float(np.mean(jump <= 1))}


def build_report(cfg):
    df = data_mod.build_abt({**cfg["data"], "seed": cfg.get("experiment", {}).get("seed", 42)})
    mcfg = cfg["model"]
    groups = df["subject_id"].values
    make = lambda: models_mod.build_model(mcfg)

    # patient hold-out (dev vs held-out test)
    dev_i, test_i = next(GroupShuffleSplit(1, test_size=0.15, random_state=0)
                         .split(df, groups=groups))
    dev, test = df.iloc[dev_i], df.iloc[test_i]

    # In-sample: fit on all, predict all
    m_all = make().fit(df, df["outcome_T"])
    stat = {"In-sample (full-data fit)": _stat_row(df["outcome_T"], m_all.predict(df), df["current_T"])}

    # CV out-of-fold on the dev set
    oof = np.empty(len(dev)); dg = dev["subject_id"].values
    for tr, te in GroupKFold(5).split(dev, dev["outcome_T"], dg):
        m = make().fit(dev.iloc[tr], dev.iloc[tr]["outcome_T"])
        oof[te] = m.predict(dev.iloc[te])
    stat["Out-of-sample (CV, out-of-fold)"] = _stat_row(dev["outcome_T"], oof, dev["current_T"])

    # Held-out patients
    m_dev = make().fit(dev, dev["outcome_T"])
    stat["Out-of-sample (held-out patients)"] = _stat_row(test["outcome_T"], m_dev.predict(test), test["current_T"])

    # Dose recommendation
    rec = {"In-sample (production model, full data)": _rec_row(m_all, df, m_all.ladder),
           "Out-of-sample (held-out patients)": _rec_row(m_dev, test, m_dev.ladder)}

    # Dosing behavior. Full-sample = out-of-fold recommendation for every row.
    ladder = m_all.ladder
    oof_rec = np.empty(len(df), dtype=int)
    for tr, te in GroupKFold(5).split(df, df["outcome_T"], groups):
        mm = make().fit(df.iloc[tr], df.iloc[tr]["outcome_T"])
        oof_rec[te] = mm.recommend(df.iloc[te], df.iloc[te]["outcome_T"].values)
    beh = {
        "In-sample (full-data fit)": _behavior(m_all, df, ladder),
        "Full sample (CV, out-of-fold)": _behavior_from(
            oof_rec, df["new_dose"].values, df["current_dose"].values, ladder),
        "Out-of-sample (held-out patients)": _behavior(m_dev, test, ladder),
    }

    return pd.DataFrame(stat).T, pd.DataFrame(rec).T, pd.DataFrame(beh).T


def _html(stat, rec, beh, cfg, stress=None):
    eng = cfg["model"].get("engine", "histgbm"); tgt = cfg["model"].get("target", "outcome_T")
    sty = ("<style>body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:24px;color:#222}"
           "h2{margin-top:28px}table{border-collapse:collapse;width:100%;margin-top:10px}"
           "th,td{border:1px solid #ddd;padding:8px 12px;text-align:right}"
           "th:first-child,td:first-child{text-align:left}th{background:#f5f6f8}"
           "p{color:#555;max-width:900px}</style>")
    note_a = ("<p>R2 (final T) is against actual OUTCOME_T. Raw T variance is dominated by "
              "regression-to-the-mean noise, so it can look weak even with real skill. "
              "<b>R2 (delta T)</b> scores predicting the change and is the more honest read. "
              "RMSE/MAE/MSE/%-within are identical either way (current_T cancels).</p>")
    note_b = ("<p>For each row we know the dose given and the T achieved. Setting desired_T to "
              "the achieved T and asking the recommender for a dose: does it match reality? "
              "'Clean subset' restricts to rows whose actual next dose is a real marketed rung.</p>")
    note_c = ("<p><b>% over/under/exact/mis-dosed</b> compare the recommended dose to the dose "
              "actually taken (accuracy). <b>% &gt;1 rung from current</b> is a guideline check "
              "(titration should avoid multi-rung jumps) and is a behavioral property of the "
              "recommender vs the patient's current dose.</p>")
    cell = lambda v: (f"{v:.0f}" if isinstance(v, float) and abs(v) >= 100
                      else (f"{v:.3f}" if isinstance(v, float) else v))
    fmt = lambda d: d.apply(lambda col: col.map(cell))
    stress_html = ""
    if stress:
        rows = "".join(f"<tr><td>{k}</td><td>{v:,.0f}</td></tr>" if k == "combinations"
                       else f"<tr><td>{k}</td><td>{v:.2f}</td></tr>" for k, v in stress.items())
        stress_html = ("<h2>Behavioral stress test (synthetic input combinations)</h2>"
                       "<p>Valid for behavioral metrics only - these depend on the model's "
                       "decision function, not on unknown ground truth. Inputs sampled from the "
                       "real data's ranges.</p>"
                       f"<table><tr><th>metric</th><th>value</th></tr>{rows}</table>")
    return (f"<html><head>{sty}</head><body>"
            f"<h1>Jatenzo dose model - metrics ({eng}, target={tgt})</h1>"
            f"<h2>Statistical accuracy</h2>{note_a}{fmt(stat).to_html()}"
            f"<h2>Dose-recommendation accuracy (inverse recommendation test)</h2>{note_b}{fmt(rec).to_html()}"
            f"<h2>Dosing behavior (over/under, mis-dose, multi-rung)</h2>{note_c}{fmt(beh).to_html()}"
            f"{stress_html}"
            "</body></html>")


def main(config_path, stress_n=0):
    cfg = yaml.safe_load(Path(config_path).read_text())
    stat, rec, beh = build_report(cfg)
    stress = stress_test(cfg, n=stress_n) if stress_n else None
    out = Path(cfg.get("experiment", {}).get("results_dir", "results")); out.mkdir(exist_ok=True, parents=True)
    name = cfg.get("experiment", {}).get("name", "run")
    stat.to_csv(out / f"{name}_statistical_accuracy.csv")
    rec.to_csv(out / f"{name}_dose_recommendation.csv")
    beh.to_csv(out / f"{name}_dosing_behavior.csv")
    (out / f"{name}_report.html").write_text(_html(stat, rec, beh, cfg, stress))
    pd.set_option("display.width", 160, "display.max_columns", 20)
    print("\n=== Statistical accuracy ===\n", stat.round(3).to_string())
    print("\n=== Dose-recommendation accuracy ===\n", rec.round(1).to_string())
    print("\n=== Dosing behavior (over/under, mis-dose, multi-rung) ===\n", beh.round(1).to_string())
    if stress:
        print(f"\n=== Behavioral stress test ({stress['combinations']:,} synthetic combos) ===")
        for k, v in stress.items():
            print(f"   {k}: {v:,.2f}" if k != "combinations" else f"   {k}: {v:,}")
    print(f"\nSaved -> {out}/{name}_report.html (+ CSVs)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.outcome_t.yaml")
    ap.add_argument("--stress", type=int, default=0, help="run behavioral stress test over N synthetic combos")
    a = ap.parse_args(); main(a.config, a.stress)
