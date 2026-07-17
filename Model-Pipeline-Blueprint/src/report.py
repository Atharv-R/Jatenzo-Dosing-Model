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

    return pd.DataFrame(stat).T, pd.DataFrame(rec).T


def _html(stat, rec, cfg):
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
    cell = lambda v: (f"{v:.0f}" if isinstance(v, float) and abs(v) >= 100
                      else (f"{v:.3f}" if isinstance(v, float) else v))
    fmt = lambda d: d.apply(lambda col: col.map(cell))
    return (f"<html><head>{sty}</head><body>"
            f"<h1>Jatenzo dose model - metrics ({eng}, target={tgt})</h1>"
            f"<h2>Statistical accuracy</h2>{note_a}{fmt(stat).to_html()}"
            f"<h2>Dose-recommendation accuracy (inverse recommendation test)</h2>{note_b}{fmt(rec).to_html()}"
            "</body></html>")


def main(config_path):
    cfg = yaml.safe_load(Path(config_path).read_text())
    stat, rec = build_report(cfg)
    out = Path(cfg.get("experiment", {}).get("results_dir", "results")); out.mkdir(exist_ok=True, parents=True)
    name = cfg.get("experiment", {}).get("name", "run")
    stat.to_csv(out / f"{name}_statistical_accuracy.csv")
    rec.to_csv(out / f"{name}_dose_recommendation.csv")
    (out / f"{name}_report.html").write_text(_html(stat, rec, cfg))
    pd.set_option("display.width", 140, "display.max_columns", 20)
    print("\n=== Statistical accuracy ===\n", stat.round(3).to_string())
    print("\n=== Dose-recommendation accuracy ===\n", rec.round(1).to_string())
    print(f"\nSaved -> {out}/{name}_report.html (+ CSVs)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.outcome_t.yaml")
    main(ap.parse_args().config)
