"""Orchestrator: read config -> build ABT -> train -> evaluate -> log.

    python src/run.py --config config.example.yaml

Swapping data / model / metrics is done entirely in the YAML; this file never changes.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))  # allow flat imports
import data as data_mod
import models as models_mod
import evaluate as eval_mod


def main(config_path: str):
    cfg = yaml.safe_load(Path(config_path).read_text())

    # ---- DATA -------------------------------------------------------------------
    dcfg = cfg["data"]
    dcfg.setdefault("seed", cfg.get("experiment", {}).get("seed", 42))
    df = data_mod.build_abt(dcfg)
    if "group_key" not in df:
        df["group_key"] = df["patient_id"]
    features = dcfg["features"] + dcfg.get("extended_features", [])
    eval_target = dcfg["target"]                    # a dose column, e.g. next_dose_rubric
    name = cfg["model"]["name"]
    kind = models_mod.MODEL_KIND.get(name, "dose")

    # Choose train target + metric family from the model kind.
    dose_metrics = {"exact_dose_accuracy", "within_one_step_accuracy", "dose_mae_steps",
                    "quadratic_weighted_kappa", "agreement_with_rubric", "agreement_with_clinician"}
    reg_metrics = {"rmse", "pct_in_band"}
    if kind == "serum":
        train_target = eval_target = "serum_T_6h"
        allowed = reg_metrics
    elif kind == "dose_via_serum":
        train_target, allowed = "serum_T_6h", dose_metrics
    else:  # plain dose model
        train_target, allowed = eval_target, dose_metrics
        if eval_target == "serum_T_6h":             # guard against a misconfigured target
            train_target = eval_target = "next_dose_rubric"

    metrics = [m for m in cfg["evaluation"]["metrics"] if m in allowed] or list(allowed)[:1]
    print(f"ABT: {len(df)} rows, {df['group_key'].nunique()} patients | "
          f"features={features} | model={name} ({kind}) | "
          f"train={train_target} eval={eval_target}")

    # ---- MODEL + EVAL -----------------------------------------------------------
    model_factory = lambda: models_mod.build_model(cfg["model"])
    folds, summary = eval_mod.run_cv(
        model_factory, df, features, train_target, eval_target, metrics, cfg["evaluation"]["cv"])

    # ---- LOG --------------------------------------------------------------------
    out = Path(cfg.get("experiment", {}).get("results_dir", "results"))
    out.mkdir(parents=True, exist_ok=True)
    name = cfg.get("experiment", {}).get("name", "run")
    folds.to_csv(out / f"{name}_folds.csv", index=False)
    summary.to_csv(out / f"{name}_summary.csv")
    (out / f"{name}_summary.json").write_text(
        json.dumps(summary["mean"].round(4).to_dict(), indent=2))

    print("\n=== CV summary (mean +/- std over folds) ===")
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
        print(summary)
    print(f"\nSaved -> {out}/{name}_*.csv/json")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.example.yaml")
    main(ap.parse_args().config)
