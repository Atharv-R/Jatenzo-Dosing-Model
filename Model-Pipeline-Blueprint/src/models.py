"""MODEL registry -- name -> estimator with a common interface.

Every model exposes fit(X, y) and predict(X) so the orchestrator never needs to know
which one it is running. Add a model = register one class. Includes:
  * baselines: rubric, existing linear calculator (proxy), always_237
  * direct-dose (Method B): linear, ordinal_logistic, random_forest, gbm_monotone
  * serum-T (Method A): serumT_then_rule (regress serum T, then apply the label rule)
"""
from __future__ import annotations
import numpy as np

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingRegressor

from rubric import LADDER, next_dose_from_rubric, dose_to_target

# kind decides how the harness targets/scores each model:
#   "dose"            -> train and score on a dose column
#   "dose_via_serum"  -> train on serum_T_6h, score on a dose column (Method A)
#   "serum"           -> train and score on serum_T_6h (regression)
MODEL_KIND = {
    "rubric": "dose", "always_237": "dose", "existing_linear_calculator": "dose",
    "ordinal_logistic": "dose", "random_forest": "dose", "gbm_monotone": "dose",
    "serumT_then_rule": "dose_via_serum",
}

_REGISTRY: dict[str, type] = {}


def register(name):
    def deco(cls):
        _REGISTRY[name] = cls
        return cls
    return deco


def build_model(cfg: dict):
    name = cfg["name"]
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](cfg)


# --------------------------------------------------------------------------- baselines
@register("rubric")
class RubricBaseline:
    """Pure FDA Table 1. Requires current_T (=6h serum) and current_dose as features."""
    def __init__(self, cfg): self.cfg = cfg
    def fit(self, X, y): return self
    def predict(self, X):
        return np.array([next_dose_from_rubric(r["current_T"], int(r["current_dose"]))
                         for _, r in X.iterrows()])


@register("always_237")
class Always237:
    def __init__(self, cfg): pass
    def fit(self, X, y): return self
    def predict(self, X): return np.full(len(X), 237)


@register("existing_linear_calculator")
class LinearCalculatorProxy:
    """Proxy for the deployed calculator: linear regression -> snap to ladder."""
    def __init__(self, cfg): self.m = LinearRegression()
    def fit(self, X, y): self.m.fit(X.values, y); return self
    def predict(self, X):
        raw = self.m.predict(X.values)
        return np.array([min(LADDER, key=lambda d: abs(d - v)) for v in raw])


# --------------------------------------------------------------------- direct-dose (B)
@register("ordinal_logistic")
class OrdinalLogistic:
    """Multinomial logistic over ladder classes (simple ordinal stand-in)."""
    def __init__(self, cfg):
        self.m = LogisticRegression(max_iter=1000, multi_class="multinomial")
    def fit(self, X, y): self.m.fit(X.values, y); return self
    def predict(self, X): return self.m.predict(X.values)


@register("random_forest")
class RandomForest:
    def __init__(self, cfg):
        hp = cfg.get("hyperparameters", {})
        self.m = RandomForestClassifier(
            n_estimators=hp.get("n_estimators", 300),
            max_depth=hp.get("max_depth", None), random_state=0)
    def fit(self, X, y): self.m.fit(X.values, y); return self
    def predict(self, X): return self.m.predict(X.values)


@register("gbm_monotone")
class GBMMonotone:
    """Monotone-constrained gradient boosting: regress dose index, enforce that higher
    current_T never raises the dose. Predicts an ordinal index, maps back to ladder."""
    def __init__(self, cfg):
        hp = cfg.get("hyperparameters", {})
        self.features = None
        self.mono = hp.get("monotone_constraints", {"current_T": -1})
        self.m = HistGradientBoostingRegressor(
            learning_rate=hp.get("learning_rate", 0.05),
            max_iter=hp.get("n_estimators", 400),
            max_depth=hp.get("max_depth", 4), random_state=0)
    def fit(self, X, y):
        self.features = list(X.columns)
        self.m.set_params(monotonic_cst=[self.mono.get(c, 0) for c in self.features])
        idx = np.array([LADDER.index(min(LADDER, key=lambda d: abs(d - v))) for v in y])
        self.m.fit(X.values, idx)
        return self
    def predict(self, X):
        idx = np.clip(np.round(self.m.predict(X.values)).astype(int), 0, len(LADDER) - 1)
        return np.array([LADDER[i] for i in idx])


# ------------------------------------------------------------------------ serum-T (A)
@register("serumT_then_rule")
class SerumTThenRule:
    """Method A. Regress 6h serum T from (dose, covariates); to recommend, predict serum
    T at each candidate dose and choose the lowest that lands in-band, then the rubric."""
    def __init__(self, cfg):
        self.m = HistGradientBoostingRegressor(random_state=0)
        self.cov = None
    def fit(self, X, y_serum):
        # y here must be serum_T_6h (regression target), not a dose.
        self.cov = [c for c in X.columns if c != "current_dose"] + ["current_dose"]
        self.m.fit(X[self.cov].values, y_serum)
        return self
    def predict(self, X):
        out = []
        for _, r in X.iterrows():
            preds = {}
            for d in LADDER:
                row = r.copy(); row["current_dose"] = d
                preds[d] = float(self.m.predict(row[self.cov].values.reshape(1, -1))[0])
            out.append(dose_to_target(preds))
        return np.array(out)
