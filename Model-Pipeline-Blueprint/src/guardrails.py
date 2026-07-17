"""Deterministic safety layer -- the 'four scenarios' guardrails.

The ML model *proposes*; these guardrails *dispose*. Every recommendation is clamped to
the label ladder, limited to one step per visit, forced to discontinue at the floor when
indicated, and downgraded to 'defer / re-measure' when inputs are out-of-distribution or
model confidence is low.
"""
from __future__ import annotations
from dataclasses import dataclass, field

from rubric import LADDER, DISCONTINUE


@dataclass
class GuardrailConfig:
    clamp_to_ladder: bool = True
    max_step_per_visit: int = 1
    enforce_discontinue_at_floor: bool = True
    ood_defer_to_rubric: bool = True
    low_confidence_action: str = "remeasure_7d"


@dataclass
class Decision:
    dose: int                      # final recommended dose (mg BID) or 0 = discontinue
    rationale: str
    flags: list[str] = field(default_factory=list)


def directional_dose(proposed_dose: int, current_dose: int,
                     current_T: float, desired_T: float,
                     *, tol: float = 25.0, start_dose: int = 237,
                     ladder: list | None = None,
                     naive_policy: str = "model") -> Decision:
    """Directional guardrails -- force the recommendation's DIRECTION to match the
    intended change in testosterone. These are the four scenarios:

      Scenario 4 (naive):    no current dose -> see naive_policy below.
      Scenario 1 (maintain): desired T within +/-tol of current T -> KEEP current dose.
      Scenario 2 (raise):    desired T above current T -> dose must step UP (magnitude
                             from the model -- a big jump toward the goal is allowed).
      Scenario 3 (lower):    desired T below current T -> dose must step DOWN.

    `proposed_dose` is what the ML recommender suggested; this only *corrects its
    direction*, it does not cap how far the model jumps toward the goal.

    naive_policy for a treatment-naive patient:
      "model"          -> trust the model's dose (snapped to a rung). Best when the goal
                          is a large T increase -- a fixed low start would be wrong.
      "standard_start" -> the label's conservative fixed start (then titrate).
    """
    rungs = ladder if ladder is not None else LADDER
    # Scenario 4: treatment-naive.
    if current_dose == 0 or current_dose not in rungs:
        if naive_policy == "standard_start":
            return Decision(start_dose, "Treatment-naive: label's standard start dose.",
                            ["naive_standard_start"])
        prop = min(rungs, key=lambda d: abs(d - proposed_dose))
        return Decision(prop, "Treatment-naive: model-recommended starting dose "
                              "(chosen to reach the target T).", ["naive_model"])

    ci = rungs.index(current_dose)
    prop = min(rungs, key=lambda d: abs(d - proposed_dose))    # snap proposal to ladder
    pi = rungs.index(prop)
    diff = desired_T - current_T

    if abs(diff) <= tol:                                       # Scenario 1: maintain
        return Decision(current_dose, "Goal is to maintain T: keep current dose.",
                        ["maintain"])
    if diff > 0:                                               # Scenario 2: raise
        ti = max(pi, ci + 1)                                   # must be strictly higher
        dose = rungs[min(ti, len(rungs) - 1)]
        flags = ["raise"] + (["forced_step_up"] if pi <= ci else [])
        return Decision(dose, "Goal is higher T: dose stepped up.", flags)
    # Scenario 3: lower
    ti = min(pi, ci - 1)                                       # must be strictly lower
    dose = rungs[max(ti, 0)]
    flags = ["lower"] + (["forced_step_down"] if pi >= ci else [])
    return Decision(dose, "Goal is lower T: dose stepped down.", flags)


def _snap(dose: int) -> int:
    if dose == DISCONTINUE:
        return DISCONTINUE
    return min(LADDER, key=lambda d: abs(d - dose))


def apply_guardrails(proposed_dose: int,
                     current_dose: int,
                     cfg: GuardrailConfig,
                     *,
                     ood: bool = False,
                     low_confidence: bool = False,
                     rubric_dose: int | None = None) -> Decision:
    """Wrap a model's proposed dose with the safety layer.

    proposed_dose : the ML model's raw suggestion (mg).
    current_dose  : patient's current dose (mg).
    ood           : True if inputs are out-of-distribution (Scenario 3).
    low_confidence: True if predictive interval crosses a band edge (Scenario 4).
    rubric_dose   : label-rule dose to fall back to when deferring.
    """
    flags: list[str] = []

    # Scenario 3: out-of-distribution -> defer to the label rule.
    if ood and cfg.ood_defer_to_rubric and rubric_dose is not None:
        return Decision(_snap(rubric_dose), "OOD inputs: deferred to FDA rubric.",
                        ["ood", "deferred_to_rubric"])

    # Scenario 4: low confidence -> don't jump; re-measure per label (7-day recheck).
    if low_confidence and cfg.low_confidence_action == "remeasure_7d":
        return Decision(_snap(current_dose),
                        "Low confidence: hold dose, re-measure serum T in 7 days.",
                        ["low_confidence", "remeasure_7d"])

    dose = proposed_dose

    if cfg.clamp_to_ladder and dose != DISCONTINUE:
        snapped = _snap(dose)
        if snapped != dose:
            flags.append("clamped_to_ladder")
        dose = snapped

    # Scenario 2: ladder boundaries + one-step limit.
    if dose != DISCONTINUE and current_dose in LADDER:
        ci, ti = LADDER.index(current_dose), LADDER.index(dose)
        if abs(ti - ci) > cfg.max_step_per_visit:
            ti = ci + cfg.max_step_per_visit * (1 if ti > ci else -1)
            dose = LADDER[ti]
            flags.append("limited_to_one_step")

    # Scenario 2: discontinue only from the floor.
    if proposed_dose == DISCONTINUE:
        if cfg.enforce_discontinue_at_floor and current_dose != LADDER[0]:
            dose = LADDER[0]
            flags.append("discontinue_blocked_not_at_floor")
        else:
            dose = DISCONTINUE

    return Decision(dose, "Model proposal passed through guardrails.", flags)
