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
