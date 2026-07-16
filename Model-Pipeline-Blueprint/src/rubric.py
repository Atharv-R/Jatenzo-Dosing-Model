"""FDA JATENZO titration logic (label Table 1).

Ground-truth clinical decision rule. Deterministic: given the serum T drawn 6h after
the morning dose and the current dose, return the next dose. This is *not* ML -- it is
the sanctioned policy that ML models are benchmarked against and that the guardrail
layer enforces.
"""
from __future__ import annotations

# Dose ladder, mg twice daily (BID), in ascending order.
LADDER = [158, 198, 237, 316, 396]

# Target band for 6h post-dose serum total testosterone (ng/dL).
BAND_LOW = 425
BAND_HIGH = 970

DISCONTINUE = 0  # sentinel: below 158 and still >970 -> discontinue treatment


def next_dose_from_rubric(serum_t_6h: float, current_dose: int) -> int:
    """Return the next JATENZO dose per FDA Table 1.

    serum_t_6h : serum total T (ng/dL) drawn 6h after the morning dose.
    current_dose : current mg BID; must be on LADDER.
    Returns next dose (mg BID), or DISCONTINUE (0).
    """
    if current_dose not in LADDER:
        # Snap to nearest ladder rung before applying the rule.
        current_dose = min(LADDER, key=lambda d: abs(d - current_dose))
    i = LADDER.index(current_dose)

    if serum_t_6h < BAND_LOW:            # too low -> step up (capped at 396)
        return LADDER[min(i + 1, len(LADDER) - 1)]
    if serum_t_6h > BAND_HIGH:           # too high -> step down, or discontinue at floor
        return DISCONTINUE if i == 0 else LADDER[i - 1]
    return current_dose                  # in band -> no change


def dose_to_target(pred_serum_by_dose: dict[int, float]) -> int:
    """Pick the lowest ladder dose whose *predicted* 6h serum T lands in-band.

    pred_serum_by_dose : {dose_mg: predicted_serum_T}. Used by Method A
    (predict serum T at each candidate dose, then choose).
    Falls back to the dose with predicted level closest to the band midpoint.
    """
    midpoint = (BAND_LOW + BAND_HIGH) / 2
    in_band = [d for d in sorted(pred_serum_by_dose)
               if BAND_LOW <= pred_serum_by_dose[d] <= BAND_HIGH]
    if in_band:
        return in_band[0]
    return min(pred_serum_by_dose,
               key=lambda d: abs(pred_serum_by_dose[d] - midpoint))
