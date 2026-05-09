"""Exit Pressure Score — unified weighted formula.

Aggregates the five exit-strategy signals into one operator-visible
score on a 0-100 scale:

  * 30% Theme Deterioration   — composite trend + below-floor streak
  * 20% Profit Preservation   — which trim ladder band is active
  * 20% Technical Exhaustion  — momentum-exhaustion signals
  * 15% Options Risk          — short-call delta / earnings (PMCC only)
  * 15% Better Opportunity    — rotation candidate score delta

Bands (operator spec):
  <40    Hold
  40-60  Trim 10-20%
  60-75  Trim 25-50%
  >75    Exit / rotate aggressively

Each sub-score is mapped to 0-100 by a small monotone function, then
weighted-summed. The score is informational — actual trim/exit
decisions remain with the individual modules. The score is written to
auto_actions per maintenance tick so operators can see exit pressure
on every position at a glance, and so we can backtest band
calibrations against future outcomes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Weight allocation matches the operator's spec (sums to 1.0).
WEIGHT_THEME_DETERIORATION   = 0.30
WEIGHT_PROFIT_PRESERVATION   = 0.20
WEIGHT_TECH_EXHAUSTION       = 0.20
WEIGHT_OPTIONS_RISK          = 0.15
WEIGHT_ROTATION_PRESSURE     = 0.15

# Band thresholds.
BAND_HOLD_MAX                = 40.0
BAND_TRIM_LIGHT_MAX          = 60.0
BAND_TRIM_HEAVY_MAX          = 75.0
# Above 75 -> aggressive exit / rotation


@dataclass
class ExitPressure:
    score: float = 0.0                              # 0-100 weighted sum
    band: str = "hold"                              # hold | trim_light | trim_heavy | aggressive
    sub_scores: dict[str, float] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    rationale: str = ""
    detail: dict[str, Any] = field(default_factory=dict)


def _theme_deterioration_subscore(
    composite: Optional[float],
    streak_days_below_floor: int,
) -> float:
    """0-100. Streak weighting: 1d=20, 2d=40, 3d=60, 4d=80, 5d=100. Composite
    below 40 adds another 20 (capped at 100)."""
    streak_part = min(100.0, streak_days_below_floor * 20.0)
    composite_bonus = 20.0 if (composite is not None and composite < 40.0) else 0.0
    return min(100.0, streak_part + composite_bonus)


def _profit_preservation_subscore(trim_band: str) -> float:
    """0-100 from the trim band the position is in."""
    return {"50pct": 33.0, "100pct": 66.0, "200pct": 100.0, "none": 0.0}.get(trim_band, 0.0)


def _tech_exhaustion_subscore(exhaustion_score: Optional[float]) -> float:
    """ExhaustionDecision.score is 0.0-1.0; scale to 0-100."""
    if exhaustion_score is None:
        return 0.0
    return max(0.0, min(100.0, exhaustion_score * 100.0))


def _options_risk_subscore(
    short_call_delta: Optional[float],
    days_to_earnings: Optional[int],
    leap_dte_days: Optional[int],
) -> float:
    """0-100 for PMCC positions. Stocks always 0.

    Three signals:
      * Short-call delta in 0.65-0.75 band  -> 50
      * Short-call delta > 0.75             -> 100
      * Earnings within 5 days              -> +20
      * LEAP DTE under 90 days              -> +30 (gamma cliff)
    """
    if short_call_delta is None and leap_dte_days is None:
        return 0.0
    score = 0.0
    if short_call_delta is not None:
        if short_call_delta >= 0.75:
            score += 100.0
        elif short_call_delta >= 0.65:
            score += 50.0
        elif short_call_delta >= 0.50:
            score += 20.0
    if days_to_earnings is not None and 0 <= days_to_earnings <= 5:
        score += 20.0
    if leap_dte_days is not None and leap_dte_days < 90:
        score += 30.0
    return min(100.0, score)


def _rotation_subscore(score_delta: Optional[float]) -> float:
    """0-100 from the rotation candidate's score advantage.
    Delta of 10 pts = 50 score; 25+ pts = 100 score."""
    if score_delta is None:
        return 0.0
    if score_delta <= 0:
        return 0.0
    return max(0.0, min(100.0, (score_delta / 25.0) * 100.0))


def compute_exit_pressure(
    *,
    # Theme inputs
    theme_composite: Optional[float] = None,
    theme_streak_days: int = 0,
    # Profit preservation
    trim_band: str = "none",
    # Technical exhaustion
    exhaustion_score: Optional[float] = None,
    # Options risk (PMCC only)
    short_call_delta: Optional[float] = None,
    days_to_earnings: Optional[int] = None,
    leap_dte_days: Optional[int] = None,
    # Rotation
    rotation_score_delta: Optional[float] = None,
) -> ExitPressure:
    """Compute the unified Exit Pressure Score and the recommended band."""
    sub = {
        "theme_deterioration": _theme_deterioration_subscore(theme_composite, theme_streak_days),
        "profit_preservation": _profit_preservation_subscore(trim_band),
        "tech_exhaustion":     _tech_exhaustion_subscore(exhaustion_score),
        "options_risk":        _options_risk_subscore(short_call_delta, days_to_earnings, leap_dte_days),
        "rotation_pressure":   _rotation_subscore(rotation_score_delta),
    }
    weights = {
        "theme_deterioration": WEIGHT_THEME_DETERIORATION,
        "profit_preservation": WEIGHT_PROFIT_PRESERVATION,
        "tech_exhaustion":     WEIGHT_TECH_EXHAUSTION,
        "options_risk":        WEIGHT_OPTIONS_RISK,
        "rotation_pressure":   WEIGHT_ROTATION_PRESSURE,
    }
    score = sum(sub[k] * weights[k] for k in sub)
    score = round(score, 1)

    if score < BAND_HOLD_MAX:
        band = "hold"
    elif score < BAND_TRIM_LIGHT_MAX:
        band = "trim_light"
    elif score < BAND_TRIM_HEAVY_MAX:
        band = "trim_heavy"
    else:
        band = "aggressive"

    # Compact rationale showing the top contributors.
    contribs = [
        (k, sub[k] * weights[k]) for k in sub
    ]
    contribs.sort(key=lambda kv: kv[1], reverse=True)
    top = [f"{k.replace('_',' ')} {sub[k]:.0f}" for k, _ in contribs[:3]
           if sub[k] > 0]
    rationale = (
        f"Exit pressure {score:.0f}/100 -> {band}"
        + (": " + ", ".join(top) if top else "")
    )

    return ExitPressure(
        score=score, band=band,
        sub_scores=sub, weights=weights,
        rationale=rationale,
        detail={
            "theme_composite": theme_composite,
            "theme_streak_days": theme_streak_days,
            "trim_band": trim_band,
            "exhaustion_score": exhaustion_score,
            "short_call_delta": short_call_delta,
            "days_to_earnings": days_to_earnings,
            "leap_dte_days": leap_dte_days,
            "rotation_score_delta": rotation_score_delta,
        },
    )
