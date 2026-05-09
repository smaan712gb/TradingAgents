"""Strategy layer — turns agent decisions into broker-ready intents.

Modules:
  pmcc      — Poor man's covered call eligibility + leg selection
  execution — Walking-limit combo executor + timing
"""

from .pmcc import (
    PmccCandidate,
    PmccEligibility,
    select_pmcc_legs,
)

__all__ = [
    "PmccCandidate",
    "PmccEligibility",
    "select_pmcc_legs",
]
