"""Execution layer — turns a candidate combo into a filled IBKR order.

Modules:
  walking_limit — atomic combo via walking-limit algorithm
"""

from .walking_limit import (
    ExecutionConfig,
    ExecutionResult,
    submit_pmcc_combo,
    submit_single_leg_option,
)

__all__ = ["ExecutionConfig", "ExecutionResult", "submit_pmcc_combo", "submit_single_leg_option"]
