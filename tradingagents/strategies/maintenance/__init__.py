"""Post-entry position management — Loop 1.

Owns every position the system has opened:

  Stocks:
    * Exit on Avoid signal from latest theme run
    * ATR-based trailing stop (2.5×ATR_30d)
    * Time-stop (no movement in 30 days, low conviction)

  PMCC (LEAP + short call):
    * Roll short call when DTE ≤ 7 OTM (capture theta cycle)
    * Roll up-and-out when short call delta ≥ 0.50 (defensive)
    * Roll LEAP forward when DTE ≤ 270 (extend time)
    * Close PMCC if LEAP delta drops < 0.65 (thesis broken)
    * Earnings hedge: buy back short call 2 sessions before earnings,
      re-sell day after the print clears.

Loop runs every 5 min during RTH. Each action goes through the same
auto-gate stack as entries (kill switch, etc.) and is audited.
"""

from .rolls       import maybe_roll_short_call, maybe_roll_leap_forward
from .exits       import maybe_exit_stock, maybe_close_pmcc
from .earnings    import days_to_earnings, earnings_hedge_due

__all__ = [
    "maybe_roll_short_call",
    "maybe_roll_leap_forward",
    "maybe_exit_stock",
    "maybe_close_pmcc",
    "days_to_earnings",
    "earnings_hedge_due",
]
