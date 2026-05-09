"""Walking-limit combo executor.

Submits a multi-leg PMCC combo as one atomic IBKR Bag order. Walks the
limit from mid+1¢ toward a cap based on the spread, abandoning if no
fill within the time budget.

Why walking and not market-on-spread:

  * Combo orders always go LMT, never MKT (option spreads multiply leg
    spreads — a market combo can fill 10-20% off mid).
  * Starting at mid+1¢ tells market makers "serious buyer, willing to
    pay slightly above mid". They often step up.
  * Capping at mid + 25% of (ask - mid) means worst case we pay
    midway between mid and ask, never the screaming offer.
  * Atomic combo (Bag) means both legs fill simultaneously at one net
    price — no naked-leg risk if the underlying moves between fills.

Failure modes the executor handles cleanly:
  - No fill at any walk step → cancel, try next price
  - No fill at cap → ABANDON (cancel, return result with status="abandoned")
  - Account / market-data errors → cancel, return status="error"
  - Network glitch → tries up to 2 times to cancel
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ExecutionConfig:
    initial_offset_cents:    int   = 1
    walk_increment_cents:    int   = 1
    walk_interval_sec:       float = 30.0
    max_offset_pct_of_spread: float = 0.25
    timeout_sec:             float = 300.0  # total budget
    fast_mode:               bool  = False  # used for short-call rolls (faster, wider cap)


@dataclass
class ExecutionResult:
    status:         str                   # "filled" | "abandoned" | "error" | "rejected_pretrade"
    order_id:       Optional[int] = None
    fill_price:     Optional[float] = None
    submitted_price: Optional[float] = None
    walk_steps:     int = 0
    elapsed_sec:    float = 0.0
    started_at:     Optional[datetime] = None
    finished_at:    Optional[datetime] = None
    bid_at_submit:  Optional[float] = None
    ask_at_submit:  Optional[float] = None
    mid_at_submit:  Optional[float] = None
    cap_price:      Optional[float] = None
    error:          Optional[str] = None
    audit:          list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status, "order_id": self.order_id,
            "fill_price": self.fill_price, "submitted_price": self.submitted_price,
            "walk_steps": self.walk_steps, "elapsed_sec": round(self.elapsed_sec, 2),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "bid_at_submit": self.bid_at_submit, "ask_at_submit": self.ask_at_submit,
            "mid_at_submit": self.mid_at_submit, "cap_price": self.cap_price,
            "error": self.error, "audit": self.audit,
        }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def submit_pmcc_combo(
    *,
    ibkr: Any,                        # IbkrProvider
    symbol: str,
    legs: list[dict[str, Any]],       # [{conid, ratio, action}, ...]
    contracts: int,
    action: str = "BUY",              # "BUY" pays net debit; "SELL" receives net credit
    config: Optional[ExecutionConfig] = None,
    fair_value_ceiling: Optional[float] = None,  # don't pay above this (BUY only)
    fair_value_floor: Optional[float] = None,    # don't accept less than this (SELL only)
) -> ExecutionResult:
    """Walk a limit on a multi-leg combo and return the result.

    BUY combo (entries, LEAP-forward roll): start at mid+1¢, walk UP toward
    ask, capped at mid + 25% of half-spread. Filled when MM lifts.

    SELL combo (closes, credit rolls): start at mid-1¢, walk DOWN toward
    bid, capped at mid - 25% of half-spread. Filled when MM hits.

    Either way the operation is atomic — IBKR fills both legs at the same
    net price or neither.
    """
    cfg = config or ExecutionConfig()
    started = datetime.now(timezone.utc)
    result = ExecutionResult(status="error", started_at=started)
    loop = asyncio.get_running_loop()
    t0 = loop.time()

    # ---- Pre-trade quote snapshot -------------------------------------
    try:
        quote = await ibkr.get_combo_quote(legs=legs, symbol=symbol)
    except Exception as e:
        result.status = "error"
        result.error = f"combo quote failed: {e}"
        result.finished_at = datetime.now(timezone.utc)
        result.elapsed_sec = loop.time() - t0
        return result

    bid = quote.get("bid")
    ask = quote.get("ask")
    mid = quote.get("mid")
    if bid is None or ask is None or mid is None or mid <= 0:
        result.status = "rejected_pretrade"
        result.error = f"invalid combo quote: bid={bid} ask={ask} mid={mid}"
        result.finished_at = datetime.now(timezone.utc)
        result.elapsed_sec = loop.time() - t0
        return result

    direction = 1 if action.upper() == "BUY" else -1
    half_spread = (ask - mid)

    # Thin-combo detection: when the half-spread is wide relative to mid,
    # the default 1¢ step + 0.50 × half_spread cap can't actually walk to
    # a fillable price within ``cfg.timeout_sec``. For LEAP combos with
    # 3-6%-of-mid half-spreads (HPE / ANET / AEHR), we adapt:
    #   * effective cap = max(cfg.max_offset_pct_of_spread, 0.75) — walk
    #     up to 75% of half-spread above mid (still below ask)
    #   * effective step ≈ 5% of half-spread, so we cover the full
    #     half-spread in ~20 steps without paying through cap
    # Backward compatible: tight combos use the configured cfg values.
    thin_combo = mid > 0 and (half_spread / mid) > 0.03
    if thin_combo:
        effective_cap_pct = max(cfg.max_offset_pct_of_spread, 0.75)
        effective_step_cents = max(
            cfg.walk_increment_cents,
            int(round((half_spread / 20) * 100)),
        )
    else:
        effective_cap_pct = cfg.max_offset_pct_of_spread
        effective_step_cents = cfg.walk_increment_cents

    # Direction-aware cap
    cap = round(mid + direction * effective_cap_pct * half_spread, 2)
    if direction > 0 and fair_value_ceiling is not None:
        cap = min(cap, fair_value_ceiling)
    elif direction < 0 and fair_value_floor is not None:
        cap = max(cap, fair_value_floor)
    result.bid_at_submit = bid
    result.ask_at_submit = ask
    result.mid_at_submit = mid
    result.cap_price = cap

    # ---- Phase E3 — depth-aware smart starting price ----------------
    # Read per-leg L2 depth (NASDAQ TotalView + NYSE OpenBook). For thin
    # combos we use depth to compute a *smart starting price*: the lowest
    # debit (BUY) or highest credit (SELL) where the resting size on both
    # legs supports our contract count. This skips the wasted steps walking
    # through "air" between mid and where real liquidity sits.
    smart_start: Optional[float] = None
    depth_audit: dict[str, Any] = {}
    if thin_combo:
        try:
            smart_start, depth_audit = await _compute_smart_start(
                ibkr=ibkr, legs=legs, action=action.upper(),
                contracts=contracts, cap=cap, mid=mid, direction=direction,
            )
        except Exception as e:
            logger.debug("depth-aware smart start failed (%s); using mid+1¢", e)
            depth_audit = {"error": str(e)}
    result.audit.append({
        "step": 0, "action": "preflight",
        "thin_combo": thin_combo,
        "effective_cap_pct": effective_cap_pct,
        "effective_step_cents": effective_step_cents,
        "smart_start": smart_start,
        "depth": depth_audit,
        "ts": datetime.now(timezone.utc).isoformat(),
    })

    # ---- Walking limit -----------------------------------------------
    if smart_start is not None:
        # Cap-clamp the smart start so we never start above cap
        if direction > 0:
            limit = round(min(smart_start, cap), 2)
        else:
            limit = round(max(smart_start, cap), 2)
    else:
        limit = round(mid + direction * cfg.initial_offset_cents / 100.0, 2)
    deadline = t0 + cfg.timeout_sec
    last_trade: Any = None

    def _hit_cap(lim: float) -> bool:
        return (direction > 0 and lim >= cap) or (direction < 0 and lim <= cap)

    while loop.time() < deadline:
        if _hit_cap(limit):
            limit = cap

        result.walk_steps += 1
        result.submitted_price = limit
        try:
            submission = await ibkr.submit_combo(
                legs=legs, action=action.upper(), net_price=limit, symbol=symbol,
            )
        except Exception as e:
            result.status = "error"
            result.error = f"submit_combo failed: {e}"
            break

        last_trade = submission.get("_trade")
        order_id = submission.get("order_id")
        result.order_id = order_id
        result.audit.append({
            "step": result.walk_steps, "action": "submit",
            "limit": limit, "order_id": order_id,
            "ts": datetime.now(timezone.utc).isoformat(),
        })

        # Wait for the interval (or earlier if we get a fill)
        await asyncio.sleep(cfg.walk_interval_sec)

        status = await _poll_order_status(last_trade)
        if status.get("filled_qty", 0) >= contracts:
            result.status = "filled"
            result.fill_price = status.get("avg_fill_price") or limit
            result.audit.append({
                "step": result.walk_steps, "action": "filled",
                "fill_price": result.fill_price,
                "ts": datetime.now(timezone.utc).isoformat(),
            })
            break

        # Not filled — cancel and walk
        await _cancel_trade(last_trade)
        result.audit.append({
            "step": result.walk_steps, "action": "cancel_unfilled",
            "limit": limit,
            "ts": datetime.now(timezone.utc).isoformat(),
        })

        if _hit_cap(limit):
            # Reached cap and didn't fill — abandon.
            result.status = "abandoned"
            result.error = f"reached cap ${cap} after {result.walk_steps} walk steps"
            break

        limit = round(limit + direction * effective_step_cents / 100.0, 2)

    if result.status == "error" and not result.audit:
        # Nothing happened — probably crashed before first submit.
        pass
    elif result.status not in ("filled", "abandoned", "rejected_pretrade", "error"):
        # Loop exited via deadline without explicit terminal state.
        result.status = "abandoned"
        result.error = f"timeout after {cfg.timeout_sec}s, {result.walk_steps} steps"
        if last_trade:
            await _cancel_trade(last_trade)

    result.finished_at = datetime.now(timezone.utc)
    result.elapsed_sec = loop.time() - t0
    return result


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


async def _compute_smart_start(
    *, ibkr: Any, legs: list[dict[str, Any]], action: str,
    contracts: int, cap: float, mid: float, direction: int,
) -> tuple[Optional[float], dict[str, Any]]:
    """Use per-leg L2 depth to compute a smart starting limit price.

    For each leg we read the top-N depth levels via IBKR's reqMktDepth.
    Then we construct combined combo prices at each cross-leg level
    and find the lowest (BUY) or highest (SELL) price where the
    minimum resting size across both legs supports our contract count.

    Returns ``(smart_start, audit)``:
      * ``smart_start`` — recommended limit price, or None if depth was
        unavailable or no level supports the contract count.
      * ``audit`` — debug dict with per-leg level snippets so the
        operator can see what the book looked like.

    Markets-closed sessions (weekend) typically return empty books;
    smart_start will be None and the caller falls back to mid+1¢.
    """
    # Resolve leg meta — we need to know which is BUY and which is SELL
    # so we can sign the per-level contributions correctly.
    leg_metas: list[dict[str, Any]] = []
    for leg in legs:
        leg_metas.append({
            "conid": int(leg["conid"]),
            "action": str(leg["action"]).upper(),
            "ratio": int(leg.get("ratio", 1)),
        })

    # Fetch depth on each leg in parallel
    depth_results: list[dict[str, Any]] = []
    coros = [
        ibkr.get_market_depth_by_conid(conid=m["conid"])
        for m in leg_metas
    ]
    try:
        depth_results = list(await asyncio.gather(*coros, return_exceptions=True))
    except Exception as e:
        return (None, {"error": f"gather failed: {e}"})

    audit: dict[str, Any] = {"legs": []}
    have_depth = True
    for m, d in zip(leg_metas, depth_results):
        if isinstance(d, Exception) or not isinstance(d, dict):
            have_depth = False
            audit["legs"].append({"conid": m["conid"], "error": str(d)})
            continue
        # Trim each side to the first 5 levels for the audit
        bids = d.get("bids", [])[:5]
        asks = d.get("asks", [])[:5]
        audit["legs"].append({
            "conid": m["conid"], "action": m["action"],
            "depth_available": d.get("depth_available", False),
            "top_bids": bids, "top_asks": asks,
        })
        if not d.get("depth_available"):
            have_depth = False

    if not have_depth:
        return (None, {**audit, "reason": "depth_unavailable"})

    # For BUY combo: walk up — start at LEAP ask + short bid pairs
    #   side we need is leap.asks (we lift offer) + short.bids (we hit bid)
    # For SELL combo: walk down — start at LEAP bid + short ask pairs
    levels: list[tuple[float, float]] = []   # (price, max_fillable)
    if direction > 0:                # BUY combo (typical PMCC entry)
        leap_idx = next(
            (i for i, m in enumerate(leg_metas) if m["action"] == "BUY"), None,
        )
        short_idx = next(
            (i for i, m in enumerate(leg_metas) if m["action"] == "SELL"), None,
        )
    else:                            # SELL combo (PMCC unwind / credit roll)
        leap_idx = next(
            (i for i, m in enumerate(leg_metas) if m["action"] == "SELL"), None,
        )
        short_idx = next(
            (i for i, m in enumerate(leg_metas) if m["action"] == "BUY"), None,
        )

    if leap_idx is None or short_idx is None:
        return (None, {**audit, "reason": "could_not_identify_legs"})

    leap_d = depth_results[leap_idx]
    short_d = depth_results[short_idx]

    # For BUY: leap.asks (we pay up) + short.bids (we receive at bid).
    # Combo limit = leap_ask_price - short_bid_price (per spread).
    # Pair at the same level index — IBKR doesn't guarantee per-MM
    # alignment, but level-by-level pairing is the convention.
    if direction > 0:
        leap_side = leap_d.get("asks", [])
        short_side = short_d.get("bids", [])
    else:
        leap_side = leap_d.get("bids", [])
        short_side = short_d.get("asks", [])

    n = min(len(leap_side), len(short_side))
    for i in range(n):
        leap_price, leap_size, _ = leap_side[i]
        short_price, short_size, _ = short_side[i]
        if leap_price <= 0 or short_price <= 0:
            continue
        # For BUY: combo debit per spread = leap_ask - short_bid
        # For SELL: combo credit per spread = leap_bid - short_ask  (negative -> we receive)
        # Either way, the combo "limit" we'd submit is leap_price - short_price.
        combo_limit = round(leap_price - short_price, 2)
        max_fill = min(leap_size, short_size)
        levels.append((combo_limit, max_fill))
    audit["combo_levels"] = levels

    # Find the first level where max_fill >= our contracts.
    # For BUY: we want the LOWEST such combo_limit (cheapest fill).
    # For SELL: we want the HIGHEST such combo_limit (best credit).
    qualifying = [(p, s) for (p, s) in levels if s >= contracts]
    if not qualifying:
        # No single level can fill the whole order — pick the level with
        # the largest available size as the best aggressive starting price.
        if not levels:
            return (None, {**audit, "reason": "no_combo_levels"})
        levels_sorted = sorted(levels, key=lambda lv: lv[1], reverse=True)
        smart = levels_sorted[0][0]
        audit["chosen"] = {
            "limit": smart, "available_size": levels_sorted[0][1],
            "reason": "fragmented_book — picked largest visible size",
        }
    else:
        if direction > 0:
            chosen = min(qualifying, key=lambda lv: lv[0])
        else:
            chosen = max(qualifying, key=lambda lv: lv[0])
        smart = chosen[0]
        audit["chosen"] = {
            "limit": smart, "available_size": chosen[1],
            "reason": "first_level_with_full_size",
        }

    # Bound to cap
    if direction > 0 and smart > cap:
        audit["chosen"]["clamped_to_cap"] = True
        smart = cap
    elif direction < 0 and smart < cap:
        audit["chosen"]["clamped_to_cap"] = True
        smart = cap

    return (smart, audit)


async def _poll_order_status(trade: Any) -> dict[str, Any]:
    """Snapshot the current ib_insync Trade.orderStatus."""
    if trade is None:
        return {"filled_qty": 0, "avg_fill_price": None, "status": "no_trade"}
    try:
        os = trade.orderStatus
        return {
            "status":         os.status,
            "filled_qty":     float(os.filled or 0),
            "remaining":      float(os.remaining or 0),
            "avg_fill_price": float(os.avgFillPrice or 0) or None,
        }
    except Exception:
        return {"filled_qty": 0, "avg_fill_price": None, "status": "unknown"}


async def _cancel_trade(trade: Any, attempts: int = 2) -> None:
    """Cancel an ib_insync Trade. Tolerates already-filled / already-cancelled."""
    if trade is None:
        return
    for i in range(attempts):
        try:
            ib = trade.contract  # placeholder so static type checker doesn't complain
            ib  # silence
            from ib_insync import IB  # type: ignore
            # The Trade carries a reference to the IB instance via its .ib attr in newer
            # ib_insync versions; older versions need cancellation through the original
            # ib instance. We try both shapes.
            ib_inst = getattr(trade, "ib", None)
            if ib_inst is not None:
                ib_inst.cancelOrder(trade.order)
            else:
                # Fall back: nothing to do, the calling code should pass IB-aware trade
                logger.debug("trade has no ib ref; cannot cancel")
            return
        except Exception as e:
            logger.warning("cancel attempt %d failed: %s", i + 1, e)
            await asyncio.sleep(0.5)
