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
    # Mid-drift abandon threshold. If the combo mid has drifted by more
    # than this fraction of the *initial* mid since order construction,
    # abandon rather than chase. 0.05 = 5% drift trips the abandon — for
    # PMCC combos at $5 net debit, that's 25c (≈ several walk steps' worth
    # of slippage avoided). Set to None (or <=0) to disable.
    abandon_on_mid_drift_pct: Optional[float] = 0.05


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
    # the default 1¢ step can't walk to a fillable price within
    # ``cfg.timeout_sec``. For LEAP combos with 3-6%-of-mid half-spreads
    # (HPE / ANET / AEHR), we adapt:
    #   * effective cap = max(cfg.max_offset_pct_of_spread, 0.50) — walk
    #     up to half of half-spread above mid (mid + 25% of full spread,
    #     i.e. a quarter of the way to the ask). The previous override
    #     value of 0.75 paid 37.5% of full spread above mid — too close
    #     to the ask for combos with $0.20+ spreads, and unilaterally
    #     overrode tighter operator caps. 0.50 caps the override at the
    #     midpoint between mid and ask.
    #   * effective step ≈ 5% of half-spread, so we cover the full
    #     half-spread in ~20 steps without paying through cap
    # Backward compatible: tight combos use the configured cfg values.
    thin_combo = mid > 0 and (half_spread / mid) > 0.03
    if thin_combo:
        effective_cap_pct = max(cfg.max_offset_pct_of_spread, 0.50)
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
    initial_mid = mid          # frozen reference for drift abandon
    current_mid = mid          # updated each step from a fresh quote
    drift_threshold = cfg.abandon_on_mid_drift_pct

    def _hit_cap(lim: float, cap_ref: float) -> bool:
        return (direction > 0 and lim >= cap_ref) or (direction < 0 and lim <= cap_ref)

    while loop.time() < deadline:
        # Re-quote before each submit so the cap tracks the *current* mid,
        # not the stale mid we captured at order construction (which can be
        # 30-60s old after the LEAP+short leg-selection round-trips). This
        # is the single biggest slippage protection: if the underlying
        # moves while we walk, the cap moves with it within the drift
        # tolerance — and if mid drifted too far, we abandon instead of
        # paying through a stale anchor.
        if result.walk_steps > 0:
            try:
                rq = await ibkr.get_combo_quote(legs=legs, symbol=symbol)
                rq_mid = rq.get("mid")
                rq_ask = rq.get("ask")
                rq_bid = rq.get("bid")
                if (rq_mid is not None and rq_mid > 0
                        and rq_ask is not None and rq_bid is not None):
                    current_mid = rq_mid
                    # Drift check vs initial mid (BUY: mid moving up is bad
                    # for us; SELL: mid moving down is bad for us).
                    if drift_threshold is not None and drift_threshold > 0:
                        drift_pct = (current_mid - initial_mid) / initial_mid
                        adverse = (direction > 0 and drift_pct > drift_threshold) \
                            or (direction < 0 and drift_pct < -drift_threshold)
                        if adverse:
                            result.status = "abandoned"
                            result.error = (
                                f"mid drifted {drift_pct*100:+.2f}% from "
                                f"${initial_mid:.2f} → ${current_mid:.2f} "
                                f"(threshold {drift_threshold*100:.1f}%)"
                            )
                            result.audit.append({
                                "step": result.walk_steps + 1,
                                "action": "abandon_mid_drift",
                                "initial_mid": initial_mid, "current_mid": current_mid,
                                "drift_pct": round(drift_pct, 4),
                                "ts": datetime.now(timezone.utc).isoformat(),
                            })
                            break
                    # Recompute the cap from the fresh mid + fresh half-spread.
                    # Don't widen past the initial cap by more than the
                    # drift_threshold worth — keeps the cap from runaway
                    # tracking when mid races against us.
                    new_half = (rq_ask - current_mid) if direction > 0 else (current_mid - rq_bid)
                    if new_half > 0:
                        new_cap = round(current_mid + direction * effective_cap_pct * new_half, 2)
                        if direction > 0:
                            cap = min(new_cap, cap * (1 + (drift_threshold or 0)))
                        else:
                            cap = max(new_cap, cap * (1 - (drift_threshold or 0)))
                        # Cap-clamp the current limit so we don't sit
                        # above the freshly-computed ceiling.
                        if direction > 0 and limit > cap:
                            limit = cap
                        elif direction < 0 and limit < cap:
                            limit = cap
            except Exception as e:
                # Re-quote failure is non-fatal: log and walk with the
                # last known cap. Surface in audit so post-mortems can see.
                result.audit.append({
                    "step": result.walk_steps + 1, "action": "requote_failed",
                    "error": str(e),
                    "ts": datetime.now(timezone.utc).isoformat(),
                })

        if _hit_cap(limit, cap):
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
            "current_mid": current_mid, "cap": cap,
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
                "slippage_vs_initial_mid": round(
                    direction * (result.fill_price - initial_mid), 4,
                ),
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

        if _hit_cap(limit, cap):
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

    # Post-walk reconciliation. The walker tracks fills via the combo
    # order's status field, but two failure modes can leave us thinking
    # we abandoned when the broker actually filled:
    #
    #   1. Fill-during-cancel race: status polled "not filled", we send
    #      cancel, broker fills before cancel lands, we move on with the
    #      walker thinking nothing happened.
    #
    #   2. NonGuaranteed leg-by-leg fills: each leg can fill independently
    #      and the combo's aggregate "filled" field doesn't always reflect
    #      that — especially when the smart router routes the legs to
    #      different exchanges.
    #
    # Either way the legs end up in the IBKR account while our DB says
    # the intent abandoned — exactly the state we need to avoid before
    # going live. Sync the truth: check the leg conids against current
    # positions; if both are there, override status to "filled".
    if result.status in ("abandoned", "error"):
        try:
            reconciled = await _reconcile_legs_with_positions(
                ibkr=ibkr, legs=legs, contracts=contracts, action=action,
            )
            if reconciled is not None:
                result.status = "filled"
                result.fill_price = reconciled["net_price"]
                result.error = None
                result.audit.append({
                    "step": result.walk_steps + 1,
                    "action": "reconciled_filled_post_walk",
                    "net_price": reconciled["net_price"],
                    "leg_costs": reconciled["leg_costs"],
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "note": "walker thought abandoned; positions show filled",
                })
                logger.warning(
                    "walker reconcile: combo for %s actually filled at %.2f "
                    "(walker had marked %s). Position truth wins.",
                    symbol, reconciled["net_price"], "abandoned" if result.error is None else "error",
                )
        except Exception as e:
            logger.warning("walker post-walk reconcile failed for %s: %s", symbol, e)
            result.audit.append({
                "step": result.walk_steps + 1,
                "action": "reconcile_error",
                "error": str(e),
                "ts": datetime.now(timezone.utc).isoformat(),
            })

    result.finished_at = datetime.now(timezone.utc)
    result.elapsed_sec = loop.time() - t0
    return result


async def _reconcile_legs_with_positions(
    *, ibkr: Any, legs: list[dict[str, Any]], contracts: int, action: str,
) -> Optional[dict[str, Any]]:
    """Check if every leg conid is present in IBKR positions at the expected qty.

    Returns ``{"net_price": float, "leg_costs": {conid: avg_cost}}`` if
    every leg is found with at least ``contracts`` qty (signed by the
    leg's action). Returns None if any leg is missing — meaning we should
    trust the walker's "not filled" verdict.

    Sign convention:
      * combo action=BUY + leg action=BUY  → expect qty >= +contracts
      * combo action=BUY + leg action=SELL → expect qty <= -contracts
      * combo action=SELL + leg action=BUY → expect qty <= -contracts (closing)
      * combo action=SELL + leg action=SELL → expect qty >= +contracts (closing)
    Combo SELL inverts leg directions because it's the unwind path.
    """
    try:
        ib = await ibkr._ensure_connected()
    except Exception:
        return None
    # ib.positions() is a sync method on the IB instance.
    try:
        raw = ib.positions()
    except Exception:
        return None
    if not raw:
        return None
    by_conid: dict[int, dict[str, Any]] = {}
    for p in raw:
        cid = int(getattr(p.contract, "conId", 0) or 0)
        if not cid:
            continue
        by_conid[cid] = {
            "qty": float(getattr(p, "position", 0) or 0),
            "avg_cost": float(getattr(p, "avgCost", 0) or 0),
            "contract": p.contract,
        }

    invert = action.upper() == "SELL"
    leg_costs: dict[int, float] = {}
    # Net price = sum over legs of (signed avg_cost) divided by multiplier.
    # For options the avgCost is premium * 100 (the multiplier); we want
    # per-share/per-contract premium for the net price.
    signed_premium_sum = 0.0
    for leg in legs:
        conid = int(leg["conid"])
        leg_action = str(leg["action"]).upper()
        ratio = int(leg.get("ratio", 1))
        pos = by_conid.get(conid)
        if pos is None:
            return None
        qty = pos["qty"]
        expected_sign = (+1 if leg_action == "BUY" else -1) * (-1 if invert else +1)
        expected_qty = expected_sign * ratio * contracts
        # Be tolerant of partial-fill accumulation from earlier walks —
        # we just need enough qty in the right direction. ARM example:
        # leg BUY at +1, we accept qty >= +1.
        if expected_sign > 0 and qty < expected_qty:
            return None
        if expected_sign < 0 and qty > expected_qty:
            return None
        leg_costs[conid] = pos["avg_cost"]
        # Premium contribution: BUY leg adds cost (we paid), SELL leg
        # subtracts (we received). avgCost is total cost basis per
        # contract for options (already × multiplier of 100).
        premium_per_contract = pos["avg_cost"]
        if leg_action == "BUY":
            signed_premium_sum += premium_per_contract
        else:
            signed_premium_sum -= premium_per_contract
    # Convert total cost-basis difference to per-share net price (÷ 100 for options).
    net_price = round(signed_premium_sum / 100.0, 2)
    if invert:
        net_price = -net_price   # SELL combos report as credit (positive)
    return {"net_price": net_price, "leg_costs": leg_costs}


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


async def submit_single_leg_option(
    *,
    ibkr: Any,                       # IbkrProvider
    conid: int,
    contracts: int,
    action: str = "BUY",
    config: Optional[ExecutionConfig] = None,
    fair_value_ceiling: Optional[float] = None,   # don't pay above this (BUY)
) -> ExecutionResult:
    """Walk a limit on a SINGLE option leg (used by the LEAPS-only strategy to
    buy a long-dated call outright — no combo, no short leg).

    Mirrors ``submit_pmcc_combo``'s discipline at single-leg scale: start near
    mid, walk toward the ask capped at mid + a bounded fraction of the
    half-spread, re-quote each step so the cap tracks the live mid, abandon on
    adverse mid drift or timeout. Quote comes from
    ``ibkr.get_option_quote_by_conid`` (bid/ask/mid/model_price fallback)."""
    cfg = config or ExecutionConfig()
    started = datetime.now(timezone.utc)
    result = ExecutionResult(status="error", started_at=started)
    loop = asyncio.get_running_loop()
    t0 = loop.time()

    async def _quote() -> tuple[Optional[float], Optional[float], Optional[float]]:
        q = await ibkr.get_option_quote_by_conid(conid=int(conid))
        b, a, m = q.get("bid"), q.get("ask"), q.get("mid")
        if m is None or m <= 0:
            mp = q.get("model_price")
            if mp and mp > 0:
                m = mp
                b = b if (b and b > 0) else mp
                a = a if (a and a > 0) else mp
        return b, a, m

    bid, ask, mid = await _quote()
    if not mid or mid <= 0:
        result.status = "rejected_pretrade"
        result.error = f"no quote for conid={conid}"
        result.finished_at = datetime.now(timezone.utc)
        result.elapsed_sec = loop.time() - t0
        return result

    direction = 1 if action.upper() == "BUY" else -1
    half = (ask - mid) if (ask and ask > mid) else max(mid * 0.03, 0.05)
    cap_pct = cfg.max_offset_pct_of_spread
    if mid > 0 and (half / mid) > 0.03:          # thin option — allow a wider cap
        cap_pct = max(cap_pct, 0.50)
    cap = round(mid + direction * cap_pct * half, 2)
    if direction > 0 and fair_value_ceiling is not None:
        cap = min(cap, fair_value_ceiling)
    result.bid_at_submit, result.ask_at_submit, result.mid_at_submit, result.cap_price = bid, ask, mid, cap

    from ib_insync import Contract, LimitOrder  # type: ignore
    ib = await ibkr._ensure_connected()
    qc = await ib.qualifyContractsAsync(Contract(conId=int(conid), exchange="SMART", currency="USD"))
    if not qc:
        result.status = "error"
        result.error = f"qualify failed for conid={conid}"
        result.finished_at = datetime.now(timezone.utc)
        result.elapsed_sec = loop.time() - t0
        return result
    contract = qc[0]

    def _hit_cap(lim: float) -> bool:
        return (direction > 0 and lim >= cap) or (direction < 0 and lim <= cap)

    limit = round(mid + direction * cfg.initial_offset_cents / 100.0, 2)
    if direction > 0:
        limit = min(limit, cap)
    else:
        limit = max(limit, cap)
    deadline = t0 + cfg.timeout_sec
    initial_mid = mid
    drift = cfg.abandon_on_mid_drift_pct
    trade: Any = None

    while loop.time() < deadline:
        if result.walk_steps > 0:
            b, a, m = await _quote()
            if m and m > 0:
                if drift and drift > 0:
                    dp = (m - initial_mid) / initial_mid
                    if (direction > 0 and dp > drift) or (direction < 0 and dp < -drift):
                        await _cancel_trade(trade)
                        result.status = "abandoned"
                        result.error = f"mid drift {dp*100:+.2f}% (${initial_mid:.2f}->${m:.2f})"
                        break
                nh = (a - m) if (a and a > m) else half
                if nh > 0:
                    newcap = round(m + direction * cap_pct * nh, 2)
                    cap = min(newcap, round(cap * (1 + (drift or 0)), 2)) if direction > 0 \
                        else max(newcap, round(cap * (1 - (drift or 0)), 2))
                    if (direction > 0 and limit > cap) or (direction < 0 and limit < cap):
                        limit = cap

        if trade is None:
            trade = ib.placeOrder(contract, LimitOrder(action, contracts, limit, tif="DAY"))
        else:
            trade.order.lmtPrice = limit
            ib.placeOrder(contract, trade.order)
        result.walk_steps += 1
        result.submitted_price = limit
        result.audit.append({"step": result.walk_steps, "limit": limit, "cap": cap,
                              "ts": datetime.now(timezone.utc).isoformat()})

        await asyncio.sleep(cfg.walk_interval_sec)
        st = await _poll_order_status(trade)
        if st.get("status") == "Filled" or (st.get("filled_qty") and st.get("remaining") == 0):
            result.status = "filled"
            result.fill_price = st.get("avg_fill_price") or limit
            result.order_id = getattr(getattr(trade, "order", None), "orderId", None)
            break
        if _hit_cap(limit):
            # At the cap and still unfilled — give it one more interval, then abandon.
            await asyncio.sleep(cfg.walk_interval_sec)
            st = await _poll_order_status(trade)
            if st.get("status") == "Filled":
                result.status = "filled"
                result.fill_price = st.get("avg_fill_price") or limit
                result.order_id = getattr(getattr(trade, "order", None), "orderId", None)
            else:
                await _cancel_trade(trade)
                result.status = "abandoned"
                result.error = f"unfilled at cap ${cap}"
            break
        limit = round(limit + direction * max(cfg.walk_increment_cents, 1) / 100.0, 2)
        if direction > 0:
            limit = min(limit, cap)
        else:
            limit = max(limit, cap)

    if result.status not in ("filled", "abandoned", "rejected_pretrade"):
        await _cancel_trade(trade)
        result.status = "abandoned"
        result.error = result.error or "timeout"
    result.finished_at = datetime.now(timezone.utc)
    result.elapsed_sec = loop.time() - t0
    return result
