"""Interactive Brokers (IBKR) execution provider — paper-only.

Uses `ib_insync` over the TWS/Gateway socket API. The agents NEVER call
this directly; the FastAPI router accepts a `TradeIntent` from the user
(after on-screen confirmation in the UI) and routes here. We expose this
as a Provider for symmetry with the data vendors and so backtests can
substitute a fake.

Hard rules in this file:

* `account_mode` must equal `"paper"`. We additionally check that the
  connected account ID starts with `D` (IBKR's paper-account prefix).
  Both gates must pass before any order is placed.
* `submit_trade` is the *only* method that places orders. There is no
  market-on-close, no bracket-from-string, no convenience helpers — keep
  the surface area tiny so live trading can never be "almost wired up".
* All calls log a structured audit line with the order intent and the
  resulting IBKR order ID. The FastAPI layer mirrors this to Postgres.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional

from .base import (
    AuthError,
    ExecutionProvider,
    LiveTradingDisabledError,
    ProviderError,
    TradeIntent,
)

logger = logging.getLogger(__name__)


def _safe_float(v: Any) -> Optional[float]:
    """ib_insync uses NaN to signal 'no quote'; collapse to None."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:        # NaN
        return None
    return f


def _safe_int(v: Any) -> Optional[int]:
    f = _safe_float(v)
    return int(f) if f is not None else None


class IbkrProvider:
    name = "ibkr"

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        client_id: Optional[int] = None,
        account_mode: Optional[str] = None,
    ) -> None:
        self.host = host or os.getenv("IBKR_HOST", "127.0.0.1")
        self.port = int(port or os.getenv("IBKR_PORT", "7497"))   # 7497 = TWS paper
        self.client_id = int(client_id or os.getenv("IBKR_CLIENT_ID", "11"))
        self.account_mode = (account_mode or os.getenv("IBKR_MODE", "paper")).lower()
        if self.account_mode != "paper":
            raise LiveTradingDisabledError()
        try:
            from ib_insync import IB  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ProviderError("ibkr", "ib_insync not installed", retryable=False) from e
        self._ib_cls = IB
        self._ib: Any = None
        self._connect_lock = asyncio.Lock()
        self._reconnect_attempts = 0

    # ------------------------------------------------------------------
    # Connection lifecycle — single instance, auto-reconnect, listener-driven
    # ------------------------------------------------------------------

    async def _ensure_connected(self) -> Any:
        """Return a connected IB instance. Reconnects if the socket dropped.

        Uses a lock so concurrent callers don't race two parallel connects.
        Single IB instance is reused — disconnects clear it via the
        `disconnectedEvent` listener so the next call sees `None` and
        reconnects fresh.
        """
        if self._ib is not None and self._ib.isConnected():
            return self._ib

        async with self._connect_lock:
            # Double-checked: another coroutine may have reconnected while we waited.
            if self._ib is not None and self._ib.isConnected():
                return self._ib

            # Clean up any stale instance before starting a new one. ib_insync
            # leaks reqIds and event handlers if you re-use a disconnected
            # instance, so build a fresh IB() each time.
            if self._ib is not None:
                try:
                    self._ib.disconnect()
                except Exception:
                    pass
                self._ib = None

            backoff = min(1.0 * (2 ** min(self._reconnect_attempts, 5)), 30.0)
            if self._reconnect_attempts > 0:
                logger.info("ibkr: reconnect attempt %d (backoff %.1fs)",
                            self._reconnect_attempts, backoff)
                await asyncio.sleep(backoff)

            ib = self._ib_cls()
            try:
                await ib.connectAsync(
                    self.host, self.port, clientId=self.client_id, timeout=10,
                )
            except Exception as e:
                self._reconnect_attempts += 1
                raise ProviderError(
                    "ibkr",
                    f"connect failed (host={self.host}:{self.port} cid={self.client_id}): {e}",
                    retryable=True,
                )

            accounts = ib.managedAccounts()
            if not accounts:
                try: ib.disconnect()
                except Exception: pass
                self._reconnect_attempts += 1
                raise AuthError("ibkr")
            acct = accounts[0]
            if not acct.startswith("D"):
                try: ib.disconnect()
                except Exception: pass
                raise LiveTradingDisabledError()

            # Wire a disconnect listener so the next call notices and reconnects.
            try:
                ib.disconnectedEvent += self._on_disconnect
            except Exception:
                pass

            self._ib = ib
            self._reconnect_attempts = 0
            logger.info("ibkr: connected paper account=%s (clientId=%d)",
                        acct, self.client_id)
            return ib

    def _on_disconnect(self) -> None:
        """ib_insync fires this when the socket drops. Reset state so the
        next ``_ensure_connected`` call rebuilds the connection."""
        logger.warning("ibkr: disconnect detected, will reconnect on next call")
        self._ib = None

    async def aclose(self) -> None:
        if self._ib is not None:
            try:
                self._ib.disconnect()
            except Exception:
                pass
        self._ib = None

    def is_connected(self) -> bool:
        """Best-effort check used by /api/health — never raises."""
        try:
            return self._ib is not None and self._ib.isConnected()
        except Exception:
            return False

    # ------------------------------------------------------------------
    # ExecutionProvider
    # ------------------------------------------------------------------

    async def get_account_summary(self) -> dict[str, Any]:
        ib = await self._ensure_connected()
        rows = await ib.accountSummaryAsync()
        return {r.tag: r.value for r in rows}

    async def get_positions(self) -> list[dict[str, Any]]:
        """Return current account positions with LIVE market data.

        First call after a restart: subscribes ``reqMktData`` for each
        contract and waits 3s for ticks to populate.

        Subsequent calls: reads from the live, persistent subscriptions —
        no re-subscribe, no wait. The Ticker objects update continuously
        as IBKR pushes ticks over the socket.

        New positions auto-subscribe on first sighting; closed positions
        get unsubscribed and dropped.
        """
        ib = await self._ensure_connected()
        positions = ib.positions()
        if not hasattr(self, "_pos_subs"):
            self._pos_subs = {}   # conId -> ticker

        # Add subscriptions for any newly-seen contract.
        new_subs = 0
        for p in positions:
            cid = getattr(p.contract, "conId", 0)
            if not cid:
                continue
            if cid in self._pos_subs:
                continue
            try:
                ticker = ib.reqMktData(p.contract, "", False, False)
                self._pos_subs[cid] = (p.contract, ticker)
                new_subs += 1
            except Exception as e:
                logger.warning("reqMktData failed for %s: %s", p.contract.symbol, e)

        # If we just subscribed new contracts, give the socket time to
        # deliver the first tick before reading. Otherwise, no wait —
        # the Ticker fields are live-streamed.
        if new_subs > 0:
            await asyncio.sleep(3.0)
            logger.info("positions: subscribed live data for %d new contracts", new_subs)

        # Build the output from current live ticker state.
        out: list[dict[str, Any]] = []
        active_conids: set[int] = set()
        for p in positions:
            cid = getattr(p.contract, "conId", 0)
            active_conids.add(cid)
            qty = float(p.position)
            avg = float(p.avgCost)
            last = avg
            sub = self._pos_subs.get(cid)
            if sub is not None:
                _contract, ticker = sub
                try:
                    cand = []
                    if hasattr(ticker, "marketPrice"):
                        try: cand.append(ticker.marketPrice())
                        except Exception: pass
                    cand.append(getattr(ticker, "last", None))
                    cand.append(getattr(ticker, "close", None))
                    cand.append(getattr(ticker, "bid", None))
                    for v in cand:
                        f = _safe_float(v)
                        if f is not None and f > 0:
                            last = f
                            break
                except Exception:
                    pass
            out.append({
                "account_id": p.account,
                "symbol": p.contract.symbol,
                "secType": p.contract.secType,
                "currency": p.contract.currency,
                "qty": qty,
                "avg_price": round(avg, 4),
                "last_price": round(last, 4),
                "pnl": round((last - avg) * qty, 2),
            })

        # Drop subscriptions for positions that closed since last call.
        stale = [cid for cid in self._pos_subs if cid not in active_conids]
        for cid in stale:
            contract, _ticker = self._pos_subs.pop(cid)
            try:
                ib.cancelMktData(contract)
            except Exception:
                pass
        if stale:
            logger.info("positions: dropped subs for %d closed positions", len(stale))

        return out

    async def get_option_chain(
        self,
        *,
        symbol: str,
        exchange: str = "SMART",
        currency: str = "USD",
    ) -> dict[str, Any]:
        """Return all listed expirations + strikes for ``symbol``.

        Wraps ``reqSecDefOptParams``. Returns a dict shaped like:
            {
              "trading_class": "NVDA",
              "underlying_conid": 76792991,
              "expirations": ["20260116", "20260220", ...],
              "strikes": [50.0, 55.0, ..., 500.0],
              "exchange": "SMART",
            }
        Caller filters by DTE / strike range when picking PMCC legs.
        """
        from ib_insync import Stock  # type: ignore
        ib = await self._ensure_connected()
        underlying = Stock(symbol, exchange, currency)
        qualified = await ib.qualifyContractsAsync(underlying)
        if not qualified:
            raise ProviderError("ibkr", f"could not qualify {symbol} as underlying")
        underlying = qualified[0]
        params = await ib.reqSecDefOptParamsAsync(
            underlyingSymbol=underlying.symbol,
            futFopExchange="",
            underlyingSecType=underlying.secType,
            underlyingConId=underlying.conId,
        )
        if not params:
            raise ProviderError("ibkr", f"no option params for {symbol}")
        # Prefer SMART when present, else first row.
        choice = next((p for p in params if p.exchange == "SMART"), params[0])
        return {
            "underlying_symbol": underlying.symbol,
            "underlying_conid": underlying.conId,
            "trading_class": choice.tradingClass,
            "exchange": choice.exchange,
            "expirations": sorted(list(choice.expirations)),
            "strikes": sorted([float(s) for s in choice.strikes]),
        }

    async def get_option_quote(
        self,
        *,
        symbol: str,
        expiry: str,            # YYYYMMDD
        strike: float,
        right: str,             # "C" or "P"
        exchange: str = "SMART",
        currency: str = "USD",
        snapshot: bool = False,  # IBKR rejects snapshot=True when generic ticks include 106
        timeout_s: float = 3.0,
    ) -> dict[str, Any]:
        """Snapshot quote + Greeks for one option contract.

        Uses ``reqMktData`` with generic tick list ``"100,101,106"``:
            100 — option volume
            101 — option open interest
            106 — option implied volatility / model Greeks (Δ Γ Vega Θ)

        Returns dict with bid/ask/last/iv/delta/gamma/theta/vega/oi.
        Failure to populate within ``timeout_s`` returns Nones for the
        missing fields; caller decides how strict to be.
        """
        from ib_insync import Option  # type: ignore
        ib = await self._ensure_connected()
        opt = Option(symbol, expiry, float(strike), right, exchange, "100", currency)
        # Trading class often "NVDA" rather than the default "NVDA"; qualify
        # so IBKR fills in conId and disambiguates the SMART routing.
        try:
            qualified = await ib.qualifyContractsAsync(opt)
            if not qualified:
                raise ProviderError("ibkr", f"could not qualify option {symbol} {expiry} {strike}{right}")
            opt = qualified[0]
        except Exception as e:
            raise ProviderError("ibkr", f"qualify option failed: {e}")

        ticker = ib.reqMktData(opt, "100,101,106", snapshot, False)
        # Wait for the snapshot fields to populate (or timeout).
        deadline = (await self._monotonic()) + timeout_s
        while (await self._monotonic()) < deadline:
            if (
                (ticker.bid is not None and ticker.bid > 0)
                or (ticker.modelGreeks and ticker.modelGreeks.delta is not None)
            ):
                break
            await asyncio.sleep(0.1)

        greeks = ticker.modelGreeks or ticker.bidGreeks or ticker.askGreeks
        # ``optPrice`` on modelGreeks is IBKR's theoretical fair value,
        # populated even when bid/ask aren't streaming (after-hours, thin
        # contracts). Used as a last-resort mid in OptionLeg.mid.
        model_price = _safe_float(getattr(greeks, "optPrice", None)) if greeks else None
        result = {
            "symbol": symbol, "expiry": expiry, "strike": float(strike), "right": right,
            "conid": opt.conId, "exchange": opt.exchange,
            "bid":   _safe_float(ticker.bid),
            "ask":   _safe_float(ticker.ask),
            "last":  _safe_float(ticker.last),
            "close": _safe_float(ticker.close),
            "model_price": model_price,
            "volume": _safe_int(ticker.volume),
            "open_interest": _safe_int(getattr(ticker, "callOpenInterest", None) if right == "C" else getattr(ticker, "putOpenInterest", None)),
            "iv":    _safe_float(getattr(greeks, "impliedVol", None)) if greeks else None,
            "delta": _safe_float(getattr(greeks, "delta", None)) if greeks else None,
            "gamma": _safe_float(getattr(greeks, "gamma", None)) if greeks else None,
            "theta": _safe_float(getattr(greeks, "theta", None)) if greeks else None,
            "vega":  _safe_float(getattr(greeks, "vega", None))  if greeks else None,
            "underlying_price": _safe_float(getattr(greeks, "undPrice", None)) if greeks else None,
        }
        ib.cancelMktData(opt)
        return result

    async def get_option_quote_by_conid(
        self, *, conid: int, timeout_s: float = 3.0,
    ) -> dict[str, Any]:
        """Quote an option leg given only its conId (no symbol/expiry/strike).

        Used by ``get_combo_quote``'s per-leg fallback when the combo bag
        itself doesn't populate ticks.
        """
        from ib_insync import Contract  # type: ignore
        ib = await self._ensure_connected()
        c = Contract(conId=int(conid), exchange="SMART", currency="USD")
        try:
            qualified = await ib.qualifyContractsAsync(c)
            if not qualified:
                return {"bid": None, "ask": None, "mid": None, "model_price": None}
            c = qualified[0]
        except Exception as e:
            logger.warning("qualify by conid %s failed: %s", conid, e)
            return {"bid": None, "ask": None, "mid": None, "model_price": None}

        ticker = ib.reqMktData(c, "100,101,106", False, False)
        try:
            deadline = (await self._monotonic()) + timeout_s
            while (await self._monotonic()) < deadline:
                if (
                    (ticker.bid is not None and ticker.bid > 0)
                    or (ticker.modelGreeks and getattr(ticker.modelGreeks, "optPrice", None))
                ):
                    break
                await asyncio.sleep(0.1)
            greeks = ticker.modelGreeks or ticker.bidGreeks or ticker.askGreeks
            model_price = _safe_float(getattr(greeks, "optPrice", None)) if greeks else None
            bid = _safe_float(ticker.bid)
            ask = _safe_float(ticker.ask)
            mid = ((bid + ask) / 2) if (bid and ask and bid > 0 and ask > 0) else None
        finally:
            ib.cancelMktData(c)
        return {"bid": bid, "ask": ask, "mid": mid, "model_price": model_price}

    async def get_auction_imbalance(
        self, *, symbol: str, exchange: str = "SMART", timeout_s: float = 2.5,
    ) -> dict[str, Any]:
        """Closing-auction imbalance for a US stock.

        IBKR generic tick 225 streams the auction values during the
        15:50-16:00 ET window: indicative match price, paired auction
        volume, and the signed auction imbalance (positive = buy
        imbalance, negative = sell pressure).

        Outside the auction window all three values come back None;
        callers treat that as "imbalance signal unavailable".
        """
        from ib_insync import Stock  # type: ignore
        ib = await self._ensure_connected()
        contract = Stock(symbol, exchange, "USD")
        try:
            qualified = await ib.qualifyContractsAsync(contract)
            if not qualified:
                return {"imbalance": None, "auction_price": None, "auction_volume": None}
            contract = qualified[0]
        except Exception as e:
            logger.warning("auction qualify failed for %s: %s", symbol, e)
            return {"imbalance": None, "auction_price": None, "auction_volume": None}

        ticker = ib.reqMktData(contract, "225", False, False)
        try:
            deadline = (await self._monotonic()) + timeout_s
            while (await self._monotonic()) < deadline:
                if (getattr(ticker, "auctionImbalance", None) is not None
                        or getattr(ticker, "auctionPrice", None) is not None):
                    break
                await asyncio.sleep(0.1)
            imbalance = _safe_float(getattr(ticker, "auctionImbalance", None))
            auction_price = _safe_float(getattr(ticker, "auctionPrice", None))
            auction_volume = _safe_float(getattr(ticker, "auctionVolume", None))
        finally:
            try:
                ib.cancelMktData(contract)
            except Exception:
                pass
        return {
            "imbalance": imbalance,
            "auction_price": auction_price,
            "auction_volume": auction_volume,
        }

    async def get_market_depth(
        self, *, contract: Any, n_rows: int = 5, timeout_s: float = 4.0,
        is_smart_depth: bool = True,
    ) -> dict[str, Any]:
        """L2 order-book depth for any qualified contract (Stock, Option, Bag).

        Subscribes via ``reqMktDepth`` (NASDAQ TotalView + NYSE OpenBook +
        Cboe BZX Depth — all in the operator's IBKR data subscription).
        Streams ``DOMBid`` / ``DOMAsk`` events; we wait briefly for the
        book to populate then snapshot the top ``n_rows`` levels.

        Returns:
            {
                "bids": [(price, size, market_maker), ...],   top-of-book first
                "asks": [(price, size, market_maker), ...],
                "fetched_at": iso,
                "depth_available": bool,
            }

        Markets-closed sessions return ``depth_available=False`` with empty
        lists; callers fall back to mid/ask-based logic.
        """
        ib = await self._ensure_connected()
        try:
            ticker = ib.reqMktDepth(
                contract, numRows=n_rows,
                isSmartDepth=is_smart_depth,
            )
        except Exception as e:
            logger.warning("reqMktDepth failed for %s: %s",
                           getattr(contract, "symbol", "?"), e)
            return {"bids": [], "asks": [], "fetched_at": None,
                    "depth_available": False, "error": str(e)}

        try:
            deadline = (await self._monotonic()) + timeout_s
            while (await self._monotonic()) < deadline:
                if (getattr(ticker, "domBids", None)
                        and len(ticker.domBids) > 0):
                    break
                if (getattr(ticker, "domAsks", None)
                        and len(ticker.domAsks) > 0):
                    break
                await asyncio.sleep(0.15)

            def _level(d: Any) -> tuple[float, float, str]:
                return (
                    _safe_float(getattr(d, "price", None)) or 0.0,
                    _safe_float(getattr(d, "size", None)) or 0.0,
                    str(getattr(d, "marketMaker", "")),
                )

            bids = [_level(d) for d in (ticker.domBids or [])][:n_rows]
            asks = [_level(d) for d in (ticker.domAsks or [])][:n_rows]
            depth_available = bool(bids or asks)
        finally:
            try:
                ib.cancelMktDepth(contract, isSmartDepth=is_smart_depth)
            except Exception:
                pass

        return {
            "bids": bids,
            "asks": asks,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "depth_available": depth_available,
        }

    async def get_market_depth_by_conid(
        self, *, conid: int, n_rows: int = 5, timeout_s: float = 4.0,
    ) -> dict[str, Any]:
        """L2 depth for a contract identified only by its conId.

        Used by the walking-limit executor to read per-leg depth and
        construct an own-side combo book when the IBKR Bag depth is
        sparse (typical for thin LEAPs).
        """
        from ib_insync import Contract  # type: ignore
        ib = await self._ensure_connected()
        c = Contract(conId=int(conid), exchange="SMART", currency="USD")
        try:
            qualified = await ib.qualifyContractsAsync(c)
            if not qualified:
                return {"bids": [], "asks": [], "fetched_at": None,
                        "depth_available": False,
                        "error": "could not qualify conid"}
            c = qualified[0]
        except Exception as e:
            return {"bids": [], "asks": [], "fetched_at": None,
                    "depth_available": False, "error": f"qualify: {e}"}
        return await self.get_market_depth(
            contract=c, n_rows=n_rows, timeout_s=timeout_s,
        )

    async def get_index_quote(
        self, *, symbol: str, exchange: str = "CBOE", timeout_s: float = 2.5,
    ) -> dict[str, Any]:
        """Real-time index quote via IBKR Index secType.

        Used by the macro-regime overlay to read VIX, VIX9D, SPX, etc.
        Returns dict with last/close; bid/ask are typically None for
        indexes (they don't trade on a book).
        """
        from ib_insync import Index  # type: ignore
        ib = await self._ensure_connected()
        contract = Index(symbol, exchange, "USD")
        try:
            qualified = await ib.qualifyContractsAsync(contract)
            if not qualified:
                return {"last": None, "close": None, "change_pct": None}
            contract = qualified[0]
        except Exception as e:
            logger.warning("index qualify failed for %s: %s", symbol, e)
            return {"last": None, "close": None, "change_pct": None}

        ticker = ib.reqMktData(contract, "", False, False)
        try:
            deadline = (await self._monotonic()) + timeout_s
            while (await self._monotonic()) < deadline:
                if (getattr(ticker, "last", None) is not None
                        or getattr(ticker, "close", None) is not None):
                    break
                await asyncio.sleep(0.1)
            last = _safe_float(ticker.last)
            close = _safe_float(ticker.close)
        finally:
            try:
                ib.cancelMktData(contract)
            except Exception:
                pass
        change_pct = None
        if last and close and close > 0:
            change_pct = (last - close) / close
        return {"last": last, "close": close, "change_pct": change_pct}

    async def _monotonic(self) -> float:
        import time
        return time.monotonic()

    async def submit_combo(
        self,
        *,
        legs: list[dict[str, Any]],
        action: str = "BUY",        # BUY = pay net debit; SELL = receive net credit
        net_price: float,
        symbol: str,
        currency: str = "USD",
        outside_rth: bool = False,
        tif: str = "DAY",
    ) -> dict[str, Any]:
        """Submit a multi-leg combo as one IBKR Bag order.

        ``legs`` is a list of leg dicts (typically two for PMCC):
            {"conid": 12345, "ratio": 1, "action": "BUY"}
            {"conid": 67890, "ratio": 1, "action": "SELL"}

        ``net_price`` is the limit on the net debit (BUY) or credit (SELL).
        For PMCC entry: action=BUY, net_price=positive net debit.
        For PMCC unwind: action=SELL, net_price=positive net credit
        (ib_insync handles sign conventions through ratio + action).

        Returns dict with order_id, status, fill snapshot.
        """
        if self.account_mode != "paper":
            raise LiveTradingDisabledError()
        from ib_insync import Bag, ComboLeg, LimitOrder  # type: ignore

        ib = await self._ensure_connected()
        bag = Bag(
            symbol=symbol, exchange="SMART", currency=currency,
            comboLegs=[
                ComboLeg(
                    conId=int(leg["conid"]),
                    ratio=int(leg.get("ratio", 1)),
                    action=str(leg["action"]).upper(),
                    exchange=leg.get("exchange", "SMART"),
                )
                for leg in legs
            ],
        )
        order = LimitOrder(
            action=action.upper(),
            totalQuantity=int(legs[0].get("ratio", 1)),  # qty of "spreads"
            lmtPrice=round(float(net_price), 2),
            outsideRth=outside_rth,
            tif=tif,
        )
        trade = ib.placeOrder(bag, order)

        # Don't block here — the executor wraps placeOrder with the walking
        # algorithm which polls trade.orderStatus.status itself. Just return
        # a snapshot so the caller has the order id immediately.
        await asyncio.sleep(0.1)
        return {
            "order_id": trade.order.orderId,
            "perm_id": trade.order.permId,
            "status": trade.orderStatus.status,
            "filled": float(trade.orderStatus.filled or 0),
            "remaining": float(trade.orderStatus.remaining or order.totalQuantity),
            "avg_fill_price": float(trade.orderStatus.avgFillPrice or 0),
            "_trade": trade,  # the executor uses this to cancel/poll
        }

    async def get_combo_quote(
        self, *, legs: list[dict[str, Any]],
        symbol: str, currency: str = "USD",
        timeout_s: float = 6.0,
    ) -> dict[str, Any]:
        """Quote bid/ask on a Bag combo so the executor knows where mid is.

        Strategy:
          1. Try the combo bag in *streaming* mode (snapshot=True returns
             empty for combos most of the time — IBKR's combo book takes
             a beat to spin up).
          2. If the bag doesn't populate within ``timeout_s``, fall back
             to summing per-leg quotes (signed by action: BUY adds ask /
             subtracts the buyer's premium, SELL subtracts bid).

        Returns ``{"bid": float|None, "ask": float|None, "mid": float|None}``.
        """
        from ib_insync import Bag, ComboLeg  # type: ignore
        ib = await self._ensure_connected()
        bag = Bag(
            symbol=symbol, exchange="SMART", currency=currency,
            comboLegs=[
                ComboLeg(
                    conId=int(leg["conid"]), ratio=int(leg.get("ratio", 1)),
                    action=str(leg["action"]).upper(),
                    exchange=leg.get("exchange", "SMART"),
                )
                for leg in legs
            ],
        )
        # Streaming (snapshot=False) — combos rarely populate in snapshot mode.
        ticker = ib.reqMktData(bag, "", False, False)
        try:
            deadline = (await self._monotonic()) + timeout_s
            while (await self._monotonic()) < deadline:
                if ticker.bid is not None and ticker.ask is not None and ticker.bid > 0 and ticker.ask > 0:
                    break
                await asyncio.sleep(0.15)
            bid = _safe_float(ticker.bid)
            ask = _safe_float(ticker.ask)
        finally:
            ib.cancelMktData(bag)

        if bid and ask and bid > 0 and ask > 0:
            return {"bid": bid, "ask": ask, "mid": (bid + ask) / 2}

        # ---- Per-leg fallback ---------------------------------------------
        # Sum signed leg quotes. For a 2-leg PMCC (BUY LEAP, SELL short call):
        #   net_debit_bid = leap.bid  - short.ask    (worst-case to enter)
        #   net_debit_ask = leap.ask  - short.bid    (best-case to exit)
        # General formula per leg (action='BUY'):
        #   contributes +ask to debit-ask, +bid to debit-bid
        # Action='SELL':
        #   contributes -bid to debit-ask, -ask to debit-bid
        try:
            leg_bid = 0.0
            leg_ask = 0.0
            ok = True
            for leg in legs:
                conid = int(leg["conid"])
                action = str(leg["action"]).upper()
                ratio = int(leg.get("ratio", 1))
                qd = await self.get_option_quote_by_conid(conid=conid)
                lb = _safe_float(qd.get("bid"))
                la = _safe_float(qd.get("ask"))
                lm = _safe_float(qd.get("mid")) or _safe_float(qd.get("model_price"))
                # If neither bid/ask nor model_price, this leg has no usable price
                if (lb is None or la is None or lb <= 0 or la <= 0) and (lm is None or lm <= 0):
                    ok = False
                    break
                # Use bid/ask if present, otherwise model_price for both sides.
                if lb is None or la is None or lb <= 0 or la <= 0:
                    lb = la = float(lm)
                if action == "BUY":
                    leg_bid += ratio * lb
                    leg_ask += ratio * la
                else:   # SELL
                    leg_bid -= ratio * la
                    leg_ask -= ratio * lb
            if ok:
                # leg_bid <= leg_ask by construction; ensure orientation
                lo, hi = (leg_bid, leg_ask) if leg_bid <= leg_ask else (leg_ask, leg_bid)
                return {"bid": round(lo, 2), "ask": round(hi, 2),
                        "mid": round((lo + hi) / 2, 2)}
        except Exception as e:
            logger.warning("combo per-leg quote fallback failed for %s: %s", symbol, e)

        return {"bid": bid, "ask": ask, "mid": None}

    async def get_historical_bars(
        self,
        *,
        symbol: str,
        duration: str = "1 Y",       # ib_insync syntax: "1 Y", "6 M", "30 D", "5000 S"
        bar_size: str = "1 day",     # "1 day" | "1 hour" | "5 mins" | etc.
        what_to_show: str = "TRADES", # "TRADES" | "MIDPOINT" | "BID" | "ASK"
        use_rth: bool = True,
        sec_type: str = "STK",
        exchange: str = "SMART",
        currency: str = "USD",
    ) -> list[Any]:
        """Fetch historical bars for a contract via reqHistoricalData.

        Used by the sector-regime calculator (ETFs) and the market analyst
        (single stocks). Returns ib_insync's BarData list; callers extract
        ``.date``, ``.open``, ``.high``, ``.low``, ``.close``, ``.volume``.

        Caches the qualified contract on the IBKR instance so repeated
        calls for the same symbol skip the qualification round-trip.
        """
        from ib_insync import Stock, Forex, Index  # type: ignore

        ib = await self._ensure_connected()
        sec_type = (sec_type or "STK").upper()
        if sec_type == "STK":
            contract = Stock(symbol, exchange, currency)
        elif sec_type == "IND":
            contract = Index(symbol, exchange, currency)
        elif sec_type == "CASH":
            contract = Forex(symbol)
        else:
            raise ProviderError("ibkr", f"unsupported secType {sec_type!r}", retryable=False)
        # Index symbols (^VIX) need ARCA/CBOE rather than SMART.
        if symbol.startswith("^"):
            contract.exchange = "CBOE"
            contract.symbol = symbol.lstrip("^")
        # Qualify so the contract has the full conId etc.
        try:
            qualified = await ib.qualifyContractsAsync(contract)
            if not qualified:
                raise ProviderError("ibkr", f"qualifyContracts returned nothing for {symbol}",
                                    retryable=False)
            contract = qualified[0]
        except Exception as e:
            raise ProviderError("ibkr", f"qualify failed for {symbol}: {e}",
                                retryable=False)

        bars = await ib.reqHistoricalDataAsync(
            contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow=what_to_show,
            useRTH=use_rth,
            formatDate=1,
            keepUpToDate=False,
        )
        return list(bars or [])

    async def submit_trade(self, intent: TradeIntent) -> dict[str, Any]:
        if intent.account_mode != "paper" or self.account_mode != "paper":
            raise LiveTradingDisabledError()
        if intent.qty <= 0:
            raise ProviderError("ibkr", "qty must be positive")
        from ib_insync import (  # type: ignore
            Stock,
            MarketOrder,
            LimitOrder,
            StopLimitOrder,
        )

        ib = await self._ensure_connected()
        contract = Stock(intent.ticker, "SMART", "USD")
        await ib.qualifyContractsAsync(contract)

        action = "BUY" if intent.side.upper() == "BUY" else "SELL"
        if intent.order_type == "MKT":
            order = MarketOrder(action, intent.qty, tif=intent.tif)
        elif intent.order_type == "LMT":
            if intent.limit_px is None:
                raise ProviderError("ibkr", "LMT order requires limit_px")
            order = LimitOrder(action, intent.qty, float(intent.limit_px), tif=intent.tif)
        elif intent.order_type == "STP_LMT":
            if intent.limit_px is None or intent.stop_px is None:
                raise ProviderError("ibkr", "STP_LMT order requires limit_px and stop_px")
            order = StopLimitOrder(
                action, intent.qty, float(intent.limit_px), float(intent.stop_px), tif=intent.tif
            )
        else:
            raise ProviderError("ibkr", f"unsupported order_type {intent.order_type!r}")

        # Belt-and-braces: IBKR's `whatIfOrder` first to surface margin and
        # commission previews — handy for the UI, but also catches malformed
        # intents before anything reaches the matching engine.
        what_if = await ib.whatIfOrderAsync(contract, order)
        trade = ib.placeOrder(contract, order)

        audit = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ticker": intent.ticker,
            "side": intent.side,
            "qty": intent.qty,
            "order_type": intent.order_type,
            "limit_px": str(intent.limit_px) if intent.limit_px else None,
            "stop_px": str(intent.stop_px) if intent.stop_px else None,
            "ibkr_order_id": trade.order.orderId,
            "perm_id": trade.order.permId,
            "what_if_initial_margin": getattr(what_if, "initMarginChange", None),
            "what_if_commission": getattr(what_if, "commission", None),
        }
        logger.info("ibkr.audit %s", audit)
        return audit

    async def cancel_order(self, order_id: str) -> dict[str, Any]:
        ib = await self._ensure_connected()
        for trade in ib.trades():
            if str(trade.order.orderId) == str(order_id):
                ib.cancelOrder(trade.order)
                return {"order_id": order_id, "cancelled": True}
        raise ProviderError("ibkr", f"order {order_id!r} not found among open trades")
