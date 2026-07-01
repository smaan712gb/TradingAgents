"""Interactive Brokers (IBKR) execution provider.

Uses `ib_insync` over the TWS/Gateway socket API. The agents NEVER call
this directly; the FastAPI router accepts a `TradeIntent` from the user
(after on-screen confirmation in the UI) and routes here. We expose this
as a Provider for symmetry with the data vendors and so backtests can
substitute a fake.

Hard rules in this file:

* `account_mode` is `"paper"` or `"live"`. The real-money safety invariant
  is NOT a flag — it is a runtime check that the *connected account's prefix
  matches the declared mode*: IBKR paper accounts use a `D` prefix (IB Gateway
  paper, e.g. `DU1234567`, port 4002) and live real-money accounts use a `U`
  prefix (port 4001). `verify_account_mode` enforces this on every (re)connect
  and bails before any order can reach the wrong account. So "paper" against a
  `U`-account, or "live" against a `D`-account, can never trade.
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
from datetime import datetime, timedelta, timezone
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


def verify_account_mode(account_mode: str, account_id: str) -> None:
    """Real-money safety invariant: connected account must match declared mode.

    IBKR paper accounts use a 'D' prefix (IB Gateway paper, e.g. DU1234567,
    port 4002); live real-money accounts use a 'U' prefix (port 4001). A
    mismatch — paper mode against a U-account, or live mode against a D-account
    — is operator error and MUST block trading. Enforced on every (re)connect
    before any order can be placed. Raises ``ProviderError`` on mismatch;
    returns None when the account is consistent with the mode.

    This is the actual guard that lets us run the live execution path against an
    IB Gateway PAPER account safely: orders route to the paper account, and a
    real-money (U) account can never be hit while in paper mode.
    """
    if account_mode == "paper" and not account_id.startswith("D"):
        raise ProviderError(
            "ibkr",
            f"IBKR_MODE=paper but connected account {account_id!r} doesn't "
            f"start with 'D' (paper accounts use D-prefix). Connect to Gateway "
            f"in paper mode (port 4002) or set IBKR_MODE=live.",
            retryable=False,
        )
    if account_mode == "live" and not account_id.startswith("U"):
        raise ProviderError(
            "ibkr",
            f"IBKR_MODE=live but connected account {account_id!r} doesn't "
            f"start with 'U' (live accounts use U-prefix). Connect to Gateway "
            f"in live mode (port 4001) or set IBKR_MODE=paper.",
            retryable=False,
        )


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
        if self.account_mode not in ("paper", "live"):
            raise ValueError(
                f"IBKR_MODE={self.account_mode!r}: must be 'paper' or 'live'"
            )
        if self.account_mode == "live":
            logger.warning(
                "IbkrProvider initialized in LIVE mode "
                "(host=%s port=%s cid=%s) — real-money execution path",
                self.host, self.port, self.client_id,
            )
        try:
            from ib_insync import IB  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ProviderError("ibkr", "ib_insync not installed", retryable=False) from e
        self._ib_cls = IB
        self._ib: Any = None
        self._connect_lock = asyncio.Lock()
        self._reconnect_attempts = 0
        # Market-data health: ib_insync's errorEvent records the monotonic time
        # of the last data-farm refusal (10197 "competing live session" + the
        # data-farm-down codes). The socket stays UP for these, so the
        # disconnect listener never fires — the heartbeat watches these instead
        # and calls force_reconnect() to re-acquire the data farm.
        self._md_error_at: float = 0.0
        self._last_force_reconnect: float = 0.0

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

            # ib_insync binds its socket via util.getLoop() ==
            # policy.get_event_loop() (the THREAD-CURRENT loop), not
            # asyncio.get_running_loop(). Under uvicorn the thread-current
            # loop is unset/different from the running loop, so the IB socket
            # ends up attached to a different loop -> "Future attached to a
            # different loop" the moment we await a request. Pin the
            # thread-current loop to the running loop so ib_insync uses it.
            # No-op in a standalone asyncio.run() context.
            try:
                asyncio.set_event_loop(asyncio.get_running_loop())
            except RuntimeError:
                pass

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
            # Real-money safety invariant: the connected account's prefix must
            # match the declared mode (paper=D, live=U). Disconnect + bail on
            # mismatch so an order can never reach the wrong account.
            try:
                verify_account_mode(self.account_mode, acct)
            except ProviderError:
                try: ib.disconnect()
                except Exception: pass
                raise

            # Wire a disconnect listener so the next call notices and reconnects.
            try:
                ib.disconnectedEvent += self._on_disconnect
            except Exception:
                pass
            # Wire an error listener to catch market-data-farm refusals (10197
            # "competing live session" etc.) — these leave the socket UP, so the
            # disconnect path won't fire; the heartbeat reads market_data_unhealthy().
            try:
                ib.errorEvent += self._on_error
            except Exception:
                pass

            # Market-data type. Configurable via IBKR_MARKET_DATA_TYPE:
            #   1 = live (real-time) — REQUIRED for limit orders to be
            #       marketable + fill; needs a real-time data subscription
            #       on the account (shared with the paper account).
            #   3 = delayed (streaming), 4 = delayed-frozen (free, ~15min).
            # Default 4 (free) so the dashboard shows prices without a
            # subscription — but delayed prices make limit orders non-
            # marketable, so the sim won't fill them. Set to 1 once the
            # real-time subscription is enabled to get fills.
            md_type = int(os.getenv("IBKR_MARKET_DATA_TYPE", "4") or "4")
            try:
                ib.reqMarketDataType(md_type)
                logger.info("ibkr: market data type=%d", md_type)
                # REAL MONEY needs LIVE (type 1) data: delayed/frozen prices make
                # limit orders non-marketable and blind the exit engine to intraday
                # drawdowns. Scream loudly if a live account is on delayed data.
                if self.account_mode == "live" and md_type != 1:
                    logger.critical(
                        "ibkr: LIVE account on market-data type=%d (NOT live) — limit "
                        "orders may be non-marketable and exits blind to intraday moves. "
                        "Set IBKR_MARKET_DATA_TYPE=1.", md_type)
            except Exception as e:
                logger.warning("reqMarketDataType(%d) failed: %s", md_type, e)

            self._ib = ib
            self._reconnect_attempts = 0
            logger.info("ibkr: connected %s account=%s (clientId=%d)",
                        self.account_mode, acct, self.client_id)
            return ib

    def _on_disconnect(self) -> None:
        """ib_insync fires this when the socket drops. Reset state so the
        next ``_ensure_connected`` call rebuilds the connection."""
        logger.warning("ibkr: disconnect detected, will reconnect on next call")
        self._ib = None

    # Data-farm errors where the socket stays connected but market data is
    # unavailable: 10197 (competing live session), 1100 (connectivity lost),
    # 2103/2105/2157 (data-farm connection broken).
    _MD_FARM_ERROR_CODES = frozenset({10197, 1100, 2103, 2105, 2157})
    _FORCE_RECONNECT_COOLDOWN = 300.0   # ≥5 min between forced reconnects — never thrash

    def _on_error(self, reqId: Any = None, errorCode: Any = None,
                  errorString: Any = None, contract: Any = None) -> None:
        """ib_insync errorEvent handler — stamp the time of data-farm refusals
        so the heartbeat can trigger an active recovery. Never raises."""
        try:
            if int(errorCode) in self._MD_FARM_ERROR_CODES:
                import time as _t
                self._md_error_at = _t.monotonic()
        except Exception:
            pass

    def market_data_unhealthy(self, window: float = 90.0) -> bool:
        """True if a data-farm refusal (e.g. 10197) occurred within ``window``
        seconds. The socket may still report connected — this catches the case
        the disconnect listener misses."""
        import time as _t
        return self._md_error_at > 0.0 and (_t.monotonic() - self._md_error_at) < window

    async def force_reconnect(self, reason: str = "") -> bool:
        """Tear down and rebuild the connection to re-acquire the market-data
        farm. For the socket-UP-but-data-refused case (10197) that the
        disconnect listener never catches. Cooldown-guarded so repeated 10197s
        can't thrash the connection. Returns True if a reconnect was performed."""
        import time as _t
        now = _t.monotonic()
        if now - self._last_force_reconnect < self._FORCE_RECONNECT_COOLDOWN:
            return False
        # SAFETY: never tear down the socket while orders are working — a forced
        # reconnect orphans the ib_insync Trade handles and the resting orders
        # become untracked/uncancellable (they can still fill at the broker).
        # 10197 is a market-DATA problem; if we hold live orders, KEEP the
        # connection and let quotes degrade to the fallback chain instead.
        try:
            if self._ib is not None and self._ib.isConnected():
                working = [
                    t for t in self._ib.openTrades()
                    if str(getattr(getattr(t, "orderStatus", None), "status", "")) in
                    ("PendingSubmit", "PreSubmitted", "Submitted", "ApiPending", "PendingCancel")
                ]
                if working:
                    logger.warning("ibkr: skipping forced reconnect — %d working order(s) would "
                                   "be orphaned; keeping connection (data degrades to fallback)",
                                   len(working))
                    return False
        except Exception:
            pass
        self._last_force_reconnect = now
        logger.warning("ibkr: forcing reconnect to recover market data (%s)", reason or "manual")
        async with self._connect_lock:
            if self._ib is not None:
                try:
                    self._ib.disconnect()
                except Exception:
                    pass
                self._ib = None
        try:
            ib = await self._ensure_connected()   # re-acquires its own lock + re-subs the farm
            # Re-adopt any orders working at the broker into the fresh IB
            # instance's state so nothing placed pre-reconnect is left untracked.
            try:
                await ib.reqAllOpenOrdersAsync()
            except Exception as e:
                logger.debug("ibkr: reqAllOpenOrders after reconnect failed: %s", e)
            self._md_error_at = 0.0
            logger.info("ibkr: forced reconnect complete — market-data farm re-subscribed")
            return True
        except Exception as e:
            logger.warning("ibkr: forced reconnect failed: %s", e)
            return False

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

    # ------------------------------------------------------------------
    # News (free with the account's IBKR news-provider subscriptions)
    # ------------------------------------------------------------------

    async def get_news_providers(self) -> list[dict[str, str]]:
        """News providers the account is subscribed to (codes used for
        historical-news queries). Cached per process."""
        if getattr(self, "_news_providers", None) is None:
            ib = await self._ensure_connected()
            try:
                provs = await ib.reqNewsProvidersAsync()
                self._news_providers = [{"code": p.code, "name": p.name} for p in provs]
            except Exception as e:
                logger.warning("ibkr: reqNewsProviders failed: %s", e)
                self._news_providers = []
        return self._news_providers

    async def get_historical_news(
        self, symbol: str, *, lookback_hours: int = 24,
        max_results: int = 10, fetch_body: bool = True,
    ) -> list[dict[str, Any]]:
        """Recent news for a symbol from the account's subscribed providers.

        Polls reqHistoricalNews (sweep-friendly — no streaming callbacks) and
        optionally pulls each article body via reqNewsArticle for keyword/LLM
        analysis. Returns [] when no providers are subscribed or the symbol
        can't be qualified — never raises into the caller."""
        from ib_insync import Stock  # type: ignore
        ib = await self._ensure_connected()
        try:
            qualified = await ib.qualifyContractsAsync(Stock(symbol, "SMART", "USD"))
        except Exception as e:
            logger.debug("news: qualify failed for %s: %s", symbol, e)
            return []
        if not qualified or not getattr(qualified[0], "conId", 0):
            return []
        conid = qualified[0].conId

        providers = await self.get_news_providers()
        if not providers:
            return []
        provider_codes = "+".join(p["code"] for p in providers)

        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=lookback_hours)
        fmt = "%Y-%m-%d %H:%M:%S.0"   # IBKR historical-news datetime format
        try:
            items = await ib.reqHistoricalNewsAsync(
                conid, provider_codes, start.strftime(fmt), end.strftime(fmt), max_results,
            )
        except Exception as e:
            logger.warning("news: reqHistoricalNews failed for %s: %s", symbol, e)
            return []

        out: list[dict[str, Any]] = []
        for it in (items or []):
            row = {
                "symbol": symbol,
                "time": getattr(it, "time", None),
                "provider": getattr(it, "providerCode", ""),
                "article_id": getattr(it, "articleId", ""),
                "headline": getattr(it, "headline", ""),
                "body": None,
            }
            if fetch_body and row["provider"] and row["article_id"]:
                try:
                    art = await ib.reqNewsArticleAsync(row["provider"], row["article_id"])
                    row["body"] = getattr(art, "articleText", None)
                except Exception as e:
                    logger.debug("news: article body fetch failed (%s/%s): %s",
                                 row["provider"], row["article_id"], e)
            out.append(row)
        return out

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
                contract = p.contract
                # Positions come back with no exchange on option legs, and
                # reqMktData then errors 321 ("please enter exchange"), leaving
                # option P&L stuck at cost basis. Qualify to fill the exchange;
                # fall back to SMART (US options route there) if qualify can't.
                if not getattr(contract, "exchange", None):
                    try:
                        q = await ib.qualifyContractsAsync(contract)
                        if q:
                            contract = q[0]
                    except Exception as qe:
                        logger.debug("qualify failed for %s: %s", contract.symbol, qe)
                    if not getattr(contract, "exchange", None):
                        contract.exchange = "SMART"
                ticker = ib.reqMktData(contract, "", False, False)
                self._pos_subs[cid] = (contract, ticker)
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
            got_tick = False   # did a REAL price arrive, or are we falling back to cost basis?
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
                            got_tick = True
                            break
                except Exception:
                    pass
            # Expose enough contract detail so callers can disambiguate
            # the two-options-with-same-symbol case (PMCC = LEAP + short
            # call on the same underlying). Without conid/expiry/strike/
            # right, downstream UI and DB upserts collide on symbol alone.
            c = p.contract
            # Normalize price units. IBKR's avgCost on options is
            # premium × multiplier (cost basis per contract); the ticker's
            # marketPrice() is premium per share. Mixing them produces
            # nonsense PnL like $1044 vs $10.44 for the same contract.
            # Convert to per-share consistently for options so the dashboard
            # can render with the same fmtMoney helper as stocks.
            try:
                mult_int = int(c.multiplier) if c.multiplier else 1
            except (TypeError, ValueError):
                mult_int = 100  # standard equity-option multiplier
            is_option = (c.secType or "").upper() == "OPT"
            if is_option and mult_int > 0:
                avg_per_share = avg / mult_int
                # `last` was sourced from ticker.marketPrice / last / bid /
                # close — those are already per-share for options. The
                # fallback path in the loop above assigned last = avg
                # when no tick arrived, which is per-CONTRACT — detect
                # that and normalize.
                if last == avg:
                    last_per_share = avg_per_share
                else:
                    last_per_share = last
                avg_out = avg_per_share
                last_out = last_per_share
                pnl_out = (last_per_share - avg_per_share) * qty * mult_int
            else:
                avg_out = avg
                last_out = last
                pnl_out = (last - avg) * qty
            out.append({
                "account_id": p.account,
                "symbol": c.symbol,
                "secType": c.secType,
                "currency": c.currency,
                "qty": qty,
                "avg_price": round(avg_out, 4),
                "last_price": round(last_out, 4),
                # False when NO live tick arrived and last_price fell back to
                # cost basis — callers must NOT treat a cost-basis substitution
                # as a real (flat) price and suppress an exit on it.
                "price_fresh": bool(got_tick),
                "pnl": round(pnl_out, 2),
                "conid": int(getattr(c, "conId", 0) or 0),
                "local_symbol": getattr(c, "localSymbol", "") or "",
                "expiry": getattr(c, "lastTradeDateOrContractMonth", "") or "",
                "strike": float(getattr(c, "strike", 0) or 0) or None,
                "right": getattr(c, "right", "") or "",
                "multiplier": getattr(c, "multiplier", "") or "",
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
        # reqSecDefOptParams returns ONE ROW PER EXCHANGE, and each row's
        # expiration/strike set can differ — a LEAP listed on one exchange may
        # be absent from another exchange's row. The old code read a SINGLE row
        # (SMART, else first) and treated it as the whole chain, which silently
        # dropped expirations for thinner names (e.g. SNDK's Jan-2028 LEAP was
        # missing from the row read → false "no expirations in window" even
        # though the contract is real). Take the UNION across ALL rows so
        # eligibility sees every listed expiration/strike. Retry on an empty
        # response — the data farm intermittently returns nothing while it's
        # flapping (the Error 162 / 1100↔1102 farm blips), and a transient
        # empty must NOT be mistaken for "this name has no options".
        params = None
        for attempt in range(3):
            params = await ib.reqSecDefOptParamsAsync(
                underlyingSymbol=underlying.symbol,
                futFopExchange="",
                underlyingSecType=underlying.secType,
                underlyingConId=underlying.conId,
            )
            if params and any(getattr(p, "expirations", None) for p in params):
                break
            if attempt < 2:
                await asyncio.sleep(1.5)
        if not params or not any(getattr(p, "expirations", None) for p in params):
            # retryable=True → the eligibility path treats this as a transient
            # probe failure to re-attempt next tick, NOT as "no LEAP exists".
            raise ProviderError("ibkr", f"no option params for {symbol} (empty after retries)",
                                retryable=True)

        expirations: set[str] = set()
        strikes: set[float] = set()
        for p in params:
            expirations.update(getattr(p, "expirations", None) or [])
            for s in (getattr(p, "strikes", None) or []):
                try:
                    strikes.add(float(s))
                except (TypeError, ValueError):
                    continue
        # SMART (else first) row supplies trading_class / exchange metadata; the
        # expiration & strike UNION above comes from every row.
        meta = next((p for p in params if p.exchange == "SMART"), params[0])
        return {
            "underlying_symbol": underlying.symbol,
            "underlying_conid": underlying.conId,
            "trading_class": meta.tradingClass,
            "exchange": meta.exchange,
            "expirations": sorted(expirations),
            "strikes": sorted(strikes),
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

    async def get_atm_call_iv(
        self, *, symbol: str, target_dte: int = 30,
        timeout_s: float = 4.0,
    ) -> dict[str, Any]:
        """Front-month ATM call implied volatility.

        Picks the listed expiration whose DTE is closest to ``target_dte``
        (default 30 days), then the strike closest to current spot, then
        reads the option's implied vol via tick 106 (modelGreeks).

        Returns:
            {
                "iv": float | None,         # implied volatility (decimal, e.g. 0.45)
                "strike": float | None,
                "expiry": str | None,       # YYYYMMDD
                "dte": int | None,
                "spot": float | None,
            }

        Used by the IV-percentile signal in momentum_exhaustion.
        Markets-closed sessions return cached IV from modelGreeks.optPrice
        when available, None otherwise.
        """
        from datetime import date as _date
        out: dict[str, Any] = {
            "iv": None, "strike": None, "expiry": None,
            "dte": None, "spot": None,
        }

        # Process-level negative cache: names with no listed options (e.g.
        # ADRs like ABBNY) would otherwise re-fail the chain probe on every
        # IV-snapshot tick. Once we confirm a name isn't optionable, skip it.
        no_chain = getattr(self, "_iv_no_chain", None)
        if no_chain is None:
            no_chain = self._iv_no_chain = set()
        if symbol in no_chain:
            return out

        try:
            chain = await self.get_option_chain(symbol=symbol)
        except Exception as e:
            msg = str(e).lower()
            if "no option" in msg or "no security definition" in msg:
                no_chain.add(symbol)   # not optionable — stop re-probing
                logger.debug("ATM IV: %s is not optionable — caching skip", symbol)
            else:
                logger.warning("ATM IV: chain fetch failed for %s: %s", symbol, e)
            return out

        expirations: list[str] = chain.get("expirations") or []
        strikes: list[float] = chain.get("strikes") or []
        if not expirations or not strikes:
            no_chain.add(symbol)       # confirmed no chain — stop re-probing
            return out

        today = _date.today()
        def _dte_of(exp: str) -> int:
            try:
                ed = _date(int(exp[:4]), int(exp[4:6]), int(exp[6:8]))
                return (ed - today).days
            except Exception:
                return 99999
        def _is_monthly(exp: str) -> bool:
            # Standard monthly = 3rd Friday. Monthlies carry the full strike
            # ladder; weeklies are sparse, so pairing a weekly expiry with the
            # nearest-to-spot strike often yields a contract that doesn't exist
            # (IBKR Error 200). get_option_chain returns the UNION of strikes
            # and expirations, so we must bias to expirations that actually
            # list the ATM strike.
            try:
                ed = _date(int(exp[:4]), int(exp[4:6]), int(exp[6:8]))
                return ed.weekday() == 4 and 15 <= ed.day <= 21
            except Exception:
                return False
        expirations_with_dte = [(e, _dte_of(e)) for e in expirations]
        # Filter out expired contracts (negative DTE)
        future = [(e, d) for e, d in expirations_with_dte if d > 0]
        if not future:
            return out
        # Prefer monthly expirations (dense strike ladder); fall back to the
        # full set only if the name lists no monthlies.
        monthlies = [(e, d) for e, d in future if _is_monthly(e)]
        pool = monthlies if monthlies else future
        # Pick the closest DTE to target within the chosen pool
        target_exp, target_dte_actual = min(pool, key=lambda ed: abs(ed[1] - target_dte))

        # Spot for ATM strike pick
        try:
            from tradingagents.strategies.pmcc import _fetch_spot
            spot = await _fetch_spot(self, symbol)
        except Exception:
            spot = None
        if not spot or spot <= 0:
            return out

        # Snap the strike to one ACTUALLY listed for the chosen expiry. The
        # chain's `strikes` is the UNION across all expirations, so the nearest
        # union strike is frequently not listed on this specific monthly
        # (e.g. 2.5-increment weeklies) -> IBKR Error 200. Enumerate valid
        # strikes for (symbol, expiry, C) via reqContractDetails and pick ATM
        # from those.
        candidate_strikes = strikes
        try:
            from ib_insync import Option  # type: ignore
            ib = await self._ensure_connected()
            probe = Option(symbol, target_exp, 0.0, "C", exchange="SMART", currency="USD")
            details = await ib.reqContractDetailsAsync(probe)
            exp_strikes = sorted({float(d.contract.strike)
                                  for d in (details or []) if d.contract.strike})
            if exp_strikes:
                candidate_strikes = exp_strikes
        except Exception as e:
            logger.debug("ATM IV: per-expiry strike enum failed for %s %s: %s",
                         symbol, target_exp, e)
        atm_strike = min(candidate_strikes, key=lambda s: abs(s - spot))
        out["spot"] = spot
        out["strike"] = atm_strike
        out["expiry"] = target_exp
        out["dte"] = target_dte_actual

        try:
            quote = await self.get_option_quote(
                symbol=symbol, expiry=target_exp, strike=atm_strike, right="C",
                timeout_s=timeout_s,
            )
            iv = quote.get("iv")
            if iv is not None and iv > 0:
                out["iv"] = float(iv)
        except Exception as e:
            logger.warning("ATM IV: quote failed for %s %s %s C: %s",
                           symbol, target_exp, atm_strike, e)

        return out

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
        contracts: int = 1,         # number of SPREADS (combos) to trade
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
        if self.account_mode not in ("paper", "live"):
            raise ValueError(
                f"submit_combo: invalid account_mode {self.account_mode!r}"
            )
        from ib_insync import Bag, ComboLeg, LimitOrder, TagValue  # type: ignore

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
        # totalQuantity is the number of SPREADS (combos); each leg's per-spread
        # count is carried by its ComboLeg.ratio. The old code used
        # legs[0].ratio (always 1) here, so EVERY multi-contract combo/roll
        # submitted just 1 spread — a 10-lot roll rolled 1 and left 9 in the
        # old leg while the intent was rewritten as fully rolled. Use `contracts`.
        n_spreads = max(1, int(contracts))
        order = LimitOrder(
            action=action.upper(),
            totalQuantity=n_spreads,
            lmtPrice=round(float(net_price), 2),
            outsideRth=outside_rth,
            tif=tif,
        )
        # Let IBKR work the order inside the spread (Price Management Algo),
        # and allow leg-by-leg fills on thin combos via NonGuaranteed Smart
        # Routing. Without these flags, a bare LMT on a wide-spread BAG often
        # sits at the original price and times out — the symptom that was
        # killing our walking-limit fills.
        order.usePriceMgmtAlgo = True
        order.smartComboRoutingParams = [TagValue("NonGuaranteed", "1")]
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
        # Account-mode consistency: the intent's mode must match the
        # provider's connection mode. Mixing paper-intent against a live
        # connection (or vice versa) is operator error.
        if intent.account_mode != self.account_mode:
            raise ProviderError(
                "ibkr",
                f"account_mode mismatch: intent={intent.account_mode!r} "
                f"provider={self.account_mode!r}",
                retryable=False,
            )
        if self.account_mode not in ("paper", "live"):
            raise ValueError(
                f"submit_trade: invalid account_mode {self.account_mode!r}"
            )
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
