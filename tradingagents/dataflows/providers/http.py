"""Shared async HTTP client used by every external-vendor provider.

Centralizes the parts that should be identical for every vendor:

* `httpx.AsyncClient` with sensible timeouts.
* Retry-with-jitter on connect errors / 5xx / 429 (via tenacity).
* Per-host concurrency cap so one slow vendor can't starve the others.
* Translation of HTTP errors to typed `ProviderError` subclasses so the
  graph can react (retry, fall back to another vendor, surface to user).
* Stripped logging — request bodies and headers are never logged in full
  because they contain API keys.

Provider classes own the *what* (paths, params, response schemas). This
module owns the *how* (transport, reliability).
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Mapping, Optional

import httpx
from tenacity import (
    AsyncRetrying,
    RetryError,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from .base import AuthError, ProviderError, RateLimitError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-vendor concurrency caps. Tune via env if you upgrade tiers.
# ---------------------------------------------------------------------------


_DEFAULT_HOST_CAPS: dict[str, int] = {
    "api.polygon.io": 10,
    "api.unusualwhales.com": 6,
    "financialmodelingprep.com": 8,
    "www.alphavantage.co": 4,
}


@dataclass
class HttpClientConfig:
    timeout_s: float = 15.0
    connect_timeout_s: float = 5.0
    max_retries: int = 4
    retry_min_wait_s: float = 0.5
    retry_max_wait_s: float = 8.0
    user_agent: str = "tradingagents-pro/0.1"


class AsyncHttpClient:
    """Reusable wrapper around httpx.AsyncClient. Construct once per
    provider, share the underlying connection pool across calls."""

    def __init__(
        self,
        provider_name: str,
        base_url: str,
        default_headers: Optional[Mapping[str, str]] = None,
        cfg: Optional[HttpClientConfig] = None,
        host_concurrency: Optional[int] = None,
    ) -> None:
        self.provider_name = provider_name
        self.cfg = cfg or HttpClientConfig()
        # HTTP/2 is a perf nicety, not a correctness requirement. Only enable
        # it if the optional `h2` dep is installed; otherwise httpx raises.
        try:
            import h2  # type: ignore  # noqa: F401
            http2 = True
        except ImportError:
            http2 = False
        self._client = httpx.AsyncClient(
            base_url=base_url,
            http2=http2,
            timeout=httpx.Timeout(self.cfg.timeout_s, connect=self.cfg.connect_timeout_s),
            headers={
                "Accept": "application/json",
                "User-Agent": self.cfg.user_agent,
                **(default_headers or {}),
            },
        )
        cap = host_concurrency or _DEFAULT_HOST_CAPS.get(httpx.URL(base_url).host or "", 8)
        self._sem = asyncio.Semaphore(cap)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def get_json(self, path: str, params: Optional[Mapping[str, Any]] = None) -> Any:
        return await self._request("GET", path, params=params)

    async def post_json(self, path: str, json: Optional[Mapping[str, Any]] = None) -> Any:
        return await self._request("POST", path, json=json)

    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[Mapping[str, Any]] = None,
        json: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        retry = AsyncRetrying(
            stop=stop_after_attempt(self.cfg.max_retries),
            wait=wait_exponential_jitter(
                initial=self.cfg.retry_min_wait_s,
                max=self.cfg.retry_max_wait_s,
            ),
            retry=retry_if_exception_type((RateLimitError, httpx.ConnectError, httpx.ReadTimeout)),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        try:
            async for attempt in retry:
                with attempt:
                    async with self._sem:
                        resp = await self._client.request(method, path, params=params, json=json)
                    return self._parse(resp, path)
        except RetryError as e:  # pragma: no cover
            raise e.last_attempt.exception()  # type: ignore[misc]

    def _parse(self, resp: httpx.Response, path: str) -> Any:
        if resp.status_code in (401, 403):
            raise AuthError(self.provider_name)
        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            raise RateLimitError(
                self.provider_name,
                retry_after_s=float(retry_after) if retry_after else None,
            )
        if 500 <= resp.status_code < 600:
            raise ProviderError(self.provider_name, f"{resp.status_code} on {path}", retryable=True)
        if resp.status_code >= 400:
            body = resp.text[:200]
            raise ProviderError(self.provider_name, f"{resp.status_code} on {path}: {body}", retryable=False)
        try:
            return resp.json()
        except ValueError:
            raise ProviderError(
                self.provider_name,
                f"non-JSON response on {path}: {resp.text[:200]}",
                retryable=False,
            )


@asynccontextmanager
async def http_client(
    provider_name: str,
    base_url: str,
    headers: Optional[Mapping[str, str]] = None,
) -> AsyncIterator[AsyncHttpClient]:
    """Convenience for one-shot scripts and tests; in the running app each
    provider holds its client for the process lifetime."""
    client = AsyncHttpClient(provider_name, base_url, default_headers=headers)
    try:
        yield client
    finally:
        await client.aclose()
