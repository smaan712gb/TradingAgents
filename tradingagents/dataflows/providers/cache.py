"""Async cache decorator with Redis primary + filesystem fallback.

Usage:

    from .cache import cached

    class PolygonProvider:
        @cached(ttl_s=60, namespace="polygon.ohlcv")
        async def get_stock_data(self, symbol: str, ...): ...

The decorator builds the cache key from the (function name, namespace,
args, kwargs). Args must be JSON-serializable or a `__cache_key__` method
must be provided. Symbols, dates, and primitive numbers are fine.

Why this and not aiocache? aiocache pulls in too many transitive deps and
its serialization story conflicts with our Pydantic-heavy surfaces. This
file is small enough to own.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
import os
import pickle
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class _Backend:
    async def get(self, key: str) -> Optional[bytes]: ...
    async def set(self, key: str, value: bytes, ttl_s: int) -> None: ...


class _RedisBackend(_Backend):
    def __init__(self, url: str) -> None:
        # Lazy import — Redis is optional.
        import redis.asyncio as redis_asyncio  # type: ignore

        self._r = redis_asyncio.from_url(url, decode_responses=False)

    async def get(self, key: str) -> Optional[bytes]:
        try:
            return await self._r.get(key)
        except Exception as e:  # pragma: no cover
            logger.warning("redis cache get failed: %s", e)
            return None

    async def set(self, key: str, value: bytes, ttl_s: int) -> None:
        try:
            await self._r.set(key, value, ex=ttl_s)
        except Exception as e:  # pragma: no cover
            logger.warning("redis cache set failed: %s", e)


class _FsBackend(_Backend):
    """Filesystem fallback. Useful for local dev without Redis and as a
    persistent cache that survives Redis evictions."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        h = hashlib.sha256(key.encode()).hexdigest()
        return self._root / h[:2] / h

    async def get(self, key: str) -> Optional[bytes]:
        p = self._path(key)
        if not p.exists():
            return None
        # Honour TTL by checking mtime — header below stores expiry epoch.
        try:
            data = await asyncio.to_thread(p.read_bytes)
            if len(data) < 8:
                return None
            expiry_epoch = int.from_bytes(data[:8], "big")
            if expiry_epoch < int(datetime.utcnow().timestamp()):
                return None
            return data[8:]
        except OSError:  # pragma: no cover
            return None

    async def set(self, key: str, value: bytes, ttl_s: int) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        expiry = int(datetime.utcnow().timestamp()) + ttl_s
        payload = expiry.to_bytes(8, "big") + value
        await asyncio.to_thread(p.write_bytes, payload)


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


_BACKEND: Optional[_Backend] = None
_BACKEND_LOCK = asyncio.Lock()


async def _get_backend() -> _Backend:
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    async with _BACKEND_LOCK:
        if _BACKEND is not None:
            return _BACKEND
        url = os.getenv("REDIS_URL")
        if url:
            try:
                backend = _RedisBackend(url)
                # Touch it to validate; any failure falls back to FS.
                await backend.set("__healthcheck__", b"1", 5)
                _BACKEND = backend
                logger.info("provider cache: using Redis at %s", _redact(url))
                return _BACKEND
            except Exception as e:  # pragma: no cover
                logger.warning("Redis unavailable, falling back to filesystem cache: %s", e)
        root = Path(os.getenv("PROVIDER_CACHE_DIR", "/tmp/tradingagents-cache"))
        _BACKEND = _FsBackend(root)
        logger.info("provider cache: using filesystem at %s", root)
        return _BACKEND


def _redact(url: str) -> str:
    """Don't log passwords from REDIS_URL."""
    try:
        from urllib.parse import urlparse
        p = urlparse(url)
        if p.password:
            return url.replace(p.password, "***")
    except Exception:
        pass
    return url


# ---------------------------------------------------------------------------
# Key building
# ---------------------------------------------------------------------------


def _serialize_for_key(obj: Any) -> Any:
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return f"D:{obj}"
    if hasattr(obj, "__cache_key__"):
        return obj.__cache_key__()
    return obj


def _build_key(namespace: str, fn_name: str, args: tuple, kwargs: dict) -> str:
    payload = {
        "ns": namespace,
        "fn": fn_name,
        "args": [_serialize_for_key(a) for a in args[1:]],  # drop self
        "kwargs": {k: _serialize_for_key(v) for k, v in sorted(kwargs.items())},
    }
    return f"ta:cache:{hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()}"


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


def cached(
    ttl_s: int,
    namespace: str,
    *,
    # Pickle, NOT json.dumps(default=str): the json default silently turns
    # any non-JSON-native return (pandas DataFrame, frozen dataclasses like
    # FlowAlert/GammaLevel, Decimal) into its str() repr on write, so the
    # cache HIT hands back a *string* — e.g. `df.empty` then throws
    # "'str' object has no attribute 'empty'". Pickle round-trips all of
    # these losslessly. The cache is our own (Redis/FS), so pickle is safe.
    serialize: Callable[[Any], bytes] = lambda v: pickle.dumps(v),
    deserialize: Callable[[bytes], Any] = lambda b: pickle.loads(b),
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    def decorator(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            key = _build_key(namespace, fn.__name__, args, kwargs)
            backend = await _get_backend()
            cached_bytes = await backend.get(key)
            if cached_bytes is not None:
                try:
                    return deserialize(cached_bytes)
                except Exception:  # pragma: no cover
                    logger.warning("cache deserialize failed for %s; recomputing", namespace)
            result = await fn(*args, **kwargs)
            try:
                await backend.set(key, serialize(result), ttl_s)
            except Exception as e:  # pragma: no cover
                logger.warning("cache set failed for %s: %s", namespace, e)
            return result

        return wrapper

    return decorator
