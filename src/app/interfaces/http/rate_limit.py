"""Bounded per-client rate limiting.

A fixed window, a request budget, and a hard cap on how many clients are
tracked so the limiter cannot itself become the memory leak.

Deliberate: in-process, single worker. Two uvicorn workers get two
independent budgets; move the counter to Redis if that ever matters.
"""

from collections import OrderedDict
from math import ceil
from threading import Lock
from time import monotonic
from typing import final

from fastapi import Request
from starlette.requests import HTTPConnection

from app.application.policies import RateLimitPolicy
from app.domain import RateLimited
from app.interfaces.http.constants import RateLimitValues


@final
class FixedWindowLimiter:
    """Counts requests per client inside a rolling fixed window."""

    def __init__(self, policy: RateLimitPolicy) -> None:
        """Create an empty limiter bound to its budget.

        :param policy: Window size, budget and tracking ceiling.
        :type policy: app.application.policies.RateLimitPolicy
        """
        self._policy: RateLimitPolicy = policy
        self._windows: OrderedDict[str, tuple[int, float]] = OrderedDict()
        self._lock: Lock = Lock()

    def reset(self) -> None:
        """Drop every tracked window. Exists for tests."""
        with self._lock:
            self._windows.clear()

    def consume(self, key: str) -> float:
        """Count one request against ``key`` and report any overflow.

        :param key: Client identity the budget is tracked against.
        :type key: str
        :returns: Seconds to wait when over budget, zero when allowed.
        :rtype: float
        """
        now: float = monotonic()
        with self._lock:
            count, ends_at = self._windows.get(
                key, RateLimitValues.empty_window
            )
            if ends_at <= now:
                count, ends_at = 0, now + self._policy.window_seconds
            count += 1
            self._windows[key] = (count, ends_at)
            self._windows.move_to_end(key)
            # Evicting the least recently seen client bounds the map, so
            # a flood of one-shot keys costs O(capacity), never
            # O(distinct callers).
            excess: int = len(self._windows) - self._policy.max_tracked_clients
            for _ in range(max(0, excess)):
                self._windows.popitem(last=False)
        if count > self._policy.max_requests:
            return ends_at - now
        return RateLimitValues.no_wait

    def check(self, key: str) -> None:
        """Enforce the budget for one caller.

        :param key: Client identity.
        :type key: str
        :raises app.domain.errors.RateLimited: When over budget.
        """
        if self._policy.max_requests <= 0:
            return
        if (wait := self.consume(key)) > RateLimitValues.no_wait:
            raise RateLimited(
                retry_after_seconds=max(
                    self._policy.minimum_retry_after_seconds, ceil(wait)
                )
            )


def _client_of(connection: HTTPConnection) -> str:
    return (
        connection.client.host
        if connection.client
        else RateLimitValues.unknown_client
    )


def enforce_rate_limit(request: Request) -> None:
    """FastAPI dependency applying the wired limiter to a request.

    :param request: Incoming request, carrying the wired limiter.
    :type request: fastapi.Request
    :raises app.domain.errors.RateLimited: When over budget.
    """
    limiter: FixedWindowLimiter = request.app.state.limiter
    limiter.check(_client_of(request))


def within_rate_limit(connection: HTTPConnection) -> bool:
    """Count one connection against its budget and report the verdict.

    The socket counterpart of :func:`enforce_rate_limit`: an upgrade
    cannot carry a 429, so this answers rather than raises and the
    caller closes the connection. Sharing the limiter with the HTTP
    routes is the point - a subscription is as much a request as a
    ``POST`` is, and it costs a poll every quarter second thereafter.

    :param connection: The upgrade request, carrying the wired limiter.
    :type connection: starlette.requests.HTTPConnection
    :returns: ``True`` when the caller is within budget.
    :rtype: bool
    """
    limiter: FixedWindowLimiter = connection.app.state.limiter
    try:
        limiter.check(_client_of(connection))
    except RateLimited:
        return False
    return True


__all__: list[str] = [
    "FixedWindowLimiter",
    "enforce_rate_limit",
    "within_rate_limit",
]
