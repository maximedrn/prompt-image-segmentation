"""Rate-limit window arithmetic. No models, no HTTP."""

import pytest

from app.application.policies import RateLimitPolicy
from app.domain import RateLimited
from app.interfaces.http.rate_limit import FixedWindowLimiter


def _limiter(
    max_requests: int = 3,
    window_seconds: float = 60.0,
    max_tracked_clients: int = 100,
) -> FixedWindowLimiter:
    """Build a limiter with an explicit budget.

    :param max_requests: Requests allowed per window.
    :type max_requests: int
    :param window_seconds: Window length.
    :type window_seconds: float
    :param max_tracked_clients: Tracking ceiling.
    :type max_tracked_clients: int
    :returns: A fresh limiter.
    :rtype: app.interfaces.http.rate_limit.FixedWindowLimiter
    """
    return FixedWindowLimiter(
        RateLimitPolicy(
            max_requests=max_requests,
            window_seconds=window_seconds,
            max_tracked_clients=max_tracked_clients,
        )
    )


def test_budget_is_enforced_per_key() -> None:
    """A key is allowed exactly its budget, then refused."""
    limiter: FixedWindowLimiter = _limiter()
    assert [limiter.consume("a") for _ in range(3)] == [0.0, 0.0, 0.0]
    assert limiter.consume("a") > 0.0
    # A different caller keeps its own budget.
    assert limiter.consume("b") == 0.0


def test_window_expiry_resets_the_budget() -> None:
    """A zero-length window has elapsed by the next call."""
    limiter: FixedWindowLimiter = _limiter(max_requests=1, window_seconds=0.0)
    assert all(limiter.consume("a") == 0.0 for _ in range(5))


def test_open_window_is_not_shortened_retroactively() -> None:
    """A live window keeps the length it was opened with."""
    limiter: FixedWindowLimiter = _limiter(max_requests=1)
    assert limiter.consume("a") == 0.0
    assert limiter.consume("a") > 0.0


def test_tracked_clients_are_bounded() -> None:
    """A flood of one-shot keys costs O(capacity), not O(callers)."""
    limiter: FixedWindowLimiter = _limiter(
        max_requests=1, max_tracked_clients=10
    )
    for index in range(500):
        limiter.consume(f"client-{index}")
    # pylint: disable=protected-access
    assert len(limiter._windows) <= 10


def test_check_raises_the_domain_failure() -> None:
    """Over budget surfaces as the typed failure, with a retry delay."""
    limiter: FixedWindowLimiter = _limiter(max_requests=1)
    limiter.check("a")
    with pytest.raises(RateLimited) as raised:
        limiter.check("a")
    assert raised.value.retry_after_seconds >= 1


def test_zero_budget_disables_the_limiter() -> None:
    """A non-positive budget means the limiter never refuses."""
    limiter: FixedWindowLimiter = _limiter(max_requests=0)
    for _ in range(50):
        limiter.check("a")
