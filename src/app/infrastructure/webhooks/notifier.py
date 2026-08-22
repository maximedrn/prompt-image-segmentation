"""Signed webhook delivery.

Implements :class:`~app.application.capabilities.JobNotifier`.

A webhook is the one place this service makes an outbound request to an
address a caller chose, which is what makes it the one place that can be
turned into a probe of whatever the container can reach. Three rules
follow, and none of them is optional:

* the destination is resolved and refused if it lands on a private,
  loopback, link-local or otherwise internal address - at acceptance so
  the caller hears about it, and again at delivery because a name that
  answered publicly minutes ago can answer differently now;
* redirects are not followed, because a public host may redirect to a
  private one after the check has passed;
* the body is signed, so the receiver can tell our delivery from anyone
  else's post to the same URL.

Retries are bounded and spaced: a receiver that is down should not be
hammered, and a queue worker should not wait on it either.
"""

from asyncio import sleep
from hashlib import sha256
from hmac import new as hmac_new
from ipaddress import IPv4Address, IPv6Address, ip_address
from json import dumps
from logging import Logger, getLogger
from socket import AF_INET, AF_INET6, SOCK_STREAM, getaddrinfo
from time import time
from typing import Final, final
from urllib.parse import urlparse

from anyio import to_thread
from httpx import AsyncClient, HTTPError, Response

from app.application.jobs import JobResult
from app.application.policies import WebhookPolicy
from app.domain import Job
from app.infrastructure.imaging.types import TextEncoding
from app.infrastructure.webhooks.constants import (
    DeliveryField,
    DeliveryLog,
    WebhookRules,
)

_logger: Final[Logger] = getLogger(__name__)


def _public(address: IPv4Address | IPv6Address) -> bool:
    """Report whether an address is one the internet can route to.

    :param address: A resolved address.
    :type address: ipaddress.IPv4Address | ipaddress.IPv6Address
    :returns: ``True`` when it is neither internal nor special.
    :rtype: bool
    """
    return not (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )


def _reachable(host: str) -> bool:
    """Report whether a hostname resolves only to public addresses.

    Every resolved address is checked, not just the first: a name that
    answers with one public and one loopback address is exactly the
    trick this exists to refuse.

    :param host: Hostname or address from the caller's URL.
    :type host: str
    :returns: ``True`` when every resolved address is routable and
        public.
    :rtype: bool
    """
    try:
        # ``type``, not ``proto``: SOCK_STREAM is a socket type, and
        # passing it as a protocol number silently drops every AAAA
        # record - which would let an AAAA pointing at loopback through
        # on the strength of a public A record alone.
        resolved = getaddrinfo(host, None, type=SOCK_STREAM)
    except OSError:
        return False
    if not resolved:
        return False
    # Destructured rather than indexed: it is what tells both type
    # checkers which of ``getaddrinfo``'s address shapes this is.
    for family, _, _, _, socket_address in resolved:
        if family not in (AF_INET, AF_INET6):
            return False
        if not _public(ip_address(str(socket_address[WebhookRules.host]))):
            return False
    return True


def sign(body: bytes, timestamp: int, secret: str) -> str:
    """Return the signature a receiver should expect for one delivery.

    The timestamp is inside the signed material rather than beside it,
    so a captured delivery cannot be replayed with a fresh header.

    :param body: Exact bytes that will be sent.
    :type body: bytes
    :param timestamp: Unix seconds the delivery was signed at.
    :type timestamp: int
    :param secret: Shared signing secret.
    :type secret: str
    :returns: ``sha256=<hex digest>``.
    :rtype: str
    """
    signed: bytes = f"{timestamp}.".encode(TextEncoding.UTF8) + body
    digest: str = hmac_new(
        secret.encode(TextEncoding.UTF8), signed, sha256
    ).hexdigest()
    return f"{WebhookRules.algorithm}={digest}"


@final
class SignedWebhookNotifier:
    """Posts a finished job to the address its caller nominated."""

    def __init__(self, transport: AsyncClient, policy: WebhookPolicy) -> None:
        """Bind a client to the rules deliveries follow.

        :param transport: HTTP client, set not to follow redirects.
        :type transport: httpx.AsyncClient
        :param policy: Signing secret, timeout and retry bounds.
        :type policy: app.application.policies.WebhookPolicy
        """
        self._client: AsyncClient = transport
        self._policy: WebhookPolicy = policy

    async def notify(
        self, job: Job, result: JobResult | None, callback_url: str
    ) -> None:
        """Deliver one terminal state, retrying a few times.

        Delivery is best effort by design: polling and the socket both
        remain, so a caller is never left with only this.

        The destination is re-checked here even though acceptance
        already checked it. Minutes can pass between the two, and a name
        the caller controls can start answering with a loopback or
        metadata address in between - which is the whole of a DNS
        rebinding attack. The residual window, between this resolution
        and the one the client makes, is the price of not writing a
        custom transport.

        :param job: The finished job.
        :type job: app.domain.Job
        :param result: What it produced, if anything.
        :type result: app.application.jobs.JobResult | None
        :param callback_url: Where the caller asked to be told.
        :type callback_url: str
        """
        if not await acceptable(callback_url, self._policy):
            _logger.warning(
                DeliveryLog.refused,
                job.identifier,
            )
            return
        body: bytes = _document(job, result)
        stamp: int = int(time())
        headers: dict[str, str] = {
            WebhookRules.signature_header: sign(
                body, stamp, self._policy.signing_secret
            ),
            WebhookRules.timestamp_header: str(stamp),
            WebhookRules.content_type_header: WebhookRules.content_type,
        }
        delay: float = self._policy.initial_backoff_seconds
        for attempt in range(1, self._policy.max_attempts + 1):
            try:
                response: Response = await self._client.post(
                    callback_url,
                    content=body,
                    headers=headers,
                    timeout=self._policy.timeout_seconds,
                )
            except HTTPError as error:
                _logger.warning(
                    DeliveryLog.failed,
                    job.identifier,
                    type(error).__name__,
                    attempt,
                )
            else:
                if response.is_success:
                    return
                _logger.warning(
                    DeliveryLog.answered,
                    job.identifier,
                    response.status_code,
                    attempt,
                )
            if attempt < self._policy.max_attempts:
                await sleep(delay)
                delay *= WebhookRules.backoff_factor
        _logger.error(
            DeliveryLog.gave_up,
            job.identifier,
            self._policy.max_attempts,
        )


def _document(job: Job, result: JobResult | None) -> bytes:
    """Serialise what a receiver is told about a finished job.

    :param job: The finished job.
    :type job: app.domain.Job
    :param result: What it produced, if anything.
    :type result: app.application.jobs.JobResult | None
    :returns: The exact bytes that will be signed and sent.
    :rtype: bytes
    """
    return dumps({
        DeliveryField.identifier: job.identifier,
        DeliveryField.state: job.state.value,
        DeliveryField.error: job.error,
        DeliveryField.result: (
            None
            if result is None or result.body is None
            # The body is a model, and ``dumps`` only speaks builtins.
            else result.body.model_dump(mode=WebhookRules.json_mode)
        ),
    }).encode(TextEncoding.UTF8)


async def acceptable(url: str, policy: WebhookPolicy) -> bool:
    """Decide whether a caller's callback URL may be delivered to.

    Checked at acceptance rather than at delivery, so a caller learns
    its URL is refused while it is still holding the connection.

    :param url: Callback URL the caller supplied.
    :type url: str
    :param policy: Rules deliveries follow.
    :type policy: app.application.policies.WebhookPolicy
    :returns: ``True`` when the URL may be posted to.
    :rtype: bool
    """
    parsed = urlparse(url)
    if parsed.scheme not in _schemes(policy):
        return False
    if not parsed.hostname:
        return False
    # Resolution blocks, and this runs on the event loop that is also
    # accepting requests.
    return await to_thread.run_sync(_reachable, parsed.hostname)


def _schemes(policy: WebhookPolicy) -> tuple[str, ...]:
    """Return the URL schemes a delivery may use.

    :param policy: Rules deliveries follow.
    :type policy: app.application.policies.WebhookPolicy
    :returns: The accepted schemes.
    :rtype: tuple[str, ...]
    """
    if policy.allow_insecure:
        return (WebhookRules.https, WebhookRules.http)
    return (WebhookRules.https,)


def client() -> AsyncClient:
    """Build the client deliveries go out on.

    :returns: A client that never follows a redirect.
    :rtype: httpx.AsyncClient
    """
    return AsyncClient(follow_redirects=False)


__all__: list[str] = ["SignedWebhookNotifier", "acceptable", "client", "sign"]
