"""Outbound deliveries: what is signed, and where they refuse to go.

No network: the address guard resolves real names, and the delivery
tests drive the notifier against a stand-in transport. A test that had
to reach the internet would be a test nobody trusts offline.
"""

# pytest collects the test functions by name; nothing in the module
# calls them, which is what ``allow-global-unused-variables`` flags.
# pylint: disable=unused-variable

from asyncio import run
from dataclasses import replace
from hashlib import sha256
from hmac import new as hmac_new
from json import dumps, loads
from typing import Final

import pytest
from httpx import AsyncClient, ConnectError, MockTransport, Request, Response

from app.application.jobs import JobResult
from app.application.policies import WebhookPolicy
from app.domain import Job, JobState, queued, start, succeed
from app.infrastructure.webhooks.constants import (
    DeliveryField,
    WebhookRules,
)
from app.infrastructure.webhooks.notifier import (
    SignedWebhookNotifier,
    acceptable,
    sign,
)
from app.interfaces.http.schemas import JsonValue
from tests.conftest import segment_body

SECRET: Final[str] = "a shared secret"
TIMESTAMP: Final[int] = 1_700_000_000
ATTEMPTS: Final[int] = 3
TIMEOUT_SECONDS: Final[float] = 1.0
NO_BACKOFF: Final[float] = 0.0
JOB_ID: Final[str] = "job-1"
ACCEPTED_AT: Final[float] = 1000.0
STARTED_AT: Final[float] = 1001.0
FINISHED_AT: Final[float] = 1002.0
PUBLIC_URL: Final[str] = "https://example.com/hook"
#: The one field a receiver branches on.
SUCCEEDED_STATE: Final[str] = JobState.SUCCEEDED


#: The policy every test starts from: same rules, no backoff to wait on.
BASE_POLICY: Final[WebhookPolicy] = WebhookPolicy(
    signing_secret=SECRET,
    timeout_seconds=TIMEOUT_SECONDS,
    max_attempts=ATTEMPTS,
    initial_backoff_seconds=NO_BACKOFF,
)


def _policy(
    *, allow_insecure: bool = False, allow_private_hosts: bool = False
) -> WebhookPolicy:
    """Build a policy that retries fast enough for a test.

    ``replace`` rather than a mapping of field names: the policy is a
    dataclass, so the one field a test ever varies is named as one.

    :param allow_insecure: Whether plain http is accepted.
    :type allow_insecure: bool
    :param allow_private_hosts: Whether internal addresses are accepted.
    :type allow_private_hosts: bool
    :returns: The policy.
    :rtype: app.application.policies.WebhookPolicy
    """
    return replace(
        BASE_POLICY,
        allow_insecure=allow_insecure,
        allow_private_hosts=allow_private_hosts,
    )


def _finished() -> Job:
    """Build a job that has something to announce.

    :returns: A succeeded job.
    :rtype: app.domain.Job
    """
    return succeed(start(queued(JOB_ID, ACCEPTED_AT), STARTED_AT), FINISHED_AT)


def test_the_signature_covers_the_body_and_its_timestamp() -> None:
    """A captured delivery cannot be replayed under a fresh stamp."""
    body: bytes = dumps({DeliveryField.identifier: JOB_ID}).encode()
    expected: str = hmac_new(
        SECRET.encode(),
        f"{TIMESTAMP}.".encode() + body,
        sha256,
    ).hexdigest()

    assert (
        sign(body, TIMESTAMP, SECRET) == f"{WebhookRules.algorithm}={expected}"
    )
    # Same body, different second: a different signature.
    assert sign(body, TIMESTAMP + 1, SECRET) != sign(body, TIMESTAMP, SECRET)
    # Same everything, wrong secret: a different signature.
    assert sign(body, TIMESTAMP, "not the secret") != sign(
        body, TIMESTAMP, SECRET
    )


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/hook",
        "https://localhost/hook",
        "https://127.0.0.1/hook",
        "https://10.0.0.1/hook",
        "https://192.168.1.10/hook",
        # The cloud metadata endpoint, which is the whole reason this
        # guard exists.
        "https://169.254.169.254/latest/meta-data",
        "file:///etc/passwd",
        "https://this-name-does-not-resolve.invalid/hook",
    ],
)
def test_internal_and_insecure_destinations_are_refused(url: str) -> None:
    """A caller cannot aim the service at what only it can reach.

    :param url: Destination under test.
    :type url: str
    """
    assert run(acceptable(url, _policy())) is False


def test_a_public_https_destination_is_accepted() -> None:
    """The guard refuses the dangerous, not everything."""
    assert run(acceptable(PUBLIC_URL, _policy())) is True


def test_a_private_host_is_reached_only_when_an_operator_says_so() -> None:
    """The escape hatch a single-network stack needs, and its default.

    The guard exists to stop a caller aiming the service at what only
    it can reach. An operator who runs the whole stack on one private
    network and says so is the case that is not.
    """
    assert run(acceptable("https://127.0.0.1/hook", _policy())) is False
    assert (
        run(
            acceptable(
                "https://127.0.0.1/hook", _policy(allow_private_hosts=True)
            )
        )
        is True
    )


def test_plain_http_is_accepted_only_when_an_operator_says_so() -> None:
    """Inside a private network a receiver may have no certificate."""
    relaxed: WebhookPolicy = _policy(allow_insecure=True)
    assert run(acceptable("http://example.com/hook", relaxed)) is True


def test_a_delivery_carries_the_outcome_and_its_signature() -> None:
    """What the receiver gets is what it can verify."""
    seen: list[Request] = []

    def handle(request: Request) -> Response:
        """Accept one delivery and remember it.

        :param request: The delivery.
        :type request: httpx.Request
        :returns: An empty success.
        :rtype: httpx.Response
        """
        seen.append(request)
        return Response(200)

    notifier: SignedWebhookNotifier = SignedWebhookNotifier(
        AsyncClient(transport=MockTransport(handle)), _policy()
    )
    job: Job = _finished()
    run(notifier.notify(job, JobResult(body=segment_body()), PUBLIC_URL))

    assert len(seen) == 1
    delivered: Request = seen[0]
    stamp: int = int(delivered.headers[WebhookRules.timestamp_header])
    assert delivered.headers[WebhookRules.signature_header] == sign(
        delivered.content, stamp, SECRET
    )
    document: dict[str, JsonValue] = loads(delivered.content)
    assert document[DeliveryField.state] == SUCCEEDED_STATE


def test_a_refusing_receiver_is_retried_and_then_left_alone() -> None:
    """Retries are bounded: a broken receiver is not a broken job."""
    attempts: list[Request] = []

    def handle(request: Request) -> Response:
        """Refuse every delivery.

        :param request: The delivery.
        :type request: httpx.Request
        :returns: A server error.
        :rtype: httpx.Response
        """
        attempts.append(request)
        return Response(500)

    notifier: SignedWebhookNotifier = SignedWebhookNotifier(
        AsyncClient(transport=MockTransport(handle)), _policy()
    )
    run(
        notifier.notify(
            _finished(), JobResult(body=segment_body()), PUBLIC_URL
        )
    )

    assert len(attempts) == ATTEMPTS


def test_a_failed_delivery_never_raises() -> None:
    """The job is already stored; the push is best effort."""

    def handle(request: Request) -> Response:
        """Fail the way an unreachable host does.

        :param request: The delivery.
        :type request: httpx.Request
        :raises ConnectError: Always.
        """
        del request
        raise ConnectError("no route to host")

    notifier: SignedWebhookNotifier = SignedWebhookNotifier(
        AsyncClient(transport=MockTransport(handle)), _policy()
    )
    # No assertion beyond returning: raising here would fail a job that
    # succeeded.
    run(
        notifier.notify(
            _finished(), JobResult(body=segment_body()), PUBLIC_URL
        )
    )


def test_a_destination_that_turned_private_is_not_delivered_to() -> None:
    """The address guard runs again at delivery, not only at acceptance.

    Minutes pass between the two, and the name belongs to the caller:
    answering publicly while the job is queued and with a loopback once
    it finishes is the whole of a DNS rebinding attack.
    """
    attempts: list[Request] = []

    def handle(request: Request) -> Response:
        """Record a delivery that should never have been made.

        :param request: The delivery.
        :type request: httpx.Request
        :returns: An empty success.
        :rtype: httpx.Response
        """
        attempts.append(request)
        return Response(200)

    notifier: SignedWebhookNotifier = SignedWebhookNotifier(
        AsyncClient(transport=MockTransport(handle)),
        _policy(allow_insecure=True),
    )
    run(
        notifier.notify(
            _finished(), JobResult(body=segment_body()), "http://127.0.0.1/x"
        )
    )
    run(
        notifier.notify(
            _finished(),
            JobResult(body=segment_body()),
            "http://169.254.169.254/x",
        )
    )

    assert not attempts
