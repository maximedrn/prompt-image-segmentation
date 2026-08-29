"""What makes two submissions the same request.

An idempotency key on its own cannot tell a retry from a mistake: the
caller controls it, and reusing one for different work is a thing that
happens. The hash is what distinguishes them, so what goes into it is
the whole decision.

Included: the image bytes, the prompt, the backend, the person mode and
the segmentation options. Change any of those and you are asking for a
different segmentation, which is a conflict rather than a retry.

Excluded: the callback URL. It is the only field a caller can honestly
change between attempts -- a first try aimed at a receiver that has
since moved, a retry aimed at the new one -- and treating that as a
conflict would refuse the retry it exists to allow. The consequence is
deliberate and worth naming: a replay answers with the first job, so the
*first* callback URL is the one that gets the result.

The image is hashed as the bytes that arrived rather than as a decoded
array. Decoding is lossy across library versions in ways that would make
the same upload hash differently after an upgrade, and the bytes are
what the caller actually sent.
"""

from dataclasses import asdict
from hashlib import sha256
from json import dumps
from typing import Final, final

from app.application.jobs import JobPayload
from app.infrastructure.imaging.types import TextEncoding

#: Separates the image digest from the settings digest. A byte that
#: cannot occur in either, so no pair of different inputs can be
#: concatenated into the same string.
_SEPARATOR: Final[bytes] = b"\x00"


@final
class HashRules:
    """How the digest is built, named so the contract is readable."""

    #: Sorted, so two payloads whose options were built in a different
    #: order still hash the same. A dictionary's insertion order is not
    #: part of what a caller asked for.
    sort_keys: bool = True
    #: No whitespace: it is not information and it varies with the
    #: serializer's defaults.
    separators: tuple[str, str] = (",", ":")


def _settings_of(payload: JobPayload) -> str:
    """Render everything about a request except its image.

    :param payload: The accepted request.
    :type payload: app.application.jobs.JobPayload
    :returns: A canonical JSON document.
    :rtype: str
    """
    return dumps(
        {
            "prompt": payload.prompt,
            "backend": payload.backend,
            "person_mode": payload.person_mode,
            "options": asdict(payload.options),
        },
        sort_keys=HashRules.sort_keys,
        separators=HashRules.separators,
    )


def request_hash(payload: JobPayload) -> str:
    """Digest the parts of a request that make it the request it is.

    :param payload: The accepted request.
    :type payload: app.application.jobs.JobPayload
    :returns: A hex SHA-256 digest.
    :rtype: str
    """
    digest = sha256()
    digest.update(sha256(payload.image).digest())
    digest.update(_SEPARATOR)
    digest.update(_settings_of(payload).encode(TextEncoding.UTF8))
    return digest.hexdigest()


__all__: list[str] = ["HashRules", "request_hash"]
