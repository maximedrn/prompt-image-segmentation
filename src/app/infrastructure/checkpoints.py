"""Remote checkpoint downloads.

Isolated so we can swap the storage backend (HuggingFace, S3, local
mount) behind a single interface without touching the model loaders.
"""

from pathlib import Path

from requests import Response, get

_CHUNK_SIZE: int = 1 << 20


def download_file(url: str, destination: Path) -> Path:
    """Stream ``url`` to ``destination``. Idempotent when file exists.

    :param url: HTTP(S) URL to fetch.
    :type url: str
    :param destination: Filesystem path where the payload is written.
        Parent directories are created if missing.
    :type destination: pathlib.Path
    :returns: ``destination`` (for chaining).
    :rtype: pathlib.Path
    :raises requests.HTTPError: On any non-2xx response.
    """
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    response: Response = get(url, stream=True, timeout=60)
    response.raise_for_status()
    with destination.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=_CHUNK_SIZE):
            if chunk:
                handle.write(chunk)
    return destination


__all__: list[str] = ["download_file"]
