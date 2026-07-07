"""Minimal HuggingFace Hub public API used by the app."""

# pylint: disable=locally-disabled
# pylint: disable=suppressed-message,useless-suppression
# pylint: disable=unused-variable,unused-argument

def hf_hub_download(
    repo_id: str,
    filename: str,
) -> str:
    """Download ``filename`` from ``repo_id`` and return its local path."""
