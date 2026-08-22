"""Closed sets the pixel plumbing uses."""

from enum import StrEnum, unique


@unique
class TextEncoding(StrEnum):
    """Character encodings used when serialising to the wire."""

    UTF8 = "utf-8"


__all__: list[str] = ["TextEncoding"]
