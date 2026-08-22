"""Closed sets the configuration is written against."""

from enum import StrEnum, unique


@unique
class AuthMode(StrEnum):
    """How the API authenticates callers."""

    BASIC = "basic"
    NONE = "none"


__all__: list[str] = ["AuthMode"]
