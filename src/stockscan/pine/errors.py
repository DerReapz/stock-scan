"""Pine interpreter errors."""

from __future__ import annotations


class PineError(Exception):
    def __init__(self, message: str, line: int | None = None):
        self.line = line
        super().__init__(f"line {line}: {message}" if line else message)


class PineSyntaxError(PineError):
    pass


class PineUnsupportedError(PineError):
    """A construct outside the supported Pine v6 subset — never guess silently."""


class PineRuntimeError(PineError):
    pass
