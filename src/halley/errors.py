"""Errors raised by the build (tracing, checking), the loader and the run.

Every error that concerns a user's model carries a :class:`Loc` pointing at the model source,
never at generated code (project rule).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Loc:
    """A position in the user's model source."""

    file: str
    line: int
    function: str

    def __str__(self) -> str:
        return f"{self.function} ({self.file}:{self.line})"


class HalleyError(Exception):
    """Base class for every error Halley raises on purpose."""


class BuildError(HalleyError):
    """The model cannot be built: units, time references, order, cycles or boundary values."""

    def __init__(self, message: str, loc: Loc | None = None) -> None:
        self.message = message
        self.loc = loc
        super().__init__(f"{message}\n  at {loc}" if loc is not None else message)


class UnitError(BuildError):
    """A unit or dimension mismatch in a formula (D11: always an error)."""


class LoadError(HalleyError):
    """Input data breaks its contract: schema, ranges, table coverage or scenario coverage."""


class RunError(HalleyError):
    """A run-time check failed: a probability outside [0, 1], a lookup key outside its table,
    a non-finite result, a built-in invariant."""

    def __init__(self, message: str, loc: Loc | None = None) -> None:
        self.message = message
        self.loc = loc
        super().__init__(f"{message}\n  at {loc}" if loc is not None else message)


class BuildWarning(UserWarning):
    """A build finding that does not stop the build: an unused variable."""
