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


class _LocatedError(HalleyError):
    """An error with a message and the model source location it points at, both of which
    survive pickling. A run's worker processes (``RunSettings.workers``) send their errors to
    the calling process as pickles, and an exception pickles by default as its class called with
    its ``args``, which hold the formatted text rather than the message and the location."""

    def __init__(self, message: str, loc: Loc | None = None) -> None:
        self.message = message
        self.loc = loc
        super().__init__(f"{message}\n  at {loc}" if loc is not None else message)

    def __reduce__(self) -> tuple[object, ...]:
        # the class called with the message and the location, then the instance's attributes
        # (``__notes__`` among them), which an exception's default reduction keeps too
        return (type(self), (self.message, self.loc), dict(self.__dict__))


class BuildError(_LocatedError):
    """The model cannot be built: units, time references, order, cycles or boundary values."""


class UnitError(BuildError):
    """A unit or dimension mismatch in a formula (D11: always an error)."""


class LoadError(_LocatedError):
    """Input data breaks its contract: schema, ranges, table coverage or scenario coverage.

    ``loc`` is set when the failure is the model source's rather than the data's (a rule of
    the model that raises, or returns something other than its booleans): it points at the
    rule's declaration."""


class RunError(_LocatedError):
    """A run-time check failed: a probability outside [0, 1], a lookup key outside its table,
    a non-finite result, a built-in invariant. One raised in a worker process reaches the
    calling process with its message and location."""


class BuildWarning(UserWarning):
    """A build finding that does not stop the build: an unused variable."""
