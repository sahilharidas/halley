"""The unit types a model writes as annotations.

These classes are never instantiated. A model method declares ``-> Count`` and reads other
variables that pyright sees as ``Count``, ``Prob[Monthly]`` and so on; the operator overloads
below are the rule table of :mod:`halley.ir.units` written for the type checker, so a unit
mistake shows in the editor before a build. At run time the tracer returns symbolic expressions
in their place and the checker applies the rule table to the graph.
"""

from __future__ import annotations

from typing import Any, overload

from halley.errors import Loc, UnitError
from halley.ir.units import (
    BOOL,
    COUNT,
    FACTOR,
    INT,
    MONEY,
    MONEY_PP,
    PROB,
    PROB_ANNUAL,
    PROB_MONTHLY,
    RATE_ANNUAL,
    RATE_MONTHLY,
    Unit,
)


def _static() -> Any:
    raise TypeError("Halley unit types are annotations; they are never instantiated")


class Annual:
    """Period tag: per year."""


class Monthly:
    """Period tag: per month."""


class Bool:
    """A condition: comparisons produce it and where() consumes it."""

    def __and__(self, other: Bool | bool) -> Bool:
        return _static()

    def __or__(self, other: Bool | bool) -> Bool:
        return _static()

    def __invert__(self) -> Bool:
        return _static()


class Int:
    """An exact integer: the time index, ages, durations and table keys."""

    def __add__(self, other: Int | int) -> Int:
        return _static()

    def __radd__(self, other: int) -> Int:
        return _static()

    def __sub__(self, other: Int | int) -> Int:
        return _static()

    def __rsub__(self, other: int) -> Int:
        return _static()

    def __mul__(self, other: Int | int) -> Int:
        return _static()

    def __rmul__(self, other: int) -> Int:
        return _static()

    def __floordiv__(self, other: Int | int) -> Int:
        return _static()

    def __mod__(self, other: Int | int) -> Int:
        return _static()

    def __truediv__(self, other: Int | int | float) -> Factor:
        return _static()

    def __rtruediv__(self, other: int | float) -> Factor:
        return _static()

    def __neg__(self) -> Int:
        return _static()

    def __lt__(self, other: Int | int | float) -> Bool:
        return _static()

    def __le__(self, other: Int | int | float) -> Bool:
        return _static()

    def __gt__(self, other: Int | int | float) -> Bool:
        return _static()

    def __ge__(self, other: Int | int | float) -> Bool:
        return _static()

    def __eq__(self, other: object) -> Bool:  # pyright: ignore[reportIncompatibleMethodOverride]
        return _static()

    def __ne__(self, other: object) -> Bool:  # pyright: ignore[reportIncompatibleMethodOverride]
        return _static()

    __hash__ = object.__hash__


class Factor:
    """A dimensionless multiplier: discount and inflation factors, proportions."""

    def __add__(self, other: Factor | float | int) -> Factor:
        return _static()

    def __radd__(self, other: float | int) -> Factor:
        return _static()

    def __sub__(self, other: Factor | float | int) -> Factor:
        return _static()

    def __rsub__(self, other: float | int) -> Factor:
        return _static()

    @overload
    def __mul__(self, other: Factor | float | int) -> Factor: ...
    @overload
    def __mul__(self, other: Money) -> Money: ...
    @overload
    def __mul__(self, other: MoneyPerPolicy) -> MoneyPerPolicy: ...
    @overload
    def __mul__(self, other: Count) -> Count: ...
    @overload
    def __mul__[P: (Annual, Monthly)](self, other: Prob[P]) -> Prob[P]: ...
    @overload
    def __mul__[P: (Annual, Monthly)](self, other: Rate[P]) -> Rate[P]: ...
    def __mul__(self, other: object) -> object:
        return _static()

    def __rmul__(self, other: float | int) -> Factor:
        return _static()

    def __truediv__(self, other: Factor | float | int) -> Factor:
        return _static()

    def __rtruediv__(self, other: float | int) -> Factor:
        return _static()

    def __pow__(self, other: float | int | Int) -> Factor:
        return _static()

    def __neg__(self) -> Factor:
        return _static()

    def __lt__(self, other: Factor | float | int) -> Bool:
        return _static()

    def __le__(self, other: Factor | float | int) -> Bool:
        return _static()

    def __gt__(self, other: Factor | float | int) -> Bool:
        return _static()

    def __ge__(self, other: Factor | float | int) -> Bool:
        return _static()

    def __eq__(self, other: object) -> Bool:  # pyright: ignore[reportIncompatibleMethodOverride]
        return _static()

    def __ne__(self, other: object) -> Bool:  # pyright: ignore[reportIncompatibleMethodOverride]
        return _static()

    __hash__ = object.__hash__


Fraction = Factor
"""A proportion, such as a commission rate: the same dimensionless unit as ``Factor``."""


class Money:
    """A currency amount for the whole in-force count of the model point."""

    def __add__(self, other: Money) -> Money:
        return _static()

    def __sub__(self, other: Money) -> Money:
        return _static()

    def __mul__(self, other: Factor | Rate[Monthly] | float | int) -> Money:
        return _static()

    def __rmul__(self, other: float | int) -> Money:
        return _static()

    @overload
    def __truediv__(self, other: Count) -> MoneyPerPolicy: ...
    @overload
    def __truediv__(self, other: Money) -> Factor: ...
    @overload
    def __truediv__(self, other: Factor | float | int) -> Money: ...
    def __truediv__(self, other: object) -> object:
        return _static()

    def __neg__(self) -> Money:
        return _static()

    def __lt__(self, other: Money | float | int) -> Bool:
        return _static()

    def __le__(self, other: Money | float | int) -> Bool:
        return _static()

    def __gt__(self, other: Money | float | int) -> Bool:
        return _static()

    def __ge__(self, other: Money | float | int) -> Bool:
        return _static()

    def __eq__(self, other: object) -> Bool:  # pyright: ignore[reportIncompatibleMethodOverride]
        return _static()

    def __ne__(self, other: object) -> Bool:  # pyright: ignore[reportIncompatibleMethodOverride]
        return _static()

    __hash__ = object.__hash__


class MoneyPerPolicy:
    """A currency amount per policy: a premium, a sum assured, an expense per policy."""

    def __add__(self, other: MoneyPerPolicy) -> MoneyPerPolicy:
        return _static()

    def __sub__(self, other: MoneyPerPolicy) -> MoneyPerPolicy:
        return _static()

    @overload
    def __mul__(self, other: Count) -> Money: ...
    @overload
    def __mul__(
        self, other: Factor | Rate[Monthly] | Prob[Monthly] | float | int
    ) -> MoneyPerPolicy: ...
    def __mul__(self, other: object) -> object:
        return _static()

    def __rmul__(self, other: float | int) -> MoneyPerPolicy:
        return _static()

    @overload
    def __truediv__(self, other: MoneyPerPolicy) -> Factor: ...
    @overload
    def __truediv__(self, other: Factor | float | int) -> MoneyPerPolicy: ...
    def __truediv__(self, other: object) -> object:
        return _static()

    def __neg__(self) -> MoneyPerPolicy:
        return _static()

    def __lt__(self, other: MoneyPerPolicy | float | int) -> Bool:
        return _static()

    def __le__(self, other: MoneyPerPolicy | float | int) -> Bool:
        return _static()

    def __gt__(self, other: MoneyPerPolicy | float | int) -> Bool:
        return _static()

    def __ge__(self, other: MoneyPerPolicy | float | int) -> Bool:
        return _static()

    def __eq__(self, other: object) -> Bool:  # pyright: ignore[reportIncompatibleMethodOverride]
        return _static()

    def __ne__(self, other: object) -> Bool:  # pyright: ignore[reportIncompatibleMethodOverride]
        return _static()

    __hash__ = object.__hash__


class Count:
    """A number of policies, real-valued: in force, deaths, lapses."""

    def __add__(self, other: Count) -> Count:
        return _static()

    def __sub__(self, other: Count) -> Count:
        return _static()

    @overload
    def __mul__(self, other: Prob[Monthly]) -> Count: ...
    @overload
    def __mul__(self, other: MoneyPerPolicy) -> Money: ...
    @overload
    def __mul__(self, other: Factor | float | int) -> Count: ...
    def __mul__(self, other: object) -> object:
        return _static()

    def __rmul__(self, other: float | int) -> Count:
        return _static()

    @overload
    def __truediv__(self, other: Count) -> Factor: ...
    @overload
    def __truediv__(self, other: Factor | float | int) -> Count: ...
    def __truediv__(self, other: object) -> object:
        return _static()

    def __neg__(self) -> Count:
        return _static()

    def __lt__(self, other: Count | float | int) -> Bool:
        return _static()

    def __le__(self, other: Count | float | int) -> Bool:
        return _static()

    def __gt__(self, other: Count | float | int) -> Bool:
        return _static()

    def __ge__(self, other: Count | float | int) -> Bool:
        return _static()

    def __eq__(self, other: object) -> Bool:  # pyright: ignore[reportIncompatibleMethodOverride]
        return _static()

    def __ne__(self, other: object) -> Bool:  # pyright: ignore[reportIncompatibleMethodOverride]
        return _static()

    __hash__ = object.__hash__


class Prob[P: (Annual, Monthly)]:
    """A probability in [0, 1], checked at run time. ``Prob[Annual]`` and ``Prob[Monthly]``
    carry the period; :func:`~halley.authoring.to_monthly_prob` converts between them."""

    def __add__(self, other: Prob[P] | float | int) -> Prob[P]:
        return _static()

    def __radd__(self, other: float | int) -> Prob[P]:
        return _static()

    def __sub__(self, other: Prob[P] | float | int) -> Prob[P]:
        return _static()

    def __rsub__(self, other: float | int) -> Prob[P]:
        return _static()

    @overload
    def __mul__(self, other: Factor | float | int) -> Prob[P]: ...
    @overload
    def __mul__(self: Prob[Monthly], other: Count) -> Count: ...
    @overload
    def __mul__(self: Prob[Monthly], other: MoneyPerPolicy) -> MoneyPerPolicy: ...
    def __mul__(self, other: object) -> object:
        return _static()

    def __rmul__(self, other: float | int) -> Prob[P]:
        return _static()

    def __truediv__(self, other: Factor | float | int) -> Prob[P]:
        return _static()

    def __pow__(self, other: float | int) -> Prob[P]:
        return _static()

    def __neg__(self) -> Prob[P]:
        return _static()

    def __lt__(self, other: Prob[P] | float | int) -> Bool:
        return _static()

    def __le__(self, other: Prob[P] | float | int) -> Bool:
        return _static()

    def __gt__(self, other: Prob[P] | float | int) -> Bool:
        return _static()

    def __ge__(self, other: Prob[P] | float | int) -> Bool:
        return _static()

    def __eq__(self, other: object) -> Bool:  # pyright: ignore[reportIncompatibleMethodOverride]
        return _static()

    def __ne__(self, other: object) -> Bool:  # pyright: ignore[reportIncompatibleMethodOverride]
        return _static()

    __hash__ = object.__hash__


class Rate[P: (Annual, Monthly)]:
    """An effective interest or inflation rate per year or per month. ``1 + rate`` is a
    :class:`Factor`; :func:`~halley.authoring.to_monthly` converts between periods."""

    def __add__(self, other: Rate[P]) -> Rate[P]:
        return _static()

    def __radd__(self, other: float | int) -> Factor:
        return _static()

    def __sub__(self, other: Rate[P]) -> Rate[P]:
        return _static()

    @overload
    def __mul__(self, other: Factor | float | int) -> Rate[P]: ...
    @overload
    def __mul__(self: Rate[Monthly], other: Money) -> Money: ...
    @overload
    def __mul__(self: Rate[Monthly], other: MoneyPerPolicy) -> MoneyPerPolicy: ...
    def __mul__(self, other: object) -> object:
        return _static()

    def __rmul__(self, other: float | int) -> Rate[P]:
        return _static()

    def __truediv__(self, other: Factor | float | int) -> Rate[P]:
        return _static()

    def __neg__(self) -> Rate[P]:
        return _static()

    def __lt__(self, other: Rate[P] | float | int) -> Bool:
        return _static()

    def __le__(self, other: Rate[P] | float | int) -> Bool:
        return _static()

    def __gt__(self, other: Rate[P] | float | int) -> Bool:
        return _static()

    def __ge__(self, other: Rate[P] | float | int) -> Bool:
        return _static()

    def __eq__(self, other: object) -> Bool:  # pyright: ignore[reportIncompatibleMethodOverride]
        return _static()

    def __ne__(self, other: object) -> Bool:  # pyright: ignore[reportIncompatibleMethodOverride]
        return _static()

    __hash__ = object.__hash__


class PerPolicy:
    """``PerPolicy[Money]`` is another name for :class:`MoneyPerPolicy`."""

    def __class_getitem__(cls, item: object) -> type[MoneyPerPolicy]:
        if item is Money:
            return MoneyPerPolicy
        raise TypeError("PerPolicy[...] takes Money")


_SIMPLE: dict[object, Unit] = {
    Money: MONEY,
    MoneyPerPolicy: MONEY_PP,
    Count: COUNT,
    Prob: PROB,
    Factor: FACTOR,
    Int: INT,
    Bool: BOOL,
}


def unit_from_annotation(annotation: object, loc: Loc | None = None) -> Unit:
    """Resolve a return annotation (a class above, possibly parameterised) to its tag."""
    origin = getattr(annotation, "__origin__", None)
    args: tuple[object, ...] = getattr(annotation, "__args__", ())
    if origin is Prob or origin is Rate:
        period = args[0] if args else None
        tags = (PROB_ANNUAL, PROB_MONTHLY) if origin is Prob else (RATE_ANNUAL, RATE_MONTHLY)
        if period is Annual:
            return tags[0]
        if period is Monthly:
            return tags[1]
        name = "Prob" if origin is Prob else "Rate"
        raise UnitError(f"{name} takes Annual or Monthly, not {period!r}", loc)
    if annotation in _SIMPLE:
        return _SIMPLE[annotation]
    if annotation is Rate:
        raise UnitError("Rate needs a period: Rate[Annual] or Rate[Monthly]", loc)
    raise UnitError(
        f"{annotation!r} is not a Halley unit; use Money, MoneyPerPolicy, Count, "
        "Prob[Annual], Prob[Monthly], Rate[Annual], Rate[Monthly], Factor, Int or Bool",
        loc,
    )
