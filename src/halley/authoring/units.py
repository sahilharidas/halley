"""The unit types a model writes as annotations.

These classes are never instantiated. A model method declares ``-> Count`` and reads other
variables that pyright sees as ``Count``, ``Prob[Monthly]`` and so on; the operator overloads
below are the rule table of :mod:`halley.ir.units` written for the type checker, so a unit
mistake shows in the editor before a build. At run time the tracer returns symbolic expressions
in their place and the checker applies the rule table to the graph.

``Rate`` carries a period and a compounding tag. The compounding tag defaults to ``Effective``
(PEP 696), so ``Rate[Annual]`` and ``Rate[Annual, Effective]`` are the same annotation and the
same unit; ``Rate[Annual, Continuous]`` is the continuously compounded rate. Python 3.12 needs
``typing_extensions.TypeVar`` for the default because ``get_type_hints`` evaluates the annotation
at build time.
"""

from __future__ import annotations

from typing import Any, Generic, overload

from typing_extensions import TypeVar

from halley.errors import Loc, UnitError
from halley.ir.units import (
    BOOL,
    COUNT,
    COVERAGE_UNITS,
    FACTOR,
    INDEX,
    INT,
    MONEY,
    MONEY_PP,
    PROB,
    PROB_ANNUAL,
    PROB_MONTHLY,
    RATE_ANNUAL,
    RATE_ANNUAL_CC,
    RATE_MONTHLY,
    Unit,
)


def _static() -> Any:
    raise TypeError("Halley unit types are annotations; they are never instantiated")


class Annual:
    """Period tag: per year."""


class Monthly:
    """Period tag: per month."""


class Effective:
    """Compounding tag: an effective rate, whose accumulation factor is ``1 + rate``."""


class Continuous:
    """Compounding tag: a continuously compounded rate (a force), which accumulates through
    ``cc_discount()`` and never through ``1 + rate``."""


_P = TypeVar("_P", Annual, Monthly)
_C = TypeVar("_C", Effective, Continuous, default=Effective)


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

    def __abs__(self) -> Int:
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
    def __mul__(self, other: Index) -> Index: ...
    @overload
    def __mul__(self, other: CoverageUnits) -> CoverageUnits: ...
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

    def __abs__(self) -> Factor:
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

    def __abs__(self) -> Money:
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

    def __abs__(self) -> MoneyPerPolicy:
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

    def __abs__(self) -> Count:
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


class Index:
    """A price index level (CPI). Two levels divide to a :class:`Factor`; a level scales by a
    factor or a number. Nothing else has a meaning: an index is not summed or compounded."""

    def __add__(self, other: Index) -> Index:
        return _static()

    def __sub__(self, other: Index) -> Index:
        return _static()

    def __mul__(self, other: Factor | float | int) -> Index:
        return _static()

    def __rmul__(self, other: float | int) -> Index:
        return _static()

    @overload
    def __truediv__(self, other: Index) -> Factor: ...
    @overload
    def __truediv__(self, other: Factor | float | int) -> Index: ...
    def __truediv__(self, other: object) -> object:
        return _static()

    def __neg__(self) -> Index:
        return _static()

    def __abs__(self) -> Index:
        return _static()

    def __lt__(self, other: Index | float | int) -> Bool:
        return _static()

    def __le__(self, other: Index | float | int) -> Bool:
        return _static()

    def __gt__(self, other: Index | float | int) -> Bool:
        return _static()

    def __ge__(self, other: Index | float | int) -> Bool:
        return _static()

    def __eq__(self, other: object) -> Bool:  # pyright: ignore[reportIncompatibleMethodOverride]
        return _static()

    def __ne__(self, other: object) -> Bool:  # pyright: ignore[reportIncompatibleMethodOverride]
        return _static()

    __hash__ = object.__hash__


class CoverageUnits:
    """IFRS 17 coverage units: a quantity of service for the whole in-force count. Not money:
    :func:`~halley.authoring.as_cu` converts an amount, and ``Money + CoverageUnits`` is an
    error."""

    def __add__(self, other: CoverageUnits) -> CoverageUnits:
        return _static()

    def __sub__(self, other: CoverageUnits) -> CoverageUnits:
        return _static()

    def __mul__(self, other: Factor | float | int) -> CoverageUnits:
        return _static()

    def __rmul__(self, other: float | int) -> CoverageUnits:
        return _static()

    @overload
    def __truediv__(self, other: CoverageUnits) -> Factor: ...
    @overload
    def __truediv__(self, other: Factor | float | int) -> CoverageUnits: ...
    def __truediv__(self, other: object) -> object:
        return _static()

    def __neg__(self) -> CoverageUnits:
        return _static()

    def __abs__(self) -> CoverageUnits:
        return _static()

    def __lt__(self, other: CoverageUnits | float | int) -> Bool:
        return _static()

    def __le__(self, other: CoverageUnits | float | int) -> Bool:
        return _static()

    def __gt__(self, other: CoverageUnits | float | int) -> Bool:
        return _static()

    def __ge__(self, other: CoverageUnits | float | int) -> Bool:
        return _static()

    def __eq__(self, other: object) -> Bool:  # pyright: ignore[reportIncompatibleMethodOverride]
        return _static()

    def __ne__(self, other: object) -> Bool:  # pyright: ignore[reportIncompatibleMethodOverride]
        return _static()

    __hash__ = object.__hash__


class Prob[P: (Annual, Monthly)]:
    """A probability in [0, 1], checked at run time. ``Prob[Annual]`` and ``Prob[Monthly]``
    carry the period; :func:`~halley.authoring.to_monthly_prob` and
    :func:`~halley.authoring.to_monthly_cf` convert between them."""

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

    def __abs__(self) -> Prob[P]:
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


class Rate(Generic[_P, _C]):
    """An interest or inflation rate per year or per month, effective unless the second tag says
    ``Continuous``. ``1 + rate`` is a :class:`Factor` for an effective rate only;
    :func:`~halley.authoring.to_monthly` and :func:`~halley.authoring.annual_step` take the
    effective annual rate, :func:`~halley.authoring.cc_discount` the continuous one.
    ``Rate[Monthly]`` is effective by definition."""

    def __add__(self, other: Rate[_P, _C]) -> Rate[_P, _C]:
        return _static()

    def __radd__(self: Rate[_P, Effective], other: float | int) -> Factor:
        return _static()

    def __sub__(self, other: Rate[_P, _C]) -> Rate[_P, _C]:
        return _static()

    def __rsub__(self: Rate[_P, Effective], other: float | int) -> Factor:
        return _static()

    @overload
    def __mul__(self, other: float | int) -> Rate[_P, _C]: ...
    @overload
    def __mul__(self: Rate[_P, Effective], other: Factor) -> Rate[_P, Effective]: ...
    @overload
    def __mul__(self: Rate[Monthly], other: Money) -> Money: ...
    @overload
    def __mul__(self: Rate[Monthly], other: MoneyPerPolicy) -> MoneyPerPolicy: ...
    def __mul__(self, other: object) -> object:
        return _static()

    def __rmul__(self, other: float | int) -> Rate[_P, _C]:
        return _static()

    def __truediv__(self, other: Factor | float | int) -> Rate[_P, _C]:
        return _static()

    def __neg__(self) -> Rate[_P, _C]:
        return _static()

    def __abs__(self) -> Rate[_P, _C]:
        return _static()

    def __lt__(self, other: Rate[_P, _C] | float | int) -> Bool:
        return _static()

    def __le__(self, other: Rate[_P, _C] | float | int) -> Bool:
        return _static()

    def __gt__(self, other: Rate[_P, _C] | float | int) -> Bool:
        return _static()

    def __ge__(self, other: Rate[_P, _C] | float | int) -> Bool:
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
    Index: INDEX,
    CoverageUnits: COVERAGE_UNITS,
    Int: INT,
    Bool: BOOL,
}

_LEGAL = (
    "Money, MoneyPerPolicy, Count, Prob[Annual], Prob[Monthly], Rate[Annual], "
    "Rate[Annual, Continuous], Rate[Monthly], Factor, Index, CoverageUnits, Int or Bool"
)


def _rate_unit(args: tuple[object, ...], loc: Loc | None) -> Unit:
    period = args[0] if args else None
    compounding = args[1] if len(args) > 1 else Effective
    if period is not Annual and period is not Monthly:
        raise UnitError(f"Rate takes Annual or Monthly, not {period!r}", loc)
    if compounding is Effective:
        return RATE_ANNUAL if period is Annual else RATE_MONTHLY
    if compounding is Continuous:
        if period is Monthly:
            raise UnitError(
                "Rate[Monthly] is effective by definition and takes no compounding tag", loc
            )
        return RATE_ANNUAL_CC
    raise UnitError(
        f"Rate takes Effective or Continuous as its second tag, not {compounding!r}", loc
    )


def unit_from_annotation(annotation: object, loc: Loc | None = None) -> Unit:
    """Resolve a return annotation (a class above, possibly parameterised) to its tag."""
    origin = getattr(annotation, "__origin__", None)
    args: tuple[object, ...] = getattr(annotation, "__args__", ())
    if origin is Rate:
        return _rate_unit(args, loc)
    if origin is Prob:
        period = args[0] if args else None
        if period is Annual:
            return PROB_ANNUAL
        if period is Monthly:
            return PROB_MONTHLY
        raise UnitError(f"Prob takes Annual or Monthly, not {period!r}", loc)
    if annotation in _SIMPLE:
        return _SIMPLE[annotation]
    if annotation is Rate:
        raise UnitError("Rate needs a period: Rate[Annual] or Rate[Monthly]", loc)
    raise UnitError(f"{annotation!r} is not a Halley unit; use {_LEGAL}", loc)
