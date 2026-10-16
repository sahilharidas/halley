"""Functions a formula may call.

``where`` is the only conditional (D8). ``at`` and ``sum_t`` read a variable at a fixed month and
sum it over its domain. ``to_monthly`` and ``to_monthly_prob`` are the two explicit period
conversions (D11): a rate or probability written per year never acts per month without one of
them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, overload

from halley import ir
from halley.authoring.product import BoundVar
from halley.authoring.units import Annual, Bool, Factor, Int, Monthly, Prob, Rate
from halley.errors import BuildError
from halley.tracing.symbolic import Sym, as_node, current_loc


@overload
def where[T](cond: Bool | bool, a: T, b: float) -> T: ...
@overload
def where[T](cond: Bool | bool, a: float, b: T) -> T: ...
@overload
def where[T](cond: Bool | bool, a: T, b: T) -> T: ...
def where(cond: Any, a: Any, b: Any) -> Any:
    """``a`` where ``cond`` holds, otherwise ``b``. Both branches are evaluated and the unused
    one discarded, so a division in a branch must be guarded in the branch itself."""
    loc = current_loc()
    return Sym(ir.Where(loc, as_node(cond, loc), as_node(a, loc), as_node(b, loc)))


def at[T](x: Callable[[Int], T], index: Int | int) -> T:
    """The value of a time variable at a fixed month: a constant or a per-policy integer."""
    loc = current_loc()
    if not isinstance(x, BoundVar) or not x.var.has_time:
        raise BuildError("at() takes a variable with a time dimension, as at(self.x, 0)", loc)
    node: Any = ir.VarAt(loc, x.var.name, as_node(index, loc))
    return Sym(node)  # pyright: ignore[reportReturnType]


def sum_t[T](x: Callable[[Int], T]) -> T:
    """The sum of a time variable over its domain, in ascending ``t`` and in a fixed order."""
    loc = current_loc()
    if not isinstance(x, BoundVar) or not x.var.has_time:
        raise BuildError("sum_t() takes a variable with a time dimension, as sum_t(self.x)", loc)
    node: Any = ir.SumT(loc, x.var.name)
    return Sym(node)  # pyright: ignore[reportReturnType]


def _call(fn: str, *args: object) -> Any:
    loc = current_loc()
    return Sym(ir.Call(loc, fn, tuple(as_node(a, loc) for a in args)))


def exp(x: Factor | float) -> Factor:
    return _call("exp", x)


def log(x: Factor | float) -> Factor:
    return _call("log", x)


def expm1(x: Factor | float) -> Factor:
    return _call("expm1", x)


def log1p(x: Factor | float) -> Factor:
    return _call("log1p", x)


def sqrt(x: Factor | float) -> Factor:
    return _call("sqrt", x)


def minimum[T](a: T, b: T | float) -> T:
    """The smaller of two values. Python's ``min`` cannot be used on symbolic values."""
    return _call("minimum", a, b)


def maximum[T](a: T, b: T | float) -> T:
    return _call("maximum", a, b)


def to_monthly(rate: Rate[Annual] | float) -> Rate[Monthly]:
    """``(1 + i) ** (1/12) - 1``: the monthly effective rate of an annual effective rate."""
    return _call("to_monthly", rate)


def to_monthly_prob(q: Prob[Annual] | float) -> Prob[Monthly]:
    """``1 - (1 - q) ** (1/12)``: the monthly probability of an annual one, under a constant
    force within the year. ``q / 12`` is a different number and is not offered."""
    return _call("to_monthly_prob", q)


def lag(x: Any, d: Any, *, min: int, max: int) -> Any:
    """A data-dependent time offset with declared bounds (D12). Not available yet: the term
    assurance and annuity reference models use constant offsets only."""
    raise BuildError(
        "lag() is not available in this milestone; the reference models use constant offsets",
        current_loc(),
    )
