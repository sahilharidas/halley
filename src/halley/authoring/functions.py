"""Functions a formula may call.

``where`` is the only conditional (D8). ``at`` and ``sum_t`` read a variable at a fixed month and
sum it over its domain. The period and compounding conversions (D11) are explicit: a rate or
probability written per year never acts per month without one of them, and a continuously
compounded rate accumulates only through ``cc_discount``. Each conversion is one ``Call`` node
with a fixed emitted form (annuity spec, "Compounding and unit conversions"), so the backend and
the ``explain`` interpreter evaluate it in the same floating-point order (D19). ``pv_month`` is
not a node: it expands at trace time into the cash flow read times the discount factor its
timing calls for.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, overload

from halley import ir
from halley.authoring.product import BoundVar
from halley.authoring.units import (
    Annual,
    Bool,
    Continuous,
    CoverageUnits,
    Factor,
    Int,
    Money,
    Monthly,
    Prob,
    Rate,
)
from halley.errors import BuildError
from halley.ir.units import FACTOR
from halley.tracing.symbolic import Sym, as_node, current_loc, offset_of


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


def pv_month[T](cf: Callable[[Int], T], t: Int, v: Callable[[Int], Factor]) -> T:
    """``cf`` at month ``t`` (written ``t`` or ``t + k``) discounted over month ``t`` by ``cf``'s
    declared timing: a start-timed flow is read as it is, a mid-timed one is multiplied by
    ``v(t) ** 0.5`` and an end-timed one by ``v(t)``, where ``v`` is the one-month discount
    factor variable. Expands at trace time into those reads, so the graph shows the model's own
    grouping and ``explain`` lists ``cf(t+k)`` and ``v(t+k)``."""
    loc = current_loc()
    if not isinstance(cf, BoundVar) or not cf.var.has_time:
        raise BuildError(
            "pv_month() takes a cash flow variable with a time dimension, as "
            "pv_month(self.cf, t + 1, self.v)",
            loc,
        )
    name = cf.var.name
    timing = cf.var.timing
    if timing is None:
        raise BuildError(
            f"pv_month({name}, ...) needs the timing of {name}; declare it "
            f"@var(timing='start'), @var(timing='mid') or @var(timing='end')",
            loc,
        )
    if not isinstance(v, BoundVar) or not v.var.has_time:
        raise BuildError(
            f"pv_month({name}, ...) takes the one-month discount factor as a variable with a time "
            f"dimension, as pv_month(self.{name}, t + 1, self.v)",
            loc,
        )
    v_unit = v.var.declared_unit()
    if v_unit != FACTOR:
        raise BuildError(
            f"pv_month({name}, ...) discounts with {v.var.name}, which is declared {v_unit}, "
            "not Factor",
            loc,
        )
    k = offset_of(t, loc, name)
    cf_read: ir.Node = ir.VarRead(loc, name, k)
    if timing == "start":
        node: ir.Node = cf_read
    elif timing == "mid":
        v_read: ir.Node = ir.VarRead(loc, v.var.name, k)
        node = ir.Binary(loc, "mul", cf_read, ir.Binary(loc, "pow", v_read, ir.Const(loc, 0.5)))
    else:
        node = ir.Binary(loc, "mul", cf_read, ir.VarRead(loc, v.var.name, k))
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


def to_monthly_cf(q: Prob[Annual] | float) -> Prob[Monthly]:
    """The monthly probability of an annual one under a constant force, computed as
    ``-expm1(log1p(-q) / 12)`` for accuracy near 0 and 1; ``q >= 1`` gives exactly 1 (the
    annuity spec's conversion, which keeps the terminal age exact)."""
    return _call("to_monthly_cf", q)


def annual_step(g: Rate[Annual] | float) -> Factor:
    """``1 + g``: the one-year escalation factor of an effective annual rate."""
    return _call("annual_step", g)


def cc_discount(s: Rate[Annual, Continuous] | float, *, months: Int | int) -> Factor:
    """``exp(-s * months / 12)``: the discount factor over ``months`` of a continuously
    compounded annual rate (an illiquidity premium, a force of interest)."""
    return _call("cc_discount", s, months)


def fwd_rate(p0: Factor | float, p1: Factor | float) -> Rate[Monthly]:
    """``p0 / p1 - 1``: the one-month forward rate between two discount factors."""
    return _call("fwd_rate", p0, p1)


def ann_fwd(p0: Factor | float, p1: Factor | float) -> Rate[Annual]:
    """``(p0 / p1) ** 12 - 1``: the one-month forward rate annualised."""
    return _call("ann_fwd", p0, p1)


def ann_spot(p: Factor | float, m: Int | int) -> Rate[Annual]:
    """``p ** (-12 / m) - 1``: the annualised spot rate of the discount factor to month ``m``
    (``m >= 1``; the caller guards ``m = 0`` with ``where``)."""
    return _call("ann_spot", p, m)


def as_cu(x: Money | float) -> CoverageUnits:
    """Declare an amount to be the coverage units of the month: the same number, in the unit
    that ``sum_t`` and the aggregation stage expect for coverage units."""
    return _call("as_cu", x)


def lag(x: Any, d: Any, *, min: int, max: int) -> Any:
    """A data-dependent time offset with declared bounds (D12). Not available yet: the term
    assurance and annuity reference models use constant offsets only."""
    raise BuildError(
        "lag() is not available in this milestone; the reference models use constant offsets",
        current_loc(),
    )
