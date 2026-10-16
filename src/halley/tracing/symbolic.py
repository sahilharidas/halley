"""Symbolic values.

While a model is traced, every read of a variable, field, table or scenario input returns a
:class:`Sym` wrapping an IR node. Arithmetic on ``Sym`` builds more nodes and records where in
the model source it happened. Anything that would need the value itself (a Python ``if``,
``min``, ``max``, ``and``, ``or``, ``float()``) is an error that names ``where()``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

from halley import ir
from halley.errors import BuildError, Loc


@dataclass(slots=True)
class TraceContext:
    file: str
    function: str
    first_line: int


_context: TraceContext | None = None


def set_context(ctx: TraceContext | None) -> None:
    global _context
    _context = ctx


def current_loc() -> Loc | None:
    """The innermost frame in the model's source file, or the method being traced."""
    ctx = _context
    if ctx is None:
        return None
    frame = sys._getframe(1)  # pyright: ignore[reportPrivateUsage]
    while frame is not None:
        if frame.f_code.co_filename == ctx.file:
            return Loc(ctx.file, frame.f_lineno, frame.f_code.co_qualname)
        frame = frame.f_back
    return Loc(ctx.file, ctx.first_line, ctx.function)


class Sym:
    """A symbolic value: the result of reading something or combining other symbolic values."""

    __slots__ = ("node",)

    def __init__(self, node: ir.Node) -> None:
        self.node = node

    def __repr__(self) -> str:
        return f"Sym({ir.render(self.node)})"

    # ----------------------------------------------------------------- arithmetic
    def _binary(self, op: str, other: object, swap: bool = False) -> Sym:
        loc = current_loc()
        a, b = self.node, as_node(other, loc)
        if swap:
            a, b = b, a
        return Sym(ir.Binary(loc, op, a, b))

    def __add__(self, other: object) -> Sym:
        return self._binary("add", other)

    def __radd__(self, other: object) -> Sym:
        return self._binary("add", other, swap=True)

    def __sub__(self, other: object) -> Sym:
        return self._binary("sub", other)

    def __rsub__(self, other: object) -> Sym:
        return self._binary("sub", other, swap=True)

    def __mul__(self, other: object) -> Sym:
        return self._binary("mul", other)

    def __rmul__(self, other: object) -> Sym:
        return self._binary("mul", other, swap=True)

    def __truediv__(self, other: object) -> Sym:
        return self._binary("truediv", other)

    def __rtruediv__(self, other: object) -> Sym:
        return self._binary("truediv", other, swap=True)

    def __floordiv__(self, other: object) -> Sym:
        return self._binary("floordiv", other)

    def __rfloordiv__(self, other: object) -> Sym:
        return self._binary("floordiv", other, swap=True)

    def __mod__(self, other: object) -> Sym:
        return self._binary("mod", other)

    def __rmod__(self, other: object) -> Sym:
        return self._binary("mod", other, swap=True)

    def __pow__(self, other: object) -> Sym:
        return self._binary("pow", other)

    def __rpow__(self, other: object) -> Sym:
        return self._binary("pow", other, swap=True)

    def __and__(self, other: object) -> Sym:
        return self._binary("and", other)

    def __rand__(self, other: object) -> Sym:
        return self._binary("and", other, swap=True)

    def __or__(self, other: object) -> Sym:
        return self._binary("or", other)

    def __ror__(self, other: object) -> Sym:
        return self._binary("or", other, swap=True)

    def __lt__(self, other: object) -> Sym:
        return self._binary("lt", other)

    def __le__(self, other: object) -> Sym:
        return self._binary("le", other)

    def __gt__(self, other: object) -> Sym:
        return self._binary("gt", other)

    def __ge__(self, other: object) -> Sym:
        return self._binary("ge", other)

    def __eq__(self, other: object) -> Sym:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self._binary("eq", other)

    def __ne__(self, other: object) -> Sym:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self._binary("ne", other)

    __hash__ = object.__hash__

    def __neg__(self) -> Sym:
        return Sym(ir.Unary(current_loc(), "neg", self.node))

    def __pos__(self) -> Sym:
        return self

    def __invert__(self) -> Sym:
        return Sym(ir.Unary(current_loc(), "not", self.node))

    # ----------------------------------------------------------------- forbidden uses
    def __bool__(self) -> bool:
        raise BuildError(
            "a Python if, min, max, and, or or not on a value that depends on data or time; "
            "write the condition with where(cond, a, b)",
            current_loc(),
        )

    def __float__(self) -> float:
        raise BuildError("float() of a symbolic value; formulas stay symbolic", current_loc())

    def __int__(self) -> int:
        raise BuildError("int() of a symbolic value; use // for integer division", current_loc())

    def __index__(self) -> int:
        raise BuildError(
            "a symbolic value cannot index a Python sequence; use a table lookup", current_loc()
        )

    def __iter__(self) -> Any:
        raise BuildError("a symbolic value cannot be iterated", current_loc())


def as_node(value: object, loc: Loc | None) -> ir.Node:
    """The node for a symbolic value or a Python constant."""
    if isinstance(value, Sym):
        return value.node
    if isinstance(value, bool):
        return ir.Const(loc, value)
    if isinstance(value, (int, float, str)):
        return ir.Const(loc, value)
    raise BuildError(
        f"a {type(value).__name__} cannot appear in a formula; formulas combine variables, "
        "model point fields, table values, scenario inputs and numbers",
        loc,
    )


def offset_of(arg: object, loc: Loc | None, what: str) -> int:
    """The constant offset ``k`` of a read written ``x(t + k)``."""
    if isinstance(arg, Sym):
        node = arg.node
        if isinstance(node, ir.TimeIndex):
            return 0
        if isinstance(node, ir.Binary) and node.op in ("add", "sub"):
            a, b = node.a, node.b
            if isinstance(a, ir.TimeIndex) and isinstance(b, ir.Const) and _is_int(b.value):
                return int(b.value) if node.op == "add" else -int(b.value)
            if (
                node.op == "add"
                and isinstance(b, ir.TimeIndex)
                and isinstance(a, ir.Const)
                and _is_int(a.value)
            ):
                return int(a.value)
        raise BuildError(
            f"the time offset of {what} must be a constant: {what}(t), {what}(t-1), {what}(t+1); "
            "data-dependent offsets go through lag() with declared bounds, and a fixed month "
            f"through at({what}, k)",
            loc,
        )
    if isinstance(arg, int) and not isinstance(arg, bool):
        raise BuildError(
            f"{what}({arg}) reads a fixed month; write at({what}, {arg}) for that, or {what}(t) "
            "for the current month",
            loc,
        )
    raise BuildError(f"{what}() takes the time index t, with a constant offset", loc)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)
