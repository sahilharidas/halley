"""Numba backend: the schedule as compiled loops, tested against the NumPy reference (D16).

The generated module has the NumPy backend's batch ABI (``run_batch(inp, H)``, the same arrays
in, the same arrays out, month quantities exactly 0 at ``t = 0``, Int and Bool arrays int64 and
bool) and calls one ``@numba.njit`` kernel per component, in schedule order. A kernel loops
explicitly: over blocks of :data:`POLICY_BLOCK` policies (``numba.prange`` when the component
has a policy dimension and :data:`PARALLEL` is set; no reduction crosses policies, D5), then
months, then the block's policies and the scenarios, and evaluates each variable's formula once
per lane and month. The formula is the NumPy backend's expression, fully parenthesised in the
IR's order (D19): a variable read at ``t + k`` is ``V_y[t + k, p, s]`` (index 0 on an axis ``y``
does not have), a model point field ``M_f[0, p, 0]``, a scenario input ``Z_s[t + k - u0, 0,
s]``, a group series ``Y_s[t + k, p, s]``; ``where`` evaluates both operands and selects, as
``np.where`` does, so a discarded branch may divide by zero (``error_model="numpy"`` gives inf
or nan, never an exception); ``sum_t`` is a sequential loop in ascending ``t``; ``at(x, k)``
reads month ``k`` per lane; ``**`` is LLVM's pow with a float exponent. Kernels compile with
``fastmath=False`` (no contraction to FMA, no reassociation) and ``cache=True``.

The run-time checks have the NumPy backend's semantics. A kernel never reads out of bounds:
before a table read it tests each key against its declared range, before an ``at``,
``scen.at`` or ``series.at`` read it tests the month, each into a temporary computed before the
statement that reads it, and on a failure it records the check in per-lane error arrays (its
place in the order the NumPy backend evaluates checks, and the range of the values that failed
there) and reads 0 instead. After the kernel the wrapper raises the NumPy backend's text for the
first failing check in that order, naming the lanes the NumPy backend names
(:func:`raise_first`), then runs the NumPy backend's own probability check on the component's
outputs; the finiteness check runs at the end, as there.

The module text is ``CompiledModel.source``. It is written to a real file, :func:`cache_dir`
``/ <ir_hash>.py``, and imported from there, so that Numba's ``cache=True`` keeps the compiled
kernels beside it between processes.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
import platform
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Final, NoReturn

import numpy as np
from numpy.typing import NDArray

from halley import ir
from halley.codegen.numpy_backend import ARRAY, BatchFn, LaneError, run_batch, value_kind
from halley.errors import BuildError, Loc
from halley.ir import units
from halley.scheduler import STEP, Component, Schedule

NAME: Final = "numba"
ARTEFACT: Final = "numba_backend.py"
CACHE_ENV: Final = "HALLEY_NUMBA_CACHE"
"""An environment variable naming the directory of the generated modules and their compiled
kernels, instead of ``<temp>/halley-numba``."""
POLICY_BLOCK: Final = 64
"""Policies one iteration of a kernel's outer loop takes: months run inside a block, so the
inner loops read contiguous lanes of each month."""
PARALLEL: Final = False
"""Whether a kernel with a policy dimension runs its blocks with ``numba.prange``."""
INSTALL_HINT: Final = "pip install halley[numba]"

_OPTIONS: Final = 'cache=True, fastmath=False, error_model="numpy", boundscheck=False'


def _numba() -> Any:
    """The ``numba`` module (imported by name: it ships no type stubs), or a BuildError saying
    how to install it."""
    try:
        return importlib.import_module("numba")
    except ImportError:
        raise BuildError(
            f"the Numba backend needs numba, which is not installed: {INSTALL_HINT}"
        ) from None


def cache_dir() -> Path:
    """Where the generated modules and Numba's cache of their kernels live: the directory
    named by ``HALLEY_NUMBA_CACHE``, or ``halley-numba`` under the system's temporary
    directory."""
    named = os.environ.get(CACHE_ENV)
    return Path(named) if named else Path(tempfile.gettempdir()) / "halley-numba"


# ----------------------------------------------------------------------------- run time


def as_input(values: Any, dtype: Any) -> NDArray[Any]:
    """An input array as a kernel's signature types it: C-contiguous and writeable (Numba
    types a read-only array, such as a group series' values, as another type). The kernels
    never write it; a read-only one is copied."""
    a = np.ascontiguousarray(values, dtype=dtype)
    return a if a.flags.writeable else a.copy()


def as_table(values: Any, dtype: Any) -> NDArray[Any]:
    """A table's dense column for a kernel (:func:`as_input`), a keyless table's 0-d value as
    one element, so that every kernel argument has a fixed type."""
    a = as_input(values, dtype)
    return a.reshape(1) if a.ndim == 0 else a


def raise_first(
    component: tuple[int, int, tuple[Any, ...], tuple[Any, ...]],
    e_pos: NDArray[np.int64],
    e_lo: NDArray[np.int64],
    e_hi: NDArray[np.int64],
    horizon: int,
    lasts: dict[str, int],
) -> NoReturn:
    """Raise the NumPy backend's :class:`LaneError` for the first check that failed in a
    kernel.

    ``component`` is ``(nb, nl, boundary, loop)``: the checks before the time loop (positions
    ``1..nb``) and those of one step of it (``nl`` of them, repeated per step after ``nb``).
    Each check is ``(kind, variable, loc, form, policy, params)``: ``form`` is ``"scalar"``
    when the value the NumPy backend checks has no lane axis (its message then names the value
    and no lanes), ``policy`` whether that value has a policy axis (without one, every lane of
    the batch fails). The lanes whose first failure is at the smallest position are the lanes
    the NumPy backend's mask holds at that check, and the range of their values there is its
    message's."""
    nb, nl, boundary, loop = component
    pos = int(e_pos[e_pos > 0].min())
    kind, variable, loc, form, policy, params = (
        boundary[pos - 1] if pos <= nb else loop[(pos - nb - 1) % nl]
    )
    there = e_pos == pos
    lo, hi = int(e_lo[there].min()), int(e_hi[there].max())
    scalar = form == "scalar"
    if kind == "key":
        key, table, k_lo, k_hi = params
        message = (
            f"lookup key {key} of table {table} is outside its declared range [{k_lo}, {k_hi}]: "
            f"value {lo} to {hi}; no silent extrapolation"
        )
    elif kind == "at":
        name, start = params
        message = (
            f"at({name}, {lo}) is outside the domain of {name}, months {start}..{horizon}"
            if scalar
            else f"at({name}, ...) reads a month outside the domain of {name}, months "
            f"{start}..{horizon}"
        )
    elif kind == "scen_at":
        (name,) = params
        message = (
            f"scen.{name}.at({lo}) is outside the scenario file's months"
            if scalar
            else f"scen.{name}.at(...) reads a month outside the scenario file"
        )
    else:  # series_at
        name, start = params
        last = lasts[name]
        message = (
            f"series.{name}.at({lo}) is outside the months of series {name}, {start}..{last}"
            if scalar
            else f"series.{name}.at(...) reads a month outside the months of series {name}, "
            f"{start}..{last}"
        )
    if scalar:
        lanes = np.ones((), dtype=np.bool_)
    elif not policy:
        lanes = np.ones((1,), dtype=np.bool_)  # no policy axis: _lane_ids names every lane
    else:
        lanes = np.asarray(there.any(axis=1))
    raise LaneError(message, str(variable), str(loc), lanes)


# ---------------------------------------------------------------------------- generation

_HELPERS: Final = f'''
@numba.njit({_OPTIONS})
def _flag(pos, value, p, s, E_pos, E_lo, E_hi):
    """Record a failed check at position pos for lane (p, s): the lane keeps its first failure
    in the NumPy backend's order and the range of the values that failed there."""
    cur = E_pos[p, s]
    if cur == 0 or pos < cur:
        E_pos[p, s] = pos
        E_lo[p, s] = value
        E_hi[p, s] = value
    elif pos == cur:
        if value < E_lo[p, s]:
            E_lo[p, s] = value
        if value > E_hi[p, s]:
            E_hi[p, s] = value


@numba.njit({_OPTIONS})
def _where(c, a, b):
    """np.where of one lane: the caller has evaluated both operands."""
    if c:
        return a
    return b


@numba.njit({_OPTIONS})
def _to_monthly_cf(q):
    """to_monthly_cf(q), the NumPy backend's emitted form for one lane."""
    return _where(q >= 1, 1.0, -np.expm1(np.log1p(-_where(q >= 1, 0.0, q)) / 12))
'''

_INT_ARRAY: Final = "int64[:, :, ::1]"
_FLOAT_ARRAY: Final = "float64[:, :, ::1]"
_LANES: Final = "int64[:, ::1]"

_CALLS: Final = {
    "exp": "np.exp",
    "log": "np.log",
    "expm1": "np.expm1",
    "log1p": "np.log1p",
    "sqrt": "np.sqrt",
    "minimum": "np.minimum",
    "maximum": "np.maximum",
    "abs": "np.abs",
}
_BINOPS: Final = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "truediv": "/",
    "floordiv": "//",
    "mod": "%",
    "lt": "<",
    "le": "<=",
    "gt": ">",
    "ge": ">=",
    "eq": "==",
    "ne": "!=",
    "and": "&",
    "or": "|",
}
_BOOLEAN_OPS: Final = frozenset({"lt", "le", "gt", "ge", "eq", "ne", "and", "or"})
_FLOAT_CALLS: Final = frozenset(
    {
        "exp",
        "log",
        "expm1",
        "log1p",
        "sqrt",
        "to_monthly",
        "to_monthly_prob",
        "to_monthly_cf",
        "annual_step",
        "cc_discount",
        "fwd_rate",
        "ann_fwd",
        "ann_spot",
    }
)


@dataclass(frozen=True, slots=True)
class _Ctx:
    mode: str  # "whole", "step" or "scalar", as in the NumPy backend
    loop: bool = False
    """A statement of a step component's time loop: its checks repeat at every step."""


@dataclass(slots=True)
class _Kernel:
    """One component's kernel as it is generated: its arguments by name with their Numba
    types, in first-use order, and its checks before the time loop and in one step of it."""

    args: dict[str, str] = field(default_factory=lambda: {})
    boundary: list[str] = field(default_factory=lambda: [])
    loop: list[str] = field(default_factory=lambda: [])
    temps: int = 0


def _loc_str(loc: Loc | None) -> str:
    return "" if loc is None else str(loc)


def _dtype(v: ir.Variable) -> tuple[str, str, str]:
    """A variable's array: its NumPy dtype in the wrapper, its Numba type in a signature and
    the value a failed read of it gives."""
    if v.unit == units.INT:
        return "np.int64", _INT_ARRAY, "0"
    if v.unit == units.BOOL:
        return "np.bool_", "boolean[:, :, ::1]", "False"
    return "np.float64", _FLOAT_ARRAY, "0.0"


def _i64(value: int | str) -> str:
    """An integer a kernel passes to a helper, as an int64: Numba types a bare literal by its
    value and would compile the helper once per value."""
    return f"np.int64({value})"


def _typed(node: ir.Node, text: str) -> str:
    """``text`` as a helper's argument: a literal int or bool constant made a plain int64 or
    bool, for the reason of :func:`_i64`."""
    if isinstance(node, ir.Const):
        if isinstance(node.value, bool):
            return f"np.bool_({text})"
        if isinstance(node.value, int):
            return _i64(text)
    return text


def _lane(dims: ir.Dims | None) -> tuple[str, str]:
    """The indices of lane ``(p, s)`` into an array of these dimensions: 0 on an axis the
    array does not have (its size is 1)."""
    d = dims or frozenset()
    return ("p" if ir.POLICY in d else "0", "s" if ir.SCENARIO in d else "0")


def _promote(a: str, b: str) -> str:
    if a == b:
        return a
    return "float" if "float" in (a, b) else "int"


def _row(k: int) -> str:
    return f"t{k:+d}" if k else "t"


class _Generator:
    def __init__(self, graph: ir.Graph, schedule: Schedule) -> None:
        self.graph = graph
        self.schedule = schedule
        self.vars = graph.variables
        self.current = ""
        self.k = _Kernel()
        self.pre: list[str] = []
        """The statements the statement being generated needs first: checked reads."""
        self.lasts: set[str] = set()
        """Series read with ``.at``: the wrapper passes their last month to the messages."""
        self.inputs: dict[str, None] = {}
        """Every kernel argument the wrapper prepares from the batch, in first-use order."""
        self.kernel_entries: list[str] = []
        self.check_entries: list[str] = []

    # ------------------------------------------------------------------ arguments
    def use(self, name: str, numba_type: str) -> str:
        self.k.args.setdefault(name, numba_type)
        return name

    def var_array(self, name: str) -> str:
        return self.use(f"V_{name}", _dtype(self.vars[name])[1])

    def field_array(self, name: str) -> str:
        decl = self.graph.fields[name]
        kind = _FLOAT_ARRAY if decl.kind == "number" else _INT_ARRAY
        return self.use(f"M_{name}", kind)

    def column(self, table: str, column: str) -> ir.ColumnDecl:
        return next(c for c in self.graph.tables[table].columns if c.name == column)

    def table_array(self, table: str, column: str) -> str:
        decl = self.graph.tables[table]
        base = "int64" if self.column(table, column).unit == units.INT else "float64"
        dims = ", ".join([":"] * (max(1, len(decl.keys)) - 1) + ["::1"])
        return self.use(f"T_{table}__{column}", f"{base}[{dims}]")

    def temp(self, prefix: str) -> str:
        self.k.temps += 1
        return f"_{prefix}{self.k.temps}"

    # ------------------------------------------------------------------ checks
    def check(self, kind: str, node: ir.Node, ctx: _Ctx, params: tuple[object, ...]) -> str:
        """Register a check of the value of ``node`` and return its position: an int64 before
        the time loop, ``(_q + j)`` inside it, where ``_q`` is the position of the step's
        first check."""
        form = "array" if value_kind(node, ctx.mode == "step") == ARRAY else "scalar"
        policy = ir.POLICY in (node.dims or frozenset())
        entry = repr((kind, self.current, _loc_str(node.loc), form, policy, params))
        if ctx.loop:
            self.k.loop.append(entry)
            return f"(_q + {len(self.k.loop) - 1})"
        self.k.boundary.append(entry)
        return _i64(len(self.k.boundary))

    def checked(self, value: str, node: ir.Node) -> str:
        """``value`` held in a temporary; an index if Numba would see a float there (a model
        point field of kind ``number`` used as a month or a key), truncated as NumPy's index
        conversion does."""
        x = self.temp("x")
        if self.kind(node) == "float" or isinstance(node, ir.Const):
            value = _i64(value)  # a constant too: Numba would type the temporary by its value
        self.pre.append(f"{x} = {value}")
        return x

    # ------------------------------------------------------------------ expressions
    def expr(self, node: ir.Node, ctx: _Ctx) -> str:
        if isinstance(node, ir.Const):
            if isinstance(node.value, bool):
                return "True" if node.value else "False"
            if isinstance(node.value, str):
                raise BuildError("a string constant can only be a table key", node.loc)
            return repr(node.value)
        if isinstance(node, ir.TimeIndex):
            if ctx.mode == "scalar":
                raise BuildError("t appears in a variable without a time dimension", node.loc)
            return "t"
        if isinstance(node, ir.VarRead):
            return self.var_read(node, ctx)
        if isinstance(node, ir.VarAt):
            return self.var_at(node, ctx)
        if isinstance(node, ir.SumT):
            return self.sum_t(node)
        if isinstance(node, ir.MpField):
            return f"{self.field_array(node.name)}[0, p, 0]"
        if isinstance(node, ir.TableLookup):
            return self.lookup(node, ctx)
        if isinstance(node, ir.ScenRead):
            if ctx.mode == "scalar":
                raise BuildError(
                    f"scen.{node.name}(t) appears in a variable without a time dimension", node.loc
                )
            z = self.use(f"Z_{node.name}", _FLOAT_ARRAY)
            u0 = self.use(f"u0_{node.name}", "int64")
            return f"{z}[{_row(node.offset)} - {u0}, 0, s]"
        if isinstance(node, ir.ScenAt):
            return self.scen_at(node, ctx)
        if isinstance(node, ir.SeriesRead):
            return self.series_read(node, ctx)
        if isinstance(node, ir.SeriesAt):
            return self.series_at(node, ctx)
        if isinstance(node, ir.RunConst):
            return self.use(f"R_{node.name}", "int64")
        if isinstance(node, ir.Unary):
            a = self.expr(node.a, ctx)
            return f"(-{a})" if node.op == "neg" else f"(not {a})"
        if isinstance(node, ir.Binary):
            if node.op == "pow":
                return self.power(node, ctx)
            if node.op not in _BINOPS:
                raise BuildError(
                    f"the Numba backend has no form for the operator {node.op}", node.loc
                )
            a = self.expr(node.a, ctx)
            b = self.expr(node.b, ctx)
            return f"({a} {_BINOPS[node.op]} {b})"
        if isinstance(node, ir.Where):
            c = _typed(node.cond, self.expr(node.cond, ctx))
            a = _typed(node.a, self.expr(node.a, ctx))
            b = _typed(node.b, self.expr(node.b, ctx))
            return f"_where({c}, {a}, {b})"
        if isinstance(node, ir.Call):
            args = [self.expr(a, ctx) for a in node.args]
            return self.call(node, args)
        raise BuildError(f"the Numba backend cannot emit a {type(node).__name__} node", node.loc)

    def var_read(self, node: ir.VarRead, ctx: _Ctx) -> str:
        v = self.vars[node.name]
        name = self.var_array(node.name)
        pi, si = _lane(v.dims)
        if not v.has_time:
            return f"{name}[0, {pi}, {si}]"
        if ctx.mode == "scalar":
            raise BuildError(
                f"{node.name}(t) appears in a variable without a time dimension", node.loc
            )
        return f"{name}[{_row(node.offset)}, {pi}, {si}]"

    def var_at(self, node: ir.VarAt, ctx: _Ctx) -> str:
        """``at(x, i)``: the month tested against x's domain ``start..H`` first."""
        v = self.vars[node.name]
        index = node.index
        if isinstance(index, ir.Const) and not isinstance(index.value, bool):
            i = int(index.value)
            if i < v.start:
                raise BuildError(
                    f"at({node.name}, {i}) reads before the domain of {node.name}, which "
                    f"starts at t = {v.start}",
                    node.loc,
                )
        x = self.checked(self.expr(index, ctx), index)
        pos = self.check("at", index, ctx, (node.name, v.start))
        pi, si = _lane(v.dims)
        out = self.temp("v")
        self.pre += [
            f"{out} = {_dtype(v)[2]}",
            f"if {x} < {v.start} or {x} > H:",
            f"    _flag({pos}, {x}, p, s, E_pos, E_lo, E_hi)",
            "else:",
            f"    {out} = {self.var_array(node.name)}[{x}, {pi}, {si}]",
        ]
        return out

    def sum_t(self, node: ir.SumT) -> str:
        """``sum_t(x)``: a sequential loop over x's domain in ascending ``t`` (D19)."""
        v = self.vars[node.name]
        pi, si = _lane(v.dims)
        name = self.var_array(node.name)
        out = self.temp("v")
        self.pre += [
            f"{out} = {name}[{v.start}, {pi}, {si}]",
            f"for _u in range({v.start + 1}, H + 1):",
            f"    {out} += {name}[_u, {pi}, {si}]",
        ]
        return out

    def scen_at(self, node: ir.ScenAt, ctx: _Ctx) -> str:
        x = self.checked(self.expr(node.index, ctx), node.index)
        pos = self.check("scen_at", node.index, ctx, (node.name,))
        z = self.use(f"Z_{node.name}", _FLOAT_ARRAY)
        u0 = self.use(f"u0_{node.name}", "int64")
        out = self.temp("v")
        self.pre += [
            f"{out} = 0.0",
            f"if {x} - {u0} < 0 or {x} - {u0} >= {z}.shape[0]:",
            f"    _flag({pos}, {x}, p, s, E_pos, E_lo, E_hi)",
            "else:",
            f"    {out} = {z}[{x} - {u0}, 0, s]",
        ]
        return out

    def series_read(self, node: ir.SeriesRead, ctx: _Ctx) -> str:
        """A group series, ``(Tn | 1, g, S | 1)``: read like a variable of its dimensions."""
        decl = self.graph.series[node.name]
        y = self.use(f"Y_{node.name}", _FLOAT_ARRAY)
        si = "s" if decl.scenario else "0"
        if node.offset is None:  # a series without a time dimension, (1, g, S)
            return f"{y}[0, p, {si}]"
        if ctx.mode == "scalar":
            raise BuildError(
                f"series.{node.name}(t) appears in a variable without a time dimension", node.loc
            )
        return f"{y}[{_row(node.offset)}, p, {si}]"

    def series_at(self, node: ir.SeriesAt, ctx: _Ctx) -> str:
        decl = self.graph.series[node.name]
        x = self.checked(self.expr(node.index, ctx), node.index)
        pos = self.check("series_at", node.index, ctx, (node.name, decl.start))
        y = self.use(f"Y_{node.name}", _FLOAT_ARRAY)
        self.lasts.add(node.name)
        si = "s" if decl.scenario else "0"
        out = self.temp("v")
        self.pre += [
            f"{out} = 0.0",
            f"if {x} < {decl.start} or {x} > {y}.shape[0] - 1:",
            f"    _flag({pos}, {x}, p, s, E_pos, E_lo, E_hi)",
            "else:",
            f"    {out} = {y}[{x}, p, {si}]",
        ]
        return out

    def lookup(self, node: ir.TableLookup, ctx: _Ctx) -> str:
        """A table read: each integer key tested against its declared range first, in the
        order the NumPy backend tests them; the read gives 0 if one failed."""
        decl = self.graph.tables[node.table]
        parts: list[str] = []
        ok: list[str] = []
        for key, key_node in zip(decl.keys, node.keys, strict=True):
            if key.enum is not None:
                if isinstance(key_node, ir.Const):
                    parts.append(str(key.enum.index(str(key_node.value))))
                else:
                    parts.append(self.expr(key_node, ctx))
                continue
            lo, hi = int(key.lo or 0), int(key.hi or 0)
            if isinstance(key_node, ir.Const):
                i = int(key_node.value)
                if i < lo or i > hi:
                    raise BuildError(
                        f"key {key.name} = {i} is outside the declared range [{lo}, {hi}] "
                        f"of table {node.table}",
                        node.loc,
                    )
                parts.append(str(i - lo))
                continue
            x = self.checked(self.expr(key_node, ctx), key_node)
            pos = self.check("key", key_node, ctx, (key.name, node.table, lo, hi))
            self.pre += [
                f"if {x} < {lo} or {x} > {hi}:",
                f"    _flag({pos}, {x}, p, s, E_pos, E_lo, E_hi)",
            ]
            ok.append(f"{lo} <= {x} <= {hi}")
            parts.append(f"{x} - {lo}" if lo else x)
        name = self.table_array(node.table, node.column)
        if not parts:
            return f"{name}[0]"  # a keyless table, one element
        if not ok:
            return f"{name}[{', '.join(parts)}]"
        out = self.temp("v")
        zero = "0" if self.column(node.table, node.column).unit == units.INT else "0.0"
        self.pre += [
            f"{out} = {zero}",
            f"if {' and '.join(ok)}:",
            f"    {out} = {name}[{', '.join(parts)}]",
        ]
        return out

    def power(self, node: ir.Binary, ctx: _Ctx) -> str:
        """``a ** b`` with a float exponent, so that it is LLVM's pow: Numba raises to an
        integer exponent by repeated multiplication, and 0.0 to a negative one with a
        ZeroDivisionError whatever the error model."""
        a = self.expr(node.a, ctx)
        b_node = node.b
        if (
            isinstance(b_node, ir.Const)
            and isinstance(b_node.value, int)
            and not isinstance(b_node.value, bool)
        ):
            return f"({a} ** {float(b_node.value)!r})"
        b = self.expr(b_node, ctx)
        if self.kind(b_node) != "float":
            b = f"float({b})"
        return f"({a} ** {b})"

    def call(self, node: ir.Call, args: list[str]) -> str:
        """The NumPy backend's emitted form of each formula function, for one lane."""
        fn = node.fn
        if fn == "to_monthly":
            return f"(((1 + {args[0]}) ** (1/12)) - 1)"
        if fn == "to_monthly_prob":
            return f"(1 - ((1 - {args[0]}) ** (1/12)))"
        if fn == "to_monthly_cf":
            return f"_to_monthly_cf({_typed(node.args[0], args[0])})"
        if fn == "annual_step":
            return f"(1 + {args[0]})"
        if fn == "cc_discount":
            return f"np.exp(((-{args[0]}) * {args[1]}) / 12)"
        if fn == "fwd_rate":
            return f"(({args[0]} / {args[1]}) - 1)"
        if fn == "ann_fwd":
            return f"((({args[0]} / {args[1]}) ** 12.0) - 1)"
        if fn == "ann_spot":
            return f"(({args[0]} ** (-12 / {args[1]})) - 1)"
        if fn == "as_cu":
            return args[0]
        if fn in _CALLS:
            return f"{_CALLS[fn]}({', '.join(args)})"
        raise BuildError(f"the Numba backend has no form for {fn}()", node.loc)

    def kind(self, node: ir.Node) -> str:
        """``"int"``, ``"float"`` or ``"bool"``: the kind of number a kernel holds for
        ``node``, which decides whether a power's exponent or a stored value needs a cast."""
        if isinstance(node, ir.Const):
            if isinstance(node.value, bool):
                return "bool"
            return "float" if isinstance(node.value, float) else "int"
        if isinstance(node, (ir.TimeIndex, ir.RunConst)):
            return "int"
        if isinstance(node, (ir.VarRead, ir.VarAt, ir.SumT)):
            unit = self.vars[node.name].unit
            return "int" if unit == units.INT else "bool" if unit == units.BOOL else "float"
        if isinstance(node, ir.MpField):
            return "float" if self.graph.fields[node.name].kind == "number" else "int"
        if isinstance(node, ir.TableLookup):
            return "int" if self.column(node.table, node.column).unit == units.INT else "float"
        if isinstance(node, ir.Unary):
            return "bool" if node.op == "not" else self.kind(node.a)
        if isinstance(node, ir.Binary):
            if node.op in _BOOLEAN_OPS:
                return "bool"
            if node.op in ("truediv", "pow"):
                return "float"
            return _promote(self.kind(node.a), self.kind(node.b))
        if isinstance(node, ir.Where):
            return _promote(self.kind(node.a), self.kind(node.b))
        if isinstance(node, ir.Call):
            if node.fn in _FLOAT_CALLS:
                return "float"
            if node.fn in ("minimum", "maximum"):
                return _promote(self.kind(node.args[0]), self.kind(node.args[1]))
            return self.kind(node.args[0])  # abs, as_cu
        return "float"  # ScenRead, ScenAt, SeriesRead, SeriesAt

    # ------------------------------------------------------------------ statements
    def store(self, name: str, row: str, value_node: ir.Node, ctx: _Ctx) -> list[str]:
        """``V_x[row, p, s] = <value>`` after the checked reads it needs; an Int variable
        takes a float value as NumPy's assignment casts it, truncated."""
        self.pre = []
        value = self.expr(value_node, ctx)
        if self.vars[name].unit == units.INT and self.kind(value_node) == "float":
            value = f"np.int64({value})"
        return [*self.pre, f"{self.var_array(name)}[{row}, p, s] = {value}"]

    def kernel(self, index: int, comp: Component) -> tuple[str, str]:
        """The source of component ``index``'s kernel, and the wrapper's call of it."""
        self.k = _Kernel()
        for name in comp.members:
            self.var_array(name)  # the outputs first
        groups: list[tuple[str | None, str | None, list[str]]] = []
        if comp.mode == STEP:
            self.step_groups(comp, groups)
        else:
            self.whole_groups(comp, groups)
        dims: frozenset[str] = frozenset()
        for name in comp.members:
            dims = dims | (self.vars[name].dims or frozenset())
        policy = ir.POLICY in dims
        parallel = policy and PARALLEL
        lanes_p = "P" if policy else "1"
        lanes_s = "S" if ir.SCENARIO in dims else "1"
        nb, nl = len(self.k.boundary), len(self.k.loop)
        body: list[str] = [
            f"    for _b in {'numba.prange' if parallel else 'range'}"
            f"(({lanes_p} + {POLICY_BLOCK - 1}) // {POLICY_BLOCK}):",
            f"        _lo = _b * {POLICY_BLOCK}",
            f"        _hi = min(_lo + {POLICY_BLOCK}, {lanes_p})",
        ]
        for header, ordinal, stmts in groups:
            pad = "        "
            if header is not None:
                body.append(pad + header)
                pad += "    "
                if ordinal is not None and nl:
                    body.append(f"{pad}_q = {nb + 1} + {ordinal} * {nl}")
            body.append(f"{pad}for p in range(_lo, _hi):")
            body.append(f"{pad}    for s in range({lanes_s}):")
            body.extend(f"{pad}        {line}" for line in stmts)
        for name in self.k.args:
            self.inputs.setdefault(name, None)
        names = [*self.k.args, "H", "P", "S"]
        types = [*self.k.args.values(), "int64", "int64", "int64"]
        if nb or nl:
            names += ["E_pos", "E_lo", "E_hi"]
            types += [_LANES] * 3
        source = "\n".join(
            [
                f"def k_{index}({', '.join(names)}):",
                f'    """component {index}: {", ".join(comp.members)} ({comp.direction}, '
                f'{comp.mode})"""',
                *body,
            ]
        )
        first = self.vars[comp.members[0]].loc
        signature = "void(" + ", ".join(types) + ")"
        where = (first.file, first.line, first.function)
        self.kernel_entries.append(
            f"    ({f'k_{index}'!r}, {signature!r}, {parallel}, {comp.members!r}, {where!r}),"
        )
        if nb or nl:
            boundary = "".join(f"{c}, " for c in self.k.boundary)
            loop = "".join(f"{c}, " for c in self.k.loop)
            self.check_entries.append(f"    ({nb}, {nl}, ({boundary}), ({loop})),")
        else:
            self.check_entries.append("    None,")
        return source, f"k_{index}({', '.join(names)})"

    def whole_groups(
        self, comp: Component, groups: list[tuple[str | None, str | None, list[str]]]
    ) -> None:
        """A non-recursive component: each member's boundary values, then its formula over
        its range of months, or once per lane without a time dimension."""
        for name in comp.members:
            v = self.vars[name]
            self.current = name
            if not v.has_time:
                groups.append((None, None, self.store(name, "0", v.expr, _Ctx("scalar"))))
                continue
            edges: list[str] = []
            if v.init is not None:
                edges += self.store(name, "0", v.init, _Ctx("scalar"))
            if v.terminal is not None:
                edges += self.store(name, "H", v.terminal, _Ctx("scalar"))
            if edges:
                groups.append((None, None, edges))
            first, last = v.formula_range
            stop = "H + 1" if last == "H" else "H"
            stmts = self.store(name, "t", v.expr, _Ctx("whole"))
            groups.append((f"for t in range({first}, {stop}):", None, stmts))

    def step_groups(
        self, comp: Component, groups: list[tuple[str | None, str | None, list[str]]]
    ) -> None:
        """A recursive component, as the NumPy backend steps it: every member's boundary
        values, the pre-loop, then each month with every member inside."""
        pre: list[str] = []
        for name in comp.members:
            v = self.vars[name]
            self.current = name
            if v.init is not None:
                pre += self.store(name, "0", v.init, _Ctx("scalar"))
            if v.terminal is not None:
                pre += self.store(name, "H", v.terminal, _Ctx("scalar"))
        step = _Ctx("step")
        each = _Ctx("step", loop=True)
        loop: list[str] = []
        if comp.direction == ir.FORWARD:
            for name in comp.members:
                v = self.vars[name]
                if v.init is None and v.start == 0:
                    self.current = name
                    pre.append("t = 0")
                    pre += self.store(name, "0", v.expr, step)
            for name in comp.members:
                self.current = name
                loop += self.store(name, "t", self.vars[name].expr, each)
            header, ordinal = "for t in range(1, H + 1):", "(t - 1)"
        else:
            # the NumPy backend's pre-loop: a member without a terminal is evaluated at t = H
            # by its formula before the loop reads it at t + 1 (design section 4)
            for name in comp.members:
                v = self.vars[name]
                if v.terminal is None:
                    self.current = name
                    pre.append("t = H")
                    pre += self.store(name, "H", v.expr, step)
            for name in comp.members:
                v = self.vars[name]
                self.current = name
                stmts = self.store(name, "t", v.expr, each)
                if v.start > 0:
                    loop += [f"if t >= {v.start}:", *(f"    {line}" for line in stmts)]
                else:
                    loop += stmts
            header, ordinal = "for t in range(H - 1, -1, -1):", "(H - 1 - t)"
        if pre:
            groups.append((None, None, pre))
        groups.append((header, ordinal, loop))

    # ------------------------------------------------------------------ the module
    def generate(self) -> str:
        g = self.graph
        kernels: list[str] = []
        calls: list[str] = []
        for i, comp in enumerate(self.schedule):
            source, call = self.kernel(i, comp)
            kernels.append(source)
            calls.append(call)
        lines: list[str] = [
            '"""Generated by the Halley Numba backend. Do not edit: errors point at the model',
            f'source, {g.model}."""',
            "",
            "import numba",
            "import numpy as np",
            "",
            "from halley.codegen.numba_backend import as_input as _as_input",
            "from halley.codegen.numba_backend import as_table as _as_table",
            "from halley.codegen.numba_backend import raise_first as _raise_first",
            "from halley.codegen.numpy_backend import _check_finite, _check_prob",
            "",
            f"IR_SHA256 = {ir.graph_hash(g)!r}",
            f"MODEL = {g.model!r}",
            _HELPERS.rstrip("\n"),
        ]
        for source in kernels:
            lines += ["", "", source]
        lines += ["", "", "KERNELS = (", *self.kernel_entries, ")", ""]
        lines += ["_CHECKS = (", *self.check_entries, ")", "", ""]
        lines += self.wrapper(calls)
        return "\n".join(lines) + "\n"

    def wrapper(self, calls: list[str]) -> list[str]:
        """``run_batch(inp, H)``: the inputs as the kernels' arguments, the variables'
        arrays as the NumPy backend allocates them, the kernels in schedule order with the
        checks after each, the finiteness checks, the arrays."""
        g = self.graph
        out = [
            "def run_batch(inp, H):",
            "    P = inp['P']",
            "    S = inp['S']",
            "    Tn = H + 1",
            "    _cur = ''",
        ]
        for name in self.inputs:
            if name.startswith("M_"):
                fname = name[2:]
                dtype = "np.float64" if g.fields[fname].kind == "number" else "np.int64"
                out.append(f"    {name} = _as_input(inp['mp.{fname}'], {dtype})")
            elif name.startswith("T_"):
                table, column = name[2:].split("__", 1)
                unit = self.column(table, column).unit
                dtype = "np.int64" if unit == units.INT else "np.float64"
                out.append(f"    {name} = _as_table(inp['tbl.{table}.{column}'], {dtype})")
            elif name.startswith("Z_"):
                out.append(f"    {name} = _as_input(inp['scen.{name[2:]}'], np.float64)")
            elif name.startswith("u0_"):
                out.append(f"    {name} = int(inp['scen.{name[3:]}.u0'])")
            elif name.startswith("Y_"):
                out.append(f"    {name} = _as_input(inp['series.{name[2:]}'], np.float64)")
            elif name.startswith("R_"):
                rname = name[2:]
                value = "H" if rname == "horizon" else f"int(inp['run.{rname}'])"
                out.append(f"    {name} = {value}")
        lasts = ", ".join(f"{n!r}: Y_{n}.shape[0] - 1" for n in sorted(self.lasts))
        out += [
            f"    _last = {{{lasts}}}",
            "    E_pos = np.zeros((P, S), dtype=np.int64)",
            "    E_lo = np.zeros((P, S), dtype=np.int64)",
            "    E_hi = np.zeros((P, S), dtype=np.int64)",
            "    with np.errstate(all='ignore'):",
        ]
        for name in sorted(self.schedule.needed):
            v = self.vars[name]
            dims = v.dims or frozenset()
            shape = (
                f"({'Tn' if v.has_time else 1}, {'P' if ir.POLICY in dims else 1}, "
                f"{'S' if ir.SCENARIO in dims else 1})"
            )
            out.append(f"        V_{name} = np.zeros({shape}, dtype={_dtype(v)[0]})")
        for i, comp in enumerate(self.schedule):
            out.append(
                f"        # component: {', '.join(comp.members)} ({comp.direction}, {comp.mode})"
            )
            out.append(f"        _cur = {comp.members[0]!r}")
            out.append(f"        {calls[i]}")
            if "E_pos" in calls[i]:
                out.append("        if E_pos.any():")
                out.append(f"            _raise_first(_CHECKS[{i}], E_pos, E_lo, E_hi, H, _last)")
            for name in comp.members:
                v = self.vars[name]
                if v.unit in units.PROBS:
                    out.append(f"        _cur = {name!r}")
                    out.append(
                        f"        _check_prob(V_{name}, {v.start}, {name!r}, {_loc_str(v.loc)!r})"
                    )
        for name in sorted(self.schedule.needed):
            v = self.vars[name]
            if v.unit not in (units.INT, units.BOOL):
                out.append(
                    f"        _check_finite(V_{name}, {v.start}, {name!r}, {_loc_str(v.loc)!r})"
                )
        items = ", ".join(f"{n!r}: V_{n}" for n in sorted(self.schedule.needed))
        out.append(f"    return {{{items}}}")
        return out


def generate(graph: ir.Graph, schedule: Schedule) -> str:
    """The Python/Numba module for a batch of the model."""
    return _Generator(graph, schedule).generate()


# ------------------------------------------------------------------------------ compiling

_LOADED: dict[str, ModuleType] = {}
"""The modules compiled in this process, by the sha256 of their source."""


def _write(path: Path, source: str) -> None:
    """Write the module where Numba's cache can find it, unless the file already holds this
    text: rewriting it would change its timestamp and so invalidate the cached kernels."""
    data = source.encode("utf-8")
    try:
        if path.read_bytes() == data:
            return
    except OSError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}-", suffix=".tmp")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def compile_source(source: str, model: str) -> BatchFn:
    """Write the generated module to :func:`cache_dir` ``/ <ir_hash>.py``, import it and
    compile its kernels for their signatures (or load them from Numba's cache); its
    ``run_batch``. A kernel Numba cannot compile is a BuildError at the model line of the
    component's first variable."""
    numba = _numba()
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    module = _LOADED.get(digest)
    if module is None:
        found = re.search(r"^IR_SHA256 = '([0-9a-f]{64})'$", source, re.M)
        if found is None:
            raise BuildError(f"the Numba backend's source for {model} names no IR_SHA256")
        ir_hash = found.group(1)
        path = cache_dir() / f"{ir_hash}.py"
        _write(path, source)
        name = f"halley_numba_{ir_hash}"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:  # pragma: no cover - a .py path always has one
            raise BuildError(f"cannot import the Numba backend's module for {model} from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module  # Numba's cache imports the kernels' module by name
        try:
            spec.loader.exec_module(module)
            _jit(module, numba)
        except BaseException:
            del sys.modules[name]
            raise
        _LOADED[digest] = module
    fn: BatchFn = module.run_batch
    return fn


def _jit(module: ModuleType, numba: Any) -> None:
    """Compile every kernel of the module for its signature, replacing the Python function."""
    kernels: tuple[Any, ...] = module.KERNELS
    for name, signature, parallel, members, where in kernels:
        py_func = getattr(module, name)
        try:
            jitted = numba.njit(
                signature,
                parallel=parallel,
                cache=True,
                fastmath=False,
                error_model="numpy",
                boundscheck=False,
            )(py_func)
        except Exception as exc:
            text = str(exc).strip()
            first = text.splitlines()[0] if text else ""
            raise BuildError(
                f"the Numba backend could not compile the kernel of {', '.join(members)}: "
                f"{type(exc).__name__}: {first}",
                Loc(*where),
            ) from exc
        setattr(module, name, jitted)


class NumbaBackend:
    """The Numba backend as a :class:`~halley.codegen.Backend`."""

    name = NAME
    artefact = ARTEFACT

    def generate(self, graph: ir.Graph, schedule: Schedule) -> str:
        _numba()  # say how to install it before anything else
        return generate(graph, schedule)

    def compile(self, source: str, model: str) -> BatchFn:
        return compile_source(source, model)

    def run_batch(
        self,
        fn: BatchFn,
        graph: ir.Graph,
        inp: dict[str, Any],
        horizon: int,
        lanes: str = "policies",
    ) -> dict[str, NDArray[Any]]:
        return run_batch(fn, graph, inp, horizon, lanes)

    def versions(self) -> dict[str, str]:
        """What computes the values: NumPy (the wrapper's arrays and checks), Numba and
        llvmlite (the kernels), whether Numba uses Intel's SVML for vectorised maths, and the
        processor LLVM compiled for."""
        numba = _numba()
        llvmlite = importlib.import_module("llvmlite")
        return {
            "numpy": np.__version__,
            "numba": str(numba.__version__),
            "llvmlite": str(llvmlite.__version__),
            "using_svml": str(bool(numba.config.USING_SVML)),
            "processor": platform.processor(),
        }
