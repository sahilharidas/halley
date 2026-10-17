"""NumPy reference backend: the executable spec (D16).

The schedule is turned into Python source that evaluates every component with NumPy over a batch
of policies by scenarios, one array per variable shaped ``(time, policy, scenario)`` with a size-1
axis for each dimension the variable does not have. Non-recursive components are evaluated for
all months at once; recursive ones step month by month, forward from ``t = 1`` or backward from
the horizon. The source keeps the model's own grouping of every expression, so float64 results
have the order the model wrote (D19), and it is saved beside the run for inspection.

Run-time checks live here too: probabilities in [0, 1], lookup keys inside their table, results
finite. Each failure is a :class:`~halley.errors.RunError` that names the variable and its source
line, never a line of generated code.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from halley import ir
from halley.errors import BuildError, Loc, RunError
from halley.ir import units
from halley.scheduler import STEP, Schedule

BatchFn = Callable[[dict[str, Any], int], dict[str, NDArray[Any]]]

_BINOPS = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "truediv": "/",
    "floordiv": "//",
    "mod": "%",
    "pow": "**",
    "lt": "<",
    "le": "<=",
    "gt": ">",
    "ge": ">=",
    "eq": "==",
    "ne": "!=",
    "and": "&",
    "or": "|",
}

_CALLS = {
    "exp": "np.exp",
    "log": "np.log",
    "expm1": "np.expm1",
    "log1p": "np.log1p",
    "sqrt": "np.sqrt",
    "minimum": "np.minimum",
    "maximum": "np.maximum",
    "abs": "np.abs",
}


class LaneError(Exception):
    """Raised inside generated code: a check failed for the lanes in ``lanes``."""

    def __init__(self, message: str, variable: str, loc: str, lanes: NDArray[np.bool_]) -> None:
        super().__init__(message)
        self.message = message
        self.variable = variable
        self.loc = loc
        self.lanes = lanes


def _lanes(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Collapse a mask over (time, policy, scenario) to the offending policies."""
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m.any(axis=2).any(axis=0)
    elif m.ndim == 2:
        m = m.any(axis=1)
    return m


def _ck(key: Any, lo: int, hi: int, table: str, name: str, variable: str, loc: str) -> Any:
    """Check a lookup key against the table's declared range; return it for indexing."""
    k = np.asarray(key)
    bad = (k < lo) | (k > hi)
    if bad.any():
        worst = k[bad]
        raise LaneError(
            f"lookup key {name} of table {table} is outside its declared range [{lo}, {hi}]: "
            f"value {int(worst.min())} to {int(worst.max())}; no silent extrapolation",
            variable,
            loc,
            _lanes(bad),
        )
    return k


def _index3(index: Any) -> NDArray[np.intp]:
    idx = np.asarray(index)
    if idx.ndim < 3:
        idx = idx.reshape((1,) * (3 - idx.ndim) + idx.shape)
    return idx.astype(np.intp)


def _at(
    values: NDArray[Any], index: Any, horizon: int, start: int, name: str, variable: str, loc: str
) -> Any:
    idx = np.asarray(index)
    if idx.ndim == 0:
        i = int(idx)
        if i < start or i > horizon:
            raise LaneError(
                f"at({name}, {i}) is outside the domain of {name}, months {start}..{horizon}",
                variable,
                loc,
                np.ones((), dtype=bool),
            )
        return values[i]
    bad = (idx < start) | (idx > horizon)
    if bad.any():
        raise LaneError(
            f"at({name}, ...) reads a month outside the domain of {name}, months "
            f"{start}..{horizon}",
            variable,
            loc,
            _lanes(bad),
        )
    return np.take_along_axis(values, _index3(idx), axis=0)


def _scen_at(values: NDArray[Any], index: Any, u0: int, name: str, variable: str, loc: str) -> Any:
    idx = np.asarray(index) - u0
    if idx.ndim == 0:
        i = int(idx)
        if i < 0 or i >= values.shape[0]:
            raise LaneError(
                f"scen.{name}.at({i + u0}) is outside the scenario file's months",
                variable,
                loc,
                np.ones((), dtype=bool),
            )
        return values[i]
    bad = (idx < 0) | (idx >= values.shape[0])
    if bad.any():
        raise LaneError(
            f"scen.{name}.at(...) reads a month outside the scenario file",
            variable,
            loc,
            _lanes(bad),
        )
    return np.take_along_axis(values, _index3(idx), axis=0)


def _sum_t(values: NDArray[Any], start: int, horizon: int) -> NDArray[Any]:
    """Sequential sum over months ``start..horizon`` in ascending order (D19)."""
    return np.cumsum(values[start : horizon + 1], axis=0)[-1:]


def _check_prob(values: NDArray[Any], start: int, variable: str, loc: str) -> None:
    v = values[start:]
    bad = (v < 0.0) | (v > 1.0) | ~np.isfinite(v)
    if bad.any():
        raise LaneError(
            f"{variable} is a probability but takes values outside [0, 1]: "
            f"{float(np.nanmin(v)):.6g} to {float(np.nanmax(v)):.6g}",
            variable,
            loc,
            _lanes(bad),
        )


def _check_finite(values: NDArray[Any], start: int, variable: str, loc: str) -> None:
    v = values[start:]
    bad = ~np.isfinite(v)
    if bad.any():
        raise LaneError(
            f"{variable} is not finite (NaN or infinity) for some policy and month",
            variable,
            loc,
            _lanes(bad),
        )


_HELPERS: dict[str, object] = {
    "np": np,
    "_ck": _ck,
    "_at": _at,
    "_scen_at": _scen_at,
    "_sum_t": _sum_t,
    "_check_prob": _check_prob,
    "_check_finite": _check_finite,
}


# ------------------------------------------------------------- what a value is at run time

SCALAR = "scalar"
"""A Python or NumPy scalar in the generated code: a literal, a run constant, ``t`` inside a
recursive component, a lookup whose keys are all scalars, and arithmetic, comparisons or a
ufunc over those alone (``np.exp(0.5)`` is a NumPy scalar)."""
ZERO_D = "0-d"
"""A 0-d array: a keyless table's value, or ``np.where`` over scalars (so ``to_monthly_cf`` of a
scalar too). Arithmetic on it gives a scalar again; indexing a table with it gives a scalar."""
ARRAY = "array"
"""An array with at least one axis: a model point field, a variable, a scenario value, ``t``
outside a recursive component, and anything built from one of those."""


def value_kind(node: ir.Node, step: bool) -> str:
    """What the generated code holds for ``node``: :data:`SCALAR`, :data:`ZERO_D` or
    :data:`ARRAY`. ``step`` says whether the reader is a member of a recursive component, where
    ``t`` is a Python int and a model point field is one row of the batch, or not, where ``t``
    is the month column ``_t[first:last]``.

    Nothing in the generator reads this; it is the contract the ``explain`` interpreter mirrors
    for ``**``. NumPy's power takes a fast path for the exponents 0.5, 2 and -1 (sqrt, square,
    reciprocal) that differs from libm's pow in the last place. Measured on NumPy 2.5: a scalar
    exponent takes it when the base is an array, a 0-d one included, while a scalar base with a
    scalar exponent is scalar power (libm's pow); a 0-d exponent takes it against any base, a
    scalar one included. An array exponent takes it when its values reach the power loop
    broadcast, which the batch's shapes decide, not the formula: a model point field
    (``V_x[1:H+1] ** M_k``, or ``V_x[t] ** m_k`` in a recursive component) takes it in a batch of
    one policy and not in a batch of three; a scenario value read inside a recursive component
    takes it for several policies and one scenario; a scalar variable read as an exponent takes
    it at almost every shape; an exponent built from the month column alone (``v ** _t[0:H+1]``,
    ``-12 / _t[1:H+1]`` in a whole-mode ``ann_spot``) took it in none of the shapes measured.

    A scalar or 0-d exponent (a literal, ``t`` inside a recursive component, a keyless table's
    value) is therefore on one path at every batch size. An array exponent whose value is 0.5, 2
    or -1 may not be: its results are bit-identical between reruns of the same batching (D19)
    and agree within 1e-12 relative with another batching (D5), which is what the annuity
    specification's "Built-in invariants" promise. The interpreter takes libm's pow for every
    array exponent.
    """
    if isinstance(node, (ir.Const, ir.RunConst)):
        return SCALAR
    if isinstance(node, ir.TimeIndex):
        return SCALAR if step else ARRAY
    if isinstance(node, ir.TableLookup):
        if not node.keys:
            return ZERO_D  # the dense array of a keyless table has no axis
        kinds = [value_kind(k, step) for k in node.keys]
        return ARRAY if ARRAY in kinds else SCALAR
    if isinstance(node, ir.Unary):
        return ARRAY if value_kind(node.a, step) == ARRAY else SCALAR
    if isinstance(node, ir.Binary):
        kinds = [value_kind(node.a, step), value_kind(node.b, step)]
        return ARRAY if ARRAY in kinds else SCALAR
    if isinstance(node, ir.Where):
        kinds = [value_kind(n, step) for n in (node.cond, node.a, node.b)]
        return ARRAY if ARRAY in kinds else ZERO_D  # np.where always returns an array
    if isinstance(node, ir.Call):
        kinds = [value_kind(a, step) for a in node.args]
        if ARRAY in kinds:
            return ARRAY
        if node.fn == "as_cu":
            return kinds[0]  # emitted as its argument
        return ZERO_D if node.fn == "to_monthly_cf" else SCALAR  # to_monthly_cf is an np.where
    return ARRAY  # VarRead, VarAt, SumT, MpField, ScenRead, ScenAt


def backend_scalar(node: ir.Node, step: bool) -> bool:
    """Whether the generated code holds ``node`` as a Python or NumPy scalar, so that ``node **
    k`` there is libm's pow and never NumPy's array fast path (see :func:`value_kind`)."""
    return value_kind(node, step) == SCALAR


# ---------------------------------------------------------------------------- generation


@dataclass(frozen=True, slots=True)
class _Ctx:
    mode: str  # "whole", "step" or "scalar"
    first: int = 0
    last: str = "H"


def _bound(last: str, k: int) -> str:
    """The exclusive upper bound of rows ``.. last + k`` as a Python expression in ``H``."""
    hi = k + 1 if last == "H" else k
    return f"H{hi:+d}" if hi else "H"


def _loc_str(loc: Loc | None) -> str:
    return "" if loc is None else str(loc)


class _Generator:
    def __init__(self, graph: ir.Graph, schedule: Schedule) -> None:
        self.graph = graph
        self.schedule = schedule
        self.vars = graph.variables
        self.lines: list[str] = []
        self.current = ""

    # ------------------------------------------------------------------ names
    @staticmethod
    def arr(name: str) -> str:
        return f"V_{name}"

    @staticmethod
    def table(table: str, column: str) -> str:
        return f"T_{table}__{column}"

    @staticmethod
    def mp(name: str, ctx: _Ctx) -> str:
        return f"m_{name}" if ctx.mode == "step" else f"M_{name}"

    # ------------------------------------------------------------------ expressions
    def expr(self, node: ir.Node, ctx: _Ctx) -> str:
        if isinstance(node, ir.Const):
            if isinstance(node.value, bool):
                return "True" if node.value else "False"
            if isinstance(node.value, str):
                raise BuildError("a string constant can only be a table key", node.loc)
            return repr(node.value)
        if isinstance(node, ir.TimeIndex):
            if ctx.mode == "step":
                return "t"
            if ctx.mode == "whole":
                return f"_t[{ctx.first}:{_bound(ctx.last, 0)}]"
            raise BuildError("t appears in a variable without a time dimension", node.loc)
        if isinstance(node, ir.VarRead):
            return self.var_read(node, ctx)
        if isinstance(node, ir.VarAt):
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
                return (
                    f"_at({self.arr(node.name)}, {i}, H, {v.start}, {node.name!r}, "
                    f"{self.current!r}, {_loc_str(node.loc)!r})"
                )
            return (
                f"_at({self.arr(node.name)}, {self.expr(index, ctx)}, H, {v.start}, "
                f"{node.name!r}, {self.current!r}, {_loc_str(node.loc)!r})"
            )
        if isinstance(node, ir.SumT):
            v = self.vars[node.name]
            return f"_sum_t({self.arr(node.name)}, {v.start}, H)"
        if isinstance(node, ir.MpField):
            return self.mp(node.name, ctx)
        if isinstance(node, ir.TableLookup):
            return self.lookup(node, ctx)
        if isinstance(node, ir.ScenRead):
            k = node.offset
            if ctx.mode == "step":
                return f"Z_{node.name}[t{k:+d}-u0_{node.name}]"
            if ctx.mode == "whole":
                return (
                    f"Z_{node.name}[{ctx.first + k}-u0_{node.name}:"
                    f"{_bound(ctx.last, k)}-u0_{node.name}]"
                )
            raise BuildError(
                f"scen.{node.name}(t) appears in a variable without a time dimension", node.loc
            )
        if isinstance(node, ir.ScenAt):
            return (
                f"_scen_at(Z_{node.name}, {self.expr(node.index, ctx)}, u0_{node.name}, "
                f"{node.name!r}, {self.current!r}, {_loc_str(node.loc)!r})"
            )
        if isinstance(node, ir.RunConst):
            return f"R_{node.name}"
        if isinstance(node, ir.Unary):
            op = "-" if node.op == "neg" else "~"
            return f"({op}{self.expr(node.a, ctx)})"
        if isinstance(node, ir.Binary):
            return f"({self.expr(node.a, ctx)} {_BINOPS[node.op]} {self.expr(node.b, ctx)})"
        if isinstance(node, ir.Where):
            return (
                f"np.where({self.expr(node.cond, ctx)}, {self.expr(node.a, ctx)}, "
                f"{self.expr(node.b, ctx)})"
            )
        if isinstance(node, ir.Call):
            args = [self.expr(a, ctx) for a in node.args]
            return self.call(node.fn, args)
        raise BuildError(f"unknown node {type(node).__name__}", node.loc)

    @staticmethod
    def call(fn: str, args: list[str]) -> str:
        """The fixed emitted form of each formula function (annuity spec, "Compounding and unit
        conversions"); the ``explain`` interpreter evaluates the same expressions."""
        if fn == "to_monthly":
            return f"(((1 + {args[0]}) ** (1/12)) - 1)"
        if fn == "to_monthly_prob":
            return f"(1 - ((1 - {args[0]}) ** (1/12)))"
        if fn == "to_monthly_cf":
            q = args[0]
            return (
                f"np.where({q} >= 1, 1.0, -np.expm1(np.log1p(-np.where({q} >= 1, 0.0, {q})) / 12))"
            )
        if fn == "annual_step":
            return f"(1 + {args[0]})"
        if fn == "cc_discount":
            return f"np.exp(((-{args[0]}) * {args[1]}) / 12)"
        if fn == "fwd_rate":
            return f"(({args[0]} / {args[1]}) - 1)"
        if fn == "ann_fwd":
            return f"((({args[0]} / {args[1]}) ** 12) - 1)"
        if fn == "ann_spot":
            return f"(({args[0]} ** (-12 / {args[1]})) - 1)"
        if fn == "as_cu":
            return args[0]
        return f"{_CALLS[fn]}({', '.join(args)})"

    def var_read(self, node: ir.VarRead, ctx: _Ctx) -> str:
        v = self.vars[node.name]
        name = self.arr(node.name)
        if not v.has_time:
            return f"{name}[0]" if ctx.mode == "step" else name
        k = node.offset
        if ctx.mode == "step":
            return f"{name}[t{k:+d}]" if k else f"{name}[t]"
        if ctx.mode == "whole":
            return f"{name}[{ctx.first + k}:{_bound(ctx.last, k)}]"
        raise BuildError(f"{node.name}(t) appears in a variable without a time dimension", node.loc)

    def lookup(self, node: ir.TableLookup, ctx: _Ctx) -> str:
        decl = self.graph.tables[node.table]
        parts: list[str] = []
        for key, key_node in zip(decl.keys, node.keys, strict=True):
            if key.enum is not None:
                if isinstance(key_node, ir.Const):
                    parts.append(str(key.enum.index(str(key_node.value))))
                else:
                    parts.append(self.expr(key_node, ctx))
            else:
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
                else:
                    parts.append(
                        f"(_ck({self.expr(key_node, ctx)}, {lo}, {hi}, {node.table!r}, "
                        f"{key.name!r}, {self.current!r}, {_loc_str(node.loc)!r}) - {lo})"
                    )
        name = self.table(node.table, node.column)
        if not parts:
            return name
        return f"{name}[{', '.join(parts)}]"

    # ------------------------------------------------------------------ statements
    def emit(self, line: str, indent: int = 1) -> None:
        self.lines.append("    " * indent + line)

    def set_current(self, name: str, indent: int = 1) -> None:
        self.current = name
        self.emit(f"_cur = {name!r}", indent)

    def generate(self) -> str:
        g = self.graph
        self.emit('"""Generated by the Halley NumPy backend. Do not edit: errors point at the', 0)
        self.emit(f'model source, {g.model}."""', 0)
        self.emit("", 0)
        self.emit("def run_batch(inp, H):", 0)
        self.emit("P = inp['P']")
        self.emit("S = inp['S']")
        self.emit("Tn = H + 1")
        self.emit("_t = np.arange(Tn, dtype=np.int64).reshape(Tn, 1, 1)")
        self.emit("_cur = ''")
        for name, decl in g.fields.items():
            if decl.readable:
                self.emit(f"M_{name} = inp['mp.{name}']")
                self.emit(f"m_{name} = M_{name}[0]")
        for tname, tdecl in g.tables.items():
            for col in tdecl.columns:
                self.emit(f"{self.table(tname, col.name)} = inp['tbl.{tname}.{col.name}']")
        for sname in g.scenarios:
            self.emit(f"Z_{sname} = inp['scen.{sname}']")
            self.emit(f"u0_{sname} = inp['scen.{sname}.u0']")
        for rname in self.run_constants():
            self.emit(f"R_{rname} = H" if rname == "horizon" else f"R_{rname} = inp['run.{rname}']")
        self.emit("with np.errstate(all='ignore'):")
        for name in sorted(self.schedule.needed):
            v = self.vars[name]
            dims = v.dims or frozenset()
            shape = (
                f"({'Tn' if v.has_time else 1}, {'P' if ir.POLICY in dims else 1}, "
                f"{'S' if ir.SCENARIO in dims else 1})"
            )
            self.emit(f"{self.arr(name)} = np.zeros({shape}, dtype={_dtype(v)})", 2)
        for comp in self.schedule:
            self.emit(f"# component: {', '.join(comp.members)} ({comp.direction}, {comp.mode})", 2)
            if comp.mode == STEP:
                self.step_component(comp.members, comp.direction)
            else:
                for name in comp.members:
                    self.whole(name)
            for name in comp.members:
                v = self.vars[name]
                if v.unit in units.PROBS:
                    self.emit(
                        f"_check_prob({self.arr(name)}, {v.start}, {name!r}, {_loc_str(v.loc)!r})",
                        2,
                    )
        for name in sorted(self.schedule.needed):
            v = self.vars[name]
            if v.unit not in (units.INT, units.BOOL):
                self.emit(
                    f"_check_finite({self.arr(name)}, {v.start}, {name!r}, {_loc_str(v.loc)!r})",
                    2,
                )
        items = ", ".join(f"{n!r}: {self.arr(n)}" for n in sorted(self.schedule.needed))
        self.emit(f"return {{{items}}}")
        return "\n".join(self.lines) + "\n"

    def run_constants(self) -> list[str]:
        """The run constants the needed variables read, in a fixed order."""
        names: set[str] = set()
        for name in self.schedule.needed:
            v = self.vars[name]
            for expr in (v.expr, v.init, v.terminal):
                if expr is None:
                    continue
                names.update(n.name for n in ir.walk(expr) if isinstance(n, ir.RunConst))
        return sorted(names)

    def whole(self, name: str) -> None:
        v = self.vars[name]
        arr = self.arr(name)
        self.set_current(name, 2)
        if not v.has_time:
            self.emit(f"{arr}[...] = {self.expr(v.expr, _Ctx('scalar'))}", 2)
            return
        if v.init is not None:
            self.emit(f"{arr}[0] = {self.expr(v.init, _Ctx('scalar'))}", 2)
        if v.terminal is not None:
            self.emit(f"{arr}[H] = {self.expr(v.terminal, _Ctx('scalar'))}", 2)
        first, last = v.formula_range
        ctx = _Ctx("whole", first, last)
        self.emit(f"{arr}[{first}:{_bound(last, 0)}] = {self.expr(v.expr, ctx)}", 2)

    def step_component(self, members: tuple[str, ...], direction: str) -> None:
        for name in members:
            v = self.vars[name]
            if v.init is not None:
                self.set_current(name, 2)
                self.emit(f"{self.arr(name)}[0] = {self.expr(v.init, _Ctx('scalar'))}", 2)
            if v.terminal is not None:
                self.set_current(name, 2)
                self.emit(f"{self.arr(name)}[H] = {self.expr(v.terminal, _Ctx('scalar'))}", 2)
        step = _Ctx("step")
        if direction == ir.FORWARD:
            for name in members:
                v = self.vars[name]
                if v.init is None and v.start == 0:
                    self.set_current(name, 2)
                    self.emit("t = 0", 2)
                    self.emit(f"{self.arr(name)}[0] = {self.expr(v.expr, step)}", 2)
            self.emit("for t in range(1, H + 1):", 2)
            for name in members:
                self.set_current(name, 3)
                self.emit(f"{self.arr(name)}[t] = {self.expr(self.vars[name].expr, step)}", 3)
        else:
            # the mirror of the forward pre-loop: a member without a terminal is defined at
            # t = H by its formula (the checker lets it read the others only within the month),
            # so it is evaluated there first, in same-period order, before the loop reads it at
            # t + 1 (the annuity's {bel, inv_exp})
            for name in members:
                v = self.vars[name]
                if v.terminal is None:
                    self.set_current(name, 2)
                    self.emit("t = H", 2)
                    self.emit(f"{self.arr(name)}[H] = {self.expr(v.expr, step)}", 2)
            self.emit("for t in range(H - 1, -1, -1):", 2)
            for name in members:
                v = self.vars[name]
                self.set_current(name, 3)
                if v.start > 0:
                    self.emit(f"if t >= {v.start}:", 3)
                    self.emit(f"{self.arr(name)}[t] = {self.expr(v.expr, step)}", 4)
                else:
                    self.emit(f"{self.arr(name)}[t] = {self.expr(v.expr, step)}", 3)


def _dtype(v: ir.Variable) -> str:
    if v.unit == units.INT:
        return "np.int64"
    if v.unit == units.BOOL:
        return "np.bool_"
    return "np.float64"


def generate(graph: ir.Graph, schedule: Schedule) -> str:
    """The Python/NumPy source for a batch of the model."""
    return _Generator(graph, schedule).generate()


def compile_source(source: str, model: str) -> BatchFn:
    namespace: dict[str, Any] = dict(_HELPERS)
    code = compile(source, f"<halley numpy backend: {model}>", "exec")
    exec(code, namespace)  # our own generated source
    fn: BatchFn = namespace["run_batch"]
    return fn


def run_batch(
    fn: BatchFn, graph: ir.Graph, inp: dict[str, Any], horizon: int
) -> dict[str, NDArray[Any]]:
    """Call the generated function; turn a failure into a :class:`RunError` that names the
    model variable and its source line."""
    try:
        return fn(inp, horizon)
    except LaneError as exc:
        v = graph.variables.get(exc.variable)
        lanes = _lane_ids(exc.lanes, inp)
        raise RunError(f"{exc.message}{lanes}", v.loc if v else None) from None
    except RunError:
        raise
    except Exception as exc:  # map anything else to the model source
        name = _current_from_traceback(exc)
        v = graph.variables.get(name) if name else None
        raise RunError(
            f"while computing {name or 'the model'}: {type(exc).__name__}: {exc}",
            v.loc if v else None,
        ) from exc


def _lane_ids(lanes: NDArray[np.bool_], inp: dict[str, Any]) -> str:
    ids = inp.get("policy_ids")
    if ids is None or lanes.ndim == 0 or lanes.shape[0] != len(ids):
        return ""
    bad = [str(i) for i in np.asarray(ids)[lanes]]
    if not bad:
        return ""
    shown = ", ".join(bad[:5]) + (f" and {len(bad) - 5} more" if len(bad) > 5 else "")
    return f"; policies {shown}"


def _current_from_traceback(exc: BaseException) -> str | None:
    tb = exc.__traceback__
    name: str | None = None
    while tb is not None:
        if tb.tb_frame.f_code.co_name == "run_batch" and "_cur" in tb.tb_frame.f_locals:
            name = str(tb.tb_frame.f_locals["_cur"])
        tb = tb.tb_next
    return name
