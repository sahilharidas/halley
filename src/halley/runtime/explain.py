"""Trace, explain and diff (Spec, Debugging).

* :func:`explain` shows, for one policy's variable at one month, the value, the formula source
  and the value of every input the formula read, evaluated for that lane with a small interpreter
  over the IR. The same interpreter is a second implementation of the model's semantics, so the
  tests also use it to cross-check the generated backend lane by lane.
* :func:`diff` compares two runs variable by variable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from halley import ir
from halley.codegen.numpy_backend import ARRAY, SCALAR, ZERO_D, float_power, value_kind
from halley.errors import HalleyError, Loc
from halley.runtime.run import RunResult


@dataclass(frozen=True, slots=True)
class Explanation:
    variable: str
    policy_id: object
    t: int | None
    scenario: int
    value: float | int | bool
    unit: str
    loc: Loc
    source: str
    inputs: tuple[tuple[str, float | int | bool | str], ...]
    """Each input the formula read, as written, with its value for this lane."""
    lane: str = "policy"
    """What a lane is: ``"policy"``, or ``"group"`` in an aggregation stage's run."""

    def __str__(self) -> str:
        when = f" at t = {self.t}" if self.t is not None else ""
        lines = [
            f"{self.variable}{when} for {self.lane} {self.policy_id} = {self.value!r} [{self.unit}]"
        ]
        lines.append(f"  defined at {self.loc}")
        for line in self.source.strip().splitlines():
            lines.append(f"    {line}")
        if self.inputs:
            lines.append("  inputs:")
            for text, value in self.inputs:
                lines.append(f"    {text} = {value!r}")
        return "\n".join(lines)


class Lane:
    """Evaluates IR nodes for one policy and scenario of a finished run."""

    def __init__(self, result: RunResult, policy_id: object, scenario: int = 0) -> None:
        if result.graph is None or result.inputs is None:
            raise HalleyError("explain needs a run made in this session (load_run has no model)")
        self.result = result
        self.graph = result.graph
        self.inputs = result.inputs
        self.p = result.index(policy_id)
        self.s = scenario
        self.H = result.H

    def value_at(self, name: str, t: int) -> float:
        a = self.result.arrays[name]
        v = self.graph.variables[name]
        if v.has_time and not 0 <= t <= self.H:
            raise HalleyError(f"{name}(t) read at t = {t}, outside 0..{self.H}")
        ti = t if a.shape[0] > 1 else 0
        pi = self.p if a.shape[1] > 1 else 0
        si = self.s if a.shape[2] > 1 else 0
        return a[ti, pi, si].item()

    def series_at(self, name: str, u: int) -> float:
        """A group series of the stage at month ``u`` for this lane's group and scenario (the
        scenario axis has size 1 after a mean)."""
        values = self.inputs.series[name].values
        if not 0 <= u < values.shape[0]:
            raise HalleyError(f"series.{name} read at t = {u}, outside 0..{values.shape[0] - 1}")
        si = self.s if values.shape[2] > 1 else 0
        return values[u, self.p, si].item()

    def eval(self, node: ir.Node, t: int | None, step: bool = False) -> Any:
        """The value of ``node`` for this lane at month ``t``. ``step`` says whether the reader
        is a member of a recursive component, which decides the path a power takes in the
        generated code (:func:`_pow`); the arithmetic itself is the same either way."""
        if isinstance(node, ir.Const):
            return node.value
        if isinstance(node, ir.TimeIndex):
            if t is None:
                raise HalleyError("t has no value in a variable without a time dimension")
            return t
        if isinstance(node, ir.VarRead):
            v = self.graph.variables[node.name]
            if not v.has_time:
                return self.value_at(node.name, 0)
            if t is None:
                raise HalleyError(f"{node.name}(t) read without a month")
            return self.value_at(node.name, t + node.offset)
        if isinstance(node, ir.VarAt):
            return self.value_at(node.name, int(self.eval(node.index, t, step)))
        if isinstance(node, ir.SumT):
            v = self.graph.variables[node.name]
            total = 0.0
            for u in range(v.start, self.H + 1):
                total += self.value_at(node.name, u)
            return total
        if isinstance(node, ir.MpField):
            mp = self.inputs.model_points
            if node.name in mp.codes:
                return mp.columns[node.name][self.p]
            return mp.columns[node.name][self.p].item()
        if isinstance(node, ir.TableLookup):
            table = self.inputs.tables[node.table]
            idx: list[int] = []
            for key, key_node in zip(table.decl.keys, node.keys, strict=True):
                value = self.eval(key_node, t, step)
                if key.enum is not None:
                    idx.append(key.enum.index(str(value)))
                else:
                    k = int(value)
                    lo, hi = int(key.lo or 0), int(key.hi or 0)
                    if not lo <= k <= hi:
                        raise HalleyError(
                            f"key {key.name} = {k} outside [{lo}, {hi}] of table {node.table}"
                        )
                    idx.append(k - lo)
            return table.values[node.column][tuple(idx)].item()
        if isinstance(node, ir.ScenRead):
            if t is None:
                raise HalleyError(f"scen.{node.name}(t) read without a month")
            sc = self.inputs.scenarios[node.name]
            return sc.values[t + node.offset - sc.u0, 0, self.s].item()
        if isinstance(node, ir.ScenAt):
            sc = self.inputs.scenarios[node.name]
            return sc.values[int(self.eval(node.index, t, step)) - sc.u0, 0, self.s].item()
        if isinstance(node, ir.SeriesRead):
            if node.offset is None:  # a series without a time dimension: its one row
                return self.series_at(node.name, 0)
            if t is None:
                raise HalleyError(f"series.{node.name}(t) read without a month")
            return self.series_at(node.name, t + node.offset)
        if isinstance(node, ir.SeriesAt):
            return self.series_at(node.name, int(self.eval(node.index, t, step)))
        if isinstance(node, ir.RunConst):
            return self.inputs.run[node.name]
        if isinstance(node, ir.Unary):
            a = self.eval(node.a, t, step)
            return -a if node.op == "neg" else (not a)
        if isinstance(node, ir.Binary):
            a = self.eval(node.a, t, step)
            b = self.eval(node.b, t, step)
            if node.op == "pow":
                if float_power(node):
                    a = float(a)  # an integer literal to an Int power, raised as a float
                return _pow(
                    a,
                    b,
                    _array_power(node.a, node.b, step),
                    array_exponent=value_kind(node.b, step) == ARRAY,
                )
            return _binary(node.op, a, b)
        if isinstance(node, ir.Where):
            a = self.eval(node.a, t, step)
            b = self.eval(node.b, t, step)
            return a if self.eval(node.cond, t, step) else b
        if isinstance(node, ir.Call):
            args = [self.eval(a, t, step) for a in node.args]
            kinds = [value_kind(a, step) for a in node.args]
            return _call(node.fn, args, kinds)
        raise HalleyError(f"unknown node {type(node).__name__}")


def _array_power(base: ir.Node, exponent: ir.Node, step: bool) -> bool:
    """Whether the generated code's ``base ** exponent`` is NumPy's power with a scalar-like
    exponent, which takes the fast paths: an array base (a 0-d one included) with a scalar or 0-d
    exponent, or a 0-d exponent against any base, a scalar one included
    (:func:`~halley.codegen.numpy_backend.value_kind`). A scalar base with a scalar exponent is
    scalar power, libm's pow as Python's ``**`` is; an array exponent goes through NumPy's power
    loop (:func:`_pow` says when the batch's shapes make NumPy decide otherwise)."""
    kb, ke = value_kind(base, step), value_kind(exponent, step)
    return ke != ARRAY and (kb != SCALAR or ke == ZERO_D)


def _binary(op: str, a: Any, b: Any) -> Any:
    """A binary operator on two lane values in Python arithmetic, which agrees with NumPy's
    element loops bit for bit except for the powers :func:`_pow` routes through NumPy."""
    try:
        if op == "add":
            return a + b
        if op == "sub":
            return a - b
        if op == "mul":
            return a * b
        if op == "truediv":
            return a / b
        if op == "floordiv":
            return a // b
        if op == "mod":
            return a % b
        if op == "pow":
            return a**b
        if op == "lt":
            return a < b
        if op == "le":
            return a <= b
        if op == "gt":
            return a > b
        if op == "ge":
            return a >= b
        if op == "eq":
            return a == b
        if op == "ne":
            return a != b
        if op == "and":
            return bool(a) and bool(b)
        if op == "or":
            return bool(a) or bool(b)
    except ZeroDivisionError:
        return math.nan  # a discarded where() branch may divide by zero
    raise HalleyError(f"unknown operator {op}")


def _pow(a: Any, b: Any, array_power: bool, array_exponent: bool = False) -> Any:
    """``a ** b`` through the path the generated code takes for this base and exponent.

    NumPy's power takes a fast path for ``** 0.5``, ``** 2`` and ``** -1`` (sqrt, square,
    reciprocal) that differs from its power loop in the last place for about one value in a
    thousand. ``array_power`` (:func:`_array_power`) says that the generated code raises with a
    scalar-like exponent through NumPy: an array base with a scalar or 0-d exponent
    (``ndarray ** 0.5``, a keyless table's 0-d value as the base included) or a 0-d exponent
    against any base (``T_rates__f[0] ** T_power__e``); the array form is then run on a
    one-element array so NumPy decides the fast path from the exponent's value exactly as it
    does for the batch. ``array_exponent`` says that the exponent is an array of the batch
    (``v_m ** _t[1:H+1]``, ``x(t) ** M_k``), which NumPy raises in its power loop: the power is
    then ``np.power`` on one-element arrays, which takes no fast path and runs the loop NumPy
    dispatches to on this machine, the C library's pow where NumPy has no vectorised power
    (Windows, Linux without AVX-512) and NumPy's own kernel where it has (an AVX-512 Linux
    machine), so the interpreter and the batch take one kernel. Otherwise the power is Python's
    ``**``, libm's pow, as the generated code's is for a scalar base with a scalar exponent
    (``T_mort__q[0, 3] ** 0.5`` for a constant-key lookup, ``(t / 12) ** 0.5`` inside a
    recursive component); an Int base to an Int exponent stays Python's exact integer power.

    An array exponent is the power loop in the generated code only when the batch's shapes do
    not take it to the fast path (:func:`~halley.codegen.numpy_backend.value_kind`). An
    exponent built from the month column alone (``v_m ** t``, ``-12 / _t[1:H+1]`` in a
    whole-mode ``ann_spot``) is the loop at every shape measured over a window of two or more
    months (over a one-month window, H = 1 for a formula from t = 1, the month column is a
    size-one array and takes the fast path at every batch size, so this interpreter differs
    from the stored value in the last place there), but a model point field exponent
    (``x(t) ** self.mp.k``) whose value is 0.5, 2 or -1 takes the fast path in a batch of one
    policy and not in a batch of three, and a scalar variable read as an exponent
    (``x ** alpha()``) takes it at almost every shape. Not mirrored: such a result depends on
    the batch composition in the last place, two batchings agree within 1e-12 relative (D5),
    and this interpreter agrees with the backend to the same tolerance, bit for bit only with a
    batch that took the loop.
    """
    if array_power:
        with np.errstate(all="ignore"):  # the generated code runs under the same errstate
            return (np.array([a]) ** b).item()
    if array_exponent and (isinstance(a, float) or isinstance(b, float)):
        with np.errstate(all="ignore"):
            return np.power(np.array([a]), np.array([b])).item()
    return _binary("pow", a, b)


def _ufunc(fn: str, x: Any) -> Any:
    """``np.<fn>`` of one value through the same NumPy loop the generated code runs over the
    batch's arrays, on a one-element contiguous array so that NumPy dispatches as it does for
    the batch (its own vectorised ``exp`` and ``log`` where this machine has them, the C
    library's otherwise), so that ``exp`` and friends agree with the backend bit for bit."""
    return getattr(np, fn)(np.array([x], dtype=np.float64)).item()


def _call(fn: str, args: list[Any], kinds: list[str]) -> Any:
    """Each formula function in the emitted form of the backend (``_Generator.call``), written
    with the same grouping so the two agree bit for bit (D19). ``kinds`` is what the generated
    code holds for each argument (:func:`~halley.codegen.numpy_backend.value_kind`), which
    routes the powers inside the helpers as :func:`_pow` routes ``**``: the base ``(1 + q)``,
    ``(1 - q)`` or ``(p0 / p1)`` is an array exactly when an argument is, and ``ann_spot``'s
    exponent ``(-12 / m)`` is a scalar unless ``m`` is an array."""
    try:
        if fn == "to_monthly":
            return _pow(1 + args[0], 1 / 12, kinds[0] == ARRAY) - 1
        if fn == "to_monthly_prob":
            return 1 - _pow(1 - args[0], 1 / 12, kinds[0] == ARRAY)
        if fn == "to_monthly_cf":
            q = args[0]
            inner = 0.0 if q >= 1 else q
            return 1.0 if q >= 1 else -_ufunc("expm1", _ufunc("log1p", -inner) / 12)
        if fn == "annual_step":
            return 1 + args[0]
        if fn == "cc_discount":
            return _ufunc("exp", ((-args[0]) * args[1]) / 12)
        if fn == "fwd_rate":
            return (args[0] / args[1]) - 1
        if fn == "ann_fwd":
            return _pow(args[0] / args[1], 12, ARRAY in kinds[:2]) - 1
        if fn == "ann_spot":
            array_power = kinds[0] != SCALAR and kinds[1] != ARRAY
            return _pow(args[0], -12 / args[1], array_power, kinds[1] == ARRAY) - 1
        if fn == "as_cu":
            return args[0]
        if fn in ("exp", "log", "expm1", "log1p", "sqrt"):
            return _ufunc(fn, args[0])
        if fn == "minimum":
            return min(args[0], args[1])
        if fn == "maximum":
            return max(args[0], args[1])
        if fn == "abs":
            return abs(args[0])
    except ZeroDivisionError:
        return math.nan  # a discarded where() branch may divide by zero
    raise HalleyError(f"unknown function {fn}")


def _leaves(node: ir.Node) -> list[ir.Node]:
    out: list[ir.Node] = []
    for n in ir.walk(node):
        if isinstance(
            n,
            (
                ir.VarRead,
                ir.VarAt,
                ir.SumT,
                ir.MpField,
                ir.TableLookup,
                ir.ScenRead,
                ir.ScenAt,
                ir.SeriesRead,
                ir.SeriesAt,
                ir.RunConst,
            ),
        ):
            out.append(n)
    return out


def explain(
    result: RunResult, policy_id: object, variable: str, t: int | None = None, scenario: int = 0
) -> Explanation:
    """The value of ``variable`` for one policy (a group, in an aggregation stage's run) at month
    ``t``, its formula and its inputs."""
    if result.graph is None:
        raise HalleyError("explain needs a run made in this session (load_run has no model)")
    v = result.graph.variables[variable]
    if v.has_time and t is None:
        raise HalleyError(f"{variable} has a time dimension; give t")
    if not v.has_time:
        t = None
    lane = Lane(result, policy_id, scenario)
    value = lane.value_at(variable, t or 0)
    step = v.direction != ir.NONE
    seen: dict[str, Any] = {}
    for leaf in _leaves(v.expr):
        text = ir.render(leaf)
        if text in seen:
            continue
        try:
            seen[text] = lane.eval(leaf, t, step)
        except HalleyError as exc:
            seen[text] = f"<{exc}>"
    return Explanation(
        variable,
        policy_id,
        t,
        scenario,
        value,
        str(v.unit),
        v.loc,
        v.source,
        tuple(seen.items()),
        "group" if result.manifest.get("outputs", {}).get("kind") == "aggregation" else "policy",
    )


def recompute(
    result: RunResult, policy_id: object, variable: str, t: int, scenario: int = 0
) -> Any:
    """Evaluate a variable's formula for one lane from the stored inputs, independently of the
    generated code. For boundary months the stored init or terminal is evaluated instead. The
    variable's direction says whether its formula runs inside a recursive component, which the
    interpreter needs only to take the same power path as the generated code."""
    if result.graph is None:
        raise HalleyError("recompute needs a run made in this session")
    v = result.graph.variables[variable]
    lane = Lane(result, policy_id, scenario)
    step = v.direction != ir.NONE
    if v.has_time:
        if t == 0 and v.init is not None:
            return lane.eval(v.init, None)  # a boundary is evaluated once, outside the loop
        if t == result.H and v.terminal is not None:
            return lane.eval(v.terminal, None)
        first, _last = v.formula_range
        if t < first:
            return 0.0
        return lane.eval(v.expr, t, step)
    return lane.eval(v.expr, None)


@dataclass(frozen=True, slots=True)
class Difference:
    variable: str
    max_abs: float
    policy_id: object
    t: int
    scenario: int
    a: float
    b: float


def diff(a: RunResult, b: RunResult, variables: list[str] | None = None) -> dict[str, Difference]:
    """Largest absolute difference per variable between two runs of the same policies."""
    if a.n_policies != b.n_policies or not np.array_equal(a.ids, b.ids):
        raise HalleyError("diff compares runs of the same policies in the same order")
    if a.H != b.H or a.n_scenarios != b.n_scenarios:
        raise HalleyError("diff compares runs with the same horizon and number of scenarios")
    names = variables or sorted(set(a.arrays) & set(b.arrays))
    out: dict[str, Difference] = {}
    for name in names:
        x: NDArray[Any] = np.asarray(a.var(name), dtype=np.float64)
        y: NDArray[Any] = np.asarray(b.var(name), dtype=np.float64)
        d = np.abs(x - y)
        if d.size == 0:
            continue
        flat = int(np.argmax(d))
        t, p, s = np.unravel_index(flat, d.shape)
        out[name] = Difference(
            name,
            float(d.max()),
            a.ids[int(p)],
            int(t),
            int(s),
            float(x[t, p, s]),
            float(y[t, p, s]),
        )
    return out
