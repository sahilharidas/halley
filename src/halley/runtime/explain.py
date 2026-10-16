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

    def __str__(self) -> str:
        when = f" at t = {self.t}" if self.t is not None else ""
        lines = [
            f"{self.variable}{when} for policy {self.policy_id} = {self.value!r} [{self.unit}]"
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

    def eval(self, node: ir.Node, t: int | None) -> Any:
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
            return self.value_at(node.name, int(self.eval(node.index, t)))
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
                value = self.eval(key_node, t)
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
            return sc.values[int(self.eval(node.index, t)) - sc.u0, 0, self.s].item()
        if isinstance(node, ir.Unary):
            a = self.eval(node.a, t)
            return -a if node.op == "neg" else (not a)
        if isinstance(node, ir.Binary):
            return _binary(node.op, self.eval(node.a, t), self.eval(node.b, t))
        if isinstance(node, ir.Where):
            a = self.eval(node.a, t)
            b = self.eval(node.b, t)
            return a if self.eval(node.cond, t) else b
        if isinstance(node, ir.Call):
            args = [self.eval(a, t) for a in node.args]
            return _call(node.fn, args)
        raise HalleyError(f"unknown node {type(node).__name__}")


def _binary(op: str, a: Any, b: Any) -> Any:
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


def _call(fn: str, args: list[Any]) -> Any:
    if fn == "to_monthly":
        return ((1 + args[0]) ** (1 / 12)) - 1
    if fn == "to_monthly_prob":
        return 1 - ((1 - args[0]) ** (1 / 12))
    if fn in ("exp", "log", "expm1", "log1p", "sqrt"):
        return getattr(math, fn)(args[0])
    if fn == "minimum":
        return min(args[0], args[1])
    if fn == "maximum":
        return max(args[0], args[1])
    if fn == "abs":
        return abs(args[0])
    raise HalleyError(f"unknown function {fn}")


def _leaves(node: ir.Node) -> list[ir.Node]:
    out: list[ir.Node] = []
    for n in ir.walk(node):
        if isinstance(
            n, (ir.VarRead, ir.VarAt, ir.SumT, ir.MpField, ir.TableLookup, ir.ScenRead, ir.ScenAt)
        ):
            out.append(n)
    return out


def explain(
    result: RunResult, policy_id: object, variable: str, t: int | None = None, scenario: int = 0
) -> Explanation:
    """The value of ``variable`` for one policy at month ``t``, its formula and its inputs."""
    if result.graph is None:
        raise HalleyError("explain needs a run made in this session (load_run has no model)")
    v = result.graph.variables[variable]
    if v.has_time and t is None:
        raise HalleyError(f"{variable} has a time dimension; give t")
    if not v.has_time:
        t = None
    lane = Lane(result, policy_id, scenario)
    value = lane.value_at(variable, t or 0)
    seen: dict[str, Any] = {}
    for leaf in _leaves(v.expr):
        text = ir.render(leaf)
        if text in seen:
            continue
        try:
            seen[text] = lane.eval(leaf, t)
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
    )


def recompute(
    result: RunResult, policy_id: object, variable: str, t: int, scenario: int = 0
) -> Any:
    """Evaluate a variable's formula for one lane from the stored inputs, independently of the
    generated code. For boundary months the stored init or terminal is evaluated instead."""
    if result.graph is None:
        raise HalleyError("recompute needs a run made in this session")
    v = result.graph.variables[variable]
    lane = Lane(result, policy_id, scenario)
    if v.has_time:
        if t == 0 and v.init is not None:
            return lane.eval(v.init, None)
        if t == result.H and v.terminal is not None:
            return lane.eval(v.terminal, None)
        first, _last = v.formula_range
        if t < first:
            return 0.0
        return lane.eval(v.expr, t)
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
