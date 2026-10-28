"""Invariants, checked on every batch of every run (Spec, Testing and validation).

The built-in ones run on the roles a model names through ``role=`` on ``@var``, with the per-unit
tolerance scales the model declares:

* H1: policies in force never negative and never increasing (no new business in v1).
* H2: decrement probabilities in [0, 1] (checked where they are computed) and total decrements
  never above the opening in-force count.
* H3: terminal values reached: every backward variable equals its terminal value at the horizon.

The product's own follow (:class:`~halley.authoring.contracts.Invariant`, in declaration order):
the term assurance's reserve roll-forward and PV identity (H4, H5), the annuity's I3 to I10.
Every invariant is evaluated per batch on a :class:`BatchView`; the results of the batches are
merged into one :class:`InvariantResult` per invariant, in ascending batch order, so the worst
residual covers every policy and the first failing policy is deterministic. A failure is a
:class:`~halley.errors.RunError` naming the invariant, the policy and the month; in an
aggregation stage, whose model points are groups, the group.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from halley import ir
from halley.authoring.contracts import Invariant
from halley.errors import RunError
from halley.ir import units
from halley.ir.units import unit_from_name
from halley.runtime.loader import Inputs, lane_noun
from halley.runtime.tolerance import RELATIVE


@dataclass(frozen=True, slots=True)
class InvariantResult:
    name: str
    passed: bool
    worst: float
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def merge_results(batches: Sequence[Sequence[InvariantResult]]) -> list[InvariantResult]:
    """One result per invariant over every batch, in ascending batch order: passed when every
    batch passed, the worst residual over the batches, and the detail of the first failing
    batch (or of the first batch when all pass)."""
    if not batches:
        return []
    names = [r.name for r in batches[0]]
    for results in batches[1:]:
        if [r.name for r in results] != names:
            raise RunError("the batches did not check the same invariants")
    merged: list[InvariantResult] = []
    for i, name in enumerate(names):
        column = [results[i] for results in batches]
        failed = [r for r in column if not r.passed]
        detail = failed[0].detail if failed else column[0].detail
        # a residual that is not a number is the worst there is, whichever batch it came from
        worst = max(math.inf if math.isnan(r.worst) else r.worst for r in column)
        merged.append(InvariantResult(name, not failed, worst, detail))
    return merged


def _first_bad(
    mask: NDArray[np.bool_], ids: NDArray[Any], t_offset: int = 0, lane: str = "policy"
) -> str:
    """``policy P05 at t = 3`` (``group G0 at t = 3`` in a stage, by ``lane``) for the first
    True in a (time, policy, scenario) mask."""
    idx = np.argwhere(mask)
    if idx.size == 0:
        return ""
    t, p, s = (int(x) for x in idx[0])
    p = min(p, len(ids) - 1)
    where = f"{lane} {ids[p]} at t = {t + t_offset}"
    if mask.shape[2] > 1:
        where += f", scenario {s}"
    return where


def check_builtin(
    graph: ir.Graph,
    out: dict[str, NDArray[Any]],
    ids: NDArray[Any],
    horizon: int,
    scales: dict[units.Unit, NDArray[np.float64]],
    lanes: str = "policies",
) -> list[InvariantResult]:
    """H1 to H3 on one batch; a failure names the first failing lane by its id under ``lanes``
    (``"groups"`` in an aggregation stage)."""
    lane = lane_noun(lanes)
    results: list[InvariantResult] = []
    roles: dict[str, list[ir.Variable]] = {}
    for v in graph.variables.values():
        if v.role is not None and v.name in out:
            roles.setdefault(v.role, []).append(v)
    in_force = roles.get("in_force", [None])[0]
    decrements = roles.get("decrement", [])
    count_scale = scales.get(units.COUNT)
    scale3 = (
        np.ones((1, 1, 1))
        if count_scale is None
        else np.asarray(count_scale, dtype=np.float64).reshape(1, -1, 1)
    )

    if in_force is not None:
        x = out[in_force.name]
        neg = x < 0
        rising = x[1:] > x[:-1]
        passed = not neg.any() and not rising.any()
        worst = float(max(-x.min(), (x[1:] - x[:-1]).max(), 0.0))
        detail = ""
        if neg.any():
            detail = f"{in_force.name} is negative for {_first_bad(neg, ids, 0, lane)}"
        elif rising.any():
            detail = f"{in_force.name} increases for {_first_bad(rising, ids, 1, lane)}"
        results.append(
            InvariantResult("H1 in force never negative, never increasing", passed, worst, detail)
        )
        if decrements:
            total = sum((out[d.name][1:] for d in decrements), np.zeros_like(x[1:]))
            opening = x[:-1]
            over = total - opening
            bad = over > RELATIVE * scale3
            passed = not bad.any()
            results.append(
                InvariantResult(
                    "H2 decrements never exceed the opening in-force count",
                    passed,
                    float(max(over.max(), 0.0)),
                    f"decrements exceed {in_force.name}(t-1) for {_first_bad(bad, ids, 1, lane)}"
                    if not passed
                    else "",
                )
            )
    probs = [v.name for v in graph.variables.values() if v.unit in units.PROBS and v.name in out]
    results.append(
        InvariantResult(
            "H2 probabilities in [0, 1]",
            True,
            0.0,
            f"checked as computed: {', '.join(probs)}" if probs else "no probability variables",
        )
    )
    for v in graph.variables.values():
        if v.terminal is None or v.name not in out:
            continue
        if isinstance(v.terminal, ir.Const) and not isinstance(v.terminal.value, bool):
            end = out[v.name][horizon]
            target = float(v.terminal.value)
            bad = end != target
            detail = ""
            if bad.any():
                detail = f"{v.name}(H) != {target} for {_first_bad(bad[None], ids, horizon, lane)}"
            results.append(
                InvariantResult(
                    f"H3 {v.name} reaches its terminal value at t = {horizon}",
                    not bad.any(),
                    float(np.abs(end - target).max()) if end.size else 0.0,
                    detail,
                )
            )
    failed = [r for r in results if not r.passed]
    if failed:
        r = failed[0]
        raise RunError(f"built-in invariant failed: {r.name}: {r.detail}")
    return results


# ------------------------------------------------------------------------- product invariants


def _read_only(values: NDArray[Any]) -> NDArray[Any]:
    """A read-only view of ``values`` (no copy): what an invariant sees of the loaded inputs,
    so a slip such as ``a.mp.premium[...] = x`` raises instead of corrupting the model points
    for every later batch and for the run's result."""
    view = values.view()
    view.flags.writeable = False
    return view


class _Fields:
    """``a.mp``: a model point field of the batch, shaped ``(1, P, 1)``, read-only."""

    def __init__(self, columns: dict[str, NDArray[Any]], lo: int, hi: int) -> None:
        self._columns = columns
        self._lo = lo
        self._hi = hi

    def __getattr__(self, name: str) -> NDArray[Any]:
        if name.startswith("_"):
            raise AttributeError(name)
        col = self._columns.get(name)
        if col is None:
            raise RunError(
                f"a.mp.{name}: the model point schema has no field {name!r}; it has "
                f"{', '.join(sorted(self._columns))}"
            )
        return _read_only(np.asarray(col[self._lo : self._hi]).reshape(1, -1, 1))


class _Run:
    """``a.run``: the run constants."""

    def __init__(self, values: dict[str, int]) -> None:
        self._values = values

    def __getattr__(self, name: str) -> int:
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in self._values:
            raise RunError(
                f"a.run.{name}: the run has no constant {name!r}; it has {', '.join(self._values)}"
            )
        return self._values[name]


class BatchView:
    """One batch after projection, as a product invariant sees it (the
    :class:`~halley.authoring.contracts.BatchView` protocol). Arrays are ``(time, policy,
    scenario)``; a variable is broadcast to ``(H + 1, P, S)``, or ``(1, P, S)`` without a time
    dimension, and memoised. ``lanes`` names the batch's lanes in messages: ``"policies"``, or
    ``"groups"`` in an aggregation stage."""

    def __init__(
        self,
        graph: ir.Graph,
        out: dict[str, NDArray[Any]],
        inputs: Inputs,
        lo: int,
        hi: int,
        scales: dict[units.Unit, NDArray[np.float64]],
        lanes: str = "policies",
    ) -> None:
        self._graph = graph
        self._out = out
        self._inputs = inputs
        self._scales = scales
        self._cache: dict[str, NDArray[Any]] = {}
        self.H = inputs.H
        self.P = hi - lo
        self.S = inputs.n_scenarios
        # read-only like every array the view returns: the ids are the loaded inputs' own (a
        # write would rename the policies of the run's result), and t is shared by every
        # invariant of the batch
        self.ids = _read_only(inputs.ids[lo:hi])
        self.lanes = lanes
        self.t = _read_only(np.arange(self.H + 1, dtype=np.int64).reshape(-1, 1, 1))
        self.mp = _Fields(inputs.model_points.columns, lo, hi)
        self.run = _Run(inputs.run)

    def var(self, name: str) -> NDArray[Any]:
        cached = self._cache.get(name)
        if cached is not None:
            return cached
        arr = self._out.get(name)
        if arr is None:
            if name in self._graph.variables:
                raise RunError(
                    f"a.var({name!r}): {name} was not generated for this run; name it in the "
                    "invariant's reads"
                )
            raise RunError(f"a.var({name!r}): the model has no variable {name!r}")
        full = np.broadcast_to(arr, (arr.shape[0], self.P, self.S))
        self._cache[name] = full
        return full

    def at(self, name: str, index: Any) -> NDArray[Any]:
        values = self.var(name)
        if values.shape[0] == 1:
            raise RunError(f"a.at({name!r}, ...): {name} has no time dimension; use a.var")
        if isinstance(index, (int, np.integer)):
            i = int(index)  # pyright: ignore[reportUnknownArgumentType]
            if i < 0 or i > self.H:
                raise RunError(f"a.at({name!r}, {i}): month {i} is outside 0..{self.H}")
            return values[i : i + 1]
        given = np.asarray(index)
        if not np.issubdtype(given.dtype, np.integer):
            # the model's at() takes an Int index (D11); a float here would be truncated silently
            raise RunError(f"a.at({name!r}, index): the index is {given.dtype}, not an integer")
        idx = given.astype(np.int64).reshape(1, -1, 1)
        if idx.shape[1] != self.P:
            raise RunError(
                f"a.at({name!r}, index): the index has {idx.shape[1]} {self.lanes}, the batch "
                f"{self.P}"
            )
        if (idx < 0).any() or (idx > self.H).any():
            raise RunError(f"a.at({name!r}, index): a month is outside 0..{self.H}")
        return _read_only(
            np.take_along_axis(values, np.broadcast_to(idx, (1, self.P, self.S)), axis=0)
        )

    def _scenario(self, name: str, lo: int, hi: int) -> NDArray[np.float64]:
        sc = self._inputs.scenarios.get(name)
        if sc is None:
            raise RunError(f"the model declares no scenario input {name!r}")
        if lo < sc.u0 or hi > sc.u_max or lo > hi:
            raise RunError(
                f"scenario input {name} covers months {sc.u0} to {sc.u_max}, not {lo} to {hi}"
            )
        return _read_only(sc.values[lo - sc.u0 : hi - sc.u0 + 1])

    def scen(self, name: str, u: int) -> NDArray[np.float64]:
        return self._scenario(name, u, u)

    def scen_range(self, name: str, lo: int, hi: int) -> NDArray[np.float64]:
        return self._scenario(name, lo, hi)

    def table(self, name: str, column: str) -> NDArray[Any]:
        t = self._inputs.tables.get(name)
        if t is None:
            raise RunError(f"the model declares no assumption table {name!r}")
        if column not in t.values:
            raise RunError(f"table {name} has no column {column!r}")
        return _read_only(t.values[column])

    def scale(self, unit: str) -> NDArray[np.float64]:
        s = self._scales.get(unit_from_name(unit))
        if s is None:
            raise RunError(
                f"a.scale({unit!r}): the model declares no tolerance scale for {unit} "
                "(tolerance_scales)"
            )
        return _read_only(np.asarray(s, dtype=np.float64).reshape(1, -1, 1))


def _as_array(value: Any, dtype: Any) -> NDArray[Any]:
    arr = np.asarray(value, dtype=dtype)
    while arr.ndim < 3:
        arr = arr[None]
    if arr.ndim > 3:
        raise RunError(f"an invariant's array has {arr.ndim} axes; (time, policy, scenario) are 3")
    return arr


def _mask(inv: Invariant, a: BatchView) -> NDArray[np.bool_] | None:
    """The ``where`` mask as a boolean array; a mask of any other dtype is refused, as the
    loader refuses a rule that does not return booleans, so a forgotten comparison cannot
    silently mask every month out."""
    if inv.where is None:
        return None
    raw = np.asarray(inv.where(a))
    if raw.dtype != np.bool_:
        raise RunError(f"where must return booleans, not {raw.dtype}")
    return _as_array(raw, np.bool_)


def _evaluate(inv: Invariant, a: BatchView) -> InvariantResult:
    try:
        lhs = _as_array(inv.lhs(a) if callable(inv.lhs) else inv.lhs, np.float64)
        rhs = _as_array(inv.rhs(a) if callable(inv.rhs) else inv.rhs, np.float64)
        lhs, rhs = np.broadcast_arrays(lhs, rhs)
        scale = _as_array(inv.scale(a, lhs, rhs) if callable(inv.scale) else inv.scale, np.float64)
        mask = _mask(inv, a)
    except RunError as exc:
        raise RunError(f"invariant {inv.name}: {exc.message}", inv.loc) from exc
    except Exception as exc:  # the invariant's own failure
        raise RunError(f"invariant {inv.name}: {type(exc).__name__}: {exc}", inv.loc) from exc
    if inv.kind == "eq":
        resid = np.abs(lhs - rhs)
    elif inv.kind == "ge":
        resid = rhs - lhs
    else:
        resid = lhs - rhs
    try:
        bound = inv.tol * scale
        # the spec's rule is |a - b| <= tol * scale, which is false for a residual that is not a
        # number (0 / 0, inf - inf), so a NaN or an infinite residual fails
        bad = ~(resid <= bound)
        if mask is not None:
            bad = bad & mask
            resid = np.where(mask, resid, 0.0)
        shape = bad.shape
        lhs, rhs, bound, resid = (np.broadcast_to(x, shape) for x in (lhs, rhs, bound, resid))
    except ValueError as exc:
        raise RunError(f"invariant {inv.name}: its arrays do not line up: {exc}", inv.loc) from exc
    if shape[1] not in (1, a.P) or shape[2] not in (1, a.S):
        raise RunError(
            f"invariant {inv.name}: its arrays are shaped {shape}; the batch has {a.P} "
            f"{a.lanes} and {a.S} scenarios",
            inv.loc,
        )
    first, last = inv.window(a.H)
    if last < first:
        raise RunError(
            f"invariant {inv.name}: months ({first}, {last}) is an empty window at H = {a.H}",
            inv.loc,
        )
    if first < 0 or last > a.H:
        raise RunError(
            f"invariant {inv.name}: months ({first}, {last}) is outside 0..{a.H} at H = {a.H}",
            inv.loc,
        )
    n_window = last - first + 1
    n_time = shape[0]
    if inv.months is not None and n_time == a.H + 1 and n_window != n_time:
        sl = slice(first, last + 1)
        lhs, rhs, bound, resid, bad = (x[sl] for x in (lhs, rhs, bound, resid, bad))
        n_time = n_window
    elif n_time not in (1, n_window):
        hint = "" if inv.months is not None else "; declare months=(first, last) for a window"
        raise RunError(
            f"invariant {inv.name}: its arrays have {n_time} months, not {n_window}{hint}",
            inv.loc,
        )
    if resid.size == 0:
        raise RunError(f"invariant {inv.name}: its arrays are empty; nothing was checked", inv.loc)
    finite = np.isfinite(resid)
    worst = float(max(resid[finite].max(), 0.0)) if finite.any() else 0.0
    if not finite.all():
        worst = math.inf
    if not bad.any():
        return InvariantResult(inv.name, True, worst, "")
    i, p, s = (int(x) for x in np.argwhere(bad)[0])
    parts: list[str] = []
    if shape[1] == a.P:
        parts.append(f"{lane_noun(a.lanes)} {a.ids[p]}")
    if n_time > 1 or n_window == 1:
        parts.append(f"t = {first + i}")
    elif inv.months is not None:
        # a one-row result under a longer window is one month of it, which one is not known
        parts.append(f"one month of {first}..{last}")
    if shape[2] > 1:
        parts.append(f"scenario {s}")
    where = ", ".join(parts) or "the batch"
    r = float(resid[i, p, s])
    verdict = (
        "is not a number"
        if math.isnan(r)
        else f"{r!r} exceeds the tolerance {float(bound[i, p, s])!r}"
    )
    detail = (
        f"{where}: lhs = {float(lhs[i, p, s])!r}, rhs = {float(rhs[i, p, s])!r}; the residual "
        f"{verdict}"
    )
    return InvariantResult(inv.name, False, worst, detail + (f" ({inv.doc})" if inv.doc else ""))


def check_declared(invariants: Sequence[Invariant], view: BatchView) -> list[InvariantResult]:
    """Evaluate the product's invariants on one batch, in declaration order, stopping at the
    first failure, which raises a :class:`~halley.errors.RunError` naming the invariant, the
    first failing policy (group, in a stage: ``view.lanes``) and month, located at the
    invariant's declaration (so a later invariant that cannot even be evaluated never hides
    it)."""
    results: list[InvariantResult] = []
    for inv in invariants:
        r = _evaluate(inv, view)
        if not r.passed:
            raise RunError(f"invariant {r.name} failed: {r.detail}", inv.loc)
        results.append(r)
    return results
