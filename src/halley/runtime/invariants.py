"""Built-in invariants, checked on every run (Spec, Testing and validation).

The model names its in-force count, its decrements and its reserve through ``role=`` on
``@var``; the checks below run on those roles with the per-unit tolerance scales the model
declares. A failure is a :class:`~halley.errors.RunError` naming the invariant, the policy and the
month.

* H1: policies in force never negative and never increasing (no new business in v1).
* H2: decrement probabilities in [0, 1] (checked where they are computed) and total decrements
  never above the opening in-force count.
* H3: terminal values reached: every backward variable equals its terminal value at the horizon.

The reserve roll-forward and the PV identity (H4, H5) need the model's accumulation rate and
timed flows, which the Halley notes do not yet bind (term spec 15.6); the term assurance
reconciliation test computes them in the spec's own form.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from halley import ir
from halley.errors import RunError
from halley.ir import units
from halley.runtime.tolerance import RELATIVE


@dataclass(frozen=True, slots=True)
class InvariantResult:
    name: str
    passed: bool
    worst: float
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _first_bad(mask: NDArray[np.bool_], ids: NDArray[Any], t_offset: int = 0) -> str:
    """``policy P05 at t = 3`` for the first True in a (time, policy, scenario) mask."""
    idx = np.argwhere(mask)
    if idx.size == 0:
        return ""
    t, p, s = (int(x) for x in idx[0])
    p = min(p, len(ids) - 1)
    where = f"policy {ids[p]} at t = {t + t_offset}"
    if mask.shape[2] > 1:
        where += f", scenario {s}"
    return where


def check_builtin(
    graph: ir.Graph,
    out: dict[str, NDArray[Any]],
    ids: NDArray[Any],
    horizon: int,
    scales: dict[units.Unit, NDArray[np.float64]],
) -> list[InvariantResult]:
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
            detail = f"{in_force.name} is negative for {_first_bad(neg, ids)}"
        elif rising.any():
            detail = f"{in_force.name} increases for {_first_bad(rising, ids, 1)}"
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
                    f"decrements exceed {in_force.name}(t-1) for {_first_bad(bad, ids, 1)}"
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
                detail = f"{v.name}(H) != {target} for {_first_bad(bad[None], ids, horizon)}"
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
