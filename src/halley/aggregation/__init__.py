"""Aggregation stage: group-level calculations on aggregated policy results (D6).

First use: the portfolio totals of the term assurance spec (section 11.3). The IFRS 17 CSM
roll-forward for one group comes with the annuity (M7). The input contract already has the shape
that v2 needs (Roadmap, doors v1 keeps open): it takes one or more projection runs, keeps the
scenario dimension, and the one-scenario requirement of the v1 CSM stage is a single check,
:func:`require_deterministic`, not a shape assumption.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from halley.errors import HalleyError, RunError
from halley.runtime.run import RunResult


def sequential_sum(values: NDArray[Any], axis: int) -> NDArray[Any]:
    """Sum along an axis sequentially, in index order, keeping the axis with length 1 (D19:
    a fixed-order reduction, not pairwise)."""
    return np.cumsum(values, axis=axis).take([-1], axis=axis)


def require_deterministic(result: RunResult) -> None:
    """The v1 CSM stage reads a deterministic run (annuity spec, G6). One place, one check."""
    if result.n_scenarios != 1:
        raise RunError(
            f"this aggregation reads a deterministic run (J = 1); the run has J = "
            f"{result.n_scenarios} scenarios. Scenario statistics in the aggregation stage are "
            "a v2 item"
        )


@dataclass(frozen=True, slots=True)
class AggregationInputs:
    """The projection runs an aggregation reads. A remeasurement (v2) reads a new-basis and an
    old-basis run of the same policies; the v1 stages read one."""

    runs: tuple[RunResult, ...]

    def __post_init__(self) -> None:
        if not self.runs:
            raise HalleyError("an aggregation needs at least one projection run")
        first = self.runs[0]
        for other in self.runs[1:]:
            if not np.array_equal(first.ids, other.ids):
                raise HalleyError("every run of an aggregation covers the same policies")
            if other.H != first.H or other.n_scenarios != first.n_scenarios:
                raise HalleyError("every run of an aggregation has the same horizon and scenarios")

    @property
    def primary(self) -> RunResult:
        return self.runs[0]


def portfolio_totals(result: RunResult, variables: list[str]) -> dict[str, NDArray[Any]]:
    """Each variable summed over policies in ascending ``policy_id``, sequentially, per month and
    scenario: shape ``(time or 1, 1, scenarios)``. Months beyond a policy's own horizon hold
    exact zeros in the projection, so the totals need no masking (term spec 11.3)."""
    out: dict[str, NDArray[Any]] = {}
    for name in variables:
        if name not in result.arrays:
            raise HalleyError(f"the run has no variable {name!r}")
        out[name] = sequential_sum(np.asarray(result.var(name), dtype=np.float64), axis=1)
    return out
