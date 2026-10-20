"""Aggregation stage: group-level calculations on aggregated policy results (D6).

First use: the portfolio totals of the term assurance spec (section 11.3). The stage itself is
written in the same ``@var`` language as a projection and checked by the same checker
(:mod:`halley.aggregation.stage`, design section 9): an :class:`Aggregation` is a ``Product``
whose model points are groups and whose extra inputs are group series, each a :class:`GroupSum`
over the runs of a :class:`Basis`. The scenario dimension is kept throughout (D4): a series is a
mean over its basis's scenarios or keeps them, and the one-scenario requirement of a stage that
reads a mean is a single check, :func:`require_deterministic`, not a shape assumption.

The CSM of one group is :mod:`halley.aggregation.csm` (``CsmRollForward``, ``NewGroup``,
``InForceGroup``); the cross-run helpers that build an in-force group's inputs, and the
recomputations the loader runs to verify them, are :mod:`halley.aggregation.inforce`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from numpy.typing import NDArray

from halley.aggregation.csm import CsmRollForward, InForceGroup, NewGroup
from halley.aggregation.inforce import (
    check_membership,
    composed_curve,
    expected_in_force,
    locked_in_cpi,
    locked_in_curve,
    rebase_expense_table,
)
from halley.aggregation.stage import (
    Aggregation,
    Basis,
    BasisInfo,
    GroupSum,
    MemberRule,
    MemberView,
    group_series,
    replay_aggregation,
    run_aggregation,
    sequential_sum,
)
from halley.errors import HalleyError, RunError

if TYPE_CHECKING:
    from halley.runtime.run import RunResult


class HasScenarios(Protocol):
    """Anything with a number of scenarios: a run, a scenario input, a group series."""

    @property
    def n_scenarios(self) -> int: ...


def require_deterministic(
    source: HasScenarios,
    what: str = "the run",
    reason: str = "this aggregation reads a deterministic run (J = 1)",
    error: type[HalleyError] = RunError,
) -> None:
    """``source`` has one scenario (J = 1). The one check of D4: a stage that reads a mean over
    scenarios runs at S = 1, so its own scenario inputs and every per-scenario series it reads
    must be deterministic (the loader raises a LoadError through this); a projection run read
    as deterministic must be too (annuity spec, G6). One place, one check."""
    n = source.n_scenarios
    if n != 1:
        raise error(
            f"{reason}; {what} has J = {n} scenarios, and a basis with J > 1 reaches a stage "
            'only through a mean, GroupSum(..., scenarios="mean") (require_deterministic)'
        )


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


__all__ = [
    "Aggregation",
    "Basis",
    "BasisInfo",
    "CsmRollForward",
    "GroupSum",
    "HasScenarios",
    "InForceGroup",
    "MemberRule",
    "MemberView",
    "NewGroup",
    "check_membership",
    "composed_curve",
    "expected_in_force",
    "group_series",
    "locked_in_cpi",
    "locked_in_curve",
    "portfolio_totals",
    "rebase_expense_table",
    "replay_aggregation",
    "require_deterministic",
    "run_aggregation",
    "sequential_sum",
]
