"""The aggregation stage's first use, the deterministic check and the basis contract shaped for
M9 to M12 (design section 9)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pytest

from halley import Basis, HalleyError, LoadError, RunError, RunResult, RunSettings, build, run
from halley.aggregation import (
    portfolio_totals,
    require_deterministic,
    sequential_sum,
)
from models import (
    VAL,
    Indexed,
    covered_points,
    run_covered,
    scenario_file,
    simple_model_points,
)


def test_portfolio_totals_are_sequential_sums_over_policies(term_run: RunResult) -> None:
    totals = portfolio_totals(term_run, ["reserve", "pols_if", "pv_net_cf"])
    for name in ("reserve", "pols_if", "pv_net_cf"):
        manual = np.cumsum(term_run.var(name), axis=1)[:, -1:, :]
        assert np.array_equal(totals[name], manual), name
    assert totals["reserve"].shape == (481, 1, 1)
    assert totals["pv_net_cf"].shape == (1, 1, 1)
    with pytest.raises(HalleyError, match="no variable"):
        portfolio_totals(term_run, ["nothing"])


def test_sequential_sum_is_left_to_right() -> None:
    values = np.array([[1e16, 1.0, -1e16]])
    assert sequential_sum(values, axis=1)[0, 0] == 0.0  # (1e16 + 1) - 1e16, not 1e16 + (1 - 1e16)


def test_the_deterministic_check_lives_in_one_place(term_run: RunResult, tmp_path: Path) -> None:
    require_deterministic(term_run)
    compiled = build(Indexed)
    res = run(
        compiled,
        model_points=simple_model_points(rem_m=6),
        scenarios={
            "df": scenario_file("df", 0, 6, 3, lambda s, u: 0.99**u),
            "cpi": scenario_file("cpi", -3, 6, 3, lambda s, u: 100.0),
        },
        settings=RunSettings(valuation_date=VAL),
        run_dir=tmp_path / "stochastic",
    )
    assert res.n_scenarios == 3
    with pytest.raises(RunError, match="J = 3"):
        require_deterministic(res)
    totals = portfolio_totals(res, ["pay"])
    assert totals["pay"].shape == (7, 1, 3)  # the scenario dimension survives aggregation


def test_basis_requires_disjoint_ids_same_date_horizon_and_j(tmp_path: Path) -> None:
    """A basis is one or more projection runs, one per product class, that agree on the
    valuation date, the horizon and the number of scenarios and hold disjoint policies (design
    section 9). It replaces AggregationInputs, which took several runs of the same policies."""
    cur = Basis("cur")
    a = run_covered(tmp_path, "a", covered_points("A"))
    b = run_covered(tmp_path, "b", covered_points("B"))
    cur.check([a, b])  # one valuation date, H = 12, J = 1, A.. and B.. apart

    with pytest.raises(LoadError, match="basis cur has no runs"):
        cur.check([])
    with pytest.raises(LoadError, match=f"policy A00 is in run {a.run_id} and in run {a.run_id}"):
        cur.check([a, a])
    overlap = run_covered(tmp_path, "overlap", covered_points("B", ids=["B07", "A01"]))
    with pytest.raises(
        LoadError, match=rf"policy A01 is in run {a.run_id} and in run {overlap.run_id}; .*disjoint"
    ):
        cur.check([a, overlap])

    later = run_covered(tmp_path, "later", covered_points("B"), valuation_date=dt.date(2027, 2, 1))
    with pytest.raises(LoadError, match=r"values at 2027-02-01 but run .* at 2027-01-01"):
        cur.check([a, later])
    shorter = run_covered(tmp_path, "shorter", covered_points("B"), horizon=6)
    with pytest.raises(LoadError, match=r"has horizon 6 but run .* has 12; .*RunSettings\.horizon"):
        cur.check([a, shorter])
    stochastic = run_covered(tmp_path, "stochastic", covered_points("B"), shocks=(1.0, 1.1))
    with pytest.raises(LoadError, match=r"has J = 2 scenarios but run .* has J = 1"):
        cur.check([a, stochastic])
