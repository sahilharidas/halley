"""The aggregation stage's first use and the input contract shaped for M9 to M12."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from halley import HalleyError, RunError, RunResult, RunSettings, build, run
from halley.aggregation import (
    AggregationInputs,
    portfolio_totals,
    require_deterministic,
    sequential_sum,
)
from models import VAL, Indexed, run_growth, scenario_file, simple_model_points


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


def test_aggregation_inputs_take_several_runs_of_the_same_policies(
    term_run: RunResult, tmp_path: Path
) -> None:
    both = AggregationInputs((term_run, term_run))
    assert both.primary is term_run
    other = run_growth(tmp_path)
    with pytest.raises(HalleyError, match="same policies"):
        AggregationInputs((term_run, other))
    with pytest.raises(HalleyError, match="at least one"):
        AggregationInputs(())
