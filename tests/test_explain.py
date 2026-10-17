# pyright: basic
"""Trace, explain and diff."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

from halley import CompiledModel, RunResult, RunSettings, diff, explain, run
from halley.products.term_assurance import VALUATION_DATE
from halley.runtime import recompute, within
from test_helpers import run_helpers


def test_explain_shows_the_formula_and_every_input(term_run: RunResult) -> None:
    e = explain(term_run, "P05", "reserve", 13)
    assert within(e.value, 46.35078406367477, 200_000)
    assert e.unit == "Money"
    assert e.loc.function.endswith("TermAssurance.reserve")
    inputs = dict(e.inputs)
    assert inputs["reserve(t+1)"] == 0.0
    assert within(inputs["net_outgo(t+1)"], 46.502524469400136, 200_000)
    assert within(inputs["v(t+1)"], 0.9967369426185624, 1)
    text = str(e)
    assert "def reserve(self, t: Int) -> Money:" in text
    assert "reserve(t+1) = 0.0" in text


def test_explain_a_scalar_and_a_lookup(term_run: RunResult) -> None:
    e = explain(term_run, "P05", "pv_net_cf")
    assert e.t is None
    assert within(e.value, -537.9608658387098, 200_000)
    q = explain(term_run, "P05", "q_ann", 3)
    assert dict(q.inputs)["mp.sex"] == "M"
    assert dict(q.inputs)["age(t)"] == 51
    assert within(q.value, 0.0025001, 1, rel=1e-5)


def test_recompute_matches_the_engine_lane_by_lane(term_run: RunResult) -> None:
    for name in ("deaths", "lapses", "pols_if", "net_outgo", "reserve", "reserve_pp", "disc"):
        for t in (0, 1, 2, 3, 13, 14, 15):
            assert recompute(term_run, "P05", name, t) == term_run.lane(name, "P05")[t], (name, t)
    assert recompute(term_run, "P05", "pv_premiums", 0) == term_run.lane("pv_premiums", "P05")[0]


def test_recompute_matches_the_generated_code_for_every_helper(tmp_path: Path) -> None:
    """The interpreter evaluates each helper's emitted form in the same floating-point order as
    the backend (D19), so the two agree bit for bit on a model that uses every helper, the
    three pv_month expansions and both run constants."""
    res = run_helpers(tmp_path)
    assert res.graph is not None
    for name, v in res.graph.variables.items():
        months = range(res.H + 1) if v.has_time else [0]
        for pid in res.ids:
            lane = res.lane(name, pid)
            for t in months:
                assert recompute(res, pid, name, t) == lane[t], (name, pid, t)
    e = explain(res, "H01", "cal_year", 3)
    assert dict(e.inputs) == {"run.m_val": 12 * 2027 + 1 - 1}
    assert e.value == 2027
    assert e.unit == "Int"
    bel = explain(res, "H01", "bel", 5)
    assert set(dict(bel.inputs)) == {
        "bel(t+1)",
        "v(t+1)",
        "ann_out(t+1)",
        "exp_out(t+1)",
        "claims(t+1)",
    }


def test_diff_locates_the_largest_change(
    compiled_term: CompiledModel,
    reference_sources: dict[str, Any],
    term_run: RunResult,
    tmp_path: Path,
) -> None:
    table = reference_sources["model_points"]
    rows = table.to_pylist()
    rows[4]["premium"] = rows[4]["premium"] * 2  # P05
    changed = run(
        compiled_term,
        model_points=pa.Table.from_pylist(rows, schema=table.schema),
        tables=reference_sources["tables"],
        settings=RunSettings(valuation_date=VALUATION_DATE),
        run_dir=tmp_path / "changed",
    )
    d = diff(term_run, changed)
    assert d["premium_income"].policy_id == "P05"
    assert d["premium_income"].t == 1
    assert d["premium_income"].max_abs == pytest.approx(46.40, abs=1e-9)
    assert d["death_outgo"].max_abs == 0.0
    assert d["reserve_0"].policy_id == "P05"
