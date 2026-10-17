# pyright: basic
"""Trace, explain and diff."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest

from halley import (
    CompiledModel,
    Factor,
    Field,
    Int,
    Money,
    Product,
    RunResult,
    RunSettings,
    Schema,
    build,
    diff,
    explain,
    pv_month,
    run,
    var,
)
from halley.authoring import int_field
from halley.products.term_assurance import VALUATION_DATE
from halley.runtime import recompute, within
from models import VAL
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


MID_SCHEMA = Schema(
    fields=(
        Field("policy_id", "string", unique=True),
        Field("disc", "number", "Factor", gt=0, le=1),
        Field("premium", "number", "MoneyPerPolicy", gt=0),
        Field("pols_if_init", "number", "Count", gt=0),
        int_field("rem_m", ge=1, le=1000),
    ),
    horizon="rem_m",
)


class MidTimed(Product):
    """A mid-month cash flow discounted by a constant monthly factor: the backward recursion's
    last step is ``cf(H) * (v(H) ** 0.5)``, which with ``cf = 1`` is ``v ** 0.5`` itself."""

    model_point = MID_SCHEMA

    @var
    def v(self, t: Int) -> Factor:
        return self.mp.disc

    @var(timing="mid")
    def cf(self, t: Int) -> Money:
        return self.mp.pols_if_init * self.mp.premium

    @var(terminal=0)
    def pv(self, t: Int) -> Money:
        return (self.pv(t + 1) * self.v(t + 1)) + pv_month(self.cf, t + 1, self.v)


def _sqrt_fast_path_values(n: int = 3) -> list[float]:
    """Discount factors in (0.9, 1) for which NumPy's ``x ** 0.5`` over an array (its sqrt fast
    path) differs in the last bit from Python's ``x ** 0.5`` (libm's pow); about one value in
    a thousand does."""
    rng = np.random.default_rng(20261017)
    found: list[float] = []
    for _ in range(50):
        xs = rng.uniform(0.9, 1.0, 20_000)
        fast = xs**0.5
        for x, f in zip(xs.tolist(), fast.tolist(), strict=True):
            if x**0.5 != f:
                found.append(x)
                if len(found) == n:
                    return found
    raise AssertionError("no value found whose sqrt fast path differs from pow")


def test_recompute_mirrors_the_power_fast_path_of_the_generated_code(tmp_path: Path) -> None:
    """``pv_month`` of a mid-month flow emits ``cf * (v ** 0.5)`` and the backend evaluates
    ``ndarray ** 0.5`` through sqrt, which differs from libm's pow (Python's ``**``) in the last
    bit for some values. Every model point here is such a value, so the interpreter agrees with
    the backend only when it takes the same path (D19 for the second implementation)."""
    discs = _sqrt_fast_path_values()
    n = len(discs)
    points = pa.table(
        {
            "policy_id": pa.array([f"D{i}" for i in range(n)], pa.string()),
            "disc": pa.array(discs, pa.float64()),
            "premium": pa.array([1.0] * n, pa.float64()),
            "pols_if_init": pa.array([1.0] * n, pa.float64()),
            "rem_m": pa.array([6] * n, pa.int32()),
        }
    )
    res = run(
        build(MidTimed),
        model_points=points,
        settings=RunSettings(valuation_date=VAL),
        run_dir=tmp_path / "mid",
    )
    for pid, d in zip(res.ids, discs, strict=True):
        lane = res.lane("pv", pid)
        # the last step is 0 * v + 1.0 * (v ** 0.5): the fast path's value, not pow's
        assert lane[res.H - 1] == float(np.sqrt(d))
        assert lane[res.H - 1] != d**0.5
        for t in range(res.H + 1):
            assert recompute(res, pid, "pv", t) == lane[t], (pid, t)
    e = explain(res, "D0", "pv", res.H - 1)
    assert dict(e.inputs)["v(t+1)"] == discs[0]


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
