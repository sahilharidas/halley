# pyright: basic
"""The NumPy reference backend: closed forms, domains, reruns, run directories, scenarios and
the run-time checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from halley import (
    Count,
    Factor,
    Int,
    LoadError,
    Money,
    Monthly,
    Prob,
    Product,
    RunError,
    RunSettings,
    at,
    build,
    load_run,
    run,
    var,
)
from halley.products.term_assurance import VALUATION_DATE, TermAssurance
from halley.runtime import recompute, within
from models import (
    SIMPLE_SCHEMA,
    VAL,
    Growth,
    Indexed,
    closed_form_growth,
    run_growth,
    scenario_file,
    simple_model_points,
)


def test_growth_matches_closed_forms(tmp_path: Path) -> None:
    res = run_growth(tmp_path)
    assert (res.H, res.n_policies, res.n_scenarios) == (12, 3, 1)
    for i, pid in enumerate(res.ids):
        x0, prem = 1.0 + i, 10.0 * (i + 1)
        want = closed_form_growth(x0, prem, 12)
        np.testing.assert_allclose(res.lane("x", pid), want["x"], rtol=1e-12, atol=0)
        assert res.lane("cf", pid)[0] == 0.0  # a month quantity is stored as 0 at t = 0
        np.testing.assert_allclose(res.lane("cf", pid)[1:], want["cf"][1:], rtol=1e-12)
        np.testing.assert_allclose(res.lane("r", pid), want["r"], rtol=1e-12)
        assert res.lane("r0", pid)[0] == res.lane("r", pid)[0]
        assert res.lane("total", pid)[0] == pytest.approx(want["cf"][1:].sum(), rel=1e-12)
        assert res.lane("cf_at_k", pid)[0] == res.lane("cf", pid)[2 + i]
        np.testing.assert_allclose(
            res.lane("ratio", pid)[1:], (want["r"] / want["x"])[1:], rtol=1e-12
        )
    masked = res.var("masked")[:, 0, 0]
    assert list(masked[:5]) == [0.0, 0.0, 0.0, 1.0, 0.5]


def test_domains_follow_the_demand_rule() -> None:
    g = build(Growth).graph
    starts = {name: v.start for name, v in g.variables.items()}
    assert starts == {
        "x": 0,  # init
        "cf": 1,  # reads x(t-1): a month quantity
        "r": 0,  # terminal
        "ratio": 1,
        "masked": 1,
        "total": 0,  # no time dimension
        "r0": 0,
        "cf_at_k": 0,
    }


def test_rerun_is_bit_identical_d19(tmp_path: Path) -> None:
    a = run_growth(tmp_path / "a")
    b = run_growth(tmp_path / "b")
    assert a.run_id == b.run_id
    for name in a.arrays:
        assert np.array_equal(a.arrays[name], b.arrays[name]), name
    strip = {"created_at"}
    assert {k: v for k, v in a.manifest.items() if k not in strip} == {
        k: v for k, v in b.manifest.items() if k not in strip
    }


def test_batches_do_not_change_results(tmp_path: Path) -> None:
    whole = run_growth(tmp_path / "whole")
    single = run_growth(tmp_path / "single", batch_size=1)
    for name in whole.arrays:
        assert within(single.var(name), whole.var(name), 1.0).all(), name


def test_run_directory_holds_inputs_code_outputs_and_manifest(tmp_path: Path) -> None:
    res = run_growth(tmp_path)
    d = res.run_dir
    for rel in (
        "manifest.json",
        "generated/numpy_backend.py",
        "inputs/model_points.parquet",
        "outputs/monthly.parquet",
        "outputs/scalars.parquet",
        "invariants.json",
    ):
        assert (d / rel).exists(), rel
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["output_schema"] == 1
    assert manifest["backend"]["name"] == "numpy"
    assert manifest["ir_sha256"] == res.compiled.ir_hash  # type: ignore[union-attr]
    copied = hashlib.sha256((d / "inputs/model_points.parquet").read_bytes()).hexdigest()
    assert manifest["inputs"]["model_points"]["sha256"] == copied
    assert manifest["dimensions"] == {"policies": 3, "scenarios": 1, "months": 13, "horizon": 12}
    assert "def run_batch(inp, H):" in (d / "generated/numpy_backend.py").read_text()
    back = load_run(d)
    assert back.run_id == res.run_id
    assert list(back.ids) == list(res.ids)
    for name in res.arrays:
        assert np.array_equal(back.var(name), res.var(name)), name
    with pytest.raises(LoadError, match="already holds a run"):
        run_growth(tmp_path)


def test_scenarios_broadcast_and_need_history(tmp_path: Path) -> None:
    compiled = build(Indexed)
    horizon = 12
    df = scenario_file("df", 0, horizon, 2, lambda s, u: (0.99 if s == 0 else 0.98) ** u)
    cpi = scenario_file("cpi", -3, horizon, 2, lambda s, u: 100.0 * (1.002 + 0.001 * s) ** u)
    res = run(
        compiled,
        model_points=simple_model_points(rem_m=horizon),
        scenarios={"df": df, "cpi": cpi},
        settings=RunSettings(valuation_date=VAL),
        run_dir=tmp_path / "indexed",
    )
    assert res.n_scenarios == 2
    assert res.var("v").shape == (horizon + 1, 3, 2)
    assert res.arrays["v"].shape == (horizon + 1, 1, 2)  # no policy dimension
    assert res.arrays["pay"].shape == (horizon + 1, 3, 2)
    for s, g in ((0, 1.002), (1, 1.003)):
        esc = res.var("esc")[1:, 0, s]
        np.testing.assert_allclose(esc, [g ** (t - 3) for t in range(1, horizon + 1)], rtol=1e-12)
        v = res.var("v")[1:, 0, s]
        np.testing.assert_allclose(v, [0.99 if s == 0 else 0.98] * horizon, rtol=1e-12)
        ann = res.var("ann_pp")[:, 1, s]
        assert ann[0] == 20.0
        np.testing.assert_allclose(ann[1:], 20.0 * np.cumprod(esc), rtol=1e-12)
    short = scenario_file("cpi", -1, horizon, 2, lambda s, u: 100.0)
    with pytest.raises(LoadError, match="reads months -2 to 9 but the file covers -1 to 12"):
        run(
            compiled,
            model_points=simple_model_points(rem_m=horizon),
            scenarios={"df": df, "cpi": short},
            settings=RunSettings(valuation_date=VAL),
            run_dir=tmp_path / "short",
        )
    uneven = scenario_file("df", 0, horizon, 3, lambda s, u: 1.0)
    with pytest.raises(LoadError, match="disagree on the number of scenarios"):
        run(
            compiled,
            model_points=simple_model_points(rem_m=horizon),
            scenarios={"df": uneven, "cpi": cpi},
            settings=RunSettings(valuation_date=VAL),
            run_dir=tmp_path / "uneven",
        )


def test_probability_range_is_checked_at_run_time(tmp_path: Path) -> None:
    class BadProb(Product):
        model_point = SIMPLE_SCHEMA

        @var
        def q(self, t: Int) -> Prob[Monthly]:
            return -0.5 + 0.0  # type: ignore[return-value]

        @var
        def x(self, t: Int) -> Count:
            return self.mp.pols_if_init * self.q(t)

    with pytest.raises(RunError, match="q is a probability") as info:
        run(
            build(BadProb),
            model_points=simple_model_points(),
            settings=RunSettings(valuation_date=VAL),
            run_dir=tmp_path / "prob",
        )
    assert info.value.loc is not None
    assert info.value.loc.function.endswith("BadProb.q")


def test_lookup_keys_outside_a_table_fail_naming_the_policies(
    reference_sources: dict[str, Any], tmp_path: Path
) -> None:
    class Old(TermAssurance):
        @var
        def age(self, t: Int) -> Int:
            return self.mp.age_at_entry + self.dur_y(t) + 100

    with pytest.raises(RunError, match=r"key age of table mort_table .* P01") as info:
        run(
            build(Old),
            model_points=reference_sources["model_points"],
            tables=reference_sources["tables"],
            settings=RunSettings(valuation_date=VALUATION_DATE),
            run_dir=tmp_path / "old",
        )
    assert info.value.loc is not None
    assert info.value.loc.function.endswith("q_ann")


def test_non_finite_results_fail(tmp_path: Path) -> None:
    class Inf(Product):
        model_point = SIMPLE_SCHEMA

        @var
        def x(self, t: Int) -> Factor:
            return 1.0 / (t - t)

    with pytest.raises(RunError, match="x is not finite"):
        run(
            build(Inf),
            model_points=simple_model_points(),
            settings=RunSettings(valuation_date=VAL),
            run_dir=tmp_path / "inf",
        )


def test_fixed_month_outside_the_horizon_fails(tmp_path: Path) -> None:
    class Late(Growth):
        @var
        def late(self) -> Money:
            return at(self.cf, 50)

    with pytest.raises(RunError, match=r"at\(cf, 50\) is outside the domain of cf, months 1..12"):
        run(
            build(Late),
            model_points=simple_model_points(),
            settings=RunSettings(valuation_date=VAL),
            run_dir=tmp_path / "late",
        )


def test_the_interpreter_agrees_with_the_generated_code(tmp_path: Path) -> None:
    res = run_growth(tmp_path)
    for pid in res.ids:
        for name in ("x", "cf", "r", "ratio", "masked"):
            for t in (0, 1, 5, 11, 12):
                assert recompute(res, pid, name, t) == res.lane(name, pid)[t], (pid, name, t)
        for name in ("total", "r0", "cf_at_k"):
            assert recompute(res, pid, name, 0) == res.lane(name, pid)[0], (pid, name)


def test_outputs_can_be_restricted_at_run_time(tmp_path: Path) -> None:
    res = run_growth(tmp_path, outputs=("r", "total"))
    assert set(res.manifest["outputs"]["variables"]) == {"r", "total"}
    back = load_run(res.run_dir)
    assert set(back.arrays) == {"r", "total"}
    assert set(res.arrays) > {"r", "total"}  # the session keeps every variable for explain
