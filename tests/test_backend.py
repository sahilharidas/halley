# pyright: basic
"""The NumPy reference backend: closed forms, domains, reruns, run directories, scenarios and
the run-time checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from halley import (
    Column,
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
    Table,
    at,
    build,
    load_run,
    run,
    var,
)
from halley.products.term_assurance import VALUATION_DATE, TermAssurance
from halley.runtime import recompute, within
from models import (
    POWER_DEN,
    SIMPLE_SCHEMA,
    VAL,
    Expensed,
    Growth,
    Indexed,
    Margin,
    Powered,
    closed_form_expensed,
    closed_form_growth,
    margin_table,
    powered_model_points,
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


def _power_offset(k: int, lo: int, horizon: int = 12) -> int:
    """The first offset ``j >= lo`` for which a month value ``(t + j) / POWER_DEN``, t in
    ``1..horizon``, raised to ``k`` differs in the last place between NumPy's fast path and
    libm's pow; skips when this libm offers none."""
    for j in range(lo, lo + 200_000):
        for t in range(1, horizon + 1):
            x = (t + j) / POWER_DEN
            if (np.array([x]) ** float(k)).item() != x ** float(k):
                return j
    pytest.skip(f"this libm's pow(x, {k}) equals NumPy's fast path on every probed month value")


def test_a_non_literal_exponent_agrees_across_batchings_d5(tmp_path: Path) -> None:
    """``x(t) ** self.mp.k`` with k = -1 and 2, outside a recursive component and inside one.
    NumPy decides its fast path for an array exponent from the shapes of the batch (see
    ``value_kind``), so a batch of one policy and a batch of three may differ in the last place.
    Each of those policies has a month value on which the fast path and libm's pow differ, so
    the comparison is not vacuous. The rule is the annuity specification's: reruns of the same
    batching are bit-identical (D19), another batching agrees within 1e-12 relative (D5)."""
    ks = [-1, 2, 3]
    js = [_power_offset(-1, 60_000), _power_offset(2, 70_000), 80_000]
    compiled = build(Powered)
    assert "(V_x[1:H+1] ** M_k)" in compiled.source
    assert "(V_x[t] ** m_k)" in compiled.source
    runs: dict[tuple[int, int], Any] = {}
    for batch_size in (1, 3):
        for rerun in (0, 1):
            runs[batch_size, rerun] = run(
                compiled,
                model_points=powered_model_points(js, ks),
                settings=RunSettings(valuation_date=VAL, batch_size=batch_size),
                run_dir=tmp_path / f"batch-{batch_size}-{rerun}",
            )
    for batch_size in (1, 3):
        a, b = runs[batch_size, 0], runs[batch_size, 1]
        assert a.run_id == b.run_id
        assert a.manifest["outputs"]["monthly_sha256"] == b.manifest["outputs"]["monthly_sha256"]
        for name in a.arrays:
            assert np.array_equal(a.arrays[name], b.arrays[name]), (batch_size, name)
    one, three = runs[1, 0], runs[3, 0]
    assert set(one.arrays) == {"x", "whole", "acc", "term"}
    for name in one.arrays:
        np.testing.assert_allclose(one.var(name), three.var(name), rtol=1e-12, atol=0, err_msg=name)
    # the batch of three takes libm's pow for the array exponent, and so does the interpreter:
    # mirroring the fast path for an array exponent would break the agreement on K0 and K1
    for pid in ("K0", "K1"):
        for name in ("whole", "term"):
            lane = three.lane(name, pid)
            for t in range(1, three.H + 1):
                assert recompute(three, pid, name, t) == lane[t], (pid, name, t)


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
    assert manifest["inputs"]["model_points"]["path"] is None  # given in memory
    assert manifest["inputs"]["model_points"]["metadata"] == {}
    assert manifest["dimensions"] == {"policies": 3, "scenarios": 1, "months": 13, "horizon": 12}
    assert manifest["outputs"]["kind"] == "projection"
    assert manifest["outputs"]["id_field"] == "policy_id"
    assert manifest["outputs"]["variables"]["x"] == {
        "time": True,
        "unit": "Count",
        "start": 0,
        "role": "in_force",
    }
    assert manifest["outputs"]["variables"]["cf"]["start"] == 1
    for name in ("monthly", "scalars"):
        digest = hashlib.sha256((d / f"outputs/{name}.parquet").read_bytes()).hexdigest()
        assert manifest["outputs"][f"{name}_sha256"] == digest
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


def test_a_table_named_path_stays_in_the_run_identity(tmp_path: Path) -> None:
    """The run identity drops each input's own ``path`` string and nothing else (D21: a run is
    addressed by what it read): a table named ``path`` keeps its hash in the identity, so
    different bytes give different run ids and the same bytes at two paths give one."""
    share = Table("path", keys=(), columns=(Column("share", "Factor", ge=0, le=1),))

    class Pathed(Growth):
        assumption_tables = (share,)

        @var
        def shared(self) -> Money:
            return self.total() * self.tables.path.share

    compiled = build(Pathed)

    def go(name: str, value: float, where: str = "a") -> Any:
        d = tmp_path / where
        d.mkdir(exist_ok=True)
        pq.write_table(pa.table({"share": pa.array([value], pa.float64())}), d / f"{name}.parquet")
        return run(
            compiled,
            model_points=simple_model_points(),
            tables={"path": d / f"{name}.parquet"},
            settings=RunSettings(valuation_date=VAL),
            run_dir=tmp_path / f"run-{name}-{where}",
        )

    half, seventy = go("half", 0.5), go("seventy", 0.7)
    assert half.run_id != seventy.run_id
    assert (
        half.manifest["inputs"]["tables"]["path"]["sha256"]
        != seventy.manifest["inputs"]["tables"]["path"]["sha256"]
    )
    moved = go("half", 0.5, where="b")
    assert moved.run_id == half.run_id
    assert (
        moved.manifest["inputs"]["tables"]["path"]["path"]
        != half.manifest["inputs"]["tables"]["path"]["path"]
    )


def test_run_constants_come_from_the_valuation_date_and_horizon(tmp_path: Path) -> None:
    class Dated(Growth):
        @var
        def cal_year(self, t: Int) -> Int:
            return (self.run.m_val + t) // 12

        @var
        def m_val(self) -> Int:
            return self.run.m_val

        @var
        def months(self) -> Int:
            return self.run.horizon

    compiled = build(Dated)
    assert "R_m_val = inp['run.m_val']" in compiled.source
    assert "R_horizon = H" in compiled.source
    assert "V_cal_year[1:H+1] = ((R_m_val + _t[1:H+1]) // 12)" in compiled.source
    res = run(
        compiled,
        model_points=simple_model_points(),
        settings=RunSettings(valuation_date=VAL),  # 2027-01-01
        run_dir=tmp_path / "dated",
    )
    assert res.arrays["m_val"].shape == (1, 1, 1)
    assert res.lane("m_val", "S00")[0] == 12 * 2027 + 1 - 1
    assert res.lane("months", "S00")[0] == 12
    assert list(res.lane("cal_year", "S00")[1:]) == [2027] * 11 + [2028]
    assert res.arrays["cal_year"].dtype == np.int64
    longer = run(
        compiled,
        model_points=simple_model_points(),
        settings=RunSettings(valuation_date=VAL.replace(month=11), horizon=20),
        run_dir=tmp_path / "longer",
    )
    assert longer.lane("m_val", "S00")[0] == 12 * 2027 + 11 - 1
    assert longer.lane("months", "S00")[0] == 20
    assert longer.run_id != res.run_id


def test_a_backward_component_of_two_members_steps_from_the_horizon(tmp_path: Path) -> None:
    """The annuity's {bel, inv_exp} (design section 4): the member without a terminal is
    evaluated at t = H before the loop, in same-period order after the terminal it reads, and
    both members then step back together with the existing start guards."""
    compiled = build(Expensed)
    source = compiled.source
    terminal = source.index("V_bel[H] = (M_pols_if_init * M_premium)")
    pre = source.index("V_inv_exp[H] = (V_ie_m[0] * V_bel[t])")
    loop = source.index("for t in range(H - 1, -1, -1):")
    assert terminal < pre < loop
    assert "t = H\n" in source[terminal:pre]
    assert (
        "V_bel[t] = ((V_cf[t+1] + (V_inv_exp[t+1] * V_v[t+1])) + (V_v[t+1] * V_bel[t+1]))"
        in source[loop:]
    )
    assert "if t >= 1:\n                V_inv_exp[t] = (V_ie_m[0] * V_bel[t])" in source[loop:]
    res = run(
        compiled,
        model_points=simple_model_points(),
        settings=RunSettings(valuation_date=VAL),
        run_dir=tmp_path / "expensed",
    )
    for i, pid in enumerate(res.ids):
        want = closed_form_expensed(1.0 + i, 10.0 * (i + 1), res.H)
        bel, inv_exp, cf = res.lane("bel", pid), res.lane("inv_exp", pid), res.lane("cf", pid)
        ie, v = res.lane("ie_m", pid)[0], res.lane("v", pid)[res.H]
        np.testing.assert_allclose(bel, want["bel"], rtol=1e-12, atol=0)
        np.testing.assert_allclose(inv_exp, want["inv_exp"], rtol=1e-12, atol=0)
        assert bel[res.H] == (1.0 + i) * (10.0 * (i + 1))  # the terminal: a year's premium
        assert inv_exp[res.H] == ie * bel[res.H] != 0.0  # the formula at H, on the terminal
        # the pre-loop value feeds the last step; had it not run, bel(H-1) would be cf + v bel
        assert bel[res.H - 1] == (cf[res.H] + inv_exp[res.H] * v) + v * bel[res.H]
        assert bel[res.H - 1] != cf[res.H] + v * bel[res.H]
        assert inv_exp[0] == 0.0  # a month quantity
        assert inv_exp[1] == ie * bel[1]
        for name in ("bel", "inv_exp"):
            for t in range(res.H + 1):
                assert recompute(res, pid, name, t) == res.lane(name, pid)[t], (name, t)


def test_boundary_values_from_variables_are_computed_before_the_recursion(
    tmp_path: Path,
) -> None:
    """``csm`` starts at the scalar ``csm_0`` and ``lc`` at ``at(fcf, 0)`` (design section 5):
    the generated code assigns both inits after the variables they read and before the loops."""
    compiled = build(Margin)
    source = compiled.source
    fcf_loop = source.index("for t in range(H - 1, -1, -1):")
    csm_0 = source.index("V_csm_0[...] = ((V_premium_total - _at(V_fcf, 0, H, 0, 'fcf', 'csm_0'")
    csm_init = source.index("V_csm[0] = V_csm_0")
    lc_init = source.index("V_lc[0] = _at(V_fcf, 0, H, 0, 'fcf', 'lc', 'Margin.<lambda>")
    assert fcf_loop < csm_0 < csm_init
    assert fcf_loop < lc_init
    res = run(
        compiled,
        model_points=simple_model_points(),
        tables={"margin": margin_table(0.5)},
        settings=RunSettings(valuation_date=VAL),
        run_dir=tmp_path / "margin",
    )
    for i, pid in enumerate(res.ids):
        fcf0 = res.lane("fcf", pid)[0]
        want_csm_0 = ((1.0 + i) * (10.0 * (i + 1)) - fcf0) * 0.5
        assert res.lane("csm_0", pid)[0] == want_csm_0
        assert res.lane("csm", pid)[0] == want_csm_0
        assert res.lane("csm", pid)[1] == want_csm_0 * 0.95
        assert res.lane("lc", pid)[0] == fcf0
        assert res.lane("lc", pid)[1] == fcf0 * 0.9
        assert recompute(res, pid, "csm", 0) == want_csm_0
        assert recompute(res, pid, "lc", 0) == fcf0
    assert res.arrays["csm_0"].shape == (1, 3, 1)


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


class Ahead(Growth):
    """``at(x, t + 1)`` reaches month H + 1 at t = H for every policy: the failing mask has a
    month axis and no policy axis."""

    @var
    def ahead(self, t: Int) -> Count:
        return at(self.x, t + 1)


def test_an_at_that_fails_for_every_lane_names_every_lane(tmp_path: Path) -> None:
    """The message names the failing policies at every batch size, not only in a batch of one,
    where the mask's single lane happened to match the batch."""
    compiled = build(Ahead)
    for batch_size, named in ((1, "S00"), (2, "S00, S01"), (3, "S00, S01, S02")):
        with pytest.raises(
            RunError,
            match=rf"at\(x, \.\.\.\) reads a month outside the domain of x, months 0\.\.12; "
            rf"policies {named}\n  at Ahead\.ahead",
        ):
            run(
                compiled,
                model_points=simple_model_points(),
                settings=RunSettings(valuation_date=VAL, batch_size=batch_size),
                run_dir=tmp_path / f"ahead-{batch_size}",
            )
