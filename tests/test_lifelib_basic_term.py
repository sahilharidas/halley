# pyright: basic
"""Reference case 2: the engine against lifelib's BasicTerm_ME, from lifelib's own inputs (D23).

The oracle is lifelib itself: the committed results under ``tests/data/lifelib`` were produced
by running BasicTerm_ME with modelx (``make_reference.py`` there), and the last test here reruns
lifelib when it is installed and checks the committed files against it. Every per-policy present
value, the monthly flows and counts of the sampled policies and the portfolio totals must agree
under the term spec's tolerance (section 1): ``|a - b| <= 1e-9 * max(|a|, |b|, scale)`` with
``scale`` ``sum_assured * policy_count`` for Money, ``policy_count`` for Count, ``sum_assured``
per policy and 1 for rates and factors. The conventions the model encodes are listed in the
docstring of :mod:`halley.products.lifelib_basic_term`; the index shift is the one to remember:
a flow Halley reports at month ``t`` is lifelib's flow at ``t - 1``, counts are unshifted.
"""

from __future__ import annotations

import hashlib
import importlib.util
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from halley import CompiledModel, RunResult, RunSettings, build, run
from halley.aggregation import portfolio_totals
from halley.products.lifelib_basic_term import (
    LIFELIB_VERSION,
    VALUATION_DATE,
    BasicTerm,
    premium_rates,
)
from halley.runtime import within

FIXTURES = Path(__file__).resolve().parent / "data" / "lifelib"
TABLES = {name: FIXTURES / f"{name}.parquet" for name in ("mort_table", "disc_rate_ann")}

STATES = ("pols_if", "pols_maturity", "pols_new_biz", "pols_if_bef_decr")
"""Counts at an instant: lifelib's index t is Halley's t."""
FLOWS = ("premiums", "claims", "expenses", "commissions", "net_cf", "pols_death", "pols_lapse")
"""Flows of a month: lifelib's index t is Halley's t + 1."""
RATES = ("mort_rate_mth", "lapse_rate")
"""The rates lifelib applies in month t, used by Halley's month t + 1."""
MONEY = {"premiums", "claims", "expenses", "commissions", "net_cf"}
PV_MONEY = ("pv_premiums", "pv_claims", "pv_expenses", "pv_commissions", "pv_net_cf")


def model_points_with_premium_rate() -> Any:
    """lifelib's model points with lifelib's premium rate joined on (convention 9)."""
    mp = pq.read_table(FIXTURES / "model_point_table.parquet")
    pt = pq.read_table(FIXTURES / "premium_table.parquet")
    rate = premium_rates(
        mp.column("age_at_entry").to_numpy(),
        mp.column("policy_term").to_numpy(),
        pt.column("age_at_entry").to_numpy(),
        pt.column("policy_term").to_numpy(),
        pt.column("premium_rate").to_numpy(),
    )
    return mp.append_column("premium_rate", pa.array(rate, pa.float64()))


@pytest.fixture(scope="module")
def compiled_basic_term() -> CompiledModel:
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # an unused variable would be a BuildWarning
        return build(BasicTerm)


@pytest.fixture(scope="module")
def basic_term_run(compiled_basic_term: CompiledModel, tmp_path_factory: Any) -> RunResult:
    """All 10,000 of lifelib's model points, in batches of 2,000."""
    return run(
        compiled_basic_term,
        model_points=model_points_with_premium_rate(),
        tables=TABLES,
        settings=RunSettings(valuation_date=VALUATION_DATE, batch_size=2000, label="lifelib"),
        run_dir=tmp_path_factory.mktemp("runs") / "lifelib-basic-term",
    )


@pytest.fixture(scope="module")
def lifelib_pv() -> Any:
    return pq.read_table(FIXTURES / "results_pv.parquet")


@pytest.fixture(scope="module")
def lifelib_monthly() -> Any:
    return pq.read_table(FIXTURES / "results_monthly.parquet")


@pytest.fixture(scope="module")
def lifelib_totals() -> Any:
    return pq.read_table(FIXTURES / "results_totals.parquet")


def _column(result: RunResult, name: str) -> Any:
    return result.inputs.model_points.columns[name]  # pyright: ignore[reportOptionalMemberAccess]


def _scales(result: RunResult) -> dict[str, Any]:
    count = _column(result, "policy_count")
    sum_assured = _column(result, "sum_assured")
    return {"money": sum_assured * count, "count": count, "per_policy": sum_assured}


def _scale_of(name: str, scales: dict[str, Any]) -> Any:
    if name in MONEY or name in PV_MONEY:
        return scales["money"]
    if name in RATES:
        return 1.0
    if name in ("net_premium_pp", "premium_pp"):
        return scales["per_policy"]
    return scales["count"]


# ------------------------------------------------------------------------------ structure


def test_dimensions_follow_lifelib(basic_term_run: RunResult, lifelib_pv: Any) -> None:
    r = basic_term_run
    assert r.n_policies == 10_000
    assert r.n_scenarios == 1
    assert r.H == 276  # max(12 * policy_term - duration_mth) = lifelib's max_proj_len - 1
    assert int(lifelib_pv.column("proj_len").to_numpy().max()) == r.H + 1
    assert np.array_equal(r.ids, lifelib_pv.column("policy_id").to_numpy())
    assert np.array_equal(_column(r, "rem_m") + 1, lifelib_pv.column("proj_len").to_numpy())
    assert [x.passed for x in r.invariants] == [True]  # probabilities only (convention 5)


def test_one_forward_component_holds_the_counts(compiled_basic_term: CompiledModel) -> None:
    recursive = [c for c in compiled_basic_term.graph.components if len(c) > 1]
    assert recursive == [
        [
            "pols_death",
            "pols_lapse",
            "pols_if",
            "pols_maturity",
            "pols_if_bef_nb",
            "pols_if_bef_decr",
        ]
    ]
    for name in ("premiums", "claims", "expenses", "commissions"):
        assert compiled_basic_term.graph.variables[name].timing == "start"


def test_premium_pp_is_lifelib_rounding_exactly(basic_term_run: RunResult, lifelib_pv: Any) -> None:
    engine = basic_term_run.var("premium_pp")[0, :, 0]
    assert np.array_equal(engine, lifelib_pv.column("premium_pp").to_numpy())
    assert np.array_equal(engine, _column(basic_term_run, "premium_pp"))
    assert np.array_equal(engine, np.around(engine, 2))  # two decimals: rounding is idempotent


# -------------------------------------------------------------------------- reconciliation


def test_present_values_reconcile_for_every_policy(
    basic_term_run: RunResult, lifelib_pv: Any
) -> None:
    scales = _scales(basic_term_run)
    for name in (*PV_MONEY, "pv_pols_if", "net_premium_pp"):
        engine = basic_term_run.var(name)[0, :, 0]
        lifelib = lifelib_pv.column(name).to_numpy()
        ok = within(engine, lifelib, _scale_of(name, scales))
        bad = np.flatnonzero(~ok)
        assert ok.all(), (
            f"{name}: {len(bad)} policies differ, first policy "
            f"{basic_term_run.ids[bad[0]]}: {engine[bad[0]]!r} vs {lifelib[bad[0]]!r}"
            if bad.size
            else ""
        )
        if name != "pv_net_cf":  # the net flow cancels; the gross values agree far tighter
            nonzero = lifelib != 0
            rel = np.abs(engine[nonzero] - lifelib[nonzero]) / np.abs(lifelib[nonzero])
            assert rel.max() < 1e-12, (name, rel.max())  # summation order and powers only
    # policies with no count have exactly nothing
    empty = _column(basic_term_run, "policy_count") == 0
    assert empty.any()
    for name in (*PV_MONEY, "pv_pols_if", "net_premium_pp"):
        assert not basic_term_run.var(name)[0, empty, 0].any(), name


def test_monthly_results_reconcile_for_the_sample(
    basic_term_run: RunResult, lifelib_monthly: Any
) -> None:
    r = basic_term_run
    scales = _scales(r)
    pid = lifelib_monthly.column("policy_id").to_numpy()
    k = lifelib_monthly.column("t").to_numpy()  # lifelib's month index
    sample = np.unique(pid)
    assert len(sample) == 56
    rem_m = _column(r, "rem_m")
    duration_mth = _column(r, "duration_mth")
    assert duration_mth[np.isin(r.ids, sample)].min() < 0  # future new business is sampled
    assert (rem_m[np.isin(r.ids, sample)] == 0).any()  # and a policy maturing at t = 0
    worst: dict[str, float] = {}
    for name in STATES + FLOWS + RATES:
        engine_all = r.var(name)
        lifelib_all = lifelib_monthly.column(name).to_numpy()
        shift = 0 if name in STATES else 1
        for p in sample:
            rows = pid == p
            i = r.index(p)
            t = k[rows] + shift
            inside = t <= r.H
            engine = engine_all[t[inside], i, 0]
            lifelib = lifelib_all[rows][inside]
            scale = _scale_of(name, scales)
            scale = scale if np.ndim(scale) == 0 else scale[i]
            ok = within(engine, lifelib, scale)
            assert ok.all(), (
                f"policy {p} {name}: first difference at lifelib t = "
                f"{k[rows][inside][np.argmin(ok)]}"
            )
            worst[name] = max(worst.get(name, 0.0), float(np.abs(engine - lifelib).max()))
            if name in FLOWS:  # lifelib's month beyond Halley's horizon has no flow
                assert not lifelib_all[rows][~inside].any(), (p, name)
    # counts, premiums, claims and commissions are bit-identical; expenses carry the inflation
    # factor's last-bit difference (convention 10)
    for name in (*STATES, "premiums", "claims", "commissions", "pols_death", "pols_lapse"):
        assert worst[name] == 0.0, (name, worst[name])
    assert worst["expenses"] < 1e-9


def test_after_maturity_everything_is_exactly_zero(basic_term_run: RunResult) -> None:
    r = basic_term_run
    rem_m = _column(r, "rem_m")
    t = np.arange(r.H + 1)[:, None]
    after = t > rem_m[None, :]  # instants after maturity
    for name in STATES:
        assert not r.var(name)[:, :, 0][after].any(), name
    after_flow = t > rem_m[None, :] + 1  # months starting after maturity
    for name in FLOWS:
        assert not r.var(name)[:, :, 0][after_flow].any(), name
    for name in FLOWS + RATES:  # month quantities are 0 at t = 0, before the first month
        assert not r.var(name)[0].any(), name


def test_in_force_properties_hold_with_new_business(basic_term_run: RunResult) -> None:
    """What the built-in H1 and H2 check, allowing the one increase at issue (convention 5)."""
    r = basic_term_run
    count = _column(r, "policy_count")
    duration_mth = _column(r, "duration_mth")
    bef_decr = r.var("pols_if_bef_decr")[:, :, 0]
    pols_if = r.var("pols_if")[:, :, 0]
    assert (bef_decr >= 0).all()
    assert (pols_if >= 0).all()
    t = np.arange(r.H + 1)[:, None]
    issue = duration_mth[None, :] + t == 0  # the instant a future policy is issued
    rising = bef_decr[1:] > bef_decr[:-1]
    assert not (rising & ~issue[1:]).any()
    assert np.array_equal(r.var("pols_new_biz")[:, :, 0] > 0, issue & (count[None, :] > 0))
    decrements = r.var("pols_death")[1:, :, 0] + r.var("pols_lapse")[1:, :, 0]
    assert (decrements <= bef_decr[:-1] + 1e-9 * count[None, :]).all()
    maturity = r.var("pols_maturity")[:, :, 0]
    matures = duration_mth[None, :] + t == 12 * _column(r, "policy_term")[None, :]
    assert np.array_equal(maturity, np.where(matures, pols_if, 0.0))
    # every policy issued on or before the horizon leaves in full: deaths, lapses or maturity
    exits = decrements.sum(axis=0) + maturity.sum(axis=0)
    assert within(exits, count, count).all()


def test_portfolio_totals_reconcile(basic_term_run: RunResult, lifelib_totals: Any) -> None:
    r = basic_term_run
    scales = _scales(r)
    money_scale = float(scales["money"].sum())
    count_scale = float(scales["count"].sum())
    totals = portfolio_totals(r, [*MONEY, "pols_if", "pols_maturity", "pols_new_biz"])
    totals.update(portfolio_totals(r, ["pols_death", "pols_lapse"]))
    assert lifelib_totals.num_rows == r.H + 1
    for name in (*MONEY, "pols_death", "pols_lapse"):  # flows: lifelib t = Halley t + 1
        lifelib = lifelib_totals.column(name).to_numpy()
        scale = money_scale if name in MONEY else count_scale
        assert within(totals[name][1:, 0, 0], lifelib[: r.H], scale).all(), name
        assert lifelib[r.H] == 0.0, name  # lifelib's last month has no flow
    for name in ("pols_if", "pols_maturity", "pols_new_biz"):
        lifelib = lifelib_totals.column(name).to_numpy()
        assert within(totals[name][:, 0, 0], lifelib, count_scale).all(), name
    assert np.array_equal(
        r.var("disc_factors")[:, 0, 0], lifelib_totals.column("disc_factors").to_numpy()
    )
    inflation = lifelib_totals.column("inflation_factor").to_numpy()
    assert within(r.var("inflation_factor")[:, 0, 0], inflation, 1.0).all()
    assert np.abs(r.var("inflation_factor")[:, 0, 0] - inflation).max() < 1e-13


# ---------------------------------------------------------------------------------- the run


def test_manifest_cites_lifelib_inputs(basic_term_run: RunResult) -> None:
    manifest = basic_term_run.manifest
    assert manifest["model"]["name"].endswith("lifelib_basic_term.BasicTerm")
    versions = {name: info["version"] for name, info in manifest["inputs"]["tables"].items()}
    assert versions == {
        "mort_table": f"lifelib-{LIFELIB_VERSION}",
        "disc_rate_ann": f"lifelib-{LIFELIB_VERSION}",
    }
    for name, info in manifest["inputs"]["tables"].items():
        assert info["sha256"] == hashlib.sha256(TABLES[name].read_bytes()).hexdigest()
        copy = basic_term_run.run_dir / info["copy"]
        assert hashlib.sha256(copy.read_bytes()).hexdigest() == info["sha256"]
    # the model points were given in memory (the premium rate joined on): no path, a copy
    assert manifest["inputs"]["model_points"]["path"] is None
    assert manifest["inputs"]["model_points"]["rows"] == 10_000
    copy = pq.read_table(basic_term_run.run_dir / manifest["inputs"]["model_points"]["copy"])
    assert "premium_rate" in copy.column_names
    assert manifest["dimensions"] == {
        "policies": 10_000,
        "scenarios": 1,
        "months": 277,
        "horizon": 276,
    }


def test_premium_rate_join_rejects_a_missing_pair() -> None:
    from halley.errors import LoadError

    pt = pq.read_table(FIXTURES / "premium_table.parquet")
    with pytest.raises(LoadError, match="age_at_entry 19, policy_term 10"):
        premium_rates(
            np.array([19]),
            np.array([10]),
            pt.column("age_at_entry").to_numpy(),
            pt.column("policy_term").to_numpy(),
            pt.column("premium_rate").to_numpy(),
        )


# ------------------------------------------------------------------- lifelib itself, if here


def _load_make_reference() -> ModuleType:
    spec = importlib.util.spec_from_file_location("make_reference", FIXTURES / "make_reference.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_live_lifelib_matches_committed_results(tmp_path: Path) -> None:
    """With ``uv sync --group lifelib``: rerun BasicTerm_ME and compare the committed files."""
    lifelib = pytest.importorskip("lifelib")
    pytest.importorskip("modelx")
    pytest.importorskip("pandas")
    pytest.importorskip("openpyxl")
    if lifelib.__version__ != LIFELIB_VERSION:
        pytest.skip(f"the fixtures are from lifelib {LIFELIB_VERSION}; {lifelib.__version__} here")
    make_reference = _load_make_reference()
    make_reference.write_inputs(make_reference.model_dir(), tmp_path)
    make_reference.run_lifelib(make_reference.model_dir(), tmp_path)
    for name in ("model_point_table", "mort_table", "disc_rate_ann", "premium_table"):
        fresh = pq.read_table(tmp_path / f"{name}.parquet")
        committed = pq.read_table(FIXTURES / f"{name}.parquet")
        assert fresh.equals(committed), name
    mp = pq.read_table(FIXTURES / "model_point_table.parquet")
    count = mp.column("policy_count").to_numpy().astype(np.float64)
    money = mp.column("sum_assured").to_numpy().astype(np.float64) * count
    pv_fresh = pq.read_table(tmp_path / "results_pv.parquet")
    pv = pq.read_table(FIXTURES / "results_pv.parquet")
    for name in pv.column_names:
        a = pv_fresh.column(name).to_numpy()
        b = pv.column(name).to_numpy()
        scale = money if name in PV_MONEY else count
        assert within(a, b, scale).all(), name  # BLAS order may differ between machines
    for name in ("results_monthly", "results_totals"):
        fresh = pq.read_table(tmp_path / f"{name}.parquet")
        committed = pq.read_table(FIXTURES / f"{name}.parquet")
        assert fresh.column_names == committed.column_names, name
        for col in committed.column_names:
            a = fresh.column(col).to_numpy()
            b = committed.column(col).to_numpy()
            assert within(a, b, float(money.sum())).all(), (name, col)
