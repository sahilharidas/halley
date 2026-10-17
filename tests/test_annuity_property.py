# pyright: basic
"""Property-based tests of the annuity (Hypothesis, D24; annuity spec "Property-based tests").

Random model points inside the schema, random valuation month-ends, flat or sloped curves and
CPI paths inside the file ranges, and the properties the spec lists: scaling, monotonicity in
age, sex and the stress parameters, the equivalences between the classes, the investment-expense
equivalence, and independence from row order, batch composition and the horizon. Every run also
checks I1 to I10 as run-time invariants.
"""

from __future__ import annotations

import calendar
import datetime as dt
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from halley import CompiledModel, RunResult, RunSettings, build, run
from halley.products import annuity_data as data
from halley.products.annuity import LevelAnnuity
from halley.runtime import within

T_FILE = 1_000
"""The generated curves cover every month a run can read, with room for a longer horizon."""
EXAMPLES = 6
RUNS = settings(max_examples=EXAMPLES, deadline=None, suppress_health_check=[HealthCheck.too_slow])

IN_FORCE = ("pols_if", "deaths", "ann_out", "exp_out", "inv_exp", "bel", "ra", "cu")
PER_LIFE = ("ann_pp", "v_be", "v_st")


def _month_end(year: int, month: int) -> dt.date:
    return dt.date(year, month, calendar.monthrange(year, month)[1])


def _from_month_index(m: int, day: int = 1) -> dt.date:
    return dt.date(m // 12, m % 12 + 1, day)


# ------------------------------------------------------------------------------ strategies


@st.composite
def valuation_dates(draw: Any) -> dt.date:
    return _month_end(draw(st.integers(2025, 2040)), draw(st.integers(1, 12)))


@st.composite
def policies(draw: Any, valuation_date: dt.date) -> dict[str, Any]:
    """One model point inside the schema at the given V: ages 50 to 110, a start date up to 30
    years back or on V + 1 day (then issued on V), positive counts, instalments and premiums."""
    m_v = 12 * valuation_date.year + valuation_date.month - 1
    a0 = draw(st.integers(600, 1320))
    months_back = draw(st.one_of(st.just(0), st.integers(0, 360)))
    start = _from_month_index(m_v + 1 - months_back)
    issue = start - dt.timedelta(days=1)  # V itself when the annuity starts on V + 1 day
    return {
        "policy_id": 1,
        "portfolio": "IA",
        "cohort": issue.year,
        "profit_bucket": "OTHER",
        "sex": draw(st.sampled_from(["F", "M"])),
        "date_of_birth": _from_month_index(m_v - a0, draw(st.integers(1, 28))),
        "issue_date": issue,
        "annuity_start_date": start,
        "pols_if_init": draw(st.floats(1e-3, 5e5)),
        "instalment_pm": draw(st.floats(1.0, 1e5)),
        "single_premium": draw(st.floats(1.0, 1e6)),
        "esc_rate": draw(st.floats(0.0, 0.10)),
    }


@st.composite
def curves(draw: Any) -> Any:
    """rf_df(t) = exp(-z(tau) tau) with a flat or sloped zero curve z(tau) = z0 + slope tau."""
    z0 = draw(st.floats(0.0, 0.06))
    slope = draw(st.floats(0.0, 0.002))
    return lambda s, t: math.exp(-(z0 + slope * (t / 12)) * (t / 12))


@st.composite
def cpi_paths(draw: Any, geometric: bool = False) -> Any:
    """I(u) = 100 (1 + g_h)^(u/12) for u <= 0 and 100 (1 + g_p)^(u/12) for u > 0; one rate
    throughout when ``geometric``."""
    g_h = draw(st.floats(0.0, 0.095) if geometric else st.floats(-0.02, 0.10))
    g_p = g_h if geometric else draw(st.floats(-0.02, 0.10))
    return lambda s, u: 100.0 * (1 + g_h) ** (u / 12) if u <= 0 else 100.0 * (1 + g_p) ** (u / 12)


@st.composite
def bases(draw: Any) -> dict[str, Any]:
    """The run's tables beside the reference mortality: expense level, investment-expense rate,
    illiquidity premium and risk-adjustment parameters with sigma_e <= z_alpha; a zero
    expense level or sigma is a real zero, not a subnormal, so a strict monotonicity holds."""
    positive_or_zero = st.one_of(st.just(0.0), st.floats(1e-3, 0.3))
    return {
        "maint_pm": draw(st.one_of(st.just(0.0), st.floats(0.01, 50.0))),
        "ie_rate": draw(st.floats(0.0, 0.005)),
        "ilp": draw(st.floats(0.0, 0.02)),
        "alpha": draw(st.floats(0.7, 0.95)),
        "sigma_l": draw(positive_or_zero),
        "sigma_e": draw(positive_or_zero),
    }


@st.composite
def cases(draw: Any, geometric: bool = False) -> dict[str, Any]:
    v = draw(valuation_dates())
    return {
        "valuation_date": v,
        "policy": draw(policies(v)),
        "curve": draw(curves()),
        "cpi": draw(cpi_paths(geometric)),
        "basis": draw(bases()),
    }


# -------------------------------------------------------------------------------- runs

MORT = {"mort_base": data.mort_base_table(), "mort_cif": data.mort_cif_table()}


@pytest.fixture(scope="module")
def compiled_level_no_inv_exp() -> CompiledModel:
    """The level class with the investment-expense switch off (D8)."""
    compiled = build(LevelAnnuity, include_inv_exp=False)
    assert compiled.graph.statics == {"include_inv_exp": False}
    return compiled


def _tables(v: dt.date, basis: dict[str, Any], **changes: Any) -> dict[str, Any]:
    b = {**basis, **changes}
    return {
        **MORT,
        "expense": data.expense_table(b["maint_pm"], f"{v.year:04d}-{v.month:02d}"),
        "inv_expense": data.inv_expense_table(b["ie_rate"]),
        "ilp": data.ilp_table(b["ilp"]),
        "ra_param": data.ra_param_table(
            b["alpha"], NormalDist().inv_cdf(b["alpha"]), b["sigma_l"], b["sigma_e"]
        ),
    }


def _scenarios(curve: Any, cpi: Any) -> dict[str, Any]:
    return {
        "rf_df": data.discount_table(curve, t_max=T_FILE),
        "cpi": data.cpi_table(cpi, u_lo=-26, u_hi=T_FILE),
    }


def _run(
    compiled: CompiledModel,
    case: dict[str, Any],
    rows: list[dict[str, Any]],
    tmp: Any,
    *,
    fixed: bool = False,
    tables: dict[str, Any] | None = None,
    scenarios: dict[str, Any] | None = None,
    **settings: Any,
) -> RunResult:
    return run(
        compiled,
        model_points=data.model_points_table(rows, fixed=fixed),
        tables=tables or _tables(case["valuation_date"], case["basis"]),
        scenarios=scenarios or _scenarios(case["curve"], case["cpi"]),
        settings=RunSettings(valuation_date=case["valuation_date"], **settings),
        run_dir=Path(tmp.mktemp("prop")) / "run",
    )


def _scale_of(res: RunResult, name: str, factor: float = 1.0) -> Any:
    """The spec's tolerance scale of a variable, per policy, from the run itself: bel(0) for
    in-force money (the floor I6 uses, which keeps a difference such as ra meaningful), v_be(0)
    per life, pols_if(0) for counts, 1 otherwise."""
    if name in ("bel", "ra", "ann_out", "exp_out", "inv_exp", "cu", "pva", "pve"):
        return factor * res.var("bel")[:1]
    if name in PER_LIFE:
        return factor * res.var("v_be")[:1]
    if name in ("pols_if", "deaths"):
        return factor * res.var("pols_if")[:1]
    return 1.0


def _same(
    a: RunResult, b: RunResult, names: tuple[str, ...], rel: float, factor: float = 1.0
) -> None:
    """Every named variable of ``a`` (times ``factor``) agrees with ``b`` within ``rel`` of
    max(|a|, |b|, the unit's scale)."""
    for name in names:
        ok = within(factor * a.var(name), b.var(name), _scale_of(b, name), rel=rel)
        assert ok.all(), (name, int(np.argwhere(~ok)[0][0]))


# -------------------------------------------------------------------------- properties


@RUNS
@given(case=cases())
def test_doubling_the_count_doubles_every_in_force_amount_bit_for_bit(
    case: dict[str, Any], compiled_annuity: dict[str, CompiledModel], tmp_path_factory: Any
) -> None:
    row = case["policy"]
    one = _run(compiled_annuity["level"], case, [row], tmp_path_factory)
    two = _run(
        compiled_annuity["level"],
        case,
        [{**row, "pols_if_init": 2 * row["pols_if_init"]}],
        tmp_path_factory,
    )
    for name in IN_FORCE:
        assert np.array_equal(2 * one.var(name), two.var(name)), name
    for name in PER_LIFE:
        assert np.array_equal(one.var(name), two.var(name)), name
    assert np.array_equal(2 * one.var("pva"), two.var("pva"))


@RUNS
@given(case=cases(), k=st.floats(0.5, 2.0))
def test_the_liability_scales_with_the_instalment_when_there_are_no_expenses(
    case: dict[str, Any],
    k: float,
    compiled_annuity: dict[str, CompiledModel],
    tmp_path_factory: Any,
) -> None:
    row = case["policy"]
    tables = _tables(case["valuation_date"], case["basis"], maint_pm=0.0)
    base = _run(compiled_annuity["level"], case, [row], tmp_path_factory, tables=tables)
    scaled = _run(
        compiled_annuity["level"],
        case,
        [{**row, "instalment_pm": k * row["instalment_pm"]}],
        tmp_path_factory,
        tables=tables,
    )
    _same(base, scaled, ("bel", "ra", "ann_out", "cu", "v_be", "v_st", "pva"), 1e-12, k)
    assert not base.var("exp_out").any()


@RUNS
@given(case=cases(), older_by=st.integers(1, 240))
def test_an_older_annuitant_and_a_male_have_lower_per_life_values(
    case: dict[str, Any],
    older_by: int,
    compiled_annuity: dict[str, CompiledModel],
    tmp_path_factory: Any,
) -> None:
    row = {**case["policy"], "sex": "M"}
    m_v = 12 * case["valuation_date"].year + case["valuation_date"].month - 1
    a0 = m_v - (12 * row["date_of_birth"].year + row["date_of_birth"].month - 1)
    assume(a0 + older_by <= 1320)
    older = {**row, "policy_id": 2, "date_of_birth": _from_month_index(m_v - a0 - older_by, 1)}
    female = {**row, "policy_id": 3, "sex": "F"}
    tables = _tables(case["valuation_date"], case["basis"], maint_pm=0.0)
    res = _run(
        compiled_annuity["level"], case, [row, older, female], tmp_path_factory, tables=tables
    )
    v_be = {pid: res.lane("v_be", pid)[0] for pid in (1, 2, 3)}
    assert v_be[2] < v_be[1]  # older, otherwise identical
    assert v_be[3] > v_be[1]  # female against male at the same age
    assert res.lane("bel", 2)[0] < res.lane("bel", 1)[0] < res.lane("bel", 3)[0]


@RUNS
@given(case=cases())
def test_the_risk_adjustment_grows_with_the_stress(
    case: dict[str, Any], compiled_annuity: dict[str, CompiledModel], tmp_path_factory: Any
) -> None:
    row = case["policy"]
    v, basis = case["valuation_date"], case["basis"]
    base = _run(compiled_annuity["level"], case, [row], tmp_path_factory)
    ra0 = base.var("ra")
    variants = {
        "sigma_l": _tables(v, basis, sigma_l=basis["sigma_l"] + 0.05),
        "sigma_e": _tables(v, basis, sigma_e=basis["sigma_e"] + 0.05),
        "alpha": _tables(v, basis, alpha=basis["alpha"] + 0.02),
    }
    for name, tables in variants.items():
        ra = _run(compiled_annuity["level"], case, [row], tmp_path_factory, tables=tables).var("ra")
        assert (ra >= ra0).all(), name  # not smaller at any t
        strictly = (
            name == "sigma_l"
            or (name == "sigma_e" and basis["maint_pm"] > 0)
            or (
                name == "alpha"
                and (basis["sigma_l"] > 0 or (basis["sigma_e"] > 0 and basis["maint_pm"] > 0))
            )
        )
        if strictly:
            assert ra[0] > ra0[0], name
    assert (ra0 >= 0).all()


@RUNS
@given(case=cases())
def test_fixed_escalation_at_zero_is_the_level_class_bit_for_bit(
    case: dict[str, Any], compiled_annuity: dict[str, CompiledModel], tmp_path_factory: Any
) -> None:
    row = case["policy"]
    level = _run(compiled_annuity["level"], case, [row], tmp_path_factory)
    fixed = _run(
        compiled_annuity["fixed"], case, [{**row, "esc_rate": 0.0}], tmp_path_factory, fixed=True
    )
    for name, arr in level.arrays.items():
        assert np.array_equal(arr, fixed.arrays[name]), name
    assert fixed.var("anniv").any()  # the mask is alive, the step is exactly 1


@RUNS
@given(case=cases(geometric=True))
def test_a_geometric_cpi_path_makes_the_cpi_class_the_fixed_class(
    case: dict[str, Any], compiled_annuity: dict[str, CompiledModel], tmp_path_factory: Any
) -> None:
    row = case["policy"]
    g = case["cpi"](0, 12) / case["cpi"](0, 0) - 1  # the path's annual rate
    assume(0.0 <= g <= 0.10)
    cpi = _run(compiled_annuity["cpi"], case, [row], tmp_path_factory)
    fixed = _run(
        compiled_annuity["fixed"], case, [{**row, "esc_rate": g}], tmp_path_factory, fixed=True
    )
    _same(cpi, fixed, ("ann_pp", "bel", "ra", "cu", "ann_out", "v_be", "v_st"), rel=1e-12)
    assert np.array_equal(cpi.var("anniv"), fixed.var("anniv"))


@RUNS
@given(case=cases())
def test_investment_expenses_equal_a_loaded_discount_curve(
    case: dict[str, Any],
    compiled_level_no_inv_exp: CompiledModel,
    compiled_annuity: dict[str, CompiledModel],
    tmp_path_factory: Any,
) -> None:
    """include_inv_exp = False on P(t) (1 + ie_m)^t gives the bel of include_inv_exp = True on
    P(t): the charge on the closing BEL is the same as a lower discount rate."""
    row = case["policy"]
    with_charge = _run(compiled_annuity["level"], case, [row], tmp_path_factory)
    ie_m = (1 + case["basis"]["ie_rate"]) ** (1 / 12) - 1
    curve = case["curve"]
    loaded = _scenarios(lambda s, t: curve(s, t) * (1 + ie_m) ** t, case["cpi"])
    without = _run(compiled_level_no_inv_exp, case, [row], tmp_path_factory, scenarios=loaded)
    assert not without.var("inv_exp").any()
    bel0 = with_charge.lane("bel", 1)[0]
    assert within(bel0, without.lane("bel", 1)[0], 0.0, rel=1e-12)
    _same(with_charge, without, ("bel", "ra", "v_be", "v_st"), rel=1e-12)
    _same(with_charge, without, ("pols_if", "ann_out", "exp_out", "cu"), rel=0.0)
    # the split of the liability differs: the payments are discounted on the loaded curve
    if ie_m > 0:  # a real charge, not a rate that rounds to nothing per month
        assert without.lane("pva", 1)[0] > with_charge.lane("pva", 1)[0]
    for res in (with_charge, without):
        pva, pve = res.lane("pva", 1)[0], res.lane("pve", 1)[0]
        assert within(pva + pve, res.lane("bel", 1)[0], 0.0, rel=1e-12)


@RUNS
@given(case=cases(), data_=st.data())
def test_permuted_rows_and_batches_give_the_same_results(
    case: dict[str, Any],
    data_: Any,
    compiled_annuity: dict[str, CompiledModel],
    tmp_path_factory: Any,
) -> None:
    """The loader sorts by policy_id, so permuted rows are bit-identical (D19); the same policy
    in batches of different composition agrees within 1e-12 (D5)."""
    v = case["valuation_date"]
    rows = [
        {**data_.draw(policies(v)), "policy_id": pid, "profit_bucket": bucket}
        for pid, bucket in ((3, "OTHER"), (1, "ONEROUS"), (2, "NO_SIGNIFICANT_RISK"))
    ]
    forward = _run(compiled_annuity["level"], case, rows, tmp_path_factory)
    reverse = _run(compiled_annuity["level"], case, rows[::-1], tmp_path_factory)
    assert list(forward.ids) == [1, 2, 3]
    for name, arr in forward.arrays.items():
        assert np.array_equal(arr, reverse.arrays[name]), name
    single = _run(compiled_annuity["level"], case, rows, tmp_path_factory, batch_size=1)
    _same(forward, single, (*IN_FORCE, *PER_LIFE, "pva", "pve"), rel=1e-12)
    alone = _run(compiled_annuity["level"], case, [rows[0]], tmp_path_factory)
    for name in (*IN_FORCE, "pva", "pve"):
        a = forward.lane(name, 3)[: alone.H + 1]
        scale = _scale_of(alone, name)[0, 0, 0]
        assert within(a, alone.lane(name, 3), scale, rel=1e-12).all(), name


@RUNS
@given(case=cases(), extra=st.integers(1, 60))
def test_results_do_not_depend_on_the_horizon(
    case: dict[str, Any],
    extra: int,
    compiled_annuity: dict[str, CompiledModel],
    tmp_path_factory: Any,
) -> None:
    row = case["policy"]
    short = _run(compiled_annuity["level"], case, [row], tmp_path_factory)
    t_end = int(short.lane("run_off_month", 1)[0])
    assert t_end == short.H
    long = _run(compiled_annuity["level"], case, [row], tmp_path_factory, horizon=t_end + extra)
    assert t_end + extra == long.H
    for name in IN_FORCE:
        a, b = short.lane(name, 1), long.lane(name, 1)[: t_end + 1]
        assert within(a, b, _scale_of(short, name)[0, 0, 0], rel=1e-12).all(), name
        assert not long.lane(name, 1)[t_end + 1 :].any(), name
    for name in PER_LIFE:
        a, b = short.lane(name, 1)[:t_end], long.lane(name, 1)[:t_end]
        assert within(a, b, _scale_of(short, name)[0, 0, 0], rel=1e-12).all(), name
    for name in ("pva", "pve"):
        scale = _scale_of(short, name)[0, 0, 0]
        assert within(short.lane(name, 1)[0], long.lane(name, 1)[0], scale, rel=1e-12), name
    assert short.lane("onerous", 1)[0] == long.lane("onerous", 1)[0]
