# pyright: basic
"""The formula helper table (annuity spec, "Compounding and unit conversions"): each helper is one
Call node with a fixed emitted form, checked for units, and ``pv_month`` expands at trace time by
the cash flow's timing. One model uses every helper so that a run cross-checks the backend against
the ``explain`` interpreter (tests/test_explain.py)."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest

from halley import (
    Annual,
    BuildError,
    Count,
    CoverageUnits,
    Factor,
    Field,
    Index,
    Int,
    Money,
    Monthly,
    Prob,
    Product,
    Rate,
    RunResult,
    RunSettings,
    ScenarioInput,
    Schema,
    UnitError,
    ann_fwd,
    ann_spot,
    annual_step,
    as_cu,
    build,
    cc_discount,
    check,
    fwd_rate,
    pv_month,
    run,
    sum_t,
    to_monthly,
    to_monthly_cf,
    trace,
    var,
    where,
)
from halley.authoring import exp, int_field
from halley.errors import Loc
from halley.ir import render
from halley.ir import units as u
from models import VAL

HELPER_SCHEMA = Schema(
    fields=(
        Field("policy_id", "string", unique=True),
        Field("pols_if_init", "number", "Count", gt=0),
        Field("premium", "number", "MoneyPerPolicy", gt=0),
        Field("q_ann", "number", "Prob[Annual]", ge=0, le=1),
        Field("esc", "number", "Rate[Annual]", ge=0),
        Field("ilp", "number", "Rate[Annual, Continuous]", ge=0),
        int_field("rem_m", ge=1, le=1000),
    ),
    horizon="rem_m",
)


def helper_model_points(rem_m: int = 12) -> Any:
    """Three policies: a small, a large and a certain annual mortality, with and without
    escalation and illiquidity premium."""
    return pa.table(
        {
            "policy_id": pa.array(["H00", "H01", "H02"], pa.string()),
            "pols_if_init": pa.array([1.0, 2.5, 100.0], pa.float64()),
            "premium": pa.array([10.0, 20.0, 30.0], pa.float64()),
            "q_ann": pa.array([0.01, 0.5, 1.0], pa.float64()),
            "esc": pa.array([0.0, 0.03, 0.1], pa.float64()),
            "ilp": pa.array([0.0, 0.005, 0.02], pa.float64()),
            "rem_m": pa.array([rem_m] * 3, pa.int32()),
        }
    )


class Helpers(Product):
    """Every formula helper once, plus the two run constants."""

    model_point = HELPER_SCHEMA

    @var
    def q_m(self, t: Int) -> Prob[Monthly]:
        return to_monthly_cf(self.mp.q_ann)

    @var
    def step(self) -> Factor:
        return annual_step(self.mp.esc)

    @var
    def v_m(self) -> Factor:
        return (1 + self.mp.esc) ** (-1 / 12)

    @var(domain="state")
    def df(self, t: Int) -> Factor:
        return (self.v_m() ** t) * cc_discount(self.mp.ilp, months=t)

    @var
    def v(self, t: Int) -> Factor:
        return self.df(t) / self.df(t - 1)

    @var
    def j(self, t: Int) -> Rate[Monthly]:
        return fwd_rate(self.df(t - 1), self.df(t))

    @var
    def j_ann(self, t: Int) -> Rate[Annual]:
        return ann_fwd(self.df(t - 1), self.df(t))

    @var
    def spot(self, t: Int) -> Rate[Annual]:
        return where(t > 0, ann_spot(self.df(t), t), 0.0)

    @var(init=lambda mp: mp.pols_if_init)
    def pols_if(self, t: Int) -> Count:
        return self.pols_if(t - 1) * (1 - self.q_m(t))

    @var(timing="start")
    def ann_out(self, t: Int) -> Money:
        return self.pols_if(t - 1) * self.mp.premium

    @var(timing="mid")
    def exp_out(self, t: Int) -> Money:
        return self.pols_if(t - 1) * (self.mp.premium * 0.1)

    @var(timing="end")
    def claims(self, t: Int) -> Money:
        return (self.pols_if(t - 1) - self.pols_if(t)) * (self.mp.premium * 100.0)

    @var
    def cu(self, t: Int) -> CoverageUnits:
        return as_cu(self.ann_out(t))

    @var
    def cu_total(self) -> CoverageUnits:
        return sum_t(self.cu)

    @var
    def gap(self, t: Int) -> Money:
        return abs(self.ann_out(t) - self.claims(t))

    @var(terminal=0)
    def bel(self, t: Int) -> Money:
        return (
            ((self.bel(t + 1) * self.v(t + 1)) + pv_month(self.ann_out, t + 1, self.v))
            + pv_month(self.exp_out, t + 1, self.v)
            + pv_month(self.claims, t + 1, self.v)
        )

    @var
    def cal_year(self, t: Int) -> Int:
        return (self.run.m_val + t) // 12

    @var
    def months_left(self, t: Int) -> Int:
        return self.run.horizon - t


def run_helpers(tmp_path: Path, **settings: Any) -> RunResult:
    return run(
        build(Helpers),
        model_points=helper_model_points(),
        settings=RunSettings(valuation_date=VAL, **settings),
        run_dir=tmp_path / "helpers",
    )


def _inside(loc: Loc | None, model: type[Product], variable: str) -> None:
    """The error points at a line of ``variable``'s method in ``model``'s own source: the right
    file, the method's qualified name and a line within the decorated method."""
    assert loc is not None, "every error carries a model source location"
    fn = vars(model)[variable].fn
    lines, first = inspect.getsourcelines(fn)
    assert loc.file == fn.__code__.co_filename, loc
    assert loc.function.endswith(f"{model.__name__}.{variable}"), loc
    assert first <= loc.line < first + len(lines), (loc, first, len(lines))


def _fails(
    model: type[Product], variable: str, match: str, error: type[Exception] = BuildError
) -> Exception:
    """``check(trace(model))`` raises ``error`` matching ``match``, located inside ``variable``."""
    with pytest.raises(error, match=match) as info:
        check(trace(model))
    exc = info.value
    _inside(getattr(exc, "loc", None), model, variable)
    return exc


def _pv_month_fails(model: type[Product], match: str) -> None:
    """``trace(model)`` fails at the ``pv_month`` call site inside the model's ``bel``."""
    with pytest.raises(BuildError, match=match) as info:
        trace(model)
    _inside(info.value.loc, model, "bel")


# --------------------------------------------------------------------------- the Call nodes


def test_helpers_are_call_nodes_with_the_declared_units() -> None:
    g = check(trace(Helpers))
    v = g.variables
    assert render(v["q_m"].expr) == "to_monthly_cf(mp.q_ann)"
    assert render(v["step"].expr) == "annual_step(mp.esc)"
    assert render(v["df"].expr) == "((v_m(t) ** t) * cc_discount(mp.ilp, t))"
    assert render(v["j"].expr) == "fwd_rate(df(t-1), df(t))"
    assert render(v["j_ann"].expr) == "ann_fwd(df(t-1), df(t))"
    assert render(v["spot"].expr) == "where((t > 0), ann_spot(df(t), t), 0.0)"
    assert render(v["cu"].expr) == "as_cu(ann_out(t))"
    assert render(v["gap"].expr) == "abs((ann_out(t) - claims(t)))"
    assert v["q_m"].expr.unit == u.PROB_MONTHLY
    assert v["step"].expr.unit == u.FACTOR
    assert v["j"].expr.unit == u.RATE_MONTHLY
    assert v["j_ann"].expr.unit == u.RATE_ANNUAL
    assert v["spot"].expr.unit == u.RATE_ANNUAL
    assert v["cu"].expr.unit == u.COVERAGE_UNITS
    assert v["cu_total"].expr.unit == u.COVERAGE_UNITS
    assert v["gap"].expr.unit == u.MONEY


def test_helpers_emit_exactly_the_spec_forms() -> None:
    source = build(Helpers).source
    expected = (
        # to_monthly_cf(q), whole mode over a model point field
        "np.where(M_q_ann >= 1, 1.0, -np.expm1(np.log1p(-np.where(M_q_ann >= 1, 0.0, "
        "M_q_ann)) / 12))",
        # annual_step(g)
        "V_step[...] = (1 + M_esc)",
        # cc_discount(s, months=t), a state evaluated from t = 0
        "np.exp(((-M_ilp) * _t[0:H+1]) / 12)",
        # fwd_rate(p0, p1) and ann_fwd(p0, p1)
        "V_j[1:H+1] = ((V_df[0:H] / V_df[1:H+1]) - 1)",
        "V_j_ann[1:H+1] = (((V_df[0:H] / V_df[1:H+1]) ** 12) - 1)",
        # ann_spot(p, m) inside the caller's where() guard
        "((V_df[1:H+1] ** (-12 / _t[1:H+1])) - 1)",
        # as_cu(x) is x
        "V_cu[1:H+1] = V_ann_out[1:H+1]",
        # abs(x)
        "np.abs((V_ann_out[1:H+1] - V_claims[1:H+1]))",
    )
    for text in expected:
        assert text in source, text


def test_abs_of_a_symbolic_value_keeps_its_unit() -> None:
    g = check(trace(Helpers))
    assert g.variables["gap"].unit == u.MONEY

    class Bad(Helpers):
        @var
        def gap(self, t: Int) -> Money:
            return abs(self.pols_if(t) > 0) * self.mp.premium  # type: ignore[operator, return-value]

    _fails(Bad, "gap", r"abs\(\) takes a quantity, not a Bool", UnitError)


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("to_monthly_cf_of_a_rate", r"to_monthly_cf\(\) converts a Prob\[Annual\], not a Rate"),
        (
            "annual_step_of_a_continuous_rate",
            r"annual_step\(\) converts an effective Rate\[Annual\], not a "
            r"Rate\[Annual, Continuous\]",
        ),
        (
            "to_monthly_of_a_continuous_rate",
            r"to_monthly\(\) converts an effective Rate\[Annual\], not a Rate\[Annual, Cont",
        ),
        (
            "cc_discount_of_an_effective_rate",
            r"cc_discount\(\) discounts with a Rate\[Annual, Continuous\], not a Rate\[Annual\]",
        ),
        ("cc_discount_months_not_int", r"the months in cc_discount\(\) is an Int, not a Money"),
        ("fwd_rate_of_money", r"fwd_rate\(\) takes two discount factors \(Factor\), not a Money"),
        ("ann_fwd_of_money", r"ann_fwd\(\) takes two discount factors \(Factor\), not a Money"),
        ("ann_spot_month_not_int", r"the month in ann_spot\(\) is an Int, not a Factor"),
        ("as_cu_of_a_count", r"as_cu\(\) converts Money to coverage units, not a Count"),
        ("money_plus_coverage_units", "coverage units are not money; convert with as_cu"),
        ("one_plus_continuous_rate", "accumulates through cc_discount"),
        ("exp_of_an_index", r"exp\(\) takes a dimensionless Factor, not Index"),
    ],
)
def test_helper_unit_mistakes_are_build_errors(name: str, message: str) -> None:
    class Bad(Helpers):
        scenario_inputs = (ScenarioInput("cpi", "Index"),)

        @var
        def wrong(self, t: Int) -> Factor:
            # the model point fields are untyped (Any) for pyright; the typed mistakes carry
            # the ignore pyright would otherwise flag, as the checker tests do
            if name == "to_monthly_cf_of_a_rate":
                return to_monthly_cf(self.mp.esc)  # type: ignore[return-value]
            if name == "annual_step_of_a_continuous_rate":
                return annual_step(self.mp.ilp)
            if name == "to_monthly_of_a_continuous_rate":
                return to_monthly(self.mp.ilp)  # type: ignore[return-value]
            if name == "cc_discount_of_an_effective_rate":
                return cc_discount(self.mp.esc, months=t)
            if name == "cc_discount_months_not_int":
                return cc_discount(self.mp.ilp, months=self.ann_out(t))  # type: ignore[arg-type]
            if name == "fwd_rate_of_money":
                return fwd_rate(self.ann_out(t), self.df(t))  # type: ignore[arg-type, return-value]
            if name == "ann_fwd_of_money":
                return ann_fwd(self.df(t), self.ann_out(t))  # type: ignore[arg-type, return-value]
            if name == "ann_spot_month_not_int":
                return ann_spot(self.df(t), self.df(t))  # type: ignore[arg-type, return-value]
            if name == "as_cu_of_a_count":
                return as_cu(self.pols_if(t))  # type: ignore[arg-type, return-value]
            if name == "money_plus_coverage_units":
                return self.ann_out(t) + self.cu(t)  # type: ignore[operator, return-value]
            if name == "one_plus_continuous_rate":
                return 1 + self.mp.ilp
            return exp(self.scen.cpi(t))

    _fails(Bad, "wrong", message, UnitError)


def test_sum_t_accepts_coverage_units_but_not_an_index() -> None:
    class Level(Product):
        model_point = HELPER_SCHEMA
        scenario_inputs = (ScenarioInput("cpi", "Index"),)

        @var
        def lvl(self, t: Int) -> Index:
            return self.scen.cpi(t)

        @var
        def ratio(self, t: Int) -> Factor:
            return self.scen.cpi(t) / self.scen.cpi(t - 12)

        @var
        def total(self) -> Index:
            return sum_t(self.lvl)

    _fails(
        Level,
        "total",
        r"sum_t\(lvl\) sums a Index; only Money, MoneyPerPolicy, Count, Factor and "
        "CoverageUnits are summed over time",
        UnitError,
    )
    assert check(trace(Helpers)).variables["cu_total"].unit == u.COVERAGE_UNITS


# ---------------------------------------------------------------------------------- pv_month


def test_pv_month_expands_by_the_cash_flows_timing() -> None:
    g = check(trace(Helpers))
    assert render(g.variables["bel"].expr) == (
        "((((bel(t+1) * v(t+1)) + ann_out(t+1)) + (exp_out(t+1) * (v(t+1) ** 0.5))) + "
        "(claims(t+1) * v(t+1)))"
    )
    source = build(Helpers).source
    assert (
        "V_bel[t] = ((((V_bel[t+1] * V_v[t+1]) + V_ann_out[t+1]) + (V_exp_out[t+1] * "
        "(V_v[t+1] ** 0.5))) + (V_claims[t+1] * V_v[t+1]))"
    ) in source


def test_pv_month_folds_the_offset_of_its_month_argument() -> None:
    class Shifted(Helpers):
        @var(terminal=0)
        def bel(self, t: Int) -> Money:
            return self.bel(t + 1) * 0.5 + pv_month(self.claims, (t + 2) - 1, self.v)

    g = trace(Shifted)
    assert render(g.variables["bel"].expr) == "((bel(t+1) * 0.5) + (claims(t+1) * v(t+1)))"


def test_pv_month_needs_a_timed_cash_flow_and_a_factor_discount_variable() -> None:
    class NoTiming(Helpers):
        @var(terminal=0)
        def bel(self, t: Int) -> Money:
            return self.bel(t + 1) * 0.5 + pv_month(self.gap, t + 1, self.v)

    _pv_month_fails(NoTiming, r"pv_month\(gap, ...\) needs the timing of gap")

    class ScalarFlow(Helpers):
        @var(terminal=0)
        def bel(self, t: Int) -> Money:
            return self.bel(t + 1) * 0.5 + pv_month(self.cu_total, t + 1, self.v)  # type: ignore[arg-type]

    _pv_month_fails(ScalarFlow, "takes a cash flow variable with a time dimension")

    class ScalarDiscount(Helpers):
        @var(terminal=0)
        def bel(self, t: Int) -> Money:
            return self.bel(t + 1) * 0.5 + pv_month(self.claims, t + 1, self.v_m)  # type: ignore[arg-type]

    _pv_month_fails(ScalarDiscount, "discount factor as a variable with a time dimension")

    class WrongDiscountUnit(Helpers):
        @var(terminal=0)
        def bel(self, t: Int) -> Money:
            return self.bel(t + 1) * 0.5 + pv_month(self.claims, t + 1, self.j)  # type: ignore[arg-type]

    _pv_month_fails(WrongDiscountUnit, r"discounts with j, which is declared Rate\[Monthly\]")

    class NotConstant(Helpers):
        @var(terminal=0)
        def bel(self, t: Int) -> Money:
            return self.bel(t + 1) * 0.5 + pv_month(self.claims, t * 2, self.v)

    _pv_month_fails(NotConstant, "the time offset of claims must be a constant")


# ------------------------------------------------------------------------------- the numbers


def test_helpers_run_and_the_helpers_give_the_textbook_numbers(tmp_path: Path) -> None:
    res = run_helpers(tmp_path)
    assert res.H == 12
    q_ann = np.array([0.01, 0.5, 1.0])
    want_q_m = np.where(q_ann >= 1, 1.0, 1 - (1 - q_ann) ** (1 / 12))
    np.testing.assert_allclose(res.var("q_m")[1, :, 0], want_q_m, rtol=1e-12)
    assert res.lane("q_m", "H02")[1] == 1.0  # q >= 1 gives exactly 1
    assert res.lane("pols_if", "H02")[1] == 0.0
    esc = np.array([0.0, 0.03, 0.1])
    np.testing.assert_array_equal(res.var("step")[0, :, 0], 1 + esc)
    ilp = np.array([0.0, 0.005, 0.02])
    for p, pid in enumerate(res.ids):
        df = res.lane("df", pid)
        want_df = (1 + esc[p]) ** (-np.arange(13) / 12) * np.exp(-ilp[p] * np.arange(13) / 12)
        np.testing.assert_allclose(df, want_df, rtol=1e-12)
        np.testing.assert_allclose(
            res.lane("j", pid)[1:], df[:-1] / df[1:] - 1, rtol=1e-12, atol=1e-15
        )
        np.testing.assert_allclose(
            res.lane("j_ann", pid)[1:], (df[:-1] / df[1:]) ** 12 - 1, rtol=1e-12, atol=1e-15
        )
        m = np.arange(1, 13)
        np.testing.assert_allclose(
            res.lane("spot", pid)[1:], df[1:] ** (-12 / m) - 1, rtol=1e-12, atol=1e-15
        )
        assert res.lane("spot", pid)[0] == 0.0
        np.testing.assert_array_equal(res.lane("cu", pid), res.lane("ann_out", pid))
    assert res.lane("df", "H00")[12] == 1.0  # no escalation, no illiquidity premium: exactly 1
    assert res.lane("cal_year", "H00")[1] == 2027
    assert list(res.lane("cal_year", "H00")[1:]) == [2027] * 11 + [2028]
    assert list(res.lane("months_left", "H00")[1:]) == list(range(11, -1, -1))
    assert res.lane("cu_total", "H00")[0] == res.lane("ann_out", "H00")[1:].cumsum()[-1]
