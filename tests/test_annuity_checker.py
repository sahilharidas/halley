# pyright: basic
"""The annuity spec's "Checker rules exercised" table, one test per row: each deliberate error
fails the build at the model source line (never at generated code), the unused variable warns
and still builds, and each data-contract error fails at load naming the file, table, rule or
row. G3 and G5 belong to the aggregation tests."""

from __future__ import annotations

import datetime as dt
import inspect
import warnings
from dataclasses import replace
from pathlib import Path
from statistics import NormalDist
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pytest

from halley import (
    Bool,
    BuildError,
    BuildWarning,
    CompiledModel,
    Count,
    Factor,
    Int,
    LoadError,
    Money,
    MoneyPerPolicy,
    Monthly,
    Product,
    Rate,
    RunError,
    RunSettings,
    as_cu,
    build,
    check,
    pv_month,
    run,
    to_monthly,
    trace,
    var,
    where,
)
from halley.authoring import Column, CoverageUnits, Key, Table, int_field
from halley.errors import Loc
from halley.products import annuity_data as data
from halley.products.annuity import (
    EXPENSE,
    ILP,
    INV_EXPENSE,
    MORT_BASE,
    MORT_CIF,
    RA_PARAM,
    SCHEMA,
    VALUATION_DATE,
    CPILinkedAnnuity,
    EscalatingAnnuity,
    FixedEscalationAnnuity,
    LevelAnnuity,
)


def _inside(loc: Loc | None, model: type[Product], variable: str) -> None:
    """The error points at a line of ``variable``'s method in ``model``'s own source."""
    assert loc is not None, "every build error carries a model source location"
    fn = vars(model)[variable].fn
    lines, first = inspect.getsourcelines(fn)
    assert Path(loc.file) == Path(fn.__code__.co_filename), loc
    assert loc.function.endswith(f"{model.__name__}.{variable}"), loc
    assert first <= loc.line < first + len(lines), (loc, first, len(lines))


def _fails(
    model: type[Product], variable: str, match: str, error: type[Exception] = BuildError
) -> Exception:
    with pytest.raises(error, match=match) as info:
        check(trace(model))
    exc = info.value
    _inside(getattr(exc, "loc", None), model, variable)
    return exc


# --------------------------------------------------------------------------------- units


def test_money_adds_only_to_money() -> None:
    class Bad(LevelAnnuity):
        @var(terminal=0, role="reserve")
        def bel(self, t: Int) -> Money:
            return self.bel(t + 1) + self.pols_if(t)  # type: ignore[operator]

    _fails(Bad, "bel", r"unit mismatch: Money \+ Count")


def test_per_policy_against_in_force() -> None:
    class Bad(LevelAnnuity):
        @var(timing="start")
        def exp_out(self, t: Int) -> Money:
            return self.pols_if(t - 1) * self.ann_out(t)  # type: ignore[operator]

    _fails(Bad, "exp_out", r"Count \* Money: Money \* Count would apply the in-force count twice")


def test_explicit_rate_conversion() -> None:
    class Bad(LevelAnnuity):
        @var
        def ie_m(self) -> Rate[Monthly]:
            return self.tables.inv_expense.ie_rate

    _fails(Bad, "ie_m", r"ie_m is declared Rate\[Monthly\] but its formula is Rate\[Annual\]")


def test_compounding_basis() -> None:
    class Bad(LevelAnnuity):
        @var
        def ie_m(self) -> Rate[Monthly]:
            return to_monthly(self.tables.ilp.ilp)

    _fails(
        Bad,
        "ie_m",
        r"to_monthly\(\) converts an effective Rate\[Annual\], not a Rate\[Annual, Continuous\]",
    )


def test_probability_period() -> None:
    class BadDeaths(LevelAnnuity):
        @var(role="decrement")
        def deaths(self, t: Int) -> Count:
            return self.pols_if(t - 1) * self.q_ann(t)  # type: ignore[operator]

    _fails(
        BadDeaths,
        "deaths",
        r"Count \* Prob\[Annual\]: an annual probability cannot multiply a count in a monthly "
        r"step; convert it with to_monthly_prob\(\)",
    )

    class BadSurvival(LevelAnnuity):
        @var(terminal=0)
        def v_be(self, t: Int) -> MoneyPerPolicy:
            p = 1 - self.q_ann(t + 1)
            nxt = self.v_be(t + 1) + self.ie_m() * self.v_be(t + 1)
            return pv_month(self.s_pp, t + 1, self.v) + p * (self.v(t + 1) * nxt)  # type: ignore[operator]

    _fails(BadSurvival, "v_be", r"unit mismatch: Prob\[Annual\] \* MoneyPerPolicy")


def test_coverage_units_are_not_money() -> None:
    class Bad(LevelAnnuity):
        @var
        def liability_and_units(self, t: Int) -> Money:
            return self.bel(t) + self.cu(t)  # type: ignore[operator]

    _fails(
        Bad,
        "liability_and_units",
        r"Money \+ CoverageUnits: coverage units are not money; convert with as_cu\(\)",
    )


def test_index_ratios() -> None:
    class Bad(CPILinkedAnnuity):
        @var(timing="start", init=lambda mp: mp.instalment_pm)
        def ann_pp(self, t: Int) -> MoneyPerPolicy:
            return self.ann_pp(t - 1) * self.scen.cpi(t - 3)

    _fails(Bad, "ann_pp", r"unit mismatch: MoneyPerPolicy \* Index")


# ------------------------------------------------------------- time, recursion and cycles


def test_a_per_policy_lag_is_not_a_constant_offset() -> None:
    lagged = replace(SCHEMA, fields=(*SCHEMA.fields, int_field("lag_m", ge=0, le=12)))

    class Bad(CPILinkedAnnuity):
        model_point = lagged

        @var
        def esc_factor(self, t: Int) -> Factor:
            lag = self.mp.lag_m
            return self.scen.cpi(t - lag) / self.scen.cpi(t - lag - 12)

    with pytest.raises(
        BuildError, match=r"the time offset of scen\.cpi must be a constant"
    ) as info:
        trace(Bad)
    _inside(info.value.loc, Bad, "esc_factor")
    assert "lag()" in info.value.message


def test_a_backward_recursion_needs_a_terminal() -> None:
    class Bad(LevelAnnuity):
        @var(role="reserve")
        def bel(self, t: Int) -> Money:
            return (
                pv_month(self.ann_out, t + 1, self.v)
                + pv_month(self.exp_out, t + 1, self.v)
                + pv_month(self.inv_exp, t + 1, self.v)
                + self.v(t + 1) * self.bel(t + 1)
            )

    _fails(Bad, "bel", "backward recursive and declares a terminal value")


def test_a_same_period_cycle_fails() -> None:
    class Bad(LevelAnnuity):
        @var(role="decrement")
        def deaths(self, t: Int) -> Count:
            return self.pols_if(t) * self.q_mth(t)

    _fails(Bad, "deaths", r"same-period cycle: \{deaths, pols_if\}")


def test_a_mixed_component_fails() -> None:
    """``deaths`` reading ``pols_if(t+1)`` puts offsets -1 and +1 in {pols_if, deaths}; the
    build fails at this line with a mixed-direction, same-period-cycle or out-of-domain error."""

    class Bad(LevelAnnuity):
        @var(role="decrement")
        def deaths(self, t: Int) -> Count:
            return self.pols_if(t + 1) * self.q_mth(t)

    exc = _fails(
        Bad, "deaths", "mixes forward and backward|same-period cycle|outside the domain of pols_if"
    )
    assert "pols_if(t+1)" in str(exc)


def test_reads_across_policies_are_not_a_projection_construct() -> None:
    """Group sums belong to the aggregation stage; a Product has no ``series`` accessor."""

    class Bad(LevelAnnuity):
        @var
        def cu_total(self, t: Int) -> Money:
            return self.series.cu(t)  # type: ignore[attr-defined]

    with pytest.raises(BuildError, match="cu_total raised AttributeError") as info:
        trace(Bad)
    _inside(info.value.loc, Bad, "cu_total")
    assert "series" in info.value.message


def test_an_unused_variable_warns_and_still_builds() -> None:
    class Extra(LevelAnnuity):
        @var
        def anniv(self, t: Int) -> Bool:
            k = t - self.mp.m_c
            return (k > 0) & (k % 12 == 0)

    with pytest.warns(BuildWarning, match="anniv is read by no other variable") as record:
        compiled = build(Extra)
    assert len(record) == 1
    assert "V_anniv" not in compiled.source
    assert "V_bel" in compiled.source


def test_the_abstract_class_is_refused_and_not_inherited() -> None:
    with pytest.raises(BuildError, match="EscalatingAnnuity is abstract"):
        build(EscalatingAnnuity)
    assert vars(EscalatingAnnuity)["abstract"] is True
    assert "abstract" not in vars(FixedEscalationAnnuity)
    assert "abstract" not in vars(CPILinkedAnnuity)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert build(CPILinkedAnnuity, cpi_lag_m=4).graph.statics == {
            "include_inv_exp": True,
            "cpi_lag_m": 4,
        }
    with pytest.raises(BuildError, match="has no build-time parameter 'cpi_lag_m'"):
        build(LevelAnnuity, cpi_lag_m=4)


# ------------------------------------------------------------------------ data contracts


def _row(**changes: Any) -> dict[str, Any]:
    return {**data.example_policy(1), **changes}


def _load(
    compiled: CompiledModel,
    tmp_path: Path,
    *,
    rows: list[dict[str, Any]] | None = None,
    tables: dict[str, Any] | None = None,
    scenarios: dict[str, Any] | None = None,
    valuation_date: Any = VALUATION_DATE,
    horizon: int | None = None,
) -> Any:
    return run(
        compiled,
        model_points=data.model_points_table(rows or [_row()]),
        tables={**data.reference_tables(), **(tables or {})},
        scenarios={**data.reference_scenarios(), **(scenarios or {})},
        settings=RunSettings(valuation_date=valuation_date, horizon=horizon),
        run_dir=tmp_path / "run",
    )


def test_the_valuation_date_must_be_a_month_end(
    compiled_annuity: dict[str, CompiledModel], tmp_path: Path
) -> None:
    with pytest.raises(
        LoadError, match="valuation date 2025-12-30: must be the last day of a calendar month"
    ):
        _load(compiled_annuity["level"], tmp_path, valuation_date=dt.date(2025, 12, 30))


def test_the_annuity_starts_on_the_first_of_a_month(
    compiled_annuity: dict[str, CompiledModel], tmp_path: Path
) -> None:
    row = _row(
        policy_id=7,
        issue_date=dt.date(2025, 11, 30),
        annuity_start_date=dt.date(2025, 12, 2),
    )
    with pytest.raises(
        LoadError,
        match=r"model points: rule day\(annuity_start_date\) = 1 fails for policies 7 \(payments",
    ):
        _load(compiled_annuity["level"], tmp_path, rows=[row])


def test_the_cpi_file_must_reach_back_fourteen_months(
    compiled_annuity: dict[str, CompiledModel], tmp_path: Path
) -> None:
    with pytest.raises(
        LoadError,
        match=r"scenario input cpi: the model reads months -14 to 655 but the file covers -13 to",
    ):
        _load(compiled_annuity["level"], tmp_path, scenarios={"cpi": data.cpi_table(u_lo=-13)})


def test_mort_cif_must_cover_the_last_calendar_year_read(tmp_path: Path) -> None:
    """A table contract ending before Y(T) = 2080 fails the input rule at load, before any
    lookup could run off the table (the run-time key check stays as the backstop)."""
    narrow = Table(
        "mort_cif",
        keys=(Key("age", lo=20, hi=120), Key("year", lo=2025, hi=2060)),
        columns=(Column("cif", "Factor", le=3),),
        metadata=MORT_CIF.metadata,
        rules=MORT_CIF.rules,
    )

    class Narrow(LevelAnnuity):
        assumption_tables = (MORT_BASE, narrow, EXPENSE, INV_EXPENSE, ILP, RA_PARAM)

    full = data.mort_cif_table()
    short = full.filter(pc.field("year") <= 2060).replace_schema_metadata(full.schema.metadata)
    with pytest.raises(
        LoadError,
        match=r"input rule the calendar years read are inside mort_cif fails \(\(m_val \+ 1\) // 12"
        r" and \(m_val \+ H\) // 12 within",
    ):
        _load(build(Narrow), tmp_path, tables={"mort_cif": short})


def test_mort_cif_must_cover_the_first_calendar_year_read(
    compiled_annuity: dict[str, CompiledModel], tmp_path: Path
) -> None:
    """The other end of the spec's key check: a valuation date in 2024 reads year 2024 in month
    1, before the 2025-based table starts. Every other contract is met (the policy issued at V,
    the expense table in June 2024 prices, the files long enough for the older life)."""
    row = _row(issue_date=dt.date(2024, 6, 30), cohort=2024, annuity_start_date=dt.date(2024, 7, 1))
    with pytest.raises(
        LoadError, match=r"input rule the calendar years read are inside mort_cif fails"
    ):
        _load(
            compiled_annuity["level"],
            tmp_path,
            rows=[row],
            tables={"expense": data.expense_table(price_base="2024-06")},
            scenarios={"rf_df": data.discount_table(t_max=673), "cpi": data.cpi_table(u_hi=673)},
            valuation_date=dt.date(2024, 6, 30),
        )


def test_the_horizon_must_cover_every_run_off_month(
    compiled_annuity: dict[str, CompiledModel], tmp_path: Path
) -> None:
    """A horizon given in the settings that stops before a policy's t_end fails at load, naming
    the rule and the first failing policy (D11), not inside an invariant after a truncated
    valuation: policy 1 runs off at 655, policy 3 at 591, and H = 600 covers only the latter."""
    rows = [_row(), data.example_policy(3)]
    with pytest.raises(
        LoadError,
        match=r"input rule the horizon covers every policy's run-off month fails for 1 of 2 "
        r"values, first for policy 1 \(a run shorter than t_end truncates the valuation\)",
    ):
        _load(compiled_annuity["level"], tmp_path, rows=rows, horizon=600)
    # one policy, one scenario: the (1,) result still names the policy, not an element position
    with pytest.raises(
        LoadError,
        match=r"input rule the horizon covers every policy's run-off month fails for policy 1 "
        r"\(a run shorter",
    ):
        _load(compiled_annuity["level"], tmp_path, horizon=300)
    assert _load(compiled_annuity["level"], tmp_path, rows=rows, horizon=655).H == 655


def _cif_table_with(age: int, year: int, value: float) -> Any:
    """The reference ``mort_cif`` with one cell changed, its metadata kept."""
    full = data.mort_cif_table()
    cols = full.to_pydict()
    i = next(
        k
        for k, (x, y) in enumerate(zip(cols["age"], cols["year"], strict=True))
        if (x, y) == (age, year)
    )
    cols["cif"][i] = value
    return pa.table(
        {
            "age": pa.array(cols["age"], pa.int16()),
            "year": pa.array(cols["year"], pa.int16()),
            "cif": pa.array(cols["cif"], pa.float64()),
        }
    ).replace_schema_metadata(full.schema.metadata)


def test_cif_at_the_limiting_age_is_one(
    compiled_annuity: dict[str, CompiledModel], tmp_path: Path
) -> None:
    with pytest.raises(
        LoadError,
        match=r"table mort_cif: rule cif\(120, y\) = 1 fails at year = 2030 \(the limiting",
    ):
        _load(
            compiled_annuity["level"],
            tmp_path,
            tables={"mort_cif": _cif_table_with(120, 2030, 0.9)},
        )


def test_q_base_times_cif_must_stay_a_probability(
    compiled_annuity: dict[str, CompiledModel], tmp_path: Path
) -> None:
    """The cross-table check names the ``mort_cif`` row (spec, "Assumption tables"): cif(119,
    2030) = 1.1 passes every per-table rule (below 3, above 0, neither the base year nor age 120)
    and fails the grid check through the male q at 119, so the failure is reported at that age
    and year of ``mort_cif``, one cell of its 101 x 126 grid."""
    with pytest.raises(
        LoadError,
        match=r"input rule q_base \* cif <= 1 on the key grid fails for 1 of 12726 values, first "
        r"at age = 119, year = 2030 of table mort_cif \(q_ann stays a probability\)",
    ):
        _load(
            compiled_annuity["level"],
            tmp_path,
            tables={"mort_cif": _cif_table_with(119, 2030, 1.1)},
        )


def test_the_money_scale_is_floored_at_a_year_of_payments(
    compiled_annuity: dict[str, CompiledModel], tmp_path: Path
) -> None:
    """An in-force policy may carry no premium (``single_premium = 0`` is allowed before V); its
    Money tolerance scale is then a year of payments, never 0, while a policy with a premium
    keeps the premium as its scale."""
    in_force = _row(
        policy_id=9,
        issue_date=dt.date(2015, 12, 31),
        cohort=2015,
        annuity_start_date=dt.date(2016, 1, 1),
        single_premium=0.0,
    )
    res = _load(compiled_annuity["level"], tmp_path, rows=[_row(), in_force])
    assert res.inputs is not None
    scale = res.inputs.model_points.columns["scale_money"]
    assert list(scale) == [190_000.0, 1_000.0 * 12]  # policy 1 its premium, policy 9 the floor


def test_sigma_e_must_not_exceed_z_alpha(
    compiled_annuity: dict[str, CompiledModel], tmp_path: Path
) -> None:
    params = data.ra_param_table(alpha=0.6, z_alpha=NormalDist().inv_cdf(0.6), sigma_e=0.3)
    with pytest.raises(LoadError, match=r"input rule sigma_e <= z_alpha fails \(keeps the expense"):
        _load(compiled_annuity["level"], tmp_path, tables={"ra_param": params})


def test_mort_base_must_cover_every_sex(
    compiled_annuity: dict[str, CompiledModel], tmp_path: Path
) -> None:
    """The enum-coverage rule for the annuity is the load-time key coverage of mort_base."""
    full = data.mort_base_table()
    males = full.filter(pc.field("sex") == "M").replace_schema_metadata(full.schema.metadata)
    with pytest.raises(
        LoadError,
        match=r"table mort_base: the rows must cover every key combination exactly once; "
        r"declared grid has 202 rows, the file 101 rows, 101 combination\(s\) missing",
    ):
        _load(compiled_annuity["level"], tmp_path, tables={"mort_base": males})


def test_the_other_table_rules_and_metadata(
    compiled_annuity: dict[str, CompiledModel], tmp_path: Path
) -> None:
    compiled = compiled_annuity["level"]
    base = data.mort_base_table()
    cols = base.to_pydict()
    cols["q"][cols["age"].index(120)] = 0.5
    open_table = pa.table(
        {
            "sex": pa.array(cols["sex"], pa.string()),
            "age": pa.array(cols["age"], pa.int16()),
            "q": pa.array(cols["q"], pa.float64()),
        }
    ).replace_schema_metadata(base.schema.metadata)
    with pytest.raises(LoadError, match=r"table mort_base: rule q\(\., 120\) = 1 fails at sex = F"):
        _load(compiled, tmp_path, tables={"mort_base": open_table})
    no_meta = base.replace_schema_metadata({b"halley.version": b"x"})
    with pytest.raises(
        LoadError, match=r"table mort_base: the file carries no schema metadata halley\.base_year"
    ):
        _load(compiled, tmp_path, tables={"mort_base": no_meta})
    with pytest.raises(
        LoadError, match="input rule expense price_base is the valuation month fails"
    ):
        _load(compiled, tmp_path, tables={"expense": data.expense_table(price_base="2025-11")})
    with pytest.raises(
        LoadError, match=r"input rule z_alpha = inv_cdf\(alpha\) fails \(within 1e-12\)"
    ):
        _load(compiled, tmp_path, tables={"ra_param": data.ra_param_table(z_alpha=0.67)})
    with pytest.raises(LoadError, match=r"input rule base_year agrees across mort_base, mort_cif"):
        _load(
            compiled,
            tmp_path,
            tables={"mort_base": base.replace_schema_metadata({b"halley.base_year": b"2024"})},
        )
    with pytest.raises(
        LoadError, match=r"scenario input rf_df: rule rf_df\(s, 0\) = 1 fails for scenario 0"
    ):
        _load(
            compiled,
            tmp_path,
            scenarios={
                "rf_df": data.discount_table(lambda s, t: 0.999 if t == 0 else data.rf_df(t))
            },
        )
    with pytest.raises(
        LoadError,
        match="rule rows with u <= 0 identical across scenarios fails for scenario 1, month -24",
    ):
        _load(
            compiled,
            tmp_path,
            scenarios={
                "rf_df": data.discount_table(n_scenarios=2),
                "cpi": data.cpi_table(lambda s, u: data.cpi_path(u) + s, n_scenarios=2),
            },
        )
    with pytest.raises(LoadError, match=r"rule 0\.5 < alpha < 1 fails"):
        _load(
            compiled,
            tmp_path,
            tables={"ra_param": data.ra_param_table(alpha=0.5, z_alpha=0.0, sigma_e=0.0)},
        )


def test_the_model_point_rules(compiled_annuity: dict[str, CompiledModel], tmp_path: Path) -> None:
    compiled = compiled_annuity["level"]
    with pytest.raises(LoadError, match=r"rule portfolio non-empty fails for policies 1 \(group"):
        _load(compiled, tmp_path, rows=[_row(portfolio="")])
    with pytest.raises(
        LoadError, match=r"rule cohort = year\(issue_date\) fails for policies 1 \(para 22\)"
    ):
        _load(compiled, tmp_path, rows=[_row(cohort=2024)])
    with pytest.raises(LoadError, match="rule issue_date <= V fails for policies 1"):
        _load(compiled, tmp_path, rows=[_row(issue_date=dt.date(2026, 1, 1), cohort=2026)])
    with pytest.raises(LoadError, match="rule issue_date < annuity_start_date fails"):
        _load(compiled, tmp_path, rows=[_row(annuity_start_date=dt.date(2025, 12, 1))])
    with pytest.raises(LoadError, match=r"rule annuity_start_date <= V \+ 1 day fails"):
        _load(compiled, tmp_path, rows=[_row(annuity_start_date=dt.date(2026, 2, 1))])
    with pytest.raises(LoadError, match="rule single_premium > 0 where issue_date = V fails"):
        _load(compiled, tmp_path, rows=[_row(single_premium=0.0)])
    with pytest.raises(LoadError, match="field a0 must be >= 600"):
        _load(compiled, tmp_path, rows=[_row(date_of_birth=dt.date(1980, 1, 1))])
    with pytest.raises(
        LoadError, match="field profit_bucket takes exactly ONEROUS, NO_SIGNIFICANT_RISK, OTHER"
    ):
        _load(compiled, tmp_path, rows=[_row(profit_bucket="PROFITABLE")])
    with pytest.raises(LoadError, match="field policy_id is not unique: 1"):
        _load(compiled, tmp_path, rows=[_row(), _row(profit_bucket="ONEROUS")])
    with pytest.raises(LoadError, match="field esc_rate is missing"):
        run(
            compiled_annuity["fixed"],
            model_points=data.model_points_table([_row()]),
            tables=data.reference_tables(),
            scenarios=data.reference_scenarios(),
            settings=RunSettings(valuation_date=VALUATION_DATE),
            run_dir=tmp_path / "fixed",
        )


# --------------------------------------------------------------- invariants are live checks


def test_a_wrong_cpi_lag_fails_i9_at_the_first_anniversary(tmp_path: Path) -> None:
    """I9 rebuilds ann_pp from the CPI path at the declared lag; a model escalating at lag 2
    while declaring 3 passes the build and fails the run at policy 3's first anniversary, the
    error located at the invariant's declaration in the product module."""

    class WrongLag(CPILinkedAnnuity):
        @var
        def esc_factor(self, t: Int) -> Factor:
            return where(self.anniv(t), self.scen.cpi(t - 2) / self.scen.cpi(t - 14), 1.0)

    with pytest.raises(
        RunError,
        match=r"invariant I9 ann_pp telescopes along the CPI path failed: policy 3, t = 13",
    ) as info:
        _load(build(WrongLag), tmp_path, rows=[data.example_policy(3)])
    assert info.value.loc is not None
    assert Path(info.value.loc.file) == Path(inspect.getsourcefile(LevelAnnuity) or "")


def test_coverage_units_other_than_the_payment_fail_i10(tmp_path: Path) -> None:
    class ExpenseUnits(LevelAnnuity):
        @var
        def cu(self, t: Int) -> CoverageUnits:
            return as_cu(self.exp_out(t))

    with pytest.raises(
        RunError, match=r"invariant I10 cu = ann_out failed: policy 1, t = 1"
    ) as info:
        _load(build(ExpenseUnits), tmp_path)
    assert info.value.loc is not None
    assert Path(info.value.loc.file) == Path(inspect.getsourcefile(LevelAnnuity) or "")


def test_the_level_class_uses_every_variable_and_the_cpi_class_needs_cpi_lag(
    compiled_annuity: dict[str, CompiledModel],
) -> None:
    level = compiled_annuity["level"].graph
    assert set(level.variables) == set(compiled_annuity["level"].schedule.needed)
    cpi = compiled_annuity["cpi"].graph
    assert "cpi_lag" in cpi.invariant_reads
    assert cpi.variables["cpi_lag"].expr.unit is not None
    assert ["bel", "inv_exp"] in level.components
    assert level.variables["ra"].start == 0
    assert level.variables["df"].start == 0  # demanded at t = 0 by v(t) and pva_t(t)
    assert level.variables["cu"].start == 1
    assert level.variables["q_ann"].unit is not None
    assert str(level.variables["cu"].unit) == "CoverageUnits"
    assert str(level.tables["ilp"].columns[0].unit) == "Rate[Annual, Continuous]"
    assert str(level.scenarios["cpi"].unit) == "Index"
    assert level.variables["onerous"].dims == {"policy", "scenario"}
    assert level.variables["exp_pp"].dims == {"scenario", "time"}
    assert level.variables["ie_m"].dims == frozenset()
