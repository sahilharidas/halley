"""Single-life immediate annuity in payment: the annuity reference case (D2, M5).

The model of ``docs/spec/annuity.md`` ("Model in Halley's authoring syntax"), written with the
engine's spellings (design section 8): a scalar is read as ``self.x()``, a fixed month as
``at(x, 0)``, the run's valuation month as ``self.run.m_val``, and the level class is extended
by the escalating variants as subclasses that override the one hook ``esc_factor`` (D9). Also
here: the model point schema, the assumption table contracts with their metadata and rules, the
scenario inputs, the input rules across them (section 6 of the design) and the invariants I3 to
I10 of the spec, checked on every run. The reference basis, the discount and CPI files and the
four example model points are built by :mod:`halley.products.annuity_data`.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import replace
from statistics import NormalDist
from typing import Any, ClassVar, cast

import numpy as np
from numpy.typing import NDArray

from halley.authoring import (
    Annual,
    BatchView,
    Bool,
    Column,
    Count,
    CoverageUnits,
    Derived,
    Factor,
    Field,
    InputRule,
    InputView,
    Int,
    Invariant,
    Key,
    LoadContext,
    Money,
    MoneyPerPolicy,
    Monthly,
    Prob,
    Product,
    Rate,
    Rule,
    ScenarioInput,
    ScenarioRule,
    ScenarioView,
    Schema,
    Static,
    Table,
    TableRule,
    TableView,
    ann_fwd,
    ann_spot,
    annual_step,
    as_cu,
    at,
    cc_discount,
    exp,
    minimum,
    pv_month,
    sum_t,
    to_monthly,
    to_monthly_cf,
    var,
    where,
)
from halley.authoring.dates import day, is_month_end, month_index, year

OMEGA = 120
"""The limiting age: q = 1 at OMEGA, so every life in force dies in its first month there."""

T_OMEGA = 12 * OMEGA + 1
"""``t_end = T_OMEGA - a0`` is a policy's first month at the limiting age."""

VALUATION_DATE = dt.date(2025, 12, 31)
"""The worked example's valuation date (spec, "Run settings")."""

BASIS_VERSION = "annuity-ref-basis-0.1"

EPS = 1e-12
"""The spec's tolerance for its invariants within a backend."""

# ------------------------------------------------------------------- model point schema


def _m_val(c: LoadContext) -> int:
    return 12 * c.valuation_date.year + c.valuation_date.month - 1


def _a0(c: LoadContext) -> NDArray[Any]:
    """Age at V in completed months, the day of birth ignored (spec, "Ages")."""
    return _m_val(c) - month_index(c.date_of_birth)


def _m_c(c: LoadContext) -> NDArray[Any]:
    """The month of the first payment relative to V; at most 1."""
    return month_index(c.annuity_start_date) - _m_val(c)


def _t_end(c: LoadContext) -> NDArray[Any]:
    return T_OMEGA - c.a0


def _scale_money(c: LoadContext) -> NDArray[Any]:
    return c.pols_if_init * c.single_premium


def _month_end(d: dt.date) -> str | None:
    return None if is_month_end(d) else "must be the last day of a calendar month (annuity spec)"


SCHEMA = Schema(
    fields=(
        Field("policy_id", "int", "Int", unique=True, ge=1, doc="Unique across the classes"),
        Field("portfolio", "string", doc="Group key"),
        Field("cohort", "int", "Int", ge=1900, le=2200, doc="Group key: the year of issue"),
        Field(
            "profit_bucket",
            "enum",
            enum=("ONEROUS", "NO_SIGNIFICANT_RISK", "OTHER"),
            doc="Group key (IFRS 17 para 16)",
        ),
        Field("sex", "enum", enum=("F", "M"), doc="Mortality key"),
        Field("date_of_birth", "date"),
        Field("issue_date", "date", doc="Initial recognition date (para 25)"),
        Field("annuity_start_date", "date", doc="First payment date; anniversaries run from it"),
        Field("pols_if_init", "number", "Count", gt=0, le=1_000_000),
        Field(
            "instalment_pm",
            "number",
            "MoneyPerPolicy",
            gt=0,
            le=1_000_000,
            doc="Monthly payment in force at V, after every escalation up to V",
        ),
        Field(
            "single_premium",
            "number",
            "MoneyPerPolicy",
            ge=0,
            doc="Received on the issue date; read only at initial recognition",
        ),
    ),
    derived=(
        Derived("a0", "Int", _a0, ge=600, le=1320, doc="age at V in months: 50 to 110 years"),
        Derived("m_c", "Int", _m_c, doc="at most 1: the annuity is in payment by month 1"),
        Derived("t_end", "Int", _t_end, doc="the first month at the limiting age"),
        Derived("scale_money", "Money", _scale_money, kind="number", ge=0),
    ),
    rules=(
        Rule("portfolio non-empty", lambda c: c.portfolio != "", "group key"),
        Rule("cohort = year(issue_date)", lambda c: c.cohort == year(c.issue_date), "para 22"),
        Rule("issue_date <= V", lambda c: c.issue_date <= c.valuation),
        Rule("issue_date < annuity_start_date", lambda c: c.issue_date < c.annuity_start_date),
        Rule(
            "day(annuity_start_date) = 1",
            lambda c: day(c.annuity_start_date) == 1,
            "payments fall on the 1st of a month",
        ),
        Rule(
            "annuity_start_date <= V + 1 day",
            lambda c: c.annuity_start_date <= c.valuation + np.timedelta64(1, "D"),
        ),
        Rule(
            "single_premium > 0 where issue_date = V",
            lambda c: (c.single_premium > 0) | (c.issue_date != c.valuation),
        ),
    ),
    horizon="t_end",
    valuation_date_rule=_month_end,
)

FIXED_SCHEMA = replace(
    SCHEMA,
    fields=(
        *SCHEMA.fields,
        Field(
            "esc_rate",
            "number",
            "Rate[Annual]",
            ge=0,
            le=0.10,
            doc="Compound escalation, once a year on the anniversary",
        ),
    ),
)

# --------------------------------------------------------- assumption table contracts (6)

MORT_BASE = Table(
    "mort_base",
    keys=(Key("sex", enum=("F", "M")), Key("age", lo=20, hi=OMEGA)),
    columns=(Column("q", "Prob[Annual]"),),
    doc="Annual mortality by sex and age last birthday, base year rates",
    metadata=("base_year",),
    rules=(
        TableRule(
            "q(., 120) = 1",
            lambda v: v.at("q", age=OMEGA) == 1.0,
            "the table closes at the limiting age",
        ),
    ),
)


def _cif_at_base_year(v: TableView) -> NDArray[Any]:
    return v.at("cif", year=int(v.metadata["base_year"])) == 1.0


MORT_CIF = Table(
    "mort_cif",
    keys=(Key("age", lo=20, hi=OMEGA), Key("year", lo=2025, hi=2150)),
    columns=(Column("cif", "Factor", le=3),),
    doc="Cumulative improvement factor by age and calendar year, from mort_improvement",
    metadata=("base_year", "mort_improvement_sha256"),
    rules=(
        TableRule("cif > 0", lambda v: v.cif > 0),
        TableRule("cif(., base_year) = 1", _cif_at_base_year, "no improvement in the base year"),
        TableRule(
            "cif(120, y) = 1",
            lambda v: v.at("cif", age=OMEGA) == 1.0,
            "the limiting age never improves",
        ),
    ),
)
EXPENSE = Table(
    "expense",
    keys=(),
    columns=(Column("maint_pm", "MoneyPerPolicy", ge=0, le=10_000),),
    doc="Maintenance expense per policy per month, in valuation-month prices",
    metadata=("price_base",),
)
INV_EXPENSE = Table(
    "inv_expense",
    keys=(),
    columns=(Column("ie_rate", "Rate[Annual]", ge=0, le=0.01),),
    doc="Investment expense, an annual effective charge on the closing BEL",
)
ILP = Table(
    "ilp",
    keys=(),
    columns=(Column("ilp", "Rate[Annual, Continuous]", ge=0, le=0.02),),
    doc="Illiquidity premium over the risk-free curve, continuously compounded",
)
RA_PARAM = Table(
    "ra_param",
    keys=(),
    columns=(
        Column("alpha", "Prob"),
        Column("z_alpha", "Factor"),
        Column("sigma_l", "Factor", ge=0, le=0.5),
        Column("sigma_e", "Factor", ge=0, le=0.5),
    ),
    doc="Risk adjustment: confidence level, its normal quantile, longevity and expense sigmas",
    rules=(TableRule("0.5 < alpha < 1", lambda v: (v.alpha > 0.5) & (v.alpha < 1)),),
)

# ------------------------------------------------------------------ scenario inputs (D4)

RF_DF = ScenarioInput(
    "rf_df",
    "Factor",
    doc="Liquid risk-free discount factor from month t to time 0, per scenario",
    gt=0,
    le=2,
    covers=(0, "H"),
    rules=(
        ScenarioRule("rf_df(s, 0) = 1", lambda v: v.at(0) == 1.0, "a discount factor starts at 1"),
    ),
)


def _history_identical(v: ScenarioView) -> NDArray[Any]:
    return np.where(v.months[:, None] <= 0, v.values == v.values[:, :1], True)


CPI = ScenarioInput(
    "cpi",
    "Index",
    doc="CPI level for calendar month m_val + u; rows with u <= 0 are history",
    gt=0,
    le=1_000_000,
    covers=(-14, "H"),
    rules=(
        ScenarioRule(
            "rows with u <= 0 identical across scenarios",
            _history_identical,
            "history is a fact, the same in every scenario",
        ),
    ),
)

# ------------------------------------------------------------------------ input rules (6)


def _base_years_agree(v: InputView) -> bool:
    base = int(v.tables["mort_base"].metadata["base_year"])
    cif = int(v.tables["mort_cif"].metadata["base_year"])
    return base == cif == v.tables["mort_cif"].key_range("year")[0]


def _z_alpha_is_the_quantile(v: InputView) -> bool:
    ra = v.tables["ra_param"]
    return abs(float(ra.z_alpha) - NormalDist().inv_cdf(float(ra.alpha))) <= 1e-12


def _price_base_is_the_valuation_month(v: InputView) -> bool:
    d = v.valuation_date
    return v.tables["expense"].metadata["price_base"] == f"{d.year:04d}-{d.month:02d}"


def _years_read_inside_mort_cif(v: InputView) -> bool:
    """The spec's load-time key check: ``cal_year(t)`` is monotone in t, so the first month and
    the last month read, ``(m_val + 1) // 12`` and ``(m_val + H) // 12``, bound every lookup."""
    lo, hi = v.tables["mort_cif"].key_range("year")
    first, last = (v.valuation_month + 1) // 12, (v.valuation_month + v.horizon) // 12
    return lo <= first and last <= hi


def _horizon_covers_every_run_off(v: InputView) -> NDArray[Any]:
    """A horizon given in the run settings must reach every policy's ``t_end``: a shorter run
    takes ``bel(H) = 0`` as terminal while lives are still in force (D11)."""
    return v.horizon >= v.model_points.t_end


INPUT_RULES = (
    InputRule(
        "q_base * cif <= 1 on the key grid",
        lambda v: v.tables["mort_base"].q[:, :, None] * v.tables["mort_cif"].cif[None, :, :] <= 1,
        "q_ann stays a probability",
    ),
    InputRule(
        "base_year agrees across mort_base, mort_cif and the year key",
        _base_years_agree,
        "the improvement factors start from the base table's year",
    ),
    InputRule("z_alpha = inv_cdf(alpha)", _z_alpha_is_the_quantile, "within 1e-12"),
    InputRule(
        "sigma_e <= z_alpha",
        lambda v: v.tables["ra_param"].sigma_e <= v.tables["ra_param"].z_alpha,
        "keeps the expense multiplier at or above 1, so the RA is never negative (I7)",
    ),
    InputRule(
        "expense price_base is the valuation month",
        _price_base_is_the_valuation_month,
        "maint_pm is in the prices of month u = 0",
    ),
    InputRule(
        "the calendar years read are inside mort_cif",
        _years_read_inside_mort_cif,
        "(m_val + 1) // 12 and (m_val + H) // 12 within the year key's range",
    ),
    InputRule(
        "the horizon covers every policy's run-off month",
        _horizon_covers_every_run_off,
        "a run shorter than t_end truncates the valuation",
    ),
)

# ------------------------------------------------------------------------- invariants (I3..I10)


def _rel(a: BatchView, lhs: NDArray[Any], rhs: NDArray[Any]) -> NDArray[Any]:
    return np.maximum(np.abs(lhs), np.abs(rhs))


def _abs_rhs(a: BatchView, lhs: NDArray[Any], rhs: NDArray[Any]) -> NDArray[Any]:
    return np.abs(rhs)


def _i3_terminals(a: BatchView) -> NDArray[Any]:
    """|bel(H)| + |v_be(H)| + |v_st(H)|: zero exactly when every terminal is reached."""
    return np.abs(a.at("bel", a.H)) + np.abs(a.at("v_be", a.H)) + np.abs(a.at("v_st", a.H))


def _i4_rhs(a: BatchView) -> NDArray[Any]:
    """[bel(t-1) - ann_out(t) - exp_out(t)] / v(t) - inv_exp(t), months 1..H."""
    bel, ann_out, exp_out = a.var("bel"), a.var("ann_out"), a.var("exp_out")
    return ((bel[:-1] - ann_out[1:]) - exp_out[1:]) / a.var("v")[1:] - a.var("inv_exp")[1:]


def _i4_scale(a: BatchView, lhs: NDArray[Any], rhs: NDArray[Any]) -> NDArray[Any]:
    """The size of the terms combined: (|bel(t-1)| + |ann_out(t)| + |exp_out(t)|) / v(t) +
    |inv_exp(t)|, so the check holds at a policy's run-off month, where bel is exactly 0."""
    bel, ann_out, exp_out = a.var("bel"), a.var("ann_out"), a.var("exp_out")
    size = (np.abs(bel[:-1]) + np.abs(ann_out[1:]) + np.abs(exp_out[1:])) / a.var("v")[1:]
    return size + np.abs(a.var("inv_exp")[1:])


def _i6_scale(a: BatchView, lhs: NDArray[Any], rhs: NDArray[Any]) -> NDArray[Any]:
    return np.maximum(_rel(a, lhs, rhs), np.abs(a.at("bel", 0)))


def _no_stress(a: BatchView) -> NDArray[Any]:
    return (a.table("ra_param", "sigma_l") == 0) & (a.table("ra_param", "sigma_e") == 0)


def _i8_fixed_rhs(a: BatchView) -> NDArray[Any]:
    """instalment_pm (1 + g)^K(t), K(t) the anniversaries in months 1..t."""
    k = np.cumsum(a.var("anniv").astype(np.int64), axis=0)
    return a.mp.instalment_pm * (1 + a.mp.esc_rate) ** k


def _i9_rhs(a: BatchView) -> NDArray[Any]:
    """instalment_pm I(n_K - L) / I(n_0 - L), months 1..H: n_K the latest anniversary month in
    1..t (or n_0 when there is none), n_0 = m_c when m_c = 1, else the latest m_c + 12k <= 0."""
    lag = int(a.var("cpi_lag")[0, 0, 0])
    m_c = a.mp.m_c
    t = a.t[1:]
    n_0 = np.where(m_c == 1, 1, m_c + 12 * ((-m_c) // 12))
    n_k = m_c + 12 * ((t - m_c) // 12)
    lo, hi = int(n_0.min()) - lag, int(n_k.max()) - lag
    cpi = a.scen_range("cpi", lo, hi)[:, 0, :]
    numer = cpi[(n_k - lag - lo)[:, :, 0]]
    denom = cpi[(n_0 - lag - lo)[0, :, 0]][None]
    return a.mp.instalment_pm * (numer / denom)


_COMMON_INVARIANTS = (
    Invariant(
        "I3 q_ann(t_end) = 1",
        lambda a: a.at("q_ann", a.mp.t_end),
        1.0,
        tol=0,
        reads=("q_ann",),
        doc="the model closes the table at the limiting age",
    ),
    Invariant(
        "I3 pols_if(t_end) = 0",
        lambda a: a.at("pols_if", a.mp.t_end),
        0.0,
        tol=0,
        doc="every life in force dies in its first month at the limiting age",
    ),
    Invariant(
        "I3 bel(H) = v_be(H) = v_st(H) = 0",
        _i3_terminals,
        0.0,
        tol=0,
        reads=("bel", "v_be", "v_st"),
    ),
    Invariant(
        "I4 roll-forward of bel",
        lambda a: a.var("bel")[1:],
        _i4_rhs,
        tol=EPS,
        scale=_i4_scale,
        months=(1, "H"),
        reads=("bel", "ann_out", "exp_out", "v", "inv_exp"),
        doc="bel(t) = [bel(t-1) - ann_out(t) - exp_out(t)] / v(t) - inv_exp(t)",
    ),
    Invariant(
        "I5 bel(0) = pva + pve",
        lambda a: a.at("bel", 0),
        lambda a: a.var("pva") + a.var("pve"),
        tol=EPS,
        scale=_rel,
        reads=("bel", "pva", "pve"),
        doc="the direct present value of the fulfilment cash flows",
    ),
    Invariant(
        "I6 bel(t) = pols_if(t) v_be(t)",
        lambda a: a.var("bel"),
        lambda a: a.var("pols_if") * a.var("v_be"),
        tol=EPS,
        scale=_i6_scale,
        reads=("bel", "pols_if", "v_be"),
        doc="the per-life recursion; floored at bel(0) where both sides are tiny",
    ),
    Invariant("I7 ra >= 0", lambda a: a.var("ra"), 0.0, kind="ge", tol=0, reads=("ra",)),
    Invariant(
        "I7 ra = 0 without stress",
        lambda a: a.var("ra"),
        0.0,
        tol=0,
        where=_no_stress,
        reads=("ra",),
        doc="sigma_l = sigma_e = 0 gives the best-estimate basis on both recursions",
    ),
    Invariant(
        "I10 cu = ann_out",
        lambda a: a.var("cu"),
        lambda a: a.var("ann_out"),
        tol=0,
        reads=("cu", "ann_out"),
    ),
    Invariant("I10 cu >= 0", lambda a: a.var("cu"), 0.0, kind="ge", tol=0, reads=("cu",)),
)

_I8_LEVEL = Invariant(
    "I8 ann_pp = instalment_pm",
    lambda a: a.var("ann_pp"),
    lambda a: a.mp.instalment_pm,
    tol=0,
    reads=("ann_pp",),
    doc="a level annuity never escalates",
)
_I8_FIXED = Invariant(
    "I8 ann_pp = instalment_pm (1 + esc_rate)^K",
    lambda a: a.var("ann_pp"),
    _i8_fixed_rhs,
    tol=EPS,
    scale=_abs_rhs,
    reads=("ann_pp", "anniv"),
    doc="K(t) counts the anniversaries in months 1..t",
)
_I9_CPI = Invariant(
    "I9 ann_pp telescopes along the CPI path",
    lambda a: a.var("ann_pp")[1:],
    _i9_rhs,
    tol=EPS,
    scale=_abs_rhs,
    months=(1, "H"),
    reads=("ann_pp", "cpi_lag"),
    doc="ann_pp(t) = instalment_pm I(n_K - L) / I(n_0 - L)",
)

# ----------------------------------------------------------------------------- the model


class LevelAnnuity(Product):
    """Single-life annuity in payment, level, paid monthly in advance (D2)."""

    include_inv_exp: Static[bool] = True  # B65(ka)(i) judgement

    model_point = SCHEMA
    assumption_tables = (MORT_BASE, MORT_CIF, EXPENSE, INV_EXPENSE, ILP, RA_PARAM)
    scenario_inputs = (RF_DF, CPI)
    input_rules = INPUT_RULES
    outputs = (
        "pols_if",
        "deaths",
        "ann_pp",
        "ann_out",
        "exp_out",
        "inv_exp",
        "bel",
        "ra",
        "cu",
        "pva",
        "pve",
        "onerous",
        "premium_total",
        "run_off_month",
        "df",
        "v",
        "fwd_ann",
        "spot_ann",
    )
    tolerance_scales: ClassVar[dict[str, str]] = {
        "Money": "scale_money",
        "MoneyPerPolicy": "instalment_pm",
        "Count": "pols_if_init",
    }
    invariants = (*_COMMON_INVARIANTS, _I8_LEVEL)

    # ---- time-derived lookup keys (D10) ----------------------------------------------
    @var
    def age(self, t: Int) -> Int:
        """Age last birthday at the start of month t, clamped at the limiting age."""
        return minimum((self.mp.a0 + t - 1) // 12, OMEGA)

    @var
    def cal_year(self, t: Int) -> Int:
        """Calendar year of month t."""
        return (self.run.m_val + t) // 12

    # ---- mortality ------------------------------------------------------------------
    @var
    def q_ann(self, t: Int) -> Prob[Annual]:
        """Closed at OMEGA whatever the tables hold."""
        q: Prob[Annual] = (
            self.tables.mort_base[self.mp.sex, self.age(t)]
            * self.tables.mort_cif[self.age(t), self.cal_year(t)]
        )
        return where(self.age(t) >= OMEGA, 1.0, q)

    @var
    def q_mth(self, t: Int) -> Prob[Monthly]:
        return to_monthly_cf(self.q_ann(t))  # guarded 1 - (1 - q)^(1/12)

    @var(role="decrement")
    def deaths(self, t: Int) -> Count:
        return self.pols_if(t - 1) * self.q_mth(t)  # Count x Prob[Monthly] -> Count

    @var(init=lambda mp: mp.pols_if_init, role="in_force")
    def pols_if(self, t: Int) -> Count:
        return self.pols_if(t - 1) - self.deaths(t)

    # ---- annuity amount -------------------------------------------------------------
    @var
    def esc_factor(self, t: Int) -> Factor:
        """Hook: the variants override this only."""
        return cast(Factor, 1.0)

    @var(timing="start", init=lambda mp: mp.instalment_pm)
    def ann_pp(self, t: Int) -> MoneyPerPolicy:
        return self.ann_pp(t - 1) * self.esc_factor(t)

    # ---- expenses -------------------------------------------------------------------
    @var(timing="start")
    def exp_pp(self, t: Int) -> MoneyPerPolicy:
        """Valuation-month prices, CPI-inflated."""
        return self.tables.expense.maint_pm * (self.scen.cpi(t - 1) / self.scen.cpi.at(0))

    @var
    def ie_m(self) -> Rate[Monthly]:
        if self.include_inv_exp:  # build-time branch (D8)
            return to_monthly(self.tables.inv_expense.ie_rate)
        return cast(Rate[Monthly], 0.0)

    # ---- cash flows -----------------------------------------------------------------
    @var(timing="start")
    def ann_out(self, t: Int) -> Money:
        return self.pols_if(t - 1) * self.ann_pp(t)  # Count x PerPolicy[Money] -> Money

    @var(timing="start")
    def exp_out(self, t: Int) -> Money:
        return self.pols_if(t - 1) * self.exp_pp(t)

    @var(timing="end")
    def inv_exp(self, t: Int) -> Money:
        if self.include_inv_exp:  # the unused branch never enters the graph
            return self.ie_m() * self.bel(t)  # Rate[Monthly] x Money: charge on the closing BEL
        return cast(Money, 0.0)

    # ---- discounting ----------------------------------------------------------------
    @var
    def df(self, t: Int) -> Factor:
        """The composed current curve P(t)."""
        return self.scen.rf_df(t) * cc_discount(self.tables.ilp.ilp, months=t)

    @var
    def v(self, t: Int) -> Factor:
        """One-month discount factor of month t."""
        return self.df(t) / self.df(t - 1)

    # ---- best-estimate liability ----------------------------------------------------
    @var(terminal=0, role="reserve")
    def bel(self, t: Int) -> Money:
        """Value at t of the fulfilment cash flows of months t+1..T."""
        return (
            pv_month(self.ann_out, t + 1, self.v)
            + pv_month(self.exp_out, t + 1, self.v)
            + pv_month(self.inv_exp, t + 1, self.v)
            + self.v(t + 1) * self.bel(t + 1)
        )

    # ---- per-life values, best estimate ---------------------------------------------
    @var(timing="start")
    def s_pp(self, t: Int) -> MoneyPerPolicy:
        return self.ann_pp(t) + self.exp_pp(t)

    @var(terminal=0)
    def v_be(self, t: Int) -> MoneyPerPolicy:
        """Value at t per life in force at t."""
        p = 1 - self.q_mth(t + 1)  # Prob[Monthly]
        nxt = self.v_be(t + 1) + self.ie_m() * self.v_be(t + 1)  # Rate[Monthly] x PerPolicy
        return pv_month(self.s_pp, t + 1, self.v) + p * (self.v(t + 1) * nxt)

    # ---- risk adjustment: one mean-one latent factor at confidence level alpha ------
    @var
    def mort_mult(self) -> Factor:
        ra = self.tables.ra_param
        return exp(-ra.sigma_l * ra.z_alpha - ra.sigma_l * ra.sigma_l / 2)

    @var
    def exp_mult(self) -> Factor:
        ra = self.tables.ra_param
        return exp(ra.sigma_e * ra.z_alpha - ra.sigma_e * ra.sigma_e / 2)

    @var
    def q_ann_st(self, t: Int) -> Prob[Annual]:
        return where(self.age(t) >= OMEGA, 1.0, self.q_ann(t) * self.mort_mult())

    @var
    def q_mth_st(self, t: Int) -> Prob[Monthly]:
        return to_monthly_cf(self.q_ann_st(t))

    @var(timing="start")
    def s_pp_st(self, t: Int) -> MoneyPerPolicy:
        return self.ann_pp(t) + self.exp_pp(t) * self.exp_mult()

    @var(terminal=0)
    def v_st(self, t: Int) -> MoneyPerPolicy:
        p = 1 - self.q_mth_st(t + 1)
        nxt = self.v_st(t + 1) + self.ie_m() * self.v_st(t + 1)
        return pv_month(self.s_pp_st, t + 1, self.v) + p * (self.v(t + 1) * nxt)

    @var(domain="state")
    def ra(self, t: Int) -> Money:
        return self.pols_if(t) * (self.v_st(t) - self.v_be(t))

    # ---- coverage units (B119) ------------------------------------------------------
    @var
    def cu(self, t: Int) -> CoverageUnits:
        """The amount payable times the expected number of lives entitled to it."""
        return as_cu(self.ann_out(t))

    # ---- per-policy scalars: direct present values, the para 47 test ---------------
    @var
    def pva_t(self, t: Int) -> Money:
        return self.df(t - 1) * self.ann_out(t)

    @var
    def pve_t(self, t: Int) -> Money:
        return (self.df(t - 1) * self.exp_out(t)) + (self.df(t) * self.inv_exp(t))

    @var
    def pva(self) -> Money:
        """Present value of the annuity payments, summed in ascending t."""
        return sum_t(self.pva_t)

    @var
    def pve(self) -> Money:
        """Present value of the maintenance and investment expenses, in ascending t."""
        return sum_t(self.pve_t)

    @var
    def premium_total(self) -> Money:
        """The policy's term of P_g: pols_if_init x single_premium."""
        return self.mp.pols_if_init * self.mp.single_premium

    @var
    def onerous(self) -> Bool:
        """The para 47 test: the premium falls short of the fulfilment cash flows at t = 0
        (FCF formed first, as G3 writes it)."""
        return (self.premium_total() - (at(self.bel, 0) + at(self.ra, 0))) < 0

    @var
    def run_off_month(self) -> Int:
        """t_end as an output, so readers can flag per-life values beyond it."""
        return self.mp.t_end

    # ---- disclosure: the annualised rates of the composed curve ---------------------
    @var
    def fwd_ann(self, t: Int) -> Rate[Annual]:
        return ann_fwd(self.df(t - 1), self.df(t))

    @var
    def spot_ann(self, t: Int) -> Rate[Annual]:
        return where(t > 0, ann_spot(self.df(t), t), 0.0)


class EscalatingAnnuity(LevelAnnuity):
    """Abstract: adds the anniversary mask. Not run on its own."""

    abstract = True

    @var
    def anniv(self, t: Int) -> Bool:
        """Month t's payment falls on an anniversary of the first payment."""
        k = t - self.mp.m_c  # months since the first payment
        return (k > 0) & (k % 12 == 0)


class FixedEscalationAnnuity(EscalatingAnnuity):
    """Payment x (1 + esc_rate) on each anniversary."""

    model_point = FIXED_SCHEMA
    invariants = (*_COMMON_INVARIANTS, _I8_FIXED)

    @var
    def esc_factor(self, t: Int) -> Factor:
        return where(self.anniv(t), annual_step(self.mp.esc_rate), 1.0)


class CPILinkedAnnuity(EscalatingAnnuity):
    """Payment x I(t - L) / I(t - L - 12) on each anniversary; no cap, no floor."""

    cpi_lag_m: Static[int] = 3  # reference lag L, in months

    invariants = (*_COMMON_INVARIANTS, _I9_CPI)

    @var
    def esc_factor(self, t: Int) -> Factor:
        lag = self.cpi_lag_m  # a build-time constant, so the offsets are fixed
        return where(self.anniv(t), self.scen.cpi(t - lag) / self.scen.cpi(t - (lag + 12)), 1.0)

    @var
    def cpi_lag(self) -> Int:
        """The lag L as a scalar, read by invariant I9 (a build-time constant otherwise)."""
        return cast(Int, self.cpi_lag_m)
