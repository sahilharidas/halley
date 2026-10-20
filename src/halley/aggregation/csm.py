"""The contractual service margin of one group: the monthly roll-forward, the recognition of a
group new at the valuation date and the period close of a group in force (design section 9;
annuity spec, "Aggregation stage: CSM for one group"; in-force spec, sections 3 to 12).

:class:`CsmRollForward` is the monthly roll-forward the annuity spec writes in evaluation order,
"Accretion and release" and "Onerous groups and the loss component": the locked-in one-month
rate and discount factor from the locked-in curve ``p_l``, the coverage units still to be
provided ``N(t)`` by a backward recursion (discounted on the locked-in curve by default, Method 1
of the 2022 agenda decision, or undiscounted by the build-time switch), the accretion, the
release in proportion to the month's units, and the loss component by the ratio of
IFRS 17 Illustrative Example 8. Every division sits inside a ``where`` whose divisor is guarded,
because both branches are evaluated. The opening balances are hooks: ``csm_0`` and ``lc_0`` (the
inits of ``csm`` and ``lc``) and ``p_g``, which a subclass defines. Beside them it computes the
lines of milestone M12 for the months on the ``cur`` basis (design section 9): the insurance
revenue of an expected run and the insurance finance expense, at the current forward rates of
the composed current curve ``df_cur``, with the invariants M1 (the LRC moves by IFIE less
revenue and the loss component's share of the service items) and M2 (the loss component's
allocation is that share less its share of the finance unwind); the loader holds ``df_cur`` to
the files the cur runs composed their curve from (``discloses_current_curve``). Its N(t) needs
every later month of the units, so it runs to its bases' run-off (``reads_to_run_off``).

:class:`NewGroup` is the stage of milestone M7: a group recognised at the valuation date, whose
``P_g`` is the sum of its members' premiums and whose opening CSM or loss component is
``max(0, P_g - FCF_g(0))`` or ``max(0, FCF_g(0) - P_g)``, with ``FCF_g = BEL_g + RA_g`` formed
first (paras 38 and 47), and whose locked-in curve is the composed curve of the runs it reads
(``recognised_at_valuation``: P_L = P). Its member rules G5 and G3 check every contract on the
runs the stage reads; its invariants I11 to I14, M1 and M2 are checked on every run. It reads a
deterministic basis (J = 1): G3 is each contract's para 47 test at one scenario, and P_L = P
names the curve of one run (annuity spec, G6 and "Scenarios", until M12).

:class:`InForceGroup` is the stage of milestone M11: a group measured before, at
``prev_valuation_date`` by the previous aggregation run its groups row names, closed for the
period ``(V_prev, V]`` in the in-force spec's order (section 9): the opening CSM accreted at the
locked-in forward rates of the period, the loss component allocated by the previous run's ratio
``F_prev(tau) / F_prev(0)``, the remeasurement rules applied to the change in locked-in
fulfilment cash flows ``FCF_g^L_new(0) - FCF_g^L_old(0)`` of its bases ``new`` and ``old``, one
release for the period on the units actually provided and those still to come, and the closing
balances, which open the months after V of the roll-forward above on the ``cur`` basis. It
stores the analysis of change of in-force section 11.2 and checks K1 to K7 beside I11 to I13,
M1 and M2. Its locked-in files are rebuilt from the basis the group locks in and its old basis
is held to the previous run (``rebuilds_locked_in``), so it reads the actual CPI history
``cpi_act`` the old and new runs' inputs were built from.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any, ClassVar, NoReturn, cast

import numpy as np
from numpy.typing import NDArray

from halley.aggregation.stage import (
    Aggregation,
    Basis,
    Disclosure,
    GroupSum,
    MemberRule,
    MemberView,
)
from halley.authoring import (
    BatchView,
    Column,
    CoverageUnits,
    Factor,
    InputRule,
    InputView,
    Int,
    Invariant,
    Key,
    Money,
    Monthly,
    Rate,
    ScenarioInput,
    ScenarioRule,
    Static,
    Table,
    TableRule,
    at,
    fwd_rate,
    minimum,
    var,
    where,
)
from halley.authoring.contracts import group_schema
from halley.errors import BuildError, LoadError, Loc

EPS = 1e-12
"""The annuity spec's tolerance for its invariants within a backend."""

ONEROUS = "ONEROUS"
"""The ``profit_bucket`` of a group of contracts onerous at initial recognition (para 16(a))."""

# ----------------------------------------------------------------------- inputs and outputs

P_L = ScenarioInput(
    "p_l",
    "Factor",
    doc="The group's locked-in curve P_L(t), re-based to the valuation date (B72(b))",
    gt=0,
    le=2,
    column="rf_df",
    covers=(0, "H"),
    rules=(
        ScenarioRule(
            "p_l(0) = 1",
            lambda v: v.at(0) == 1.0,
            "the locked-in curve is re-based to the valuation date",
        ),
    ),
)
"""The locked-in curve, built by :func:`halley.aggregation.inforce.locked_in_curve` (its value
column is ``rf_df``, as a discount file's is): for a group new at V, the composed curve of the
runs it reads (annuity spec, "Locked-in rates"), which the loader verifies for
:class:`NewGroup` (``recognised_at_valuation``)."""

DF_CUR = ScenarioInput(
    "df_cur",
    "Factor",
    doc="The composed current curve P(t) of the cur basis's runs, one scenario: the para 120 curve",
    gt=0,
    le=2,
    column="rf_df",
    covers=(0, "H"),
    rules=(
        ScenarioRule(
            "df_cur(0) = 1",
            lambda v: v.at(0) == 1.0,
            "the current curve discounts to the valuation date",
        ),
    ),
)
"""The composed current curve, built by :func:`halley.aggregation.inforce.composed_curve` from
the discount file and the ILP table the cur basis's runs read (value column ``rf_df``): the
curve their BEL and RA unwind on, whose forward rates the M12 lines read (design section 9) and
whose file the manifest discloses as the para 120 curve. The loader requires its metadata to
name the ``rf_df`` and ``ilp`` files every cur run read and recomputes it from them
(:attr:`CsmRollForward.discloses_current_curve`); M1 checks its values on every run. One
scenario: the stage reads mean series, so the loader holds every scenario input of the stage to
``require_deterministic`` (the one J = 1 check, D4)."""

ROLL_OUTPUTS = (
    "cu_g",
    "fcf_g",
    "bel_g",
    "ra_g",
    "j_l",
    "d_l",
    "d_cu",
    "cu_next",
    "units",
    "csm_open",
    "acc",
    "csm_pre",
    "release",
    "csm",
    "lc",
    "lc_alloc",
    "lrc",
    "ann_out_g",
    "exp_out_g",
    "inv_exp_g",
    "f",
    "ifie_bel",
    "ra_fin",
    "ra_rel",
    "sar",
    "svc",
    "lc_svc",
    "revenue",
    "ifie",
)
"""What :class:`CsmRollForward` stores: the group series it reads, every line of the
roll-forward tables of the annuity spec and the M12 lines (revenue and IFIE), by month. The
CSM table's opening line is ``csm_open``; its changes for future service are 0 in every month
of the roll (annuity spec, "Roll-forward table"; in-force section 10: the next remeasurement is
at the next period close), so they are not a stored line, and an in-force group's change is
its close's, a figure of the in-force stage."""

DISCOUNT_RATES = (
    "Bottom-up (B80): a liquid risk-free curve plus an illiquidity premium, continuously "
    "compounded, composed as P(t) = rf_df(t) x exp(-ilp x t / 12). The fulfilment cash flows are "
    "discounted at the current composed curve; the CSM accretes at the forward rates of the "
    "composed curve at the group's initial recognition, re-based to the valuation date (B72(b))."
)
"""The approach to discount rates (para 117(c)(iii)), annuity spec "Discounting". How the
coverage units are weighted depends on the build (``discount_coverage_units``), so it is a note
(:func:`roll_notes`), not part of this fixed string."""

RISK_ADJUSTMENT = (
    "Confidence-level technique on one latent factor: the risk adjustment for non-financial risk "
    "corresponds to the disclosed confidence level for the expected present value of the future "
    "cash flows (annuity payments and expenses) under systematic longevity and expense risk, "
    "over the full run-off. Random variation in individual deaths and expenses is assumed "
    "diversified. No diversification benefit is taken between contracts' systematic exposures "
    "or between longevity and expense risk."
)
"""The approach to the risk adjustment (paras 117(c)(ii), 119), annuity spec "Disclosure"."""

NOTE_MONTHLY_STEP = (
    "The CSM is allocated monthly: a reporting period's release is the sum of its monthly "
    "releases. With discounted coverage units this differs slightly from one release per period, "
    "because a monthly step discounts within the period (annuity spec, open question 4)."
)
NOTE_TIMING_CONVEXITY = (
    "A deterministic run on the market-implied curve and CPI path stands for the "
    "probability-weighted mean of para 33(a); the timing convexity adjustment between the lagged "
    "index dates and the payment dates is ignored as immaterial (annuity spec, 'Scenarios')."
)
NOTE_PERIOD_ALLOCATION = (
    "The loss component is allocated over a reporting period by the previous run's ratio "
    "F_prev(tau) / F_prev(0), the monthly ratio telescoped, so the period allocation is the "
    "previous run's own projected LC(tau) (in-force spec, C11)."
)
NOTE_LOCKED_IN_RUNS = (
    "The locked-in runs (bases old and new) are ordinary projections that read the locked-in "
    "curve P_L as their rf_df with an illiquidity premium table of 0.0, so exp(0) = 1 leaves P_L "
    "unchanged (in-force spec, C17)."
)
"""The notes a stage's manifest discloses (design section 9, "Outputs"). The first two are the
annuity spec's ("Outputs", "Manifest") and every CSM stage's; the last two belong to the period
close of an in-force group, whose stage adds them."""

NOTE_UNITS_DISCOUNTED = (
    "Coverage units (B119): the units of a month are the payments expected to be validly "
    "claimable in it, and the units still to be provided are discounted on the locked-in curve "
    "(Method 1 of the IFRS Interpretations Committee's 2022 agenda decision; annuity spec, "
    "'Coverage units')."
)
NOTE_UNITS_UNDISCOUNTED = (
    "Coverage units (B119): the units of a month are the payments expected to be validly "
    "claimable in it, undiscounted, so every month's units are weighted equally "
    "(discount_coverage_units = False; annuity spec, 'Coverage units')."
)
"""The coverage-unit approach a CSM stage discloses, by its ``discount_coverage_units``
(:func:`roll_notes`)."""


def roll_notes(statics: Mapping[str, object]) -> tuple[str, ...]:
    """The notes of a CSM roll-forward's manifest: the monthly allocation step, the ignored
    timing convexity and the coverage-unit approach the stage was built with, from its statics
    (``discount_coverage_units``), so that a stage built with undiscounted units does not
    disclose discounted ones. The loader calls it with the compiled stage's statics."""
    units = NOTE_UNITS_DISCOUNTED if statics["discount_coverage_units"] else NOTE_UNITS_UNDISCOUNTED
    return (NOTE_MONTHLY_STEP, NOTE_TIMING_CONVEXITY, units)


# -------------------------------------------------------------------------------- invariants


def _guard(x: NDArray[Any]) -> NDArray[Any]:
    """A divisor that is itself where it is positive and 1 elsewhere, so a masked ratio is
    finite and the comparison is decided by the mask."""
    return np.where(x > 0, x, 1.0)


def _rel(a: BatchView, lhs: NDArray[Any], rhs: NDArray[Any]) -> NDArray[Any]:
    return np.maximum(np.abs(lhs), np.abs(rhs))


def _next(x: NDArray[Any]) -> NDArray[Any]:
    """``x(t + 1)`` at row ``t``, zero at row ``H``: a step from month t to month t + 1 is
    checked at row t, so the arrays keep the run's length whatever ``H`` is."""
    return np.concatenate([x[1:], np.zeros_like(x[:1])], axis=0)


def _i11_rhs(a: BatchView) -> NDArray[Any]:
    """CSM(t - 1) + Acc(t) - R(t), months 1..H."""
    csm = a.var("csm")
    return (csm[:-1] + a.var("acc")[1:]) - a.var("release")[1:]


def _i11_scale(a: BatchView, lhs: NDArray[Any], rhs: NDArray[Any]) -> NDArray[Any]:
    """CSM(t - 1) + |Acc(t)| + R(t): the size of the terms combined."""
    return (a.var("csm")[:-1] + np.abs(a.var("acc")[1:])) + a.var("release")[1:]


def _from_the_last_units(a: BatchView) -> NDArray[np.bool_]:
    """The months from the last month with CU_g > 0 on (from month 1 when no month has units):
    the release of that month is the whole CSM before it, since N(t) is exactly 0 there."""
    covered = np.where(a.var("cu_g") > 0, a.t, 0)
    last = np.maximum(covered.max(axis=0, keepdims=True), 1)
    return np.asarray(a.t >= last)


def _per_unit(a: BatchView) -> NDArray[Any]:
    """R(t) / CU_g(t), guarded where CU_g(t) = 0."""
    return a.var("release") / _guard(a.var("cu_g"))


def _i12_csm_per_unit(a: BatchView) -> NDArray[Any]:
    """CSM(t) / N(t), guarded where N(t) = 0."""
    return a.var("csm") / _guard(a.var("cu_next"))


def _i12_where(a: BatchView) -> NDArray[np.bool_]:
    return np.asarray((a.var("cu_g") > 0) & (a.var("cu_next") > 0) & (a.var("csm") > 0))


def _i12_step_rhs(a: BatchView) -> NDArray[Any]:
    """[R(t) / CU_g(t)] x a_L(t + 1) x [CU_g(t + 1) + d(t + 1) x N(t + 1)] / U(t + 1) at row t,
    with a_L(t + 1) = P_L(t) / P_L(t + 1) and d the units' one-month discount factor (d_L, or 1
    for undiscounted units)."""
    p = a.scen_range("p_l", 0, a.H)
    a_l = np.concatenate([p[:-1] / p[1:], np.ones_like(p[:1])], axis=0)
    units_next = _next(a.var("cu_g")) + _next(a.var("d_cu")) * _next(a.var("cu_next"))
    return ((_per_unit(a) * a_l) * units_next) / _guard(_next(a.var("units")))


def _i12_step_where(a: BatchView) -> NDArray[np.bool_]:
    cu = a.var("cu_g")
    return np.asarray((cu > 0) & (_next(cu) > 0))


def _i13_lhs(a: BatchView) -> NDArray[Any]:
    """LC(t) / F(t), guarded where F(t) = 0."""
    return a.var("lc") / _guard(a.var("fcf_g"))


def _i13_rhs(a: BatchView) -> NDArray[Any]:
    """LC(0) / F(0), guarded where F(0) = 0."""
    return a.at("lc", 0) / _guard(a.at("fcf_g", 0))


def _i13_where(a: BatchView) -> NDArray[np.bool_]:
    return np.asarray((a.var("fcf_g") > 0) & (a.at("fcf_g", 0) > 0) & (a.at("lc", 0) > 0))


def _p_g(a: BatchView, lhs: NDArray[Any], rhs: NDArray[Any]) -> NDArray[Any]:
    """P_g, the group's scale (the annuity spec's "Quantities")."""
    return a.var("p_g")


def _m1_lhs(a: BatchView) -> NDArray[Any]:
    """LRC(t) - LRC(t - 1), months 1..H."""
    lrc = a.var("lrc")
    return lrc[1:] - lrc[:-1]


def _m1_rhs(a: BatchView) -> NDArray[Any]:
    """IFIE(t) - revenue(t) - LC_svc(t), left to right, months 1..H."""
    return (a.var("ifie")[1:] - a.var("revenue")[1:]) - a.var("lc_svc")[1:]


def _m2_rhs(a: BatchView) -> NDArray[Any]:
    """LC_svc(t) - SAR(t) x [IFIE_BEL(t) + RA_fin(t)], months 1..H."""
    finance = a.var("ifie_bel")[1:] + a.var("ra_fin")[1:]
    return a.var("lc_svc")[1:] - a.var("sar")[1:] * finance


ROLL_INVARIANTS = (
    Invariant(
        "I11 CSM(t) = CSM(t-1) + Acc(t) - R(t)",
        lambda a: a.var("csm")[1:],
        _i11_rhs,
        tol=EPS,
        scale=_i11_scale,
        months=(1, "H"),
        reads=("csm", "acc", "release"),
        doc="the roll-forward lines sum to the closing CSM",
    ),
    Invariant("I11 CSM(t) >= 0", lambda a: a.var("csm"), 0.0, kind="ge", tol=0, reads=("csm",)),
    Invariant(
        "I11 CSM(t) = 0 from the last month with CU_g > 0",
        lambda a: a.var("csm"),
        0.0,
        tol=0,
        where=_from_the_last_units,
        reads=("csm", "cu_g"),
        doc="the last month's release is the whole CSM, since no units remain after it",
    ),
    Invariant(
        "I12 R(t) / CU_g(t) = CSM(t) / N(t)",
        _per_unit,
        _i12_csm_per_unit,
        tol=EPS,
        scale=_rel,
        where=_i12_where,
        reads=("release", "cu_g", "csm", "cu_next"),
        doc="B119(b): the CSM is allocated equally to each unit, where CU_g, N and CSM are > 0",
    ),
    Invariant(
        "I12 R(t+1) / CU_g(t+1) = [R(t) / CU_g(t)] a_L(t+1) [CU_g(t+1) + d(t+1) N(t+1)] / U(t+1)",
        lambda a: _next(_per_unit(a)),
        _i12_step_rhs,
        tol=EPS,
        scale=_rel,
        where=_i12_step_where,
        reads=("release", "cu_g", "d_cu", "cu_next", "units"),
        doc="the release per unit steps by the accretion and the units, where CU_g(t), "
        "CU_g(t+1) > 0; checked at row t",
    ),
    Invariant(
        "I13 LC(t) / F(t) = LC(0) / F(0)",
        _i13_lhs,
        _i13_rhs,
        tol=EPS,
        scale=_rel,
        months=(1, "H"),
        where=_i13_where,
        reads=("lc", "fcf_g"),
        doc="the loss component's share of F is constant without a remeasurement, where F > 0",
    ),
    Invariant(
        "I13 LC(t) = 0 where F(t) = 0",
        lambda a: a.var("lc"),
        0.0,
        tol=0,
        months=(1, "H"),
        where=lambda a: np.asarray(a.var("fcf_g") == 0),
        reads=("lc", "fcf_g"),
    ),
    Invariant(
        "M1 LRC(t) - LRC(t-1) = IFIE(t) - revenue(t) - LC_svc(t)",
        _m1_lhs,
        _m1_rhs,
        tol=EPS,
        scale=_p_g,
        months=(1, "H"),
        reads=("lrc", "ifie", "revenue", "lc_svc", "p_g"),
        doc="the LRC moves by the finance expense less the revenue and the loss component's "
        "share of the service items (B121, B124, para 87), on the current curve df_cur",
    ),
    Invariant(
        "M2 LCA(t) = LC_svc(t) - SAR(t) [IFIE_BEL(t) + RA_fin(t)]",
        lambda a: a.var("lc_alloc")[1:],
        _m2_rhs,
        tol=EPS,
        scale=_p_g,
        months=(1, "H"),
        reads=("lc_alloc", "lc_svc", "sar", "ifie_bel", "ra_fin", "p_g"),
        doc="the loss component's allocation is its share of the service items less its share "
        "of the finance unwind, which stays in IFIE (paras 50(a), 51)",
    ),
)
"""The annuity spec's I11, I12 and I13 ("Built-in invariants"), each part an Invariant, and the
design's M1 and M2 (section 9, M12; scale P_g, 1e-12). I11's exact zero covers the last month
with units itself (where N is exactly 0, so the release is the whole CSM), which the literal
"after" leaves vacuous when that month is H (in-force open question 17). M1 and M2 hold only
when ``df_cur`` is the curve the cur runs' BEL and RA unwind on, so they also check that
curve."""


def _i14_lhs(a: BatchView) -> NDArray[Any]:
    """CSM(0) in an ONEROUS group, LC(0) in any other: 0 when G3 held for every contract."""
    return np.where(a.mp.profit_bucket == ONEROUS, a.var("csm_0"), a.var("lc_0"))


I14 = Invariant(
    "I14 G3 for the group: CSM(0) = 0 if ONEROUS, else LC(0) = 0",
    _i14_lhs,
    0.0,
    tol=0,
    reads=("csm_0", "lc_0"),
    doc="every contract of an ONEROUS group is onerous, so P_g < FCF_g(0), and no contract of "
    "another group is, so P_g >= FCF_g(0); the member rule G3 checks each contract at load",
)
"""The annuity spec's I14 as the stage can check it on its own figures: the group-level
consequence of G3 holding for every contract. The per-contract test is the member rule
:data:`G3`, which the loader evaluates on the runs' ``onerous`` outputs."""

# ------------------------------------------------------------------------------ member rules


def _g3(v: MemberView) -> NDArray[np.bool_]:
    """Contract i is onerous (its run's ``onerous`` output, ``P_i - FCF_i(0) < 0`` with FCF
    formed first) exactly when its ``profit_bucket`` is ONEROUS. At one scenario: with J > 1
    the run's ``onerous`` is pathwise, and only the probability-weighted mean of FCF_i(0) is a
    para 33(a) estimate (annuity spec, "Labelling"); that test is left with G6 to M12."""
    from halley.aggregation import require_deterministic  # the one J = 1 check (D4)

    require_deterministic(
        v,
        f"basis {v.basis} run {v.run_id}",
        "G3 reads each contract's para 47 test, onerous, at one scenario",
        LoadError,
    )
    onerous = np.asarray(v.output("onerous")[0, :, 0], dtype=np.bool_)
    return np.asarray(onerous == (v.mp.profit_bucket == ONEROUS), dtype=np.bool_)


def _g5(v: MemberView) -> NDArray[np.bool_]:
    """Every contract was issued on the valuation date."""
    return np.asarray(v.mp.issue_date == v.valuation_date, dtype=np.bool_)


G3 = MemberRule(
    "G3 a contract is onerous exactly in an ONEROUS group",
    _g3,
    reads=("onerous",),
    doc="for a group new at V, contract i is onerous iff P_i - FCF_i(0) < 0 with "
    "P_i = pols_if_init_i x single_premium_i and FCF_i(0) = bel_i(0) + ra_i(0); an onerous "
    "contract outside an ONEROUS group, or a non-onerous one inside, is an error (paras 16(a), "
    "47)",
)
G5 = MemberRule(
    "G5 every contract is new at V",
    _g5,
    doc="issue_date = V for every contract of a group recognised at V (paras 25, 28); an "
    "in-force group is measured by the in-force stage",
)

# ------------------------------------------------------------------------------------ classes


def _defined_at(cls: type) -> Loc | None:
    """Where a class is defined, for an error about the class rather than one of its lines."""
    try:
        file = inspect.getsourcefile(cls)
        _lines, line = inspect.getsourcelines(cls)
    except (OSError, TypeError):
        return None
    return None if file is None else Loc(file, line, cls.__qualname__)


def _undefined_hook(stage: object, name: str) -> NoReturn:
    """The BuildError of a hook the class being built does not define: located at that class,
    the user's subclass, not at the hook's line here (errors point at the model source)."""
    cls = type(stage)
    raise BuildError(
        f"{cls.__qualname__} does not define {name}, a hook of CsmRollForward: a subclass "
        "defines csm_0 and lc_0 (the opening balances) and p_g (the group's premiums)",
        _defined_at(cls),
    )


class CsmRollForward(Aggregation):
    """The monthly CSM and loss-component roll-forward of one group on the ``cur`` basis
    (annuity spec, "Accretion and release", "Onerous groups and the loss component"), in the
    spec's evaluation order and grouping, and the M12 lines of its months, revenue and IFIE
    (design section 9). A subclass defines the hooks ``csm_0``, ``lc_0`` (the opening balances,
    the inits of ``csm`` and ``lc``) and ``p_g`` (the premiums of the group, the scale of M1,
    M2 and of the in-force close)."""

    abstract = True

    discount_coverage_units: Static[bool] = True  # Method 1 of the 2022 agenda decision

    model_point = group_schema()
    bases = (Basis("cur", doc="the projection runs at the valuation date, one per class"),)
    series = (
        GroupSum("cu_g", "cur", "cu", "CoverageUnits", doc="CU_g(t), t = 1..H"),
        GroupSum("fcf_g", "cur", ("bel", "ra"), "Money", start=0, doc="FCF_g = BEL_g + RA_g"),
        GroupSum("bel_g", "cur", "bel", "Money", start=0, doc="BEL_g(t), t = 0..H"),
        GroupSum("ra_g", "cur", "ra", "Money", start=0, doc="RA_g(t), t = 0..H"),
        GroupSum("ann_out_g", "cur", "ann_out", "Money", doc="expected payments of month t"),
        GroupSum("exp_out_g", "cur", "exp_out", "Money", doc="expected expenses of month t"),
        GroupSum("inv_exp_g", "cur", "inv_exp", "Money", doc="investment expenses of month t"),
    )
    scenario_inputs = (P_L, DF_CUR)
    outputs: ClassVar[tuple[str, ...] | None] = ROLL_OUTPUTS
    invariants = ROLL_INVARIANTS
    disclosures: ClassVar[dict[str, Disclosure]] = {
        "discount_rates": DISCOUNT_RATES,
        "risk_adjustment": RISK_ADJUSTMENT,
        "notes": roll_notes,  # by discount_coverage_units, as built
    }
    reads_to_run_off: ClassVar[bool] = True  # N(t) sums the units of every month after t
    discloses_current_curve: ClassVar[bool] = True  # df_cur: the cur runs' composed curve

    # ---- hooks: the opening balances and the premiums --------------------------------
    @var
    def csm_0(self) -> Money:
        """CSM(0), the init of ``csm``: a subclass defines it."""
        _undefined_hook(self, "csm_0")

    @var
    def lc_0(self) -> Money:
        """LC(0), the init of ``lc``: a subclass defines it."""
        _undefined_hook(self, "lc_0")

    @var
    def p_g(self) -> Money:
        """P_g, the group's premiums: a subclass defines it."""
        _undefined_hook(self, "p_g")

    # ---- the group series, stored ------------------------------------------------------
    @var
    def cu_g(self, t: Int) -> CoverageUnits:
        return self.series.cu_g(t)

    @var(domain="state")
    def fcf_g(self, t: Int) -> Money:
        """F(t): the present value of future outflows plus the RA; there are no inflows."""
        return self.series.fcf_g(t)

    @var(domain="state")
    def bel_g(self, t: Int) -> Money:
        return self.series.bel_g(t)

    @var(domain="state")
    def ra_g(self, t: Int) -> Money:
        return self.series.ra_g(t)

    # ---- the locked-in curve -----------------------------------------------------------
    @var
    def j_l(self, t: Int) -> Rate[Monthly]:
        """The locked-in one-month accretion rate j_L(t) = P_L(t - 1) / P_L(t) - 1."""
        return fwd_rate(self.scen.p_l(t - 1), self.scen.p_l(t))

    @var
    def d_l(self, t: Int) -> Factor:
        """The locked-in one-month discount factor d_L(t) = P_L(t) / P_L(t - 1)."""
        return self.scen.p_l(t) / self.scen.p_l(t - 1)

    @var
    def d_cu(self, t: Int) -> Factor:
        """d(t) of the coverage units: d_L(t) for discounted units, 1 for undiscounted."""
        if self.discount_coverage_units:  # build-time branch (D8)
            return self.d_l(t)
        return cast(Factor, 1.0)

    # ---- coverage units (B119) ---------------------------------------------------------
    @var(terminal=0)
    def cu_next(self, t: Int) -> CoverageUnits:
        """N(t): the units of the months after t, valued at the end of month t."""
        return self.series.cu_g(t + 1) + self.d_cu(t + 1) * self.cu_next(t + 1)

    @var
    def units(self, t: Int) -> CoverageUnits:
        """U(t): the current month's units, undiscounted, and the units still to come."""
        return self.series.cu_g(t) + self.cu_next(t)

    # ---- accretion and release ---------------------------------------------------------
    @var
    def csm_open(self, t: Int) -> Money:
        """The opening CSM of month t, CSM(t - 1): the first line of the roll-forward table."""
        return self.csm(t - 1)

    @var
    def acc(self, t: Int) -> Money:
        """Acc(t), interest accretion at the locked-in rate (para 44(b), B72(b))."""
        return self.j_l(t) * self.csm(t - 1)

    @var
    def csm_pre(self, t: Int) -> Money:
        return self.csm(t - 1) + self.acc(t)

    @var
    def release(self, t: Int) -> Money:
        """R(t), the release in proportion to the month's units (para 44(e), B119)."""
        u = self.units(t)
        share = self.series.cu_g(t) / where(u > 0, u, 1.0)
        return where(u > 0, self.csm_pre(t) * share, self.csm_pre(t))

    @var(init=lambda g: g.csm_0())
    def csm(self, t: Int) -> Money:
        return self.csm_pre(t) - self.release(t)

    # ---- the loss component (paras 49 to 52) -------------------------------------------
    @var(init=lambda g: g.lc_0())
    def lc(self, t: Int) -> Money:
        """LC(t) = LC(t - 1) x F(t) / F(t - 1), and 0 once F(t - 1) = 0."""
        f0 = self.series.fcf_g(t - 1)
        ratio = self.series.fcf_g(t) / where(f0 > 0, f0, 1.0)
        return where(f0 > 0, self.lc(t - 1) * ratio, 0.0)

    @var
    def lc_alloc(self, t: Int) -> Money:
        """LCA(t), the systematic allocation of the month (para 50(a))."""
        return self.lc(t - 1) - self.lc(t)

    # ---- the liability for remaining coverage ------------------------------------------
    @var(domain="state")
    def lrc(self, t: Int) -> Money:
        """LRC(t) = FCF_g(t) + CSM(t); the LC is part of the FCF, not added to it."""
        return self.series.fcf_g(t) + self.csm(t)

    @var
    def lrc_0(self) -> Money:
        return self.series.fcf_g.at(0) + self.csm_0()

    # ---- M12: insurance revenue and IFIE of the months on the cur basis (design 9) -------
    @var
    def ann_out_g(self, t: Int) -> Money:
        return self.series.ann_out_g(t)

    @var
    def exp_out_g(self, t: Int) -> Money:
        return self.series.exp_out_g(t)

    @var
    def inv_exp_g(self, t: Int) -> Money:
        return self.series.inv_exp_g(t)

    @var
    def f(self, t: Int) -> Rate[Monthly]:
        """The current one-month forward rate f(t) = P(t - 1) / P(t) - 1 of ``df_cur``."""
        return fwd_rate(self.scen.df_cur(t - 1), self.scen.df_cur(t))

    @var
    def ifie_bel(self, t: Int) -> Money:
        """The BEL's unwind at the current forward rate, on the opening BEL less the month's
        payments and expenses, which are made at its start (para 87)."""
        opening = (self.series.bel_g(t - 1) - self.series.ann_out_g(t)) - self.series.exp_out_g(t)
        return self.f(t) * opening

    @var
    def ra_fin(self, t: Int) -> Money:
        """The RA's finance part: its unwind at the current forward rate (para 81)."""
        return self.f(t) * self.series.ra_g(t - 1)

    @var
    def ra_rel(self, t: Int) -> Money:
        """The RA released for the month's service: the opening RA and its unwind, less the
        closing RA (para 81, B124(b)(ii))."""
        return (self.series.ra_g(t - 1) + self.ra_fin(t)) - self.series.ra_g(t)

    @var
    def sar(self, t: Int) -> Factor:
        """SAR(t) = LC(t - 1) / F(t - 1), the loss component's share of the opening outflows
        and RA (paras 50(a), 51)."""
        f0 = self.series.fcf_g(t - 1)
        return self.lc(t - 1) / where(f0 > 0, f0, 1.0)

    @var
    def svc(self, t: Int) -> Money:
        """The month's service items: the expected payments, expenses and investment expenses,
        and the RA released (B124)."""
        cash = (self.series.ann_out_g(t) + self.series.exp_out_g(t)) + self.series.inv_exp_g(t)
        return cash + self.ra_rel(t)

    @var
    def lc_svc(self, t: Int) -> Money:
        """The loss component's share of the service items, which revenue excludes (B124)."""
        return self.sar(t) * self.svc(t)

    @var
    def revenue(self, t: Int) -> Money:
        """Insurance revenue of an expected run (B121, B124): the service items less the loss
        component's share, and the CSM release."""
        return (self.svc(t) - self.lc_svc(t)) + self.release(t)

    @var
    def ifie(self, t: Int) -> Money:
        """Insurance finance expense (para 87): the BEL and RA unwind at the current rates and
        the CSM accretion at the locked-in rates. The loss component's share of the unwind
        stays in it."""
        return (self.ifie_bel(t) + self.ra_fin(t)) + self.acc(t)


class NewGroup(CsmRollForward):
    """A group recognised at the valuation date (milestone M7; annuity spec, "Initial
    recognition"): every member was issued on it (G5), the premiums P_g are the members' and
    the opening CSM or loss component is the gap between P_g and FCF_g(0), with the FCF formed
    first. G3 holds for every contract (the member rule at load, I14 on the group's figures).
    Its locked-in curve is the composed curve of the runs it reads, P_L = P, which the loader
    verifies from ``p_l``'s metadata (``recognised_at_valuation``)."""

    recognised_at_valuation: ClassVar[bool] = True  # P_L = P (annuity spec, "Locked-in rates")
    series = (
        *CsmRollForward.series,
        GroupSum(
            "premium_total",
            "cur",
            "premium_total",
            "Money",
            time=False,
            doc="P_g = the sum of pols_if_init x single_premium in ascending policy_id",
        ),
    )
    outputs: ClassVar[tuple[str, ...] | None] = (*ROLL_OUTPUTS, "p_g", "csm_0", "lc_0", "lrc_0")
    invariants = (*ROLL_INVARIANTS, I14)
    member_rules = (G5, G3)  # a group new at V first, then the para 47 test of each contract

    @var
    def p_g(self) -> Money:
        return self.series.premium_total()

    @var
    def gap(self) -> Money:
        """P_g - FCF_g(0): FCF formed first, never (P_g - BEL_g) - RA_g."""
        return self.p_g() - self.series.fcf_g.at(0)

    @var
    def csm_0(self) -> Money:
        """CSM(0) = max(0, P_g - FCF_g(0)) (para 38)."""
        return where(self.gap() > 0, self.gap(), 0.0)

    @var
    def lc_0(self) -> Money:
        """LC(0) = max(0, FCF_g(0) - P_g) (para 47)."""
        loss = self.series.fcf_g.at(0) - self.p_g()
        return where(loss > 0, loss, 0.0)


# ------------------------------------------------------------ the in-force group (M11, M9)

P_REC = ScenarioInput(
    "p_rec",
    "Factor",
    doc="The group's composed curve at recognition P_rec(t), not re-based (B72(b))",
    gt=0,
    le=2,
    column="rf_df",
    covers=(0, "H"),
    rules=(
        ScenarioRule(
            "p_rec(0) = 1",
            lambda v: v.at(0) == 1.0,
            "the curve at recognition discounts to the recognition date",
        ),
    ),
)
"""The composed curve at the group's recognition, ``P_rec(t) = rf_df_rec(t) x exp(-ilp x t /
12)`` for t = 0..tau0 + H (in-force 5.1), built by
:func:`halley.aggregation.inforce.locked_in_curve` (value column ``rf_df``): the period close
accretes the opening CSM at its forward factor ``P_rec(tau0 - tau) / P_rec(tau0)`` (in-force
9.2). The loader holds its metadata to the group's locked-in hashes and rebuilds it from them
(:attr:`InForceGroup.rebuilds_locked_in`); :data:`P_REC_COVERS` requires the months to the
run-off."""

CPI_ACT_FIRST_MONTH = -26
"""The first month of the actual CPI history at V: the runs read ``u >= -14``, and ``-26``
leaves twelve months of margin for the splice (in-force 3.6 and 4). A limit of this version:
the table's key is a dense grid, so the history holds exactly the months -26..0, and the
expense rebase, which reads ``I_act(-tau)``, serves periods of at most 26 months."""

CPI_ACT = Table(
    "cpi_act",
    keys=(Key("u", lo=CPI_ACT_FIRST_MONTH, hi=0),),
    columns=(Column("cpi", "Index", le=1_000_000),),
    doc="The actual CPI history as known at V, u = -26..0; the month u = 0 is an estimate (C9)",
    metadata=("estimated_rows",),
    rules=(TableRule("cpi > 0", lambda v: v.cpi > 0),),
)
"""The actual CPI history at V (in-force 3.6, 5.2), one path with no scenario column. No formula
reads it: the stage reads it so that the loader can rebuild, bit for bit, the files its old and
new runs read that were built from it, the expense table rebased to V prices
(``rebase_expense_table``, in-force 5.4) and the locked-in CPI path (``locked_in_cpi``, 5.2),
which must name this file by ``halley.cpi_act_sha256`` and ``halley.act_sha256`` (design
sections 9 and 10; the loader's ``_check_actual_cpi``). As an input of the stage it is copied
into the run directory and enters the run id (D21); its ``estimated_rows`` flag, the C9
estimate of the valuation month, enters the manifest. It holds exactly the months -26..0
(:data:`CPI_ACT_FIRST_MONTH`)."""


def _p_rec_covers(v: InputView) -> bool:
    """``p_rec`` covers months ``0..tau0 + H`` for the groups of the run (in-force 5.1, 12.4).
    One answer for the file: the run measures one group (G1), and a one-element result would
    line up with the cpi_act table's key as well as with the group."""
    p = v.scenarios["p_rec"]
    tau0 = np.asarray(v.model_points.tau0, dtype=np.int64)
    return bool(np.all((p.u0 <= 0) & (p.u_max >= tau0 + v.horizon)))


P_REC_COVERS = InputRule(
    "the locked-in curve covers [0, tau0 + H]",
    _p_rec_covers,
    "p_rec, the composed curve at recognition, reaches the run-off of the group's membership "
    "from its recognition date (in-force 5.1, 12.4)",
)
"""The load-time check of in-force 12.4 on the locked-in discount file, which the stage reads as
``p_rec``: the read itself needs only months ``i_acc`` and ``i_rec``, inside ``0..H``."""

CLOSE_OUTPUTS = (
    "p_g",
    "csm_open_per",
    "lc_open_per",
    "cu_act_per",
    "delta_fcf_l",
    "csm_acc",
    "acc_per",
    "lc_alloc_per",
    "lca_per",
    "rev",
    "raw",
    "csm_adj",
    "lc_adj",
    "csm_fs",
    "lc_fs",
    "lc_loss",
    "lc_reversal",
    "n_0",
    "u_per",
    "r_per",
    "csm_0",
    "lc_0",
    "lrc_0",
    "aoc_a",
    "aoc_b",
    "aoc_c",
    "aoc_d",
    "aoc_e",
    "aoc_ba",
    "aoc_cb",
    "aoc_dc",
    "aoc_ed",
)
"""What :class:`InForceGroup` stores beside :data:`ROLL_OUTPUTS`: P_g, the group inputs of the
close (the opening CSM and loss component, ``csm_prev`` and ``lc_prev``, and the period's units
``cu_act``), the change in locked-in fulfilment cash flows, every step of the period close in
the spec's order (in-force 9.2 to 9.6), which with the loss component's losses and reversals
(``lc_fs`` split by sign) are every line of the period's roll-forward tables (11.1), and the
analysis of change of the fulfilment cash flows, five values and four differences (11.2)."""

AOC_LABELS = (
    "aoc_a: A, the previous run's FCF at its t = 0; FCF at recognition in the first period",
    "aoc_b: B = F_prev(tau), the previous run's FCF at V, on its own basis and curve",
    "aoc_c: C = FCF_g^L_old(0), at V on the locked-in financial inputs, the previous tables and "
    "the expected in-force",
    "aoc_d: D = FCF_g^L_new(0), at V on the locked-in financial inputs, the current tables and "
    "the actual in-force",
    "aoc_e: E = FCF_g^cur(0), at V on the current basis: the balance sheet",
    "aoc_ba: B - A, the previous run's expected movement over the period",
    "aoc_cb: C - B, CPI experience over the period on the future cash flows (IFIE, B128(a))",
    "aoc_dc: D - C, delta_FCF_L, which adjusts the CSM or the loss component (paras 44(c), 48, "
    "50(b))",
    "aoc_ed: E - D, the change in financial assumptions at V (IFIE, B97(a), para 87)",
)
"""The labels of the analysis of change (in-force 11.2), which the manifest's disclosures carry
under ``analysis_of_change``: reported, not interpreted, for the revenue and IFIE of M12."""

NOTE_PERIOD_RELEASE = (
    "The period close releases the adjusted CSM once for the reporting period, in proportion to "
    "the coverage units actually provided in it (cu_act, undiscounted) against those still to be "
    "provided at V (N(0)); one release per period differs from twelve monthly steps. The months "
    "after V are allocated monthly (in-force spec, C12 and open question 2)."
)
"""The in-force stage's note on the CSM allocation step, in place of :data:`NOTE_MONTHLY_STEP`,
which says that a period's release is the sum of its monthly releases: true of the months after
V, not of the period the stage closes."""


def in_force_notes(statics: Mapping[str, object]) -> tuple[str, ...]:
    """The notes of an in-force stage's manifest (design section 9, "Outputs"): the period's
    single release with the monthly step after V (:data:`NOTE_PERIOD_RELEASE`), the ignored
    timing convexity, the coverage-unit approach the stage was built with, the C11 period
    allocation of the loss component and the C17 locked-in runs."""
    _monthly, convexity, units = roll_notes(statics)
    return (NOTE_PERIOD_RELEASE, convexity, units, NOTE_PERIOD_ALLOCATION, NOTE_LOCKED_IN_RUNS)


def _k1_rhs(a: BatchView) -> NDArray[Any]:
    """csm_prev + acc_per + csm_fs - R_per, left to right: the opening line and the movements
    of the period's CSM table."""
    return ((a.var("csm_open_per") + a.var("acc_per")) + a.var("csm_fs")) - a.var("r_per")


def _k2_rhs(a: BatchView) -> NDArray[Any]:
    """lc_prev - lca_per + lc_fs, left to right: the opening line and the movements of the
    period's loss-component table."""
    return (a.var("lc_open_per") - a.var("lca_per")) + a.var("lc_fs")


def _k3_lhs(a: BatchView) -> NDArray[Any]:
    return a.var("csm_adj") - a.var("lc_adj")


def _k3_rhs(a: BatchView) -> NDArray[Any]:
    """(CSM_acc - LC_alloc) - delta_FCF_L."""
    return (a.var("csm_acc") - a.var("lc_alloc_per")) - a.var("delta_fcf_l")


def _k5_lhs(a: BatchView) -> NDArray[Any]:
    """R_per / cu_act, guarded where cu_act = 0."""
    return a.var("r_per") / _guard(a.var("cu_act_per"))


def _k5_rhs(a: BatchView) -> NDArray[Any]:
    """CSM(0) / N(0), guarded where N(0) = 0."""
    return a.var("csm_0") / _guard(a.var("n_0"))


def _k5_where(a: BatchView) -> NDArray[np.bool_]:
    return np.asarray((a.var("cu_act_per") > 0) & (a.var("csm_0") > 0) & (a.var("n_0") > 0))


CLOSE_INVARIANTS = (
    Invariant(
        "K1 CSM(0) = csm_prev + acc_per + csm_fs - R_per",
        lambda a: a.var("csm_0"),
        _k1_rhs,
        tol=EPS,
        scale=_p_g,
        reads=("csm_0", "csm_open_per", "acc_per", "csm_fs", "r_per", "p_g"),
        doc="the lines of the period's CSM roll-forward sum to the closing CSM (in-force 11.1)",
    ),
    Invariant(
        "K2 LC(0) = lc_prev - lca_per + lc_fs",
        lambda a: a.var("lc_0"),
        _k2_rhs,
        tol=EPS,
        scale=_p_g,
        reads=("lc_0", "lc_open_per", "lca_per", "lc_fs", "p_g"),
        doc="the lines of the period's loss-component roll-forward sum to the closing LC",
    ),
    Invariant(
        "K3 CSM_adj - LC_adj = (CSM_acc - LC_alloc) - delta_FCF_L",
        _k3_lhs,
        _k3_rhs,
        tol=EPS,
        scale=_p_g,
        reads=("csm_adj", "lc_adj", "csm_acc", "lc_alloc_per", "delta_fcf_l", "p_g"),
        doc="the remeasurement rules conserve CSM - LC less the change (in-force 9.4)",
    ),
    Invariant(
        "K4 CSM_acc >= 0", lambda a: a.var("csm_acc"), 0.0, kind="ge", tol=0, reads=("csm_acc",)
    ),
    Invariant(
        "K4 LC_alloc >= 0",
        lambda a: a.var("lc_alloc_per"),
        0.0,
        kind="ge",
        tol=0,
        reads=("lc_alloc_per",),
    ),
    Invariant(
        "K4 CSM_adj >= 0", lambda a: a.var("csm_adj"), 0.0, kind="ge", tol=0, reads=("csm_adj",)
    ),
    Invariant(
        "K4 LC_adj >= 0", lambda a: a.var("lc_adj"), 0.0, kind="ge", tol=0, reads=("lc_adj",)
    ),
    Invariant("K4 CSM(0) >= 0", lambda a: a.var("csm_0"), 0.0, kind="ge", tol=0, reads=("csm_0",)),
    Invariant("K4 LC(0) >= 0", lambda a: a.var("lc_0"), 0.0, kind="ge", tol=0, reads=("lc_0",)),
    Invariant(
        "K4 min(CSM_adj, LC_adj) = 0",
        lambda a: np.minimum(a.var("csm_adj"), a.var("lc_adj")),
        0.0,
        tol=0,
        reads=("csm_adj", "lc_adj"),
        doc="at most one of CSM_adj and LC_adj is positive",
    ),
    Invariant(
        "K4 min(CSM(0), LC(0)) = 0",
        lambda a: np.minimum(a.var("csm_0"), a.var("lc_0")),
        0.0,
        tol=0,
        reads=("csm_0", "lc_0"),
        doc="at most one of CSM(0) and LC(0) is positive",
    ),
    Invariant(
        "K5 R_per / cu_act = CSM(0) / N(0)",
        _k5_lhs,
        _k5_rhs,
        tol=EPS,
        scale=_rel,
        where=_k5_where,
        reads=("r_per", "cu_act_per", "csm_0", "n_0"),
        doc="equal allocation at the period level, where cu_act > 0 and CSM(0) > 0 (B119)",
    ),
    Invariant(
        "K6 CSM(t) >= 0",
        lambda a: a.var("csm"),
        0.0,
        kind="ge",
        tol=0,
        months=(1, "H"),
        reads=("csm",),
    ),
    Invariant(
        "K6 LC(t) >= 0",
        lambda a: a.var("lc"),
        0.0,
        kind="ge",
        tol=0,
        months=(1, "H"),
        reads=("lc",),
    ),
    Invariant(
        "K7 LC_alloc = LC^prev(tau)",
        lambda a: a.var("lc_alloc_per"),
        lambda a: a.mp.lc_prev_tau,
        tol=EPS,
        scale=_p_g,
        reads=("lc_alloc_per", "p_g"),
        doc="the period allocation is the loss component the previous run projected for V, its "
        "monthly lc(tau) (in-force 9.3, C11)",
    ),
)
"""The in-force spec's K1 to K7 (section 12.2) with its scales (P_g; ``max(|lhs|, |rhs|)`` for
K5) at its ``eps = 1e-12``, K4's inequalities and "at most one positive" exactly; checked on
every run beside :data:`ROLL_INVARIANTS` (I11 to I13, M1, M2) on the months after V."""


class InForceGroup(CsmRollForward):
    """One reporting period's measurement of a group in force (milestone M11; in-force spec,
    sections 3 to 11): the period close of section 9 in its order and grouping, then the months
    after V (section 10) on the ``cur`` basis from the closing balances.

    Bases ``old`` (the locked-in runs on the previous non-financial tables and the expected
    in-force), ``new`` (the locked-in runs on the current tables and the actual in-force) and
    ``cur`` (the balance sheet's runs, whose units give ``N``); the group fields of a group
    measured before and its locked-in basis by hash (``group_schema(previous=True,
    locked_in=True)``), among them the fields the loader derives from the previous run named by
    ``prev_run_id`` and from the dates: ``f_prev_0``, ``f_prev_tau``, ``lc_prev_tau``,
    ``p_g_prev``, ``tau0``, ``tau``, ``i_acc = tau0 - tau``, ``i_rec = tau0``. The stage's own
    curves are the locked-in curve re-based to V ``p_l``, the composed curve at recognition
    ``p_rec`` and the current curve ``df_cur``; ``cpi_act`` is read for the provenance of the
    locked-in runs' inputs. P_g is carried forward from recognition (``p_g_prev``), the scale of
    every K, M1 and M2. The close reads the group's inputs once each, through the lines it
    stores (``csm_open_per``, ``lc_open_per``, ``cu_act_per``) and ``fcf_prev_0``,
    ``fcf_prev_tau``, so that K1 and K2 sum the stored lines of the tables of in-force 11.1 and
    K5 divides by the units the release used. The loader holds the old basis to the previous
    run, the new basis's model points to the cur basis's and every locked-in file to the group's
    basis and tau0 (``rebuilds_locked_in``)."""

    model_point = group_schema(previous=True, locked_in=True)
    bases = (
        Basis(
            "old",
            doc="the locked-in runs on the previous tables and the expected in-force (5.5)",
        ),
        Basis("new", doc="the locked-in runs on the current tables and the actual in-force"),
        Basis("cur", doc="the current-basis runs at the valuation date, one per class"),
    )
    series = (
        *CsmRollForward.series,
        GroupSum("fcf_old", "old", ("bel", "ra"), "Money", start=0, doc="FCF_g^L_old(t)"),
        GroupSum("fcf_new", "new", ("bel", "ra"), "Money", start=0, doc="FCF_g^L_new(t)"),
    )
    scenario_inputs = (P_L, DF_CUR, P_REC)
    assumption_tables = (CPI_ACT,)
    input_rules = (P_REC_COVERS,)
    outputs: ClassVar[tuple[str, ...] | None] = (*ROLL_OUTPUTS, *CLOSE_OUTPUTS)
    invariants = (*ROLL_INVARIANTS, *CLOSE_INVARIANTS)
    disclosures: ClassVar[dict[str, Disclosure]] = {
        "discount_rates": DISCOUNT_RATES,
        "risk_adjustment": RISK_ADJUSTMENT,
        "notes": in_force_notes,  # by discount_coverage_units, as built
        "analysis_of_change": AOC_LABELS,
    }
    rebuilds_locked_in: ClassVar[bool] = True  # p_l, p_rec, the locked-in runs' curve and CPI

    @var
    def p_g(self) -> Money:
        """P_g, the group's premiums at recognition, carried forward by every run."""
        return self.group.p_g_prev

    # ---- the group inputs of the close: the opening lines of 11.1, the period's units and
    # ---- the previous run's FCF (3.3, 3.4), each read once from the group ----------------
    @var
    def csm_open_per(self) -> Money:
        """csm_prev, the closing CSM at V_prev: the opening line of the period's CSM table."""
        return self.group.csm_prev

    @var
    def lc_open_per(self) -> Money:
        """lc_prev, the closing loss component at V_prev: the opening line of its table."""
        return self.group.lc_prev

    @var
    def cu_act_per(self) -> CoverageUnits:
        """cu_act, the coverage units actually provided in the period, undiscounted (9.5)."""
        return self.group.cu_act

    @var
    def fcf_prev_0(self) -> Money:
        """F_prev(0), the previous run's FCF at its own t = 0, V_prev (3.4)."""
        return self.group.f_prev_0

    @var
    def fcf_prev_tau(self) -> Money:
        """F_prev(tau), the previous run's FCF at V, on its own basis and curve (3.4)."""
        return self.group.f_prev_tau

    # ---- section 8: the change in locked-in fulfilment cash flows ---------------------
    @var
    def delta_fcf_l(self) -> Money:
        """delta_FCF_L = FCF_g^L_new(0) - FCF_g^L_old(0) (paras 44(c), B96(b), B72(c))."""
        return self.series.fcf_new.at(0) - self.series.fcf_old.at(0)

    # ---- 9.2 step 1: accretion -----------------------------------------------------------
    @var
    def csm_acc(self) -> Money:
        """CSM_acc = csm_prev x P_rec(tau0 - tau) / P_rec(tau0), left to right as the spec
        writes it: the opening CSM at the locked-in forward rates of the period (para 44(b))."""
        p_rec = self.scen.p_rec
        return (self.csm_open_per() * p_rec.at(self.group.i_acc)) / p_rec.at(self.group.i_rec)

    @var
    def acc_per(self) -> Money:
        """The period's interest accretion, a line of the CSM roll-forward."""
        return self.csm_acc() - self.csm_open_per()

    # ---- 9.3 step 2: the systematic allocation of the loss component --------------------
    @var
    def lc_alloc_per(self) -> Money:
        """LC_alloc = lc_prev x F_prev(tau) / F_prev(0), and 0 where F_prev(0) = 0 (paras
        50(a), 51; C11)."""
        f0 = self.fcf_prev_0()
        ratio = self.fcf_prev_tau() / where(f0 > 0, f0, 1.0)
        return where(f0 > 0, self.lc_open_per() * ratio, 0.0)

    @var
    def lca_per(self) -> Money:
        """The period's systematic allocation, a line of the loss-component roll-forward."""
        return self.lc_open_per() - self.lc_alloc_per()

    # ---- 9.4 step 3: the remeasurement rules ---------------------------------------------
    @var
    def rev(self) -> Money:
        """min(LC_alloc, -delta_FCF_L), the reversal of the second rule (paras 50(b),
        44(c)(ii)); read only where that rule applies."""
        return minimum(self.lc_alloc_per(), -self.delta_fcf_l())

    @var
    def raw(self) -> Money:
        """CSM_acc - delta_FCF_L, the third rule before the para 48(a) floor; read only where
        that rule applies."""
        return self.csm_acc() - self.delta_fcf_l()

    @var
    def csm_adj(self) -> Money:
        """CSM_adj by the three rules of in-force 9.4: CSM_acc where LC_alloc > 0 and the change
        is not favourable; CSM_acc plus what the change reverses beyond the loss component where
        it is; otherwise CSM_acc - delta_FCF_L, floored at 0 (para 48(a))."""
        lc, delta = self.lc_alloc_per(), self.delta_fcf_l()
        loss_rules = where(delta >= 0, self.csm_acc(), self.csm_acc() + (-delta - self.rev()))
        return where(lc > 0, loss_rules, where(self.raw() < 0, 0.0, self.raw()))

    @var
    def lc_adj(self) -> Money:
        """LC_adj by the same rules: the change added to the loss component, or the reversal
        taken from it, or, without a loss component, the excess of the change over CSM_acc."""
        lc, delta = self.lc_alloc_per(), self.delta_fcf_l()
        loss_rules = where(delta >= 0, lc + delta, lc - self.rev())
        return where(lc > 0, loss_rules, where(self.raw() < 0, -self.raw(), 0.0))

    @var
    def csm_fs(self) -> Money:
        """The CSM's changes for future service, a line of its roll-forward (44(c), B96)."""
        return self.csm_adj() - self.csm_acc()

    @var
    def lc_fs(self) -> Money:
        """The loss component's losses (> 0, para 48(a)) or reversals (< 0, para 50(b))."""
        return self.lc_adj() - self.lc_alloc_per()

    @var
    def lc_loss(self) -> Money:
        """Losses on remeasurement: lc_fs where positive, a line of the LC table (48(a))."""
        return where(self.lc_fs() > 0, self.lc_fs(), 0.0)

    @var
    def lc_reversal(self) -> Money:
        """Reversals of losses: lc_fs where negative, a line of the LC table (50(b))."""
        return where(self.lc_fs() < 0, self.lc_fs(), 0.0)

    # ---- 9.5 step 4: release -------------------------------------------------------------
    @var
    def n_0(self) -> CoverageUnits:
        """N(0): the units of the months after V, from the cur basis on the locked-in curve."""
        return at(self.cu_next, 0)

    @var
    def u_per(self) -> CoverageUnits:
        """The units of the period, actually provided and undiscounted, and those to come."""
        return self.cu_act_per() + self.n_0()

    @var
    def r_per(self) -> Money:
        """R_per = CSM_adj x cu_act / (cu_act + N(0)), the whole CSM_adj where no coverage
        remains (para 44(e), B119; Method 1 at the period level)."""
        u = self.u_per()
        share = self.cu_act_per() / where(u > 0, u, 1.0)
        return where(u > 0, self.csm_adj() * share, self.csm_adj())

    # ---- 9.6 step 5: the closing balances, the opening balances of the months after V ------
    @var
    def csm_0(self) -> Money:
        """CSM(0) = CSM_adj - R_per."""
        return self.csm_adj() - self.r_per()

    @var
    def lc_0(self) -> Money:
        """LC(0) = LC_adj."""
        return self.lc_adj()

    # ---- 11.2: the analysis of change of the fulfilment cash flows -----------------------
    @var
    def aoc_a(self) -> Money:
        """A: the previous run's FCF at its t = 0; FCF at recognition in the first period."""
        return self.fcf_prev_0()

    @var
    def aoc_b(self) -> Money:
        """B = F_prev(tau): the previous run's FCF at V."""
        return self.fcf_prev_tau()

    @var
    def aoc_c(self) -> Money:
        """C = FCF_g^L_old(0)."""
        return self.series.fcf_old.at(0)

    @var
    def aoc_d(self) -> Money:
        """D = FCF_g^L_new(0)."""
        return self.series.fcf_new.at(0)

    @var
    def aoc_e(self) -> Money:
        """E = FCF_g^cur(0), the balance sheet."""
        return at(self.fcf_g, 0)

    @var
    def aoc_ba(self) -> Money:
        """B - A: the previous run's expected movement over the period."""
        return self.aoc_b() - self.aoc_a()

    @var
    def aoc_cb(self) -> Money:
        """C - B: CPI experience over the period, on the locked-in curve (IFIE)."""
        return self.aoc_c() - self.aoc_b()

    @var
    def aoc_dc(self) -> Money:
        """D - C: delta_FCF_L."""
        return self.aoc_d() - self.aoc_c()

    @var
    def aoc_ed(self) -> Money:
        """E - D: the change in financial assumptions at V (IFIE)."""
        return self.aoc_e() - self.aoc_d()


__all__ = [
    "AOC_LABELS",
    "CLOSE_INVARIANTS",
    "CLOSE_OUTPUTS",
    "CPI_ACT",
    "CPI_ACT_FIRST_MONTH",
    "DF_CUR",
    "DISCOUNT_RATES",
    "G3",
    "G5",
    "I14",
    "NOTE_LOCKED_IN_RUNS",
    "NOTE_MONTHLY_STEP",
    "NOTE_PERIOD_ALLOCATION",
    "NOTE_PERIOD_RELEASE",
    "NOTE_TIMING_CONVEXITY",
    "NOTE_UNITS_DISCOUNTED",
    "NOTE_UNITS_UNDISCOUNTED",
    "P_L",
    "P_REC",
    "P_REC_COVERS",
    "RISK_ADJUSTMENT",
    "ROLL_INVARIANTS",
    "ROLL_OUTPUTS",
    "CsmRollForward",
    "InForceGroup",
    "NewGroup",
    "in_force_notes",
    "roll_notes",
]
