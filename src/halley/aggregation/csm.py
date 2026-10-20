"""The contractual service margin of one group: the monthly roll-forward and the recognition of a
group new at the valuation date (design section 9; annuity spec, "Aggregation stage: CSM for one
group").

:class:`CsmRollForward` is the monthly roll-forward the annuity spec writes in evaluation order,
"Accretion and release" and "Onerous groups and the loss component": the locked-in one-month
rate and discount factor from the locked-in curve ``p_l``, the coverage units still to be
provided ``N(t)`` by a backward recursion (discounted on the locked-in curve by default, Method 1
of the 2022 agenda decision, or undiscounted by the build-time switch), the accretion, the
release in proportion to the month's units, and the loss component by the ratio of
IFRS 17 Illustrative Example 8. Every division sits inside a ``where`` whose divisor is guarded,
because both branches are evaluated. The opening balances are hooks: ``csm_0`` and ``lc_0`` (the
inits of ``csm`` and ``lc``) and ``p_g``, which a subclass defines.

:class:`NewGroup` is the stage of milestone M7: a group recognised at the valuation date, whose
``P_g`` is the sum of its members' premiums and whose opening CSM or loss component is
``max(0, P_g - FCF_g(0))`` or ``max(0, FCF_g(0) - P_g)``, with ``FCF_g = BEL_g + RA_g`` formed
first (paras 38 and 47). Its member rules G5 and G3 check every contract on the runs the
stage reads; its invariants I11 to I14 are checked on every run.
"""

from __future__ import annotations

from typing import Any, ClassVar, cast

import numpy as np
from numpy.typing import NDArray

from halley.aggregation.stage import Aggregation, Basis, GroupSum, MemberRule, MemberView
from halley.authoring import (
    BatchView,
    CoverageUnits,
    Factor,
    Int,
    Invariant,
    Money,
    Monthly,
    Rate,
    ScenarioInput,
    ScenarioRule,
    Static,
    fwd_rate,
    var,
    where,
)
from halley.authoring.contracts import group_schema
from halley.errors import LoadError

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
runs it reads (annuity spec, "Locked-in rates")."""

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
    "acc",
    "csm_pre",
    "release",
    "csm",
    "lc",
    "lc_alloc",
    "lrc",
)
"""What :class:`CsmRollForward` stores: the group series it reads and every line of the
roll-forward tables of the annuity spec, by month."""

DISCOUNT_RATES = (
    "Bottom-up (B80): a liquid risk-free curve plus an illiquidity premium, continuously "
    "compounded, composed as P(t) = rf_df(t) x exp(-ilp x t / 12). The fulfilment cash flows are "
    "discounted at the current composed curve; the CSM accretes at the forward rates of the "
    "composed curve at the group's initial recognition, re-based to the valuation date (B72(b)), "
    "and the coverage units are discounted on that locked-in curve."
)
"""The approach to discount rates (para 117(c)(iii)), annuity spec "Discounting"."""

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
)
"""The annuity spec's I11, I12 and I13 ("Built-in invariants"), each part an Invariant; I11's
exact zero covers the last month with units itself (where N is exactly 0, so the release is the
whole CSM), which the literal "after" leaves vacuous when that month is H (in-force open
question 17)."""


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
    formed first) exactly when its ``profit_bucket`` is ONEROUS."""
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


class CsmRollForward(Aggregation):
    """The monthly CSM and loss-component roll-forward of one group on the ``cur`` basis
    (annuity spec, "Accretion and release", "Onerous groups and the loss component"), in the
    spec's evaluation order and grouping. A subclass defines the hooks ``csm_0``, ``lc_0`` (the
    opening balances, the inits of ``csm`` and ``lc``) and ``p_g`` (the premiums of the group,
    the K scale of the in-force close)."""

    abstract = True

    discount_coverage_units: Static[bool] = True  # Method 1 of the 2022 agenda decision

    model_point = group_schema()
    bases = (Basis("cur", doc="the projection runs at the valuation date, one per class"),)
    series = (
        GroupSum("cu_g", "cur", "cu", "CoverageUnits", doc="CU_g(t), t = 1..H"),
        GroupSum("fcf_g", "cur", ("bel", "ra"), "Money", start=0, doc="FCF_g = BEL_g + RA_g"),
        GroupSum("bel_g", "cur", "bel", "Money", start=0, doc="BEL_g(t), t = 0..H"),
        GroupSum("ra_g", "cur", "ra", "Money", start=0, doc="RA_g(t), t = 0..H"),
    )
    scenario_inputs = (P_L,)
    outputs: ClassVar[tuple[str, ...] | None] = ROLL_OUTPUTS
    invariants = ROLL_INVARIANTS
    disclosures: ClassVar[dict[str, str | tuple[str, ...]]] = {
        "discount_rates": DISCOUNT_RATES,
        "risk_adjustment": RISK_ADJUSTMENT,
        "notes": (NOTE_MONTHLY_STEP, NOTE_TIMING_CONVEXITY),
    }

    # ---- hooks: the opening balances and the premiums --------------------------------
    @var
    def csm_0(self) -> Money:
        """CSM(0), the init of ``csm``: a subclass defines it."""
        raise NotImplementedError("csm_0 is a hook of CsmRollForward; a subclass defines it")

    @var
    def lc_0(self) -> Money:
        """LC(0), the init of ``lc``: a subclass defines it."""
        raise NotImplementedError("lc_0 is a hook of CsmRollForward; a subclass defines it")

    @var
    def p_g(self) -> Money:
        """P_g, the group's premiums: a subclass defines it."""
        raise NotImplementedError("p_g is a hook of CsmRollForward; a subclass defines it")

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


class NewGroup(CsmRollForward):
    """A group recognised at the valuation date (milestone M7; annuity spec, "Initial
    recognition"): every member was issued on it (G5), the premiums P_g are the members' and
    the opening CSM or loss component is the gap between P_g and FCF_g(0), with the FCF formed
    first. G3 holds for every contract (the member rule at load, I14 on the group's figures)."""

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


__all__ = [
    "DISCOUNT_RATES",
    "G3",
    "G5",
    "I14",
    "NOTE_LOCKED_IN_RUNS",
    "NOTE_MONTHLY_STEP",
    "NOTE_PERIOD_ALLOCATION",
    "NOTE_TIMING_CONVEXITY",
    "P_L",
    "RISK_ADJUSTMENT",
    "ROLL_INVARIANTS",
    "ROLL_OUTPUTS",
    "CsmRollForward",
    "NewGroup",
]
