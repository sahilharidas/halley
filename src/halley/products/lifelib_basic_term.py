"""lifelib's BasicTerm_ME in Halley: reference case 2 (D23).

A reproduction of ``basiclife/BasicTerm_ME`` from lifelib (https://lifelib.io, MIT licence,
version 0.17.2) from lifelib's own sample inputs: its 10,000 model points, its mortality table by
age and duration, its annual spot discount curve and its premium rate table. Every variable keeps
lifelib's cell name and lifelib's definition; the test ``tests/test_lifelib_basic_term.py``
reconciles the monthly cash flows, the per-policy present values and the portfolio totals to
lifelib's own results. Nothing here depends on reference case 1; the term assurance spec's
section 15.5 lists where that model deliberately differs from this one.

**Conventions encoded, and where lifelib's results had to be read differently**

1. *Month index.* lifelib's month ``t`` runs from time ``t`` to ``t + 1`` and its flows carry
   the index of the month's start; Halley's month ``t`` runs from ``t - 1`` to ``t``. So a flow
   Halley reports at ``t`` is lifelib's flow at ``t - 1`` (premiums, claims, expenses,
   commissions, net_cf, pols_death, pols_lapse, and the rates mort_rate_mth and lapse_rate that
   month uses). Quantities at an instant are not shifted: pols_if and the other counts,
   duration_mth, duration, age, disc_factors and inflation_factor agree index for index.
2. *Timing.* lifelib values every flow of month ``t`` at its start and discounts it with
   ``disc_factors(t)``. Here the flows declare ``timing="start"`` and the present values use
   ``disc_factors(t - 1)``, the factor at the start of Halley's month ``t``.
3. *Discounting.* A spot curve by projection year: ``disc_rate_mth(t)`` is
   ``(1 + zero_spot[t // 12]) ** (1/12) - 1`` and ``disc_factors(t)`` is
   ``(1 + disc_rate_mth(t)) ** (-t)``, lifelib's two formulas with the same operations.
4. *In-force counts.* lifelib's ``pols_if(t)`` is the count at time ``t`` before maturity
   (``pols_if_at(t, "BEF_MAT")``); maturity, then new business, then the month's deaths and
   lapses follow, in that order. The three counts are ``pols_if``, ``pols_if_bef_nb`` and
   ``pols_if_bef_decr`` here, exactly lifelib's ``pols_if_at`` timings. Deaths are taken from
   the count before decrements, lapses from the survivors of death. The recursion reads
   ``pols_if_bef_decr(t - 1)``, and the checker requires an init value on every variable a
   forward recursion reads at ``t - 1``, although its formula would give the same value at
   ``t = 0``: the init is lifelib's ``pols_if_at(0, "BEF_DECR")`` written from the model point.
5. *New business.* lifelib's ``duration_mth`` is negative for business issued after the
   valuation date: ``policy_count`` enters at the instant ``duration_mth(t) == 0`` with the
   acquisition expense, and ``pols_if_init`` is ``policy_count`` only when ``duration_mth > 0``.
   46 of the sample points are issued at ``t = 0`` and 1,730 later. Halley's built-in in-force
   invariant H1 (never increasing) assumes no new business, so no variable carries
   ``role="in_force"`` or ``role="decrement"``; the reconciliation test checks the equivalent
   properties itself.
6. *Maturity.* At the instant ``duration_mth(t) == 12 * policy_term``, everything in
   ``pols_if(t)`` matures. 31 sample points mature at ``t = 0``. The run horizon is the largest
   ``12 * policy_term - duration_mth`` over the file (276 months), one less than lifelib's
   ``max_proj_len``: lifelib's last month has zero flows for every policy.
7. *Mortality.* ``mort_table[age, min(duration, 5)]``, sex ignored. lifelib looks the pair up
   with ``reindex(fill_value=0)``, so a pair outside the table gives a rate of 0 without an
   error; in the sample that happens only before a future new business policy is issued
   (negative duration), where the count is 0 anyway. Halley never extrapolates a table, so the
   rate is ``where(duration >= 0, lookup, 0)`` with the keys clamped into the table for the
   discarded branch: the clamps never touch a value lifelib reads from the table. An age above
   120 would be a run error here, not a 0 (the sample reaches 101).
8. *Lapse.* lifelib's ``max(0.1 - 0.02 * duration, 0.02)`` is a formula, not a table. Halley's
   unit rules make arithmetic on an ``Int`` a ``Factor``, so ``lapse_rate`` is typed ``Factor``
   rather than ``Prob[Annual]`` and its monthly conversion ``1 - (1 - lapse_rate) ** (1/12)`` is
   written out (``to_monthly_prob`` takes a ``Prob[Annual]``). The operations are lifelib's, so
   the values are bit-identical, including the ``0.020000000000000004`` that
   ``0.1 - 0.02 * 4`` gives at duration 4. (The ``Int`` annotation class has no ``float * Int``
   overload, so that line reads the duration as ``Any`` for pyright; the rule table is fine.)
9. *Premium.* lifelib's ``premium_pp`` is ``round(sum_assured * premium_table[age_at_entry,
   policy_term], 2)``. Halley has no rounding in formulas and no table lookup at load, so the
   rate is joined onto the model points before the run (:func:`premium_rates`, lifelib's
   ``reindex`` on the pair) as the column ``premium_rate``, and ``premium_pp`` is a load-time
   derived field, ``np.around(sum_assured * premium_rate, 2)`` as lifelib computes it.
10. *Inflation.* lifelib's ``(1 + 0.01) ** (t / 12)`` needs a ``Factor ** Factor`` power that
    the unit rules do not have; ``inflation_factor`` is ``(1 + to_monthly(0.01)) ** t``, the
    same value up to last-bit rounding (about 1e-14 relative at the horizon).
11. *Commissions.* lifelib's ``(duration(t) == 0) * premiums(t)`` is ``where(duration == 0,
    premiums, 0)``: a ``Bool`` does not multiply ``Money`` here.
12. *Expenses.* ``300 * pols_new_biz + pols_if_bef_decr * 60 / 12 * inflation_factor`` in
    lifelib's order of operations. lifelib's ``expense_acq``, ``expense_maint`` and
    ``inflation_rate`` constants are variables returning literals here: a literal takes the
    unit its variable declares, which a ``cast`` states for pyright.
13. *Present values.* lifelib sums each flow times ``disc_factors`` as a matrix product (BLAS
    order); ``sum_t`` sums in ascending ``t`` (D19). The per-policy values agree to last-bit
    rounding. ``net_premium_pp`` and ``pv_pols_if`` are lifelib cells that BasicTerm_ME does not
    use by default; they are reproduced too, with lifelib's ``nan_to_num`` written as a
    ``where`` on ``pv_pols_if > 0``. ``loading_prem`` is unused in BasicTerm_ME and is left out.
14. *Inputs.* lifelib's workbooks are committed as Parquet under ``tests/data/lifelib`` with the
    values unchanged; the mortality table is unpivoted to one row per age and duration, the
    premium table's two-level index becomes two key columns. ``policy_id`` is lifelib's integer
    index; ``sex`` is loaded and validated but, as in lifelib, not read. lifelib has no
    valuation date: :data:`VALUATION_DATE` only labels the run manifest.
15. *Reading lifelib's results.* Per-policy present values come from the ``pv_*`` cells,
    counts from ``pols_if_at``, monthly flows from the cells by month, portfolio totals from
    ``result_cf()`` and ``result_pols()``, which sum the policies in index order with Python's
    ``sum``, as Halley's ``portfolio_totals`` does.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, ClassVar, cast

import numpy as np
from numpy.typing import NDArray

from halley.authoring import (
    Annual,
    Column,
    Count,
    Derived,
    Factor,
    Field,
    Int,
    Key,
    LoadContext,
    Money,
    MoneyPerPolicy,
    Monthly,
    Prob,
    Product,
    Rate,
    Schema,
    Table,
    maximum,
    minimum,
    sum_t,
    to_monthly,
    to_monthly_prob,
    var,
    where,
)
from halley.errors import LoadError

VALUATION_DATE = dt.date(2027, 1, 1)
"""lifelib has no valuation date; this one labels the manifest (reference case 1's date)."""

LIFELIB_VERSION = "0.17.2"
"""The lifelib release the fixtures under ``tests/data/lifelib`` were taken from."""

EXPENSE_ACQ = 300.0
"""lifelib's ``expense_acq()``: acquisition expense per new policy."""
EXPENSE_MAINT = 60.0
"""lifelib's ``expense_maint()``: annual maintenance expense per policy in force."""
INFLATION_RATE = 0.01
"""lifelib's ``inflation_rate()``: annual expense inflation."""

# ------------------------------------------------------------------------ model point schema


def _premium_pp(c: LoadContext) -> NDArray[Any]:
    """lifelib's ``premium_pp()``: ``np.around(sum_assured() * prem_rates, 2)``."""
    return np.around(c.sum_assured * c.premium_rate, 2)


def _rem_m(c: LoadContext) -> NDArray[Any]:
    return 12 * c.policy_term - c.duration_mth


def _scale_money(c: LoadContext) -> NDArray[Any]:
    return c.sum_assured * c.policy_count


SCHEMA = Schema(
    fields=(
        Field("policy_id", "int", "Int", unique=True, ge=1, doc="lifelib's point index"),
        Field("age_at_entry", "int", "Int", ge=18, le=120, doc="Age at issue, whole years"),
        Field("sex", "enum", enum=("M", "F"), doc="Loaded as lifelib's input; not read"),
        Field("policy_term", "int", "Int", ge=1, doc="Policy term in years"),
        Field("policy_count", "number", "Count", ge=0, doc="Policies in the model point"),
        Field("sum_assured", "number", "MoneyPerPolicy", gt=0),
        Field(
            "duration_mth",
            "int",
            "Int",
            doc="Months in force at t = 0; 0 or negative for business issued at t = -duration_mth",
        ),
        Field(
            "premium_rate",
            "number",
            "Factor",
            ge=0,
            doc="premium_table[age_at_entry, policy_term], joined before the run (premium_rates)",
        ),
    ),
    derived=(
        Derived("premium_pp", "MoneyPerPolicy", _premium_pp, kind="number", ge=0),
        Derived(
            "rem_m",
            "Int",
            _rem_m,
            ge=0,
            doc="12 * policy_term - duration_mth: months to maturity, lifelib's proj_len - 1",
        ),
        Derived("scale_money", "Money", _scale_money, kind="number", ge=0),
    ),
    horizon="rem_m",
)


def premium_rates(
    age_at_entry: NDArray[np.int64],
    policy_term: NDArray[np.int64],
    table_age_at_entry: NDArray[np.int64],
    table_policy_term: NDArray[np.int64],
    table_premium_rate: NDArray[np.float64],
) -> NDArray[np.float64]:
    """lifelib's ``premium_table.reindex(MultiIndex(age_at_entry, policy_term))``: the premium
    rate per unit of sum assured of each model point, to be loaded as the model point column
    ``premium_rate``. A pair missing from the table is an error (lifelib's reindex would give
    NaN)."""
    rates = {
        (int(a), int(p)): float(r)
        for a, p, r in zip(table_age_at_entry, table_policy_term, table_premium_rate, strict=True)
    }
    out = np.empty(len(age_at_entry), dtype=np.float64)
    for i, (a, p) in enumerate(zip(age_at_entry, policy_term, strict=True)):
        try:
            out[i] = rates[(int(a), int(p))]
        except KeyError:
            raise LoadError(
                f"premium_table has no rate for age_at_entry {int(a)}, policy_term {int(p)}"
            ) from None
    return out


def _pols_if_init(mp: Any) -> Count:
    """lifelib's ``pols_if_init()``: ``policy_count`` where ``duration_mth > 0``, else 0."""
    return where(mp.duration_mth > 0, mp.policy_count, 0)


def _pols_if_bef_decr_init(mp: Any) -> Count:
    """lifelib's ``pols_if_at(0, "BEF_DECR")``: ``pols_if_init() - pols_maturity(0) +
    pols_new_biz(0)``, each term from the model point."""
    maturity_0 = where(mp.duration_mth == mp.policy_term * 12, _pols_if_init(mp), 0)
    new_biz_0 = where(mp.duration_mth == 0, mp.policy_count, 0)
    return _pols_if_init(mp) - maturity_0 + new_biz_0


# ----------------------------------------------------------------- assumption table contracts

MORT_TABLE = Table(
    "mort_table",
    keys=(Key("age", lo=18, hi=120), Key("duration", lo=0, hi=5)),
    columns=(Column("mort_rate", "Prob[Annual]"),),
    doc="lifelib's mort_table: annual mortality by attained age and duration capped at 5",
)
DISC_RATE_ANN = Table(
    "disc_rate_ann",
    keys=(Key("year", lo=0, hi=150),),
    columns=(Column("zero_spot", "Rate[Annual]"),),
    doc="lifelib's disc_rate_ann: annual spot rate by projection year",
)


# ---------------------------------------------------------------------------------- the model


class BasicTerm(Product):
    """lifelib's BasicTerm_ME, cell for cell. Flows at ``t`` are lifelib's at ``t - 1``."""

    model_point = SCHEMA
    assumption_tables = (MORT_TABLE, DISC_RATE_ANN)
    outputs = (
        "pols_if",
        "pols_maturity",
        "pols_new_biz",
        "pols_if_bef_decr",
        "pols_death",
        "pols_lapse",
        "mort_rate_mth",
        "lapse_rate",
        "premiums",
        "claims",
        "expenses",
        "commissions",
        "net_cf",
        "disc_factors",
        "inflation_factor",
        "premium_pp",
        "pv_premiums",
        "pv_claims",
        "pv_expenses",
        "pv_commissions",
        "pv_net_cf",
        "pv_pols_if",
        "net_premium_pp",
    )
    tolerance_scales: ClassVar[dict[str, str]] = {
        "Money": "scale_money",
        "MoneyPerPolicy": "sum_assured",
        "Count": "policy_count",
    }

    # ---- constants ------------------------------------------------------------------
    # A literal takes the unit its variable declares; the cast says so to pyright.
    @var
    def expense_acq(self) -> MoneyPerPolicy:
        return cast(MoneyPerPolicy, EXPENSE_ACQ)

    @var
    def expense_maint(self) -> MoneyPerPolicy:
        return cast(MoneyPerPolicy, EXPENSE_MAINT)

    @var
    def inflation_rate(self) -> Rate[Annual]:
        return cast(Rate[Annual], INFLATION_RATE)

    # ---- durations and age at time t (unshifted) ------------------------------------
    @var(domain="state")
    def duration_mth(self, t: Int) -> Int:
        return self.mp.duration_mth + t

    @var(domain="state")
    def duration(self, t: Int) -> Int:
        return self.duration_mth(t) // 12

    @var(domain="state")
    def age(self, t: Int) -> Int:
        return self.mp.age_at_entry + self.duration(t)

    # ---- counts at time t: maturity, then new business, then the month's decrements ----
    @var(init=_pols_if_init)
    def pols_if(self, t: Int) -> Count:
        """lifelib's pols_if_at(t, "BEF_MAT"): survivors of the month before maturity."""
        return self.pols_if_bef_decr(t - 1) - self.pols_lapse(t) - self.pols_death(t)

    @var(domain="state")
    def pols_maturity(self, t: Int) -> Count:
        return where(self.duration_mth(t) == self.mp.policy_term * 12, self.pols_if(t), 0)

    @var(domain="state")
    def pols_if_bef_nb(self, t: Int) -> Count:
        return self.pols_if(t) - self.pols_maturity(t)

    @var(domain="state")
    def pols_new_biz(self, t: Int) -> Count:
        return where(self.duration_mth(t) == 0, self.mp.policy_count, 0)

    @var(init=_pols_if_bef_decr_init)
    def pols_if_bef_decr(self, t: Int) -> Count:
        """lifelib's pols_if_at(t, "BEF_DECR"). The recursion reads it at t - 1, so the checker
        wants its own init: lifelib's value at t = 0, written from the model point."""
        return self.pols_if_bef_nb(t) + self.pols_new_biz(t)

    # ---- rates of month t: lifelib's rates at t - 1 -----------------------------------
    @var
    def mort_rate(self, t: Int) -> Prob[Annual]:
        """mort_table[age, min(duration, 5)]; 0 before issue, where lifelib's reindex finds no
        row. The clamps keep the discarded branch inside the table (convention 7)."""
        d = self.duration(t - 1)
        age_key = maximum(self.age(t - 1), 18)
        dur_key = minimum(maximum(d, 0), 5)
        return where(d >= 0, self.tables.mort_table[age_key, dur_key], 0)

    @var
    def mort_rate_mth(self, t: Int) -> Prob[Monthly]:
        return to_monthly_prob(self.mort_rate(t))  # 1 - (1 - q) ** (1/12)

    @var
    def lapse_rate(self, t: Int) -> Factor:
        """max(0.1 - 0.02 * duration, 0.02): a Factor, not a Prob[Annual] (convention 8)."""
        d: Any = self.duration(t - 1)  # the rule table makes a float times an Int a Factor
        return maximum(0.1 - 0.02 * d, 0.02)

    @var
    def lapse_rate_mth(self, t: Int) -> Factor:
        return 1 - (1 - self.lapse_rate(t)) ** (1 / 12)

    # ---- decrements of month t --------------------------------------------------------
    @var
    def pols_death(self, t: Int) -> Count:
        return self.pols_if_bef_decr(t - 1) * self.mort_rate_mth(t)

    @var
    def pols_lapse(self, t: Int) -> Count:
        return (self.pols_if_bef_decr(t - 1) - self.pols_death(t)) * self.lapse_rate_mth(t)

    # ---- per-policy amounts -----------------------------------------------------------
    @var
    def claim_pp(self, t: Int) -> MoneyPerPolicy:
        return self.mp.sum_assured

    @var
    def premium_pp(self) -> MoneyPerPolicy:
        """round(sum_assured * premium_table[age_at_entry, policy_term], 2), at load."""
        return self.mp.premium_pp

    @var(domain="state")
    def inflation_factor(self, t: Int) -> Factor:
        return (1 + to_monthly(self.inflation_rate())) ** t  # lifelib: 1.01 ** (t / 12)

    # ---- flows of month t, at its start -----------------------------------------------
    @var(timing="start")
    def premiums(self, t: Int) -> Money:
        return self.premium_pp() * self.pols_if_bef_decr(t - 1)

    @var(timing="start")
    def claims(self, t: Int) -> Money:
        return self.claim_pp(t) * self.pols_death(t)

    @var(timing="start")
    def commissions(self, t: Int) -> Money:
        return where(self.duration(t - 1) == 0, self.premiums(t), 0)

    @var(timing="start")
    def expenses(self, t: Int) -> Money:
        maint = self.pols_if_bef_decr(t - 1) * self.expense_maint() / 12
        return self.expense_acq() * self.pols_new_biz(t - 1) + maint * self.inflation_factor(t - 1)

    @var
    def net_cf(self, t: Int) -> Money:
        return self.premiums(t) - self.claims(t) - self.expenses(t) - self.commissions(t)

    # ---- discounting ------------------------------------------------------------------
    @var(domain="state")
    def disc_rate_mth(self, t: Int) -> Rate[Monthly]:
        return to_monthly(self.tables.disc_rate_ann[t // 12])  # (1 + spot) ** (1/12) - 1

    @var(domain="state")
    def disc_factors(self, t: Int) -> Factor:
        return (1 + self.disc_rate_mth(t)) ** (-t)

    @var
    def pv_premiums_t(self, t: Int) -> Money:
        return self.premiums(t) * self.disc_factors(t - 1)

    @var
    def pv_claims_t(self, t: Int) -> Money:
        return self.claims(t) * self.disc_factors(t - 1)

    @var
    def pv_expenses_t(self, t: Int) -> Money:
        return self.expenses(t) * self.disc_factors(t - 1)

    @var
    def pv_commissions_t(self, t: Int) -> Money:
        return self.commissions(t) * self.disc_factors(t - 1)

    @var
    def pv_pols_if_t(self, t: Int) -> Count:
        return self.pols_if_bef_decr(t - 1) * self.disc_factors(t - 1)

    # ---- per-policy present values --------------------------------------------------
    @var
    def pv_premiums(self) -> Money:
        return sum_t(self.pv_premiums_t)

    @var
    def pv_claims(self) -> Money:
        return sum_t(self.pv_claims_t)

    @var
    def pv_expenses(self) -> Money:
        return sum_t(self.pv_expenses_t)

    @var
    def pv_commissions(self) -> Money:
        return sum_t(self.pv_commissions_t)

    @var
    def pv_net_cf(self) -> Money:
        return self.pv_premiums() - self.pv_claims() - self.pv_expenses() - self.pv_commissions()

    @var
    def pv_pols_if(self) -> Count:
        return sum_t(self.pv_pols_if_t)

    @var
    def net_premium_pp(self) -> MoneyPerPolicy:
        return where(self.pv_pols_if() > 0, self.pv_claims() / self.pv_pols_if(), 0)
