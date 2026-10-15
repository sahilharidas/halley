"""Regenerate the printed numbers of docs/spec/term-assurance.md (Halley reference case 1, v0.1).

Implements the document's formulas exactly as written, in plain float64 scalars with the standard
library only. It prints every computed number the document quotes: the basis checkpoints of
section 6, the dates and derived constants of sections 4 and 13, the tolerance scales of section 1,
the worked example and scalars of section 14, and the magnitudes of the named simplifications in
sections 5, 7.2 and 14.5. It also checks every invariant of section 12 for all ten reference model
points, at their own horizon and in a batch to H = 480.

This is not the reference implementation and not the engine. The spreadsheet is built from the
document's formulas, and the engine reconciles to the spreadsheet (section 1). If a formula in the
document changes, change it here too and rerun:

    python docs/spec/term_assurance_numbers.py
"""

from __future__ import annotations

import math
from datetime import date

# ------------------------------------------------------------------ run setting and basis (6)
VAL_DATE = date(2027, 1, 1)

I_ANN = 0.04  # Rate[Annual], effective
E_ANN = 0.03  # Rate[Annual], effective
I_M = (1 + I_ANN) ** (1 / 12) - 1  # Rate[Monthly]
V_M = (1 + I_ANN) ** (-1 / 12)  # one-month discount factor
E_M = (1 + E_ANN) ** (1 / 12) - 1  # Rate[Monthly]
ONE_INFL = 1 + E_M

GM = {  # sex: (A, B, c)
    "M": (0.0002, 0.000017, 1.1),
    "F": (0.00015, 0.000011, 1.1),
}
AGE_LO, AGE_HI = 18, 120
LAPSE_TABLE = [0.15, 0.10, 0.08, 0.06, 0.05, 0.04]  # key min(dur_y, 5)
COMM_TABLE = [0.20, 0.02]  # key min(dur_y, 1)
MAINT_BASE = 5.00  # Money per policy, time-0 prices, per month
CLAIM_EXP_BASE = 250.00  # Money per policy (per death), time-0 prices
TWELFTH = 1 / 12


def q_gm(sex: str, x: int) -> float:
    a, b, c = GM[sex]
    # order: c^x, B*c^x, *(c-1), /ln(c), A+, negate, exp, 1-
    return 1 - math.exp(-(a + b * c**x * (c - 1) / math.log(c)))


MORT = {s: {x: q_gm(s, x) for x in range(AGE_LO, AGE_HI + 1)} for s in ("M", "F")}


def to_monthly_prob(p: float) -> float:
    return 1 - (1 - p) ** TWELFTH


# ------------------------------------------------------------------ dates (4, 5)
def months_between(d1: date, d2: date) -> int:
    return 12 * (d2.year - d1.year) + (d2.month - d1.month) - (1 if d2.day < d1.day else 0)


def age_last_birthday(birth: date, on: date) -> int:
    return on.year - birth.year - (1 if (on.month, on.day) < (birth.month, birth.day) else 0)


# ------------------------------------------------------------------ model points (3, 13)
def model_point(pid, sex, birth, entry, maturity, prem_term_y, sum_assured, pols_if_init):
    mp = {
        "policy_id": pid,
        "sex": sex,
        "birth_date": birth,
        "entry_date": entry,
        "maturity_date": maturity,
        "prem_term_y": prem_term_y,
        "sum_assured": float(sum_assured),
        "pols_if_init": float(pols_if_init),
        "premium": None,
    }
    assert sex in ("M", "F")
    assert entry <= VAL_DATE
    assert maturity.day == entry.day
    assert maturity > entry
    assert prem_term_y >= 1
    mp["age_at_entry"] = age_last_birthday(birth, entry)
    mp["dur_m"] = months_between(entry, VAL_DATE)
    mp["term_m"] = months_between(entry, maturity)
    mp["rem_m"] = mp["term_m"] - mp["dur_m"]
    mp["prem_term_m"] = 12 * prem_term_y
    mp["prem_rem_m"] = max(mp["prem_term_m"] - mp["dur_m"], 0)
    mp["dur_y_last"] = (mp["term_m"] - 1) // 12
    assert 18 <= mp["age_at_entry"] <= 65, pid
    assert 12 <= mp["term_m"] <= 480, pid
    assert mp["prem_term_m"] <= mp["term_m"], pid
    assert mp["dur_m"] >= 0, pid
    assert mp["rem_m"] >= 1, pid
    assert mp["age_at_entry"] + mp["dur_y_last"] <= AGE_HI, pid
    return mp


D = date
REF = [
    model_point("P01", "M", D(1986, 3, 15), D(2024, 7, 1), D(2044, 7, 1), 20, 250000, 1),
    model_point("P02", "F", D(1991, 11, 2), D(2027, 1, 1), D(2052, 1, 1), 25, 400000, 1),
    model_point("P03", "M", D(1966, 5, 20), D(2007, 1, 15), D(2027, 1, 15), 20, 150000, 1),
    model_point("P04", "F", D(1976, 8, 9), D(2015, 3, 20), D(2045, 3, 20), 20, 300000, 1),
    model_point("P05", "M", D(1975, 4, 18), D(2018, 2, 10), D(2028, 2, 10), 9, 200000, 1),
    model_point("P06", "F", D(2003, 2, 28), D(2026, 11, 10), D(2036, 11, 10), 10, 100000, 1),
    model_point("P07", "M", D(1980, 6, 1), D(2020, 2, 29), D(2040, 2, 29), 20, 500000, 2.5),
    model_point("P08", "M", D(1958, 12, 31), D(2010, 12, 31), D(2030, 12, 31), 10, 120000, 1),
    model_point("P09", "F", D(1996, 10, 31), D(2026, 12, 15), D(2066, 12, 15), 40, 1000000, 1),
    model_point("P10", "M", D(1990, 5, 5), D(2022, 2, 1), D(2032, 2, 1), 5, 200000, 1),
]

PV_COLS = [
    "pv_premiums_t",
    "pv_commission_t",
    "pv_maint_expense_t",
    "pv_death_outgo_t",
    "pv_claim_expense_t",
]
MONTH_COLS = [
    "dur_y",
    "age",
    "q_ann",
    "q_m",
    "w_ann",
    "w_m",
    "comm_rate",
    "infl",
    "deaths",
    "lapses",
    "maturities",
    "pols_if",
    "premium_pp",
    "commission_pp",
    "maint_pp",
    "claim_exp_pp",
    "premium_income",
    "commission",
    "maint_expense",
    "death_outgo",
    "claim_expense",
    "net_cf",
    "net_outgo",
    "v",
    "disc",
    "pv_net_cf_t",
    *PV_COLS,
    "reserve",
    "reserve_pp",
    "reserve_floored",
]


# ------------------------------------------------------------------ projection (5 to 10)
def project(
    sex,
    age_at_entry,
    dur_m,
    rem_m,
    prem_rem_m,
    dur_y_last,
    premium,
    sum_assured,
    pols_if_init,
    H,
    q_over_12=False,
    w_over_12=False,
    mid_claims=False,
):
    """One policy for t = 0 .. H. Month quantities are None at t = 0 (undefined)."""
    r = {c: [None] * (H + 1) for c in MONTH_COLS}
    r["pols_if"][0] = pols_if_init
    r["infl"][0] = 1.0
    r["disc"][0] = 1.0
    for t in range(1, H + 1):
        row = {}
        dur_y = min((dur_m + t - 1) // 12, dur_y_last)
        age = age_at_entry + dur_y
        q_ann = MORT[sex][age]
        in_term = t <= rem_m
        in_prem = t <= prem_rem_m
        if q_over_12:  # named simplification, measured only
            q_m = q_ann / 12 if in_term else 0.0
        else:
            q_m = to_monthly_prob(q_ann) if in_term else 0.0
        w_ann = LAPSE_TABLE[min(dur_y, 5)] if in_prem else 0.0
        w_m = (w_ann / 12) if w_over_12 else to_monthly_prob(w_ann)
        comm_rate = COMM_TABLE[min(dur_y, 1)]
        infl = ONE_INFL**t
        open_ = r["pols_if"][t - 1]
        deaths = open_ * q_m
        lapses = (open_ - deaths) * w_m
        maturities = (open_ - deaths - lapses) if t == rem_m else 0.0
        pols_if = open_ - deaths - lapses - maturities
        premium_pp = premium if in_prem else 0.0
        commission_pp = premium_pp * comm_rate
        maint_pp = MAINT_BASE * infl
        claim_exp_pp = CLAIM_EXP_BASE * infl
        premium_income = open_ * premium_pp
        commission = premium_income * comm_rate
        maint_expense = open_ * maint_pp
        death_outgo = deaths * sum_assured
        claim_expense = deaths * claim_exp_pp
        net_cf = premium_income - commission - maint_expense - death_outgo - claim_expense
        net_outgo = (
            death_outgo + claim_expense + maint_expense - (premium_income - commission) * (1 + I_M)
        )
        v = V_M
        disc = V_M**t
        disc_prev = r["disc"][t - 1]
        d_claim = V_M ** (t - 0.5) if mid_claims else disc  # mid_claims: measured only
        if mid_claims:
            pv_net_cf_t = (
                (premium_income - commission) * disc_prev
                - maint_expense * disc
                - (death_outgo + claim_expense) * d_claim
            )
        else:
            pv_net_cf_t = (premium_income - commission) * disc_prev - (
                maint_expense + death_outgo + claim_expense
            ) * disc
        pv_premiums_t = premium_income * disc_prev
        pv_commission_t = commission * disc_prev
        pv_maint_expense_t = maint_expense * disc
        pv_death_outgo_t = death_outgo * d_claim
        pv_claim_expense_t = claim_expense * d_claim
        row = {
            "dur_y": dur_y,
            "age": age,
            "q_ann": q_ann,
            "q_m": q_m,
            "w_ann": w_ann,
            "w_m": w_m,
            "comm_rate": comm_rate,
            "infl": infl,
            "deaths": deaths,
            "lapses": lapses,
            "maturities": maturities,
            "pols_if": pols_if,
            "premium_pp": premium_pp,
            "commission_pp": commission_pp,
            "maint_pp": maint_pp,
            "claim_exp_pp": claim_exp_pp,
            "premium_income": premium_income,
            "commission": commission,
            "maint_expense": maint_expense,
            "death_outgo": death_outgo,
            "claim_expense": claim_expense,
            "net_cf": net_cf,
            "net_outgo": net_outgo,
            "v": v,
            "disc": disc,
            "pv_net_cf_t": pv_net_cf_t,
            "pv_premiums_t": pv_premiums_t,
            "pv_commission_t": pv_commission_t,
            "pv_maint_expense_t": pv_maint_expense_t,
            "pv_death_outgo_t": pv_death_outgo_t,
            "pv_claim_expense_t": pv_claim_expense_t,
        }
        for name, val in row.items():
            r[name][t] = val
    # backward recursion (10)
    r["reserve"][H] = 0.0
    for t in range(H - 1, -1, -1):
        r["reserve"][t] = (r["reserve"][t + 1] + r["net_outgo"][t + 1]) * V_M
    for t in range(H + 1):
        p = r["pols_if"][t]
        r["reserve_pp"][t] = r["reserve"][t] / p if p > 0 else 0.0
        r["reserve_floored"][t] = r["reserve"][t] if r["reserve"][t] > 0 else 0.0
    # scalars (9.3, 11.2): running sums of the pv_*_t columns in ascending t
    s = dict.fromkeys(
        (
            "pv_premiums",
            "pv_commission",
            "pv_maint_expense",
            "pv_death_outgo",
            "pv_claim_expense",
            "sum_pv_net_cf_t",
            "exits",
        ),
        0.0,
    )
    for t in range(1, H + 1):
        s["pv_premiums"] += r["pv_premiums_t"][t]
        s["pv_commission"] += r["pv_commission_t"][t]
        s["pv_maint_expense"] += r["pv_maint_expense_t"][t]
        s["pv_death_outgo"] += r["pv_death_outgo_t"][t]
        s["pv_claim_expense"] += r["pv_claim_expense_t"][t]
        s["sum_pv_net_cf_t"] += r["pv_net_cf_t"][t]
        s["exits"] += r["deaths"][t] + r["lapses"][t] + r["maturities"][t]
    s["pv_net_cf"] = (
        s["pv_premiums"]
        - s["pv_commission"]
        - s["pv_maint_expense"]
        - s["pv_death_outgo"]
        - s["pv_claim_expense"]
    )
    s["reserve_0"] = r["reserve"][0]
    s["reserve_pp_0"] = r["reserve_pp"][0]
    s["reserve_floored_0"] = r["reserve_floored"][0]
    s["maturities_n"] = r["maturities"][rem_m] if rem_m <= H else 0.0
    r["scalars"] = s
    r["H"] = H
    return r


def project_mp(mp, H=None, **kw):
    return project(
        mp["sex"],
        mp["age_at_entry"],
        mp["dur_m"],
        mp["rem_m"],
        mp["prem_rem_m"],
        mp["dur_y_last"],
        mp["premium"],
        mp["sum_assured"],
        mp["pols_if_init"],
        mp["rem_m"] if H is None else H,
        **kw,
    )


# ------------------------------------------------------------------ pricing rule (outside the model)
def price(mp) -> float:
    """round(1.25 * break-even monthly premium at entry on this basis, 2)."""
    unit = project(
        mp["sex"],
        mp["age_at_entry"],
        0,
        mp["term_m"],
        mp["prem_term_m"],
        mp["dur_y_last"],
        1.0,
        mp["sum_assured"],
        1.0,
        mp["term_m"],
    )
    s = unit["scalars"]
    outgo = s["pv_maint_expense"] + s["pv_death_outgo"] + s["pv_claim_expense"]
    return round(1.25 * outgo / (s["pv_premiums"] - s["pv_commission"]), 2)


for _mp in REF:
    _mp["premium"] = price(_mp)


# ------------------------------------------------------------------ invariants (12)
EPS = 1e-9


def within(a, b, scale):
    return abs(a - b) <= EPS * max(abs(a), abs(b), scale)


def check(mp, r, label):
    H, n = r["H"], mp["rem_m"]
    sm = mp["sum_assured"] * mp["pols_if_init"]  # Money (in-force) scale
    sp = mp["sum_assured"]  # Money per policy scale
    sc = mp["pols_if_init"]  # Count scale
    s = r["scalars"]
    worst = {}

    def rec(k, v):
        worst[k] = max(worst.get(k, 0.0), abs(v))

    def ok(cond, name):
        assert cond, f"{label} {mp['policy_id']}: {name}"

    for t in range(1, H + 1):
        ok(r["pols_if"][t] >= 0 and r["pols_if"][t] <= r["pols_if"][t - 1], "H1")
        ok(0 <= r["q_m"][t] <= 1 and 0 <= r["w_m"][t] <= 1, "H2 range")
        ok(0 <= r["q_ann"][t] <= 1 and 0 <= r["w_ann"][t] <= 1, "H2 range annual")
        dec = r["deaths"][t] + r["lapses"][t] + r["maturities"][t]
        ok(dec <= r["pols_if"][t - 1] + EPS * sc, "H2 bounded")
        rec("H2 overshoot", max(dec - r["pols_if"][t - 1], 0.0))
    ok(r["reserve"][H] == 0.0 and r["reserve"][n] == 0.0 and r["pols_if"][n] == 0.0, "H3")
    for t in range(1, H + 1):  # H4 roll-forward
        lhs = (
            (r["reserve"][t - 1] + r["premium_income"][t] - r["commission"][t]) * (1 + I_M)
            - r["maint_expense"][t]
            - r["death_outgo"][t]
            - r["claim_expense"][t]
        )
        rec("H4", lhs - r["reserve"][t])
        ok(within(lhs, r["reserve"][t], sm), "H4")
    ok(within(r["reserve"][0], -s["pv_net_cf"], sm), "H5a")  # H5a
    rec("H5a", r["reserve"][0] + s["pv_net_cf"])
    for t in range(0, H):  # H5b, inner sum accumulated in ascending s
        acc = 0.0
        for u in range(t + 1, H + 1):
            acc += (r["death_outgo"][u] + r["claim_expense"][u] + r["maint_expense"][u]) * r[
                "disc"
            ][u] - (r["premium_income"][u] - r["commission"][u]) * r["disc"][u - 1]
        lhs = r["reserve"][t] * r["disc"][t]
        rec("H5b", lhs - acc)
        ok(within(lhs, acc, sm), "H5b")
    for t in range(1, H + 1):  # P1
        if t != n:
            ok(r["maturities"][t] == 0.0, "P1")
    ok(r["maturities"][n] == r["pols_if"][n - 1] - r["deaths"][n] - r["lapses"][n], "P1 n")
    for t in range(1, H + 1):  # P2
        if t > mp["prem_rem_m"]:
            ok(r["premium_income"][t] == 0.0 and r["commission"][t] == 0.0, "P2")
            ok(r["w_ann"][t] == 0.0 and r["w_m"][t] == 0.0 and r["lapses"][t] == 0.0, "P2 w")
        else:
            ok(r["premium_income"][t] == r["pols_if"][t - 1] * mp["premium"], "P2 prem")
    ok(sum(1 for t in range(1, H + 1) if r["premium_pp"][t] != 0.0) == mp["prem_rem_m"], "P3")
    for t in range(1, H + 1):  # P4
        ok(AGE_LO <= r["age"][t] <= AGE_HI, "P4 age")
        if t >= 2:
            ok(r["dur_y"][t] - r["dur_y"][t - 1] in (0, 1), "P4 step")
    for t in range(0, n):  # P5 per-policy recursion
        u = t + 1
        rhs = -(r["premium_pp"][u] - r["commission_pp"][u]) + V_M * (
            r["q_m"][u] * (mp["sum_assured"] + r["claim_exp_pp"][u])
            + r["maint_pp"][u]
            + (1 - r["q_m"][u]) * (1 - r["w_m"][u]) * r["reserve_pp"][u]
        )
        rec("P5", r["reserve_pp"][t] - rhs)
        ok(within(r["reserve_pp"][t], rhs, sp), "P5")
    for t in range(0, H + 1):  # P6
        ok(r["reserve_floored"][t] >= 0 and r["reserve_floored"][t] >= r["reserve"][t], "P6")
    for t in range(n + 1, H + 1):  # P7 exact zeros after the term
        for name in (
            "pols_if",
            "deaths",
            "lapses",
            "maturities",
            "q_m",
            "w_ann",
            "w_m",
            "premium_pp",
            "commission_pp",
            "premium_income",
            "commission",
            "maint_expense",
            "death_outgo",
            "claim_expense",
            "net_cf",
            "net_outgo",
            "pv_net_cf_t",
            *PV_COLS,
            "reserve",
            "reserve_pp",
            "reserve_floored",
        ):
            ok(r[name][t] == 0.0, f"P7 {name}")
        ok(r["age"][t] == r["age"][n] and r["q_ann"][t] == r["q_ann"][n], "P7 clamp")
    rec("P8", s["exits"] - mp["pols_if_init"])  # P8 all exit
    ok(within(s["exits"], mp["pols_if_init"], sc), "P8")
    rec("P9", s["pv_net_cf"] - s["sum_pv_net_cf_t"])  # P9
    ok(within(s["pv_net_cf"], s["sum_pv_net_cf_t"], sm), "P9")
    for t in range(1, H + 1):  # P10 unit identities
        ok(r["death_outgo"][t] == r["deaths"][t] * mp["sum_assured"], "P10 death")
        ok(r["premium_income"][t] == r["pols_if"][t - 1] * r["premium_pp"][t], "P10 prem")
        ok(r["maint_expense"][t] == r["pols_if"][t - 1] * r["maint_pp"][t], "P10 maint")
    return worst


def check_batch_independence(mp, own, batch):
    for name in MONTH_COLS:
        for t in range(0, mp["rem_m"] + 1):
            assert own[name][t] == batch[name][t], f"{mp['policy_id']} {name} t={t}"
    for k, v in own["scalars"].items():
        assert v == batch["scalars"][k], f"{mp['policy_id']} {k}"


# ------------------------------------------------------------------ dependent-rate bound (7.2, 14.5)
def dep_death_diff(q_m: float, w_m: float) -> float:
    """Signed difference: the exact dependent monthly death probability under constant forces of
    both decrements, minus the deaths-first rate q_m (section 7.2). It is negative, about
    -q_m * w_m / 2, because deaths-first overstates deaths. Zero when either rate is zero."""
    if q_m <= 0.0 or w_m <= 0.0:
        return 0.0
    mu_d = -math.log(1 - q_m)
    mu_w = -math.log(1 - w_m)
    dep = mu_d / (mu_d + mu_w) * (1 - (1 - q_m) * (1 - w_m))
    return dep - q_m


# ------------------------------------------------------------------ formatting
def g6(x):
    if x is None:
        return "n/a"
    if isinstance(x, int):
        return str(x)
    if x == 0:
        return "0"
    return f"{x:.6g}"


def main():
    print("=== constants (section 6) ===")
    print(f"1/12 = {TWELFTH!r}")
    print(f"i_m = {I_M!r}   1 + i_m = {1 + I_M!r}")
    print(f"v_m = {V_M!r}")
    print(f"v_m*(1+i_m) - 1 = {V_M * (1 + I_M) - 1!r}")
    print(f"e_m = {E_M!r}   1+e_m = {ONE_INFL!r}")
    print(f"(c-1)/ln(c) = {(1.1 - 1) / math.log(1.1)!r}   ln(1.1) = {math.log(1.1)!r}")
    a, b, c = GM["M"]
    cx = c**50
    print("--- chain for (M, 50) ---")
    print(f"c^50 = {cx!r}")
    print(f"B*c^50 = {b * cx!r}")
    print(
        f"B*c^50*(c-1) = {b * cx * (c - 1)!r}   (c - 1 = {c - 1!r}; "
        f"with a typed 0.1 instead: {b * cx * 0.1!r})"
    )
    print(f"B*c^50*(c-1)/ln(c) = {b * cx * (c - 1) / math.log(c)!r}")
    print(f"A + ... = {a + b * cx * (c - 1) / math.log(c)!r}")
    print(f"q_ann(M,50) = {MORT['M'][50]!r}")
    print(f"q_m(M,50) = {to_monthly_prob(MORT['M'][50])!r}")
    print(f"max q: q(M,120) = {MORT['M'][120]!r}   q(F,120) = {MORT['F'][120]!r}")
    print(
        "--- (c - 1) from the c cell versus a typed 0.1, at the 15 digits a spreadsheet shows ---"
    )
    for sx, x in (("M", 50), ("M", 70), ("F", 53)):
        a_, b_, c_ = GM[sx]
        typed = 1 - math.exp(-(a_ + b_ * c_**x * 0.1 / math.log(c_)))
        cell, other = f"{MORT[sx][x]:.15g}", f"{typed:.15g}"
        print(
            f"q_ann({sx},{x}): {cell} from the cell, {other} typed; "
            f"{'differ' if cell != other else 'identical'} at 15 digits"
        )
        if (sx, x) != ("M", 50):  # the spec's two checks must be visible in a spreadsheet
            assert cell != other, (sx, x)
    print("--- mortality checkpoints (8 s.f.) ---")
    for x in (18, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 80, 90, 100, 104, 110, 120):
        qm, qf = MORT["M"][x], MORT["F"][x]
        print(
            f"| {x} | {qm:.8g} | {to_monthly_prob(qm):.8g} | {qf:.8g} | {to_monthly_prob(qf):.8g} |"
        )
    print("--- age last birthday bias, q(x + 1/2) / q(x) - 1 (section 5) ---")
    bias = {}
    for sx in ("M", "F"):
        a_, b_, c_ = GM[sx]
        bias[sx] = {
            x: (1 - math.exp(-(a_ + b_ * c_ ** (x + 0.5) * (c_ - 1) / math.log(c_)))) / MORT[sx][x]
            - 1
            for x in range(18, 105)  # every mortality age a valid model point reaches (3.3)
        }
        shown = (20, 30, 40, 50, 60, 70, 80, 85, 90, 100, 104)
        print(sx, "  ".join(f"{x}: {bias[sx][x]:.2%}" for x in shown))
        peak = max(bias[sx], key=bias[sx].get)
        band = [x for x in range(60, 105) if 0.0465 <= bias[sx][x] < 0.0485]
        print(
            f"  peak {bias[sx][peak]:.3%} at age {peak}; rounds to 4.7% or 4.8% "
            f"from age {band[0]} to {band[-1]}"
        )
    gaps = {x: abs(bias["M"][x] - bias["F"][x]) for x in range(18, 105)}
    close = max(x for x in gaps if all(gaps[y] < 0.002 for y in range(18, x + 1)))
    print(
        f"female within 0.2 points of male up to age {close}; "
        f"gap {gaps[100]:.3%} at 100, {gaps[104]:.3%} at 104"
    )
    print(f"Gompertz limit c^(1/2) - 1 = {1.1**0.5 - 1:.2%}")
    print("--- lapse monthly (10 s.f.) ---")
    print(" | ".join(f"{to_monthly_prob(w):.10g}" for w in LAPSE_TABLE))
    print("--- infl / disc checkpoints (repr) ---")
    for t in (1, 12, 14, 120, 480):
        print(f"infl({t}) = {ONE_INFL**t!r}   disc({t}) = {V_M**t!r}")

    print("\n=== reference model points (section 13) ===")
    for mp in REF:
        print(
            f"| {mp['policy_id']} | {mp['sex']} | {mp['birth_date']} | {mp['entry_date']} | "
            f"{mp['maturity_date']} | {mp['prem_term_y']} | {mp['sum_assured']:g} | "
            f"{mp['premium']:.2f} | {mp['pols_if_init']:g} |"
        )
    print(
        "| id | age_at_entry | dur_m | term_m | rem_m | prem_term_m | prem_rem_m | dur_y_last | dur_y(1) | age(1) | age(rem_m) |"
    )
    for mp in REF:
        d1 = min(mp["dur_m"] // 12, mp["dur_y_last"])
        dn = min((mp["dur_m"] + mp["rem_m"] - 1) // 12, mp["dur_y_last"])
        print(
            f"| {mp['policy_id']} | {mp['age_at_entry']} | {mp['dur_m']} | {mp['term_m']} | "
            f"{mp['rem_m']} | {mp['prem_term_m']} | {mp['prem_rem_m']} | {mp['dur_y_last']} | "
            f"{d1} | {mp['age_at_entry'] + d1} | {mp['age_at_entry'] + dn} |"
        )
    print("--- portfolio tolerance scales (section 1): sum of per-policy scales ---")
    print(
        f"sum of sum_assured * pols_if_init = {sum(mp['sum_assured'] * mp['pols_if_init'] for mp in REF):,.0f}"
    )
    print(f"sum of pols_if_init = {sum(mp['pols_if_init'] for mp in REF)!r}")
    print(f"total monthly rows (sum of rem_m) = {sum(mp['rem_m'] for mp in REF)}")
    print("--- dates and day counts (sections 4.1, 4.3, 13, 17) ---")

    def date_t(t):  # section 4.1: first of the month t months after the valuation date
        k = VAL_DATE.year * 12 + VAL_DATE.month - 1 + t
        return date(k // 12, k % 12 + 1, 1)

    print("date(t): " + ", ".join(f"t = {t}: {date_t(t)}" for t in (1, 14, 119, 480)))
    by_id = {mp["policy_id"]: mp for mp in REF}
    p06, p08, p09, p10 = by_id["P06"], by_id["P08"], by_id["P09"], by_id["P10"]
    e06 = date.fromisoformat(str(p06["entry_date"]))
    boundary = date(
        e06.year + (e06.month - 1 + p06["dur_m"]) // 12,
        (e06.month - 1 + p06["dur_m"]) % 12 + 1,
        e06.day,
    )
    print(
        f"P06: in force {p06['dur_m']} month and {(VAL_DATE - boundary).days} days at the "
        f"valuation date; projected maturity date({p06['rem_m']}) = {date_t(p06['rem_m'])}, "
        f"{(date_t(p06['rem_m']) - date.fromisoformat(str(p06['maturity_date']))).days} days "
        f"after maturity_date"
    )
    print(
        f"P09: entered {(VAL_DATE - date.fromisoformat(str(p09['entry_date']))).days} days "
        f"before the valuation date, dur_m = {p09['dur_m']}"
    )
    unclamped = p08["prem_term_m"] - p08["dur_m"]
    print(f"P08: unclamped prem_term_m - dur_m = {unclamped} ({-unclamped // 12} years ago)")
    print(
        f"P10: prem_rem_m = {p10['prem_rem_m']}, then {p10['rem_m'] - p10['prem_rem_m']} "
        f"paid-up months"
    )

    HB = max(mp["rem_m"] for mp in REF)
    runs, worst_all = {}, {}
    for mp in REF:
        own = project_mp(mp)
        batch = project_mp(mp, H=HB)
        w1, w2 = check(mp, own, "own"), check(mp, batch, "batch")
        check_batch_independence(mp, own, batch)
        for k in set(w1) | set(w2):
            worst_all[k] = max(worst_all.get(k, 0.0), w1.get(k, 0.0), w2.get(k, 0.0))
        runs[mp["policy_id"]] = own
    print(f"\nbatch horizon H = {HB}")
    print("all invariants H1-H5 and P1-P10 passed for every policy at own horizon and at H =", HB)
    print("P11 batch independence: bit-identical for every policy")
    print("worst absolute residuals:", {k: f"{v:.2e}" for k, v in sorted(worst_all.items())})

    mp = REF[4]
    r = runs["P05"]
    n = mp["rem_m"]
    print("\n=== worked example P05 (section 14) ===")
    print(
        f"premium = {mp['premium']}  age_at_entry={mp['age_at_entry']} dur_m={mp['dur_m']} "
        f"term_m={mp['term_m']} rem_m={n} prem_term_m={mp['prem_term_m']} "
        f"prem_rem_m={mp['prem_rem_m']} dur_y_last={mp['dur_y_last']}"
    )
    ts = [0, 1, 2, 3, 13, 14]
    print("| variable | " + " | ".join(f"t = {t}" for t in ts) + " |")
    for cn in MONTH_COLS:
        print(f"| `{cn}` | " + " | ".join(g6(r[cn][t]) for t in ts) + " |")
    print("--- full precision, selected ---")
    for cn in (
        "q_ann",
        "q_m",
        "w_m",
        "deaths",
        "lapses",
        "pols_if",
        "maint_expense",
        "death_outgo",
        "claim_expense",
        "net_outgo",
        "disc",
        "reserve",
        "reserve_pp",
    ):
        print(f"{cn}: " + ", ".join(f"t={t}: {r[cn][t]!r}" for t in (1, 2, 3, 13, 14)))
    print("reserve(0) =", repr(r["reserve"][0]))
    print("--- 14.2 prose: full-precision left-hand sides, month 1 ---")
    print(f"1 - (1 - q_ann(1))^(1/12) with q_ann(1) = {r['q_ann'][1]!r} -> {r['q_m'][1]!r}")
    print(f"(1 - q_m(1)) * w_m(1) = (1 - {r['q_m'][1]!r}) * {r['w_m'][1]!r} = {r['lapses'][1]!r}")
    print(f"q_m(1) * 200000 = {r['death_outgo'][1]!r}")
    print(f"commission(1) = 46.40 * 0.02 = {r['commission'][1]!r}")
    print(
        f"retyped 6 s.f. lhs: 1-(1-0.00229121)^(1/12) = {1 - (1 - 0.00229121) ** TWELFTH!r}; "
        f"(1-0.000191135)*0.00339605 = {(1 - 0.000191135) * 0.00339605!r}; "
        f"0.000191135*200000 = {0.000191135 * 200000!r}"
    )
    print("--- scalars P05 (10 s.f. and repr) ---")
    for k in (
        "pv_premiums",
        "pv_commission",
        "pv_maint_expense",
        "pv_death_outgo",
        "pv_claim_expense",
        "pv_net_cf",
        "reserve_0",
        "reserve_pp_0",
        "reserve_floored_0",
        "maturities_n",
        "sum_pv_net_cf_t",
    ):
        print(f"{k} = {r['scalars'][k]:.10g}   ({r['scalars'][k]!r})")
    s = r["scalars"]
    print(f"H5a residual reserve_0 + pv_net_cf = {s['reserve_0'] + s['pv_net_cf']:.2e}")
    print(f"P9 residual pv_net_cf - sum_pv_net_cf_t = {s['pv_net_cf'] - s['sum_pv_net_cf_t']:.2e}")
    print("--- hand-checkable steps P05 ---")
    print(
        f"A: reserve({n - 1}) = (0 + net_outgo({n})) * v_m = (0 + {r['net_outgo'][n]!r}) * {V_M!r} = {r['reserve'][n - 1]!r}"
    )
    print(
        f"B: reserve(0) = (reserve(1) + net_outgo(1)) * v_m = ({r['reserve'][1]!r} + {r['net_outgo'][1]!r}) * v_m = {r['reserve'][0]!r}"
    )
    rf = (r["reserve"][0] + r["premium_income"][1] - r["commission"][1]) * (1 + I_M)
    print(
        f"C: (reserve(0) + premium_income(1) - commission(1)) * (1 + i_m) = ({r['reserve'][0]!r} + {r['premium_income'][1]!r} - {r['commission'][1]!r}) * (1 + i_m) = {rf!r}"
    )
    end = rf - r["maint_expense"][1] - r["death_outgo"][1] - r["claim_expense"][1]
    print(
        f"   - maint {r['maint_expense'][1]!r} - death {r['death_outgo'][1]!r} - claim_exp {r['claim_expense'][1]!r} = {end!r}; reserve(1) = {r['reserve'][1]!r}; residual {end - r['reserve'][1]:.2e}"
    )
    u = 1
    rhs = -(r["premium_pp"][u] - r["commission_pp"][u]) + V_M * (
        r["q_m"][u] * (mp["sum_assured"] + r["claim_exp_pp"][u])
        + r["maint_pp"][u]
        + (1 - r["q_m"][u]) * (1 - r["w_m"][u]) * r["reserve_pp"][u]
    )
    print(
        f"D: P5 at t = 0: rhs = {rhs!r}; reserve_pp(0) = {r['reserve_pp'][0]!r}; diff = {r['reserve_pp'][0] - rhs:.2e}"
    )

    print("\n=== scalars for all reference points (6 s.f.) ===")
    keys = (
        "pv_premiums",
        "pv_commission",
        "pv_maint_expense",
        "pv_death_outgo",
        "pv_claim_expense",
        "pv_net_cf",
        "reserve_0",
        "reserve_pp_0",
        "reserve_floored_0",
        "maturities_n",
    )
    tot = {k: 0.0 for k in keys if k not in ("reserve_pp_0", "maturities_n")}
    for mp in REF:
        sc = runs[mp["policy_id"]]["scalars"]
        print(
            f"| {mp['policy_id']} | {mp['rem_m']} | " + " | ".join(g6(sc[k]) for k in keys) + " |"
        )
        for k in tot:
            tot[k] += sc[k]
    print("| total | | " + " | ".join(g6(tot[k]) if k in tot else "" for k in keys) + " |")
    print("--- sign of reserve ---")
    for mp in REF:
        res = runs[mp["policy_id"]]["reserve"]
        neg = [t for t in range(0, mp["rem_m"] + 1) if res[t] < 0]
        pk = max(range(0, mp["rem_m"] + 1), key=lambda t: res[t])
        span = f"t = {neg[0]} .. {neg[-1]} ({len(neg)} months)" if neg else "never"
        print(f"{mp['policy_id']}: negative {span}; peak {res[pk]:.6g} at t = {pk}")
    p3 = runs["P03"]
    print("--- P03 one-row check (repr) ---")
    for cn in (
        "q_ann",
        "q_m",
        "deaths",
        "lapses",
        "maturities",
        "pols_if",
        "premium_income",
        "commission",
        "maint_expense",
        "death_outgo",
        "claim_expense",
        "net_outgo",
        "reserve",
    ):
        print(f"P03 {cn}(1) = {p3[cn][1]!r}")
    print(f"P03 reserve(0) = {p3['reserve'][0]!r}")
    p10 = runs["P10"]
    print(
        f"P10: premium_income(1) = {p10['premium_income'][1]!r}, premium_income(2) = {p10['premium_income'][2]!r}, lapses(1) = {p10['lapses'][1]!r}, lapses(2) = {p10['lapses'][2]!r}"
    )
    p8 = runs["P08"]
    print(
        f"P08: age(1) = {p8['age'][1]}, age(48) = {p8['age'][48]}, pv_premiums = {p8['scalars']['pv_premiums']!r}"
    )

    print("\n=== magnitudes of named simplifications (P01) ===")
    mp = REF[0]
    base = runs["P01"]
    b = base["scalars"]
    print(f"base: pv_death_outgo = {b['pv_death_outgo']!r}; reserve_0 = {b['reserve_0']!r}")
    for label, kw in (
        ("q_ann/12 only (w_m unchanged)", {"q_over_12": True}),
        ("w_ann/12 only (q_m unchanged)", {"w_over_12": True}),
        ("both q_ann/12 and w_ann/12", {"q_over_12": True, "w_over_12": True}),
    ):
        a1 = project_mp(mp, **kw)["scalars"]
        print(
            f"{label}: pv_death_outgo = {a1['pv_death_outgo']!r} ({a1['pv_death_outgo'] / b['pv_death_outgo'] - 1:+.4%}); "
            f"reserve_0 {b['reserve_0']:.6g} -> {a1['reserve_0']:.6g} ({a1['reserve_0']!r})"
        )
    a2 = project_mp(mp, mid_claims=True)["scalars"]
    print(
        f"mid-month claims instead of end of month: pv_death_outgo {a2['pv_death_outgo'] / b['pv_death_outgo'] - 1:+.4%}; reserve_0 {b['reserve_0']:.6g} -> {-a2['pv_net_cf']:.6g}"
    )
    print(
        "--- deaths-first versus simultaneous constant forces (exact dependent death rate minus q_m) ---"
    )
    worst_qw = max(base["q_m"][t] * base["w_m"][t] / 2 for t in range(1, mp["rem_m"] + 1))
    print(f"P01: order term q_m*w_m/2 at most {worst_qw:.2e} per month")
    # Every difference is negative (deaths-first overstates deaths), so "worst" is the minimum.
    best = (0.0, None, None)
    for mpx in REF:
        rr = runs[mpx["policy_id"]]
        for t in range(1, mpx["rem_m"] + 1):
            d = dep_death_diff(rr["q_m"][t], rr["w_m"][t])
            if d < best[0]:
                best = (d, mpx["policy_id"], t)
    diffs = [dep_death_diff(base["q_m"][t], base["w_m"][t]) for t in range(1, mp["rem_m"] + 1)]
    print(
        f"P01: exact difference at worst {min(diffs):.3e} per month; "
        f"largest difference {max(diffs):.3e} (never positive)"
    )
    d, pid, t = best
    rr = runs[pid]
    span = [
        u
        for u in range(1, REF[[m["policy_id"] for m in REF].index(pid)]["rem_m"] + 1)
        if abs(dep_death_diff(rr["q_m"][u], rr["w_m"][u]) - d) < 1e-20
    ]
    print(
        f"all ten reference points: most negative {d:.3e} ({pid}, t = {span[0]} to {span[-1]}, "
        f"age {rr['age'][t]}, w_ann {rr['w_ann'][t]}); q_m*w_m/2 there = {rr['q_m'][t] * rr['w_m'][t] / 2:.3e}"
    )
    w0 = to_monthly_prob(LAPSE_TABLE[0])
    y0 = min(
        (dep_death_diff(to_monthly_prob(MORT[s_][x]), w0), s_, x)
        for s_ in ("M", "F")
        for x in range(18, 66)
    )
    qx = to_monthly_prob(MORT[y0[1]][y0[2]])
    print(
        f"new entrant, policy year 0 (lapse key 0, w_ann 0.15), sex in M/F, entry age 18 to 65: "
        f"most negative {y0[0]:.3e} ({y0[1]}, age {y0[2]}), {y0[0] / qx:.2%} of q_m; "
        f"q_m*w_m/2 = {qx * w0 / 2:.3e}"
    )
    q104 = to_monthly_prob(MORT["M"][104])
    w5 = to_monthly_prob(LAPSE_TABLE[5])
    corner = dep_death_diff(q104, w5)
    print(
        f"schema corner (M, mortality age 104, premiums payable, lapse key 5, w_ann 0.04): "
        f"exact {corner:.4e}; q_m*w_m/2 = {q104 * w5 / 2:.3e}; q_m = {q104:.4e}; "
        f"ratio {corner / q104:.2%} of q_m"
    )
    # admissible pairs: entry age 18..65, completed policy year k = 0..39 with premiums payable,
    # mortality age = entry age + k (at most 104), lapse key min(k, 5)
    space = [
        (
            dep_death_diff(
                to_monthly_prob(MORT[s_][x0 + k]), to_monthly_prob(LAPSE_TABLE[min(k, 5)])
            ),
            to_monthly_prob(MORT[s_][x0 + k]),
            s_,
            x0 + k,
            min(k, 5),
        )
        for s_ in ("M", "F")
        for x0 in range(18, 66)
        for k in range(0, 40)
    ]
    absw = min(space)
    relw = min(space, key=lambda z: z[0] / z[1])
    print(
        f"check over the whole admissible space (sex, entry age 18 to 65, policy year 0 to 39, "
        f"premiums payable): most negative {absw[0]:.4e} at (sex, mortality age, lapse key) = "
        f"{absw[2:]}; largest relative to q_m {relw[0] / relw[1]:.3%} at {relw[2:]}; "
        f"any positive: {any(z[0] > 0 for z in space)}"
    )
    print(
        f"P07 (count 2.5): reserve_0 {runs['P07']['scalars']['reserve_0']!r}, reserve_pp_0 {runs['P07']['scalars']['reserve_pp_0']!r}"
    )


if __name__ == "__main__":
    main()
