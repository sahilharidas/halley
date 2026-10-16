"""Regenerate the printed numbers of docs/spec/in-force-group.md (roadmap milestone M10, document version 0.1).

Implements the document's formulas exactly as written, in plain float64 scalars with the standard
library only. It prints every computed number the document quotes: the basis checkpoints of
section 6, the re-based locked-in curve and the spliced locked-in CPI path of section 5, the four
projection runs of each variant (sections 5.5 and 7), the change in locked-in fulfilment cash
flows (section 8), the period close (section 9), the months after it (section 10), the analysis
of change and the magnitudes of the named simplifications (section 14). It checks every
invariant of section 12 for both variants.

It also carries a second, independent implementation of the headline figures (implementation B:
direct present values, suffix sums and products rather than the month-by-month recursions of
implementation A) and reports how closely the two agree, as the annuity spec's authors did.

This is not the reference implementation and not the engine. The spreadsheet is built from the
document's formulas, and the engine reconciles to the spreadsheet (section 1). If a formula in the
document changes, change it here too and rerun:

    python docs/spec/in_force_group_numbers.py
"""

from __future__ import annotations

import math

# ------------------------------------------------------------------ dates and time (4)
OMEGA = 120  # limiting age: q = 1 at OMEGA
T_OMEGA = 12 * OMEGA + 1  # T_i = T_OMEGA - a0 is the first month at age OMEGA


def mi(year: int, month: int) -> int:
    """Month index of a date: 12 * year + month - 1. The day is ignored."""
    return 12 * year + month - 1


M_REC = mi(2025, 12)  # recognition date 2025-12-31, a month-end
M_V = mi(2026, 12)  # valuation date 2026-12-31, a month-end
M_PREV = M_REC  # previous valuation date: the recognition date (first subsequent measurement)
TAU0 = M_V - M_REC  # months from recognition to V: 12
TAU = M_V - M_PREV  # length of the reporting period in months: 12
M_START = mi(2026, 1)  # annuity_start_date 2026-01-01 for every contract
CPI_LAG = 3  # L, months

# ------------------------------------------------------------------ non-financial tables (6)
A_MK, B_MK, C_MK = 0.00022, 2.7e-6, 1.124  # Standard Ultimate Survival Model (Makeham)
BASE_YEAR = 2025
MULT_V2 = 0.96  # version 2 base table: 0.96 x version 1 below OMEGA (lighter mortality)
FEMALE_SETBACK = 4


def q_makeham(x: int) -> float:
    # order: c^x, B*c^x, *(c-1), /ln(c), -A -, exp, 1-
    return 1 - math.exp(-A_MK - B_MK * C_MK**x * (C_MK - 1) / math.log(C_MK))


def build_mort_base(version: int) -> dict[tuple[str, int], float]:
    tbl: dict[tuple[str, int], float] = {}
    for x in range(20, OMEGA + 1):
        for sex, shift in (("M", 0), ("F", FEMALE_SETBACK)):
            if x == OMEGA:
                tbl[(sex, x)] = 1.0
            else:
                q = q_makeham(x - shift)
                tbl[(sex, x)] = q if version == 1 else MULT_V2 * q
    return tbl


def r_imp(x: int, y: int) -> float:
    h = min(1.0, (y - 2026) / 20)
    w = 1 - 3 * h**2 + 2 * h**3
    g = 1.0 if x <= 85 else ((110 - x) / 25 if x < 110 else 0.0)
    return g * (0.010 + 0.010 * w)


def build_cif() -> dict[tuple[int, int], float]:
    tbl: dict[tuple[int, int], float] = {}
    for x in range(20, OMEGA + 1):
        c = 1.0
        tbl[(x, BASE_YEAR)] = c
        for y in range(BASE_YEAR + 1, 2151):
            c = c * (1 - r_imp(x, y))  # ascending year order
            tbl[(x, y)] = c
    return tbl


MORT_V1 = build_mort_base(1)
MORT_V2 = build_mort_base(2)
CIF = build_cif()  # one version; unchanged at V

MAINT_V1 = 5.00  # PerPolicy[Money] per month, December 2025 prices (version 1)
MAINT_V2 = 5.30  # December 2026 prices (version 2)
IE_RATE = 0.0015  # Rate[Annual, Effective]
IE_M = (1 + IE_RATE) ** (1 / 12) - 1  # Rate[Monthly]
ILP = 0.0050  # Rate[Annual, Continuous]
ALPHA = 0.75
Z_ALPHA = 0.6744897501960817  # Phi^-1(0.75), a literal
SIGMA_L = 0.10
SIGMA_E = 0.10
MORT_MULT = math.exp(-SIGMA_L * Z_ALPHA - SIGMA_L * SIGMA_L / 2)
EXP_MULT = math.exp(SIGMA_E * Z_ALPHA - SIGMA_E * SIGMA_E / 2)


def to_monthly_cf(q: float) -> float:
    """Prob[Annual] -> Prob[Monthly], guarded constant-force conversion."""
    if q >= 1:
        return 1.0
    return -math.expm1(math.log1p(-q) / 12)


# ------------------------------------------------------------------ curves and CPI (5)
def rf_df(t: int, level: float) -> float:
    tau = t / 12
    z = level - 0.015 * math.exp(-tau / 10)
    return math.exp(-z * tau)


def p_rec(t: int) -> float:
    """Composed curve at recognition, P_rec(t) for t = 0..T_rec: the locked-in basis."""
    return rf_df(t, 0.040) * math.exp(-ILP * t / 12)


def p_cur(t: int) -> float:
    """Composed current curve at V."""
    return rf_df(t, 0.045) * math.exp(-ILP * t / 12)


def p_lock(t: int) -> float:
    """Locked-in curve re-based to V."""
    return p_rec(TAU0 + t) / p_rec(TAU0)


I_REC_BASE = 100.0  # December 2025 estimate at recognition
I_DEC25_PUB = 100.1  # December 2025 published value, known at V
G_HIST, G_REC, G_ACT, G_CUR = 1.04, 1.025, 1.032, 1.03


def i_rec(u: int) -> float:
    """CPI path of the recognition run, indexed from the recognition month."""
    if u <= 0:
        return I_REC_BASE * G_HIST ** (u / 12)
    return I_REC_BASE * G_REC ** (u / 12)


def i_act(u: int) -> float:
    """Actual CPI history as known at V, indexed from the valuation month; u <= 0."""
    if u <= -(TAU + 1):
        return I_REC_BASE * G_HIST ** ((u + TAU) / 12)
    return I_DEC25_PUB * G_ACT ** ((u + TAU) / 12)


def i_lock(u: int) -> float:
    """Locked-in CPI path at V: actual history, then locked-in expected inflation."""
    if u <= 0:
        return i_act(u)
    return i_act(0) * i_rec(TAU0 + u) / i_rec(TAU0)


def i_cur(u: int) -> float:
    """Current CPI path at V: actual history, then current expected inflation."""
    if u <= 0:
        return i_act(u)
    return i_act(0) * G_CUR ** (u / 12)


# ------------------------------------------------------------------ contracts and groups (3, 13)
def make_mp(pid, cls, sex, dob, n_rec, inst):
    return {
        "policy_id": pid,
        "cls": cls,
        "sex": sex,
        "dob": dob,
        "n_rec": float(n_rec),
        "inst": float(inst),
        "single_premium": None,
    }


GROUPS = {
    "P": {
        "bucket": "OTHER",
        "k": 1.04,
        "mps": [
            make_mp(11, "level", "M", (1947, 8, 15), 100, 1000.00),
            make_mp(12, "cpi", "F", (1945, 11, 30), 50, 800.00),
        ],
        "n_act": {11: 98.0, 12: 50.0},
        # payments actually made in (V_prev, V]: 1000 x 1187 life-months + 800 x 600 life-months
        "cu_act": 1000.00 * 1187 + 800.00 * 600,
        "cu_act_text": "1000.00 x 1187 + 800.00 x 600 life-months",
    },
    "O": {
        "bucket": "ONEROUS",
        "k": 0.98,
        "mps": [
            make_mp(21, "level", "M", (1947, 8, 15), 100, 1000.00),
            make_mp(22, "cpi", "F", (1945, 11, 30), 50, 800.00),
        ],
        "n_act": {21: 95.0, 22: 47.0},
        "cu_act": 1000.00 * 1170 + 800.00 * 588,
        "cu_act_text": "1000.00 x 1170 + 800.00 x 588 life-months",
    },
}

COLS = [
    "age",
    "cal_year",
    "q_ann",
    "q_mth",
    "deaths",
    "pols_if",
    "anniv",
    "esc_factor",
    "ann_pp",
    "exp_pp",
    "ann_out",
    "exp_out",
    "df",
    "v",
    "s_pp",
    "q_ann_st",
    "q_mth_st",
    "s_pp_st",
    "cu",
    "inv_exp",
    "bel",
    "v_be",
    "v_st",
    "ra",
]


# ------------------------------------------------------------------ one run's inputs (5.5)
def make_run(name, m_v, mort, maint, df, cpi, inforce, inst, mps):
    """A projection run: one scenario (J = 1), one set of non-financial tables."""
    a0 = {mp["policy_id"]: m_v - mi(mp["dob"][0], mp["dob"][1]) for mp in mps}
    return {
        "name": name,
        "m_v": m_v,
        "mort": mort,
        "cif": CIF,
        "maint": maint,
        "ie_m": IE_M,
        "scenarios": {0: (df, cpi)},  # scenario id -> (composed curve, CPI path); J = 1
        "inforce": inforce,
        "inst": inst,
        "mps": mps,
        "a0": a0,
        "T": T_OMEGA - min(a0.values()),
    }


# ------------------------------------------------------------------ implementation A (6)
def project(mp, run):
    """One policy for t = 0 .. T by the recursions of section 6. Month quantities are None at
    t = 0 except the states pols_if, ann_pp, df, bel, v_be, v_st, ra."""
    assert len(run["scenarios"]) == 1, "G6: the aggregation stage reads J = 1"
    dfn, cpi = run["scenarios"][0]
    T, m_v = run["T"], run["m_v"]
    pid, sex, cls = mp["policy_id"], mp["sex"], mp["cls"]
    a0 = run["a0"][pid]
    m_c = M_START - m_v
    mort, cif, maint, ie_m = run["mort"], run["cif"], run["maint"], run["ie_m"]
    r = {c: [None] * (T + 1) for c in COLS}
    r["pols_if"][0] = run["inforce"][pid]
    r["ann_pp"][0] = run["inst"][pid]
    r["df"][0] = dfn(0)
    i0 = cpi(0)
    for t in range(1, T + 1):
        x = min(OMEGA, (a0 + t - 1) // 12)
        y = (m_v + t) // 12
        q_ann = 1.0 if x >= OMEGA else mort[(sex, x)] * cif[(x, y)]
        q_mth = to_monthly_cf(q_ann)
        open_ = r["pols_if"][t - 1]
        deaths = open_ * q_mth
        pols_if = open_ - deaths
        k = t - m_c
        anniv = cls == "cpi" and k > 0 and k % 12 == 0
        esc = cpi(t - CPI_LAG) / cpi(t - CPI_LAG - 12) if anniv else 1.0
        ann_pp = r["ann_pp"][t - 1] * esc
        exp_pp = maint * (cpi(t - 1) / i0)
        ann_out = open_ * ann_pp
        exp_out = open_ * exp_pp
        df = dfn(t)
        v = df / r["df"][t - 1]
        s_pp = ann_pp + exp_pp
        q_ann_st = 1.0 if x >= OMEGA else q_ann * MORT_MULT
        q_mth_st = to_monthly_cf(q_ann_st)
        s_pp_st = ann_pp + exp_pp * EXP_MULT
        row = {
            "age": x,
            "cal_year": y,
            "q_ann": q_ann,
            "q_mth": q_mth,
            "deaths": deaths,
            "pols_if": pols_if,
            "anniv": anniv,
            "esc_factor": esc,
            "ann_pp": ann_pp,
            "exp_pp": exp_pp,
            "ann_out": ann_out,
            "exp_out": exp_out,
            "df": df,
            "v": v,
            "s_pp": s_pp,
            "q_ann_st": q_ann_st,
            "q_mth_st": q_mth_st,
            "s_pp_st": s_pp_st,
            "cu": ann_out,
        }
        for name, val in row.items():
            r[name][t] = val
    # backward recursions (6.7, 6.8)
    r["bel"][T] = 0.0
    r["v_be"][T] = 0.0
    r["v_st"][T] = 0.0
    r["inv_exp"][T] = ie_m * r["bel"][T]
    for t in range(T - 1, -1, -1):
        u = t + 1
        r["inv_exp"][u] = ie_m * r["bel"][u]
        r["bel"][t] = (
            r["ann_out"][u]
            + r["exp_out"][u]
            + r["inv_exp"][u] * r["v"][u]
            + r["v"][u] * r["bel"][u]
        )
        r["v_be"][t] = r["s_pp"][u] + (1 - r["q_mth"][u]) * (
            r["v"][u] * (r["v_be"][u] + ie_m * r["v_be"][u])
        )
        r["v_st"][t] = r["s_pp_st"][u] + (1 - r["q_mth_st"][u]) * (
            r["v"][u] * (r["v_st"][u] + ie_m * r["v_st"][u])
        )
    for t in range(T + 1):
        r["ra"][t] = r["pols_if"][t] * (r["v_st"][t] - r["v_be"][t])
    # scalars (7.2): sums in ascending t
    pva = pve = 0.0
    for t in range(1, T + 1):
        pva += r["df"][t - 1] * r["ann_out"][t]
        pve += r["df"][t - 1] * r["exp_out"][t] + r["df"][t] * r["inv_exp"][t]
    r["scalars"] = {
        "bel_0": r["bel"][0],
        "ra_0": r["ra"][0],
        "ann_pp_1": r["ann_pp"][1],
        "pva": pva,
        "pve": pve,
        "t_end": T_OMEGA - a0,
    }
    r["T"] = T
    r["a0"] = a0
    r["m_c"] = m_c
    return r


def run_group(run):
    """Project every policy of a run and form the group totals in ascending policy_id (7.1)."""
    pols = {mp["policy_id"]: project(mp, run) for mp in run["mps"]}
    T = run["T"]
    g = {"bel": [0.0] * (T + 1), "ra": [0.0] * (T + 1), "cu": [None] + [0.0] * T}
    for pid in sorted(pols):
        for t in range(T + 1):
            g["bel"][t] += pols[pid]["bel"][t]
            g["ra"][t] += pols[pid]["ra"][t]
        for t in range(1, T + 1):
            g["cu"][t] += pols[pid]["cu"][t]
    g["fcf"] = [g["bel"][t] + g["ra"][t] for t in range(T + 1)]
    run["pols"] = pols
    run["g"] = g
    return run


# ------------------------------------------------------------------ aggregation stage (7 to 10)
def recognise(run):
    """Initial recognition at the run's valuation date (7.3)."""
    p_g = 0.0
    for mp in sorted(run["mps"], key=lambda m: m["policy_id"]):
        p_g += run["inforce"][mp["policy_id"]] * mp["single_premium"]
    fcf0 = run["g"]["fcf"][0]
    gap = p_g - fcf0
    csm0 = gap if gap > 0 else 0.0
    gap_l = fcf0 - p_g
    lc0 = gap_l if gap_l > 0 else 0.0
    return {"p_g": p_g, "fcf_0": fcf0, "csm_0": csm0, "lc_0": lc0}


def cu_next(cu, p_l, T):
    """N(t) for t = 0 .. T: units of months after t, valued at t on the locked-in curve (10)."""
    n = [0.0] * (T + 1)
    for t in range(T - 1, -1, -1):
        n[t] = cu[t + 1] + (p_l(t + 1) / p_l(t)) * n[t + 1]
    return n


def remeasure(csm, lc, delta):
    """The remeasurement rules of section 9.4, applied to (CSM_acc, LC_alloc, delta FCF^L)."""
    if lc > 0 and delta >= 0:
        return csm, lc + delta
    if lc > 0:  # delta < 0
        rev = min(lc, -delta)
        lc = lc - rev
        csm = csm + (-delta - rev)
        return csm, lc
    csm = csm - delta
    if csm < 0:
        return 0.0, -csm
    return csm, 0.0


def close_period(csm_prev, lc_prev, cu_act, f_prev_0, f_prev_tau, delta, n_0):
    """The period close of section 9, steps 1 to 5, in the order of para 44 and B119(b)."""
    assert csm_prev >= 0
    assert lc_prev >= 0
    assert not (csm_prev > 0 and lc_prev > 0), "at most one of csm_prev, lc_prev is positive"
    s = {}
    s["csm_acc"] = csm_prev * p_rec(TAU0 - TAU) / p_rec(TAU0)  # step 1
    s["acc_per"] = s["csm_acc"] - csm_prev
    s["lc_alloc"] = lc_prev * (f_prev_tau / f_prev_0) if f_prev_0 > 0 else 0.0  # step 2
    s["lca_per"] = lc_prev - s["lc_alloc"]
    s["csm_adj"], s["lc_adj"] = remeasure(s["csm_acc"], s["lc_alloc"], delta)  # step 3
    s["csm_fs"] = s["csm_adj"] - s["csm_acc"]
    s["lc_fs"] = s["lc_adj"] - s["lc_alloc"]
    u = cu_act + n_0
    s["r_per"] = s["csm_adj"] * (cu_act / u) if u > 0 else s["csm_adj"]  # step 4
    s["csm_0"] = s["csm_adj"] - s["r_per"]  # step 5
    s["lc_0"] = s["lc_adj"]
    return s


def monthly_roll(csm0, lc0, cu, f, p_l, T):
    """Months 1 .. T after the close (10): accretion, release, loss-component allocation."""
    n = cu_next(cu, p_l, T)
    r = {
        "n": n,
        "j_l": [None] * (T + 1),
        "acc": [None] * (T + 1),
        "csm_pre": [None] * (T + 1),
        "u": [None] * (T + 1),
        "r": [None] * (T + 1),
        "csm": [csm0] + [None] * T,
        "lc": [lc0] + [None] * T,
        "lca": [None] * (T + 1),
    }
    for t in range(1, T + 1):
        j = p_l(t - 1) / p_l(t) - 1
        acc = j * r["csm"][t - 1]
        pre = r["csm"][t - 1] + acc
        u = cu[t] + n[t]
        rel = pre * (cu[t] / u) if u > 0 else pre
        r["j_l"][t] = j
        r["acc"][t] = acc
        r["csm_pre"][t] = pre
        r["u"][t] = u
        r["r"][t] = rel
        r["csm"][t] = pre - rel
        r["lc"][t] = r["lc"][t - 1] * (f[t] / f[t - 1]) if f[t - 1] > 0 else 0.0
        r["lca"][t] = r["lc"][t - 1] - r["lc"][t]
    return r


# ------------------------------------------------------------------ implementation B (14.9)
def direct(mp, run, p_l=None):
    """Headline figures of one policy by direct discounting: survival products, telescoped
    instalments, suffix sums of discounted cash flows, no month-by-month recursion."""
    dfn, cpi = run["scenarios"][0]
    T, m_v = run["T"], run["m_v"]
    pid, sex, cls = mp["policy_id"], mp["sex"], mp["cls"]
    a0 = run["a0"][pid]
    m_c = M_START - m_v
    mort, cif, maint, ie_m = run["mort"], run["cif"], run["maint"], run["ie_m"]
    n0, inst = run["inforce"][pid], run["inst"][pid]
    # survival from time 0 to time t, best estimate and stressed (running products, not counts)
    tp, tp_st = [1.0] * (T + 1), [1.0] * (T + 1)
    for t in range(1, T + 1):
        x = min(OMEGA, (a0 + t - 1) // 12)
        y = (m_v + t) // 12
        q = 1.0 if x >= OMEGA else mort[(sex, x)] * cif[(x, y)]
        q_st = 1.0 if x >= OMEGA else q * MORT_MULT
        tp[t] = tp[t - 1] * (1 - to_monthly_cf(q))
        tp_st[t] = tp_st[t - 1] * (1 - to_monthly_cf(q_st))
    # instalment by the telescoped CPI form (invariant I9) or the constant
    n_0 = m_c if m_c == 1 else max(m_c + 12 * k for k in range(0, 200) if m_c + 12 * k <= 0)
    ann = [None] * (T + 1)
    for t in range(1, T + 1):
        if cls == "cpi":
            annivs = [m for m in range(1, t + 1) if m - m_c > 0 and (m - m_c) % 12 == 0]
            n_k = annivs[-1] if annivs else n_0
            ann[t] = inst * cpi(n_k - CPI_LAG) / cpi(n_0 - CPI_LAG)
        else:
            ann[t] = inst
    i0 = cpi(0)
    exp_pp = [None] + [maint * (cpi(t - 1) / i0) for t in range(1, T + 1)]
    df = [dfn(t) for t in range(T + 1)]
    acc = [(1 + ie_m) ** t for t in range(T + 1)]
    # suffix sums: S(t) = sum over s > t of cash flow(s) * df(s-1) * (1 + ie_m)^(s-1)
    s_be, s_st, s_cf, s_ann, s_exp = [0.0] * (T + 1), [0.0] * (T + 1), [0.0] * (T + 1), 0.0, 0.0
    for s in range(T, 0, -1):
        w = df[s - 1] * acc[s - 1]
        s_be[s - 1] = s_be[s] + tp[s - 1] * (ann[s] + exp_pp[s]) * w
        s_st[s - 1] = s_st[s] + tp_st[s - 1] * (ann[s] + exp_pp[s] * EXP_MULT) * w
        s_cf[s - 1] = s_cf[s] + n0 * tp[s - 1] * (ann[s] + exp_pp[s]) * w
    bel = [s_cf[t] / (df[t] * acc[t]) for t in range(T + 1)]
    # v_st conditions on stressed survival, so the in-force RA carries tp(t) / tp_st(t)
    ra = [
        n0 * (tp[t] / tp_st[t] * s_st[t] - s_be[t]) / (df[t] * acc[t]) if tp_st[t] > 0 else 0.0
        for t in range(T + 1)
    ]
    for t in range(1, T + 1):
        s_ann += df[t - 1] * n0 * tp[t - 1] * ann[t]
        s_exp += df[t - 1] * n0 * tp[t - 1] * exp_pp[t] + df[t] * ie_m * bel[t]
    cu = [None] + [n0 * tp[t - 1] * ann[t] for t in range(1, T + 1)]
    out = {
        "bel": bel,
        "ra": ra,
        "cu": cu,
        "pols_if": [n0 * tp[t] for t in range(T + 1)],
        "scalars": {"bel_0": bel[0], "ra_0": ra[0], "ann_pp_1": ann[1], "pva": s_ann, "pve": s_exp},
    }
    if p_l is not None:  # N(t) as a direct sum of discounted units
        suffix = [0.0] * (T + 1)
        for m in range(T, 0, -1):
            suffix[m - 1] = suffix[m] + cu[m] * p_l(m - 1)
        out["n"] = [suffix[t] / p_l(t) for t in range(T + 1)]
    return out


def direct_group(run, p_l=None):
    pols = {mp["policy_id"]: direct(mp, run, p_l) for mp in run["mps"]}
    T = run["T"]
    g = {"bel": [0.0] * (T + 1), "ra": [0.0] * (T + 1), "cu": [None] + [0.0] * T}
    if p_l is not None:
        g["n"] = [0.0] * (T + 1)
    for pid in sorted(pols):
        for t in range(T + 1):
            g["bel"][t] += pols[pid]["bel"][t]
            g["ra"][t] += pols[pid]["ra"][t]
            if p_l is not None:
                g["n"][t] += pols[pid]["n"][t]
        for t in range(1, T + 1):
            g["cu"][t] += pols[pid]["cu"][t]
    g["fcf"] = [g["bel"][t] + g["ra"][t] for t in range(T + 1)]
    return {"pols": pols, "g": g}


def monthly_roll_direct(csm0, lc0, cu, f, n, p_l, T):
    """CSM(t) as a product of monthly factors and LC(t) as a ratio to t = 0."""
    csm, lc, rel = [csm0] + [None] * T, [lc0] + [None] * T, [None] * (T + 1)
    for t in range(1, T + 1):
        a_l = p_l(t - 1) / p_l(t)
        u = cu[t] + n[t]
        csm[t] = csm[t - 1] * a_l * (n[t] / u) if u > 0 else 0.0
        rel[t] = csm[t - 1] * a_l * (cu[t] / u) if u > 0 else csm[t - 1] * a_l
        lc[t] = lc0 * (f[t] / f[0]) if f[0] > 0 else 0.0
    return {"csm": csm, "lc": lc, "r": rel}


# ------------------------------------------------------------------ invariants (12)
EPS = 1e-9


def within(a, b, scale):
    return abs(a - b) <= EPS * max(abs(a), abs(b), scale)


def rel(a, b):
    d = max(abs(a), abs(b))
    return abs(a - b) / d if d > 0 else 0.0


def check_run(run, label, worst):
    """I1 to I10 on every policy of a run (12.1)."""
    T = run["T"]

    def rec(k, v):
        worst[k] = max(worst.get(k, 0.0), abs(v))

    for pid, r in run["pols"].items():
        tag = f"{label} {pid}"
        t_end = r["scalars"]["t_end"]
        scale = r["bel"][0] if r["bel"][0] > 0 else 1.0
        for t in range(1, T + 1):
            assert 0 <= r["pols_if"][t] <= r["pols_if"][t - 1], f"{tag} I1 t={t}"
            for c in ("q_ann", "q_mth", "q_ann_st", "q_mth_st"):
                assert 0 <= r[c][t] <= 1, f"{tag} I2 {c}"
            assert r["deaths"][t] <= r["pols_if"][t - 1], f"{tag} I2 deaths"
            assert r["cu"][t] == r["ann_out"][t] >= 0, f"{tag} I10"
        assert r["bel"][T] == r["v_be"][T] == r["v_st"][T] == 0.0, f"{tag} I3"
        assert r["q_ann"][t_end] == 1.0, f"{tag} I3 omega q"
        assert r["pols_if"][t_end] == 0.0, f"{tag} I3 omega pols_if"
        for t in range(t_end + 1, T + 1):
            for c in ("pols_if", "bel", "ra"):
                assert r[c][t] == 0.0, f"{tag} I3+ {c}"
        for t in range(1, T + 1):  # I4 roll-forward
            lhs = (r["bel"][t - 1] - r["ann_out"][t] - r["exp_out"][t]) / r["v"][t] - r["inv_exp"][
                t
            ]
            s4 = (abs(r["bel"][t - 1]) + abs(r["ann_out"][t]) + abs(r["exp_out"][t])) / r["v"][
                t
            ] + abs(r["inv_exp"][t])
            rec("I4", (lhs - r["bel"][t]) / s4 if s4 > 0 else 0.0)
            assert abs(lhs - r["bel"][t]) <= 1e-12 * s4 + 1e-300, f"{tag} I4 t={t}"
        acc = 0.0  # I5 direct PV in ascending t
        for t in range(1, T + 1):
            acc += (
                r["df"][t - 1] * (r["ann_out"][t] + r["exp_out"][t]) + r["df"][t] * r["inv_exp"][t]
            )
        rec("I5", rel(acc, r["bel"][0]))
        assert within(acc, r["bel"][0], scale), f"{tag} I5"
        for t in range(T + 1):  # I6
            lhs, rhs = r["bel"][t], r["pols_if"][t] * r["v_be"][t]
            rec("I6", abs(lhs - rhs) / max(abs(lhs), abs(rhs), scale))
            assert abs(lhs - rhs) <= 1e-12 * max(abs(lhs), abs(rhs), scale), f"{tag} I6 t={t}"
            assert r["ra"][t] >= 0, f"{tag} I7"
        m_c = r["m_c"]
        n_0 = m_c if m_c == 1 else max(m_c + 12 * k for k in range(0, 200) if m_c + 12 * k <= 0)
        _, cpi = run["scenarios"][0]
        inst = run["inst"][pid]
        cls = next(m["cls"] for m in run["mps"] if m["policy_id"] == pid)
        for t in range(1, T + 1):  # I8 / I9
            if cls == "level":
                assert r["ann_pp"][t] == inst, f"{tag} I8"
            else:
                annivs = [m for m in range(1, t + 1) if m - m_c > 0 and (m - m_c) % 12 == 0]
                n_k = annivs[-1] if annivs else n_0
                rhs = inst * cpi(n_k - CPI_LAG) / cpi(n_0 - CPI_LAG)
                rec("I9", rel(r["ann_pp"][t], rhs))
                assert abs(r["ann_pp"][t] - rhs) <= 1e-12 * abs(rhs), f"{tag} I9 t={t}"


def check_close(v, worst):
    """K1 to K7 and I11 to I13 on the period close and the months after it (12.2)."""
    s, g_in, n = v["close"], v["inputs"], v["roll"]
    scale = g_in["p_g"]

    def rec(k, x):
        worst[k] = max(worst.get(k, 0.0), abs(x))

    # K1, K2: the roll-forward lines sum to the closing balances
    k1 = (g_in["csm_prev"] + s["acc_per"] + s["csm_fs"] - s["r_per"]) - s["csm_0"]
    k2 = (g_in["lc_prev"] - s["lca_per"] + s["lc_fs"]) - s["lc_0"]
    rec("K1", k1 / scale)
    rec("K2", k2 / scale)
    assert abs(k1) <= 1e-12 * scale, "K1"
    assert abs(k2) <= 1e-12 * scale, "K2"
    # K3: conservation through the remeasurement rules
    k3 = (s["csm_adj"] - s["lc_adj"]) - ((s["csm_acc"] - s["lc_alloc"]) - v["delta"])
    rec("K3", k3 / scale)
    assert abs(k3) <= 1e-12 * scale, "K3"
    # K4: non-negativity, at most one positive, before and after the release
    for a, b in ((s["csm_adj"], s["lc_adj"]), (s["csm_0"], s["lc_0"])):
        assert a >= 0, "K4"
        assert b >= 0, "K4"
        assert not (a > 0 and b > 0), "K4 both positive"
    assert s["csm_acc"] >= 0, "K4 pre"
    assert s["lc_alloc"] >= 0, "K4 pre"
    # K5: equal allocation at the period level
    if g_in["cu_act"] > 0 and s["csm_0"] > 0:
        lhs, rhs = s["r_per"] / g_in["cu_act"], s["csm_0"] / n["n"][0]
        rec("K5", rel(lhs, rhs))
        assert within(lhs, rhs, 0.0), "K5"
    # K7: the period allocation equals the previous run's monthly loss component at tau
    lc_rec_tau = v["rec_roll"]["lc"][TAU]
    rec("K7", rel(s["lc_alloc"], lc_rec_tau))
    assert within(s["lc_alloc"], lc_rec_tau, scale), "K7"
    # K6, I11, I12, I13 over months 1..T
    T = len(n["csm"]) - 1
    cu, f = v["cur"]["g"]["cu"], v["cur"]["g"]["fcf"]
    last = max([t for t in range(1, T + 1) if cu[t] > 0], default=0)
    for t in range(1, T + 1):
        assert n["csm"][t] >= 0, "K6"
        assert n["lc"][t] >= 0, "K6"
        lhs = n["csm"][t - 1] + n["acc"][t] - n["r"][t]
        s11 = n["csm"][t - 1] + abs(n["acc"][t]) + n["r"][t]
        rec("I11", (lhs - n["csm"][t]) / s11 if s11 > 0 else 0.0)
        assert abs(lhs - n["csm"][t]) <= 1e-12 * s11 + 1e-300, "I11"
        if t > last:
            assert n["csm"][t] == 0.0, "I11 after coverage"
        if cu[t] > 0 and n["n"][t] > 0 and n["csm"][t] > 0:
            a, b = n["r"][t] / cu[t], n["csm"][t] / n["n"][t]
            rec("I12", rel(a, b))
            assert within(a, b, 0.0), "I12"
        if f[t] > 0 and n["lc"][0] > 0:
            a, b = n["lc"][t] / f[t], n["lc"][0] / f[0]
            rec("I13", rel(a, b))
            assert within(a, b, 0.0), "I13"
        if f[t] == 0:
            assert n["lc"][t] == 0.0, "I13 zero"


# ------------------------------------------------------------------ formatting
def g6(x):
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int):
        return str(x)
    if x == 0:
        return "0"
    return f"{x:.6g}"


def g10(x):
    """10 significant figures with thousands separators, as the annuity spec prints its values."""
    if x == 0:
        return "0"
    digits = len(str(int(abs(x)))) if abs(x) >= 1 else 1
    return f"{x:,.{max(0, 10 - digits)}f}"


# ------------------------------------------------------------------ the two variants
def build_variant(key):
    """Every run and every figure of one variant (P or O), sections 5 to 10 and 14."""
    spec = GROUPS[key]
    mps = spec["mps"]
    ids = [mp["policy_id"] for mp in mps]
    n_rec = {mp["policy_id"]: mp["n_rec"] for mp in mps}
    inst = {mp["policy_id"]: mp["inst"] for mp in mps}
    # --- REC: the recognition run (section 5.5), which is also the previous run
    rec_run = run_group(make_run("REC", M_REC, MORT_V1, MAINT_V1, p_rec, i_rec, n_rec, inst, mps))
    for mp in mps:  # pricing rule, outside the model (13): round(k * FCF_i(0) / n, 2)
        r = rec_run["pols"][mp["policy_id"]]
        mp["single_premium"] = round(spec["k"] * (r["bel"][0] + r["ra"][0]) / mp["n_rec"], 2)
    recog = recognise(rec_run)
    for mp in mps:  # G3 at recognition
        r = rec_run["pols"][mp["policy_id"]]
        p_i = mp["n_rec"] * mp["single_premium"]
        onerous = p_i - (r["bel"][0] + r["ra"][0]) < 0
        assert onerous == (spec["bucket"] == "ONEROUS"), f"G3 {mp['policy_id']}"
    rec_roll = monthly_roll(
        recog["csm_0"], recog["lc_0"], rec_run["g"]["cu"], rec_run["g"]["fcf"], p_rec, rec_run["T"]
    )
    # --- what the previous run supplies (3.4)
    n_exp = {pid: rec_run["pols"][pid]["pols_if"][TAU] for pid in ids}
    f_prev_0, f_prev_tau = rec_run["g"]["fcf"][0], rec_run["g"]["fcf"][TAU]
    inputs = {
        "csm_prev": recog["csm_0"],
        "lc_prev": recog["lc_0"],
        "cu_act": spec["cu_act"],
        "p_g": recog["p_g"],
        "n_exp": n_exp,
        "f_prev_0": f_prev_0,
        "f_prev_tau": f_prev_tau,
        "cu_exp": sum(rec_run["g"]["cu"][t] for t in range(1, TAU + 1)),
    }
    # --- the three runs at V (5.5)
    maint_old = MAINT_V1 * i_act(0) / i_act(-TAU)  # expense rebase (5.4)
    n_act = spec["n_act"]
    old = run_group(make_run("OLD", M_V, MORT_V1, maint_old, p_lock, i_lock, n_exp, inst, mps))
    new = run_group(make_run("NEW", M_V, MORT_V2, MAINT_V2, p_lock, i_lock, n_act, inst, mps))
    cur = run_group(make_run("CUR", M_V, MORT_V2, MAINT_V2, p_cur, i_cur, n_act, inst, mps))
    exp_only = run_group(make_run("EXP", M_V, MORT_V1, maint_old, p_lock, i_lock, n_act, inst, mps))
    delta = new["g"]["fcf"][0] - old["g"]["fcf"][0]  # section 8
    n = cu_next(cur["g"]["cu"], p_lock, cur["T"])
    close = close_period(
        inputs["csm_prev"], inputs["lc_prev"], inputs["cu_act"], f_prev_0, f_prev_tau, delta, n[0]
    )
    roll = monthly_roll(
        close["csm_0"], close["lc_0"], cur["g"]["cu"], cur["g"]["fcf"], p_lock, cur["T"]
    )
    # --- implementation B
    b = {name: direct_group(run) for name, run in (("REC", rec_run), ("OLD", old), ("NEW", new))}
    b["CUR"] = direct_group(cur, p_lock)
    delta_b = b["NEW"]["g"]["fcf"][0] - b["OLD"]["g"]["fcf"][0]
    close_b = close_period(
        inputs["csm_prev"],
        inputs["lc_prev"],
        inputs["cu_act"],
        f_prev_0,
        f_prev_tau,
        delta_b,
        b["CUR"]["g"]["n"][0],
    )
    roll_b = monthly_roll_direct(
        close_b["csm_0"],
        close_b["lc_0"],
        b["CUR"]["g"]["cu"],
        b["CUR"]["g"]["fcf"],
        b["CUR"]["g"]["n"],
        p_lock,
        cur["T"],
    )
    return {
        "key": key,
        "spec": spec,
        "mps": mps,
        "ids": ids,
        "rec": rec_run,
        "recog": recog,
        "rec_roll": rec_roll,
        "inputs": inputs,
        "maint_old": maint_old,
        "old": old,
        "new": new,
        "cur": cur,
        "exp_only": exp_only,
        "delta": delta,
        "n": n,
        "close": close,
        "roll": roll,
        "b": b,
        "delta_b": delta_b,
        "close_b": close_b,
        "roll_b": roll_b,
    }


def compare_ab(v):
    """Max relative difference between implementations A and B on the headline figures (14.9)."""
    worst, count = 0.0, 0
    pairs = []
    for name in ("REC", "OLD", "NEW", "CUR"):
        a_run, b_run = v[name.lower()], v["b"][name]
        for pid in v["ids"]:
            for k in ("bel_0", "ra_0", "pva", "pve", "ann_pp_1"):
                pairs.append(
                    (
                        f"{name} {pid} {k}",
                        a_run["pols"][pid]["scalars"][k],
                        b_run["pols"][pid]["scalars"][k],
                    )
                )
        pairs.append((f"{name} FCF_g(0)", a_run["g"]["fcf"][0], b_run["g"]["fcf"][0]))
    pairs.append(("delta FCF^L", v["delta"], v["delta_b"]))
    for k in ("csm_acc", "lc_alloc", "csm_adj", "lc_adj", "r_per", "csm_0", "lc_0"):
        pairs.append((k, v["close"][k], v["close_b"][k]))
    pairs.append(("N(0)", v["n"][0], v["b"]["CUR"]["g"]["n"][0]))
    for t in range(1, 13):
        pairs.append((f"CU_g({t})", v["cur"]["g"]["cu"][t], v["b"]["CUR"]["g"]["cu"][t]))
        pairs.append((f"R({t})", v["roll"]["r"][t], v["roll_b"]["r"][t]))
        pairs.append((f"CSM({t})", v["roll"]["csm"][t], v["roll_b"]["csm"][t]))
        pairs.append((f"LC({t})", v["roll"]["lc"][t], v["roll_b"]["lc"][t]))
    pairs.append(("F_prev(tau)", v["inputs"]["f_prev_tau"], v["b"]["REC"]["g"]["fcf"][TAU]))
    for label, a, bb in pairs:
        d = rel(a, bb)
        if a == 0.0 or bb == 0.0:
            assert a == bb, f"A/B zero mismatch {label}: {a!r} {bb!r}"
            d = 0.0
        count += 1
        if d > worst:
            worst = d
            worst_label = label
    return worst, count, worst_label


# ------------------------------------------------------------------ the shifted world (12.3, 14.8)
def shifted_world(v):
    """Z1: with no experience, no basis change and no change in financial inputs, the runs at V
    are the recognition run shifted by tau months. Z2: a new run with the old inputs gives
    delta FCF^L = 0 exactly."""
    mps, ids, rec_run = v["mps"], v["ids"], v["rec"]
    inst = {pid: v["rec"]["inst"][pid] for pid in ids}
    n_exp = v["inputs"]["n_exp"]

    def i_act_z(u):  # actual CPI equal to the recognition path
        return i_rec(TAU + u)

    def i_lock_z(u):
        return i_act_z(u) if u <= 0 else i_act_z(0) * i_rec(TAU0 + u) / i_rec(TAU0)

    maint_z = MAINT_V1 * i_act_z(0) / i_act_z(-TAU)
    old_z = run_group(make_run("OLD-Z", M_V, MORT_V1, maint_z, p_lock, i_lock_z, n_exp, inst, mps))
    new_z = run_group(make_run("NEW-Z", M_V, MORT_V1, maint_z, p_lock, i_lock_z, n_exp, inst, mps))
    delta_z = new_z["g"]["fcf"][0] - old_z["g"]["fcf"][0]
    assert delta_z == 0.0, "Z2"
    T = old_z["T"]
    worst = 0.0
    for pid in ids:
        a, b = old_z["pols"][pid], rec_run["pols"][pid]
        scale = b["bel"][TAU]
        for t in range(T + 1):
            for c in ("pols_if", "bel", "ra"):
                worst = max(worst, abs(a[c][t] - b[c][TAU + t]) / scale)
                assert abs(a[c][t] - b[c][TAU + t]) <= 1e-12 * scale, f"Z1 {pid} {c} t={t}"
        for t in range(1, T + 1):
            worst = max(worst, abs(a["cu"][t] - b["cu"][TAU + t]) / scale)
            worst = max(worst, abs(a["ann_pp"][t] - b["ann_pp"][TAU + t]) / b["ann_pp"][TAU + t])
            assert abs(a["cu"][t] - b["cu"][TAU + t]) <= 1e-12 * scale, f"Z1 cu {pid} t={t}"
    n_z = cu_next(old_z["g"]["cu"], p_lock, T)
    n_rec = v["rec_roll"]["n"]
    worst = max(worst, rel(n_z[0], n_rec[TAU]))
    assert within(old_z["g"]["fcf"][0], v["inputs"]["f_prev_tau"], v["inputs"]["p_g"]), "Z1 FCF"
    worst = max(worst, rel(old_z["g"]["fcf"][0], v["inputs"]["f_prev_tau"]))
    # named simplification: one release per period against twelve monthly steps (14.8)
    csm_prev, lc_prev = v["recog"]["csm_0"], v["recog"]["lc_0"]
    close_z = close_period(
        csm_prev,
        lc_prev,
        v["inputs"]["cu_exp"],
        v["inputs"]["f_prev_0"],
        v["inputs"]["f_prev_tau"],
        0.0,
        n_z[0],
    )
    return {
        "worst": worst,
        "fcf_z": old_z["g"]["fcf"][0],
        "n_z0": n_z[0],
        "close_z": close_z,
        "csm_rec_12": v["rec_roll"]["csm"][TAU],
        "lc_rec_12": v["rec_roll"]["lc"][TAU],
        "delta_z": delta_z,
    }


def annuity_spec_examples():
    """The projection of section 6 is the vault's annuity spec model restated. This reproduces the
    values that note prints for its examples 1 to 3 (policy 1 level, policy 3 CPI-linked, both
    new at 2025-12-31), to every printed digit."""
    mps = [
        make_mp(1, "level", "M", (1960, 6, 15), 1, 1000.0),
        make_mp(3, "cpi", "M", (1955, 2, 20), 1, 1200.0),
    ]
    run = run_group(
        make_run(
            "AS",
            M_REC,
            MORT_V1,
            MAINT_V1,
            p_rec,
            i_rec,
            {1: 1.0, 3: 1.0},
            {1: 1000.0, 3: 1200.0},
            mps,
        )
    )
    expected = {
        (1, "bel_0"): "179192.148",
        (1, "ra_0"): "2648.249534",
        (1, "pva"): "175222.1063",
        (1, "pve"): "3970.041706",
        (3, "bel_0"): "234466.2049",
        (3, "ra_0"): "5607.325647",
        (3, "pva"): "229894.987",
        (3, "pve"): "4571.217872",
    }
    for (pid, k), want in expected.items():
        got = f"{run['pols'][pid]['scalars'][k]:.10g}"
        assert got == want, (pid, k, got, want)
    a13 = run["pols"][3]["ann_pp"][13]
    assert f"{a13:.10g}" == "1232.981869", a13
    one = [mps[0]]
    run1 = run_group(
        make_run("AS1", M_REC, MORT_V1, MAINT_V1, p_rec, i_rec, {1: 1.0}, {1: 1000.0}, one)
    )
    one[0]["single_premium"] = 190000.0
    rg = recognise(run1)
    roll = monthly_roll(
        rg["csm_0"], rg["lc_0"], run1["g"]["cu"], run1["g"]["fcf"], p_rec, run1["T"]
    )
    assert f"{rg['csm_0']:.10g}" == "8159.602426", rg
    assert f"{roll['csm'][12]:.10g}" == "7854.527657", roll["csm"][12]
    assert f"{roll['r'][1]:.10g}" == "46.5678638", roll["r"][1]
    assert f"{roll['r'][12]:.10g}" == "46.31473923", roll["r"][12]
    one[0]["single_premium"] = 175000.0
    rg2 = recognise(run1)
    roll2 = monthly_roll(
        rg2["csm_0"], rg2["lc_0"], run1["g"]["cu"], run1["g"]["fcf"], p_rec, run1["T"]
    )
    assert f"{rg2['lc_0']:.10g}" == "6840.397574", rg2
    assert f"{roll2['lc'][12]:.10g}" == "6585.155578", roll2["lc"][12]
    print(
        f"annuity spec examples reproduced: T = {run['T']}; policy 1 bel(0) = {run['pols'][1]['scalars']['bel_0']!r}, "
        f"ra(0) = {run['pols'][1]['scalars']['ra_0']!r}; policy 3 bel(0) = {run['pols'][3]['scalars']['bel_0']!r}, "
        f"ann_pp(13) = {a13!r}; example 1 CSM(0) = {rg['csm_0']!r}, CSM(12) = {roll['csm'][12]!r}, R(1) = {roll['r'][1]!r}; "
        f"example 2 LC(0) = {rg2['lc_0']!r}, LC(12) = {roll2['lc'][12]!r}"
    )


def main():
    print("=== the projection of section 6 against the annuity spec's examples ===")
    annuity_spec_examples()
    print("\n=== dates and constants (sections 4 to 6) ===")
    print(f"m_rec = {M_REC}  m_V = {M_V}  tau0 = {TAU0}  tau = {TAU}  m_start = {M_START}")
    print(f"m_c at recognition = {M_START - M_REC}; m_c at V = {M_START - M_V}")
    print(f"ie_m = {IE_M!r}   1 + ie_m = {1 + IE_M!r}")
    print(f"mort_mult = {MORT_MULT!r}   exp_mult = {EXP_MULT!r}   z_alpha = {Z_ALPHA!r}")
    print(f"(c - 1) / ln(c) = {(C_MK - 1) / math.log(C_MK)!r}")
    print("--- mortality checkpoints (base v1, base v2 = 0.96 x v1; 8 s.f.) ---")
    for x in (65, 70, 75, 79, 80, 81, 85, 90, 100, 110, 119, 120):
        print(
            f"| {x} | {MORT_V1[('M', x)]:.8g} | {MORT_V1[('F', x)]:.8g} | {MORT_V2[('M', x)]:.8g} | {MORT_V2[('F', x)]:.8g} |"
        )
    print(
        f"check q_M(65) = {MORT_V1[('M', 65)]:.6f}, q_M(100) = {MORT_V1[('M', 100)]:.6f} (SOA table, 6 d.p.)"
    )
    print("--- improvement checkpoints ---")
    for x, y in (
        (65, 2026),
        (65, 2036),
        (65, 2046),
        (95, 2026),
        (110, 2030),
        (79, 2027),
        (81, 2027),
    ):
        print(f"r({x}, {y}) = {r_imp(x, y)!r}   cif({x}, {y}) = {CIF[(x, y)]!r}")
    print("--- curves (repr) ---")
    for t in (0, 1, 11, 12, 13, 24, 120, 489, 501):
        print(f"t = {t}: P_rec = {p_rec(t)!r}   P_cur = {p_cur(t)!r}")
    for t in (0, 1, 12, 13, 120, 489):
        print(
            f"t = {t}: P_L = {p_lock(t)!r}   j_L({t}) = {(p_lock(t - 1) / p_lock(t) - 1) if t >= 1 else float('nan')!r}"
        )
    print(
        f"accretion factor over the period P_rec(0) / P_rec(12) = {p_rec(TAU0 - TAU) / p_rec(TAU0)!r}"
    )
    print("--- CPI (repr) ---")
    for u in (-14, -13, -12, -2, -1, 0):
        print(f"u = {u}: I_act = {i_act(u)!r}   (I_rec({TAU + u}) = {i_rec(TAU + u)!r})")
    for u in (1, 10, 12, 13, 22, 24, 489):
        print(
            f"u = {u}: I_L = {i_lock(u)!r}   I_cur = {i_cur(u)!r}   I_rec({TAU0 + u}) = {i_rec(TAU0 + u)!r}"
        )
    e1 = i_act(-2) / i_act(-14)
    print(
        f"E(1) at V = I_act(-2) / I_act(-14) = {e1!r}; expected at recognition E_rec(13) = I_rec(10) / I_rec(-2) = {i_rec(10) / i_rec(-2)!r}"
    )
    print(
        f"E_L(13) = I_L(10) / I_L(-2) = {i_lock(10) / i_lock(-2)!r}; E_cur(13) = {i_cur(10) / i_cur(-2)!r}"
    )
    print(
        f"expense rebase factor I_act(0) / I_act(-12) = {i_act(0) / i_act(-TAU)!r}; maint_pm_old = {MAINT_V1 * i_act(0) / i_act(-TAU)!r}; maint_pm_v2 = {MAINT_V2}"
    )
    print(
        f"expected over the period at recognition I_rec(12) / I_rec(0) = {i_rec(12) / i_rec(0)!r}"
    )

    variants = {k: build_variant(k) for k in ("P", "O")}
    worst_runs: dict[str, float] = {}
    for v in variants.values():
        for name in ("rec", "old", "new", "cur"):
            check_run(v[name], f"{v['key']}-{name.upper()}", worst_runs)

    print("\n=== contracts and derived values (sections 3, 13) ===")
    for v in variants.values():
        for mp in v["mps"]:
            pid = mp["policy_id"]
            a0r, a0v = v["rec"]["a0"][pid], v["cur"]["a0"][pid]
            print(
                f"| {pid} | {mp['cls']} | {mp['sex']} | {mp['dob'][0]}-{mp['dob'][1]:02d}-{mp['dob'][2]:02d} | "
                f"{mp['n_rec']:g} | {mp['inst']:.2f} | {mp['single_premium']:.2f} | {v['spec']['n_act'][pid]:g} | "
                f"a0_rec {a0r} ({a0r // 12}y{a0r % 12}m) T_rec {T_OMEGA - a0r} | a0_V {a0v} ({a0v // 12}y{a0v % 12}m) T_V {T_OMEGA - a0v} | x(1) {v['cur']['pols'][pid]['age'][1]} |"
            )
        print(
            f"group {v['key']}: T_rec = {v['rec']['T']}, T at V = {v['cur']['T']}, cu_act = {v['spec']['cu_act']!r} = {v['spec']['cu_act_text']}"
        )

    for v in variants.values():
        k = v["key"]
        print(f"\n=== variant {k} ({v['spec']['bucket']}) ===")
        print("--- recognition run REC at 2025-12-31 (sections 7.3, 3.4) ---")
        for pid in v["ids"]:
            s = v["rec"]["pols"][pid]["scalars"]
            print(
                f"{pid}: bel(0) = {s['bel_0']!r}  ra(0) = {s['ra_0']!r}  pva = {s['pva']!r}  pve = {s['pve']!r}  per life FCF = {(s['bel_0'] + s['ra_0']) / v['rec']['inforce'][pid]!r}"
            )
            n12 = v["rec"]["pols"][pid]["pols_if"][TAU]
            print(
                f"{pid}: expected pols_if(12) = {n12!r} (expected deaths {v['rec']['inforce'][pid] - n12!r}); actual at V = {v['spec']['n_act'][pid]!r} (actual deaths {v['rec']['inforce'][pid] - v['spec']['n_act'][pid]!r})"
            )
        rg = v["recog"]
        print(
            f"P_g = {rg['p_g']!r}  FCF_g(0) = {rg['fcf_0']!r}  CSM(0) = {rg['csm_0']!r}  LC(0) = {rg['lc_0']!r}"
        )
        print(
            f"F_prev(0) = {v['inputs']['f_prev_0']!r}  F_prev(12) = {v['inputs']['f_prev_tau']!r}  ratio = {v['inputs']['f_prev_tau'] / v['inputs']['f_prev_0']!r}"
        )
        print(
            f"expected units of the period sum CU_g^rec(1..12) = {v['inputs']['cu_exp']!r}  (actual cu_act = {v['inputs']['cu_act']!r})"
        )
        print(
            f"diagnostic: REC monthly roll CSM(12) = {v['rec_roll']['csm'][TAU]!r}  LC(12) = {v['rec_roll']['lc'][TAU]!r}  N^rec(12) = {v['rec_roll']['n'][TAU]!r}"
        )
        print("--- the three runs at V (sections 5.5 and 7) ---")
        print(f"maint_pm_old = {v['maint_old']!r}")
        for name in ("old", "new", "cur", "exp_only"):
            run = v[name]
            for pid in v["ids"]:
                s = run["pols"][pid]["scalars"]
                pr = run["pols"][pid]
                print(
                    f"{run['name']} {pid}: pols_if(0) = {run['inforce'][pid]!r}  ann_pp(1) = {s['ann_pp_1']!r}  bel(0) = {s['bel_0']!r}  ra(0) = {s['ra_0']!r}  pva = {s['pva']!r}  pve = {s['pve']!r}  v_be(0) = {pr['v_be'][0]!r}  v_st(0) = {pr['v_st'][0]!r}"
                )
            print(
                f"{run['name']}: BEL_g(0) = {run['g']['bel'][0]!r}  RA_g(0) = {run['g']['ra'][0]!r}  FCF_g(0) = {run['g']['fcf'][0]!r}"
            )
        print("--- change in locked-in fulfilment cash flows (section 8) ---")
        d_exp = v["exp_only"]["g"]["fcf"][0] - v["old"]["g"]["fcf"][0]
        d_bas = v["new"]["g"]["fcf"][0] - v["exp_only"]["g"]["fcf"][0]
        print(
            f"delta FCF^L = {v['new']['g']['fcf'][0]!r} - {v['old']['g']['fcf'][0]!r} = {v['delta']!r}"
        )
        print(
            f"  diagnostic split: in-force experience {d_exp!r} + basis change {d_bas!r} = {d_exp + d_bas!r}"
        )
        print("--- period close (section 9) ---")
        c, inp = v["close"], v["inputs"]
        print(
            f"csm_prev = {inp['csm_prev']!r}  lc_prev = {inp['lc_prev']!r}  cu_act = {inp['cu_act']!r}  N(0) = {v['n'][0]!r}"
        )
        print(f"1 accrete: CSM_acc = {c['csm_acc']!r}  (accretion {c['acc_per']!r})")
        print(f"2 allocate: LC_alloc = {c['lc_alloc']!r}  (allocation {c['lca_per']!r})")
        print(
            f"3 remeasure: CSM_adj = {c['csm_adj']!r}  LC_adj = {c['lc_adj']!r}  (CSM change for future service {c['csm_fs']!r}; LC loss/reversal {c['lc_fs']!r})"
        )
        print(
            f"4 release: R_per = {c['r_per']!r}  share cu_act / (cu_act + N(0)) = {inp['cu_act'] / (inp['cu_act'] + v['n'][0])!r}"
        )
        print(f"5 close: CSM(0) = {c['csm_0']!r}  LC(0) = {c['lc_0']!r}")
        print(f"LRC(0) = FCF_g^cur(0) + CSM(0) = {v['cur']['g']['fcf'][0] + c['csm_0']!r}")
        print("--- months 1..12 after the close (section 10) ---")
        r = v["roll"]
        print("| t | CU_g(t) | N(t) | j_L(t) | Acc(t) | R(t) | CSM(t) | LC(t) | LCA(t) | F(t) |")
        for t in range(0, 13):
            if t == 0:
                print(
                    f"| 0 | n/a | {g10(r['n'][0])} | n/a | n/a | n/a | {g10(r['csm'][0])} | {g10(r['lc'][0])} | n/a | {g10(v['cur']['g']['fcf'][0])} |"
                )
            else:
                print(
                    f"| {t} | {g10(v['cur']['g']['cu'][t])} | {g10(r['n'][t])} | {r['j_l'][t]:.10g} | {g10(r['acc'][t])} | {g10(r['r'][t])} | {g10(r['csm'][t])} | {g10(r['lc'][t])} | {g10(r['lca'][t])} | {g10(v['cur']['g']['fcf'][t])} |"
                )
        print(
            f"full precision: CSM(1) = {r['csm'][1]!r}  CSM(12) = {r['csm'][12]!r}  LC(1) = {r['lc'][1]!r}  LC(12) = {r['lc'][12]!r}  R(1) = {r['r'][1]!r}"
        )
        last = max(t for t in range(1, v["cur"]["T"] + 1) if v["cur"]["g"]["cu"][t] > 0)
        print(
            f"last month with CU_g > 0: {last}; CSM({last}) = {r['csm'][last]!r}; CSM(T) = {r['csm'][-1]!r}; LC(T) = {r['lc'][-1]!r}"
        )
        print("--- analysis of change of the fulfilment cash flows (sections 11.2 and 14.7) ---")
        a_, b_, c_, d_, e_ = (
            rg["fcf_0"],
            inp["f_prev_tau"],
            v["old"]["g"]["fcf"][0],
            v["new"]["g"]["fcf"][0],
            v["cur"]["g"]["fcf"][0],
        )
        print(f"A FCF^rec_g(0) = {a_!r}")
        print(f"B F_prev(12)   = {b_!r}   B - A = {b_ - a_!r}")
        print(
            f"C FCF^L_old(0) = {c_!r}   C - B = {c_ - b_!r}   (CPI experience on future cash flows)"
        )
        print(f"D FCF^L_new(0) = {d_!r}   D - C = {d_ - c_!r}   (delta FCF^L)")
        print(f"E FCF^cur(0)   = {e_!r}   E - D = {e_ - d_!r}   (financial assumptions at V)")
        worst_ab, n_ab, label_ab = compare_ab(v)
        print(
            f"--- implementations A and B: {n_ab} values compared, max relative difference {worst_ab:.2e} ({label_ab}) ---"
        )
        assert worst_ab < 1e-11, worst_ab

    print("\n=== remeasurement branches (section 9.4, table of 14.5) ===")
    vp, vo = variants["P"], variants["O"]
    csm_a, lc_a = vp["close"]["csm_acc"], vo["close"]["lc_alloc"]
    dp = abs(vp["delta"])
    cases = [
        ("a", csm_a, 0.0, dp),
        ("b", csm_a, 0.0, 1.25 * csm_a),
        ("c", csm_a, 0.0, -dp),
        ("d", 0.0, lc_a, dp),
        ("e", 0.0, lc_a, -0.5 * lc_a),
        ("f", 0.0, lc_a, -1.5 * lc_a),
    ]
    print("| case | CSM_acc | LC_alloc | delta FCF^L | CSM_adj | LC_adj | conservation residual |")
    for name, csm, lc, d in cases:
        csm2, lc2 = remeasure(csm, lc, d)
        resid = (csm2 - lc2) - ((csm - lc) - d)
        assert abs(resid) <= 1e-12 * max(csm, lc, abs(d)), name
        assert csm2 >= 0, name
        assert lc2 >= 0, name
        assert not (csm2 > 0 and lc2 > 0), name
        print(
            f"| {name} | {g10(csm)} | {g10(lc)} | {g10(d)} | {g10(csm2)} | {g10(lc2)} | {resid:.2e} |"
        )
    print(
        f"variant P follows case {'a' if vp['delta'] >= 0 and vp['close']['csm_adj'] > 0 else '?'}; variant O follows case ",
        end="",
    )
    if vo["delta"] >= 0:
        print("d")
    elif -vo["delta"] <= vo["close"]["lc_alloc"]:
        print("e (partial reversal)")
    else:
        print("f (full reversal, remainder to the CSM)")

    print("\n=== invariants (section 12) ===")
    worst_close: dict[str, float] = {}
    for v in variants.values():
        check_close(v, worst_close)
    print(
        "I1-I10 passed on every policy of every run of both variants; worst relative residuals:",
        {k: f"{x:.2e}" for k, x in sorted(worst_runs.items())},
    )
    print(
        "K1-K7, I11-I13 passed on both variants; worst relative residuals:",
        {k: f"{x:.2e}" for k, x in sorted(worst_close.items())},
    )
    try:  # G6: a run with two scenarios is rejected by the aggregation stage
        mp0 = vp["mps"][0]
        bad = make_run(
            "BAD",
            M_V,
            MORT_V2,
            MAINT_V2,
            p_cur,
            i_cur,
            vp["spec"]["n_act"],
            dict.fromkeys(vp["ids"], 1.0),
            vp["mps"],
        )
        bad["scenarios"][1] = (p_lock, i_lock)
        project(mp0, bad)
        raise SystemExit("G6 did not fire")
    except AssertionError as e:
        print(f"G6: J = 2 rejected ({e})")
    print("--- shifted world Z1, Z2 (sections 12.3, 14.8) ---")
    for v in variants.values():
        z = shifted_world(v)
        print(
            f"{v['key']}: Z1 worst relative difference run-at-V against REC shifted by 12 months: {z['worst']:.2e}; FCF_z(0) = {z['fcf_z']!r} against F_prev(12) = {v['inputs']['f_prev_tau']!r}; N_z(0) = {z['n_z0']!r} against N^rec(12) = {v['rec_roll']['n'][TAU]!r}"
        )
        print(f"{v['key']}: Z2 delta FCF^L with unchanged inputs = {z['delta_z']!r}")
        cz = z["close_z"]
        print(
            f"{v['key']}: named simplification, one release per period: period-form CSM(0) = {cz['csm_0']!r} against twelve monthly steps CSM^rec(12) = {z['csm_rec_12']!r}; relative difference {(cz['csm_0'] / z['csm_rec_12'] - 1) if z['csm_rec_12'] else 0.0:+.3%}; LC: {cz['lc_0']!r} against {z['lc_rec_12']!r}"
        )


if __name__ == "__main__":
    main()
