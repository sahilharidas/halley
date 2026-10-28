# pyright: basic
"""The reference inputs of the annuity spec as Arrow tables.

The synthetic basis of ``docs/spec/annuity.md`` ("Worked example"): the Makeham base table with
the four-year female setback and q = 1 at 120, the improvement rates and the cumulative
improvement factors, the expense, investment-expense, illiquidity-premium and risk-adjustment
tables, the risk-free discount file and the CPI file, and the four example model points as three
class files. Every formula is written in the evaluation order the spec prints, so each value
equals ``docs/spec/in_force_group_numbers.py`` bit for bit (tests/test_annuity.py).
``docs/spec/annuity_inputs.py`` writes the tables to Parquet beside the spec.

Also here: the reference inputs of ``docs/spec/in-force-group.md`` (sections 3, 5 and 13), the
in-force group of milestone M10: its model points at recognition and at V, its groups rows with
the three locked-in hashes, the tables of versions 1 and 2, the curves and CPI paths, and the
files the cross-run helpers of :mod:`halley.aggregation.inforce` build from them (the rebased
expense table, the locked-in curve and CPI path, the composed current curve), each equal to the
in-force number script bit for bit (tests/test_inforce_helpers.py).
``docs/spec/in_force_group_inputs.py`` writes them beside the in-force spec. pyarrow ships no
type stubs, so, by the convention of every module that builds or reads Arrow tables
(``runtime/loader.py``, ``runtime/run.py``, ``products/term_assurance_data.py``), this one is
checked in pyright's basic mode and its builders return ``Any``; the model itself stays strict.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import math
from collections.abc import Callable
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from halley.aggregation.inforce import (
    composed_curve,
    locked_in_cpi,
    locked_in_curve,
    rebase_expense_table,
)
from halley.products.annuity import BASIS_VERSION, OMEGA, VALUATION_DATE
from halley.runtime.run import RunResult

# ------------------------------------------------------------------------- mortality
A_MK, B_MK, C_MK = 0.00022, 2.7e-6, 1.124
"""The Standard Ultimate Survival Model (Makeham), SOA 2022."""
AGE_LO = 20
BASE_YEAR = 2025
YEAR_HI = 2150
FEMALE_SETBACK = 4

# ------------------------------------------------------------------------ expenses and RA
MAINT_PM = 5.00
"""Per policy per month, in December 2025 prices."""
PRICE_BASE = f"{VALUATION_DATE.year:04d}-{VALUATION_DATE.month:02d}"
IE_RATE = 0.0015
ILP = 0.0050
ALPHA = 0.75
Z_ALPHA = 0.6744897501960817
"""Phi^-1(0.75); implementations use this literal (spec, "Risk adjustment parameters")."""
SIGMA_L = 0.10
SIGMA_E = 0.10

# -------------------------------------------------------------------- curves and CPI path
RF_LEVEL = 0.040
RF_SHORT = 0.015
T_FILE = 655
"""The discount and CPI files hold every month up to T = 655 (policy 1's run-off)."""
CPI_BASE = 100.0
CPI_HISTORY = 1.04
CPI_PROJECTED = 1.025
CPI_U_LO = -24
"""The CPI file starts at u = -24; the model needs u >= -14."""


def q_makeham(x: int) -> float:
    """Spec "Mortality", in the order written: c^x, B c^x, (c - 1), / ln c, -A -, exp, 1 -."""
    return 1 - math.exp(-A_MK - B_MK * C_MK**x * (C_MK - 1) / math.log(C_MK))


def q_base(sex: str, x: int) -> float:
    """The base table: the female row is the male row four years younger; q = 1 at OMEGA."""
    if x == OMEGA:
        return 1.0
    return q_makeham(x - FEMALE_SETBACK if sex == "F" else x)


def improvement_rate(x: int, y: int) -> float:
    """r(x, y): the convergence form of SOA MP-2014 formula B.1 with Halley's parameters."""
    h = min(1.0, (y - 2026) / 20)
    w = 1 - 3 * h**2 + 2 * h**3
    g = 1.0 if x <= 85 else ((110 - x) / 25 if x < 110 else 0.0)
    return g * (0.010 + 0.010 * w)


def cumulative_improvement() -> dict[tuple[int, int], float]:
    """cif(x, y): 1 in the base year, then cif(x, y - 1) (1 - r(x, y)) in ascending year."""
    out: dict[tuple[int, int], float] = {}
    for x in range(AGE_LO, OMEGA + 1):
        c = 1.0
        out[(x, BASE_YEAR)] = c
        for y in range(BASE_YEAR + 1, YEAR_HI + 1):
            c = c * (1 - improvement_rate(x, y))
            out[(x, y)] = c
    return out


def rf_df(t: int, level: float = RF_LEVEL) -> float:
    """exp(-z(tau) tau) with z(tau) = level - 0.015 exp(-tau / 10), tau = t / 12."""
    tau = t / 12
    z = level - RF_SHORT * math.exp(-tau / 10)
    return math.exp(-z * tau)


def cpi_path(u: int) -> float:
    """100 x 1.04^(u/12) for u <= 0 (history), 100 x 1.025^(u/12) for u >= 0."""
    if u <= 0:
        return CPI_BASE * CPI_HISTORY ** (u / 12)
    return CPI_BASE * CPI_PROJECTED ** (u / 12)


# ------------------------------------------------------------------------------ tables


def _meta(**extra: str) -> dict[bytes, bytes]:
    meta = {b"halley.version": BASIS_VERSION.encode(), b"halley.basis": BASIS_VERSION.encode()}
    meta.update({f"halley.{k}".encode(): v.encode() for k, v in extra.items()})
    return meta


def parquet_bytes(table: Any) -> bytes:
    """The Parquet serialisation of a table, as the loader hashes an in-memory table."""
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


def sha256(table: Any) -> str:
    return hashlib.sha256(parquet_bytes(table)).hexdigest()


def mort_base_table(multiplier: float = 1.0) -> Any:
    """``mort_base``: sex, age, q. A multiplier below 1 is the in-force spec's version 2 (lighter
    mortality at every age below OMEGA)."""
    sexes: list[str] = []
    ages: list[int] = []
    qs: list[float] = []
    for sex in ("F", "M"):
        for x in range(AGE_LO, OMEGA + 1):
            q = q_base(sex, x)
            sexes.append(sex)
            ages.append(x)
            qs.append(q if x == OMEGA or multiplier == 1.0 else multiplier * q)
    return pa.table(
        {
            "sex": pa.array(sexes, pa.string()),
            "age": pa.array(ages, pa.int16()),
            "q": pa.array(qs, pa.float64()),
        }
    ).replace_schema_metadata(
        _meta(base_year=str(BASE_YEAR), source="Standard Ultimate Survival Model (Makeham)")
    )


def mort_improvement_table() -> Any:
    """``mort_improvement``: age, year, r for y = base_year + 1..2150. Not read by the model;
    its hash is carried by ``mort_cif``."""
    ages = [x for x in range(AGE_LO, OMEGA + 1) for _ in range(BASE_YEAR + 1, YEAR_HI + 1)]
    years = [y for _ in range(AGE_LO, OMEGA + 1) for y in range(BASE_YEAR + 1, YEAR_HI + 1)]
    rates = [improvement_rate(x, y) for x, y in zip(ages, years, strict=True)]
    return pa.table(
        {
            "age": pa.array(ages, pa.int16()),
            "year": pa.array(years, pa.int16()),
            "r": pa.array(rates, pa.float64()),
        }
    ).replace_schema_metadata(_meta(base_year=str(BASE_YEAR), source="SOA MP-2014 formula B.1"))


def mort_cif_table(improvement: Any | None = None) -> Any:
    """``mort_cif``: age, year, cif for y = base_year..2150, with the improvement table's hash
    in its metadata."""
    improvement = mort_improvement_table() if improvement is None else improvement
    cif = cumulative_improvement()
    keys = sorted(cif)
    return pa.table(
        {
            "age": pa.array([x for x, _ in keys], pa.int16()),
            "year": pa.array([y for _, y in keys], pa.int16()),
            "cif": pa.array([cif[k] for k in keys], pa.float64()),
        }
    ).replace_schema_metadata(
        _meta(base_year=str(BASE_YEAR), mort_improvement_sha256=sha256(improvement))
    )


def expense_table(maint_pm: float = MAINT_PM, price_base: str = PRICE_BASE) -> Any:
    return pa.table({"maint_pm": pa.array([maint_pm], pa.float64())}).replace_schema_metadata(
        _meta(price_base=price_base)
    )


def inv_expense_table(ie_rate: float = IE_RATE) -> Any:
    return pa.table({"ie_rate": pa.array([ie_rate], pa.float64())}).replace_schema_metadata(_meta())


def ilp_table(ilp: float = ILP) -> Any:
    return pa.table({"ilp": pa.array([ilp], pa.float64())}).replace_schema_metadata(_meta())


def ra_param_table(
    alpha: float = ALPHA,
    z_alpha: float = Z_ALPHA,
    sigma_l: float = SIGMA_L,
    sigma_e: float = SIGMA_E,
) -> Any:
    return pa.table(
        {
            "alpha": pa.array([alpha], pa.float64()),
            "z_alpha": pa.array([z_alpha], pa.float64()),
            "sigma_l": pa.array([sigma_l], pa.float64()),
            "sigma_e": pa.array([sigma_e], pa.float64()),
        }
    ).replace_schema_metadata(_meta())


def reference_tables() -> dict[str, Any]:
    """The six assumption tables the model reads, from the spec's formulas."""
    return {
        "mort_base": mort_base_table(),
        "mort_cif": mort_cif_table(),
        "expense": expense_table(),
        "inv_expense": inv_expense_table(),
        "ilp": ilp_table(),
        "ra_param": ra_param_table(),
    }


# ----------------------------------------------------------------------- scenario files


def discount_table(
    fn: Callable[[int, int], float] = lambda s, t: rf_df(t),
    t_max: int = T_FILE,
    n_scenarios: int = 1,
    **metadata: str,
) -> Any:
    """``discount``: scenario, t, rf_df for t = 0..t_max and every scenario."""
    scen = [s for s in range(n_scenarios) for _ in range(t_max + 1)]
    months = [t for _ in range(n_scenarios) for t in range(t_max + 1)]
    values = [fn(s, t) for s, t in zip(scen, months, strict=True)]
    return pa.table(
        {
            "scenario": pa.array(scen, pa.int32()),
            "t": pa.array(months, pa.int32()),
            "rf_df": pa.array(values, pa.float64()),
        }
    ).replace_schema_metadata(_meta(**metadata))


def cpi_table(
    fn: Callable[[int, int], float] = lambda s, u: cpi_path(u),
    u_lo: int = CPI_U_LO,
    u_hi: int = T_FILE,
    n_scenarios: int = 1,
    **metadata: str,
) -> Any:
    """``cpi``: scenario, u, cpi for u = u_lo..u_hi and every scenario."""
    scen = [s for s in range(n_scenarios) for _ in range(u_lo, u_hi + 1)]
    months = [u for _ in range(n_scenarios) for u in range(u_lo, u_hi + 1)]
    values = [fn(s, u) for s, u in zip(scen, months, strict=True)]
    return pa.table(
        {
            "scenario": pa.array(scen, pa.int32()),
            "u": pa.array(months, pa.int32()),
            "cpi": pa.array(values, pa.float64()),
        }
    ).replace_schema_metadata(_meta(**metadata))


def reference_scenarios() -> dict[str, Any]:
    """The discount file (z_rf(tau) = 0.040 - 0.015 exp(-tau / 10), t = 0..655) and the CPI
    file (u = -24..655), keyed by the model's scenario input names."""
    return {
        "rf_df": discount_table(
            source="z_rf(tau) = 0.040 - 0.015 exp(-tau / 10), rf_df = exp(-z tau)"
        ),
        "cpi": cpi_table(
            source="100 x 1.04^(u/12) for u <= 0, 100 x 1.025^(u/12) for u >= 0",
            estimated_rows="none",
        ),
    }


# ------------------------------------------------------------------------ model points

EXAMPLE_MODEL_POINTS: tuple[dict[str, Any], ...] = (
    {
        "cls": "level",
        "policy_id": 1,
        "portfolio": "IA",
        "cohort": 2025,
        "profit_bucket": "OTHER",
        "sex": "M",
        "date_of_birth": dt.date(1960, 6, 15),
        "issue_date": dt.date(2025, 12, 31),
        "annuity_start_date": dt.date(2026, 1, 1),
        "pols_if_init": 1.0,
        "instalment_pm": 1_000.00,
        "single_premium": 190_000.00,
    },
    {
        "cls": "fixed",
        "policy_id": 2,
        "portfolio": "IA",
        "cohort": 2025,
        "profit_bucket": "OTHER",
        "sex": "F",
        "date_of_birth": dt.date(1957, 10, 10),
        "issue_date": dt.date(2025, 12, 31),
        "annuity_start_date": dt.date(2026, 1, 1),
        "pols_if_init": 1.0,
        "instalment_pm": 800.00,
        "single_premium": 220_000.00,
        "esc_rate": 0.03,
    },
    {
        "cls": "cpi",
        "policy_id": 3,
        "portfolio": "IA",
        "cohort": 2025,
        "profit_bucket": "OTHER",
        "sex": "M",
        "date_of_birth": dt.date(1955, 2, 20),
        "issue_date": dt.date(2025, 12, 31),
        "annuity_start_date": dt.date(2026, 1, 1),
        "pols_if_init": 1.0,
        "instalment_pm": 1_200.00,
        "single_premium": 250_000.00,
    },
    {
        "cls": "level",
        "policy_id": 4,
        "portfolio": "IA",
        "cohort": 2025,
        "profit_bucket": "ONEROUS",
        "sex": "M",
        "date_of_birth": dt.date(1960, 6, 15),
        "issue_date": dt.date(2025, 12, 31),
        "annuity_start_date": dt.date(2026, 1, 1),
        "pols_if_init": 1.0,
        "instalment_pm": 1_000.00,
        "single_premium": 175_000.00,
    },
)
"""Spec "Model points": the four example policies, exactly as printed; ``cls`` names the class
file (``mp_level``, ``mp_fixed``, ``mp_cpi``)."""

CLASS_FILES = {"level": "mp_level", "fixed": "mp_fixed", "cpi": "mp_cpi"}


def model_points_table(
    rows: list[dict[str, Any]], *, fixed: bool = False, valuation_date: dt.date = VALUATION_DATE
) -> Any:
    """A model point file with the spec's Arrow types: int64 id, dictionary-encoded portfolio,
    int16 cohort, enums as strings, date32 dates, float64 amounts; ``esc_rate`` for the fixed
    class. A ``cls`` entry on a row is ignored; ``valuation_date`` is recorded in the metadata."""
    cols: dict[str, Any] = {
        "policy_id": pa.array([int(r["policy_id"]) for r in rows], pa.int64()),
        "portfolio": pa.array([r["portfolio"] for r in rows], pa.string()).dictionary_encode(),
        "cohort": pa.array([int(r["cohort"]) for r in rows], pa.int16()),
        "profit_bucket": pa.array([r["profit_bucket"] for r in rows], pa.string()),
        "sex": pa.array([r["sex"] for r in rows], pa.string()),
        "date_of_birth": pa.array([r["date_of_birth"] for r in rows], pa.date32()),
        "issue_date": pa.array([r["issue_date"] for r in rows], pa.date32()),
        "annuity_start_date": pa.array([r["annuity_start_date"] for r in rows], pa.date32()),
        "pols_if_init": pa.array([float(r["pols_if_init"]) for r in rows], pa.float64()),
        "instalment_pm": pa.array([float(r["instalment_pm"]) for r in rows], pa.float64()),
        "single_premium": pa.array([float(r["single_premium"]) for r in rows], pa.float64()),
    }
    if fixed:
        cols["esc_rate"] = pa.array([float(r["esc_rate"]) for r in rows], pa.float64())
    return pa.table(cols).replace_schema_metadata(_meta(valuation_date=valuation_date.isoformat()))


def example_model_points() -> dict[str, Any]:
    """The three class files of the worked example: ``mp_level`` (policies 1 and 4),
    ``mp_fixed`` (policy 2) and ``mp_cpi`` (policy 3)."""
    out: dict[str, Any] = {}
    for cls, name in CLASS_FILES.items():
        rows = [r for r in EXAMPLE_MODEL_POINTS if r["cls"] == cls]
        out[name] = model_points_table(rows, fixed=cls == "fixed")
    return out


def example_policy(policy_id: int) -> dict[str, Any]:
    """One example row by its id, with ``cls``."""
    return next(dict(r) for r in EXAMPLE_MODEL_POINTS if r["policy_id"] == policy_id)


# ------------------------------------------------------------- the in-force group (in-force spec)

IF_RECOGNITION_DATE = dt.date(2025, 12, 31)
"""V_rec, the recognition date and the previous valuation date (in-force 3.1)."""
IF_VALUATION_DATE = dt.date(2026, 12, 31)
"""V, the in-force group's valuation date."""
IF_TAU0 = 12
"""Months from recognition to V; also tau, the period (in-force C1)."""
IF_T_REC = 501
"""The recognition run's horizon, 12 x 120 + 1 - 940 (in-force 13)."""
IF_T = 489
"""The horizon of every run at V."""
IF_CPI_U_LO = -26
"""The first month of the CPI files at V: u >= -14 is read, -26 leaves twelve months of margin."""
RF_LEVEL_CUR = 0.045
"""The long rate of the current curve at V, 50 basis points above recognition's (5.3)."""
MORT_MULT_V2 = 0.96
"""Version 2 of the base table: 0.96 x version 1 below OMEGA (lighter mortality, 3.5)."""
MAINT_PM_V2 = 5.30
"""Version 2 of the expense table, in December 2026 prices."""
CPI_DEC25_PUBLISHED = 100.1
"""The December 2025 index as published, revised from the 100.0 estimated at recognition."""
CPI_ACTUAL = 1.032
"""Actual inflation over the period, a year."""
CPI_CURRENT = 1.03
"""Expected inflation at V, a year."""
IF_BASIS_VERSION = "in-force-ref-basis-0.1"


def price_base(d: dt.date) -> str:
    """The ``YYYY-MM`` price base of a valuation date (``halley.price_base``)."""
    return f"{d.year:04d}-{d.month:02d}"


def cpi_actual(u: int) -> float:
    """I_act(u), u <= 0: actual history as known at V. Before December 2025 the recognition
    path's history (unchanged); from December 2025, published at 100.1, 3.2% a year (5.2)."""
    if u <= -(IF_TAU0 + 1):
        return CPI_BASE * CPI_HISTORY ** ((u + IF_TAU0) / 12)
    return CPI_DEC25_PUBLISHED * CPI_ACTUAL ** ((u + IF_TAU0) / 12)


def cpi_current(u: int) -> float:
    """I_cur(u): the actual history, then current expected inflation of 3.0% a year from the
    actual level at V (5.3)."""
    if u <= 0:
        return cpi_actual(u)
    return cpi_actual(0) * CPI_CURRENT ** (u / 12)


IN_FORCE_MODEL_POINTS: dict[str, tuple[dict[str, Any], ...]] = {
    variant: tuple(
        {
            "cls": cls,
            "policy_id": pid,
            "portfolio": "IA",
            "cohort": 2025,
            "profit_bucket": bucket,
            "sex": sex,
            "date_of_birth": dob,
            "issue_date": IF_RECOGNITION_DATE,
            "annuity_start_date": dt.date(2026, 1, 1),
            "pols_if_init": n_rec,
            "instalment_pm": instalment,
            "single_premium": premium,
            "pols_if_at_v": n_act,
        }
        for pid, cls, sex, dob, n_rec, instalment, premium, n_act in rows
    )
    for variant, bucket, rows in (
        (
            "P",
            "OTHER",
            (
                (11, "level", "M", dt.date(1947, 8, 15), 100.0, 1000.00, 126250.60, 98.0),
                (12, "cpi", "F", dt.date(1945, 11, 30), 50.0, 800.00, 132916.18, 50.0),
            ),
        ),
        (
            "O",
            "ONEROUS",
            (
                (21, "level", "M", dt.date(1947, 8, 15), 100.0, 1000.00, 118966.91, 95.0),
                (22, "cpi", "F", dt.date(1945, 11, 30), 50.0, 800.00, 125247.94, 47.0),
            ),
        ),
    )
}
"""In-force section 13: the same two lives under each variant's policy ids, with the lives at
recognition, the instalment (unchanged at V: the first anniversary falls in month 1 of the runs
at V), the premium and the actual survivors at V (``pols_if_at_v``)."""

IN_FORCE_CU_ACT = {"P": 1000.00 * 1187 + 800.00 * 600, "O": 1000.00 * 1170 + 800.00 * 588}
"""``cu_act``: the payments actually made in the period, two totals (in-force 13)."""

IN_FORCE_BUCKETS = {"P": "OTHER", "O": "ONEROUS"}


def in_force_model_points(variant: str, at: str) -> dict[str, Any]:
    """The two class files of a variant, ``mp_level`` and ``mp_cpi``, at recognition (``at =
    "rec"``: the lives at recognition) or at V (``"v"``: the actual survivors, the instalment
    at V and the premium carried from recognition)."""
    if at not in ("rec", "v"):
        raise ValueError(f"at is 'rec' or 'v', not {at!r}")
    when = IF_RECOGNITION_DATE if at == "rec" else IF_VALUATION_DATE
    out: dict[str, Any] = {}
    for cls in ("level", "cpi"):
        rows = [dict(r) for r in IN_FORCE_MODEL_POINTS[variant] if r["cls"] == cls]
        if at == "v":
            for r in rows:
                r["pols_if_init"] = r["pols_if_at_v"]
        out[f"mp_{cls}"] = model_points_table(rows, valuation_date=when)
    return out


def _if_meta(**extra: str) -> dict[bytes, bytes]:
    meta = {
        b"halley.version": IF_BASIS_VERSION.encode(),
        b"halley.basis": IF_BASIS_VERSION.encode(),
    }
    meta.update({f"halley.{k}".encode(): v.encode() for k, v in extra.items()})
    return meta


def in_force_tables() -> dict[str, Any]:
    """The assumption tables of in-force 3.5: version 1 (the basis at recognition, the annuity
    spec's) and version 2 (adopted at V) of the base table and the expense table, each expense
    table with its ``price_base``; the tables that do not change; and ``ilp_zero``, the ILP
    table of 0.0 the locked-in runs read (C17: ``exp(0) = 1`` leaves ``P_L`` unchanged)."""
    return {
        "mort_base_v1": mort_base_table(),
        "mort_base_v2": mort_base_table(MORT_MULT_V2),
        "mort_cif": mort_cif_table(),
        "mort_improvement": mort_improvement_table(),
        "expense_v1": expense_table(MAINT_PM, price_base(IF_RECOGNITION_DATE)),
        "expense_v2": expense_table(MAINT_PM_V2, price_base(IF_VALUATION_DATE)),
        "inv_expense": inv_expense_table(),
        "ilp": ilp_table(),
        "ilp_zero": ilp_table(0.0),
        "ra_param": ra_param_table(),
    }


def cpi_actual_table() -> Any:
    """``cpi_act``: u = -26..0, the history as known at V; one path, so no scenario column. The
    December 2026 index (u = 0) is an estimate at V, flagged and used as actual (C9)."""
    months = list(range(IF_CPI_U_LO, 1))
    return pa.table(
        {
            "u": pa.array(months, pa.int32()),
            "cpi": pa.array([cpi_actual(u) for u in months], pa.float64()),
        }
    ).replace_schema_metadata(
        _if_meta(
            source="100 x 1.04^((u + 12)/12) for u <= -13, 100.1 x 1.032^((u + 12)/12) after",
            estimated_rows="0",
        )
    )


def in_force_scenarios() -> dict[str, Any]:
    """The scenario files of in-force 3.6 that are data: the risk-free curves at recognition
    (t = 0..501) and at V (t = 0..489), the recognition run's CPI path (u = -24..501, indexed
    from the recognition month, whose u = 0 was an estimate then), the actual history at V and
    the current CPI path at V (u = -26..489)."""
    return {
        "discount_rec": discount_table(
            t_max=IF_T_REC,
            source="z_rec(tau) = 0.040 - 0.015 exp(-tau / 10), rf_df = exp(-z tau)",
        ),
        "discount_cur": discount_table(
            lambda s, t: rf_df(t, RF_LEVEL_CUR),
            t_max=IF_T,
            source="z_cur(tau) = 0.045 - 0.015 exp(-tau / 10), rf_df = exp(-z tau)",
        ),
        "cpi_rec": cpi_table(
            u_hi=IF_T_REC,
            source="100 x 1.04^(u/12) for u <= 0, 100 x 1.025^(u/12) for u >= 0",
            estimated_rows="0",
        ),
        "cpi_act": cpi_actual_table(),
        "cpi_cur": cpi_table(
            lambda s, u: cpi_current(u),
            u_lo=IF_CPI_U_LO,
            u_hi=IF_T,
            source="I_act(u) for u <= 0, I_act(0) x 1.03^(u/12) for u > 0",
            estimated_rows="0",
        ),
    }


def in_force_derived(tables: dict[str, Any], scenarios: dict[str, Any]) -> dict[str, Any]:
    """The files the cross-run helpers build from the data (design section 10), each naming its
    sources by hash: ``expense_old``, version 1 rebased to V prices with actual CPI (5.4);
    ``p_rec`` and ``p_l``, the composed curve at recognition and the locked-in curve re-based to
    V (5.1); ``cpi_locked``, the locked-in CPI splice (5.2); ``df_cur``, the composed current
    curve at V (5.3), the para 120 curve."""
    p_rec, p_l = locked_in_curve(scenarios["discount_rec"], tables["ilp"], IF_TAU0, IF_T)
    return {
        "expense_old": rebase_expense_table(
            tables["expense_v1"], scenarios["cpi_act"], IF_TAU0, price_base(IF_VALUATION_DATE)
        ),
        "p_rec": p_rec,
        "p_l": p_l,
        "cpi_locked": locked_in_cpi(scenarios["cpi_act"], scenarios["cpi_rec"], IF_TAU0, IF_T),
        "df_cur": composed_curve(scenarios["discount_cur"], tables["ilp"], IF_T),
    }


def _groups_table(columns: dict[str, Any]) -> Any:
    return pa.table(columns).replace_schema_metadata(_if_meta())


def recognition_groups(variant: str) -> Any:
    """The groups row of a variant at recognition: the group key and the recognition date (a
    group new at V needs no other group input)."""
    return _groups_table(
        {
            "portfolio": pa.array(["IA"], pa.string()),
            "cohort": pa.array([2025], pa.int16()),
            "profit_bucket": pa.array([IN_FORCE_BUCKETS[variant]], pa.string()),
            "recognition_date": pa.array([IF_RECOGNITION_DATE], pa.date32()),
        }
    )


def in_force_groups(variant: str, tables: dict[str, Any], scenarios: dict[str, Any]) -> Any:
    """The groups row of a variant at V (in-force 3.3, 13) as far as it is data: the key, the
    dates, ``cu_act`` and the locked-in basis by hash (the recognition discount file, the ILP
    table and the recognition CPI file). The fields read from the recognition run,
    ``prev_run_id``, ``csm_prev`` and ``lc_prev``, are never retyped: :func:`with_previous_run`
    adds them from that run."""
    hashes = {
        "discount_rec_sha256": sha256(scenarios["discount_rec"]),
        "ilp_sha256": sha256(tables["ilp"]),
        "cpi_rec_sha256": sha256(scenarios["cpi_rec"]),
    }
    return _groups_table(
        {
            "portfolio": pa.array(["IA"], pa.string()),
            "cohort": pa.array([2025], pa.int16()),
            "profit_bucket": pa.array([IN_FORCE_BUCKETS[variant]], pa.string()),
            "recognition_date": pa.array([IF_RECOGNITION_DATE], pa.date32()),
            "prev_valuation_date": pa.array([IF_RECOGNITION_DATE], pa.date32()),
            "cu_act": pa.array([IN_FORCE_CU_ACT[variant]], pa.float64()),
            **{name: pa.array([value], pa.string()) for name, value in hashes.items()},
        }
    )


def with_previous_run(groups: Any, previous: RunResult) -> Any:
    """A groups file at V with the fields read from the previous aggregation run (in-force 3.3,
    3.4): ``csm_prev`` and ``lc_prev``, its closing ``csm(0)`` and ``lc(0)`` at full precision,
    and ``prev_run_id``, its run id."""
    gid = previous.ids.tolist()[0]
    return (
        groups.append_column("csm_prev", pa.array([float(previous.lane("csm", gid)[0])]))
        .append_column("lc_prev", pa.array([float(previous.lane("lc", gid)[0])]))
        .append_column("prev_run_id", pa.array([previous.run_id], pa.string()))
    )


def in_force_files() -> dict[str, Any]:
    """Every file ``docs/spec/in_force_group_inputs.py`` writes, by name: the model points per
    class, variant and date (``mp_level_P_rec``, ``mp_cpi_O_v``), the groups rows at recognition
    and at V, the tables, the scenario files and the files the helpers derive from them."""
    tables = in_force_tables()
    scenarios = in_force_scenarios()
    out: dict[str, Any] = {}
    for variant in ("P", "O"):
        for at in ("rec", "v"):
            for name, table in in_force_model_points(variant, at).items():
                out[f"{name}_{variant}_{at}"] = table
        out[f"groups_{variant}_rec"] = recognition_groups(variant)
        out[f"groups_{variant}"] = in_force_groups(variant, tables, scenarios)
    return {**out, **tables, **scenarios, **in_force_derived(tables, scenarios)}
