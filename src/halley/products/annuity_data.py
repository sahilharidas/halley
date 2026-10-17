# pyright: basic
"""The reference inputs of the annuity spec as Arrow tables.

The synthetic basis of ``docs/spec/annuity.md`` ("Worked example"): the Makeham base table with
the four-year female setback and q = 1 at 120, the improvement rates and the cumulative
improvement factors, the expense, investment-expense, illiquidity-premium and risk-adjustment
tables, the risk-free discount file and the CPI file, and the four example model points as three
class files. Every formula is written in the evaluation order the spec prints, so each value
equals ``docs/spec/in_force_group_numbers.py`` bit for bit (tests/test_annuity.py).
``docs/spec/annuity_inputs.py`` writes the tables to Parquet beside the spec. pyarrow has no type
stubs, so this module is checked in pyright's basic mode; the model itself stays strict.
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

from halley.products.annuity import BASIS_VERSION, OMEGA, VALUATION_DATE

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


def model_points_table(rows: list[dict[str, Any]], *, fixed: bool = False) -> Any:
    """A model point file with the spec's Arrow types: int64 id, dictionary-encoded portfolio,
    int16 cohort, enums as strings, date32 dates, float64 amounts; ``esc_rate`` for the fixed
    class. A ``cls`` entry on a row is ignored."""
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
    return pa.table(cols).replace_schema_metadata(_meta(valuation_date=VALUATION_DATE.isoformat()))


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
