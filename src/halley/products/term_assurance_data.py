# pyright: basic
"""The reference inputs of the term assurance spec as Arrow tables.

The synthetic basis of section 6 (five assumption tables, 6.8) and the ten reference model points
of section 13, built from the formulas and numbers printed there. ``docs/spec/
term_assurance_inputs.py`` writes them to Parquet beside the spec (C37). pyarrow has no type
stubs, so this module is checked in pyright's basic mode; the model itself stays strict.
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Any

import pyarrow as pa

from halley.products.term_assurance import BASIS_VERSION, VALUATION_DATE

GOMPERTZ_MAKEHAM: dict[str, tuple[float, float, float]] = {
    "M": (0.0002, 0.000017, 1.1),
    "F": (0.00015, 0.000011, 1.1),
}
AGE_LO, AGE_HI = 18, 120
LAPSE_RATES = (0.15, 0.10, 0.08, 0.06, 0.05, 0.04)
COMMISSION_RATES = (0.20, 0.02)
INTEREST = 0.04
EXPENSE_INFLATION = 0.03
MAINT_BASE = 5.00
CLAIM_EXP_BASE = 250.00


def q_ann_formula(sex: str, x: int) -> float:
    """Section 6.1, in exactly the order written: c^x, B*c^x, *(c-1), /ln(c), A+, -, exp, 1-."""
    a, b, c = GOMPERTZ_MAKEHAM[sex]
    return 1 - math.exp(-(a + b * c**x * (c - 1) / math.log(c)))


def _meta(version: str, **extra: str) -> dict[bytes, bytes]:
    meta = {b"halley.version": version.encode(), b"halley.basis": BASIS_VERSION.encode()}
    meta.update({f"halley.{k}".encode(): v.encode() for k, v in extra.items()})
    return meta


def reference_tables() -> dict[str, Any]:
    """The five assumption tables of section 6.8 as Arrow tables, from the formulas."""
    sexes = [s for s in ("M", "F") for _ in range(AGE_LO, AGE_HI + 1)]
    ages = [x for _ in ("M", "F") for x in range(AGE_LO, AGE_HI + 1)]
    mort = pa.table(
        {
            "sex": pa.array(sexes, pa.string()),
            "age": pa.array(ages, pa.int16()),
            "q_ann": pa.array([q_ann_formula(s, x) for s, x in zip(sexes, ages, strict=True)]),
        }
    ).replace_schema_metadata(_meta("term-mort-0.1", source="Gompertz-Makeham, spec 6.1"))
    lapse = pa.table(
        {
            "dur_y": pa.array(list(range(6)), pa.int16()),
            "w_ann": pa.array(list(LAPSE_RATES), pa.float64()),
        }
    ).replace_schema_metadata(_meta("term-lapse-0.1"))
    comm = pa.table(
        {
            "dur_y": pa.array([0, 1], pa.int16()),
            "comm_rate": pa.array(list(COMMISSION_RATES), pa.float64()),
        }
    ).replace_schema_metadata(_meta("term-comm-0.1"))
    rates = pa.table(
        {
            "name": pa.array(["i", "e"], pa.string()),
            "rate": pa.array([INTEREST, EXPENSE_INFLATION], pa.float64()),
        }
    ).replace_schema_metadata(_meta("term-rates-0.1"))
    expenses = pa.table(
        {
            "name": pa.array(["maint_base", "claim_exp_base"], pa.string()),
            "amount": pa.array([MAINT_BASE, CLAIM_EXP_BASE], pa.float64()),
        }
    ).replace_schema_metadata(_meta("term-expenses-0.1"))
    return {
        "mort_table": mort,
        "lapse_table": lapse,
        "comm_table": comm,
        "rates": rates,
        "expenses": expenses,
    }


REFERENCE_MODEL_POINTS: tuple[tuple[str, str, str, str, str, int, float, float, float], ...] = (
    ("P01", "M", "1986-03-15", "2024-07-01", "2044-07-01", 20, 250_000, 52.96, 1),
    ("P02", "F", "1991-11-02", "2027-01-01", "2052-01-01", 25, 400_000, 51.64, 1),
    ("P03", "M", "1966-05-20", "2007-01-15", "2027-01-15", 20, 150_000, 39.99, 1),
    ("P04", "F", "1976-08-09", "2015-03-20", "2045-03-20", 20, 300_000, 68.98, 1),
    ("P05", "M", "1975-04-18", "2018-02-10", "2028-02-10", 9, 200_000, 46.40, 1),
    ("P06", "F", "2003-02-28", "2026-11-10", "2036-11-10", 10, 100_000, 10.76, 1),
    ("P07", "M", "1980-06-01", "2020-02-29", "2040-02-29", 20, 500_000, 105.55, 2.5),
    ("P08", "M", "1958-12-31", "2010-12-31", "2030-12-31", 10, 120_000, 128.12, 1),
    ("P09", "F", "1996-10-31", "2026-12-15", "2066-12-15", 40, 1_000_000, 116.62, 1),
    ("P10", "M", "1990-05-05", "2022-02-01", "2032-02-01", 5, 200_000, 39.27, 1),
)
"""Section 13: the ten reference model points, exactly as printed."""


def reference_model_points() -> Any:
    """The model point file of section 13 with the Arrow types of section 3.1."""
    cols = list(zip(*REFERENCE_MODEL_POINTS, strict=True))

    def dates(values: tuple[Any, ...]) -> Any:
        return pa.array([dt.date.fromisoformat(str(v)) for v in values], pa.date32())

    return pa.table(
        {
            "policy_id": pa.array(cols[0], pa.string()),
            "sex": pa.array(cols[1], pa.string()),
            "birth_date": dates(cols[2]),
            "entry_date": dates(cols[3]),
            "maturity_date": dates(cols[4]),
            "prem_term_y": pa.array(cols[5], pa.int16()),
            "sum_assured": pa.array([float(v) for v in cols[6]], pa.float64()),
            "premium": pa.array([float(v) for v in cols[7]], pa.float64()),
            "pols_if_init": pa.array([float(v) for v in cols[8]], pa.float64()),
        }
    ).replace_schema_metadata(_meta("term-mp-0.1", valuation_date=VALUATION_DATE.isoformat()))
