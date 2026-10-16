# pyright: basic
"""Property-based reconciliation (D24): random model points inside the schema of the term
assurance spec, the engine against the spec's number script, under the spec's tolerance."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pyarrow as pa
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from halley import CompiledModel, RunSettings, run
from halley.products.term_assurance import VALUATION_DATE
from halley.products.term_assurance_data import reference_model_points, reference_tables
from halley.runtime import within
from test_term_assurance import SCALARS, STATE_COLS, _scale


def _add_months(d: dt.date, months: int) -> dt.date:
    k = d.year * 12 + d.month - 1 + months
    return dt.date(k // 12, k % 12 + 1, d.day)


@st.composite
def model_points(draw: Any) -> dict[str, Any]:
    months_back = draw(st.integers(min_value=0, max_value=479))
    day = 1 if months_back == 0 else draw(st.integers(min_value=1, max_value=28))
    entry = _add_months(VALUATION_DATE, -months_back).replace(day=day)
    term_m = draw(st.integers(min_value=12, max_value=480))
    assume(term_m - months_back >= 1)  # at least one month of cover left
    maturity = _add_months(entry, term_m)
    age = draw(st.integers(min_value=18, max_value=65))
    offset_days = draw(st.integers(min_value=0, max_value=364))
    birth = entry.replace(year=entry.year - age) - dt.timedelta(days=offset_days)
    return {
        "policy_id": "R01",
        "sex": draw(st.sampled_from(["M", "F"])),
        "birth_date": birth,
        "entry_date": entry,
        "maturity_date": maturity,
        "prem_term_y": draw(st.integers(min_value=1, max_value=term_m // 12)),
        "sum_assured": draw(st.floats(min_value=1_000, max_value=10_000_000)),
        "premium": draw(st.floats(min_value=0.5, max_value=5_000)),
        "pols_if_init": draw(st.sampled_from([1.0, 2.5, 100.0, 0.25])),
    }


@pytest.fixture(scope="module")
def tables() -> dict[str, Any]:
    return reference_tables()


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(mp=model_points())
def test_random_policies_reconcile_with_the_spec_script(
    mp: dict[str, Any],
    oracle: ModuleType,
    compiled_term: CompiledModel,
    tables: dict[str, Any],
    tmp_path_factory: Any,
) -> None:
    try:
        ref = oracle.model_point(
            mp["policy_id"],
            mp["sex"],
            mp["birth_date"],
            mp["entry_date"],
            mp["maturity_date"],
            mp["prem_term_y"],
            mp["sum_assured"],
            mp["pols_if_init"],
        )
    except AssertionError:
        assume(False)
        return
    ref["premium"] = mp["premium"]
    table = pa.Table.from_pylist([mp], schema=reference_model_points().schema)
    res = run(
        compiled_term,
        model_points=table,
        tables=tables,
        settings=RunSettings(valuation_date=VALUATION_DATE),
        run_dir=Path(tmp_path_factory.mktemp("prop")) / "run",
    )
    expected = oracle.project_mp(ref)
    n = ref["rem_m"]
    assert n == res.H
    for col in oracle.MONTH_COLS:
        first = 0 if col in STATE_COLS else 1
        engine = res.lane(col, "R01")[first : n + 1]
        want = np.array(expected[col][first : n + 1], dtype=np.float64)
        assert within(engine, want, _scale(col, ref)).all(), col
    for name in SCALARS:
        assert within(res.lane(name, "R01")[0], expected["scalars"][name], _scale(name, ref)), name
    assert all(r.passed for r in res.invariants)
