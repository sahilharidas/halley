# pyright: basic
"""Data contracts at load: model point schema, derived fields, rules, tables and their metadata."""

from __future__ import annotations

import datetime as dt
from typing import Any

import pyarrow as pa
import pytest

from halley import LoadError, RunSettings, build, run
from halley.products.term_assurance import SCHEMA, VALUATION_DATE, TermAssurance
from halley.products.term_assurance_data import reference_model_points, reference_tables
from halley.runtime import load_model_points, load_table

# Section 13 of the spec: derived per-policy constants the loader must reproduce.
SPEC_DERIVED = {
    "P01": (38, 30, 240, 210, 240, 210, 19),
    "P02": (35, 0, 300, 300, 300, 300, 24),
    "P03": (40, 239, 240, 1, 240, 1, 19),
    "P04": (38, 141, 360, 219, 240, 99, 29),
    "P05": (42, 106, 120, 14, 108, 2, 9),
    "P06": (23, 1, 120, 119, 120, 119, 9),
    "P07": (39, 82, 240, 158, 240, 158, 19),
    "P08": (52, 192, 240, 48, 120, 0, 19),
    "P09": (30, 0, 480, 480, 480, 480, 39),
    "P10": (31, 59, 120, 61, 60, 1, 9),
}
DERIVED_NAMES = (
    "age_at_entry",
    "dur_m",
    "term_m",
    "rem_m",
    "prem_term_m",
    "prem_rem_m",
    "dur_y_last",
)


@pytest.fixture(scope="module")
def graph() -> Any:
    return build(TermAssurance).graph


def _rows(**changes: Any) -> Any:
    """The reference model points with the first row changed."""
    table = reference_model_points()
    rows = table.to_pylist()
    rows[0].update(changes)
    return pa.Table.from_pylist(rows, schema=table.schema)


def _load(table: Any, graph: Any, valuation: dt.date = VALUATION_DATE) -> Any:
    return load_model_points(graph, SCHEMA, table, valuation)


def test_derived_fields_match_the_spec_table(graph: Any) -> None:
    mp = _load(reference_model_points(), graph)
    for i, pid in enumerate(mp.ids):
        got = tuple(int(mp.columns[name][i]) for name in DERIVED_NAMES)
        assert got == SPEC_DERIVED[pid], pid


def test_rows_are_sorted_by_policy_id(graph: Any) -> None:
    table = reference_model_points()
    shuffled = table.take([3, 0, 9, 1, 2, 8, 4, 7, 5, 6])
    mp = _load(shuffled, graph)
    assert list(mp.ids) == sorted(mp.ids)
    assert mp.columns["sum_assured"][list(mp.ids).index("P09")] == 1_000_000.0


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"sex": "X"}, "takes exactly M, F"),
        ({"sum_assured": 0.0}, "sum_assured must be > 0"),
        ({"premium": 200_000.0}, "premium must be <= 100000"),
        ({"policy_id": "P02"}, "not unique"),
        ({"birth_date": dt.date(2010, 1, 1)}, "age_at_entry must be >= 18"),
        ({"maturity_date": dt.date(2044, 7, 2)}, r"maturity_date\.day = entry_date\.day"),
        ({"maturity_date": dt.date(2026, 7, 1)}, "rem_m must be >= 1"),
        (
            {"entry_date": dt.date(2027, 2, 1), "maturity_date": dt.date(2047, 2, 1)},
            "dur_m must be >= 0",
        ),
        ({"prem_term_y": 30}, "prem_term_m <= term_m"),
        ({"prem_term_y": 0}, "prem_term_y must be >= 1"),
    ],
)
def test_model_point_contract_violations(graph: Any, changes: dict[str, Any], message: str) -> None:
    with pytest.raises(LoadError, match=message):
        _load(_rows(**changes), graph)


def test_missing_field_and_wrong_type(graph: Any) -> None:
    table = reference_model_points()
    with pytest.raises(LoadError, match="field sex is missing"):
        _load(table.drop_columns(["sex"]), graph)
    as_text = table.set_column(
        table.schema.get_field_index("birth_date"),
        "birth_date",
        pa.array([str(d) for d in table.column("birth_date").to_pylist()]),
    )
    with pytest.raises(LoadError, match="must be date32"):
        _load(as_text, graph)


def test_valuation_date_rule(graph: Any) -> None:
    with pytest.raises(LoadError, match="first day of a calendar month"):
        _load(reference_model_points(), graph, dt.date(2027, 1, 15))


def test_table_contracts(graph: Any) -> None:
    tables = reference_tables()
    mort = tables["mort_table"]
    loaded = load_table(graph.tables["mort_table"], mort)
    assert loaded.version == "term-mort-0.1"
    assert loaded.values["q_ann"].shape == (2, 103)
    assert loaded.values["q_ann"][0, 50 - 18] == 0.00229121307525193  # the spec's (M, 50) chain
    males_only = mort.filter(pa.array([s == "M" for s in mort.column("sex").to_pylist()]))
    with pytest.raises(LoadError, match="cover every key combination"):
        load_table(graph.tables["mort_table"], males_only)
    duplicated = pa.concat_tables([mort, mort.slice(0, 1)])
    with pytest.raises(LoadError, match="cover every key combination"):
        load_table(graph.tables["mort_table"], duplicated)
    rows = mort.to_pylist()
    rows[0]["q_ann"] = 1.5
    with pytest.raises(LoadError, match=r"above 1\.0"):
        load_table(graph.tables["mort_table"], pa.Table.from_pylist(rows, schema=mort.schema))
    rows = mort.to_pylist()
    rows[0]["age"] = 17
    with pytest.raises(LoadError, match="outside its declared range"):
        load_table(graph.tables["mort_table"], pa.Table.from_pylist(rows, schema=mort.schema))
    rates = tables["rates"]
    with pytest.raises(LoadError, match="key column name is missing"):
        load_table(graph.tables["rates"], rates.drop_columns(["name"]))


def test_tables_given_must_match_the_model(tmp_path: Any) -> None:
    compiled = build(TermAssurance)
    tables = reference_tables()
    missing = {k: v for k, v in tables.items() if k != "rates"}
    with pytest.raises(LoadError, match="assumption table 'rates', which was not given"):
        run(
            compiled,
            model_points=reference_model_points(),
            tables=missing,
            settings=RunSettings(valuation_date=VALUATION_DATE),
            run_dir=tmp_path / "missing",
        )
    extra = dict(tables, bonus=tables["rates"])
    with pytest.raises(LoadError, match="declares no such table"):
        run(
            compiled,
            model_points=reference_model_points(),
            tables=extra,
            settings=RunSettings(valuation_date=VALUATION_DATE),
            run_dir=tmp_path / "extra",
        )
