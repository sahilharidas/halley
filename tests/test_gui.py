# pyright: basic
"""The GUI (``halley.gui``): its API on the reference products and on a model file of the tests'
own, its server's checks on who may call it, and its pages.

The API is called through the routes the server dispatches to, without HTTP: a request is a
``Request`` and an answer is what the route returns, held to strict JSON as the server would
send it. A build or a run is a job; the tests wait for it. The server is tested over HTTP on a
free port of the loopback interface. A run the GUI makes must be the run the same call makes in
Python, with the same run id, and a run read back from its directory must be reproducible from
it alone (D21).
"""

from __future__ import annotations

import ast
import builtins
import datetime as dt
import http.client
import importlib.resources
import json
import math
import os
import re
import shutil
import socket
import sys
import threading
import time
import types
import typing
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import halley
import halley.gui
from halley import CompiledModel, RunResult, RunSettings, run
from halley.aggregation import portfolio_totals
from halley.errors import HalleyError
from halley.gui import __main__ as cli
from halley.gui import describe, examples
from halley.gui.app import App, _guarded, _replaced, _rerun_warnings
from halley.gui.catalog import (
    Catalog,
    ModelFileError,
    exception_text,
    format_quietly,
    is_library_file,
    recorded_statics,
    rerun_misses,
    statics_of,
)
from halley.gui.jobs import CANCELLED, DONE, Jobs
from halley.gui.server import STATIC_FILES, TOKEN_HEADER, ApiError, Request, jsonable, make_server
from halley.products.annuity import CPILinkedAnnuity
from halley.products.term_assurance import VALUATION_DATE
from halley.products.term_assurance_data import reference_model_points, reference_tables

TERM = "halley.products.term_assurance:TermAssurance"
REFERENCE_INPUTS = Path(__file__).resolve().parent.parent / "docs" / "spec" / "term-assurance"


def call(
    app: App, method: str, route: str, query: dict[str, Any] | None = None, **body: Any
) -> Any:
    """A route's answer, held to the strict JSON the server sends."""
    params = {k: str(v) for k, v in (query or {}).items()}
    answer = app.routes()[(method, route)](Request(method, route, params, body))
    json.dumps(jsonable(answer), allow_nan=False)
    return answer


def refused(
    app: App, method: str, route: str, query: dict[str, Any] | None = None, **body: Any
) -> ApiError:
    with pytest.raises(ApiError) as info:
        call(app, method, route, query, **body)
    return info.value


def finish(app: App, answer: Any) -> dict[str, Any]:
    """Wait for the job an answer started and return it as the page reads it."""
    job_id = answer["job"]["id"]
    app.jobs.wait(job_id, timeout=600)
    job = call(app, "GET", "/api/job", {"id": job_id})
    assert job["state"] in ("done", "failed"), job
    return job


def done(app: App, answer: Any) -> Any:
    job = finish(app, answer)
    assert job["state"] == "done", job["error"]
    return job["result"]


@pytest.fixture
def app(tmp_path: Path) -> Iterator[App]:
    made = App(tmp_path / "runs")
    yield made
    made.close()


def build_term(app: App) -> str:
    return done(app, call(app, "POST", "/api/build", model=TERM, backend="numpy"))["build"]


# ---------------------------------------------------------------------------------- models


def test_the_catalog_lists_the_shipped_models_and_what_they_declare(app: App) -> None:
    models = {m["id"]: m for m in call(app, "GET", "/api/models")["models"]}
    assert models[TERM]["kind"] == "projection"
    assert models[TERM]["examples"] == 1
    # the flag is the class's own: subclasses of the abstract escalating annuity are buildable
    assert models["halley.products.annuity:EscalatingAnnuity"]["abstract"] is True
    assert models["halley.products.annuity:FixedEscalationAnnuity"]["abstract"] is False
    assert models["halley.aggregation.csm:NewGroup"]["kind"] == "aggregation"
    assert models["halley.aggregation.csm:NewGroup"]["abstract"] is False
    term = call(app, "GET", "/api/model", {"id": TERM})
    assert term["valuation_date"] == VALUATION_DATE.isoformat()
    assert [f["name"] for f in term["schema"]["fields"]][:2] == ["policy_id", "sex"]
    assert {t["name"] for t in term["tables"]} == set(reference_tables())
    assert term["statics"] == []
    assert [e["id"] for e in term["examples"]] == ["term-reference"]


def test_build_time_parameters_are_found_as_the_tracer_finds_them() -> None:
    params = {p.name: p for p in statics_of(CPILinkedAnnuity)}
    assert params["cpi_lag_m"].kind == "int"
    assert params["cpi_lag_m"].default == 3
    assert params["include_inv_exp"].kind == "bool"
    assert params["include_inv_exp"].default is True


def test_a_build_describes_every_variable(app: App) -> None:
    build = call(app, "GET", "/api/build", {"id": build_term(app)})
    variables = {v["name"]: v for v in build["variables"]}
    assert len(variables) == 49  # the tutorial's count (docs/tutorial/reading-a-run.md)
    reserve = variables["reserve"]
    assert reserve["direction"] == "backward"
    assert reserve["role"] == "reserve"
    assert [r["label"] for r in reserve["reads"]] == ["reserve(t+1)", "net_outgo(t+1)", "v(t+1)"]
    assert "reserve" in variables["reserve"]["read_by"]
    assert reserve["loc"]["function"] == "TermAssurance.reserve"
    assert reserve["terminal"] == "0"
    assert build["warnings"] == []
    assert build["stale"] is False
    source = call(app, "GET", "/api/build/source", {"id": build["id"]})
    assert source["artefact"] == "numpy_backend.py"
    assert "def run_batch" in source["source"]


# ------------------------------------------------------------------------------------ runs


def test_a_run_of_the_reference_example_is_the_python_run(
    app: App, compiled_term: CompiledModel, tmp_path: Path
) -> None:
    build = build_term(app)
    settings = {"valuation_date": VALUATION_DATE.isoformat()}
    check = done(
        app,
        call(
            app,
            "POST",
            "/api/inputs/check",
            build=build,
            inputs={"example": "term-reference"},
            settings=settings,
        ),
    )
    assert check == {"policies": 10, "scenarios": 1, "horizon": 480, "batches": 1, "processes": 1}
    made = done(
        app,
        call(
            app,
            "POST",
            "/api/run",
            build=build,
            inputs={"example": "term-reference"},
            settings=settings,
            name="reference",
        ),
    )
    python = run(
        compiled_term,
        model_points=reference_model_points(),
        tables=reference_tables(),
        settings=RunSettings(valuation_date=VALUATION_DATE),
        run_dir=tmp_path / "python",
    )
    assert made["run_id"] == python.run_id
    assert Path(made["run"]) == (tmp_path / "runs" / "reference").resolve()
    summary = call(app, "GET", "/api/run", {"dir": made["run"]})
    assert summary["explainable"] is True
    assert summary["in_session"] is True
    assert [i["passed"] for i in summary["invariants"]] == [True] * 7
    listed = call(app, "GET", "/api/runs")["runs"]
    assert [(r["name"], r["run_id"], r["in_session"]) for r in listed] == [
        ("reference", python.run_id, True)
    ]


def test_a_run_from_files_has_the_run_id_of_the_same_files_run_in_python(
    app: App, term_run: RunResult
) -> None:
    build = build_term(app)
    inputs = {
        "model_points": {"path": str(REFERENCE_INPUTS / "model_points.parquet")},
        # a path as Windows' Copy as path gives it, in quotes
        "tables": {n: f'"{REFERENCE_INPUTS / f"{n}.parquet"}"' for n in reference_tables()},
    }
    made = done(
        app,
        call(
            app,
            "POST",
            "/api/run",
            build=build,
            inputs=inputs,
            settings={"valuation_date": VALUATION_DATE.isoformat(), "label": "from files"},
            name="files",
        ),
    )
    # the label is recorded, not in the run id
    assert made["run_id"] == term_run.run_id


def test_lanes_totals_and_policies_read_the_run(app: App) -> None:
    build = build_term(app)
    made = done(
        app,
        call(
            app,
            "POST",
            "/api/run",
            build=build,
            inputs={"example": "term-reference"},
            settings={"valuation_date": "2027-01-01"},
            name="r",
        ),
    )
    summary = call(app, "GET", "/api/run", {"dir": made["run"]})
    ids = list(summary["ids"])
    p05 = ids.index("P05")
    lanes = call(
        app, "GET", "/api/run/lanes", {"dir": made["run"], "var": "reserve", "p": f"{p05},0"}
    )
    assert lanes["time"] is True
    assert len(lanes["t"]) == 481
    assert [s["id"] for s in lanes["series"]] == ["P05", "P01"]
    result = app._result(made["run"])[0]  # the session's own result
    assert np.array_equal(lanes["series"][0]["values"], result.lane("reserve", "P05"))
    totals = call(app, "GET", "/api/run/totals", {"dir": made["run"], "var": "reserve"})
    want = portfolio_totals(result, ["reserve"])["reserve"][:, 0, 0]
    assert np.array_equal(totals["series"][0]["values"], want)
    scalar = call(app, "GET", "/api/run/lanes", {"dir": made["run"], "var": "reserve_0", "p": "0"})
    assert scalar["time"] is False
    assert scalar["t"] is None
    page = call(
        app,
        "GET",
        "/api/run/policies",
        {
            "dir": made["run"],
            "var": "reserve",
            "t": 0,
            "sort": "value",
            "order": "desc",
            "limit": 3,
        },
    )
    values = result.var("reserve")[0, :, 0]
    assert page["total"] == 10
    assert [r["value"] for r in page["rows"]] == sorted(values.tolist(), reverse=True)[:3]
    assert "policy_id" not in page["fields"]
    assert "rem_m" in page["fields"]  # a derived field, which a run made in the session holds
    with pytest.raises(ApiError, match="outside"):
        call(app, "GET", "/api/run/lanes", {"dir": made["run"], "var": "reserve", "p": "10"})
    with pytest.raises(ApiError, match="no 'nope'"):
        call(app, "GET", "/api/run/lanes", {"dir": made["run"], "var": "nope"})


def test_explain_gives_each_input_a_target_and_recomputes_bit_for_bit(app: App) -> None:
    build = build_term(app)
    made = done(
        app,
        call(
            app,
            "POST",
            "/api/run",
            build=build,
            inputs={"example": "term-reference"},
            settings={"valuation_date": "2027-01-01"},
            name="e",
        ),
    )
    p05 = list(call(app, "GET", "/api/run", {"dir": made["run"]})["ids"]).index("P05")
    e = call(
        app, "GET", "/api/run/explain", {"dir": made["run"], "p": p05, "var": "reserve", "t": 13}
    )
    result = app._result(made["run"])[0]
    assert e["value"] == result.lane("reserve", "P05")[13]
    assert e["text"] == str(halley.explain(result, "P05", "reserve", 13))
    assert [(i["text"], i["target"]) for i in e["inputs"]] == [
        ("reserve(t+1)", {"kind": "variable", "var": "reserve", "t": 14}),
        ("net_outgo(t+1)", {"kind": "variable", "var": "net_outgo", "t": 14}),
        ("v(t+1)", {"kind": "variable", "var": "v", "t": 14}),
    ]
    assert e["recomputed"]["same"] is True
    assert e["boundary"] is None
    lookup = call(
        app, "GET", "/api/run/explain", {"dir": made["run"], "p": p05, "var": "q_ann", "t": 1}
    )
    targets = {i["text"]: i["target"] for i in lookup["inputs"]}
    assert targets["tables.mort_table.q_ann[mp.sex, age(t)]"] == {
        "kind": "table",
        "name": "mort_table",
        "column": "q_ann",
    }
    assert targets["mp.sex"] == {"kind": "field", "name": "sex"}
    # a scalar variable read in a formula is rendered x(t) and opens without a month
    pv = call(app, "GET", "/api/run/explain", {"dir": made["run"], "p": p05, "var": "pv_net_cf"})
    assert pv["t"] is None
    assert {"kind": "variable", "var": "pv_premiums", "t": None} in [
        i["target"] for i in pv["inputs"]
    ]


def test_explain_says_when_a_month_takes_a_boundary_value(app: App) -> None:
    build = build_term(app)
    made = done(
        app,
        call(
            app,
            "POST",
            "/api/run",
            build=build,
            inputs={"example": "term-reference"},
            settings={"valuation_date": "2027-01-01"},
            name="b",
        ),
    )
    e = call(app, "GET", "/api/run/explain", {"dir": made["run"], "p": 0, "var": "pols_if", "t": 0})
    assert e["boundary"]["kind"] == "init"
    assert e["boundary"]["expr"] == "mp.pols_if_init"
    assert all(i["unused"] for i in e["inputs"])
    assert e["recomputed"]["same"] is True
    end = call(
        app, "GET", "/api/run/explain", {"dir": made["run"], "p": 0, "var": "reserve", "t": 480}
    )
    assert end["boundary"]["kind"] == "terminal"
    with pytest.raises(ApiError, match=r"monthly: give t in 0\.\.480"):
        call(
            app, "GET", "/api/run/explain", {"dir": made["run"], "p": 0, "var": "reserve", "t": 481}
        )


def test_a_run_read_back_is_explained_after_a_rerun_from_its_own_inputs(
    app: App, term_run: RunResult
) -> None:
    call(app, "POST", "/api/runs/open", path=str(term_run.run_dir))
    summary = call(app, "GET", "/api/run", {"dir": str(term_run.run_dir)})
    assert summary["explainable"] is False
    assert summary["run_id"] == term_run.run_id
    error = refused(
        app, "GET", "/api/run/explain", {"dir": str(term_run.run_dir), "var": "reserve", "t": 0}
    )
    assert error.status == 409
    assert error.detail == {"rerun": True}
    again = done(app, call(app, "POST", "/api/run/rerun", dir=str(term_run.run_dir)))
    assert again["same_run_id"] is True
    assert again["outputs"]["bit_for_bit"] is True
    assert again["outputs"]["differ"] == []
    assert Path(again["run"]).parent == (app.runs_root / "reruns")
    assert call(app, "GET", "/api/run", {"dir": again["run"]})["explainable"] is True
    e = call(
        app, "GET", "/api/run/explain", {"dir": again["run"], "p": 4, "var": "reserve", "t": 13}
    )
    assert e["policy"] == "P05"
    # the label and the workers are not in the run id; the settings are the recorded ones
    rerun = app._result(again["run"])[0]
    assert rerun.manifest["settings"] == term_run.manifest["settings"]


def test_a_comparison_names_what_differs(app: App, compiled_term: CompiledModel) -> None:
    root = app.runs_root
    base = run(
        compiled_term,
        model_points=reference_model_points(),
        tables=reference_tables(),
        settings=RunSettings(valuation_date=VALUATION_DATE),
        run_dir=root / "four",
    )
    tables = reference_tables()
    rates = tables["rates"]
    tables["rates"] = rates.set_column(
        rates.schema.get_field_index("rate"), "rate", pa.array([0.05, 0.03])
    )
    other = run(
        compiled_term,
        model_points=reference_model_points(),
        tables=tables,
        settings=RunSettings(valuation_date=VALUATION_DATE),
        run_dir=root / "five",
    )
    c = call(app, "GET", "/api/compare", {"a": str(base.run_dir), "b": str(other.run_dir)})
    assert c["same_run_id"] is False
    assert [row["what"] for row in c["identity"] if not row["same"]] == ["table rates"]
    assert c["comparable"] is True
    reserve = next(v for v in c["variables"] if v["variable"] == "reserve")
    # the tutorial's diff: P09 at month 148, 3987.65 at 4% against 3194.08 at 5%
    assert (reserve["policy"], reserve["t"]) == ("P09", 148)
    assert (round(reserve["a"], 2), round(reserve["b"], 2)) == (3987.65, 3194.08)
    assert reserve["unit"] == "Money"
    assert c["variables"] == sorted(c["variables"], key=lambda r: -r["max_abs"])
    same = call(app, "GET", "/api/compare", {"a": str(base.run_dir), "b": str(base.run_dir)})
    assert same["same_run_id"] is True
    assert all(row["max_abs"] == 0 for row in same["variables"])


def test_the_inputs_and_files_of_a_run_are_read_from_its_directory(
    app: App, term_run: RunResult
) -> None:
    call(app, "POST", "/api/runs/open", path=str(term_run.run_dir))
    rates = call(
        app,
        "GET",
        "/api/run/input",
        {"dir": str(term_run.run_dir), "kind": "table", "name": "rates"},
    )
    assert [c["name"] for c in rates["columns"]] == ["name", "rate"]
    assert rates["rows"] == [["i", 0.04], ["e", 0.03]]
    assert rates["sha256"] == term_run.manifest["inputs"]["tables"]["rates"]["sha256"]
    points = call(
        app, "GET", "/api/run/input", {"dir": str(term_run.run_dir), "kind": "model_points"}
    )
    assert points["total"] == 10
    assert (
        points["rows"][0][points["columns"].index({"name": "birth_date", "type": "date32[day]"})]
        == "1986-03-15"
    )
    manifest = call(app, "GET", "/api/run/file", {"dir": str(term_run.run_dir), "name": "manifest"})
    assert json.loads(manifest["text"])["run_id"] == term_run.run_id
    generated = call(
        app, "GET", "/api/run/file", {"dir": str(term_run.run_dir), "name": "generated"}
    )
    assert generated["path"].endswith("numpy_backend.py")
    # a loaded run's policy table comes from its copy of the model points, in the run's order
    page = call(
        app, "GET", "/api/run/policies", {"dir": str(term_run.run_dir), "sort": "id", "limit": 2}
    )
    assert [r["id"] for r in page["rows"]] == ["P01", "P02"]
    assert page["fields"][:2] == ["sex", "birth_date"]


# ------------------------------------------------------------------- model files and errors


def write_model(
    folder: Path,
    stem: str,
    *,
    broken: bool = False,
    unique: bool = True,
    head: str = "",
    body: str = "",
) -> Path:
    """The tutorial's term model as a file; ``broken`` forgets the policies in force in the
    premium, the tutorial's first mistake, ``unique`` says whether the ids must be unique, and
    ``head`` and ``body`` are further lines at the module's top and in the class."""
    premium = "self.mp.premium" if broken else "self.pols_if(t - 1) * self.mp.premium"
    source = f'''"""A model file of the GUI's tests."""

from halley import Column, Count, Field, Int, Key, Money, Monthly, Prob, Product, Schema, Table
from halley import Static, to_monthly_prob, var, where
from halley.authoring import int_field
{head}
SCHEMA = Schema(
    fields=(
        Field("policy_id", "string", unique={unique}),
        int_field("age", ge=18, le=80),
        Field("sum_assured", "number", "MoneyPerPolicy", gt=0),
        Field("premium", "number", "MoneyPerPolicy", gt=0),
        Field("pols_if_init", "number", "Count", gt=0),
        int_field("rem_m", ge=1, le=600),
    ),
    horizon="rem_m",
)
MORT = Table("mort", keys=(Key("age", lo=18, hi=130),), columns=(Column("q_ann", "Prob[Annual]"),))
V_M = (1 + 0.04) ** (-1 / 12)


class Term(Product):
    """The tutorial's term assurance."""

    model_point = SCHEMA
    assumption_tables = (MORT,)
{body}
    @var
    def age(self, t: Int) -> Int:
        return self.mp.age + (t - 1) // 12

    @var
    def q_m(self, t: Int) -> Prob[Monthly]:
        return where(t <= self.mp.rem_m, to_monthly_prob(self.tables.mort[self.age(t)]), 0)

    @var(role="decrement")
    def deaths(self, t: Int) -> Count:
        return self.pols_if(t - 1) * self.q_m(t)

    @var(init=lambda mp: mp.pols_if_init, role="in_force")
    def pols_if(self, t: Int) -> Count:
        return self.pols_if(t - 1) - self.deaths(t)

    @var(timing="start")
    def premium_income(self, t: Int) -> Money:
        return where(t <= self.mp.rem_m, {premium}, 0)

    @var(timing="end")
    def death_outgo(self, t: Int) -> Money:
        return self.deaths(t) * self.mp.sum_assured

    @var(terminal=0, role="reserve")
    def reserve(self, t: Int) -> Money:
        return (self.reserve(t + 1) + self.death_outgo(t + 1) - self.premium_income(t + 1)) * V_M
'''
    path = folder / f"{stem}.py"
    path.write_text(source, encoding="utf-8")
    return path


def _stem(tmp_path: Path, what: str) -> str:
    """A module name of the test's own: a model file is imported under its file name for the
    whole session, so two tests must not share one."""
    return f"gui_{what}_{re.sub(r'[^0-9a-zA-Z]', '_', tmp_path.name)}"


def test_a_model_file_loads_builds_and_runs_on_csv_model_points(app: App, tmp_path: Path) -> None:
    path = write_model(tmp_path, _stem(tmp_path, "term"))
    loaded = call(app, "POST", "/api/models/load", path=str(path))
    model_id = f"{path.stem}:Term"
    assert loaded["file"]["models"] == [model_id]
    assert loaded["models"][0]["doc"] == "The tutorial's term assurance."
    build = done(app, call(app, "POST", "/api/build", model=model_id))["build"]
    (tmp_path / "points.csv").write_text(
        "policy_id,age,sum_assured,premium,pols_if_init,rem_m\n"
        "A,35,100000,20,1,120\nB,50,250000,90,1,60\nC,62,50000,60,1,36\n",
        encoding="utf-8",
    )
    mort = pa.table(
        {
            "age": list(range(18, 131)),
            "q_ann": [min(1.0, 0.0005 * 1.09 ** (x - 18)) for x in range(18, 131)],
        }
    )
    pq.write_table(mort, tmp_path / "mort.parquet")
    inputs = {
        "model_points": str(tmp_path / "points.csv"),
        "tables": {"mort": str(tmp_path / "mort.parquet")},
    }
    made = done(
        app,
        call(
            app,
            "POST",
            "/api/run",
            build=build,
            inputs=inputs,
            settings={"valuation_date": "2027-01-01"},
            name="mine",
        ),
    )
    result = app._result(made["run"])[0]
    # the tutorial's reserves of policies A, B and C (docs/tutorial/your-own-data.md)
    assert [round(float(result.lane("reserve", p)[0]), 2) for p in "ABC"] == [
        648.35,
        5634.01,
        1367.45,
    ]
    e = call(app, "GET", "/api/run/explain", {"dir": made["run"], "p": 0, "var": "reserve", "t": 0})
    assert Path(e["loc"]["file"]) == path.resolve()
    bad = tmp_path / "bad.csv"
    bad.write_text(
        "policy_id,age,sum_assured,premium,pols_if_init,rem_m\nA,35,100000,20,1,120\nB,50,-250000,90,1,60\n",
        encoding="utf-8",
    )
    failed = finish(
        app,
        call(
            app,
            "POST",
            "/api/inputs/check",
            build=build,
            inputs={**inputs, "model_points": str(bad)},
            settings={"valuation_date": "2027-01-01"},
        ),
    )
    assert failed["state"] == "failed"
    assert failed["error"]["kind"] == "load"
    assert (
        failed["error"]["message"]
        == "model points: field sum_assured must be > 0; policies B break this"
    )


def test_a_build_error_points_at_the_model_files_own_line(app: App, tmp_path: Path) -> None:
    path = write_model(tmp_path, _stem(tmp_path, "forgot"), broken=True)
    call(app, "POST", "/api/models/load", path=str(path))
    failed = finish(app, call(app, "POST", "/api/build", model=f"{path.stem}:Term"))
    error = failed["error"]
    assert failed["state"] == "failed"
    assert error["kind"] == "unit"
    assert error["message"] == "premium_income is declared Money but its formula is MoneyPerPolicy"
    assert Path(error["loc"]["file"]) == path.resolve()
    here = [line for line in error["snippet"] if line["here"]]
    assert [line["text"].strip() for line in here] == ['@var(timing="start")']
    assert "traceback" not in error  # one of Halley's own errors needs none


def test_a_file_that_does_not_import_is_shown_at_its_line_and_keeps_the_last_good_models(
    app: App, tmp_path: Path
) -> None:
    path = write_model(tmp_path, _stem(tmp_path, "edited"))
    call(app, "POST", "/api/models/load", path=str(path))
    model_id = f"{path.stem}:Term"
    first = app.catalog.get(model_id)
    assert first is not None
    path.write_text(
        path.read_text(encoding="utf-8").replace("class Term(Product):", "class Term(Product)"),
        encoding="utf-8",
    )
    error = refused(app, "POST", "/api/models/load", path=str(path))
    assert error.status == 422
    detail: Any = error.detail
    assert detail["type"] == "SyntaxError"
    assert detail["loc"]["line"] == next(line["line"] for line in detail["snippet"] if line["here"])
    assert app.catalog.get(model_id) is first  # the file's models stay as they were


def test_a_model_file_named_like_an_imported_module_is_refused(app: App, tmp_path: Path) -> None:
    clash = tmp_path / "json.py"
    clash.write_text('"""Shadows the standard library."""\n', encoding="utf-8")
    error = refused(app, "POST", "/api/models/load", path=str(clash))
    assert error.status == 422
    assert "already imported" in error.message
    assert sys.modules["json"].__name__ == "json"
    odd = tmp_path / "my-model.py"
    odd.write_text("", encoding="utf-8")
    assert "identifier" in refused(app, "POST", "/api/models/load", path=str(odd)).message


def test_a_request_is_refused_before_anything_runs(app: App, tmp_path: Path) -> None:
    build = build_term(app)
    settings = {"valuation_date": "2027-01-01"}
    example = {"example": "term-reference"}
    unknown = refused(
        app,
        "POST",
        "/api/run",
        build=build,
        inputs=example,
        settings={**settings, "outputs": ["nope"]},
        name="x",
    )
    assert unknown.status == 400
    assert "nope is not among them" in unknown.message
    assert (
        refused(
            app,
            "POST",
            "/api/run",
            build=build,
            inputs=example,
            settings={"valuation_date": "1 Jan"},
            name="x",
        ).status
        == 400
    )
    assert (
        refused(
            app, "POST", "/api/run", build=build, inputs=example, settings=settings, name="../x"
        ).status
        == 400
    )
    assert (
        refused(
            app,
            "POST",
            "/api/run",
            build=build,
            inputs=example,
            settings={**settings, "batch_size": 0},
            name="x",
        ).status
        == 400
    )
    assert (
        refused(
            app,
            "POST",
            "/api/run",
            build=build,
            inputs=example,
            settings={**settings, "workers": True},
            name="x",
        ).status
        == 400
    )
    missing = refused(
        app,
        "POST",
        "/api/run",
        build=build,
        inputs={"model_points": str(tmp_path / "none.parquet")},
        settings=settings,
        name="x",
    )
    assert "there is no file" in missing.message
    extra = refused(
        app,
        "POST",
        "/api/run",
        build=build,
        inputs={
            "model_points": str(REFERENCE_INPUTS / "model_points.parquet"),
            "tables": {"mort": "m.parquet"},
        },
        settings=settings,
        name="x",
    )
    assert "declares no assumption table mort" in extra.message
    done(
        app, call(app, "POST", "/api/run", build=build, inputs=example, settings=settings, name="x")
    )
    assert (
        refused(
            app, "POST", "/api/run", build=build, inputs=example, settings=settings, name="x"
        ).status
        == 409
    )
    stage = done(app, call(app, "POST", "/api/build", model="halley.aggregation.csm:NewGroup"))[
        "build"
    ]
    assert (
        "aggregation stage"
        in refused(
            app, "POST", "/api/run", build=stage, inputs=example, settings=settings, name="y"
        ).message
    )
    assert refused(app, "GET", "/api/build", {"id": "build-999"}).status == 404


def test_an_empty_outputs_choice_stores_what_the_model_stores(app: App) -> None:
    """An empty tuple of outputs stores what None does but is another run id (run.py), so the
    GUI sends None for an empty choice."""
    build = build_term(app)
    made = done(
        app,
        call(
            app,
            "POST",
            "/api/run",
            build=build,
            inputs={"example": "term-reference"},
            settings={"valuation_date": "2027-01-01", "outputs": []},
            name="o",
        ),
    )
    result = app._result(made["run"])[0]
    assert result.manifest["settings"]["outputs"] is None


# ------------------------------------------------------------------------------ the server


@pytest.fixture
def server(app: App) -> Iterator[Any]:
    made = make_server(app.routes(), "127.0.0.1", 0, token="secret-token")
    thread = threading.Thread(target=made.serve_forever, daemon=True)
    thread.start()
    yield made
    made.shutdown()
    made.server_close()


def fetch(
    server: Any,
    method: str,
    path: str,
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
) -> tuple[int, dict[str, str], bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=30)
    try:
        conn.request(
            method, path, body=body, headers={"Host": f"127.0.0.1:{server.port}", **(headers or {})}
        )
        response = conn.getresponse()
        return response.status, {k.lower(): v for k, v in response.getheaders()}, response.read()
    finally:
        conn.close()


def test_the_pages_are_served_with_a_policy_that_loads_nothing_from_elsewhere(server: Any) -> None:
    status, headers, body = fetch(server, "GET", "/")
    assert status == 200
    assert headers["content-type"] == "text/html; charset=utf-8"
    assert "default-src 'none'" in headers["content-security-policy"]
    assert headers["x-frame-options"] == "DENY"
    assert b'src="/app.js"' in body
    for path in STATIC_FILES:
        assert fetch(server, "GET", path)[0] == 200
    for path in ("/static/app.js", "/../pyproject.toml", "/index.html/..", "/%2e%2e/README.md"):
        assert fetch(server, "GET", path)[0] == 404


def test_the_api_answers_only_requests_with_the_token_to_its_own_address(server: Any) -> None:
    good = {TOKEN_HEADER: "secret-token"}
    status, headers, body = fetch(server, "GET", "/api/session", good)
    assert status == 200
    assert headers["cache-control"] == "no-store"
    assert json.loads(body)["halley"] == halley.__version__
    assert fetch(server, "GET", "/api/session")[0] == 403
    assert fetch(server, "GET", "/api/session", {TOKEN_HEADER: "secret-tokem"})[0] == 403
    # a page of another site reaching this port through a rebound host name names that host
    rebound = fetch(server, "GET", "/api/session", {**good, "Host": f"evil.example:{server.port}"})
    assert rebound[0] == 403
    assert fetch(server, "GET", "/", {"Host": f"localhost:{server.port}"})[0] == 200
    # a cross-origin call with a custom header is preflighted, and the preflight is not answered
    assert fetch(server, "OPTIONS", "/api/session", {"Origin": "https://evil.example"})[0] == 501
    assert fetch(server, "POST", "/api/session", good)[0] in (405, 415)
    assert fetch(server, "GET", "/api/nope", good)[0] == 404
    text = fetch(server, "POST", "/api/build", {**good, "Content-Type": "text/plain"}, b"{}")
    assert text[0] == 415
    status, _headers, body = fetch(
        server, "POST", "/api/build", {**good, "Content-Type": "application/json"}, b"{nope"
    )
    assert status == 400
    assert json.loads(body)["error"]["message"].startswith("the body is not JSON")


def test_an_engine_error_reaches_the_page_as_json(server: Any) -> None:
    good = {TOKEN_HEADER: "secret-token", "Content-Type": "application/json"}
    body = json.dumps({"path": "C:/no/such/model.py"}).encode()
    status, _headers, raw = fetch(server, "POST", "/api/models/load", good, body)
    assert status == 422
    assert "there is no file" in json.loads(raw)["error"]["message"]


def test_jsonable_keeps_every_bit_and_names_what_json_cannot_hold() -> None:
    value = {
        "a": np.array([0.1, -0.0, 1e-300]),
        "b": np.array([1.0, math.nan, math.inf, -math.inf]),
        "c": np.int64(7),
        "d": np.datetime64("2027-01-01"),
        "e": (np.bool_(True), dt.date(2027, 1, 1), Path("x")),
    }
    out = jsonable(value)
    text = json.dumps(out, allow_nan=False)
    back = json.loads(text)
    assert back["a"] == [0.1, -0.0, 1e-300]
    assert math.copysign(1.0, back["a"][1]) == -1.0
    assert back["b"] == [1.0, "NaN", "Infinity", "-Infinity"]
    assert back["c"] == 7
    assert back["d"] == "2027-01-01"
    assert back["e"] == [True, "2027-01-01", "x"]


def test_the_pages_ship_with_the_package() -> None:
    files = importlib.resources.files("halley.gui").joinpath("static")
    for name, _media in STATIC_FILES.values():
        assert files.joinpath(name).is_file(), name
    index = files.joinpath("index.html").read_text(encoding="utf-8")
    assert '<link rel="stylesheet" href="/app.css" />' in index
    script = files.joinpath("app.js").read_text(encoding="utf-8")
    assert 'from "/chart.js"' in script
    # every value a page shows is set as text: no HTML is built from data
    for name in ("app.js", "chart.js"):
        text = files.joinpath(name).read_text(encoding="utf-8")
        assert "innerHTML" not in text, name
        assert "insertAdjacentHTML" not in text, name


def test_the_gui_needs_nothing_beyond_halleys_dependencies() -> None:
    folder = Path(halley.gui.__file__).parent
    allowed = {"numpy", "pyarrow", "halley", *sys.stdlib_module_names}
    for path in folder.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            for name in names:
                assert name.split(".")[0] in allowed, (path.name, name)


# ------------------------------------------------------------------------ jobs and the launcher


def test_a_queued_job_can_be_cancelled_and_a_running_one_finishes() -> None:
    jobs = Jobs()
    gate = threading.Event()
    started = threading.Event()

    def first_work() -> str:
        started.set()
        gate.wait(30)
        return "first"

    try:
        first = jobs.submit("run", "first", first_work, lambda e: {"message": str(e)})
        assert started.wait(30)
        second = jobs.submit("run", "second", lambda: "second", lambda e: {"message": str(e)})
        assert jobs.cancel(second.id) is True
        assert jobs.cancel(first.id) is False  # running: a run cannot be stopped part way
        gate.set()
        jobs.wait(first.id, timeout=30)
        ran, skipped = jobs.get(first.id), jobs.get(second.id)
        assert ran is not None
        assert skipped is not None
        assert ran.as_dict()["result"] == "first"
        assert ran.state == DONE
        assert skipped.state == CANCELLED
        failing = jobs.submit(
            "build", "fails", lambda: 1 / 0, lambda e: {"message": type(e).__name__}
        )
        jobs.wait(failing.id, timeout=30)
        assert failing.as_dict()["error"] == {"message": "ZeroDivisionError"}
    finally:
        gate.set()
        jobs.close()


def test_the_launcher_takes_a_free_port_when_its_own_is_taken(app: App) -> None:
    args = cli.parse_args([])
    assert (args.host, args.port, args.no_browser) == ("127.0.0.1", cli.DEFAULT_PORT, False)
    holder = make_server(app.routes(), "127.0.0.1", 0)
    try:
        other = cli.bind(app, "127.0.0.1", holder.port)
        try:
            assert other.port != holder.port
            assert other.browser_url().startswith(f"http://127.0.0.1:{other.port}/#token=")
        finally:
            other.server_close()
    finally:
        holder.server_close()


# ------------------------------------------------------------------- what the review found


def mort_table() -> Any:
    ages = list(range(18, 131))
    return pa.table({"age": ages, "q_ann": [min(1.0, 0.0005 * 1.09 ** (x - 18)) for x in ages]})


def tutorial_points(**columns: Any) -> Any:
    data: dict[str, Any] = {
        "policy_id": ["A", "B", "C"],
        "age": [35, 50, 62],
        "sum_assured": [100_000.0, 250_000.0, 50_000.0],
        "premium": [20.0, 90.0, 60.0],
        "pols_if_init": [1.0, 1.0, 1.0],
        "rem_m": [120, 60, 36],
    }
    data.update(columns)
    return pa.table(data)


def run_example(app: App, build: str, name: str) -> dict[str, Any]:
    settings = {"valuation_date": "2027-01-01"}
    inputs = {"example": "term-reference"}
    return done(
        app, call(app, "POST", "/api/run", build=build, inputs=inputs, settings=settings, name=name)
    )


def test_a_page_address_reads_only_run_directories_the_session_knows(
    app: App, term_run: RunResult
) -> None:
    """Every GET can be sent by the page's router from its address alone, so a site that sends
    the GUI's tab to an address of its choosing must not make the server open a path of its
    choosing: a network share, say, which Windows would connect to with the user's sign-in."""
    outside = str(term_run.run_dir)
    query = {"dir": outside, "name": "manifest", "kind": "model_points"}
    for route in ("/api/run", "/api/run/file", "/api/run/input"):
        error = refused(app, "GET", route, query)
        assert error.status == 404
        assert "open it on the Runs page first" in error.message
    share = r"\\no-such-host.invalid\share\run" if sys.platform == "win32" else "/no/such/run"
    assert (
        "open it on the Runs page first" in refused(app, "GET", "/api/run", {"dir": share}).message
    )
    assert refused(app, "GET", "/api/compare", {"a": outside, "b": outside}).status == 404
    assert call(app, "POST", "/api/runs/open", path=outside)["run_id"] == term_run.run_id
    assert call(app, "GET", "/api/run", {"dir": outside})["run_id"] == term_run.run_id


def test_a_relative_run_directory_is_opened_from_the_working_folder_then_the_runs_folder(
    app: App, compiled_term: CompiledModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    settings = RunSettings(valuation_date=VALUATION_DATE)
    points, tables = reference_model_points(), reference_tables()
    elsewhere = run(
        compiled_term,
        model_points=points,
        tables=tables,
        settings=settings,
        run_dir=tmp_path / "elsewhere" / "r",
    )
    filed = run(
        compiled_term,
        model_points=points,
        tables=tables,
        settings=settings,
        run_dir=app.runs_root / "filed",
    )
    opened = call(app, "POST", "/api/runs/open", path="elsewhere/r")
    assert opened["dir"] == str(elsewhere.run_dir.resolve())
    assert call(app, "POST", "/api/runs/open", path="filed")["dir"] == str(filed.run_dir.resolve())


def test_reruns_of_a_long_named_run_each_get_a_name_of_their_own(app: App) -> None:
    """A rerun's folder is ``<name>-rerun``, then ``-rerun-2`` and so on, cut from the name's
    end, never the number's: a name of 100 characters gave the same folder for every number,
    and the search for a free one never ended, holding the session's lock."""
    name = "n" * 100
    made: list[Path] = []
    for _ in range(4):
        target = app._rerun_dir(name, "rerun")
        assert len(target.name) <= 100
        assert target not in made
        target.mkdir(parents=True)
        app._release(target)
        made.append(target)
    ends = ("-rerun", "-rerun-2", "-rerun-3", "-rerun-4")
    assert all(p.name.endswith(end) for p, end in zip(made, ends, strict=True))


def test_a_cancelled_run_gives_its_name_back(app: App) -> None:
    gate = threading.Event()
    started = threading.Event()

    def hold() -> None:
        started.set()
        gate.wait(60)

    build = build_term(app)
    blocker = app.jobs.submit("run", "hold", hold, lambda e: {"message": str(e)})
    body = {
        "build": build,
        "inputs": {"example": "term-reference"},
        "settings": {"valuation_date": "2027-01-01"},
    }
    try:
        assert started.wait(30)
        queued = call(app, "POST", "/api/run", name="x", **body)
        assert refused(app, "POST", "/api/run", name="x", **body).status == 409
        assert call(app, "POST", "/api/job/cancel", id=queued["job"]["id"]) == {"cancelled": True}
        again = call(app, "POST", "/api/run", name="x", **body)  # the name is free again
    finally:
        gate.set()
    app.jobs.wait(blocker.id, timeout=30)
    assert Path(done(app, again)["run"]).name == "x"


def test_a_run_directory_used_again_shows_the_run_it_holds_now(
    app: App, compiled_term: CompiledModel
) -> None:
    """The tutorial deletes its runs folder and runs again into the same directories: a run the
    session holds for a directory is dropped when the directory's manifest has changed."""
    first = run_example(app, build_term(app), "again")
    assert call(app, "GET", "/api/run/policies", {"dir": first["run"], "limit": 1})["total"] == 10
    shutil.rmtree(first["run"])
    second = run(
        compiled_term,
        model_points=reference_model_points().slice(0, 3),
        tables=reference_tables(),
        settings=RunSettings(valuation_date=VALUATION_DATE),
        run_dir=first["run"],
    )
    summary = call(app, "GET", "/api/run", {"dir": first["run"]})
    assert summary["run_id"] == second.run_id != first["run_id"]
    assert summary["in_session"] is False
    page = call(app, "GET", "/api/run/policies", {"dir": first["run"], "limit": 5})
    assert page["total"] == 3


def test_verifying_a_run_made_in_the_session_keeps_it_explainable(app: App) -> None:
    made = run_example(app, build_term(app), "v")
    checked = call(app, "GET", "/api/run", {"dir": made["run"], "verify": "1"})
    assert (checked["verified"], checked["explainable"], checked["in_session"]) == (
        True,
        True,
        True,
    )
    Path(made["run"], "outputs", "scalars.parquet").write_bytes(b"changed")
    error = refused(app, "GET", "/api/run", {"dir": made["run"], "verify": "1"})
    assert "does not match the sha256" in error.message


def test_the_policy_finder_gives_every_place_of_an_id(app: App) -> None:
    made = run_example(app, build_term(app), "f")
    found = call(app, "GET", "/api/run/find", {"dir": made["run"], "id": "P05"})
    assert found == {"id": "P05", "p": [4], "count": 1}
    assert call(app, "GET", "/api/run/find", {"dir": made["run"], "id": "Z"})["count"] == 0


def test_build_time_settings_are_the_ones_python_would_pass(app: App, tmp_path: Path) -> None:
    """An untouched build form sends nothing, so the GUI builds what build(cls) builds; a typed
    value is read by its parameter's type; and a rerun takes the manifest's values as JSON kept
    them, so a run made in Python with defaults that JSON cannot write exactly is rerun."""
    head = "import enum\n\n\nclass Basis(enum.Enum):\n    BEST = 'best estimate'\n"
    body = (
        "    shock: Static[float] = 1\n"
        "    weights: Static[tuple[float, ...]] = (1.0, 0.5)\n"
        "    basis: Static[Basis] = Basis.BEST\n"
    )
    path = write_model(tmp_path, _stem(tmp_path, "statics"), head=head, body=body)
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    entry = app.catalog.get(model_id)
    assert entry is not None
    statics = {p["name"]: p for p in call(app, "GET", "/api/model", {"id": model_id})["statics"]}
    assert statics["weights"]["text"] == "[1.0, 0.5]"
    assert statics["shock"]["text"] == "1"
    python = halley.build(entry.cls)
    untouched = done(app, call(app, "POST", "/api/build", model=model_id, statics={}))["build"]
    assert app._build(untouched).compiled.ir_hash == python.ir_hash
    typed = {"shock": "2", "weights": "[2, 1]"}
    made = done(app, call(app, "POST", "/api/build", model=model_id, statics=typed))["build"]
    built = app._build(made).compiled.graph.statics
    assert (repr(built["shock"]), repr(built["weights"])) == ("2.0", "(2.0, 1.0)")
    wrong = refused(app, "POST", "/api/build", model=model_id, statics={"shock": "two"})
    assert wrong.status == 400
    done_in_python = run(
        python,
        model_points=tutorial_points(),
        tables={"mort": mort_table()},
        settings=RunSettings(valuation_date=VALUATION_DATE),
        run_dir=app.runs_root / "python",
    )
    again = done(app, call(app, "POST", "/api/run/rerun", dir=str(done_in_python.run_dir)))
    assert again["same_run_id"] is True
    assert again["outputs"]["bit_for_bit"] is True


def test_a_model_defined_in_a_script_is_refused_without_running_the_script(
    app: App, tmp_path: Path
) -> None:
    script = tmp_path / "script_model.py"
    script.write_text("raise SystemExit('the script ran')\n", encoding="utf-8")
    with pytest.raises(ModelFileError, match="__main__"):
        app.catalog.entry_for("__main__.Term", str(script))
    assert "script_model" not in sys.modules


def test_a_reload_brings_in_an_edited_helper_module(app: App, tmp_path: Path) -> None:
    stem = _stem(tmp_path, "helped")
    helper = tmp_path / f"{stem}_basis.py"
    helper.write_text("LOADING = 1.0\n", encoding="utf-8")
    path = write_model(tmp_path, stem, head=f"from {stem}_basis import LOADING\n")
    source = path.read_text(encoding="utf-8")
    path.write_text(
        source.replace("* self.mp.sum_assured", "* self.mp.sum_assured * LOADING"),
        encoding="utf-8",
    )
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]

    def formula() -> str:
        made = done(app, call(app, "POST", "/api/build", model=model_id))["build"]
        built = call(app, "GET", "/api/build", {"id": made})
        return next(v["formula"] for v in built["variables"] if v["name"] == "death_outgo")

    assert "1.0" in formula()
    time.sleep(0.05)
    helper.write_text("LOADING = 1.5\n", encoding="utf-8")
    assert call(app, "GET", "/api/models")["files"][0]["changed"] is True
    call(app, "POST", "/api/models/load", path=str(path))
    assert "1.5" in formula()


def test_a_model_whose_declarations_slip_still_shows_its_page(app: App, tmp_path: Path) -> None:
    """A one-element tuple written without its comma: the page still shows the class, with what
    went wrong, and the build gives the tracer's own message at the class."""
    body = (
        "    invariants = (Invariant('in force', lhs=lambda v: v.var('pols_if'), kind='ge',"
        " reads=('pols_if',)))\n"
    )
    head = "from halley import Invariant\n"
    path = write_model(tmp_path, _stem(tmp_path, "slip"), head=head, body=body)
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    page = call(app, "GET", "/api/model", {"id": model_id})
    assert page["error"]["type"] == "TypeError"
    failed = finish(app, call(app, "POST", "/api/build", model=model_id))
    assert failed["state"] == "failed"
    assert "trailing comma" in failed["error"]["message"]


def test_a_job_that_raises_any_exception_fails(app: App) -> None:
    class Abort(BaseException):
        pass

    def work() -> None:
        raise Abort("stop")

    job = app.jobs.submit("build", "abort", work, lambda e: {"message": type(e).__name__})
    app.jobs.wait(job.id, timeout=30)
    assert job.state == "failed"
    assert job.error == {"message": "Abort"}


def test_one_damaged_run_directory_does_not_hide_the_others(app: App, term_run: RunResult) -> None:
    damaged = app.runs_root / "damaged"
    damaged.mkdir(parents=True)
    manifest = json.loads(Path(term_run.run_dir, "manifest.json").read_text(encoding="utf-8"))
    manifest["settings"] = None
    (damaged / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    shutil.copytree(term_run.run_dir, app.runs_root / "good")
    rows = {r["name"]: r for r in call(app, "GET", "/api/runs")["runs"]}
    assert "AttributeError" in rows["damaged"]["error"]
    assert rows["good"]["run_id"] == term_run.run_id


def test_a_stage_run_made_with_run_is_rerun_with_its_series(app: App, tmp_path: Path) -> None:
    """A stage run by ``run()`` on its group series has no groups file or membership to replay
    from: it is run again on its own copies of the series."""
    from halley.aggregation import group_series
    from models import MEMBERS, Release, release_bases, run_stage

    stage = run_stage(
        app.runs_root, Release, group_series(Release, release_bases(tmp_path), MEMBERS)
    )
    again = done(app, call(app, "POST", "/api/run/rerun", dir=str(stage.run_dir)))
    assert again["same_run_id"] is True
    assert again["outputs"]["bit_for_bit"] is True
    e = call(app, "GET", "/api/run/explain", {"dir": again["run"], "p": 0, "var": "csm", "t": 1})
    assert e["lane"] == "group"


def test_compare_points_at_the_place_of_a_repeated_id(app: App, tmp_path: Path) -> None:
    path = write_model(tmp_path, _stem(tmp_path, "dup"), unique=False)
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    entry = app.catalog.get(model_id)
    assert entry is not None
    compiled = halley.build(entry.cls)
    settings = RunSettings(valuation_date=VALUATION_DATE)
    ids = ["A", "A", "B"]
    a = run(
        compiled,
        model_points=tutorial_points(policy_id=ids),
        tables={"mort": mort_table()},
        settings=settings,
        run_dir=app.runs_root / "a",
    )
    b = run(
        compiled,
        model_points=tutorial_points(policy_id=ids, sum_assured=[100_000.0, 300_000.0, 50_000.0]),
        tables={"mort": mort_table()},
        settings=settings,
        run_dir=app.runs_root / "b",
    )
    rows = call(app, "GET", "/api/compare", {"a": str(a.run_dir), "b": str(b.run_dir)})["variables"]
    row = next(v for v in rows if v["variable"] == "reserve")
    assert row["p"] == 1
    assert row["a"] == a.var("reserve")[row["t"], 1, 0]
    page = call(app, "GET", "/api/run/policies", {"dir": str(a.run_dir), "sort": "id"})
    assert [r["fields"][0] for r in page["rows"]] == [35, 50, 62]  # each A's own age


def test_explain_agrees_with_a_bool_held_before_its_domain(app: App) -> None:
    """Before a Bool variable's first month the interpreter gives 0.0 where the run holds False:
    the same value."""
    model = "halley.products.annuity:FixedEscalationAnnuity"
    build = done(app, call(app, "POST", "/api/build", model=model))["build"]
    settings = {"valuation_date": "2025-12-31"}
    inputs = {"example": "annuity-fixed"}
    made = done(
        app,
        call(app, "POST", "/api/run", build=build, inputs=inputs, settings=settings, name="fixed"),
    )
    e = call(app, "GET", "/api/run/explain", {"dir": made["run"], "p": 0, "var": "anniv", "t": 0})
    assert e["value"] is False
    assert e["recomputed"]["same"] is True


def test_the_server_answers_the_default_port_without_it_and_ipv6(app: App) -> None:
    server = make_server(app.routes(), "127.0.0.1", 0)
    try:
        server.server_address = ("127.0.0.1", 80)
        assert {"127.0.0.1", "127.0.0.1:80", "localhost", "[::1]"} <= server.hosts()
    finally:
        server.server_close()
    try:
        six = make_server(app.routes(), "::1", 0)
    except OSError:
        pytest.skip("this machine has no IPv6 loopback")
    try:
        assert six.browser_url().startswith(f"http://[::1]:{six.port}/")
        assert f"[::1]:{six.port}" in six.hosts()
    finally:
        six.server_close()


def test_a_refusal_reaches_a_client_that_sent_a_body(server: Any) -> None:
    """A refusal made before the body is read was lost on Windows, which resets a connection
    closed with data unread: the server reads the body first."""
    for _ in range(40):
        status, _headers, _body = fetch(
            server, "POST", "/api/build", {"Content-Type": "text/plain"}, b"{}"
        )
        assert status == 403  # no token


# --------------------------------------------- build-time settings, helper modules, memory


STATICS_HEAD = (
    "import datetime as dt\nimport enum\n\n\n"
    "class Basis(enum.Enum):\n    BEST = 'best estimate'\n    PRUDENT = 'prudent'\n"
)
STATICS_BODY = (
    "    shock: Static[float] = 1.0\n"
    "    weights: Static[tuple[float, ...]] = (1.0, 0.5)\n"
    "    basis: Static[Basis] = Basis.BEST\n"
    "    start: Static[dt.date] = dt.date(2027, 1, 1)\n"
    "    cap: Static[float | None] = 1.0\n"
    "    flag: Static[bool] = False\n"
)


def statics_model(app: App, tmp_path: Path, what: str) -> tuple[str, Any]:
    """A model file whose class declares build-time parameters of every kind of field."""
    path = write_model(tmp_path, _stem(tmp_path, what), head=STATICS_HEAD, body=STATICS_BODY)
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    entry = app.catalog.get(model_id)
    assert entry is not None
    return model_id, entry.cls


def ir_of(app: App, build_id: str) -> str:
    return app._build(build_id).compiled.ir_hash


def shown_build(app: App, model_id: str, build_id: str) -> dict[str, Any]:
    builds = call(app, "GET", "/api/model", {"id": model_id})["builds"]
    return next(b for b in builds if b["id"] == build_id)


def test_build_again_builds_the_build_the_form_shows(app: App, tmp_path: Path) -> None:
    """The form shows a build's settings as the server writes them from the values, and sends
    only the fields changed from what it shows, with the build it shows. So an untouched Build
    again builds that build again, IR hash and all, whatever the settings' types (a float, a
    tuple of floats, an enum, a date, an optional float); a field set back to the default's
    own text builds the default."""
    model_id, cls = statics_model(app, tmp_path, "again")
    params = {p["name"]: p for p in call(app, "GET", "/api/model", {"id": model_id})["statics"]}
    assert params["basis"]["input"] == "choice"
    assert params["basis"]["choices"] == ["BEST", "PRUDENT"]
    assert params["start"]["input"] == "date"
    assert params["cap"]["input"] == "json"
    defaults = {n: p["text"] for n, p in params.items()}
    assert defaults == {
        "shock": "1.0",
        "weights": "[1.0, 0.5]",
        "basis": "BEST",
        "start": "2027-01-01",
        "cap": "1.0",
        "flag": "false",
    }
    first = done(app, call(app, "POST", "/api/build", model=model_id, statics={}))["build"]
    assert ir_of(app, first) == halley.build(cls).ir_hash
    assert shown_build(app, model_id, first)["statics_text"] == defaults
    again = call(app, "POST", "/api/build", model=model_id, statics={}, from_build=first)
    assert ir_of(app, done(app, again)["build"]) == ir_of(app, first)

    changed = {
        "basis": "PRUDENT",
        "start": "2028-06-30",
        "weights": "[1, 0.5]",
        "cap": "2",
        "flag": True,
    }
    other = done(
        app, call(app, "POST", "/api/build", model=model_id, statics=changed, from_build=first)
    )["build"]
    basis = sys.modules[cls.__module__].Basis
    expected = halley.build(
        cls,
        basis=basis.PRUDENT,
        start=dt.date(2028, 6, 30),
        weights=(1.0, 0.5),
        cap=2.0,
        flag=True,
    )
    assert ir_of(app, other) == expected.ir_hash
    texts = shown_build(app, model_id, other)["statics_text"]
    assert (texts["basis"], texts["weights"], texts["cap"], texts["flag"]) == (
        "PRUDENT",
        "[1.0, 0.5]",
        "2.0",
        "true",
    )
    untouched = call(app, "POST", "/api/build", model=model_id, statics={}, from_build=other)
    assert ir_of(app, done(app, untouched)["build"]) == expected.ir_hash
    reverted = {**{n: defaults[n] for n in changed}, "flag": False}
    back = call(app, "POST", "/api/build", model=model_id, statics=reverted, from_build=other)
    assert ir_of(app, done(app, back)["build"]) == halley.build(cls).ir_hash

    wrong = refused(app, "POST", "/api/build", model=model_id, statics={"basis": "LAX"})
    assert wrong.status == 400
    assert "BEST, PRUDENT" in wrong.message
    elsewhere = refused(app, "POST", "/api/build", model=TERM, statics={}, from_build=first)
    assert elsewhere.status == 400


def test_a_rerun_builds_the_settings_the_manifest_records(app: App, tmp_path: Path) -> None:
    """A run made in Python with settings that differ from the defaults only in type (1 against
    1.0, a tuple of ints against one of floats), and with an enum's member and a date other than
    the defaults, is rerun with the same run id; a record that does not read back is refused
    before anything is built."""
    _model_id, cls = statics_model(app, tmp_path, "rerun")
    basis = sys.modules[cls.__module__].Basis
    compiled = halley.build(
        cls, shock=1, weights=(1, 0.5), basis=basis.PRUDENT, start=dt.date(2028, 6, 30)
    )
    made = run(
        compiled,
        model_points=tutorial_points(),
        tables={"mort": mort_table()},
        settings=RunSettings(valuation_date=VALUATION_DATE),
        run_dir=app.runs_root / "python-statics",
    )
    again = done(app, call(app, "POST", "/api/run/rerun", dir=str(made.run_dir)))
    assert again["same_run_id"] is True
    assert again["outputs"]["bit_for_bit"] is True
    with pytest.raises(HalleyError, match="none of its members"):
        recorded_statics(cls, {"basis": "<Basis.GONE: 'gone'>"})
    with pytest.raises(HalleyError, match="does not read back as a date"):
        recorded_statics(cls, {"start": "next spring"})
    assert recorded_statics(cls, {"shock": 1.0, "flag": False}) == {}


def loading_model(folder: Path, stem: str, helper: str, extra: str = "") -> Path:
    """The tutorial's model, its death outgo times the LOADING a helper module defines."""
    path = write_model(folder, stem, head=f"from {helper} import LOADING\n{extra}")
    source = path.read_text(encoding="utf-8")
    loaded = source.replace("* self.mp.sum_assured", "* self.mp.sum_assured * LOADING")
    path.write_text(loaded, encoding="utf-8")
    return path


def death_outgo(app: App, model_id: str) -> str:
    """The death outgo's formula as a build of the model now gives it."""
    made = done(app, call(app, "POST", "/api/build", model=model_id))["build"]
    built = call(app, "GET", "/api/build", {"id": made})
    return next(v["formula"] for v in built["variables"] if v["name"] == "death_outgo")


def edit(path: Path, old: str, new: str) -> None:
    time.sleep(0.05)  # a modification time the file system tells from the last
    path.write_text(path.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")


def test_a_helper_two_files_use_marks_both_changed(app: App, tmp_path: Path) -> None:
    """Two model files import one assumptions module: an edit to it marks both changed, though
    only the first one's import ran it, and a reload of the second builds with the edit."""
    stem = _stem(tmp_path, "shared")
    helper = tmp_path / f"{stem}_basis.py"
    helper.write_text("LOADING = 1.0\n", encoding="utf-8")
    ids = []
    for which in ("a", "b"):
        path = loading_model(tmp_path, f"{stem}_{which}", f"{stem}_basis")
        ids.append(call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0])
    files = call(app, "GET", "/api/models")["files"]
    assert [f["helpers"] for f in files] == [[f"{stem}_basis"], [f"{stem}_basis"]]
    edit(helper, "1.0", "1.5")
    assert [f["changed"] for f in call(app, "GET", "/api/models")["files"]] == [True, True]
    call(app, "POST", "/api/models/load", path=files[1]["path"])
    assert "1.5" in death_outgo(app, ids[1])
    call(app, "POST", "/api/models/load", path=files[0]["path"])
    assert "1.5" in death_outgo(app, ids[0])


def test_a_failed_load_leaves_no_helper_untracked(app: App, tmp_path: Path) -> None:
    """A file that imports a helper and then fails: the helper goes with the failed import, so
    the next load runs it again and tracks it."""
    stem = _stem(tmp_path, "failed")
    helper = tmp_path / f"{stem}_basis.py"
    helper.write_text("LOADING = 1.0\n", encoding="utf-8")
    path = loading_model(tmp_path, stem, f"{stem}_basis", extra="undefined_name\n")
    assert refused(app, "POST", "/api/models/load", path=str(path)).status == 422
    assert f"{stem}_basis" not in sys.modules
    edit(path, "undefined_name\n", "")
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    edit(helper, "1.0", "1.5")
    assert call(app, "GET", "/api/models")["files"][0]["changed"] is True
    call(app, "POST", "/api/models/load", path=str(path))
    assert "1.5" in death_outgo(app, model_id)


def test_a_model_file_another_imports_keeps_one_module(app: App, tmp_path: Path) -> None:
    """b.py, a model file, is imported by a.py. Loading b after a takes the module a's import
    made, as the file was when it ran, so an edit made since shows as a change; a reload of a
    reloads b first, and leaves b's module to b, so b's models stay the classes its module
    holds (as a run's worker processes need)."""
    stem = _stem(tmp_path, "pair")
    b = loading_model(tmp_path, f"{stem}_b", "math", extra="LOADING = 1.0\n")
    edit(b, "from math import LOADING\n", "")
    a = loading_model(tmp_path, f"{stem}_a", f"{stem}_b")
    a_id = call(app, "POST", "/api/models/load", path=str(a))["file"]["models"][0]
    edit(b, "LOADING = 1.0", "LOADING = 1.5")
    loaded_b = call(app, "POST", "/api/models/load", path=str(b))["file"]
    assert loaded_b["changed"] is True
    b_id = loaded_b["models"][0]
    assert "1.0" in death_outgo(app, b_id)  # the module a's import made, until b is reloaded
    call(app, "POST", "/api/models/load", path=str(a))
    assert "1.5" in death_outgo(app, b_id)
    assert "1.5" in death_outgo(app, a_id)
    for _ in range(2):
        entry = app.catalog.get(b_id)
        assert entry is not None
        assert sys.modules[f"{stem}_b"].Term is entry.cls
        call(app, "POST", "/api/models/load", path=str(a))


def test_python_s_own_modules_are_never_the_user_s(tmp_path: Path) -> None:
    """Python's standard library (its compiled modules too) and installed packages are
    libraries, however their paths are written; a file of the user's is not."""
    import json as json_module
    import unicodedata

    assert is_library_file(json_module.__file__)
    folder = os.path.dirname(json_module.__file__)
    assert is_library_file(os.path.join(folder, os.pardir, "json", "__init__.py"))
    compiled = getattr(unicodedata, "__file__", None)  # in DLLs on Windows; built in elsewhere
    assert compiled is None or is_library_file(compiled)
    assert is_library_file(np.__file__)
    mine = tmp_path / "mine.py"
    mine.write_text("x = 1\n", encoding="utf-8")
    assert not is_library_file(str(mine))


def test_an_error_in_a_model_file_points_at_its_own_line(app: App, tmp_path: Path) -> None:
    """A model file whose top level fails inside Python's own code: the error is placed at the
    file's line that called it, not in the standard library."""
    line = "SETTINGS = json.loads('{\"loading\": 1.0')"
    path = write_model(tmp_path, _stem(tmp_path, "decode"), head=f"import json\n{line}\n")
    error = refused(app, "POST", "/api/models/load", path=str(path))
    assert error.status == 422
    loc: Any = error.detail["loc"]
    assert Path(loc["file"]).resolve() == path.resolve()
    assert loc["line"] == path.read_text(encoding="utf-8").splitlines().index(line) + 1


def test_the_session_lets_go_of_the_run_used_least_recently(
    app: App, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Beyond the runs it may hold, the session lets go of the one used least recently, not the
    one made first; explain of a run it let go of says so, and offers the rerun."""
    monkeypatch.setattr("halley.gui.app.MAX_SESSION_RUNS", 2)
    build = build_term(app)
    first = run_example(app, build, "lru-1")["run"]
    second = run_example(app, build, "lru-2")["run"]
    query = {"dir": first, "p": 4, "var": "reserve", "t": 1}
    call(app, "GET", "/api/run/explain", query)  # the first is now the most recently used
    run_example(app, build, "lru-3")
    assert call(app, "GET", "/api/run/explain", query)["value"] is not None
    let_go = refused(app, "GET", "/api/run/explain", {**query, "dir": second})
    assert let_go.status == 409
    assert "let go" in let_go.message
    assert let_go.detail["rerun"] is True


def test_a_runs_folder_at_a_drive_root_reads_its_runs(term_run: RunResult) -> None:
    """A runs folder at the root of a drive (or a share, or /) holds every run beneath it."""
    at_root = App(Path(Path(term_run.run_dir).anchor))
    try:
        answer = call(at_root, "GET", "/api/run", {"dir": str(term_run.run_dir)})
        assert answer["run_id"] == term_run.run_id
    finally:
        at_root.close()


def test_a_wildcard_bind_answers_the_address_this_computer_routes_from() -> None:
    assert cli.routed_addresses() <= set(cli.own_names("0.0.0.0"))
    assert cli.own_names("127.0.0.1") == ()


# ------------------------------------------------- reloads keep one copy of each module


def string_annotated(path: Path) -> None:
    """A model file with ``from __future__ import annotations``, as the shipped products have."""
    first, rest = path.read_text(encoding="utf-8").split("\n", 1)
    path.write_text(f"{first}\nfrom __future__ import annotations\n{rest}", encoding="utf-8")


def subclass_file(folder: Path, stem: str, base: str, name: str, extra: str = "") -> Path:
    path = folder / f"{stem}.py"
    path.write_text(
        f'"""{name}."""\n\nfrom {base} import Term\n{extra}\n\nclass {name}(Term):\n'
        f'    """{name}."""\n',
        encoding="utf-8",
    )
    return path


def test_a_reload_keeps_one_copy_of_a_shared_base_module(app: App, tmp_path: Path) -> None:
    """Two product files subclass a base model in a shared module whose annotations are strings:
    a reload runs both again with one new copy of the shared module, so each still builds what
    Python builds."""
    stem = _stem(tmp_path, "base")
    head = "import enum\n\n\nclass Basis(enum.Enum):\n    BEST = 'best estimate'\n"
    body = "    basis: Static[Basis] = Basis.BEST\n"
    string_annotated(write_model(tmp_path, f"{stem}_shared", head=head, body=body))
    paths = [subclass_file(tmp_path, f"{stem}_{n}", f"{stem}_shared", n.title()) for n in "ab"]
    ids = [call(app, "POST", "/api/models/load", path=str(p))["file"]["models"][0] for p in paths]
    before = done(app, call(app, "POST", "/api/build", model=ids[0]))["build"]
    edit(paths[1], '"""B."""\n\nfrom', '"""B, edited."""\n\nfrom')
    call(app, "POST", "/api/models/load", path=str(paths[1]))
    assert [f["changed"] for f in call(app, "GET", "/api/models")["files"]] == [False, False]
    after = done(app, call(app, "POST", "/api/build", model=ids[0]))["build"]
    assert ir_of(app, after) == ir_of(app, before)


def test_a_reload_runs_each_file_once_with_one_copy_of_what_both_use(
    app: App, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """b.py subclasses a.py's model and both import an enums module: an edit to a.py, then a
    reload of b, runs a and then b, each once, with the one enums module, so b's model builds as
    Python builds it."""
    stem = _stem(tmp_path, "nested")
    enums = tmp_path / f"{stem}_enums.py"
    enums.write_text(
        "import enum\n\n\nclass Basis(enum.Enum):\n    BEST = 'best'\n    PRUDENT = 'prudent'\n",
        encoding="utf-8",
    )
    head = f"from {stem}_enums import Basis\n"
    a = write_model(
        tmp_path, f"{stem}_a", head=head, body="    basis: Static[Basis] = Basis.BEST\n"
    )
    loading = "(1.5 if self.basis is Basis.PRUDENT else 1.0)"
    edit(a, "* self.mp.sum_assured\n", f"* self.mp.sum_assured * {loading}\n")
    b = subclass_file(
        tmp_path, f"{stem}_b", f"{stem}_a", "Prudent", f"from {stem}_enums import Basis\n"
    )
    edit(b, '    """Prudent."""\n', '    """Prudent."""\n\n    basis = Basis.PRUDENT\n')
    call(app, "POST", "/api/models/load", path=str(a))
    b_id = call(app, "POST", "/api/models/load", path=str(b))["file"]["models"][0]
    assert "1.5" in death_outgo(app, b_id)
    ran: list[str] = []
    real = app.catalog._exec
    monkeypatch.setattr(
        app.catalog, "_exec", lambda name, file: ran.append(name) or real(name, file)
    )
    edit(a, '"""A model file', '"""An edited model file')
    call(app, "POST", "/api/models/load", path=str(b))
    assert ran == [f"{stem}_a", f"{stem}_b"]
    assert "1.5" in death_outgo(app, b_id)
    assert [f["changed"] for f in call(app, "GET", "/api/models")["files"]] == [False, False]


def test_a_reload_takes_out_no_module_the_file_does_not_import_again(
    app: App, tmp_path: Path
) -> None:
    """other.py names a module product.py uses only under ``if __name__ == "__main__":``: after
    a reload, product.py has imported that module again, and still builds."""
    stem = _stem(tmp_path, "guarded")
    (tmp_path / f"{stem}_shared.py").write_text("LOADING = 1.5\n", encoding="utf-8")
    product = loading_model(tmp_path, f"{stem}_product", f"{stem}_shared")
    other = write_model(tmp_path, f"{stem}_other")
    edit(other, "V_M = ", f'if __name__ == "__main__":\n    import {stem}_shared\nV_M = ')
    product_id = call(app, "POST", "/api/models/load", path=str(product))["file"]["models"][0]
    call(app, "POST", "/api/models/load", path=str(other))
    edit(other, '"""A model file', '"""An edited model file')
    call(app, "POST", "/api/models/load", path=str(other))
    assert f"{stem}_shared" in sys.modules
    assert "1.5" in death_outgo(app, product_id)


def test_a_module_imported_by_a_formula_or_by_importlib_is_tracked(
    app: App, tmp_path: Path
) -> None:
    """A helper a formula imports when the build runs it, and one the file's top level reaches
    through ``importlib.import_module`` without binding it: an edit to either shows as a change,
    and a reload builds with it."""
    stem = _stem(tmp_path, "lazy")
    (tmp_path / f"{stem}_basis.py").write_text("LOADING = 1.0\n", encoding="utf-8")
    (tmp_path / f"{stem}_rates.py").write_text("SHOCK = 1.0\n", encoding="utf-8")
    head = f"import importlib\n\nSHOCK = importlib.import_module('{stem}_rates').SHOCK\n"
    path = write_model(tmp_path, stem, head=head)
    edit(
        path,
        "        return self.deaths(t) * self.mp.sum_assured\n",
        f"        from {stem}_basis import LOADING\n\n"
        "        return self.deaths(t) * self.mp.sum_assured * LOADING * SHOCK\n",
    )
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    assert "* 1.0) * 1.0)" in death_outgo(app, model_id)
    assert call(app, "GET", "/api/models")["files"][0]["helpers"] == [
        f"{stem}_basis",
        f"{stem}_rates",
    ]
    for helper, name in ((f"{stem}_basis", "LOADING"), (f"{stem}_rates", "SHOCK")):
        edit(tmp_path / f"{helper}.py", f"{name} = 1.0", f"{name} = 1.5")
        assert call(app, "GET", "/api/models")["files"][0]["changed"] is True
        call(app, "POST", "/api/models/load", path=str(path))
    assert "* 1.5) * 1.5)" in death_outgo(app, model_id)


def test_a_model_file_imported_before_it_is_loaded_shows_an_edit_made_since(
    app: App, tmp_path: Path
) -> None:
    """A model file imported before the catalogue loaded it (by the program that started the
    GUI, say): nothing records which version of the file that import ran, so it shows as
    changed until a reload imports it afresh, which builds the file as it is now."""
    stem = _stem(tmp_path, "early")
    path = loading_model(tmp_path, stem, "math", extra="LOADING = 1.0\n")
    edit(path, "from math import LOADING\n", "")
    sys.path.append(str(tmp_path))
    importlib.import_module(stem)
    edit(path, "LOADING = 1.0", "LOADING = 1.5")
    loaded = call(app, "POST", "/api/models/load", path=str(path))["file"]
    assert loaded["changed"] is True
    assert "1.0" in death_outgo(app, loaded["models"][0])
    call(app, "POST", "/api/models/reload")
    assert call(app, "GET", "/api/models")["changed"] is False
    assert "1.5" in death_outgo(app, loaded["models"][0])


def test_a_rerun_reads_back_members_of_int_str_and_flag_enums(app: App, tmp_path: Path) -> None:
    """A manifest records an int or str enum's member by its value, and a combined flag by its
    repr: a run made with members other than the defaults is rerun with the same run id."""
    head = (
        "import enum\n\n\nclass Level(enum.IntEnum):\n    LOW = 1\n    HIGH = 2\n\n\n"
        "class Mode(enum.StrEnum):\n    ANNUAL = 'annual'\n    MONTHLY = 'monthly'\n\n\n"
        "class Tier(str, enum.Enum):\n    BASE = 'base'\n    HIGH = 'high'\n\n\n"
        "class Perm(enum.Flag):\n    R = 1\n    W = 2\n"
    )
    body = (
        "    level: Static[Level] = Level.LOW\n    mode: Static[Mode] = Mode.ANNUAL\n"
        "    tier: Static[Tier] = Tier.BASE\n    perm: Static[Perm] = Perm.R\n"
    )
    path = write_model(tmp_path, _stem(tmp_path, "enums"), head=head, body=body)
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    entry = app.catalog.get(model_id)
    assert entry is not None
    m = sys.modules[entry.cls.__module__]
    compiled = halley.build(
        entry.cls,
        level=m.Level.HIGH,
        mode=m.Mode.MONTHLY,
        tier=m.Tier.HIGH,
        perm=m.Perm.R | m.Perm.W,
    )
    made = run(
        compiled,
        model_points=tutorial_points(),
        tables={"mort": mort_table()},
        settings=RunSettings(valuation_date=VALUATION_DATE),
        run_dir=app.runs_root / "python-enums",
    )
    again = done(app, call(app, "POST", "/api/run/rerun", dir=str(made.run_dir)))
    assert again["same_run_id"] is True
    call(app, "POST", "/api/runs/open", path=str(made.run_dir))
    shown = call(app, "GET", "/api/run", {"dir": str(made.run_dir)})
    assert json.loads(shown["statics_text"]) == made.manifest["model"]["statics"]


def test_enums_and_dates_inside_optional_and_tuple_settings(app: App, tmp_path: Path) -> None:
    """An enum's member or a date inside an Optional or a tuple: the form writes them by name and
    in ISO form inside the JSON and reads them back as the type says, and a rerun reads them
    back from the manifest's reprs."""
    head = (
        "import datetime as dt\nimport enum\n\n\n"
        "class Basis(enum.Enum):\n    BEST = 'best estimate'\n    PRUDENT = 'prudent'\n"
    )
    body = (
        "    maybe: Static[Basis | None] = None\n"
        "    when: Static[dt.date | None] = None\n"
        "    dates: Static[tuple[dt.date, ...]] = (dt.date(2027, 1, 1),)\n"
    )
    path = write_model(tmp_path, _stem(tmp_path, "nested_kinds"), head=head, body=body)
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    entry = app.catalog.get(model_id)
    assert entry is not None
    m = sys.modules[entry.cls.__module__]
    params = {
        p["name"]: p["text"] for p in call(app, "GET", "/api/model", {"id": model_id})["statics"]
    }
    assert params == {"maybe": "null", "when": "null", "dates": '["2027-01-01"]'}
    typed = {"maybe": '"PRUDENT"', "when": '"2028-06-30"', "dates": '["2027-01-01", "2028-01-01"]'}
    made = done(app, call(app, "POST", "/api/build", model=model_id, statics=typed))["build"]
    values = {
        "maybe": m.Basis.PRUDENT,
        "when": dt.date(2028, 6, 30),
        "dates": (dt.date(2027, 1, 1), dt.date(2028, 1, 1)),
    }
    expected = halley.build(entry.cls, **values)
    assert ir_of(app, made) == expected.ir_hash
    assert shown_build(app, model_id, made)["form"] == typed
    again = call(app, "POST", "/api/build", model=model_id, statics={}, from_build=made)
    assert ir_of(app, done(app, again)["build"]) == expected.ir_hash
    python = run(
        expected,
        model_points=tutorial_points(),
        tables={"mort": mort_table()},
        settings=RunSettings(valuation_date=VALUATION_DATE),
        run_dir=app.runs_root / "python-nested",
    )
    other = App(tmp_path / "other-runs")  # a session without the build
    try:
        rerun = done(other, call(other, "POST", "/api/run/rerun", dir=str(python.run_dir)))
        assert rerun["same_run_id"] is True
    finally:
        other.close()


def test_a_rerun_of_a_dict_setting_in_another_order_than_the_default(
    app: App, tmp_path: Path
) -> None:
    """A manifest writes a dict's keys sorted, so a dict setting in another order than the
    default's is recorded as the default is: the rerun tries the record as given too."""
    body = "    table: Static[dict[str, float]] = {'b': 2.0, 'a': 1.0}\n"
    path = write_model(tmp_path, _stem(tmp_path, "dict"), body=body)
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    entry = app.catalog.get(model_id)
    assert entry is not None
    for name, table in (("sorted", {"a": 1.0, "b": 2.0}), ("default", None)):
        compiled = (
            halley.build(entry.cls) if table is None else halley.build(entry.cls, table=table)
        )
        made = run(
            compiled,
            model_points=tutorial_points(),
            tables={"mort": mort_table()},
            settings=RunSettings(valuation_date=VALUATION_DATE),
            run_dir=app.runs_root / f"python-dict-{name}",
        )
        again = done(app, call(app, "POST", "/api/run/rerun", dir=str(made.run_dir)))
        assert again["same_run_id"] is True, name


def test_a_rerun_that_cannot_rebuild_the_run_says_why_and_keeps_no_build(
    app: App, tmp_path: Path
) -> None:
    """A run made with a list for a tuple setting: JSON records a list as a tuple's record, so
    the rerun cannot rebuild the run's IR; it says so, and keeps none of the builds it tried."""
    body = "    seq: Static[tuple[float, ...]] = (1.0, 0.5)\n"
    path = write_model(tmp_path, _stem(tmp_path, "listed"), body=body)
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    entry = app.catalog.get(model_id)
    assert entry is not None
    made = run(
        halley.build(entry.cls, seq=[1.0, 0.5]),
        model_points=tutorial_points(),
        tables={"mort": mort_table()},
        settings=RunSettings(valuation_date=VALUATION_DATE),
        run_dir=app.runs_root / "python-listed",
    )
    failed = finish(app, call(app, "POST", "/api/run/rerun", dir=str(made.run_dir)))
    assert failed["state"] == "failed"
    assert "cannot record exactly" in failed["error"]["message"]
    assert call(app, "GET", "/api/model", {"id": model_id})["builds"] == []


def test_build_again_after_a_reload_builds_the_class_as_it_is_now(app: App, tmp_path: Path) -> None:
    """The page shows a build of the file as it was: Build again builds the class as it is now
    with the settings the form shows, a setting that was the old default taking the new one,
    and one the form cannot write (an object of a class of the file's) the default."""
    head = (
        "import dataclasses\nimport enum\n\n\n"
        "class Basis(enum.Enum):\n    BEST = 'best estimate'\n    PRUDENT = 'prudent'\n\n\n"
        "@dataclasses.dataclass(frozen=True)\nclass Terms:\n    loading: float = 1.5\n"
    )
    body = (
        "    loading: Static[float] = 1.0\n"
        "    bases: Static[tuple[Basis, ...]] = (Basis.PRUDENT,)\n"
        "    terms: Static[Terms] = Terms()\n"
    )
    path = write_model(tmp_path, _stem(tmp_path, "stale"), head=head, body=body)
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    defaults = done(app, call(app, "POST", "/api/build", model=model_id, statics={}))["build"]
    loaded = call(app, "POST", "/api/build", model=model_id, statics={"loading": "1.25"})
    changed = done(app, loaded)["build"]
    edit(path, "(Basis.PRUDENT,)", "(Basis.BEST,)")
    call(app, "POST", "/api/models/load", path=str(path))
    entry = app.catalog.get(model_id)
    assert entry is not None
    for shown, expected in ((defaults, halley.build(entry.cls)), (changed, None)):
        assert shown_build(app, model_id, shown)["stale"] is True
        again = call(app, "POST", "/api/build", model=model_id, statics={}, from_build=shown)
        want = expected or halley.build(entry.cls, loading=1.25)
        assert ir_of(app, done(app, again)["build"]) == want.ir_hash


def test_a_reload_waits_for_the_job_that_is_running(app: App, tmp_path: Path) -> None:
    """A reload replaces the modules a running build or run uses, so it is refused while one
    runs; a file not loaded before still loads."""
    path = write_model(tmp_path, _stem(tmp_path, "busy"))
    call(app, "POST", "/api/models/load", path=str(path))
    started, release = threading.Event(), threading.Event()

    def work() -> None:
        started.set()
        release.wait(30)

    waiting = app._held(None, work)  # as a build holds the engine
    job = app.jobs.submit(
        "build", "Build a model that waits", waiting, lambda e: {"message": str(e)}
    )
    assert started.wait(10)
    try:
        refusal = refused(app, "POST", "/api/models/load", path=str(path))
        assert refusal.status == 409
        assert "Build a model that waits" in refusal.message
        other = write_model(tmp_path, _stem(tmp_path, "busy_new"))
        assert call(app, "POST", "/api/models/load", path=str(other))["file"]["models"]
    finally:
        release.set()
        app.jobs.wait(job.id, timeout=30)


def test_relative_names_and_the_environment_s_folder() -> None:
    """A frame of an installed package's compiled code has a relative name, which is never a
    model file's; a file in the environment's own folder, beside its packages, may be."""
    assert is_library_file(os.path.join("pyarrow", "error.pxi"))
    assert not is_library_file(os.path.join(sys.prefix, "term.py"))


def test_a_wildcard_server_answers_any_of_its_addresses(app: App) -> None:
    """Listening on every interface, the server answers a request to any IP address of its own
    with its port, whichever network it came in on; a name it does not know stays refused, and
    a server on the loopback interface answers no other address."""
    wide = make_server(app.routes(), "0.0.0.0", 0)
    try:
        port = wide.port
        assert wide.answers(f"10.0.0.5:{port}")
        assert wide.answers(f"[2001:db8::5]:{port}")
        assert not wide.answers(f"evil.example:{port}")
        assert not wide.answers(f"10.0.0.5:{port + 1}")
    finally:
        wide.server_close()
    loop = make_server(app.routes(), "127.0.0.1", 0)
    try:
        assert not loop.answers(f"10.0.0.5:{loop.port}")
        assert loop.answers(f"localhost:{loop.port}")
    finally:
        loop.server_close()


def test_explain_says_a_run_was_let_go_only_of_the_run_it_let_go(
    app: App, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A directory whose run the session let go of and that now holds another run: explain does
    not call the other run this session's."""
    monkeypatch.setattr("halley.gui.app.MAX_SESSION_RUNS", 1)
    build = build_term(app)
    first = run_example(app, build, "gone-1")["run"]
    run_example(app, build, "gone-2")
    shutil.rmtree(first)
    reference = examples.example("term-reference")
    assert reference is not None
    points, tables, scenarios = reference.make()
    run(
        app._build(build).compiled,
        model_points=points,
        tables=tables,
        scenarios=scenarios,
        settings=RunSettings(valuation_date=VALUATION_DATE, batch_size=3),
        run_dir=first,
    )
    query = {"dir": first, "p": 4, "var": "reserve", "t": 1}
    refusal = refused(app, "GET", "/api/run/explain", query)
    assert refusal.status == 409
    assert "let go" not in refusal.message


# ----------------------------------------------- one reload for all the files, imports by thread


def package(folder: Path, name: str, files: dict[str, str]) -> None:
    root = folder / name
    root.mkdir()
    for relative, text in files.items():
        (root / relative).write_text(text, encoding="utf-8")


def test_a_reload_takes_a_sibling_s_edit_through_a_package(app: App, tmp_path: Path) -> None:
    """A constants module of an assumptions package takes a value from its sibling by a relative
    import, and a namespace package (a folder without __init__.py) holds another: an edit to
    either reaches the build after a reload."""
    stem = _stem(tmp_path, "pkgs")
    package(
        tmp_path,
        f"{stem}_assumptions",
        {
            "__init__.py": "",
            "rates.py": "LOADING = 1.0\n",
            "basis.py": "from .rates import LOADING\n\nBASIS_LOADING = LOADING\n",
        },
    )
    package(tmp_path, f"{stem}_space", {"shock.py": "SHOCK = 1.0\n"})
    head = f"from {stem}_assumptions.basis import BASIS_LOADING\nfrom {stem}_space import shock\n"
    path = write_model(tmp_path, stem, head=head)
    edit(path, "* self.mp.sum_assured\n", "* self.mp.sum_assured * BASIS_LOADING * shock.SHOCK\n")
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    assert "* 1.0) * 1.0)" in death_outgo(app, model_id)
    edit(tmp_path / f"{stem}_assumptions" / "rates.py", "1.0", "1.5")
    edit(tmp_path / f"{stem}_space" / "shock.py", "1.0", "1.25")
    assert call(app, "GET", "/api/models")["changed"] is True
    call(app, "POST", "/api/models/reload")
    assert "* 1.5) * 1.25)" in death_outgo(app, model_id)
    assert call(app, "GET", "/api/models")["changed"] is False


def test_a_package_s_submodules_go_with_it(app: App, tmp_path: Path) -> None:
    """Two files use two submodules of one package through the package: a reload runs the
    package again with both, so neither file finds the package without the submodule it uses."""
    stem = _stem(tmp_path, "subs")
    files = {"__init__.py": "", "one.py": "LOADING = 1.0\n", "two.py": "SHOCK = 1.0\n"}
    package(tmp_path, f"{stem}_pkg", files)
    paths = []
    for which, name in (("a", "one.LOADING"), ("b", "two.SHOCK")):
        path = write_model(tmp_path, f"{stem}_{which}", head=f"import {stem}_pkg.{name[:3]}\n")
        edit(path, "* self.mp.sum_assured\n", f"* self.mp.sum_assured * {stem}_pkg.{name}\n")
        paths.append(path)
    ids = [call(app, "POST", "/api/models/load", path=str(p))["file"]["models"][0] for p in paths]
    edit(tmp_path / f"{stem}_pkg" / "one.py", "1.0", "1.5")
    call(app, "POST", "/api/models/reload")
    assert "* 1.5)" in death_outgo(app, ids[0])
    assert "* 1.0)" in death_outgo(app, ids[1])


def test_a_file_loaded_while_another_builds_keeps_its_own_imports(app: App, tmp_path: Path) -> None:
    """A build on the job thread and a load on another thread each record only the modules they
    imported themselves."""
    stem = _stem(tmp_path, "threads")
    for helper in ("ha", "hb"):
        (tmp_path / f"{stem}_{helper}.py").write_text("LOADING = 1.0\n", encoding="utf-8")
    a = write_model(tmp_path, f"{stem}_a", head="import time\n")
    edit(
        a,
        "        return self.deaths(t) * self.mp.sum_assured\n",
        f"        time.sleep(1.5)\n        from {stem}_ha import LOADING\n\n"
        "        return self.deaths(t) * self.mp.sum_assured * LOADING\n",
    )
    a_id = call(app, "POST", "/api/models/load", path=str(a))["file"]["models"][0]
    job = call(app, "POST", "/api/build", model=a_id)
    deadline = time.monotonic() + 10
    while app.jobs.running() is None and time.monotonic() < deadline:
        time.sleep(0.02)
    b = loading_model(tmp_path, f"{stem}_b", f"{stem}_hb")
    call(app, "POST", "/api/models/load", path=str(b))
    done(app, job)
    helpers = {f["module"]: f["helpers"] for f in call(app, "GET", "/api/models")["files"]}
    assert helpers == {f"{stem}_a": [f"{stem}_ha"], f"{stem}_b": [f"{stem}_hb"]}


def test_a_module_the_numba_backend_generated_is_not_the_user_s(app: App, tmp_path: Path) -> None:
    generated = tmp_path / "halley_numba_0123_abcd.py"
    generated.write_text("X = 1\n", encoding="utf-8")
    for name, mine in (("halley_numba_0123_abcd", False), ("mine", True)):
        module = types.ModuleType(name)
        module.__file__ = str(generated)
        assert (app.catalog._user_file(name, module) is not None) is mine


def test_a_rerun_of_settings_build_took_beyond_their_declared_types(
    app: App, tmp_path: Path
) -> None:
    """build() checks only a setting declared a plain class: a tuple of floats for a tuple of
    ints, a float for an optional int, a text a Literal does not name, and names for a tuple of
    an enum's members all build. A run made with them is rerun with them, as recorded, and the
    IR hash says they are the run's."""
    head = (
        "import enum\nfrom typing import Literal\n\n\n"
        "class Basis(enum.Enum):\n    BEST = 'best estimate'\n    PRUDENT = 'prudent'\n"
    )
    body = (
        "    counts: Static[tuple[int, ...]] = (1, 2)\n"
        "    n: Static[int | None] = None\n"
        "    level: Static[Literal['low', 'high']] = 'low'\n"
        "    bases: Static[tuple[Basis, ...]] = (Basis.BEST,)\n"
    )
    path = write_model(tmp_path, _stem(tmp_path, "loose"), head=head, body=body)
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    entry = app.catalog.get(model_id)
    assert entry is not None
    settings = {"counts": (1.0, 2.0), "n": 2.5, "level": "medium", "bases": ("PRUDENT",)}
    made = run(
        halley.build(entry.cls, **settings),
        model_points=tutorial_points(),
        tables={"mort": mort_table()},
        settings=RunSettings(valuation_date=VALUATION_DATE),
        run_dir=app.runs_root / "python-loose",
    )
    again = done(app, call(app, "POST", "/api/run/rerun", dir=str(made.run_dir)))
    assert again["same_run_id"] is True


def test_build_again_after_a_reload_carries_the_build_s_own_values(
    app: App, tmp_path: Path
) -> None:
    """A build of the file as it was carries its settings to the class as it is now: a whole
    number given for a float as it was, an enum's member by its name though its value was
    edited, and a flag's combined members; an empty flag as the default builds too."""
    head = (
        "import enum\n\n\nclass Basis(enum.Enum):\n    BEST = 'best estimate'\n"
        "    PRUDENT = 'prudent'\n\n\nclass Perm(enum.Flag):\n    R = 1\n    W = 2\n"
    )
    body = (
        "    shock: Static[float] = 1.0\n"
        "    basis: Static[Basis] = Basis.BEST\n"
        "    perm: Static[Perm] = Perm(0)\n"
    )
    path = write_model(tmp_path, _stem(tmp_path, "carry"), head=head, body=body)
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    entry = app.catalog.get(model_id)
    assert entry is not None
    m = sys.modules[entry.cls.__module__]
    given = {"shock": 2, "basis": m.Basis.PRUDENT, "perm": m.Perm.R | m.Perm.W}
    made = run(
        halley.build(entry.cls, **given),
        model_points=tutorial_points(),
        tables={"mort": mort_table()},
        settings=RunSettings(valuation_date=VALUATION_DATE),
        run_dir=app.runs_root / "python-carry",
    )
    assert done(app, call(app, "POST", "/api/run/rerun", dir=str(made.run_dir)))["same_run_id"]
    rebuilt = call(app, "GET", "/api/model", {"id": model_id})["builds"][0]
    shown = {"shock": "2", "basis": "PRUDENT", "perm": "R|W"}
    assert rebuilt["form"] == shown
    edit(path, "PRUDENT = 'prudent'", "PRUDENT = 'prudent, edited'")
    call(app, "POST", "/api/models/reload")
    entry = app.catalog.get(model_id)
    assert entry is not None
    m = sys.modules[entry.cls.__module__]
    stale = shown_build(app, model_id, rebuilt["id"])
    assert stale["stale"] is True
    assert stale["form"] == shown
    again = call(app, "POST", "/api/build", model=model_id, statics={}, from_build=rebuilt["id"])
    want = halley.build(entry.cls, shock=2, basis=m.Basis.PRUDENT, perm=m.Perm.R | m.Perm.W)
    assert ir_of(app, done(app, again)["build"]) == want.ir_hash
    defaults = done(app, call(app, "POST", "/api/build", model=model_id, statics={}))["build"]
    assert ir_of(app, defaults) == halley.build(entry.cls).ir_hash


def test_the_reload_route_waits_for_the_job_that_is_running(app: App, tmp_path: Path) -> None:
    path = write_model(tmp_path, _stem(tmp_path, "reload_route"))
    call(app, "POST", "/api/models/load", path=str(path))
    started, release = threading.Event(), threading.Event()

    def work() -> None:
        started.set()
        release.wait(30)

    waiting = app._held(None, work)  # as a run holds the engine
    job = app.jobs.submit("run", "Run a model that waits", waiting, lambda e: {"message": str(e)})
    assert started.wait(10)
    try:
        assert refused(app, "POST", "/api/models/reload").status == 409
    finally:
        release.set()
        app.jobs.wait(job.id, timeout=30)
    files = call(app, "POST", "/api/models/reload")["files"]
    assert [f["path"] for f in files] == [str(path.resolve())]


def test_a_server_on_one_address_answers_the_computer_s_name() -> None:
    assert socket.gethostname() in cli.own_names("10.0.0.5")


def test_the_overview_writes_a_setting_s_letters_as_they_are(app: App, tmp_path: Path) -> None:
    body = "    insurer: Static[str] = 'Zürich Leben'\n"
    path = write_model(tmp_path, _stem(tmp_path, "letters"), body=body)
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    entry = app.catalog.get(model_id)
    assert entry is not None
    made = run(
        halley.build(entry.cls),
        model_points=tutorial_points(),
        tables={"mort": mort_table()},
        settings=RunSettings(valuation_date=VALUATION_DATE),
        run_dir=app.runs_root / "python-letters",
    )
    call(app, "POST", "/api/runs/open", path=str(made.run_dir))
    shown = call(app, "GET", "/api/run", {"dir": str(made.run_dir)})["statics_text"]
    assert "Zürich Leben" in shown


# ------------------------------------------------ round 5: reloads that settle, settings carried


def test_a_package_with_a_namespace_subfolder_reloads(app: App, tmp_path: Path) -> None:
    """A package whose subfolder has no __init__.py: a reload takes the subfolder's namespace
    package out with the package, so the import runs afresh, with or without an edit."""
    stem = _stem(tmp_path, "nsub")
    package(tmp_path, f"{stem}_pkg", {"__init__.py": ""})
    (tmp_path / f"{stem}_pkg" / "tables").mkdir()
    rates = tmp_path / f"{stem}_pkg" / "tables" / "mortality.py"
    rates.write_text("LOADING = 1.0\n", encoding="utf-8")
    path = write_model(tmp_path, stem, head=f"import {stem}_pkg.tables.mortality\n")
    edit(
        path,
        "* self.mp.sum_assured\n",
        f"* self.mp.sum_assured * {stem}_pkg.tables.mortality.LOADING\n",
    )
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    call(app, "POST", "/api/models/reload")
    assert "* 1.0)" in death_outgo(app, model_id)
    edit(rates, "1.0", "1.5")
    call(app, "POST", "/api/models/reload")
    assert "* 1.5)" in death_outgo(app, model_id)
    assert call(app, "GET", "/api/models")["changed"] is False


def test_a_failed_load_leaves_no_submodule_on_its_package(app: App, tmp_path: Path) -> None:
    """A reload, and a first load, that fail after importing a namespace package's submodule:
    the submodule goes from the package as well as from sys.modules, so the next import of it
    runs its file as it is then."""
    stem = _stem(tmp_path, "nsfail")
    package(tmp_path, f"{stem}_ns", {"a.py": "X = 1.0\n", "b.py": "Y = 1.0\n"})
    path = write_model(tmp_path, stem, head=f"from {stem}_ns import a\n")
    edit(path, "* self.mp.sum_assured\n", "* self.mp.sum_assured * a.X\n")
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    edit(path, f"from {stem}_ns import a\n", f"from {stem}_ns import a, b\nundefined_name\n")
    assert refused(app, "POST", "/api/models/reload").status == 422
    edit(path, "undefined_name\n", "")
    edit(path, "* a.X\n", "* a.X * b.Y\n")
    edit(tmp_path / f"{stem}_ns" / "b.py", "1.0", "2.0")
    call(app, "POST", "/api/models/reload")
    assert "* 1.0) * 2.0)" in death_outgo(app, model_id)
    other = write_model(
        tmp_path, f"{stem}_first", head=f"from {stem}_ns import b\nundefined_name\n"
    )
    package(tmp_path, f"{stem}_ns2", {"c.py": "Z = 1.0\n"})
    edit(
        other, f"from {stem}_ns import b\n", f"from {stem}_ns import b\nfrom {stem}_ns2 import c\n"
    )
    assert refused(app, "POST", "/api/models/load", path=str(other)).status == 422
    assert not hasattr(sys.modules.get(f"{stem}_ns2"), "c")


def test_a_reload_with_no_edit_leaves_the_files_unchanged(app: App, tmp_path: Path) -> None:
    """Files that use one another, or a package's submodules, or a module one imports lazily
    and another at its top: after a reload with no edit, nothing has changed."""
    stem = _stem(tmp_path, "settle")
    (tmp_path / f"{stem}_h.py").write_text("LOADING = 1.0\n", encoding="utf-8")
    b = loading_model(tmp_path, f"{stem}_b", "math", extra="LOADING = 1.0\n")
    edit(b, "from math import LOADING\n", "")
    a = loading_model(tmp_path, f"{stem}_a", f"{stem}_b")
    lazy = write_model(tmp_path, f"{stem}_lazy")
    edit(
        lazy,
        "        return self.deaths(t) * self.mp.sum_assured\n",
        f"        from {stem}_h import LOADING\n\n"
        "        return self.deaths(t) * self.mp.sum_assured * LOADING\n",
    )
    top = loading_model(tmp_path, f"{stem}_top", f"{stem}_h")
    ids = {}
    for path in (a, b, lazy, top):
        ids[path.stem] = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    death_outgo(app, ids[f"{stem}_lazy"])
    assert call(app, "GET", "/api/models")["changed"] is False
    for _ in range(2):
        call(app, "POST", "/api/models/reload")
        assert call(app, "GET", "/api/models")["changed"] is False


def test_a_reload_lets_go_of_a_file_that_is_gone(app: App, tmp_path: Path) -> None:
    """A loaded file renamed or deleted since: a reload lets it go, with its models, says so,
    and reloads the others."""
    stem = _stem(tmp_path, "gone")
    draft = write_model(tmp_path, f"{stem}_draft")
    work = loading_model(tmp_path, f"{stem}_work", "math", extra="LOADING = 1.0\n")
    edit(work, "from math import LOADING\n", "")
    draft_id = call(app, "POST", "/api/models/load", path=str(draft))["file"]["models"][0]
    work_id = call(app, "POST", "/api/models/load", path=str(work))["file"]["models"][0]
    draft.rename(tmp_path / f"{stem}_kept.py")
    edit(work, "LOADING = 1.0", "LOADING = 1.5")
    answer = call(app, "POST", "/api/models/reload")
    assert answer["dropped"] == [str(draft.resolve())]
    assert app.catalog.get(draft_id) is None
    assert "* 1.5)" in death_outgo(app, work_id)


def test_a_file_saved_part_way_through_an_edit_keeps_the_pages(app: App, tmp_path: Path) -> None:
    """A loaded file saved with a syntax error: the pages still answer (Python 3.12's inspect
    parses the file), and a reload says where the file is wrong."""
    path = write_model(tmp_path, _stem(tmp_path, "midedit"))
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    edit(path, "V_M = (1 + 0.04)", "V_M = ((1 + 0.04)")
    assert call(app, "GET", "/api/models")["changed"] is True
    assert call(app, "GET", "/api/model", {"id": model_id})["id"] == model_id
    error = refused(app, "POST", "/api/models/reload")
    assert error.status == 422
    loc: Any = error.detail["loc"]
    assert Path(loc["file"]).resolve() == path.resolve()


def test_a_rerun_does_not_reload_the_files(app: App, tmp_path: Path) -> None:
    """A run whose class its loaded file no longer defines: the rerun is refused, and does not
    run the files again (which is a reload, the user's to ask for)."""
    stem = _stem(tmp_path, "renamed")
    path = write_model(tmp_path, stem)
    other = write_model(tmp_path, f"{stem}_other")
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    other_id = call(app, "POST", "/api/models/load", path=str(other))["file"]["models"][0]
    entry = app.catalog.get(model_id)
    assert entry is not None
    made = run(
        halley.build(entry.cls),
        model_points=tutorial_points(),
        tables={"mort": mort_table()},
        settings=RunSettings(valuation_date=VALUATION_DATE),
        run_dir=app.runs_root / "python-renamed",
    )
    edit(path, "class Term(Product):", "class TermV2(Product):")
    call(app, "POST", "/api/models/reload")
    built = done(app, call(app, "POST", "/api/build", model=other_id))["build"]
    failed = finish(app, call(app, "POST", "/api/run/rerun", dir=str(made.run_dir)))
    assert failed["state"] == "failed"
    assert "not a model the GUI has loaded" in failed["error"]["message"]
    assert shown_build(app, other_id, built)["stale"] is False


def test_build_again_carries_optional_tuples_and_the_old_defaults(app: App, tmp_path: Path) -> None:
    """A stale build's optional tuple of an enum's members is carried by name; its old defaults
    are read off the old class, though its string annotations no longer resolve once the enum
    is renamed; and a dict keeps its key order."""
    head = (
        "import enum\n\n\nclass Basis(enum.Enum):\n    BEST = 'best estimate'\n"
        "    PRUDENT = 'prudent'\n"
    )
    body = (
        "    many: Static[tuple[Basis, ...] | None] = None\n"
        "    loading: Static[float] = 1.0\n"
        "    table: Static[dict[str, float]] = {'b': 2.0, 'a': 1.0}\n"
    )
    path = write_model(tmp_path, _stem(tmp_path, "carry2"), head=head, body=body)
    string_annotated(path)
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    typed = {"many": '["PRUDENT"]', "table": '{"a": 1.0, "b": 2.0}'}
    built = done(app, call(app, "POST", "/api/build", model=model_id, statics=typed))["build"]
    edit(path, "class Basis(enum.Enum):", "class Kind(enum.Enum):")
    edit(path, "Static[tuple[Basis, ...] | None]", "Static[tuple[Kind, ...] | None]")
    edit(path, "loading: Static[float] = 1.0", "loading: Static[float] = 1.5")
    call(app, "POST", "/api/models/reload")
    entry = app.catalog.get(model_id)
    assert entry is not None
    kind = sys.modules[entry.cls.__module__].Kind
    stale = shown_build(app, model_id, built)
    assert stale["stale"] is True
    assert stale["form"]["many"] == '["PRUDENT"]'
    assert stale["form"]["loading"] == "1.5"
    assert stale["form"]["table"] == '{"a": 1.0, "b": 2.0}'
    again = call(app, "POST", "/api/build", model=model_id, statics={}, from_build=built)
    want = halley.build(entry.cls, many=(kind.PRUDENT,), table={"a": 1.0, "b": 2.0})
    assert ir_of(app, done(app, again)["build"]) == want.ir_hash


def test_a_check_s_imports_are_the_file_s_helpers(app: App, tmp_path: Path) -> None:
    """An input rule that imports a module of the user's when a check runs it: the module is
    one of the file's helpers, so its edit shows as a change, and a reload runs it again."""
    stem = _stem(tmp_path, "rule")
    (tmp_path / f"{stem}_limits.py").write_text("MAX_AGE = 80\n", encoding="utf-8")
    head = (
        "from halley import InputRule\n\n\n"
        f"def _young(v):\n    from {stem}_limits import MAX_AGE\n\n"
        "    return bool((v.model_points['age'] <= MAX_AGE).all())\n"
    )
    body = "    input_rules = (InputRule('young enough', _young),)\n"
    path = write_model(tmp_path, stem, head=head, body=body)
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    build = done(app, call(app, "POST", "/api/build", model=model_id))["build"]
    (tmp_path / "points.csv").write_text(
        "policy_id,age,sum_assured,premium,pols_if_init,rem_m\nA,35,1000,10,1,12\n",
        encoding="utf-8",
    )
    (tmp_path / "mort.parquet").write_bytes(b"")
    pq.write_table(mort_table(), tmp_path / "mort.parquet")
    inputs = {
        "model_points": str(tmp_path / "points.csv"),
        "tables": {"mort": str(tmp_path / "mort.parquet")},
    }
    settings = {"valuation_date": "2027-01-01"}
    done(app, call(app, "POST", "/api/inputs/check", build=build, inputs=inputs, settings=settings))
    files = call(app, "GET", "/api/models")["files"]
    assert f"{stem}_limits" in files[0]["helpers"]
    edit(tmp_path / f"{stem}_limits.py", "80", "30")
    assert call(app, "GET", "/api/models")["changed"] is True


def test_a_reload_in_another_session_marks_this_one_s_files_changed(
    app: App, tmp_path: Path
) -> None:
    """Two sessions in one process share sys.modules: a reload in one replaces the modules the
    other's models come from, which the other then shows as a change."""
    path = write_model(tmp_path, _stem(tmp_path, "twoapps"))
    call(app, "POST", "/api/models/load", path=str(path))
    other = App(tmp_path / "other-runs")
    try:
        call(other, "POST", "/api/models/load", path=str(path))
        call(other, "POST", "/api/models/reload")
        assert call(app, "GET", "/api/models")["changed"] is True
    finally:
        other.close()


def test_a_run_with_worker_processes_needs_the_class_its_module_holds(
    app: App, tmp_path: Path
) -> None:
    path = write_model(tmp_path, _stem(tmp_path, "pickle"))
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    build = done(app, call(app, "POST", "/api/build", model=model_id))["build"]
    call(app, "POST", "/api/models/reload")
    settings = {"valuation_date": "2027-01-01", "workers": 2}
    refusal = refused(
        app, "POST", "/api/run", build=build, inputs={"example": "x"}, settings=settings, name="w"
    )
    assert refusal.status in (404, 409)


def test_a_dict_setting_out_of_sorted_order_is_warned_of(app: App, tmp_path: Path) -> None:
    """A dict a build is given whose keys are not sorted: its run's manifest records them
    sorted, so a rerun from the directory cannot rebuild it, which the build warns of."""
    body = "    table: Static[dict[str, float]] = {'b': 2.0, 'a': 1.0}\n"
    path = write_model(tmp_path, _stem(tmp_path, "dictwarn"), body=body)
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]

    def warned(statics: dict[str, str]) -> bool:
        made = done(app, call(app, "POST", "/api/build", model=model_id, statics=statics))
        return any("not in sorted order" in w for w in app._build(made["build"]).warnings)

    # the default reruns whatever its order, a rerun trying the class's default first
    assert warned({}) is False
    assert warned({"table": '{"c": 3.0, "a": 1.0}'}) is True
    assert warned({"table": '{"a": 1.0, "c": 3.0}'}) is False


def test_the_form_s_json_is_a_tuple_where_the_type_is_one(app: App, tmp_path: Path) -> None:
    """A tuple setting typed with items its type does not declare (floats for a tuple of ints)
    is built as the tuple a rerun reads back, so the GUI's own run reruns in a later session;
    a text a tuple of dates was given reruns as that text; letters stay letters."""
    body = (
        "    counts: Static[tuple[int, ...]] = (1, 2)\n"
        "    dates: Static[tuple[dt.date, ...]] = (dt.date(2027, 1, 1),)\n"
        "    cities: Static[tuple[str, ...]] = ('Zürich', 'Genève')\n"
    )
    path = write_model(
        tmp_path, _stem(tmp_path, "loosejson"), head="import datetime as dt\n", body=body
    )
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    params = {p["name"]: p for p in call(app, "GET", "/api/model", {"id": model_id})["statics"]}
    assert params["cities"]["text"] == '["Zürich", "Genève"]'
    entry = app.catalog.get(model_id)
    assert entry is not None
    built = call(app, "POST", "/api/build", model=model_id, statics={"counts": "[1.0, 2.0]"})
    assert (
        ir_of(app, done(app, built)["build"]) == halley.build(entry.cls, counts=(1.0, 2.0)).ir_hash
    )
    made = run(
        halley.build(entry.cls, dates=("2028-01-01",)),
        model_points=tutorial_points(),
        tables={"mort": mort_table()},
        settings=RunSettings(valuation_date=VALUATION_DATE),
        run_dir=app.runs_root / "python-isodate",
    )
    other = App(tmp_path / "other-runs")
    try:
        again = done(other, call(other, "POST", "/api/run/rerun", dir=str(made.run_dir)))
        assert again["same_run_id"] is True
    finally:
        other.close()


def test_a_flag_setting_takes_its_members_joined(app: App, tmp_path: Path) -> None:
    head = "import enum\n\n\nclass Perm(enum.Flag):\n    R = 1\n    W = 2\n    X = 4\n"
    body = "    perm: Static[Perm] = Perm.R | Perm.W\n"
    path = write_model(tmp_path, _stem(tmp_path, "flags"), head=head, body=body)
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    param = call(app, "GET", "/api/model", {"id": model_id})["statics"][0]
    assert (param["input"], param["text"], param["choices"]) == ("flags", "R|W", ["R", "W", "X"])
    entry = app.catalog.get(model_id)
    assert entry is not None
    perm = sys.modules[entry.cls.__module__].Perm
    built = call(app, "POST", "/api/build", model=model_id, statics={"perm": "W|X"})
    assert (
        ir_of(app, done(app, built)["build"])
        == halley.build(entry.cls, perm=perm.W | perm.X).ir_hash
    )
    assert refused(app, "POST", "/api/build", model=model_id, statics={"perm": "R|Q"}).status == 400


# ------------------------------------------------------- round 6: failures that clean up, stamps


def test_a_failed_load_of_a_package_with_a_namespace_subfolder_cleans_up(
    app: App, tmp_path: Path
) -> None:
    """A first load, and a reload, that fail after importing a new package with a namespace
    subfolder take the subfolder out too, so the load works once the mistake is fixed."""
    for what in ("first", "again"):
        stem = _stem(tmp_path, f"nsfail_{what}")
        package(tmp_path, f"{stem}_pkg", {"__init__.py": ""})
        (tmp_path / f"{stem}_pkg" / "tables").mkdir()
        (tmp_path / f"{stem}_pkg" / "tables" / "mortality.py").write_text(
            "LOADING = 1.0\n", encoding="utf-8"
        )
        head = f"from {stem}_pkg.tables.mortality import LOADING\nundefined_name\n"
        path = write_model(tmp_path, stem, head="" if what == "again" else head)
        if what == "again":
            call(app, "POST", "/api/models/load", path=str(path))
            edit(path, '"""A model file', f'{head}"""A model file')
            assert refused(app, "POST", "/api/models/reload").status == 422
        else:
            assert refused(app, "POST", "/api/models/load", path=str(path)).status == 422
        assert f"{stem}_pkg.tables" not in sys.modules
        edit(path, "undefined_name\n", "")
        if what == "again":
            call(app, "POST", "/api/models/reload")
        else:
            call(app, "POST", "/api/models/load", path=str(path))
        assert f"{stem}_pkg.tables.mortality" in sys.modules


def test_a_file_the_gui_runs_itself_is_stamped_when_its_run_begins(
    app: App, tmp_path: Path
) -> None:
    """A model file whose import the watch once saw (a build's formula imported it), then run
    by a reload itself: its own run's start counts, so after a reload with no edit since,
    nothing has changed."""
    stem = _stem(tmp_path, "stamp")
    b = loading_model(tmp_path, f"{stem}_b", "math", extra="LOADING = 1.0\n")
    edit(b, "from math import LOADING\n", "")
    a = write_model(tmp_path, f"{stem}_a")
    edit(
        a,
        "        return self.deaths(t) * self.mp.sum_assured\n",
        f"        from {stem}_b import LOADING\n\n"
        "        return self.deaths(t) * self.mp.sum_assured * LOADING\n",
    )
    a_id = call(app, "POST", "/api/models/load", path=str(a))["file"]["models"][0]
    death_outgo(app, a_id)
    call(app, "POST", "/api/models/load", path=str(b))
    edit(b, "LOADING = 1.0", "LOADING = 1.5")
    assert call(app, "GET", "/api/models")["changed"] is True
    for _ in range(2):
        call(app, "POST", "/api/models/reload")
        assert call(app, "GET", "/api/models")["changed"] is False
    assert "* 1.5)" in death_outgo(app, a_id)


def test_a_model_file_saved_while_it_runs_shows_as_changed(app: App, tmp_path: Path) -> None:
    """A file saved after its run began is not what ran: it shows as changed."""
    stem = _stem(tmp_path, "saved")
    path = write_model(tmp_path, stem)
    edit(path, "V_M = ", f"import os\nos.utime(r'{path}')\nV_M = ")
    call(app, "POST", "/api/models/load", path=str(path))
    assert call(app, "GET", "/api/models")["changed"] is True


def test_a_class_a_function_made_is_left_to_run(app: App, tmp_path: Path) -> None:
    """The workers guard refuses only a class its module has replaced since the build; a class
    a function made, which no name of the module reaches, is run() 's to judge."""
    stem = _stem(tmp_path, "factory")
    path = write_model(tmp_path, stem)
    edit(path, "class Term(Product):", "def _make():\n  class Term(Product):")
    source = path.read_text(encoding="utf-8")
    head, _, rest = source.partition("def _make():\n  class Term(Product):")
    body = "\n".join("  " + line if line else line for line in rest.splitlines())
    path.write_text(
        f"{head}def _make():\n  class Term(Product):{body}\n  return Term\n\n\nTerm = _make()\n",
        encoding="utf-8",
    )
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    build = done(app, call(app, "POST", "/api/build", model=model_id))["build"]
    inputs = {
        "model_points": str(tmp_path / "points.parquet"),
        "tables": {"mort": str(tmp_path / "mort.parquet")},
    }
    pq.write_table(tutorial_points(), tmp_path / "points.parquet")
    pq.write_table(mort_table(), tmp_path / "mort.parquet")
    settings = {"valuation_date": "2027-01-01", "workers": 2}
    made = done(
        app,
        call(
            app,
            "POST",
            "/api/run",
            build=build,
            inputs=inputs,
            settings=settings,
            name="factory-run",
        ),
    )
    assert made["run_id"]


def test_build_again_of_a_default_dict_is_not_warned_of(app: App, tmp_path: Path) -> None:
    body = "    table: Static[dict[str, float]] = {'c': 3.0, 'a': 1.0}\n"
    path = write_model(tmp_path, _stem(tmp_path, "dictagain"), body=body)
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    first = done(app, call(app, "POST", "/api/build", model=model_id))["build"]
    again = done(app, call(app, "POST", "/api/build", model=model_id, statics={}, from_build=first))
    assert not any("not in sorted order" in w for w in app._build(again["build"]).warnings)


def test_a_second_reload_waits_for_the_first(app: App, tmp_path: Path) -> None:
    """Two reloads at once (the side list's control and a model page's banner, say): the second
    waits for the first rather than being refused as if a job ran."""
    stem = _stem(tmp_path, "tworeloads")
    flag = tmp_path / f"{stem}.slow"
    path = write_model(
        tmp_path,
        stem,
        head=(
            f"import time\nfrom pathlib import Path\n"
            f"if Path(r'{flag}').exists():\n    time.sleep(1.5)\n"
        ),
    )
    call(app, "POST", "/api/models/load", path=str(path))
    flag.write_text("", encoding="utf-8")
    first: list[object] = []
    worker = threading.Thread(target=lambda: first.append(call(app, "POST", "/api/models/reload")))
    worker.start()
    time.sleep(0.3)
    second = call(app, "POST", "/api/models/reload")
    worker.join(30)
    assert first
    assert second["files"]


def test_loading_a_loaded_file_that_is_gone_says_so(app: App, tmp_path: Path) -> None:
    stem = _stem(tmp_path, "gone_load")
    path = write_model(tmp_path, stem)
    call(app, "POST", "/api/models/load", path=str(path))
    kept = tmp_path / f"{stem}_kept.py"
    path.rename(kept)
    refusal = refused(app, "POST", "/api/models/load", path=str(path))
    assert refusal.status == 404
    # the page leaves the file the error names out of its note of the files let go
    assert Path(str(refusal.detail["file"])).name == path.name
    assert refusal.detail["dropped"] == [refusal.detail["file"]]


def test_dict_settings_read_their_values_by_type(app: App, tmp_path: Path) -> None:
    """A dict's values are read by its value type, in the form's JSON and from a manifest: a
    tuple, an enum's member by its name, a date in ISO form."""
    head = (
        "import datetime as dt\nimport enum\n\n\n"
        "class Basis(enum.Enum):\n    BEST = 'best estimate'\n    PRUDENT = 'prudent'\n"
    )
    body = (
        "    legs: Static[dict[str, tuple[float, ...]]] = {'a': (1.0, 0.5)}\n"
        "    by_basis: Static[dict[str, Basis]] = {'x': Basis.BEST}\n"
        "    by_date: Static[dict[str, dt.date]] = {'x': dt.date(2027, 1, 1)}\n"
    )
    path = write_model(tmp_path, _stem(tmp_path, "dictvals"), head=head, body=body)
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    entry = app.catalog.get(model_id)
    assert entry is not None
    m = sys.modules[entry.cls.__module__]
    typed = {
        "legs": '{"a": [2.0, 0.5]}',
        "by_basis": '{"x": "PRUDENT"}',
        "by_date": '{"x": "2028-01-01"}',
    }
    values = {
        "legs": {"a": (2.0, 0.5)},
        "by_basis": {"x": m.Basis.PRUDENT},
        "by_date": {"x": dt.date(2028, 1, 1)},
    }
    built = done(app, call(app, "POST", "/api/build", model=model_id, statics=typed))["build"]
    expected = halley.build(entry.cls, **values)
    assert ir_of(app, built) == expected.ir_hash
    made = run(
        expected,
        model_points=tutorial_points(),
        tables={"mort": mort_table()},
        settings=RunSettings(valuation_date=VALUATION_DATE),
        run_dir=app.runs_root / "python-dictvals",
    )
    other = App(tmp_path / "other-runs")
    try:
        again = done(other, call(other, "POST", "/api/run/rerun", dir=str(made.run_dir)))
        assert again["same_run_id"] is True
    finally:
        other.close()


def test_a_load_during_a_build_leaves_nothing_changed(app: App, tmp_path: Path) -> None:
    """A build whose formula imports a helper, and a file loaded meanwhile that imports it at
    its top: once the build has recorded its import, nothing reads as changed."""
    stem = _stem(tmp_path, "midbuild")
    (tmp_path / f"{stem}_h.py").write_text("LOADING = 1.0\n", encoding="utf-8")
    a = write_model(tmp_path, f"{stem}_a", head="import time\n")
    edit(
        a,
        "        return self.deaths(t) * self.mp.sum_assured\n",
        f"        from {stem}_h import LOADING\n\n        time.sleep(1.5)\n"
        "        return self.deaths(t) * self.mp.sum_assured * LOADING\n",
    )
    a_id = call(app, "POST", "/api/models/load", path=str(a))["file"]["models"][0]
    job = call(app, "POST", "/api/build", model=a_id)
    deadline = time.monotonic() + 10
    while f"{stem}_h" not in sys.modules and time.monotonic() < deadline:
        time.sleep(0.02)
    b = loading_model(tmp_path, f"{stem}_b", f"{stem}_h")
    call(app, "POST", "/api/models/load", path=str(b))
    done(app, job)
    assert call(app, "GET", "/api/models")["changed"] is False


# ------------------------------------------------------ round 7: threads, clocks, dicts' keys


def test_a_failed_load_during_a_build_leaves_the_build_s_imports(app: App, tmp_path: Path) -> None:
    """A first load takes no engine, so a build may be importing a helper on its own thread when
    another file fails to load: the failure takes out only what its own import brought in, so
    the build ends well, and its record of the helper shows an edit to it."""
    stem = _stem(tmp_path, "failmid")
    flag = tmp_path / f"{stem}.loading"
    helper = tmp_path / f"{stem}_h.py"
    helper.write_text("import time\ntime.sleep(1.0)\nLOADING = 1.0\n", encoding="utf-8")
    a = write_model(tmp_path, f"{stem}_a", head="import time\nfrom pathlib import Path\n")
    edit(
        a,
        "        return self.deaths(t) * self.mp.sum_assured\n",
        f"        deadline = time.monotonic() + 10\n"
        f"        while not Path(r'{flag}').exists() and time.monotonic() < deadline:\n"
        f"            time.sleep(0.01)\n"
        f"        from {stem}_h import LOADING\n\n"
        "        return self.deaths(t) * self.mp.sum_assured * LOADING\n",
    )
    a_id = call(app, "POST", "/api/models/load", path=str(a))["file"]["models"][0]
    job = call(app, "POST", "/api/build", model=a_id)
    # the other file, once its load has begun, lets the build import the helper, and fails
    # while that import runs
    c = write_model(
        tmp_path,
        f"{stem}_c",
        head=(
            f"import sys\nimport time\nfrom pathlib import Path\n\nPath(r'{flag}').write_text('')\n"
            f"deadline = time.monotonic() + 10\n"
            f"while '{stem}_h' not in sys.modules and time.monotonic() < deadline:\n"
            f"    time.sleep(0.01)\n"
            f"time.sleep(0.2)\nundefined_name\n"
        ),
    )
    assert refused(app, "POST", "/api/models/load", path=str(c)).status == 422
    assert done(app, job)["build"]
    assert f"{stem}_h" in sys.modules
    assert call(app, "GET", "/api/models")["changed"] is False
    edit(helper, "LOADING = 1.0", "LOADING = 2.0")
    assert call(app, "GET", "/api/models")["changed"] is True


def test_a_model_file_loaded_while_a_build_imports_it_is_taken_whole(
    app: App, tmp_path: Path
) -> None:
    """A file a build's formula is importing when it is loaded: the load waits for that import,
    as Python's own import would, and takes its models; once the build is done, nothing reads
    as changed."""
    stem = _stem(tmp_path, "adoptmid")
    b = write_model(tmp_path, f"{stem}_b", head="import time\n\ntime.sleep(1.0)\nLOADING = 1.0\n")
    a = write_model(tmp_path, f"{stem}_a")
    edit(
        a,
        "        return self.deaths(t) * self.mp.sum_assured\n",
        f"        from {stem}_b import LOADING\n\n"
        "        return self.deaths(t) * self.mp.sum_assured * LOADING\n",
    )
    a_id = call(app, "POST", "/api/models/load", path=str(a))["file"]["models"][0]
    job = call(app, "POST", "/api/build", model=a_id)
    deadline = time.monotonic() + 10
    while f"{stem}_b" not in sys.modules and time.monotonic() < deadline:
        time.sleep(0.01)
    loaded = call(app, "POST", "/api/models/load", path=str(b))
    assert [m["name"] for m in loaded["models"]] == ["Term"]
    done(app, job)
    assert call(app, "GET", "/api/models")["changed"] is False


def test_a_file_loaded_while_a_build_runs_reads_unchanged_meanwhile(
    app: App, tmp_path: Path
) -> None:
    """A file whose top imports a helper a running build's formula imported: the file's record
    of the helper is the build's import, not one of unknown age, so nothing reads as changed
    while the build still runs."""
    stem = _stem(tmp_path, "midrun")
    (tmp_path / f"{stem}_h.py").write_text("LOADING = 1.0\n", encoding="utf-8")
    a = write_model(tmp_path, f"{stem}_a", head="import time\n")
    edit(
        a,
        "        return self.deaths(t) * self.mp.sum_assured\n",
        f"        from {stem}_h import LOADING\n\n        time.sleep(2.0)\n"
        "        return self.deaths(t) * self.mp.sum_assured * LOADING\n",
    )
    a_id = call(app, "POST", "/api/models/load", path=str(a))["file"]["models"][0]
    job = call(app, "POST", "/api/build", model=a_id)
    deadline = time.monotonic() + 10
    while f"{stem}_h" not in sys.modules and time.monotonic() < deadline:
        time.sleep(0.01)
    b = loading_model(tmp_path, f"{stem}_b", f"{stem}_h")
    call(app, "POST", "/api/models/load", path=str(b))
    assert app.jobs.running() is not None
    assert call(app, "GET", "/api/models")["changed"] is False
    done(app, job)
    assert call(app, "GET", "/api/models")["changed"] is False


def test_a_model_file_dated_ahead_of_the_clock_is_not_taken_as_changed(
    app: App, tmp_path: Path
) -> None:
    """A file whose modification time is ahead of this computer's clock (unpacked from an
    archive made where the clock reads later, say) was not saved while it ran: after its load
    and a reload nothing reads as changed, and an edit still does."""
    stem = _stem(tmp_path, "ahead")
    helper = tmp_path / f"{stem}_h.py"
    helper.write_text("LOADING = 1.0\n", encoding="utf-8")
    path = loading_model(tmp_path, stem, f"{stem}_h")
    later = time.time() + 3600
    for file in (path, helper):
        os.utime(file, (later, later))
    call(app, "POST", "/api/models/load", path=str(path))
    assert call(app, "GET", "/api/models")["changed"] is False
    call(app, "POST", "/api/models/reload")
    assert call(app, "GET", "/api/models")["changed"] is False
    edit(helper, "LOADING = 1.0", "LOADING = 2.0")
    assert call(app, "GET", "/api/models")["changed"] is True


def test_a_helper_another_session_imports_again_still_shows_an_edit(
    app: App, tmp_path: Path
) -> None:
    """Two sessions in one process whose files share a helper: one's reload imports the helper
    again, which leaves the other's models with the helper as it was; an edit after shows in the
    other session too."""
    stem = _stem(tmp_path, "sharedhelper")
    helper = tmp_path / f"{stem}_h.py"
    helper.write_text("LOADING = 1.0\n", encoding="utf-8")
    a = loading_model(tmp_path, f"{stem}_a", f"{stem}_h")
    b = loading_model(tmp_path, f"{stem}_b", f"{stem}_h")
    call(app, "POST", "/api/models/load", path=str(a))
    other = App(tmp_path / "other-runs")
    try:
        call(other, "POST", "/api/models/load", path=str(b))
        call(other, "POST", "/api/models/reload")
        assert call(app, "GET", "/api/models")["changed"] is False
        edit(helper, "LOADING = 1.0", "LOADING = 2.0")
        assert call(app, "GET", "/api/models")["changed"] is True
    finally:
        other.close()


def test_a_folder_made_a_package_is_imported_as_one_at_a_reload(app: App, tmp_path: Path) -> None:
    """A namespace package at the top of the user's folder (no __init__.py anywhere) is taken out
    at a reload with the modules under it, so once the folder holds an __init__.py the reload
    imports it as the package a new Python process would."""
    stem = _stem(tmp_path, "nsroot")
    folder = tmp_path / f"{stem}_ns"
    (folder / "sub").mkdir(parents=True)
    (folder / "sub" / "mortality.py").write_text("LOADING = 1.0\n", encoding="utf-8")
    path = write_model(tmp_path, stem, head=f"from {stem}_ns.sub.mortality import LOADING\n")
    call(app, "POST", "/api/models/load", path=str(path))
    (folder / "__init__.py").write_text("SHOCK = 2.0\n", encoding="utf-8")
    edit(path, f"from {stem}_ns.sub.mortality import LOADING\n", f"from {stem}_ns import SHOCK\n")
    call(app, "POST", "/api/models/reload")
    assert sys.modules[f"{stem}_ns"].SHOCK == 2.0


def test_a_helper_from_a_zip_archive_is_not_taken_as_changed(app: App, tmp_path: Path) -> None:
    """A module imported from a zip archive on sys.path is an installed package's to the GUI: its
    file is a member of the archive, which has no time of its own to compare."""
    stem = _stem(tmp_path, "zipped")
    archive = tmp_path / f"{stem}.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(f"{stem}_h.py", "LOADING = 1.0\n")
    sys.path.append(str(archive))
    try:
        path = loading_model(tmp_path, stem, f"{stem}_h")
        call(app, "POST", "/api/models/load", path=str(path))
        assert call(app, "GET", "/api/models")["changed"] is False
        call(app, "POST", "/api/models/reload")
        assert call(app, "GET", "/api/models")["changed"] is False
    finally:
        sys.path.remove(str(archive))


def run_and_rerun_elsewhere(app: App, tmp_path: Path, build: str, name: str) -> dict[str, Any]:
    """A run of a build on the tutorial's points, then its rerun in a new session."""
    pq.write_table(tutorial_points(), tmp_path / "points.parquet")
    pq.write_table(mort_table(), tmp_path / "mort.parquet")
    inputs = {
        "model_points": str(tmp_path / "points.parquet"),
        "tables": {"mort": str(tmp_path / "mort.parquet")},
    }
    settings = {"valuation_date": "2027-01-01"}
    made = done(
        app, call(app, "POST", "/api/run", build=build, inputs=inputs, settings=settings, name=name)
    )
    other = App(tmp_path / f"{name}-other-runs")
    try:
        return done(other, call(other, "POST", "/api/run/rerun", dir=made["run"]))
    finally:
        other.close()


def test_the_rerun_warning_follows_what_a_rerun_tries(app: App, tmp_path: Path) -> None:
    """A build warns of a setting a rerun from its run's directory would not give back, and of
    no other: a dict without a default typed unsorted is warned of; two dicts whose defaults are
    unsorted, one left at its default and one typed sorted, are not, and their run reruns in a
    new session, a rerun trying each such dict both ways."""
    body = (
        "    need: Static[dict[str, float]]\n"
        "    x: Static[dict[str, float]] = {'m': 1.0, 'f': 2.0}\n"
        "    y: Static[dict[str, float]] = {'m': 1.0, 'f': 2.0}\n"
    )
    path = write_model(tmp_path, _stem(tmp_path, "dictmix"), body=body)
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]

    def built(statics: dict[str, str]) -> tuple[str, list[str]]:
        made = done(app, call(app, "POST", "/api/build", model=model_id, statics=statics))
        return made["build"], app._build(made["build"]).warnings

    _first, warned = built({"need": '{"b": 2.0, "a": 1.0}'})
    assert [w for w in warned if "need" in w and "not in sorted order" in w]
    second, warned = built({"need": '{"a": 1.0, "b": 2.0}', "y": '{"f": 2.0, "m": 1.0}'})
    assert not [w for w in warned if "rerun" in w]
    again = run_and_rerun_elsewhere(app, tmp_path, second, "dictmix")
    assert again["same_run_id"] is True


def test_rerun_misses_names_what_a_manifest_cannot_record_exactly(app: App, tmp_path: Path) -> None:
    """A NumPy number, which a manifest records as a plain one, and a dict in neither sorted order
    nor its default's; the default itself, in its own order, and a sorted dict are given back."""
    body = (
        "    shock: Static[float] = 1.0\n"
        "    table: Static[dict[str, float]] = {'c': 3.0, 'a': 1.0}\n"
    )
    path = write_model(tmp_path, _stem(tmp_path, "misses"), body=body)
    entry = app.catalog.get(app.catalog.load_file(path).models[0])
    assert entry is not None
    cls = entry.cls
    table = {"c": 3.0, "a": 1.0}
    assert rerun_misses(cls, {"shock": 1.0, "table": table}).names == []
    assert rerun_misses(cls, {"shock": 1.0, "table": {"a": 1.0, "c": 3.0}}).names == []
    assert rerun_misses(cls, {"shock": np.float64(1.5), "table": table}).names == ["shock"]
    unsorted = {"c": 3.0, "b": 2.0, "a": 1.0}
    assert rerun_misses(cls, {"shock": 1.0, "table": unsorted}).names == ["table"]


def test_dict_keys_are_read_by_their_type(app: App, tmp_path: Path) -> None:
    """A dict's keys are read by the dict's key type, as its values are: in the form, an enum's
    member by its name, a date in ISO form and a number as JSON writes it; from a manifest, as it
    records them, so a Python run with such keys reruns in a new session."""
    head = (
        "import datetime as dt\nimport enum\n\n\n"
        "class Basis(enum.Enum):\n    BEST = 'best estimate'\n    PRUDENT = 'prudent'\n"
    )
    body = (
        "    by_basis: Static[dict[Basis, float]] = {Basis.BEST: 1.0}\n"
        "    by_date: Static[dict[dt.date, float]] = {dt.date(2027, 1, 1): 1.0}\n"
        "    by_age: Static[dict[int, float]] = {40: 1.0}\n"
    )
    path = write_model(tmp_path, _stem(tmp_path, "dictkeys"), head=head, body=body)
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    entry = app.catalog.get(model_id)
    assert entry is not None
    m = sys.modules[entry.cls.__module__]
    shown = {
        p["name"]: p["text"] for p in call(app, "GET", "/api/model", {"id": model_id})["statics"]
    }
    assert shown == {
        "by_basis": '{"BEST": 1.0}',
        "by_date": '{"2027-01-01": 1.0}',
        "by_age": '{"40": 1.0}',
    }
    typed = {
        "by_basis": '{"BEST": 1.0, "PRUDENT": 2.0}',
        "by_date": '{"2028-01-01": 2.0}',
        "by_age": '{"41": 2.0}',
    }
    values = {
        "by_basis": {m.Basis.BEST: 1.0, m.Basis.PRUDENT: 2.0},
        "by_date": {dt.date(2028, 1, 1): 2.0},
        "by_age": {41: 2.0},
    }
    built = done(app, call(app, "POST", "/api/build", model=model_id, statics=typed))["build"]
    expected = halley.build(entry.cls, **values)
    assert ir_of(app, built) == expected.ir_hash
    made = run(
        expected,
        model_points=tutorial_points(),
        tables={"mort": mort_table()},
        settings=RunSettings(valuation_date=VALUATION_DATE),
        run_dir=app.runs_root / "python-dictkeys",
    )
    other = App(tmp_path / "other-runs")
    try:
        again = done(other, call(other, "POST", "/api/run/rerun", dir=str(made.run_dir)))
        assert again["same_run_id"] is True
    finally:
        other.close()


def test_a_run_with_values_their_type_does_not_take_reruns(app: App, tmp_path: Path) -> None:
    """A Python run whose settings hold, beside values that fit their types, values that do not
    (a text in a dict of tuples, a tuple of ints where floats are declared, inside a tuple):
    each is read as far as it fits, a list inside a tuple as a tuple, so the run reruns."""
    body = (
        "    legs: Static[dict[str, tuple[float, ...]]] = {'a': (1.0, 0.5)}\n"
        "    nested: Static[tuple[tuple[float, float], ...]] = ((1.0, 2.0),)\n"
    )
    path = write_model(tmp_path, _stem(tmp_path, "misfits"), body=body)
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    entry = app.catalog.get(model_id)
    assert entry is not None
    values = {"legs": {"a": (2.0, 0.5), "b": "flat"}, "nested": ((1.0, (2, 3)),)}
    made = run(
        halley.build(entry.cls, **values),
        model_points=tutorial_points(),
        tables={"mort": mort_table()},
        settings=RunSettings(valuation_date=VALUATION_DATE),
        run_dir=app.runs_root / "python-misfits",
    )
    other = App(tmp_path / "other-runs")
    try:
        again = done(other, call(other, "POST", "/api/run/rerun", dir=str(made.run_dir)))
        assert again["same_run_id"] is True
    finally:
        other.close()


# ------------------------------------------- round 8: packages under a race, keys, rerun tries


@pytest.mark.parametrize("regular", [True, False], ids=["package", "namespace"])
def test_a_failed_load_keeps_the_package_a_build_imports_from(
    app: App, tmp_path: Path, regular: bool
) -> None:
    """A build importing a module of a package that a file failing to load meanwhile imported
    first: the clean-up leaves the package, which the build's import looks up when it ends."""
    stem = _stem(tmp_path, f"pkgrace_{regular}")
    flag = tmp_path / f"{stem}.loading"
    folder = tmp_path / f"{stem}_pkg"
    folder.mkdir()
    if regular:
        (folder / "__init__.py").write_text("", encoding="utf-8")
    (folder / "h.py").write_text("LOADING = 1.0\n", encoding="utf-8")
    (folder / "k.py").write_text("import time\ntime.sleep(1.0)\nLOADING = 1.0\n", encoding="utf-8")
    a = write_model(tmp_path, f"{stem}_a", head="import time\nfrom pathlib import Path\n")
    edit(
        a,
        "        return self.deaths(t) * self.mp.sum_assured\n",
        f"        deadline = time.monotonic() + 10\n"
        f"        while not Path(r'{flag}').exists() and time.monotonic() < deadline:\n"
        f"            time.sleep(0.01)\n"
        f"        from {stem}_pkg.k import LOADING\n\n"
        "        return self.deaths(t) * self.mp.sum_assured * LOADING\n",
    )
    a_id = call(app, "POST", "/api/models/load", path=str(a))["file"]["models"][0]
    job = call(app, "POST", "/api/build", model=a_id)
    c = write_model(
        tmp_path,
        f"{stem}_c",
        head=(
            f"import sys\nimport time\nfrom pathlib import Path\n\nimport {stem}_pkg.h\n\n"
            f"Path(r'{flag}').write_text('')\n"
            f"deadline = time.monotonic() + 10\n"
            f"while '{stem}_pkg.k' not in sys.modules and time.monotonic() < deadline:\n"
            f"    time.sleep(0.01)\n"
            f"time.sleep(0.2)\nundefined_name\n"
        ),
    )
    assert refused(app, "POST", "/api/models/load", path=str(c)).status == 422
    assert done(app, job)["build"]
    # the package stays whole, with what the failing load imported under it
    assert f"{stem}_pkg" in sys.modules
    assert f"{stem}_pkg.h" in sys.modules
    call(app, "POST", "/api/models/reload")


def test_a_failed_load_leaves_no_namespace_subfolder_without_its_parent(
    app: App, tmp_path: Path
) -> None:
    """A load, and a reload, that fail inside a namespace subfolder of a namespace package take
    out the whole tree they imported, so a file importing the subfolder once the mistake is
    fixed loads."""
    for what in ("load", "reload"):
        stem = _stem(tmp_path, f"nstree_{what}")
        folder = tmp_path / f"{stem}_ns"
        (folder / "mortality").mkdir(parents=True)
        (folder / "loadings.py").write_text("LOADING = 1.0\n", encoding="utf-8")
        tables = folder / "mortality" / "tables.py"
        tables.write_text("SHOCK = 1.0\nundefined_name\n", encoding="utf-8")
        head = f"import {stem}_ns.loadings\nimport {stem}_ns.mortality.tables\n"
        c = write_model(tmp_path, f"{stem}_c", head="" if what == "reload" else head)
        if what == "reload":
            call(app, "POST", "/api/models/load", path=str(c))
            edit(c, '"""A model file', f'{head}"""A model file')
            assert refused(app, "POST", "/api/models/reload").status == 422
            edit(c, head, "")
        else:
            assert refused(app, "POST", "/api/models/load", path=str(c)).status == 422
        assert f"{stem}_ns.mortality" not in sys.modules
        edit(tables, "undefined_name\n", "")
        d = write_model(
            tmp_path, f"{stem}_d", head=f"from {stem}_ns.mortality.tables import SHOCK\n"
        )
        assert call(app, "POST", "/api/models/load", path=str(d))["models"]


def test_a_namespace_package_a_file_reads_is_taken_out_at_a_reload(
    app: App, tmp_path: Path
) -> None:
    """A folder without an __init__.py that a file imports itself (to read its data, say), with
    no module under it: the folder given an __init__.py shows as a change, and the reload
    imports it as the package a new Python process would."""
    stem = _stem(tmp_path, "nsdata")
    folder = tmp_path / f"{stem}_tables"
    folder.mkdir()
    (folder / "rates.csv").write_text("age,rate\n40,0.001\n", encoding="utf-8")
    path = write_model(
        tmp_path,
        stem,
        head=f"import importlib.resources\n\nDATA = importlib.resources.files('{stem}_tables')\n",
    )
    call(app, "POST", "/api/models/load", path=str(path))
    assert call(app, "GET", "/api/models")["changed"] is False
    (folder / "__init__.py").write_text("SHOCK = 2.0\n", encoding="utf-8")
    assert call(app, "GET", "/api/models")["changed"] is True
    edit(path, "\nDATA = ", f"\nfrom {stem}_tables import SHOCK\n\nDATA = ")
    call(app, "POST", "/api/models/reload")
    assert sys.modules[f"{stem}_tables"].SHOCK == 2.0
    assert call(app, "GET", "/api/models")["changed"] is False


def test_a_helper_dated_a_little_ahead_of_the_clock_is_not_taken_as_saved(
    app: App, tmp_path: Path
) -> None:
    """A helper dated two seconds ahead of the clock, imported at the top of a file whose top
    level then takes longer than that: its time as its import began is its time after, so it
    was not saved while it ran."""
    stem = _stem(tmp_path, "aheadhelper")
    helper = tmp_path / f"{stem}_h.py"
    helper.write_text("LOADING = 1.0\n", encoding="utf-8")
    later = time.time() + 2
    os.utime(helper, (later, later))
    path = loading_model(tmp_path, stem, f"{stem}_h", extra="import time\ntime.sleep(3)\n")
    call(app, "POST", "/api/models/load", path=str(path))
    assert call(app, "GET", "/api/models")["changed"] is False


def test_tuple_and_empty_flag_keys_read_back_from_a_manifest(app: App, tmp_path: Path) -> None:
    """Keys a manifest records as str() of tuples holding enum members and dates (a rate table
    keyed by sex and age, say), and of a flag's empty value: a Python run with them reruns in a
    new session, and the form's own text builds the same model."""
    head = (
        "import datetime as dt\nimport enum\n\n\n"
        "class Sex(enum.Enum):\n    F = 'female'\n    M = 'male'\n\n\n"
        "class Perm(enum.Flag):\n    R = 1\n    W = 2\n"
    )
    body = (
        "    by_sex_age: Static[dict[tuple[Sex, int], float]] = {(Sex.F, 40): 1.0}\n"
        "    by_day: Static[dict[tuple[dt.date, int], float]] = {(dt.date(2027, 1, 1), 1): 1.0}\n"
        "    by_perm: Static[dict[Perm, float]] = {Perm.R: 1.0}\n"
    )
    path = write_model(tmp_path, _stem(tmp_path, "tuplekeys"), head=head, body=body)
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    entry = app.catalog.get(model_id)
    assert entry is not None
    m = sys.modules[entry.cls.__module__]
    values = {
        "by_sex_age": {(m.Sex.F, 40): 1.0, (m.Sex.M, 40): 1.2},
        "by_day": {(dt.date(2028, 1, 1), 2): 2.0},
        "by_perm": {m.Perm(0): 1.0},
    }
    expected = halley.build(entry.cls, **values)
    typed = {
        "by_sex_age": '{"[\\"F\\", 40]": 1.0, "[\\"M\\", 40]": 1.2}',
        "by_day": '{"[\\"2028-01-01\\", 2]": 2.0}',
        "by_perm": '{"": 1.0}',
    }
    built = done(app, call(app, "POST", "/api/build", model=model_id, statics=typed))["build"]
    assert ir_of(app, built) == expected.ir_hash
    assert not [w for w in app._build(built).warnings if "rerun" in w]
    made = run(
        expected,
        model_points=tutorial_points(),
        tables={"mort": mort_table()},
        settings=RunSettings(valuation_date=VALUATION_DATE),
        run_dir=app.runs_root / "python-tuplekeys",
    )
    other = App(tmp_path / "other-runs")
    try:
        again = done(other, call(other, "POST", "/api/run/rerun", dir=str(made.run_dir)))
        assert again["same_run_id"] is True
    finally:
        other.close()


def test_a_rerun_tries_every_dict_sorted_before_mixes(app: App, tmp_path: Path) -> None:
    """Five dicts whose defaults are unsorted, all given sorted (as a settings file written with
    sorted keys gives them): a rerun tries all at their defaults, then all sorted, so the run
    reruns; with seven, a mix of three sorted the tries do not reach is warned of as such."""
    body = "".join(
        f"    d{i}: Static[dict[str, float]] = {{'b': 2.0, 'a': 1.0}}\n" for i in range(7)
    )
    path = write_model(tmp_path, _stem(tmp_path, "manydicts"), body=body)
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    entry = app.catalog.get(model_id)
    assert entry is not None
    statics: dict[str, Any] = {f"d{i}": {"a": 1.0, "b": 2.0} for i in range(7)}
    assert rerun_misses(entry.cls, statics).names == []
    made = run(
        halley.build(entry.cls, **statics),
        model_points=tutorial_points(),
        tables={"mort": mort_table()},
        settings=RunSettings(valuation_date=VALUATION_DATE),
        run_dir=app.runs_root / "python-manydicts",
    )
    other = App(tmp_path / "other-runs")
    try:
        again = done(other, call(other, "POST", "/api/run/rerun", dir=str(made.run_dir)))
        assert again["same_run_id"] is True
    finally:
        other.close()
    mixed = {f"d{i}": {"a": 1.0, "b": 2.0} if i >= 4 else {"b": 2.0, "a": 1.0} for i in range(7)}
    missed = rerun_misses(entry.cls, mixed)
    assert missed.names == []
    assert missed.beyond == ["d4", "d5", "d6"]
    typed = {f"d{i}": '{"a": 1.0, "b": 2.0}' for i in range(4, 7)}
    built = done(app, call(app, "POST", "/api/build", model=model_id, statics=typed))["build"]
    assert [w for w in app._build(built).warnings if "mixes" in w]


def test_the_dict_warning_gives_the_order_a_manifest_writes(app: App, tmp_path: Path) -> None:
    """A dict keyed by an int enum's members: a manifest sorts its keys by their values, as
    text, not by the names the form shows, so the warning gives the order that reruns."""
    head = "import enum\n\n\nclass Level(enum.IntEnum):\n    LOW = 1\n    HIGH = 2\n"
    body = "    by_level: Static[dict[Level, float]] = {Level.LOW: 1.0}\n"
    path = write_model(tmp_path, _stem(tmp_path, "levelorder"), head=head, body=body)
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]

    def warned(text: str) -> list[str]:
        made = done(
            app, call(app, "POST", "/api/build", model=model_id, statics={"by_level": text})
        )
        return [w for w in app._build(made["build"]).warnings if "by_level" in w]

    assert ["(LOW, HIGH)" in w for w in warned('{"HIGH": 2.0, "LOW": 1.0}')] == [True]
    assert warned('{"LOW": 1.0, "HIGH": 2.0}') == []


# ------------------------------------- round 9: what a failed load keeps, tracebacks, warnings


@pytest.mark.parametrize("regular", [True, False], ids=["package", "namespace"])
def test_a_package_a_failed_load_keeps_for_a_build_is_whole_and_recorded(
    app: App, tmp_path: Path, regular: bool
) -> None:
    """The package above a running build's import, which a file failing to load meanwhile
    imported first, stays whole, with what the load imported under it (its own code may hold
    it), and recorded: the build works, and the package's edit, or its folder made a package,
    shows as a change that one reload builds."""
    stem = _stem(tmp_path, f"keptpkg_{regular}")
    flag = tmp_path / f"{stem}.loading"
    folder = tmp_path / f"{stem}_pkg"
    folder.mkdir()
    init = folder / "__init__.py"
    if regular:
        init.write_text("from . import h\n\nSCALE = 1.0\n", encoding="utf-8")
    (folder / "h.py").write_text("BASE = 1.0\n", encoding="utf-8")
    (folder / "k.py").write_text("import time\ntime.sleep(1.0)\nLOADING = 1.0\n", encoding="utf-8")
    a = write_model(tmp_path, f"{stem}_a", head="import time\nfrom pathlib import Path\n")
    scale = f"{stem}_pkg.SCALE" if regular else "1.0"
    edit(
        a,
        "        return self.deaths(t) * self.mp.sum_assured\n",
        f"        deadline = time.monotonic() + 10\n"
        f"        while not Path(r'{flag}').exists() and time.monotonic() < deadline:\n"
        f"            time.sleep(0.01)\n"
        f"        import {stem}_pkg.h\n        import {stem}_pkg.k\n\n"
        f"        return self.deaths(t) * self.mp.sum_assured * {stem}_pkg.h.BASE * {scale}\n",
    )
    a_id = call(app, "POST", "/api/models/load", path=str(a))["file"]["models"][0]
    job = call(app, "POST", "/api/build", model=a_id)
    c = write_model(
        tmp_path,
        f"{stem}_c",
        head=(
            f"import sys\nimport time\nfrom pathlib import Path\n\nimport {stem}_pkg.h\n\n"
            f"Path(r'{flag}').write_text('')\n"
            f"deadline = time.monotonic() + 10\n"
            f"while '{stem}_pkg.k' not in sys.modules and time.monotonic() < deadline:\n"
            f"    time.sleep(0.01)\n"
            f"time.sleep(0.2)\nundefined_name\n"
        ),
    )
    assert refused(app, "POST", "/api/models/load", path=str(c)).status == 422
    assert done(app, job)["build"]
    assert f"{stem}_pkg.h" in sys.modules
    assert call(app, "GET", "/api/models")["changed"] is False
    if regular:
        edit(init, "SCALE = 1.0", "SCALE = 2.0")
    else:
        init.write_text("SCALE = 2.0\n", encoding="utf-8")
    assert call(app, "GET", "/api/models")["changed"] is True
    call(app, "POST", "/api/models/reload")
    formula = death_outgo(app, a_id)
    if regular:
        assert formula.endswith("* 2.0)")
    else:
        assert getattr(sys.modules[f"{stem}_pkg"], "SCALE", None) == 2.0


def test_a_failed_load_takes_out_a_folder_it_imported_as_a_namespace(
    app: App, tmp_path: Path
) -> None:
    """A file that fails importing a name from a folder without an __init__.py: the folder's
    namespace package goes with the failure, so once the folder is given an __init__.py with
    the name, the file loads, as a new Python process would load it."""
    stem = _stem(tmp_path, "nsfailed")
    folder = tmp_path / f"{stem}_tables"
    folder.mkdir()
    (folder / "rates.csv").write_text("age,rate\n", encoding="utf-8")
    c = write_model(tmp_path, f"{stem}_c", head=f"from {stem}_tables import SHOCK\n")
    assert refused(app, "POST", "/api/models/load", path=str(c)).status == 422
    assert f"{stem}_tables" not in sys.modules
    (folder / "__init__.py").write_text("SHOCK = 2.0\n", encoding="utf-8")
    assert call(app, "POST", "/api/models/load", path=str(c))["models"]


def test_a_failed_load_s_traceback_imports_nothing_again(app: App, tmp_path: Path) -> None:
    """A file that fails importing a name its helper lacks: the error is written out while the
    helper is still imported (Python 3.12 suggests a near name by importing the module again),
    so the clean-up's taking the helper out holds, and the file loads once the name is added."""
    stem = _stem(tmp_path, "suggest")
    helper = tmp_path / f"{stem}_h.py"
    helper.write_text("LOADINGS = 1.0\n", encoding="utf-8")
    c = write_model(tmp_path, f"{stem}_c", head=f"from {stem}_h import LOADING\n")
    refusal = refused(app, "POST", "/api/models/load", path=str(c))
    assert refusal.status == 422
    assert "LOADINGS" in str(refusal.detail.get("traceback", ""))
    assert f"{stem}_h" not in sys.modules
    edit(helper, "LOADINGS = 1.0", "LOADING = 1.0")
    assert call(app, "POST", "/api/models/load", path=str(c))["models"]


def test_the_rerun_warnings_name_each_cause(app: App, tmp_path: Path) -> None:
    """Dicts given sorted in a mix a rerun's tries do not reach, beside a dict typed in another
    key order: each is warned of with its own cause, the mix by the tries' cap, the other dict by
    its keys' order; and a dict held inside a setting is named by where it is."""
    body = "".join(
        f"    d{i}: Static[dict[str, float]] = {{'b': 2.0, 'a': 1.0}}\n" for i in range(7)
    )
    body += "    need: Static[dict[str, float]]\n"
    body += "    nested: Static[dict[str, dict[str, float]]] = {'x': {'a': 1.0}}\n"
    path = write_model(tmp_path, _stem(tmp_path, "causes"), body=body)
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    typed = {f"d{i}": '{"a": 1.0, "b": 2.0}' for i in range(4, 7)}
    typed["need"] = '{"b": 2.0, "a": 1.0}'
    typed["nested"] = '{"x": {"c": 3.0, "a": 1.0}}'
    made = done(app, call(app, "POST", "/api/build", model=model_id, statics=typed))["build"]
    warned = [w for w in app._build(made).warnings if "rerun" in w]
    assert len(warned) == 3
    mix, need, nested = warned
    assert mix.startswith("build-time settings d4, d5, d6 are dicts given sorted")
    assert "64 mixes" in mix
    assert need.startswith("build-time setting need is a dict")
    assert "(a, b)" in need
    assert nested.startswith('build-time setting nested holds a dict, at ["x"],')
    assert "(a, c)" in nested


def test_a_tuple_key_holding_infinity_reads_back(app: App, tmp_path: Path) -> None:
    """An open-ended band key (infinity in a tuple key), which a manifest records as ``(inf,
    1)``: a Python run with it reruns in a new session."""
    body = "    bands: Static[dict[tuple[float, int], float]] = {(65.0, 1): 1.0}\n"
    path = write_model(tmp_path, _stem(tmp_path, "infkey"), body=body)
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    entry = app.catalog.get(model_id)
    assert entry is not None
    bands = {(65.0, 1): 1.0, (math.inf, 1): 1.5}
    made = run(
        halley.build(entry.cls, bands=bands),
        model_points=tutorial_points(),
        tables={"mort": mort_table()},
        settings=RunSettings(valuation_date=VALUATION_DATE),
        run_dir=app.runs_root / "python-infkey",
    )
    other = App(tmp_path / "other-runs")
    try:
        again = done(other, call(other, "POST", "/api/run/rerun", dir=str(made.run_dir)))
        assert again["same_run_id"] is True
    finally:
        other.close()


# ------------------------------- round 10: errors written safely, loads beside builds, reads

SLIPS = {
    "exceptions": (
        "class Problems(Exception):\n"
        "    pass\n\n\n"
        "slip = Problems('two problems')\n"
        "slip.exceptions = ['a rate', 'a table']\n"
        "raise slip\n"
    ),
    "getattr": (
        "class Box:\n"
        "    def __getattr__(self, name):\n"
        "        raise KeyError(name)\n\n"
        "    def rate(self):\n"
        "        return undefined_rate\n\n\n"
        "Box().rate()\n"
    ),
    "str": (
        "class Silent(Exception):\n"
        "    def __str__(self):\n"
        "        raise RuntimeError('no text')\n\n\n"
        "raise Silent()\n"
    ),
    "str_exit": (
        "class Quit(Exception):\n"
        "    def __str__(self):\n"
        "        raise SystemExit('from str')\n\n\n"
        "raise Quit()\n"
    ),
    "getattr_exit": (
        "class Box:\n"
        "    def __getattr__(self, name):\n"
        "        raise SystemExit(name)\n\n"
        "    def rate(self):\n"
        "        return undefined_rate\n\n\n"
        "Box().rate()\n"
    ),
    "name_fixed": (
        "class Missing(NameError):\n"
        "    @property\n"
        "    def name(self):\n"
        "        return self.args[0]\n\n\n"
        "class Box:\n"
        "    def __getattr__(self, name):\n"
        "        raise KeyError(name)\n\n"
        "    def rate(self):\n"
        "        raise Missing('undefined_rate')\n\n\n"
        "Box().rate()\n"
    ),
    "cause_exit": (
        "class Odd(Exception):\n"
        "    @property\n"
        "    def __cause__(self):\n"
        "        raise SystemExit('cause')\n\n\n"
        "raise Odd('odd')\n"
    ),
    "group_nonexc": (
        "class Odd(ExceptionGroup):\n"
        "    @property\n"
        "    def exceptions(self):\n"
        "        return ('a rate', 2)\n\n\n"
        "raise Odd('odd', [ValueError('v')])\n"
    ),
    "class_exit": (
        "class Odd(Exception):\n"
        "    @property\n"
        "    def __class__(self):\n"
        "        raise SystemExit('class')\n\n\n"
        "raise Odd('odd')\n"
    ),
    "traceback_exit": (
        "class Odd(Exception):\n"
        "    @property\n"
        "    def __traceback__(self):\n"
        "        raise SystemExit('traceback')\n\n\n"
        "raise Odd('odd')\n"
    ),
    "syntax_filename": (
        "class Odd(SyntaxError):\n"
        "    @property\n"
        "    def filename(self):\n"
        "        raise SystemExit('filename')\n\n\n"
        "raise Odd('odd')\n"
    ),
    "str_format": (
        "class Text(str):\n"
        "    def __format__(self, spec):\n"
        "        raise SystemExit('format')\n\n\n"
        "class Odd(Exception):\n"
        "    def __str__(self):\n"
        "        return Text('odd text')\n\n\n"
        "raise Odd()\n"
    ),
}


@pytest.mark.parametrize("slip", sorted(SLIPS))
def test_an_error_python_cannot_write_plainly_is_refused_at_the_file_s_line(
    app: App, tmp_path: Path, slip: str
) -> None:
    """A file raising an error whose writing trips (an attribute named ``exceptions`` holding
    text, a ``__getattr__`` raising the ``KeyError`` or the ``SystemExit`` a ``NameError``'s
    suggestion meets, one whose ``name`` cannot be set to hush it, a ``__str__`` that raises,
    ``SystemExit`` too, or gives a text whose own ``__format__`` raises; a ``__cause__``,
    ``__class__``, ``__traceback__`` or ``filename`` property that raises, a group whose members
    are no exceptions): the load is refused at the file's line and leaves none of its modules
    imported, again and again; a reload that meets it puts every other file back."""
    stem = _stem(tmp_path, f"slip_{slip}")
    (tmp_path / f"{stem}_h.py").write_text("LOADING = 1.0\n", encoding="utf-8")
    c = write_model(tmp_path, f"{stem}_c", head=f"import {stem}_h\n\n{SLIPS[slip]}")
    for _ in range(2):
        refusal = refused(app, "POST", "/api/models/load", path=str(c))
        assert refusal.status == 422
        loc = typing.cast(dict[str, Any], refusal.detail["loc"])
        assert Path(loc["file"]).resolve() == c.resolve()
        assert "Traceback" in str(refusal.detail["traceback"])
        assert f"{stem}_c" not in sys.modules
        assert f"{stem}_h" not in sys.modules
    b = loading_model(tmp_path, f"{stem}_b", f"{stem}_h")
    b_id = call(app, "POST", "/api/models/load", path=str(b))["file"]["models"][0]
    a = write_model(tmp_path, f"{stem}_a")
    call(app, "POST", "/api/models/load", path=str(a))
    edit(a, '"""A model file', f'{SLIPS[slip]}"""A model file')
    assert refused(app, "POST", "/api/models/reload").status == 422
    assert f"{stem}_b" in sys.modules
    assert f"{stem}_h" in sys.modules
    assert done(app, call(app, "POST", "/api/build", model=b_id))["build"]


def test_format_quietly_writes_what_python_s_own_writing_trips_on() -> None:
    """A ``NameError`` in a method of a class whose ``__getattr__`` raises a ``KeyError``, which
    Python 3.12's suggestion of ``self.<name>`` calls, and an exception group: each is written,
    and the error is left as it was."""

    class Box:
        def __getattr__(self, name: str) -> object:
            raise KeyError(name)

        def rate(self) -> object:
            return eval("undefined_rate")

    try:
        Box().rate()
    except NameError as exc:
        error = exc
    else:
        raise AssertionError("no NameError")
    written = format_quietly(error)
    assert written.rstrip().endswith("NameError: name 'undefined_rate' is not defined")
    assert error.name == "undefined_rate"
    group = ExceptionGroup("two", [ValueError("a rate"), KeyError("a table")])
    assert "a table" in format_quietly(group)


def test_a_failed_load_beside_a_build_keeps_what_the_build_s_module_holds(
    app: App, tmp_path: Path
) -> None:
    """A build whose formula imports a module that imports a helper a file failing to load
    meanwhile imported first: the helper stays imported, and recorded, so an edit to it shows,
    and one reload builds what Python builds."""
    stem = _stem(tmp_path, "heldhelper")
    flag = tmp_path / f"{stem}.loading"
    helper = tmp_path / f"{stem}_h.py"
    helper.write_text("LOADING = 1.0\n", encoding="utf-8")
    (tmp_path / f"{stem}_k.py").write_text(f"import {stem}_h as h\n", encoding="utf-8")
    a = write_model(tmp_path, f"{stem}_a", head="import time\nfrom pathlib import Path\n")
    edit(
        a,
        "        return self.deaths(t) * self.mp.sum_assured\n",
        f"        deadline = time.monotonic() + 10\n"
        f"        while not Path(r'{flag}').exists() and time.monotonic() < deadline:\n"
        f"            time.sleep(0.01)\n"
        f"        import {stem}_k\n\n"
        f"        return self.deaths(t) * self.mp.sum_assured * {stem}_k.h.LOADING\n",
    )
    a_id = call(app, "POST", "/api/models/load", path=str(a))["file"]["models"][0]
    job = call(app, "POST", "/api/build", model=a_id)
    c = write_model(
        tmp_path,
        f"{stem}_c",
        head=(
            f"import sys\nimport time\nfrom pathlib import Path\n\nimport {stem}_h\n\n"
            f"Path(r'{flag}').write_text('')\n"
            f"deadline = time.monotonic() + 10\n"
            f"while '{stem}_k' not in sys.modules and time.monotonic() < deadline:\n"
            f"    time.sleep(0.01)\n"
            f"time.sleep(0.2)\nundefined_name\n"
        ),
    )
    assert refused(app, "POST", "/api/models/load", path=str(c)).status == 422
    assert done(app, job)["build"]
    assert sys.modules[f"{stem}_k"].h is sys.modules[f"{stem}_h"]
    assert call(app, "GET", "/api/models")["changed"] is False
    edit(helper, "LOADING = 1.0", "LOADING = 1.5")
    assert call(app, "GET", "/api/models")["changed"] is True
    call(app, "POST", "/api/models/reload")
    assert death_outgo(app, a_id).endswith("* 1.5)")


def test_a_failed_load_beside_a_build_leaves_a_package_s_submodule_on_it(
    app: App, tmp_path: Path
) -> None:
    """A package a build is importing, whose own code binds a submodule (``from . import h``)
    that a file failing to load meanwhile imported first: the submodule stays, on the package
    too, so that build and every later one work, and an edit to it shows."""
    stem = _stem(tmp_path, "initbind")
    folder = tmp_path / f"{stem}_pkg"
    folder.mkdir()
    (folder / "__init__.py").write_text(
        "import time\n\ntime.sleep(1.5)\nfrom . import h\n", encoding="utf-8"
    )
    h = folder / "h.py"
    h.write_text("BASE = 1.0\n", encoding="utf-8")
    a = write_model(tmp_path, f"{stem}_a")
    edit(
        a,
        "        return self.deaths(t) * self.mp.sum_assured\n",
        f"        import {stem}_pkg\n\n"
        f"        return self.deaths(t) * self.mp.sum_assured * {stem}_pkg.h.BASE\n",
    )
    a_id = call(app, "POST", "/api/models/load", path=str(a))["file"]["models"][0]
    job = call(app, "POST", "/api/build", model=a_id)
    c = write_model(
        tmp_path,
        f"{stem}_c",
        head=(
            f"import sys\nimport time\n\n"
            f"deadline = time.monotonic() + 10\n"
            f"while '{stem}_pkg' not in sys.modules and time.monotonic() < deadline:\n"
            f"    time.sleep(0.01)\n"
            f"import {stem}_pkg.h\n\n"
            f"time.sleep(0.2)\nundefined_name\n"
        ),
    )
    assert refused(app, "POST", "/api/models/load", path=str(c)).status == 422
    assert done(app, job)["build"]
    assert sys.modules[f"{stem}_pkg"].h is sys.modules[f"{stem}_pkg.h"]
    assert death_outgo(app, a_id).endswith("* 1.0)")
    assert call(app, "GET", "/api/models")["changed"] is False
    edit(h, "BASE = 1.0", "BASE = 2.0")
    assert call(app, "GET", "/api/models")["changed"] is True
    call(app, "POST", "/api/models/reload")
    assert death_outgo(app, a_id).endswith("* 2.0)")


def test_a_tuple_key_reads_back_infinity_inside_an_untyped_item(app: App, tmp_path: Path) -> None:
    """A tuple key whose items are untyped holding a tuple with infinity in it, which a manifest
    records as ``((inf, 1), 2)``: a Python run with it reruns in a new session."""
    body = "    bands: Static[dict[tuple, float]] = {((65.0, 1), 2): 1.0}\n"
    path = write_model(tmp_path, _stem(tmp_path, "deepinf"), body=body)
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    entry = app.catalog.get(model_id)
    assert entry is not None
    bands = {((65.0, 1), 2): 1.0, ((math.inf, 1), 2): 1.5}
    assert rerun_misses(entry.cls, {"bands": bands}).names == []
    made = run(
        halley.build(entry.cls, bands=bands),
        model_points=tutorial_points(),
        tables={"mort": mort_table()},
        settings=RunSettings(valuation_date=VALUATION_DATE),
        run_dir=app.runs_root / "python-deepinf",
    )
    other = App(tmp_path / "other-runs")
    try:
        again = done(other, call(other, "POST", "/api/run/rerun", dir=str(made.run_dir)))
        assert again["same_run_id"] is True
    finally:
        other.close()


def test_rerun_misses_names_every_cause_beside_a_setting_that_does_not_read_back(
    app: App, tmp_path: Path
) -> None:
    """An object a manifest records as its repr, which a rerun refuses, beside dicts given
    sorted in a mix the tries do not reach and a dict typed in another key order: each is named,
    so one build's warnings say everything that keeps its run from a rerun."""
    head = (
        "class Basis:\n"
        "    def __init__(self, name):\n"
        "        self.name = name\n\n"
        "    def __repr__(self):\n"
        "        return f'Basis({self.name!r})'\n"
    )
    body = "".join(
        f"    d{i}: Static[dict[str, float]] = {{'b': 2.0, 'a': 1.0}}\n" for i in range(7)
    )
    body += "    basis: Static[Basis] = Basis('best')\n    need: Static[dict[str, float]]\n"
    path = write_model(tmp_path, _stem(tmp_path, "unreadmix"), head=head, body=body)
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    entry = app.catalog.get(model_id)
    assert entry is not None
    statics: dict[str, Any] = {
        f"d{i}": {"a": 1.0, "b": 2.0} if i >= 4 else {"b": 2.0, "a": 1.0} for i in range(7)
    }
    statics["basis"] = sys.modules[entry.cls.__module__].Basis("stress")
    statics["need"] = {"b": 2.0, "a": 1.0}
    missed = rerun_misses(entry.cls, statics)
    assert missed.names == ["basis", "need"]
    assert missed.beyond == ["d4", "d5", "d6"]


def test_a_date_and_time_with_an_offset_reads_back_from_a_manifest(
    app: App, tmp_path: Path
) -> None:
    """A date and time typed with an offset or in UTC, which a manifest records by its repr with
    its time zone: the build gives no rerun warning, and a Python run with one reruns in a new
    session."""
    head = "import datetime as dt\n"
    body = "    stamp: Static[dt.datetime] = dt.datetime(2027, 1, 1, 9, 0)\n"
    path = write_model(tmp_path, _stem(tmp_path, "offsetstamp"), head=head, body=body)
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    for text in ("2027-01-01T09:00:00+01:00", "2027-01-01T09:00:00Z"):
        made = done(app, call(app, "POST", "/api/build", model=model_id, statics={"stamp": text}))
        assert not [w for w in app._build(made["build"]).warnings if "rerun" in w]
    entry = app.catalog.get(model_id)
    assert entry is not None
    stamp = dt.datetime(2027, 1, 1, 9, 0, tzinfo=dt.timezone(dt.timedelta(hours=-5), "EST"))
    made_run = run(
        halley.build(entry.cls, stamp=stamp),
        model_points=tutorial_points(),
        tables={"mort": mort_table()},
        settings=RunSettings(valuation_date=VALUATION_DATE),
        run_dir=app.runs_root / "python-offsetstamp",
    )
    other = App(tmp_path / "other-runs")
    try:
        again = done(other, call(other, "POST", "/api/run/rerun", dir=str(made_run.run_dir)))
        assert again["same_run_id"] is True
    finally:
        other.close()


# ------------------- round 11: errors whatever they raise, objects in sys.modules, list items


def test_format_quietly_holds_whatever_the_user_s_code_raises() -> None:
    """A ``__str__`` that raises ``SystemExit``, a suggestion that meets a ``__getattr__``
    raising it, and a ``NameError`` whose ``name`` cannot be set, beside one whose can: each is
    written, and every error is left as it was."""

    class QuitError(Exception):
        def __str__(self) -> str:
            raise SystemExit("from str")

    assert exception_text(QuitError()) == "<exception str() failed>"
    assert format_quietly(QuitError()).rstrip().endswith("<exception str() failed>")

    class Exits:
        def __getattr__(self, name: str) -> object:
            raise SystemExit(name)

        def rate(self) -> object:
            return eval("undefined_rate")

    try:
        Exits().rate()
    except NameError as exc:
        trip = exc
    else:
        raise AssertionError("no NameError")
    written = format_quietly(trip)
    assert written.rstrip().endswith("NameError: name 'undefined_rate' is not defined")
    assert trip.name == "undefined_rate"

    class MissingError(NameError):
        @property
        def name(self) -> object:  # pyright: ignore[reportIncompatibleVariableOverride]
            return self.args[0]

    class Box:
        def __getattr__(self, name: str) -> object:
            raise KeyError(name)

        def rate(self) -> object:
            try:
                raise MissingError("undefined_rate")
            except MissingError:
                return eval("other_rate")

    try:
        Box().rate()
    except NameError as exc:
        error = exc
    else:
        raise AssertionError("no NameError")
    written = format_quietly(error)
    assert written.startswith("Traceback (most recent call last):")
    assert written.rstrip().endswith("NameError: name 'other_rate' is not defined")
    assert error.name == "other_rate"


def test_a_fault_answers_500_with_its_traceback_whatever_it_raises() -> None:
    """A route's fault that is no engine error, whose text cannot be had: the answer is a 500
    with its type, the text Python's traceback writes for it, and the traceback."""

    class OddError(RuntimeError):
        def __str__(self) -> str:
            raise SystemExit("no text")

    def broken(request: Request) -> object:
        raise OddError()

    with pytest.raises(ApiError) as info:
        _guarded(broken)(Request("GET", "/api/broken", {}, {}))
    assert info.value.status == 500
    assert info.value.message == "OddError: <exception str() failed>"
    assert info.value.detail["type"] == "OddError"
    assert "Traceback (most recent call last):" in str(info.value.detail["traceback"])


def test_an_error_s_notes_are_written_as_python_writes_them() -> None:
    """A note whose text cannot be had is ``<note str() failed>``, the others kept, and notes
    that are a text rather than a list of them are written as the text's repr."""

    class Odd(str):
        def __str__(self) -> str:
            raise ValueError("no text")

    error = ValueError("a rate")
    error.add_note("first")
    typing.cast(list[str], error.__notes__).append(Odd("second"))
    assert describe.describe_error(error)["notes"] == ["first", "<note str() failed>"]
    text = ValueError("a table")
    text.__notes__ = "one note"  # pyright: ignore[reportAttributeAccessIssue]
    assert describe.describe_error(text)["notes"] == ["'one note'"]


def test_a_reload_stopped_by_what_is_no_exception_puts_everything_back(
    app: App, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reload stopped part-way by what is no ``Exception`` (a ``KeyboardInterrupt``): every
    module, file and model is put back as it was, and it goes on as it is."""
    stem = _stem(tmp_path, "interrupted")
    (tmp_path / f"{stem}_h.py").write_text("LOADING = 1.0\n", encoding="utf-8")
    b = loading_model(tmp_path, f"{stem}_b", f"{stem}_h")
    b_id = call(app, "POST", "/api/models/load", path=str(b))["file"]["models"][0]
    before = {n: sys.modules[n] for n in (f"{stem}_b", f"{stem}_h")}
    files = [f.as_dict() for f in app.catalog.files()]

    def stopped(*args: object, **kwargs: object) -> object:
        raise KeyboardInterrupt

    with monkeypatch.context() as patched:
        patched.setattr(Catalog, "_record", stopped)
        with pytest.raises(KeyboardInterrupt):
            app.catalog.reload()
    assert {n: sys.modules.get(n) for n in before} == before
    assert [f.as_dict() for f in app.catalog.files()] == files
    assert app.catalog.get(b_id) is not None
    assert death_outgo(app, b_id).endswith("* 1.0)")


def object_helper(path: Path, loading: str, *, strict: bool) -> None:
    """A helper module that puts an object in its own place in ``sys.modules``, whose
    ``__getattr__`` reads a dict: for a name it lacks, a ``KeyError`` when ``strict``, an
    ``AttributeError`` otherwise."""
    lookup = (
        "        return self._values[name]\n"
        if strict
        else "        try:\n"
        "            return self._values[name]\n"
        "        except KeyError:\n"
        "            raise AttributeError(name) from None\n"
    )
    path.write_text(
        "import sys\n\n\n"
        "class _Basis:\n"
        "    def __init__(self, values):\n"
        "        self._values = values\n\n"
        "    def __getattr__(self, name):\n"
        f"{lookup}\n\n"
        f"sys.modules[__name__] = _Basis({{'LOADING': {loading}}})\n",
        encoding="utf-8",
    )


def object_model(folder: Path, stem: str, helper: str, extra: str = "") -> Path:
    """The tutorial's model, its death outgo times the LOADING of a helper it imports whole."""
    path = write_model(folder, stem, head=f"import {helper}\n{extra}")
    source = path.read_text(encoding="utf-8")
    loaded = source.replace("* self.mp.sum_assured", f"* self.mp.sum_assured * {helper}.LOADING")
    path.write_text(loaded, encoding="utf-8")
    return path


def test_a_helper_that_puts_an_object_in_its_place_is_followed(app: App, tmp_path: Path) -> None:
    """A helper that puts an object in its own place in ``sys.modules``, which ``import`` gives:
    the catalogue follows it by the file that import ran, as a helper of each file that imports
    it, so an edit to it shows as a change and a reload builds what it says now, as a new Python
    process would."""
    stem = _stem(tmp_path, "objhelper")
    helper = tmp_path / f"{stem}_basis.py"
    object_helper(helper, "1.0", strict=False)
    path = object_model(tmp_path, f"{stem}_m", f"{stem}_basis")
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    second = object_model(tmp_path, f"{stem}_n", f"{stem}_basis")
    call(app, "POST", "/api/models/load", path=str(second))
    files = call(app, "GET", "/api/models")["files"]
    assert [f["helpers"] for f in files] == [[f"{stem}_basis"], [f"{stem}_basis"]]
    assert death_outgo(app, model_id).endswith("* 1.0)")
    assert call(app, "GET", "/api/models")["changed"] is False
    edit(helper, "'LOADING': 1.0", "'LOADING': 1.5")
    assert call(app, "GET", "/api/models")["changed"] is True
    call(app, "POST", "/api/models/reload")
    assert death_outgo(app, model_id).endswith("* 1.5)")
    assert call(app, "GET", "/api/models")["changed"] is False


def test_an_object_in_sys_modules_whose_getattr_raises_does_not_stop_a_load(
    app: App, tmp_path: Path
) -> None:
    """A helper's object in ``sys.modules`` whose ``__getattr__`` raises a ``KeyError`` for any
    name it lacks (``__loader__``, ``__file__``): a file that imports it loads and builds, and
    one that fails after importing another is refused at its own line, that helper taken out.
    The objects are taken out at the end, since code that reads every module's ``__file__``
    (Hypothesis's, in a later test) trips on one, as the catalogue no longer does."""
    stem = _stem(tmp_path, "strictobj")
    try:
        object_helper(tmp_path / f"{stem}_basis.py", "1.25", strict=True)
        path = object_model(tmp_path, f"{stem}_m", f"{stem}_basis")
        model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
        assert death_outgo(app, model_id).endswith("* 1.25)")
        object_helper(tmp_path / f"{stem}_other.py", "2.0", strict=True)
        c = object_model(tmp_path, f"{stem}_c", f"{stem}_other", extra="undefined_name\n")
        refusal = refused(app, "POST", "/api/models/load", path=str(c))
        assert refusal.status == 422
        loc = typing.cast(dict[str, Any], refusal.detail["loc"])
        assert Path(loc["file"]).resolve() == c.resolve()
        assert f"{stem}_other" not in sys.modules
        assert f"{stem}_c" not in sys.modules
    finally:
        for name in (f"{stem}_basis", f"{stem}_other"):
            sys.modules.pop(name, None)


def test_a_list_setting_is_read_by_its_item_type(app: App, tmp_path: Path) -> None:
    """Settings declared lists (and a Sequence) of dates, an enum's members, tuples and floats,
    typed as JSON: each item is read by the item type, as a tuple's items are, so the build is
    the one the Python call makes, with no rerun warning; and a run made in Python with them
    reruns in a new session, its manifest's lists read back so too."""
    head = (
        "import datetime as dt\nimport enum\nfrom collections.abc import Sequence\n\n\n"
        "class Sex(enum.Enum):\n    M = 'male'\n    F = 'female'\n"
    )
    body = (
        "    dates: Static[list[dt.date]] = [dt.date(2027, 1, 1)]\n"
        "    sexes: Static[list[Sex]] = [Sex.F]\n"
        "    pairs: Static[list[tuple[float, int]]] = [(1.0, 1)]\n"
        "    weights: Static[list[float]] = [1.0]\n"
        "    stamps: Static[Sequence[dt.datetime]] = [dt.datetime(2027, 1, 1, 9, 0)]\n"
    )
    path = write_model(tmp_path, _stem(tmp_path, "lists"), head=head, body=body)
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    entry = app.catalog.get(model_id)
    assert entry is not None
    sex = sys.modules[entry.cls.__module__].Sex
    typed = {
        "dates": '["2028-02-03"]',
        "sexes": '["M", "F"]',
        "pairs": "[[2, 3]]",
        "weights": "[2, 3.5]",
        "stamps": '["2027-01-01T09:00:00+01:00"]',
    }
    values: dict[str, Any] = {
        "dates": [dt.date(2028, 2, 3)],
        "sexes": [sex.M, sex.F],
        "pairs": [(2.0, 3)],
        "weights": [2.0, 3.5],
        "stamps": [dt.datetime(2027, 1, 1, 9, 0, tzinfo=dt.timezone(dt.timedelta(hours=1)))],
    }
    made = done(app, call(app, "POST", "/api/build", model=model_id, statics=typed))["build"]
    python = halley.build(entry.cls, **values)
    assert app._build(made).compiled.ir_hash == python.ir_hash
    assert not [w for w in app._build(made).warnings if "rerun" in w]
    assert rerun_misses(entry.cls, values).names == []
    made_run = run(
        python,
        model_points=tutorial_points(),
        tables={"mort": mort_table()},
        settings=RunSettings(valuation_date=VALUATION_DATE),
        run_dir=app.runs_root / "python-lists",
    )
    other = App(tmp_path / "other-runs")
    try:
        again = done(other, call(other, "POST", "/api/run/rerun", dir=str(made_run.run_dir)))
        assert again["same_run_id"] is True
    finally:
        other.close()


# ----------------- round 12: exceptions read past their own properties, module objects, keys


def test_format_quietly_reads_an_exception_past_its_own_properties() -> None:
    """An exception whose ``__cause__`` or ``__traceback__`` is a property that raises, and a
    group whose members are no exceptions: each is written with its frames, read as Python keeps
    them. A ``name`` whose setter sets it and then raises is put back too."""

    class CauseError(Exception):
        @property
        def __cause__(self) -> object:  # pyright: ignore[reportIncompatibleVariableOverride]
            raise SystemExit("cause")

    class TracebackError(Exception):
        @property
        def __traceback__(self) -> object:  # pyright: ignore[reportIncompatibleVariableOverride]
            raise SystemExit("traceback")

    class OddGroupError(ExceptionGroup):
        @property
        def exceptions(self) -> object:  # pyright: ignore[reportIncompatibleMethodOverride]
            return ("a rate", 2)

    def raised(make: typing.Callable[[], object]) -> BaseException:
        """What ``make`` raises, or the exception it makes, raised."""
        try:
            made = make()
            if isinstance(made, BaseException):
                raise made
        except BaseException as exc:
            return exc
        raise AssertionError("nothing raised")

    for error in (
        raised(lambda: CauseError("odd")),
        raised(lambda: TracebackError("odd")),
        raised(lambda: OddGroupError("odd", [ValueError("v")])),
    ):
        written = format_quietly(error)
        assert written.startswith("Traceback (most recent call last):")
        assert "test_gui.py" in written, written

    class SetsThenRaisesError(NameError):
        _held: object = "b_name"

        @property
        def name(self) -> object:  # pyright: ignore[reportIncompatibleVariableOverride]
            return self._held

        @name.setter
        def name(self, value: object) -> None:
            self._held = value
            raise RuntimeError("no")

    class Box:
        def __getattr__(self, name: str) -> object:
            raise KeyError(name)

        def rate(self) -> object:
            raise SetsThenRaisesError("b_name")

    sets = raised(lambda: Box().rate())
    format_quietly(sets)
    assert typing.cast(NameError, sets).name == "b_name"


def test_a_sequence_in_a_dict_key_is_read_as_the_tuple_a_key_holds(
    app: App, tmp_path: Path
) -> None:
    """A dict whose key is, or holds, a Sequence: typed as JSON, a key's lists are the tuples a
    key holds, as the Python call has them, and a Python run with such keys reruns."""
    head = "from collections.abc import Sequence\n"
    body = (
        "    pairs: Static[dict[tuple[Sequence[int], int], float]] = {((1, 2), 3): 1.0}\n"
        "    seqs: Static[dict[Sequence[int], float]] = {(1, 2): 1.0}\n"
    )
    path = write_model(tmp_path, _stem(tmp_path, "seqkeys"), head=head, body=body)
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    entry = app.catalog.get(model_id)
    assert entry is not None
    typed = {"pairs": '{"[[1, 2], 3]": 2.0}', "seqs": '{"[1, 2]": 2.0, "[3]": 1.5}'}
    values: dict[str, Any] = {"pairs": {((1, 2), 3): 2.0}, "seqs": {(1, 2): 2.0, (3,): 1.5}}
    made = done(app, call(app, "POST", "/api/build", model=model_id, statics=typed))["build"]
    python = halley.build(entry.cls, **values)
    assert app._build(made).compiled.ir_hash == python.ir_hash
    assert not [w for w in app._build(made).warnings if "rerun" in w]
    made_run = run(
        python,
        model_points=tutorial_points(),
        tables={"mort": mort_table()},
        settings=RunSettings(valuation_date=VALUATION_DATE),
        run_dir=app.runs_root / "python-seqkeys",
    )
    other = App(tmp_path / "other-runs")
    try:
        again = done(other, call(other, "POST", "/api/run/rerun", dir=str(made_run.run_dir)))
        assert again["same_run_id"] is True
    finally:
        other.close()


def test_a_literal_of_enum_members_is_typed_and_carried_by_name(app: App, tmp_path: Path) -> None:
    """A setting declared a Literal of an enum's members, alone and as a list's item: typed by a
    member's name, it builds the member, as the Python call does; a Python run with it reruns;
    and Build again after a reload that edits the enum carries the member by its name."""
    head = (
        "import enum\nfrom typing import Literal\n\n\n"
        "class Basis(enum.Enum):\n    BEST = 'best estimate'\n    PRUDENT = 'prudent'\n"
    )
    body = (
        "    basis: Static[Literal[Basis.BEST, Basis.PRUDENT]] = Basis.BEST\n"
        "    bases: Static[list[Literal[Basis.BEST, Basis.PRUDENT]]] = [Basis.BEST]\n"
    )
    path = write_model(tmp_path, _stem(tmp_path, "literalenum"), head=head, body=body)
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    entry = app.catalog.get(model_id)
    assert entry is not None
    m = sys.modules[entry.cls.__module__]
    typed = {"basis": '"PRUDENT"', "bases": '["PRUDENT", "BEST"]'}
    made = done(app, call(app, "POST", "/api/build", model=model_id, statics=typed))["build"]
    given = {"basis": m.Basis.PRUDENT, "bases": [m.Basis.PRUDENT, m.Basis.BEST]}
    python = halley.build(entry.cls, **given)
    assert ir_of(app, made) == python.ir_hash
    made_run = run(
        python,
        model_points=tutorial_points(),
        tables={"mort": mort_table()},
        settings=RunSettings(valuation_date=VALUATION_DATE),
        run_dir=app.runs_root / "python-literalenum",
    )
    assert done(app, call(app, "POST", "/api/run/rerun", dir=str(made_run.run_dir)))["same_run_id"]
    edit(path, "PRUDENT = 'prudent'", "PRUDENT = 'prudent, edited'")
    call(app, "POST", "/api/models/reload")
    entry = app.catalog.get(model_id)
    assert entry is not None
    m = sys.modules[entry.cls.__module__]
    again = call(app, "POST", "/api/build", model=model_id, statics={}, from_build=made)
    want = halley.build(entry.cls, basis=m.Basis.PRUDENT, bases=[m.Basis.PRUDENT, m.Basis.BEST])
    assert ir_of(app, done(app, again)["build"]) == want.ir_hash


def test_a_union_reads_a_typed_list_by_its_list_option_wherever_it_comes(
    app: App, tmp_path: Path
) -> None:
    """A union whose dict or set option comes before its list or tuple option: a typed list is
    read by the list's or the tuple's item type, as the Python call has it, and a Python run with
    such values reruns."""
    head = "import enum\n\n\nclass Sex(enum.Enum):\n    M = 'male'\n    F = 'female'\n"
    body = (
        "    a: Static[dict[str, float] | list[Sex]] = [Sex.F]\n"
        "    b: Static[set[int] | list[float]] = [1.0]\n"
        "    c: Static[dict[str, float] | tuple[float, ...]] = (1.0,)\n"
    )
    path = write_model(tmp_path, _stem(tmp_path, "unionorder"), head=head, body=body)
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    entry = app.catalog.get(model_id)
    assert entry is not None
    sex = sys.modules[entry.cls.__module__].Sex
    typed = {"a": '["M"]', "b": "[2, 3.5]", "c": "[2, 3.5]"}
    values: dict[str, Any] = {"a": [sex.M], "b": [2.0, 3.5], "c": (2.0, 3.5)}
    made = done(app, call(app, "POST", "/api/build", model=model_id, statics=typed))["build"]
    python = halley.build(entry.cls, **values)
    assert ir_of(app, made) == python.ir_hash
    made_run = run(
        python,
        model_points=tutorial_points(),
        tables={"mort": mort_table()},
        settings=RunSettings(valuation_date=VALUATION_DATE),
        run_dir=app.runs_root / "python-unionorder",
    )
    other = App(tmp_path / "other-runs")
    try:
        again = done(other, call(other, "POST", "/api/run/rerun", dir=str(made_run.run_dir)))
        assert again["same_run_id"] is True
    finally:
        other.close()


def test_an_enum_whose_own_missing_raises_builds_as_python_builds_it(
    app: App, tmp_path: Path
) -> None:
    """Enums whose ``_missing_`` raises for a value that is no member (the docs' lower-casing
    example meeting a float, a lookup by name meeting another text): the untouched form, and
    typed values, build what the Python call builds, without a warning that cannot be told."""
    head = (
        "import enum\n\n\n"
        "class Build(enum.StrEnum):\n"
        "    DEBUG = 'debug'\n"
        "    OPTIMIZED = 'optimized'\n\n"
        "    @classmethod\n"
        "    def _missing_(cls, value):\n"
        "        value = value.lower()\n"
        "        for member in cls:\n"
        "            if member.value == value:\n"
        "                return member\n"
        "        return None\n\n\n"
        "class Kind(enum.Enum):\n"
        "    A = 'a'\n"
        "    B = 'b'\n\n"
        "    @classmethod\n"
        "    def _missing_(cls, value):\n"
        "        return cls[str(value).upper()]\n"
    )
    body = (
        "    steps: Static[list[Build | float]] = [Build.DEBUG, 1.5]\n"
        "    kinds: Static[list[Kind]] = [Kind.A, 'other']\n"
    )
    path = write_model(tmp_path, _stem(tmp_path, "ownmissing"), head=head, body=body)
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    entry = app.catalog.get(model_id)
    assert entry is not None
    m = sys.modules[entry.cls.__module__]
    untouched = done(app, call(app, "POST", "/api/build", model=model_id, statics={}))["build"]
    assert ir_of(app, untouched) == halley.build(entry.cls).ir_hash
    typed = {"steps": '["OPTIMIZED", 2.5]', "kinds": '["B", "other"]'}
    made = done(app, call(app, "POST", "/api/build", model=model_id, statics=typed))["build"]
    want = halley.build(entry.cls, steps=[m.Build.OPTIMIZED, 2.5], kinds=[m.Kind.B, "other"])
    assert ir_of(app, made) == want.ir_hash
    assert not [w for w in app._build(made).warnings if "cannot be told" in w]


def module_helper(path: Path, loading: str, *, made: str) -> None:
    """A helper module that puts a module object it made in its own place in ``sys.modules``:
    an instance of a ``ModuleType`` subclass, or a ``ModuleType`` it fills in, without a spec,
    given its import's own (``made="spec"``) or given one that names no file
    (``made="nolocation"``)."""
    if made == "subclass":
        source = (
            "import sys\nimport types\n\n\n"
            f"class _Settings(types.ModuleType):\n    LOADING = {loading}\n\n\n"
            "sys.modules[__name__] = _Settings(__name__)\n"
        )
    else:
        spec = {
            "spec": "_settings.__spec__ = __spec__\n",
            "nolocation": "_settings.__spec__ = importlib.machinery.ModuleSpec(__name__, None)\n",
        }.get(made, "")
        source = (
            "import importlib.machinery\nimport sys\nimport types\n\n"
            f"_settings = types.ModuleType(__name__)\n{spec}_settings.LOADING = {loading}\n"
            "sys.modules[__name__] = _settings\n"
        )
    path.write_text(source, encoding="utf-8")


@pytest.mark.parametrize("made", ["subclass", "plain", "spec", "nolocation"])
def test_a_helper_that_puts_a_module_object_it_made_in_its_place_is_followed(
    app: App, tmp_path: Path, made: str
) -> None:
    """A helper that puts a module object it made in its own place, which has no file of its own:
    followed by the file its import ran, so an edit to it shows and a reload builds what it says
    now, a failed load takes such a helper it imported out, and its own file, loaded as a model
    file, is refused saying it puts a module object in its own place."""
    stem = _stem(tmp_path, f"modobj_{made}")
    helper = tmp_path / f"{stem}_basis.py"
    module_helper(helper, "1.0", made=made)
    path = object_model(tmp_path, f"{stem}_m", f"{stem}_basis")
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    assert death_outgo(app, model_id).endswith("* 1.0)")
    assert call(app, "GET", "/api/models")["files"][0]["helpers"] == [f"{stem}_basis"]
    assert call(app, "GET", "/api/models")["changed"] is False
    edit(helper, "LOADING = 1.0", "LOADING = 1.5")
    assert call(app, "GET", "/api/models")["changed"] is True
    call(app, "POST", "/api/models/reload")
    assert death_outgo(app, model_id).endswith("* 1.5)")
    module_helper(tmp_path / f"{stem}_other.py", "2.0", made=made)
    c = object_model(tmp_path, f"{stem}_c", f"{stem}_other", extra="undefined_name\n")
    assert refused(app, "POST", "/api/models/load", path=str(c)).status == 422
    assert f"{stem}_other" not in sys.modules
    refusal = refused(app, "POST", "/api/models/load", path=str(helper))
    assert "puts a module object in its own place in sys.modules" in refusal.message


def test_a_model_file_that_puts_an_object_in_its_place_is_not_taken_as_changed(
    app: App, tmp_path: Path
) -> None:
    """A model file whose import puts an object in its own place in ``sys.modules``: it loads
    with its models, and is not changed after the load or after a reload, nothing edited."""
    stem = _stem(tmp_path, "selfobj")
    path = write_model(tmp_path, stem, head="import sys\n")
    tail = "\n\nclass _Basis:\n    pass\n\n\nsys.modules[__name__] = _Basis()\n"
    path.write_text(path.read_text(encoding="utf-8") + tail, encoding="utf-8")
    loaded = call(app, "POST", "/api/models/load", path=str(path))
    assert loaded["file"]["models"] == [f"{stem}:Term"]
    assert call(app, "GET", "/api/models")["changed"] is False
    call(app, "POST", "/api/models/reload")
    assert call(app, "GET", "/api/models")["changed"] is False


def strict_helper(path: Path, loading: str) -> None:
    """A helper that puts an object in its own place whose every attribute is read from a dict:
    a ``KeyError`` for any name it lacks, ``__class__`` included."""
    path.write_text(
        "import sys\n\n\n"
        "class _Basis:\n"
        "    def __init__(self, values):\n"
        "        object.__setattr__(self, '_values', values)\n\n"
        "    def __getattribute__(self, name):\n"
        "        return object.__getattribute__(self, '_values')[name]\n\n\n"
        f"sys.modules[__name__] = _Basis({{'LOADING': {loading}}})\n",
        encoding="utf-8",
    )


def test_an_object_that_answers_no_attribute_at_all_does_not_stop_the_catalogue(
    app: App, tmp_path: Path
) -> None:
    """A helper's object that raises a ``KeyError`` for every attribute it lacks, ``__class__``
    included: told apart by its own class, so a file that imports it loads, lists it and
    builds, and one that fails after importing another is refused at its own line, that helper
    taken out. The objects are taken out at the end (see the test before the last)."""
    stem = _stem(tmp_path, "strictattr")
    try:
        strict_helper(tmp_path / f"{stem}_basis.py", "1.25")
        path = object_model(tmp_path, f"{stem}_m", f"{stem}_basis")
        model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
        assert call(app, "GET", "/api/models")["files"][0]["helpers"] == [f"{stem}_basis"]
        assert death_outgo(app, model_id).endswith("* 1.25)")
        strict_helper(tmp_path / f"{stem}_other.py", "2.0")
        c = object_model(tmp_path, f"{stem}_c", f"{stem}_other", extra="undefined_name\n")
        refusal = refused(app, "POST", "/api/models/load", path=str(c))
        assert refusal.status == 422
        loc = typing.cast(dict[str, Any], refusal.detail["loc"])
        assert Path(loc["file"]).resolve() == c.resolve()
        assert f"{stem}_other" not in sys.modules
    finally:
        for name in (f"{stem}_basis", f"{stem}_other"):
            sys.modules.pop(name, None)


def test_a_helper_loaded_as_a_model_file_says_its_object_is_in_its_place(
    app: App, tmp_path: Path
) -> None:
    """A helper whose object, in its own place in ``sys.modules``, raises ``SystemExit`` for a
    name it lacks, loaded as a model file after another file imported it: refused, saying the
    file puts that object in its place, not that it should be renamed."""
    stem = _stem(tmp_path, "exitobj")
    try:
        helper = tmp_path / f"{stem}_basis.py"
        helper.write_text(
            "import sys\n\n\n"
            "class _Basis:\n"
            "    LOADING = 1.0\n\n"
            "    def __getattr__(self, name):\n"
            "        raise SystemExit(name)\n\n\n"
            "sys.modules[__name__] = _Basis()\n",
            encoding="utf-8",
        )
        path = object_model(tmp_path, f"{stem}_m", f"{stem}_basis")
        call(app, "POST", "/api/models/load", path=str(path))
        refusal = refused(app, "POST", "/api/models/load", path=str(helper))
        assert refusal.status == 422
        assert "puts a _Basis object in its own place in sys.modules" in refusal.message
    finally:
        sys.modules.pop(f"{stem}_basis", None)


def test_a_build_whose_settings_cannot_be_read_back_warns_rather_than_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reading a build's settings back as a manifest records them runs code of the user's (an
    enum's own ``_missing_``), which may raise anything: the build that Python's ``build()``
    made is warned of, not failed."""

    def raising(cls: type, statics: object) -> object:
        raise SystemExit("missing")

    monkeypatch.setattr("halley.gui.app.rerun_misses", raising)
    warned = _rerun_warnings(object, {})
    assert len(warned) == 1
    assert "cannot be told" in warned[0]
    assert "SystemExit: missing" in warned[0]


# ---------------- round 13: module objects never read, keys and members by more of their types


def strict_module_helper(path: Path, loading: str) -> None:
    """A helper that puts a module object it made in its own place, whose ``__getattr__`` reads a
    dict: a ``KeyError`` for any name it lacks, ``__file__`` included."""
    path.write_text(
        "import sys\nimport types\n\n\n"
        "class _Settings(types.ModuleType):\n"
        f"    _values = {{'LOADING': {loading}}}\n\n"
        "    def __getattr__(self, name):\n"
        "        return type(self)._values[name]\n\n\n"
        "sys.modules[__name__] = _Settings(__name__)\n",
        encoding="utf-8",
    )


def test_a_module_object_whose_getattr_raises_is_followed_without_reading_it(
    app: App, tmp_path: Path
) -> None:
    """A helper's module object whose ``__getattr__`` raises a ``KeyError`` for any name it
    lacks: a file that imports it loads, again too, lists it, builds, and after an edit to it a
    reload builds what it says. The object is taken out at the end (see the strict tests)."""
    stem = _stem(tmp_path, "strictmod")
    try:
        helper = tmp_path / f"{stem}_basis.py"
        strict_module_helper(helper, "1.25")
        path = object_model(tmp_path, f"{stem}_m", f"{stem}_basis")
        model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
        assert call(app, "GET", "/api/models")["files"][0]["helpers"] == [f"{stem}_basis"]
        assert death_outgo(app, model_id).endswith("* 1.25)")
        edit(helper, "'LOADING': 1.25", "'LOADING': 1.5")
        assert call(app, "GET", "/api/models")["changed"] is True
        again = call(app, "POST", "/api/models/load", path=str(path))
        assert again["file"]["models"] == [model_id]
        assert death_outgo(app, model_id).endswith("* 1.5)")
        assert call(app, "GET", "/api/models")["changed"] is False
    finally:
        sys.modules.pop(f"{stem}_basis", None)


def test_a_model_file_another_runs_as_a_module_of_its_own_making_is_adopted(
    app: App, tmp_path: Path
) -> None:
    """A model file another model file runs into a module object it made, with the file as its
    ``__file__`` and no spec (the recipe before ``importlib.util``): loaded, it is adopted with
    its models, as a module of its own file."""
    stem = _stem(tmp_path, "execmod")
    b = write_model(tmp_path, f"{stem}_b")
    head = (
        "import sys\nimport types\n\n"
        f"_b = types.ModuleType('{stem}_b')\n"
        f"_b.__file__ = r'{b}'\n"
        f"sys.modules['{stem}_b'] = _b\n"
        f"exec(compile(open(r'{b}', encoding='utf-8').read(), r'{b}', 'exec'), _b.__dict__)\n"
    )
    a = write_model(tmp_path, f"{stem}_a", head=head)
    call(app, "POST", "/api/models/load", path=str(a))
    loaded = call(app, "POST", "/api/models/load", path=str(b))
    assert loaded["file"]["models"] == [f"{stem}_b:Term"]


def test_keys_holding_sequences_of_members_and_dates_read_back_from_a_manifest(
    app: App, tmp_path: Path
) -> None:
    """Dict keys that are, or hold, sequences of an enum's members, dates or a Literal's members,
    a bare Sequence key and a MutableMapping's keys: typed, each builds the Python call's IR,
    and a Python run with them reruns in a new session, their manifest's keys read back."""
    head = (
        "import collections.abc\nimport datetime as dt\nimport enum\n"
        "from collections.abc import MutableMapping, Sequence\nfrom typing import Literal\n\n\n"
        "class Basis(enum.Enum):\n    BEST = 'best estimate'\n    PRUDENT = 'prudent'\n"
    )
    body = (
        "    a: Static[dict[tuple[Sequence[Basis], int], float]] = {((Basis.BEST,), 1): 1.0}\n"
        "    b: Static[dict[Sequence[dt.date], float]] = {(dt.date(2027, 1, 1),): 1.0}\n"
        "    c: Static[dict[tuple[Literal[Basis.BEST, Basis.PRUDENT], int], float]] = "
        "{(Basis.BEST, 1): 1.0}\n"
        "    d: Static[dict[collections.abc.Sequence, float]] = {(1, 2): 1.0}\n"
        "    e: Static[MutableMapping[Basis, float]] = {Basis.BEST: 1.0}\n"
    )
    path = write_model(tmp_path, _stem(tmp_path, "seqmembers"), head=head, body=body)
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    entry = app.catalog.get(model_id)
    assert entry is not None
    basis = sys.modules[entry.cls.__module__].Basis
    typed = {
        "a": '{"[[\\"PRUDENT\\"], 2]": 2.0}',
        "b": '{"[\\"2028-02-03\\"]": 2.0}',
        "c": '{"[\\"PRUDENT\\", 2]": 2.0}',
        "d": '{"[1, 2]": 2.0}',
        "e": '{"PRUDENT": 2}',
    }
    values: dict[str, Any] = {
        "a": {((basis.PRUDENT,), 2): 2.0},
        "b": {(dt.date(2028, 2, 3),): 2.0},
        "c": {(basis.PRUDENT, 2): 2.0},
        "d": {(1, 2): 2.0},
        "e": {basis.PRUDENT: 2.0},
    }
    made = done(app, call(app, "POST", "/api/build", model=model_id, statics=typed))["build"]
    python = halley.build(entry.cls, **values)
    assert ir_of(app, made) == python.ir_hash
    assert not [w for w in app._build(made).warnings if "rerun" in w]
    made_run = run(
        python,
        model_points=tutorial_points(),
        tables={"mort": mort_table()},
        settings=RunSettings(valuation_date=VALUATION_DATE),
        run_dir=app.runs_root / "python-seqmembers",
    )
    other = App(tmp_path / "other-runs")
    try:
        again = done(other, call(other, "POST", "/api/run/rerun", dir=str(made_run.run_dir)))
        assert again["same_run_id"] is True
    finally:
        other.close()


def test_a_newtype_keeps_its_place_in_a_union_and_a_list_reads_as_far_as_it_fits(
    app: App, tmp_path: Path
) -> None:
    """A NewType read as its supertype, keeping its declared place before an int or str enum, and
    a list that fits its list or tuple option only in part beside a set option: each builds the
    Python call's IR."""
    head = (
        "import enum\nfrom typing import NewType\n\n"
        "Rate = NewType('Rate', float)\nName = NewType('Name', str)\n\n\n"
        "class Lvl(enum.IntEnum):\n    LOW = 1\n    HIGH = 2\n\n\n"
        "class Code(enum.StrEnum):\n    A = 'a'\n    B = 'b'\n\n\n"
        "class Sex(enum.Enum):\n    M = 'male'\n    F = 'female'\n"
    )
    body = (
        "    rate: Static[Rate | Lvl] = Lvl.LOW\n"
        "    name: Static[Name | Code] = Code.A\n"
        "    alone: Static[Rate] = 1.0\n"
        "    mix: Static[set[int] | list[Sex]] = [Sex.F]\n"
        "    tup: Static[frozenset[int] | tuple[Sex, ...]] = (Sex.F,)\n"
    )
    path = write_model(tmp_path, _stem(tmp_path, "newtype"), head=head, body=body)
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    entry = app.catalog.get(model_id)
    assert entry is not None
    sex = sys.modules[entry.cls.__module__].Sex
    statics = {p["name"]: p for p in call(app, "GET", "/api/model", {"id": model_id})["statics"]}
    assert statics["alone"]["input"] == "float"
    typed = {"rate": "2", "name": '"b"', "alone": "2", "mix": '["M", 5]', "tup": '["M", 5]'}
    made = done(app, call(app, "POST", "/api/build", model=model_id, statics=typed))["build"]
    want = halley.build(entry.cls, rate=2.0, name="b", alone=2.0, mix=[sex.M, 5], tup=(sex.M, 5))
    assert ir_of(app, made) == want.ir_hash


def test_an_enum_s_own_code_never_stands_for_a_value_the_python_call_has(
    app: App, tmp_path: Path
) -> None:
    """Enums whose own code does what the GUI cannot foresee: a ``_missing_`` that raises
    ``SystemExit``, one that gives a default member for anything, a flag's that refuses only 0,
    and a text that looks like a member's repr holding a list: each typed value builds what the
    Python call builds, and no reading of them fails the build."""
    head = (
        "import enum\n\n\n"
        "class Kind(enum.Enum):\n"
        "    A = 'a'\n"
        "    B = 'b'\n\n"
        "    @classmethod\n"
        "    def _missing_(cls, value):\n"
        "        raise SystemExit(f'no Kind {value!r}')\n\n\n"
        "class Other(enum.Enum):\n"
        "    A = 'a'\n"
        "    OTHER = 'other'\n\n"
        "    @classmethod\n"
        "    def _missing_(cls, value):\n"
        "        return cls.OTHER\n\n\n"
        "class Perm(enum.Flag):\n"
        "    R = 1\n"
        "    W = 2\n\n"
        "    @classmethod\n"
        "    def _missing_(cls, value):\n"
        "        if value == 0:\n"
        "            raise ValueError('no empty Perm')\n"
        "        return super()._missing_(value)\n"
    )
    body = (
        "    k: Static[Kind | float] = Kind.A\n"
        "    o: Static[Other | float] = Other.A\n"
        "    perm: Static[Perm] = Perm.R\n"
        "    texts: Static[list[Kind | str]] = [Kind.A]\n"
    )
    path = write_model(tmp_path, _stem(tmp_path, "enumcode"), head=head, body=body)
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    entry = app.catalog.get(model_id)
    assert entry is not None
    m = sys.modules[entry.cls.__module__]
    typed = {"k": "2.5", "o": "2.5", "perm": "R|W", "texts": '["B", "<X.Y: {[1]: 2}>"]'}
    made = done(app, call(app, "POST", "/api/build", model=model_id, statics=typed))["build"]
    want = halley.build(
        entry.cls, k=2.5, o=2.5, perm=m.Perm.R | m.Perm.W, texts=[m.Kind.B, "<X.Y: {[1]: 2}>"]
    )
    assert ir_of(app, made) == want.ir_hash
    assert not [w for w in app._build(made).warnings if "cannot be told" in w]


def test_a_model_whose_module_is_an_object_that_raises_is_still_shown(
    app: App, tmp_path: Path
) -> None:
    """A model file that puts in its own place an object whose every attribute raises
    ``SystemExit``: it loads, its page answers, and its build fails as Python's does, as a
    failed job. The object is taken out at the end."""
    stem = _stem(tmp_path, "selfexit")
    try:
        path = write_model(tmp_path, stem, head="import sys\n")
        tail = (
            "\n\nclass _Strict:\n"
            "    def __getattribute__(self, name):\n"
            "        raise SystemExit(name)\n\n\n"
            "sys.modules[__name__] = _Strict()\n"
        )
        path.write_text(path.read_text(encoding="utf-8") + tail, encoding="utf-8")
        model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
        shown = call(app, "GET", "/api/model", {"id": model_id})
        assert shown["valuation_date"] is None
        job = finish(app, call(app, "POST", "/api/build", model=model_id))
        assert job["state"] == "failed"
    finally:
        sys.modules.pop(stem, None)


# ---------------- round 14: what holds a name, a union's parts, members by name, one import


def test_a_union_leaves_a_well_typed_list_to_the_option_that_takes_it(
    app: App, tmp_path: Path
) -> None:
    """A list or dict well typed for a union's option the GUI takes as JSON gives it (an
    ``Iterable``), or for a ``type`` alias, read by its value, beside a tuple or dict option that
    reads none of it: each builds the Python call's IR, the value as given; and a setting
    declared a ``type`` alias is read by its value (a list of floats: 2 is 2.0)."""
    head = (
        "import enum\nfrom collections.abc import Iterable\n\n"
        "type Weights = list[float]\ntype Counts = dict[str, int]\n\n\n"
        "class Sex(enum.Enum):\n    M = 'male'\n    F = 'female'\n"
    )
    body = (
        "    pair: Static[tuple[float, float] | Weights] = (1.0, 2.0)\n"
        "    pit: Static[tuple[float, float] | Iterable[float]] = (1.0, 2.0)\n"
        "    wsex: Static[Weights | tuple[Sex, ...]] = [1.0]\n"
        "    dsc: Static[dict[Sex, float] | Counts] = {'a': 1}\n"
        "    rates: Static[Weights] = [1.0]\n"
    )
    path = write_model(tmp_path, _stem(tmp_path, "aliasunion"), head=head, body=body)
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    entry = app.catalog.get(model_id)
    assert entry is not None
    typed = {
        "pair": "[1.5, 2.5, 3.5]",
        "pit": "[1.5, 2.5, 3.5]",
        "wsex": "[1.5, 2.5]",
        "dsc": '{"a": 2}',
        "rates": "[2, 3]",
    }
    made = done(app, call(app, "POST", "/api/build", model=model_id, statics=typed))["build"]
    want = halley.build(
        entry.cls,
        pair=[1.5, 2.5, 3.5],
        pit=[1.5, 2.5, 3.5],
        wsex=[1.5, 2.5],
        dsc={"a": 2},
        rates=[2.0, 3.0],
    )
    assert ir_of(app, made) == want.ir_hash


def test_a_file_named_like_a_built_in_module_is_not_taken_for_it(app: App, tmp_path: Path) -> None:
    """A file of the user's named like a module built into Python, which a model imports: Python
    imports its own, never the file, so the file is no helper, an edit to it changes nothing, and
    loading it is refused as a name Python itself holds."""
    candidates = ("xxsubtype", "cmath", "array", "mmap", "audioop")
    built_in = next(
        (n for n in candidates if n in sys.builtin_module_names and n not in sys.modules), None
    )
    if built_in is None:
        pytest.skip("no module built into this Python is left to import")
    folder = tmp_path / "shadow"
    folder.mkdir()
    shadow = folder / f"{built_in}.py"
    shadow.write_text("X = 1\n", encoding="utf-8")
    path = write_model(folder, _stem(tmp_path, "shadow"), head=f"import {built_in}\n")
    call(app, "POST", "/api/models/load", path=str(path))
    assert call(app, "GET", "/api/models")["files"][0]["helpers"] == []
    edit(shadow, "X = 1", "X = 2")
    assert call(app, "GET", "/api/models")["changed"] is False
    refusal = refused(app, "POST", "/api/models/load", path=str(shadow))
    assert "from Python itself" in refusal.message


def test_an_enum_s_own_repr_or_label_never_takes_a_value_meant_for_another_option(
    app: App, tmp_path: Path
) -> None:
    """An enum whose ``__repr__`` raises for a member no setting uses, and one whose ``__str__``
    gives a label: a number, or the label's text, typed into a union with them builds what the
    Python call builds; and a Python run of the first reruns."""
    head = (
        "import enum\n\n\n"
        "class RE(enum.Enum):\n"
        "    B = 'b'\n"
        "    A = 'a'\n\n"
        "    def __repr__(self):\n"
        "        if self.name == 'B':\n"
        "            raise SystemExit('no repr for B')\n"
        "        return super().__repr__()\n\n\n"
        "class Basis(enum.Enum):\n"
        "    BEST = 'be'\n\n"
        "    def __str__(self):\n"
        "        return 'Best estimate'\n"
    )
    body = "    rs: Static[RE | float] = 1.5\n    b: Static[Basis | str] = Basis.BEST\n"
    path = write_model(tmp_path, _stem(tmp_path, "enumrepr"), head=head, body=body)
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    entry = app.catalog.get(model_id)
    assert entry is not None
    typed = {"rs": "2.5", "b": '"Best estimate"'}
    made = done(app, call(app, "POST", "/api/build", model=model_id, statics=typed))["build"]
    want = halley.build(entry.cls, rs=2.5, b="Best estimate")
    assert ir_of(app, made) == want.ir_hash
    assert not [w for w in app._build(made).warnings if "cannot be told" in w]
    made_run = run(
        halley.build(entry.cls, rs=2.5),
        model_points=tutorial_points(),
        tables={"mort": mort_table()},
        settings=RunSettings(valuation_date=VALUATION_DATE),
        run_dir=app.runs_root / "python-enumrepr",
    )
    assert done(app, call(app, "POST", "/api/run/rerun", dir=str(made_run.run_dir)))["same_run_id"]


def test_format_quietly_runs_a_suggestion_s_import_no_more_than_python_does(
    tmp_path: Path,
) -> None:
    """An ``ImportError`` of the user's own class whose ``name_from`` cannot be hushed, naming a
    module not imported whose top level raises ``SystemExit``: Python's writing imports it once,
    and so does format_quietly."""
    gone = _stem(tmp_path, "gone")
    (tmp_path / f"{gone}.py").write_text(
        "import builtins\n\n"
        "builtins.__dict__.setdefault('_halley_gone_runs', []).append(1)\n"
        "raise SystemExit('gone')\n",
        encoding="utf-8",
    )

    class OddError(ImportError):
        @property
        def name_from(self) -> object:  # pyright: ignore[reportIncompatibleVariableOverride]
            return "LOADING"

    sys.path.insert(0, str(tmp_path))
    try:
        try:
            raise OddError("no LOADING", name=gone)
        except OddError as exc:
            error = exc
        written = format_quietly(error)
        assert written.rstrip().endswith("no LOADING")
        assert builtins.__dict__.get("_halley_gone_runs") == [1]
    finally:
        sys.path.remove(str(tmp_path))
        builtins.__dict__.pop("_halley_gone_runs", None)
        sys.modules.pop(gone, None)


def test_a_replaced_class_check_holds_whatever_its_module_raises() -> None:
    """Whether a build's class was replaced, read through a module whose own ``__getattr__``
    raises ``SystemExit`` for a name it lacks: not replaced, as for a name the module does not
    reach, rather than an exception out of the run's route."""
    module = types.ModuleType("gui_replaced_raises")

    def lacking(name: str) -> object:
        raise SystemExit(name)

    module.__getattr__ = lacking  # pyright: ignore[reportAttributeAccessIssue]
    sys.modules[module.__name__] = module
    try:
        made = type("Gone", (), {"__module__": module.__name__})
        assert _replaced(made) is False
    finally:
        sys.modules.pop(module.__name__, None)


def test_a_model_whose_module_class_reads_only_its_dict_lists_its_models(
    app: App, tmp_path: Path
) -> None:
    """A model file that gives its own module a class whose ``__getattribute__`` reads only the
    module's namespace (a ``KeyError`` for ``__dict__`` and any name it lacks): its models are read
    from the namespace as Python keeps it, so it loads and lists them, and its page answers. The
    module is taken out at the end."""
    stem = _stem(tmp_path, "strictclass")
    try:
        path = write_model(tmp_path, stem, head="import sys\nimport types\n")
        tail = (
            "\n\nclass _Strict(types.ModuleType):\n"
            "    def __getattribute__(self, name):\n"
            "        return object.__getattribute__(self, '__dict__')[name]\n\n\n"
            "sys.modules[__name__].__class__ = _Strict\n"
        )
        path.write_text(path.read_text(encoding="utf-8") + tail, encoding="utf-8")
        loaded = call(app, "POST", "/api/models/load", path=str(path))
        assert loaded["file"]["models"] == [f"{stem}:Term"]
        assert call(app, "GET", "/api/model", {"id": f"{stem}:Term"})["name"] == "Term"
    finally:
        sys.modules.pop(stem, None)


def test_a_helper_that_puts_another_module_in_its_place_is_followed_by_its_own_file(
    app: App, tmp_path: Path
) -> None:
    """A helper that puts another module of the user's in its own place (it chooses a basis):
    followed by its own file as well as the module it chose, so an edit that chooses another
    shows as a change and a reload builds it, and its own file, loaded, is refused saying it puts
    a module object in its own place."""
    stem = _stem(tmp_path, "alias")
    (tmp_path / f"{stem}_prudent.py").write_text("LOADING = 1.5\n", encoding="utf-8")
    (tmp_path / f"{stem}_best.py").write_text("LOADING = 1.0\n", encoding="utf-8")
    helper = tmp_path / f"{stem}_basis.py"
    helper.write_text(
        f"import importlib\nimport sys\n\nsys.modules[__name__] = "
        f"importlib.import_module('{stem}_prudent')\n",
        encoding="utf-8",
    )
    path = object_model(tmp_path, f"{stem}_m", f"{stem}_basis")
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    assert death_outgo(app, model_id).endswith("* 1.5)")
    edit(helper, f"'{stem}_prudent'", f"'{stem}_best'")
    assert call(app, "GET", "/api/models")["changed"] is True
    call(app, "POST", "/api/models/reload")
    assert death_outgo(app, model_id).endswith("* 1.0)")
    refusal = refused(app, "POST", "/api/models/load", path=str(helper))
    assert "puts a module object in its own place in sys.modules" in refusal.message


@pytest.mark.parametrize("made", ["bare", "ran", "spec"])
def test_a_model_file_that_puts_an_empty_module_object_in_its_place_is_refused(
    app: App, tmp_path: Path, made: str
) -> None:
    """A model file whose import puts in its own place a module object holding its ``__file__``
    but none of the models it defines (code run in it or not, given the import's spec or not),
    imported first by another file: loaded, it is refused saying so, not listed without its
    models."""
    stem = _stem(tmp_path, f"emptymod_{made}")
    m = write_model(tmp_path, f"{stem}_m", head="import sys\nimport types\n")
    more = {
        "ran": "exec('LOADING = 1.0', _copy.__dict__)\n",
        "spec": "_copy.__spec__ = __spec__\nexec('LOADING = 1.0', _copy.__dict__)\n",
    }.get(made, "")
    tail = (
        "\n\n_copy = types.ModuleType(__name__)\n"
        f"_copy.__file__ = __file__\n{more}"
        "sys.modules[__name__] = _copy\n"
    )
    m.write_text(m.read_text(encoding="utf-8") + tail, encoding="utf-8")
    a = write_model(tmp_path, f"{stem}_a", head=f"import {stem}_m\n")
    call(app, "POST", "/api/models/load", path=str(a))
    refusal = refused(app, "POST", "/api/models/load", path=str(m))
    assert "puts a module object in its own place in sys.modules" in refusal.message


# ---------------- round 15: modules by their own files, aliases, members by name


def test_a_helper_that_names_itself_otherwise_is_followed_by_its_own_file(
    app: App, tmp_path: Path
) -> None:
    """A helper that sets its own ``__name__``, imported before the GUI loads the model that
    imports it: still the module of its own file, so it is the model's helper, an edit to it
    shows and a reload builds what it says now."""
    stem = _stem(tmp_path, "setname")
    helper = tmp_path / f"{stem}_basis.py"
    helper.write_text("LOADING = 1.0\n__name__ = 'elsewhere'\n", encoding="utf-8")
    sys.path.insert(0, str(tmp_path))
    try:
        importlib.import_module(f"{stem}_basis")
        path = object_model(tmp_path, f"{stem}_m", f"{stem}_basis")
        model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
        assert call(app, "GET", "/api/models")["files"][0]["helpers"] == [f"{stem}_basis"]
        call(app, "POST", "/api/models/reload")  # imported before the GUI looked: a reload
        assert call(app, "GET", "/api/models")["changed"] is False
        edit(helper, "LOADING = 1.0", "LOADING = 1.5")
        assert call(app, "GET", "/api/models")["changed"] is True
        call(app, "POST", "/api/models/reload")
        assert death_outgo(app, model_id).endswith("* 1.5)")
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop(f"{stem}_basis", None)


def test_a_helper_that_runs_another_file_as_itself_is_followed_by_both(
    app: App, tmp_path: Path
) -> None:
    """A helper whose import runs another file of the user's as the module of its own name
    (importlib's recipe) and puts that in its place: an edit to either file shows, a reload
    builds what they say now, and the helper's own file, loaded, is refused saying what it puts
    in its place."""
    stem = _stem(tmp_path, "selfexec")
    (tmp_path / f"{stem}_prudent.py").write_text("LOADING = 1.5\n", encoding="utf-8")
    best = tmp_path / f"{stem}_best.py"
    best.write_text("LOADING = 1.0\n", encoding="utf-8")
    helper = tmp_path / f"{stem}_basis.py"
    helper.write_text(
        "import importlib.util\nimport sys\nfrom pathlib import Path\n\n"
        "_spec = importlib.util.spec_from_file_location(\n"
        f"    __name__, Path(__file__).with_name('{stem}_prudent.py')\n"
        ")\n"
        "_module = importlib.util.module_from_spec(_spec)\n"
        "_spec.loader.exec_module(_module)\n"
        "sys.modules[__name__] = _module\n",
        encoding="utf-8",
    )
    path = object_model(tmp_path, f"{stem}_m", f"{stem}_basis")
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    assert death_outgo(app, model_id).endswith("* 1.5)")
    assert call(app, "GET", "/api/models")["changed"] is False
    edit(helper, f"'{stem}_prudent.py'", f"'{stem}_best.py'")
    assert call(app, "GET", "/api/models")["changed"] is True
    call(app, "POST", "/api/models/reload")
    assert death_outgo(app, model_id).endswith("* 1.0)")
    edit(best, "LOADING = 1.0", "LOADING = 2.0")
    assert call(app, "GET", "/api/models")["changed"] is True
    call(app, "POST", "/api/models/reload")
    assert death_outgo(app, model_id).endswith("* 2.0)")
    refusal = refused(app, "POST", "/api/models/load", path=str(helper))
    assert f"puts a module of another file, {best}, in its own place" in refusal.message


def test_a_file_named_like_a_namespace_package_gets_both_remedies(app: App, tmp_path: Path) -> None:
    """A model file named like a namespace package already imported (a folder without an
    __init__.py): refused with both remedies, not as a name Python itself holds."""
    stem = _stem(tmp_path, "nsname")
    (tmp_path / stem).mkdir()
    sys.path.insert(0, str(tmp_path))
    try:
        importlib.import_module(stem)
    finally:
        sys.path.remove(str(tmp_path))
    try:
        folder = tmp_path / "models"
        folder.mkdir()
        refusal = refused(app, "POST", "/api/models/load", path=str(write_model(folder, stem)))
        assert "from Python itself" not in refusal.message
        assert f"otherwise rename {stem}.py" in refusal.message
    finally:
        sys.modules.pop(stem, None)


def test_a_literal_of_members_whose_repr_or_str_raises_builds_and_reruns(
    app: App, tmp_path: Path
) -> None:
    """A ``Literal`` of an enum's members, one of whose ``__repr__`` raises and, in another enum,
    whose ``__str__`` raises, in settings that use other members: the model's page answers, a
    build typed into it gives the Python call's IR without a warning, and a Python run reruns."""
    head = (
        "import enum\nfrom typing import Literal\n\n\n"
        "class RB(enum.Enum):\n"
        "    B = 'b'\n"
        "    A = 'a'\n\n"
        "    def __repr__(self):\n"
        "        if self.name == 'B':\n"
        "            raise KeyError('no repr for B')\n"
        "        return super().__repr__()\n\n\n"
        "class SB(enum.Enum):\n"
        "    B = 'b'\n"
        "    A = 'a'\n\n"
        "    def __str__(self):\n"
        "        if self.name == 'B':\n"
        "            raise KeyError('no str for B')\n"
        "        return super().__str__()\n"
    )
    body = (
        "    lit: Static[Literal[RB.A, RB.B] | float] = 1.5\n"
        "    dkl: Static[dict[Literal[SB.B, SB.A], float]] = {SB.A: 1.0}\n"
    )
    path = write_model(tmp_path, _stem(tmp_path, "litrepr"), head=head, body=body)
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    entry = app.catalog.get(model_id)
    assert entry is not None
    shown = call(app, "GET", "/api/model", {"id": model_id})
    assert "Literal[<RB.A: 'a'>, RB.B] | float" in json.dumps(shown)
    made = done(app, call(app, "POST", "/api/build", model=model_id, statics={"lit": "2.5"}))
    assert ir_of(app, made["build"]) == halley.build(entry.cls, lit=2.5).ir_hash
    assert app._build(made["build"]).warnings == []
    made_run = run(
        halley.build(entry.cls),
        model_points=tutorial_points(),
        tables={"mort": mort_table()},
        settings=RunSettings(valuation_date=VALUATION_DATE),
        run_dir=app.runs_root / "python-litrepr",
    )
    assert done(app, call(app, "POST", "/api/run/rerun", dir=str(made_run.run_dir)))["same_run_id"]


def test_members_are_typed_as_python_writes_them_never_by_a_label(app: App, tmp_path: Path) -> None:
    """A plain enum whose ``__repr__`` gives a label: the label typed into a union with ``str``,
    or with a ``Literal`` of it, is the text it is; a member typed as ``str()`` writes it
    (``Sex.M``, ``Perm.R|W``) is that member, in a ``Literal`` and in a flag's field: each builds
    the Python call's IR."""
    head = (
        "import enum\nfrom typing import Literal\n\n\n"
        "class RL(enum.Enum):\n"
        "    A = 1\n\n"
        "    def __repr__(self):\n"
        "        return 'Label A'\n\n\n"
        "class Sex(enum.Enum):\n    M = 'male'\n    F = 'female'\n\n\n"
        "class Perm(enum.Flag):\n    R = 1\n    W = 2\n"
    )
    body = (
        "    rl: Static[RL | str] = 'y'\n"
        "    rll: Static[Literal[RL.A] | str] = 'y'\n"
        "    sx: Static[Literal[Sex.F, Sex.M] | str] = 'y'\n"
        "    lk: Static[dict[Literal[Sex.F, Sex.M], float]] = {Sex.F: 1.0}\n"
        "    perm: Static[Perm] = Perm.R\n"
    )
    path = write_model(tmp_path, _stem(tmp_path, "labels"), head=head, body=body)
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    entry = app.catalog.get(model_id)
    assert entry is not None
    module = sys.modules[entry.cls.__module__]
    typed = {
        "rl": '"Label A"',
        "rll": '"Label A"',
        "sx": '"Sex.M"',
        "lk": '{"Sex.M": 2.5, "female": 1.5}',
        "perm": "Perm.R|W",
    }
    made = done(app, call(app, "POST", "/api/build", model=model_id, statics=typed))["build"]
    want = halley.build(
        entry.cls,
        rl="Label A",
        rll="Label A",
        sx=module.Sex.M,
        lk={module.Sex.M: 2.5, module.Sex.F: 1.5},
        perm=module.Perm.R | module.Perm.W,
    )
    assert ir_of(app, made) == want.ir_hash


def test_a_build_with_a_member_its_class_writes_like_another_is_warned(
    app: App, tmp_path: Path
) -> None:
    """An enum whose ``__repr__`` writes every member alike, which a formula tests: a build with
    a member a manifest's record does not give back is warned that a run of it cannot be rerun;
    one with the member a rerun reads back is not."""
    head = (
        "import enum\n\n\n"
        "class RX(enum.Enum):\n"
        "    B = 2\n"
        "    A = 1\n\n"
        "    def __repr__(self):\n"
        "        return 'X'\n"
    )
    body = "    basis: Static[RX] = RX.B\n"
    path = write_model(tmp_path, _stem(tmp_path, "samerepr"), head=head, body=body)
    loading = "(1.5 if self.basis is RX.A else 1.0)"
    edit(path, "* self.mp.sum_assured\n", f"* self.mp.sum_assured * {loading}\n")
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    made = done(app, call(app, "POST", "/api/build", model=model_id, statics={"basis": "A"}))
    warned = app._build(made["build"]).warnings
    assert any("unless the member makes no difference to the model" in w for w in warned)
    made = done(app, call(app, "POST", "/api/build", model=model_id, statics={"basis": "B"}))
    assert app._build(made["build"]).warnings == []


def test_type_aliases_are_read_by_their_values_generic_or_lazy(app: App, tmp_path: Path) -> None:
    """Settings declared with generic ``type`` aliases (their arguments given in their places),
    an ``Annotated`` alias, and an alias naming what its module does not define: the model's page
    answers, and a build typed into the first three gives the Python call's IR."""
    head = (
        "import enum\nfrom typing import Annotated\n\n"
        "type Pair[T] = tuple[T, T]\n"
        "type Swap[A, B] = dict[B, A]\n"
        "type Rates = Annotated[list[float], 'per annum']\n"
        "type Lazy = Undefined\n"
        "type Id[T] = T\n"
        "type Row[*Ts] = tuple[*Ts]\n\n\n"
        "def halt():\n    raise SystemExit('no value')\n\n\n"
        "type Halt = halt()\n\n\n"
        "class Sex(enum.Enum):\n    M = 'male'\n    F = 'female'\n"
    )
    body = (
        "    pair: Static[Pair[float]] = (1.0, 2.0)\n"
        "    swap: Static[Swap[float, Sex]] = {Sex.M: 1.0}\n"
        "    rates: Static[Rates] = [1.0]\n"
        "    lazy: Static[Lazy] = 1\n"
        "    halted: Static[Halt] = 1\n"
        "    idf: Static[Id[float]] = 1.0\n"
        "    ids: Static[Id[Sex]] = Sex.F\n"
        "    row: Static[Row[float, Sex]] = (1.0, Sex.F)\n"
    )
    path = write_model(tmp_path, _stem(tmp_path, "generic"), head=head, body=body)
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    entry = app.catalog.get(model_id)
    assert entry is not None
    shown = call(app, "GET", "/api/model", {"id": model_id})
    assert '"lazy"' in json.dumps(shown)
    assert '"halted"' in json.dumps(shown)
    module = sys.modules[entry.cls.__module__]
    typed = {
        "pair": "[1.5, 2]",
        "swap": '{"F": 2}',
        "rates": "[2, 3]",
        "idf": "2",
        "ids": "M",
        "row": '[2, "M"]',
    }
    made = done(app, call(app, "POST", "/api/build", model=model_id, statics=typed))["build"]
    want = halley.build(
        entry.cls,
        pair=(1.5, 2.0),
        swap={module.Sex.F: 2.0},
        rates=[2.0, 3.0],
        idf=2.0,
        ids=module.Sex.M,
        row=(2.0, module.Sex.M),
    )
    assert ir_of(app, made) == want.ir_hash


def test_a_union_gives_a_list_to_an_option_that_holds_it_or_reads_its_parts(
    app: App, tmp_path: Path
) -> None:
    """A list typed into a union whose tuple option reads only part of it, beside an
    ``Iterable`` that holds it as it is; and a list of dicts whose keys fit only in part, beside
    a set option, which no JSON value is: each builds the Python call's IR."""
    head = (
        "import enum\nfrom collections.abc import Iterable\n\n"
        "type Its = Iterable[float]\n\n\n"
        "class Sex(enum.Enum):\n    M = 'male'\n    F = 'female'\n"
    )
    body = (
        "    pti: Static[tuple[float, str] | Iterable[float]] = (1.0, 'a')\n"
        "    pal: Static[tuple[float, str] | Its] = (1.0, 'a')\n"
        "    q5: Static[set[int] | list[dict[Sex, float]]] = []\n"
        "    q6: Static[set[int] | list[dict[Sex, float] | None]] = []\n"
    )
    path = write_model(tmp_path, _stem(tmp_path, "holds"), head=head, body=body)
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    entry = app.catalog.get(model_id)
    assert entry is not None
    module = sys.modules[entry.cls.__module__]
    typed = {
        "pti": "[1.5, 2.5]",
        "pal": "[1.5, 2.5]",
        "q5": '[{"M": 1, "X": 2}]',
        "q6": '[{"M": 1, "X": 2}]',
    }
    made = done(app, call(app, "POST", "/api/build", model=model_id, statics=typed))["build"]
    want = halley.build(
        entry.cls,
        pti=[1.5, 2.5],
        pal=[1.5, 2.5],
        q5=[{module.Sex.M: 1.0, "X": 2.0}],
        q6=[{module.Sex.M: 1.0, "X": 2.0}],
    )
    assert ir_of(app, made) == want.ir_hash


def test_a_reload_refuses_a_file_an_earlier_one_imports_that_puts_another_module_in_its_place(
    app: App, tmp_path: Path
) -> None:
    """Model file a imports model file b, both loaded; b, edited to put another module in its own
    place, is refused at a reload, saying so, as a load after such an import is, rather than run
    a second time: its top level runs once, as Python's import runs it, and the files and their
    models are as they were."""
    stem = _stem(tmp_path, "reloadself")
    (tmp_path / f"{stem}_impl.py").write_text("LOADING = 1.0\n", encoding="utf-8")
    b = write_model(tmp_path, f"{stem}_b", head="import builtins\n")
    a = write_model(tmp_path, f"{stem}_a", head=f"import {stem}_b\n")
    call(app, "POST", "/api/models/load", path=str(a))
    call(app, "POST", "/api/models/load", path=str(b))
    before = [(f.module, f.models, sorted(f.helpers)) for f in app.catalog.files()]
    tail = (
        "\n\nbuiltins.__dict__.setdefault('_halley_reloadself_runs', []).append(1)\n"
        "import importlib  # noqa: E402\nimport sys  # noqa: E402\n\n"
        f"sys.modules[__name__] = importlib.import_module('{stem}_impl')\n"
    )
    edit(b, "* V_M\n", "* V_M\n" + tail)
    try:
        refusal = refused(app, "POST", "/api/models/reload")
        assert "puts a module object in its own place in sys.modules" in refusal.message
        assert builtins.__dict__.get("_halley_reloadself_runs") == [1]
        assert [(f.module, f.models, sorted(f.helpers)) for f in app.catalog.files()] == before
    finally:
        builtins.__dict__.pop("_halley_reloadself_runs", None)


# ---------------- round 16: built-in modules by name, a reload's adoption, aliases' shapes


def test_a_helper_that_puts_a_built_in_module_in_its_place_is_followed_by_its_own_file(
    app: App, tmp_path: Path
) -> None:
    """A helper whose import puts a module built into Python (math) in its own place: followed
    by its own file, so an edit to it that chooses another module shows and a reload builds it,
    and its own file, loaded, is refused saying what it puts in its place."""
    stem = _stem(tmp_path, "builtinalias")
    (tmp_path / f"{stem}_best.py").write_text("LOADING = 2.0\n", encoding="utf-8")
    helper = tmp_path / f"{stem}_basis.py"
    helper.write_text(
        "import importlib\nimport math\nimport sys\n\nsys.modules[__name__] = math\n",
        encoding="utf-8",
    )
    head = f"import {stem}_basis\n\nLOADING = getattr({stem}_basis, 'LOADING', 1.0)\n"
    path = write_model(tmp_path, f"{stem}_m", head=head)
    edit(path, "* self.mp.sum_assured\n", "* self.mp.sum_assured * LOADING\n")
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    assert death_outgo(app, model_id).endswith("* 1.0)")
    assert call(app, "GET", "/api/models")["files"][0]["helpers"] == [f"{stem}_basis"]
    edit(helper, "= math", f"= importlib.import_module('{stem}_best')")
    assert call(app, "GET", "/api/models")["changed"] is True
    call(app, "POST", "/api/models/reload")
    assert death_outgo(app, model_id).endswith("* 2.0)")
    edit(helper, f"= importlib.import_module('{stem}_best')", "= math")
    call(app, "POST", "/api/models/reload")
    refusal = refused(app, "POST", "/api/models/load", path=str(helper))
    assert "puts a module object in its own place in sys.modules" in refusal.message


def test_a_failed_reload_leaves_an_edit_to_a_helper_that_runs_another_file_showing(
    app: App, tmp_path: Path
) -> None:
    """A helper that runs another file as itself, edited to run a third; a reload that fails at
    another loaded file (whose top level raises while a flag file is there) imports the helper
    again on its way: once the flag goes, the edit still shows as a change, and a reload builds
    it."""
    stem = _stem(tmp_path, "failedself")
    (tmp_path / f"{stem}_prudent.py").write_text("LOADING = 1.5\n", encoding="utf-8")
    (tmp_path / f"{stem}_best.py").write_text("LOADING = 3.0\n", encoding="utf-8")
    helper = tmp_path / f"{stem}_basis.py"
    helper.write_text(
        "import importlib.util\nimport sys\nfrom pathlib import Path\n\n"
        "_spec = importlib.util.spec_from_file_location(\n"
        f"    __name__, Path(__file__).with_name('{stem}_prudent.py')\n"
        ")\n"
        "_module = importlib.util.module_from_spec(_spec)\n"
        "_spec.loader.exec_module(_module)\n"
        "sys.modules[__name__] = _module\n",
        encoding="utf-8",
    )
    path = object_model(tmp_path, f"{stem}_m", f"{stem}_basis")
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    flag = tmp_path / f"{stem}_flag.txt"
    guard = (
        "from pathlib import Path\n\n"
        f"if Path(__file__).with_name('{flag.name}').exists():\n"
        "    raise ValueError('the flag is up')\n"
    )
    other = write_model(tmp_path, f"{stem}_z", head=guard)
    call(app, "POST", "/api/models/load", path=str(other))
    assert death_outgo(app, model_id).endswith("* 1.5)")
    edit(helper, f"'{stem}_prudent.py'", f"'{stem}_best.py'")
    assert call(app, "GET", "/api/models")["changed"] is True
    flag.write_text("up\n", encoding="utf-8")
    assert refused(app, "POST", "/api/models/reload").status == 422
    flag.unlink()
    assert call(app, "GET", "/api/models")["changed"] is True
    call(app, "POST", "/api/models/reload")
    assert death_outgo(app, model_id).endswith("* 3.0)")


@pytest.mark.parametrize("made", ["recipe", "object"])
def test_a_reload_refuses_a_file_an_earlier_one_imports_that_puts_its_own_like_module_in_place(
    app: App, tmp_path: Path, made: str
) -> None:
    """Model file m, loaded, edited to put in its own place a module of another file run as m
    (importlib's recipe), or a module object holding m's ``__file__`` and spec but none of its
    models; then a, loaded too, is edited to import m first: the reload refuses m, saying so,
    and the files and their models are as they were."""
    stem = _stem(tmp_path, f"reloadown_{made}")
    (tmp_path / f"{stem}_impl.py").write_text("LOADING = 1.0\n", encoding="utf-8")
    m = write_model(tmp_path, f"{stem}_m", head="import importlib.util\nimport sys\nimport types\n")
    a = write_model(tmp_path, f"{stem}_a")
    call(app, "POST", "/api/models/load", path=str(a))
    call(app, "POST", "/api/models/load", path=str(m))
    before = [(f.module, f.models, sorted(f.helpers)) for f in app.catalog.files()]
    if made == "recipe":
        tail = (
            "\n\n_spec = importlib.util.spec_from_file_location(\n"
            f"    __name__, __file__.replace('{stem}_m.py', '{stem}_impl.py')\n"
            ")\n"
            "_impl = importlib.util.module_from_spec(_spec)\n"
            "_spec.loader.exec_module(_impl)\n"
            "sys.modules[__name__] = _impl\n"
        )
        why = "puts a module of another file"
    else:
        tail = (
            "\n\n_copy = types.ModuleType(__name__)\n"
            "_copy.__file__ = __file__\n_copy.__spec__ = __spec__\n"
            "sys.modules[__name__] = _copy\n"
        )
        why = "puts a module object in its own place in sys.modules"
    edit(m, "* V_M\n", "* V_M\n" + tail)
    edit(
        a, '"""A model file of the GUI\'s tests."""\n', f'"""A model file."""\n\nimport {stem}_m\n'
    )
    refusal = refused(app, "POST", "/api/models/reload")
    assert why in refusal.message
    assert [(f.module, f.models, sorted(f.helpers)) for f in app.catalog.files()] == before


def test_a_union_with_a_typed_dict_takes_it_as_json_gives_it(app: App, tmp_path: Path) -> None:
    """A setting declared a union of a ``TypedDict``, which ``isinstance`` cannot check, and a
    list of an enum's members: a dict and a list typed into it each build the Python call's IR,
    and a Python run of its default reruns."""
    head = (
        "import enum\nfrom typing import TypedDict\n\n\n"
        "class Cfg(TypedDict):\n    a: int\n\n\n"
        "class Sex(enum.Enum):\n    M = 'male'\n    F = 'female'\n"
    )
    body = "    either: Static[Cfg | list[Sex]] = {'a': 1}\n"
    path = write_model(tmp_path, _stem(tmp_path, "typeddict"), head=head, body=body)
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    entry = app.catalog.get(model_id)
    assert entry is not None
    module = sys.modules[entry.cls.__module__]
    for typed, given in (('{"a": 2}', {"a": 2}), ('["M"]', [module.Sex.M])):
        made = call(app, "POST", "/api/build", model=model_id, statics={"either": typed})
        want = halley.build(entry.cls, either=given)
        assert ir_of(app, done(app, made)["build"]) == want.ir_hash
    made_run = run(
        halley.build(entry.cls),
        model_points=tutorial_points(),
        tables={"mort": mort_table()},
        settings=RunSettings(valuation_date=VALUATION_DATE),
        run_dir=app.runs_root / "python-typeddict",
    )
    assert done(app, call(app, "POST", "/api/run/rerun", dir=str(made_run.run_dir)))["same_run_id"]


# ---------------- round 17: a Literal's members by name first, why a rerun misses, odd options


def test_a_literal_s_members_are_read_by_name_before_any_value(app: App, tmp_path: Path) -> None:
    """Literal keys and items whose members' values are other members' names, a text literal
    that is a member's value, an int enum's key typed by its value, and a flag's combined key
    typed with its names in another order: each builds the Python call's IR, a member's name
    taken for that member, as the enum's own field takes it."""
    head = (
        "import enum\nfrom typing import Literal\n\n\n"
        "class Code(enum.Enum):\n    A = 'B'\n    B = 'x'\n\n\n"
        "class Sex(enum.Enum):\n    M = 'male'\n    F = 'female'\n\n\n"
        "class Band(enum.IntEnum):\n    LOW = 1\n    HIGH = 2\n\n\n"
        "class Perm(enum.Flag):\n    R = 1\n    W = 2\n    X = 4\n"
    )
    body = (
        "    ck: Static[dict[Literal[Code.A, Code.B], float]] = {Code.B: 1.0}\n"
        "    lv: Static[list[Literal[Code.A, Code.B]]] = [Code.B]\n"
        "    tk: Static[dict[Literal[Sex.M, Sex.F, 'male'], float]] = {Sex.M: 1.0}\n"
        "    bk: Static[dict[Band, float]] = {Band.LOW: 1.0}\n"
        "    lb: Static[dict[Literal[Band.LOW, Band.HIGH], float]] = {Band.LOW: 1.0}\n"
        "    fk: Static[dict[Literal[Perm.R | Perm.W, Perm.X], float]] = {Perm.X: 1.0}\n"
    )
    path = write_model(tmp_path, _stem(tmp_path, "litnames"), head=head, body=body)
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    entry = app.catalog.get(model_id)
    assert entry is not None
    m = sys.modules[entry.cls.__module__]
    typed = {
        "ck": '{"B": 2.0}',
        "lv": '["B"]',
        "tk": '{"male": 2.0}',
        "bk": '{"2": 2.5}',
        "lb": '{"2": 2.5}',
        "fk": '{"W|R": 2.5}',
    }
    made = done(app, call(app, "POST", "/api/build", model=model_id, statics=typed))["build"]
    want = halley.build(
        entry.cls,
        ck={m.Code.B: 2.0},
        lv=[m.Code.B],
        tk={"male": 2.0},
        bk={m.Band.HIGH: 2.5},
        lb={m.Band.HIGH: 2.5},
        fk={m.Perm.R | m.Perm.W: 2.5},
    )
    assert ir_of(app, made) == want.ir_hash


def test_a_rerun_miss_is_put_down_to_a_member_written_alike_only_when_it_is_why(
    app: App, tmp_path: Path
) -> None:
    """A tuple setting holding an enum's member its class writes as it writes another, which a
    rerun gives back as itself, beside a text a manifest reads back as a date: the build's
    warning names the text's reading, not the member."""
    head = (
        "import datetime as dt\nimport enum\n\n\n"
        "class RX(enum.Enum):\n    A = 1\n    B = 2\n\n"
        "    def __repr__(self):\n        return 'X'\n"
    )
    body = "    rt: Static[tuple[RX, dt.date | str]] = (RX.A, 'x')\n"
    path = write_model(tmp_path, _stem(tmp_path, "alikewhy"), head=head, body=body)
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    typed = {"rt": '["A", "datetime.date(2027, 1, 1)"]'}
    made = done(app, call(app, "POST", "/api/build", model=model_id, statics=typed))["build"]
    warned = app._build(made).warnings
    assert any("build-time setting rt does not read back" in w for w in warned)
    assert not any("makes no difference" in w for w in warned)


def test_a_union_s_unchecked_option_holds_a_value_and_an_alias_s_union_reads_its_parts(
    app: App, tmp_path: Path
) -> None:
    """A union whose option ``isinstance`` cannot check for the value typed (a ``Protocol`` that
    is not runtime-checkable, a class whose ``__instancecheck__`` raises, ``SystemExit`` too, a
    ``TypedDict``) takes the list or dict as JSON gives it; and a list of dicts that fit in part,
    beside a set, under an alias of a union or in an item type that holds one, is read as far
    as it fits: each builds the Python call's IR."""
    head = (
        "import enum\nfrom typing import Protocol, TypedDict\n\n\n"
        "class Sex(enum.Enum):\n    M = 'male'\n    F = 'female'\n\n\n"
        "type LU = list[dict[Sex, float]] | None\n"
        "type OD = dict[Sex, float] | None\n\n\n"
        "class HasLen(Protocol):\n    def __len__(self) -> int: ...\n\n\n"
        "class _Unit(type):\n"
        "    def __instancecheck__(cls, obj):\n"
        "        return obj is not None and obj.unit == 'rate'\n\n\n"
        "class Rate(metaclass=_Unit):\n    pass\n\n\n"
        "class _Halt(type):\n"
        "    def __instancecheck__(cls, obj):\n"
        "        if obj is None:\n            return False\n"
        "        raise SystemExit('no check')\n\n\n"
        "class Halted(metaclass=_Halt):\n    pass\n\n\n"
        "class TD(TypedDict):\n    M: int\n    x: int\n"
    )
    body = (
        "    hl: Static[tuple[float, str] | HasLen] = (1.0, 'a')\n"
        "    dr: Static[tuple[float, str] | Rate] = (1.0, 'a')\n"
        "    sx: Static[tuple[float, str] | Halted] = (1.0, 'a')\n"
        "    td: Static[dict[Sex, int] | TD] = {Sex.M: 1}\n"
        "    lu: Static[set[int] | LU] = None\n"
        "    od: Static[set[int] | list[OD | int]] = [1]\n"
    )
    path = write_model(tmp_path, _stem(tmp_path, "unchecked"), head=head, body=body)
    model_id = call(app, "POST", "/api/models/load", path=str(path))["file"]["models"][0]
    entry = app.catalog.get(model_id)
    assert entry is not None
    m = sys.modules[entry.cls.__module__]
    typed = {
        "hl": "[1.5, 2.5]",
        "dr": "[1.5, 2.5]",
        "sx": "[1.5, 2.5]",
        "td": '{"M": 1, "x": 2}',
        "lu": '[{"M": 1, "X": 2}]',
        "od": '[{"M": 1, "X": 2}]',
    }
    made = done(app, call(app, "POST", "/api/build", model=model_id, statics=typed))["build"]
    want = halley.build(
        entry.cls,
        hl=[1.5, 2.5],
        dr=[1.5, 2.5],
        sx=[1.5, 2.5],
        td={"M": 1, "x": 2},
        lu=[{m.Sex.M: 1.0, "X": 2.0}],
        od=[{m.Sex.M: 1.0, "X": 2.0}],
    )
    assert ir_of(app, made) == want.ir_hash
