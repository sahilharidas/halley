# pyright: basic
"""Input contracts (design section 6): table metadata and rules, input rules, scenario input
bounds, rules, value column, coverage and month column, dictionary-encoded strings, the
metadata and per-variable entries in the manifest, the output hashes, the id field, and a run
id that does not depend on where the input files were."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from halley import Invariant, LoadError, RunSettings, ScenarioInput, build, load_run, run
from halley.authoring import ScenarioView, TableView
from halley.runtime import load_table
from models import (
    VAL,
    Grouped,
    Growth,
    Ruled,
    group_model_points,
    ruled_scenarios,
    ruled_tables,
    run_ruled,
    scenario_file,
    simple_model_points,
)

# ------------------------------------------------------------------------------- tables


def test_tables_with_metadata_and_rules_load(tmp_path: Path) -> None:
    res = run_ruled(tmp_path)
    assert res.n_scenarios == 2
    assert res.lane("q", "S00")[1] == 0.02  # age 21
    assert res.lane("q", "S00")[12] == 1.0  # clamped at 25
    inputs = res.inputs
    assert inputs is not None
    view = inputs.tables["mort"].view()
    assert isinstance(view, TableView)
    assert view.keys == ("sex", "age")
    assert view.shape == (2, 6)
    assert view.key_range("age") == (20, 25)
    assert list(view.key_values("sex")) == ["M", "F"]
    assert view.at("q", sex="F", age=25).shape == (1, 1)
    assert view.metadata == {"version": "ruled-0.1", "base_year": "2027"}


def test_missing_required_metadata_names_the_table_and_key(tmp_path: Path) -> None:
    with pytest.raises(
        LoadError, match=r"table mort.*carries no schema metadata halley\.base_year"
    ):
        run_ruled(tmp_path, tables=ruled_tables(base_year=None))


def test_table_rule_failure_names_the_table_rule_and_first_key(tmp_path: Path) -> None:
    with pytest.raises(
        LoadError, match=r"table mort.*: rule q\(\., 25\) = 1 fails at sex = M \(the table closes"
    ):
        run_ruled(tmp_path, tables=ruled_tables(q_last=0.5))
    with pytest.raises(LoadError, match=r"table params.*: rule z_alpha = inv_cdf\(alpha\) fails"):
        run_ruled(tmp_path, tables=ruled_tables(z_alpha=0.7))


def test_table_rule_failure_names_the_file_when_given_a_path(tmp_path: Path) -> None:
    bad = tmp_path / "mort.parquet"
    pq.write_table(ruled_tables(q_last=0.5)["mort"], bad)
    with pytest.raises(LoadError, match=r"mort\.parquet\): rule q\(\., 25\) = 1 fails at sex = M"):
        run_ruled(tmp_path, tables={**ruled_tables(), "mort": bad})


def test_load_table_without_a_contract_checks_the_declaration_only() -> None:
    graph = build(Ruled).graph
    loaded = load_table(graph.tables["mort"], ruled_tables(base_year=None, q_last=0.5)["mort"])
    assert loaded.values["q"][0, 5] == 0.5
    assert loaded.version == "ruled-0.1"


# -------------------------------------------------------------------------- input rules


def test_input_rules_run_after_every_input_is_loaded(tmp_path: Path) -> None:
    with pytest.raises(LoadError, match="input rule base_year is the valuation year fails"):
        run_ruled(tmp_path, tables=ruled_tables(base_year="2020"))
    with pytest.raises(
        LoadError,
        match=r"input rule every k is at most the horizon fails for 1 of 3 values \(a fixed",
    ):
        run_ruled(tmp_path, horizon=3)


# ---------------------------------------------------------------------------- scenarios


def test_scenario_value_column_month_column_and_bounds(tmp_path: Path) -> None:
    res = run_ruled(tmp_path)
    assert res.lane("df", "S00")[0] == 1.0
    assert res.lane("df", "S00", scenario=1)[2] == 0.98**2
    assert res.manifest["inputs"]["scenarios"]["cpi"]["months"] == [-14, 12]
    assert res.manifest["inputs"]["scenarios"]["df"]["months"] == [0, 12]
    inputs = res.inputs
    assert inputs is not None
    view = inputs.scenarios["cpi"].view()
    assert isinstance(view, ScenarioView)
    assert (view.u0, view.u_max, view.n_scenarios) == (-14, 12, 2)
    assert view.at(0).shape == (2,)
    assert view.rows(-14, 0).shape == (15, 2)
    assert list(view.months[:2]) == [-14, -13]


def test_scenario_file_with_both_t_and_u_is_rejected(tmp_path: Path) -> None:
    scen = ruled_scenarios()
    df = scen["df"]
    both = df.append_column("u", df.column("t"))
    with pytest.raises(LoadError, match="month column is named t or u, not both"):
        run_ruled(tmp_path, scenarios={**scen, "df": both})
    neither = df.drop_columns(["t"])
    with pytest.raises(LoadError, match=r"month column t \(or u\) is missing"):
        run_ruled(tmp_path, scenarios={**scen, "df": neither})


def test_scenario_value_column_is_the_declared_one(tmp_path: Path) -> None:
    scen = ruled_scenarios()
    renamed = scen["df"].rename_columns(["scenario", "t", "df"])
    with pytest.raises(LoadError, match=r"column rf_df is missing \(needs scenario, t, rf_df\)"):
        run_ruled(tmp_path, scenarios={**scen, "df": renamed})


def test_scenario_bounds_name_the_scenario_and_month(tmp_path: Path) -> None:
    scen = ruled_scenarios()
    zero = scenario_file("rf_df", 0, 12, 2, lambda s, u: 0.0 if (s, u) == (1, 3) else 1.0)
    with pytest.raises(LoadError, match=r"values must be > 0; scenario 1, month 3 has 0\.0"):
        run_ruled(tmp_path, scenarios={**scen, "df": zero})
    big = scenario_file("rf_df", 0, 12, 2, lambda s, u: 3.0 if (s, u) == (0, 12) else 1.0)
    with pytest.raises(LoadError, match=r"values must be <= 2; scenario 0, month 12 has 3\.0"):
        run_ruled(tmp_path, scenarios={**scen, "df": big})


def test_scenario_rules_name_the_first_failing_row(tmp_path: Path) -> None:
    with pytest.raises(
        LoadError,
        match=r"rule df\(s, 0\) = 1 fails for scenario 0 \(a discount factor starts at 1\)",
    ):
        run_ruled(tmp_path, scenarios=ruled_scenarios(df0=0.99))
    with pytest.raises(
        LoadError, match="rule history identical across scenarios fails for scenario 1, month -14"
    ):
        run_ruled(tmp_path, scenarios=ruled_scenarios(history_shift=0.5))


def test_covers_demands_more_than_the_months_read(tmp_path: Path) -> None:
    # the model reads cpi(t-3) and cpi(t-15) up to H = 12, so months -14..9; covers says ..H
    with pytest.raises(LoadError, match="reads months -14 to 12 but the file covers -14 to 10"):
        run_ruled(tmp_path, scenarios=ruled_scenarios(cpi_hi=10))
    with pytest.raises(LoadError, match="reads months -14 to 12 but the file covers -13 to 12"):
        run_ruled(tmp_path, scenarios=ruled_scenarios(cpi_lo=-13))


def test_covers_is_spelled_as_a_month_or_h() -> None:
    assert ScenarioInput("x", "Factor", covers=(-2, "H-1")).covers == (-2, "H-1")
    with pytest.raises(ValueError, match="a month is an int, 'H' or 'H-k'"):
        ScenarioInput("x", "Factor", covers=(0, "T"))
    with pytest.raises(ValueError, match="invariant bad: months: a month is an int, 'H' or"):
        Invariant("bad", 1.0, months=(0, "H+1"))


# -------------------------------------------------------------------- model point strings


def test_dictionary_encoded_string_fields_load(tmp_path: Path) -> None:
    mp = simple_model_points()
    encoded = mp.set_column(0, "policy_id", mp.column("policy_id").dictionary_encode())
    assert pa.types.is_dictionary(encoded.column("policy_id").type)
    res = run(
        build(Growth),
        model_points=encoded,
        settings=RunSettings(valuation_date=VAL),
        run_dir=tmp_path / "dict",
    )
    assert list(res.ids) == ["S00", "S01", "S02"]


# ----------------------------------------------------------------------------- manifest


def test_manifest_copies_input_metadata_and_describes_every_stored_variable(
    tmp_path: Path,
) -> None:
    res = run_ruled(tmp_path)
    manifest = res.manifest
    tables = manifest["inputs"]["tables"]
    assert tables["mort"]["metadata"] == {"version": "ruled-0.1", "base_year": "2027"}
    assert tables["mort"]["version"] == "ruled-0.1"
    assert tables["params"]["metadata"] == {"version": "params-7"}  # the bare key
    assert tables["params"]["version"] == "params-7"
    assert manifest["inputs"]["model_points"]["metadata"] == {}
    assert manifest["inputs"]["scenarios"]["df"]["metadata"] == {}
    outputs = manifest["outputs"]
    assert outputs["kind"] == "projection"
    assert outputs["id_field"] == "policy_id"
    assert outputs["variables"]["x"] == {"time": True, "unit": "Money", "start": 1, "role": None}
    assert outputs["variables"]["df"] == {"time": True, "unit": "Factor", "start": 0, "role": None}
    assert outputs["variables"]["alpha"] == {
        "time": False,
        "unit": "Factor",
        "start": 0,
        "role": None,
    }
    for name in ("monthly", "scalars"):
        digest = hashlib.sha256((res.run_dir / outputs[name]).read_bytes()).hexdigest()
        assert outputs[f"{name}_sha256"] == digest
    on_disk = json.loads((res.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk["outputs"] == outputs


def test_outputs_are_keyed_on_the_schema_id_field(tmp_path: Path) -> None:
    res = run(
        build(Grouped),
        model_points=group_model_points(),
        settings=RunSettings(valuation_date=VAL),
        run_dir=tmp_path / "grouped",
    )
    assert res.manifest["outputs"]["id_field"] == "group_id"
    assert res.manifest["outputs"]["variables"]["x"]["role"] == "in_force"
    monthly = pq.read_table(res.run_dir / "outputs" / "monthly.parquet")
    assert monthly.column_names[:3] == ["group_id", "scenario", "t"]
    scalars = pq.read_table(res.run_dir / "outputs" / "scalars.parquet")
    assert scalars.column_names == ["group_id", "scenario", "total"]
    back = load_run(res.run_dir)
    assert list(back.ids) == ["G0", "G1"]
    assert np.array_equal(back.var("x"), res.var("x"))
    # an older manifest without the field reads policy_id
    manifest_path = res.run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["outputs"]["id_field"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(KeyError, match="policy_id"):
        load_run(res.run_dir)


def test_run_id_does_not_depend_on_the_input_paths(tmp_path: Path) -> None:
    mp = simple_model_points()
    paths: list[Path] = []
    for name in ("here", "there"):
        d = tmp_path / name
        d.mkdir()
        pq.write_table(mp, d / "mp.parquet")
        paths.append(d / "mp.parquet")
    compiled = build(Growth)
    results = [
        run(
            compiled,
            model_points=p,
            settings=RunSettings(valuation_date=VAL),
            run_dir=p.parent / "run",
        )
        for p in paths
    ]
    a, b = results
    assert a.run_id == b.run_id
    assert (
        a.manifest["inputs"]["model_points"]["path"] != b.manifest["inputs"]["model_points"]["path"]
    )
    assert (
        a.manifest["inputs"]["model_points"]["sha256"]
        == b.manifest["inputs"]["model_points"]["sha256"]
    )
    assert a.manifest["inputs"]["model_points"]["path"].endswith("mp.parquet")


def test_the_label_does_not_enter_the_run_id_but_the_horizon_does(tmp_path: Path) -> None:
    compiled = build(Growth)

    def go(name: str, **settings: Any) -> Any:
        return run(
            compiled,
            model_points=simple_model_points(),
            settings=RunSettings(valuation_date=VAL, **settings),
            run_dir=tmp_path / name,
        )

    assert go("a").run_id == go("b", label="relabelled").run_id
    assert go("c").run_id != go("d", horizon=6).run_id
