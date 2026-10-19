# pyright: basic
"""The aggregation stage's loader and runs (design section 9, "Loader and runs"; section 10's
loader-side provenance checks): ``run_aggregation`` on bases read from run directories, the
membership from the runs' model point copies, G1, G4, the ``groups`` rules, the previous run by
``prev_run_id`` and the group fields derived from it, the structural and locked-in hashes, the
provenance hook, the run directory's copies, manifest blocks and identity, ``load_run`` and
``replay_aggregation``. Toy projection and stage models (``Member``, ``Recognise``, ``Roll`` in
``tests/models.py``) stand in for the annuity and the CSM classes of stage B3."""

from __future__ import annotations

import dataclasses
import datetime as dt
import functools
import hashlib
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from halley import (
    Basis,
    BuildError,
    Column,
    CompiledModel,
    Derived,
    Factor,
    Field,
    GroupSum,
    Int,
    LoadError,
    Money,
    RunError,
    RunResult,
    RunSettings,
    ScenarioInput,
    Schema,
    Table,
    UnitError,
    build,
    load_run,
    run,
    var,
)
from halley.aggregation import stage as stage_module
from halley.aggregation.stage import (
    Aggregation,
    ProvenanceCheck,
    find_run,
    group_series,
    register_provenance,
    replay_aggregation,
    run_aggregation,
)
from halley.authoring import int_field
from halley.authoring.contracts import group_schema
from halley.runtime.loader import table_of
from models import (
    EXPENSE,
    ILP,
    RECOGNISE_OUTPUTS,
    ROLL_OUTPUTS,
    V_NOW,
    V_REC,
    Member,
    Recognise,
    Release,
    Roll,
    compiled_member,
    covered_points,
    cpi_path,
    curve,
    group_row,
    keyless,
    member_points,
    run_covered,
    run_member,
)

GID = "IA-2025-OTHER"
A, B = [1, 2, 3], [11, 12]
"""The two class runs' policies at recognition; policy 2 dies before the valuation date."""
A_NOW = [1, 3]


@functools.cache
def compiled(stage: type[Aggregation]) -> CompiledModel:
    return build(stage)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(table: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path


def _edited(source: Path, dest: Path, edit: Callable[[dict[str, Any]], object]) -> Path:
    """A copy of a run directory whose manifest ``edit`` changed: the manifest describes the
    files and records their hashes, but no hash covers the manifest itself."""
    shutil.copytree(source, dest)
    path = dest / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    edit(manifest)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return dest


def _ascending_sum(runs: list[RunResult], name: str, t: int) -> float:
    """A variable at month ``t`` summed over every policy of the runs in ascending policy id,
    with plain floats."""
    lanes = sorted((int(pid), r) for r in runs for pid in r.ids.tolist())
    total = 0.0
    for pid, r in lanes:
        total += float(r.lane(name, pid)[t])
    return total


class Chain:
    """A recognition stage at ``V_REC`` on two class runs, then an in-force stage at ``V_NOW``
    on the old, new and cur bases (two class runs each), with the recognition run as its
    previous run: design section 9's chain in miniature, filed under one runs root."""

    def __init__(self, base: Path) -> None:
        self.base = base
        self.root = base / "runs"
        folder = base / "files"
        tables = {
            "rf_rec": curve(0.04),
            "cpi_rec": cpi_path(0.025),
            "ilp": keyless("ilp", 0.005),
            "ra_param": keyless("alpha", 0.75),
            "expense_v1": keyless("maint", 0.5),
            "expense_v2": keyless("maint", 0.6),
            "rf_cur": curve(0.045),
            "cpi_cur": cpi_path(0.031),
        }
        self.files = {n: _write(t, folder / f"{n}.parquet") for n, t in tables.items()}
        self.hashes = {n: _sha(p) for n, p in self.files.items()}
        locked = {"source_sha256": self.hashes["rf_rec"], "ilp_sha256": self.hashes["ilp"]}
        self.files["p_l"] = _write(curve(0.041, meta=locked), folder / "p_l.parquet")
        cpi_locked = cpi_path(0.03, meta={"rec_sha256": self.hashes["cpi_rec"]})
        self.files["cpi_locked"] = _write(cpi_locked, folder / "cpi_locked.parquet")
        self.hashes.update({n: _sha(self.files[n]) for n in ("p_l", "cpi_locked")})
        self.rec = self.rec_basis(self.root)
        self.recognised = self.recognise(self.root / "recognised")
        self.old = [
            self.member(self.root / "old-a", A, expense="expense_v1"),
            self.member(self.root / "old-b", B, scale=1.5, expense="expense_v1"),
        ]
        self.new = [
            self.member(self.root / "new-a", A_NOW),
            self.member(self.root / "new-b", B, scale=1.5),
        ]
        self.cur = [
            self.member(self.root / "cur-a", A_NOW, rf="rf_cur", cpi="cpi_cur"),
            self.member(self.root / "cur-b", B, scale=1.5, rf="rf_cur", cpi="cpi_cur"),
        ]
        self.in_force = self.roll(self.root / "in-force")

    def member(
        self,
        run_dir: Path,
        ids: list[int],
        *,
        at: dt.date = V_NOW,
        rf: str = "p_l",
        cpi: str = "cpi_locked",
        expense: str = "expense_v2",
        ilp: str = "ilp",
        ra: str = "ra_param",
        horizon: int | None = None,
        points: Any = None,
        **kw: Any,
    ) -> RunResult:
        """A ``Member`` class run: at ``V_NOW`` on the locked-in curve and CPI path and the
        current expense table unless told otherwise; 24 months at recognition, 12 after."""
        h = horizon if horizon is not None else (24 if at == V_REC else 12)
        return run_member(
            run_dir,
            points if points is not None else member_points(ids, rem_m=h, **kw),
            rf_df=self.files.get(rf, rf),
            cpi=self.files.get(cpi, cpi),
            expense=self.files.get(expense, expense),
            ilp=self.files.get(ilp, ilp),
            ra_param=self.files.get(ra, ra),
            valuation_date=at,
            horizon=h,
        )

    def rec_basis(self, folder: Path, **kw: Any) -> list[RunResult]:
        """The recognition runs: classes A and B at ``V_REC`` on the recognition basis."""
        common: dict[str, Any] = {
            "at": V_REC,
            "rf": "rf_rec",
            "cpi": "cpi_rec",
            "expense": "expense_v1",
            **kw,
        }
        return [
            self.member(folder / "rec-a", A, **common),
            self.member(folder / "rec-b", B, scale=1.5, **common),
        ]

    def recognise(
        self,
        run_dir: Path,
        *,
        stage: type[Aggregation] = Recognise,
        bases: Any = None,
        groups: Any = None,
        **settings: Any,
    ) -> RunResult:
        settings.setdefault("runs_root", self.root)
        return run_aggregation(
            compiled(stage),
            groups=group_row() if groups is None else groups,
            bases={"cur": self.rec} if bases is None else bases,
            settings=RunSettings(valuation_date=V_REC, **settings),
            run_dir=run_dir,
        )

    def groups(self, previous: RunResult | None = None, **fields: Any) -> Any:
        """The in-force stage's groups file: the recognition run's closing balances and id,
        the locked-in hashes; ``fields`` override."""
        prev = previous if previous is not None else self.recognised
        row: dict[str, Any] = {
            "prev_valuation_date": V_REC,
            "csm_prev": float(prev.lane("csm", GID)[0]),
            "lc_prev": float(prev.lane("lc", GID)[0]),
            "cu_act": 25.0,
            "prev_run_id": prev.run_id,
            "discount_rec_sha256": self.hashes["rf_rec"],
            "ilp_sha256": self.hashes["ilp"],
            "cpi_rec_sha256": self.hashes["cpi_rec"],
            "j": 0.003,
        }
        row.update(fields)
        return group_row(**row)

    def bases(self, **override: Any) -> dict[str, Any]:
        return {"old": self.old, "new": self.new, "cur": self.cur, **override}

    def roll(
        self,
        run_dir: Path,
        *,
        stage: type[Aggregation] = Roll,
        groups: Any = None,
        bases: Any = None,
        scenarios: Any = None,
        previous: Path | None = None,
        **settings: Any,
    ) -> RunResult:
        settings.setdefault("runs_root", self.root)
        return run_aggregation(
            compiled(stage),
            groups=self.groups() if groups is None else groups,
            bases=self.bases() if bases is None else bases,
            scenarios={"p_l": self.files["p_l"]} if scenarios is None else scenarios,
            previous=previous,
            settings=RunSettings(valuation_date=V_NOW, **settings),
            run_dir=run_dir,
        )


@pytest.fixture(scope="module")
def chain(tmp_path_factory: Any) -> Chain:
    return Chain(tmp_path_factory.mktemp("chain"))


# ------------------------------------------------------------------------------ the chain


def test_an_in_force_stage_runs_on_its_bases_and_its_previous_run(chain: Chain) -> None:
    """The in-force stage reads three bases from their run directories and the recognition run
    by its id; the group fields derived from the previous run and the dates are what that run
    stored, and the series sum each basis's members in ascending policy id."""
    r0, r1 = chain.recognised, chain.in_force
    assert r1.kind == "aggregation"
    assert r0.kind == "aggregation"
    assert r1.ids.tolist() == [GID]
    assert r1.manifest["outputs"]["id_field"] == "group_id"
    derived = pq.read_table(r1.run_dir / "inputs" / "derived_groups.parquet")
    assert derived.column_names == [
        "group_id",
        "f_prev_0",
        "f_prev_tau",
        "lc_prev_tau",
        "p_g_prev",
        "tau0",
        "tau",
        "i_acc",
        "i_rec",
    ]
    row = derived.to_pylist()[0]
    fcf_g, lc = r0.lane("fcf_g", GID), r0.lane("lc", GID)
    assert row["group_id"] == GID
    assert row["f_prev_0"] == fcf_g[0]
    assert row["f_prev_tau"] == fcf_g[12]
    assert row["lc_prev_tau"] == lc[12]
    assert row["p_g_prev"] == r0.lane("p_g", GID)[0]
    assert (row["tau0"], row["tau"], row["i_acc"], row["i_rec"]) == (12, 12, 0, 12)
    assert r1.lane("aoc", GID)[0] == (fcf_g[12] - fcf_g[0]) + lc[12]
    assert r1.lane("months", GID)[0] == 12
    assert r1.lane("p_g", GID)[0] == r0.lane("p_g", GID)[0]
    old, new = _ascending_sum(chain.old, "fcf", 0), _ascending_sum(chain.new, "fcf", 0)
    assert r1.lane("delta", GID)[0] == new - old
    cu_g = [_ascending_sum(chain.cur, "cu", t) for t in range(13)]
    assert r1.lane("units", GID)[12] == cu_g[12]
    groups = r1.manifest["membership"]["groups"]
    assert r1.manifest["membership"]["key"] == ["portfolio", "cohort", "profit_bucket"]
    assert groups == {GID: {"old": [1, 2, 3, 11, 12], "new": [1, 3, 11, 12], "cur": [1, 3, 11, 12]}}
    assert r0.manifest["membership"]["groups"] == {GID: {"cur": [1, 2, 3, 11, 12]}}
    assert r0.manifest["previous"] is None


def test_the_run_directory_copies_what_the_stage_read_d21(chain: Chain) -> None:
    """D21: the groups file, the derived group fields, every series with a mean's pathwise
    sibling and the scenario files are copied and hashed; the manifest records the previous
    run, every run of every basis (directories relative to runs_root), the locked-in hashes,
    the provenance checks and the disclosures; runs_root and workers are recorded."""
    r1 = chain.in_force
    d, m = r1.run_dir, r1.manifest
    inputs = m["inputs"]
    assert inputs["groups"]["copy"] == "inputs/groups.parquet"
    assert pq.read_table(d / "inputs" / "groups.parquet").equals(chain.groups())
    for key in ("groups", "derived_groups", "model_points"):
        assert _sha(d / inputs[key]["copy"]) == inputs[key]["sha256"], key
    assert inputs["derived_groups"]["columns"][0] == "group_id"
    model_points = pq.read_table(d / "inputs" / "model_points.parquet")
    assert model_points.column("group_id").to_pylist() == [GID]
    assert model_points.column("f_prev_tau").to_pylist() == [
        chain.recognised.lane("fcf_g", GID)[12]
    ]
    assert (d / "inputs" / "scenarios" / "p_l.parquet").read_bytes() == chain.files[
        "p_l"
    ].read_bytes()
    assert set(inputs["series"]) == {"cu_g", "fcf_g", "fcf_old", "fcf_new"}
    for name, entry in inputs["series"].items():
        assert (entry["basis"], entry["scenarios"], entry["unit"]) == (
            {"fcf_old": "old", "fcf_new": "new"}.get(name, "cur"),
            "mean",
            "CoverageUnits" if name == "cu_g" else "Money",
        ), name
        assert entry["copy"] == f"inputs/series/{name}.parquet"
        assert _sha(d / entry["copy"]) == entry["sha256"]
        pathwise = entry["pathwise"]
        assert pathwise["copy"] == f"inputs/series/{name}.pathwise.parquet"
        assert _sha(d / pathwise["copy"]) == pathwise["sha256"]
        assert pathwise["scenarios"] == 1
        table = pq.read_table(d / pathwise["copy"])
        assert table.schema.metadata[b"halley.scenarios"] == b"pathwise"
        runs = {"old": chain.old, "new": chain.new}.get(entry["basis"], chain.cur)
        assert [s["run_id"] for s in entry["source_runs"]] == [r.run_id for r in runs]
        for source, r in zip(entry["source_runs"], runs, strict=True):
            assert source["dir"] == r.run_dir.name  # relative to runs_root
            assert source["manifest_sha256"] == _sha(r.run_dir / "manifest.json")
            assert source["outputs_sha256"] == {
                "monthly": r.manifest["outputs"]["monthly_sha256"],
                "scalars": r.manifest["outputs"]["scalars_sha256"],
            }
    assert inputs["series"]["fcf_g"]["start"] == 0
    assert inputs["series"]["cu_g"]["start"] == 1
    assert m["previous"] == {
        "run_id": chain.recognised.run_id,
        "dir": "recognised",
        "manifest_sha256": _sha(chain.recognised.run_dir / "manifest.json"),
    }
    for basis, runs in (("old", chain.old), ("new", chain.new), ("cur", chain.cur)):
        assert [e["run_id"] for e in m["bases"][basis]] == [r.run_id for r in runs]
        assert [e["dir"] for e in m["bases"][basis]] == [r.run_dir.name for r in runs]
        assert [e["manifest_sha256"] for e in m["bases"][basis]] == [
            _sha(r.run_dir / "manifest.json") for r in runs
        ]
    locked = m["locked_in"][GID]
    assert {k: v for k, v in locked.items() if k != "verified_by"} == {
        "discount_rec_sha256": chain.hashes["rf_rec"],
        "ilp_sha256": chain.hashes["ilp"],
        "cpi_rec_sha256": chain.hashes["cpi_rec"],
    }
    locked_runs = [(b, r) for b, rs in (("old", chain.old), ("new", chain.new)) for r in rs]
    for name, meta in (("discount_rec_sha256", "source_sha256"), ("ilp_sha256", "ilp_sha256")):
        # the stage's p_l, then the curve every locked-in run discounts with: p_l as its rf_df
        assert locked["verified_by"][name] == [
            {"file": "scenario input p_l", "sha256": chain.hashes["p_l"], "metadata": meta},
            *(
                {
                    "file": f"basis {b} run {r.run_id}: scenario input rf_df",
                    "sha256": chain.hashes["p_l"],
                    "metadata": meta,
                }
                for b, r in locked_runs
            ),
        ], name
    assert locked["verified_by"]["cpi_rec_sha256"] == [
        {
            "file": f"basis {b} run {r.run_id}: scenario input cpi",
            "sha256": chain.hashes["cpi_locked"],
            "metadata": "rec_sha256",
        }
        for b, r in locked_runs
    ]
    assert m["invariants_sha256"] == _sha(d / "invariants.json")
    assert m["provenance"] == []
    assert m["disclosures"] == {"confidence_level": 0.75, "para_120_curve_sha256": None}
    assert m["settings"]["runs_root"] == str(chain.root)
    assert m["settings"]["workers"] == 1


def test_aggregation_rerun_is_bit_identical_d19(chain: Chain, tmp_path: Path) -> None:
    """D19: the same inputs give the same run id and the same bits, whether the bases are
    given as runs of the session or as their directories, and whatever the label, the number
    of workers or where the run is filed (none of which the identity covers)."""
    again = chain.roll(tmp_path / "again")
    assert again.run_id == chain.in_force.run_id
    for name in chain.in_force.variables:
        assert np.array_equal(again.arrays[name], chain.in_force.arrays[name]), name
    assert (
        again.manifest["outputs"]["monthly_sha256"]
        == chain.in_force.manifest["outputs"]["monthly_sha256"]
    )
    by_path = {b: [r.run_dir for r in rs] for b, rs in chain.bases().items()}
    other = chain.roll(tmp_path / "by-path", bases=by_path, label="by path", workers=2)
    assert other.run_id == chain.in_force.run_id
    assert other.manifest["settings"]["workers"] == 2
    for name in chain.in_force.variables:
        assert np.array_equal(other.arrays[name], chain.in_force.arrays[name]), name
    bumped = chain.roll(tmp_path / "bumped", groups=chain.groups(cu_act=26.0))
    assert bumped.run_id != chain.in_force.run_id  # the groups file is in the identity


def test_aggregation_run_round_trips_through_load_run_d21(chain: Chain, tmp_path: Path) -> None:
    """``load_run`` reads an aggregation run back, every file verified first with
    ``verify=True``: the group ids, the stored outputs bit for bit, the kind and the units; a
    changed output file is a LoadError naming it."""
    r1 = chain.in_force
    loaded = load_run(r1.run_dir, verify=True)
    assert loaded.kind == "aggregation"
    assert loaded.ids.tolist() == [GID]
    assert loaded.H == 12
    assert loaded.n_scenarios == 1
    assert loaded.manifest == r1.manifest
    assert sorted(loaded.arrays) == sorted(r1.manifest["outputs"]["variables"])
    for name in loaded.arrays:
        assert np.array_equal(loaded.var(name), r1.var(name)), name
    assert loaded.variable_info("csm").unit == "Money"
    assert loaded.variable_info("months").unit == "Int"
    assert not loaded.variable_info("months").time
    copy = tmp_path / "copy"
    shutil.copytree(r1.run_dir, copy)
    scalars = pq.read_table(copy / "outputs" / "scalars.parquet")
    changed = scalars.set_column(
        scalars.column_names.index("csm_0"),
        "csm_0",
        pa.array([scalars.column("csm_0")[0].as_py() + 1.0]),
    )
    pq.write_table(changed, copy / "outputs" / "scalars.parquet")
    with pytest.raises(LoadError, match="scalars outputs") as info:
        load_run(copy, verify=True)
    assert str(copy / "outputs" / "scalars.parquet") in str(info.value)
    assert "does not match the sha256" in str(info.value)
    # the invariant results are hashed too: a record edited to pass is caught
    assert r1.manifest["invariants_sha256"] == _sha(r1.run_dir / "invariants.json")
    passed = tmp_path / "passed"
    shutil.copytree(r1.run_dir, passed)
    results = json.loads((passed / "invariants.json").read_text(encoding="utf-8"))
    results[0]["passed"] = not results[0]["passed"]
    (passed / "invariants.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with pytest.raises(LoadError, match="invariant results") as info:
        load_run(passed, verify=True)
    assert str(passed / "invariants.json") in str(info.value)


def test_aggregation_replays_from_its_own_directory_d21(chain: Chain, tmp_path: Path) -> None:
    """D21: with every base directory and the previous run moved away, the run is re-run from
    its own copies alone: the same run id and bit-identical arrays. A changed copy is a
    LoadError naming it; another stage cannot replay the run."""
    tree = tmp_path / "tree"
    shutil.copytree(chain.root, tree)
    gone = tmp_path / "gone"
    gone.mkdir()
    for d in sorted(tree.iterdir()):
        if d.name != "in-force":
            shutil.move(d, gone / d.name)
    assert [d.name for d in tree.iterdir()] == ["in-force"]
    replay = replay_aggregation(compiled(Roll), tree / "in-force", run_dir=tmp_path / "replayed")
    assert replay.run_id == chain.in_force.run_id
    for name in chain.in_force.variables:
        assert np.array_equal(replay.arrays[name], chain.in_force.arrays[name]), name
    m = replay.manifest
    assert m["replayed_from"]["run_id"] == chain.in_force.run_id
    assert m["replayed_from"]["matches"] is True
    assert "differences" not in m["replayed_from"]
    on_disk = json.loads((replay.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk["replayed_from"] == m["replayed_from"]
    for key in ("previous", "bases", "membership", "locked_in", "provenance", "disclosures"):
        assert m[key] == chain.in_force.manifest[key], key
    assert m["inputs"]["scenarios"]["p_l"]["path"].startswith(str(tree / "in-force"))
    assert m["inputs"]["groups"]["sha256"] == chain.in_force.manifest["inputs"]["groups"]["sha256"]
    for name, entry in m["inputs"]["series"].items():
        mine = chain.in_force.manifest["inputs"]["series"][name]
        assert entry["values_sha256"] == mine["values_sha256"], name
    assert load_run(replay.run_dir, verify=True).run_id == chain.in_force.run_id

    with pytest.raises(LoadError, match="ir_sha256"):
        replay_aggregation(compiled(Recognise), tree / "in-force", run_dir=tmp_path / "other")
    with pytest.raises(LoadError, match="no run made by run_aggregation"):
        replay_aggregation(compiled(Roll), gone / "old-a", run_dir=tmp_path / "projection")
    series = tree / "in-force" / "inputs" / "series" / "fcf_old.parquet"
    table = pq.read_table(series)
    values = table.column("value").to_pylist()
    values[0] += 1e-9
    pq.write_table(table.set_column(3, "value", pa.array(values)), series)
    with pytest.raises(LoadError, match="series fcf_old") as info:
        replay_aggregation(compiled(Roll), tree / "in-force", run_dir=tmp_path / "tampered")
    assert str(series) in str(info.value)


def test_same_bytes_at_two_paths_give_one_run_id(chain: Chain, tmp_path: Path) -> None:
    """A projection run is addressed by what it read, not where: the same model point bytes at
    two paths give one run id, as do another label, number of workers or runs root; other
    bytes give another id."""
    points = member_points(A, rem_m=12)
    first = _write(points, tmp_path / "here" / "points.parquet")
    second = tmp_path / "there" / "elsewhere.parquet"
    second.parent.mkdir()
    shutil.copyfile(first, second)
    a = chain.member(tmp_path / "a", A, points=first)
    b = chain.member(tmp_path / "b", A, points=second)
    assert (
        a.manifest["inputs"]["model_points"]["path"] != b.manifest["inputs"]["model_points"]["path"]
    )
    assert a.run_id == b.run_id
    settings = RunSettings(
        valuation_date=V_NOW, horizon=12, label="x", workers=3, runs_root=tmp_path
    )
    c = run(
        compiled_member(),
        model_points=second,
        tables={n: chain.files[n] for n in ("ilp", "ra_param")}
        | {"expense": chain.files["expense_v2"]},
        scenarios={"rf_df": chain.files["p_l"], "cpi": chain.files["cpi_locked"]},
        settings=settings,
        run_dir=tmp_path / "c",
    )
    assert c.run_id == a.run_id
    assert c.manifest["settings"]["runs_root"] == str(tmp_path)
    other = chain.member(tmp_path / "d", A, points=member_points(A, rem_m=12, scale=1.01))
    assert other.run_id != a.run_id


# ----------------------------------------------------------------------------- the bases


def test_basis_rejects_duplicate_policy_ids(chain: Chain, tmp_path: Path) -> None:
    """The runs of a basis hold disjoint policies: a policy in two class runs is a LoadError
    naming it and both runs."""
    dup = chain.member(
        tmp_path / "dup", [3, 13], at=V_REC, rf="rf_rec", cpi="cpi_rec", expense="expense_v1"
    )
    first = chain.rec[0]
    with pytest.raises(
        LoadError, match=f"policy 3 is in run {first.run_id} and in run {dup.run_id}"
    ):
        chain.recognise(tmp_path / "stage", bases={"cur": [first, dup]})


def test_a_tampered_base_file_is_a_load_error_naming_it(chain: Chain, tmp_path: Path) -> None:
    """Every file of a basis run is verified against its manifest's sha256 before the stage
    reads it: a changed output, or a changed model point copy, is a LoadError naming the
    file."""
    copy = tmp_path / "cur-a"
    shutil.copytree(chain.cur[0].run_dir, copy)
    monthly = copy / "outputs" / "monthly.parquet"
    table = pq.read_table(monthly)
    fcf = table.column("fcf").to_pylist()
    fcf[0] *= 1.000001
    pq.write_table(table.set_column(table.column_names.index("fcf"), "fcf", pa.array(fcf)), monthly)
    bases = chain.bases(cur=[copy, chain.cur[1]])
    with pytest.raises(LoadError, match=r"basis cur run .*monthly outputs") as info:
        chain.roll(tmp_path / "stage", bases=bases)
    assert str(monthly) in str(info.value)
    assert "does not match the sha256" in str(info.value)

    copy2 = tmp_path / "cur-a-2"
    shutil.copytree(chain.cur[0].run_dir, copy2)
    copied = copy2 / "inputs" / "model_points.parquet"
    pq.write_table(member_points(A_NOW, rem_m=12, cohort=2024), copied)
    with pytest.raises(LoadError, match="model points") as info:
        chain.roll(tmp_path / "stage", bases=chain.bases(cur=[copy2, chain.cur[1]]))
    assert str(copied) in str(info.value)

    shutil.copytree(chain.cur[0].run_dir, tmp_path / "cur-a-3")
    (tmp_path / "cur-a-3" / "outputs" / "scalars.parquet").unlink()
    with pytest.raises(LoadError, match="is missing, though its run's manifest records it"):
        chain.roll(tmp_path / "stage", bases=chain.bases(cur=[tmp_path / "cur-a-3", chain.cur[1]]))


def test_the_bases_are_checked_before_anything_is_summed(chain: Chain, tmp_path: Path) -> None:
    """The bases given are the bases declared, each a list of projection runs; a basis's runs
    share the valuation date and horizon (Basis.check), which the stage's valuation date
    equals; a projection is not run as a stage."""
    out = tmp_path / "stage"
    with pytest.raises(LoadError, match="reads basis old, which was not given"):
        chain.roll(out, bases={"new": chain.new, "cur": chain.cur})
    with pytest.raises(LoadError, match="basis 'extra' was given but Roll declares no such basis"):
        chain.roll(out, bases=chain.bases(extra=chain.cur))
    with pytest.raises(LoadError, match="basis cur takes a list of runs"):
        chain.roll(out, bases=chain.bases(cur=chain.cur[0]))
    with pytest.raises(LoadError, match="holds an aggregation run; a basis reads projection runs"):
        chain.roll(out, bases=chain.bases(cur=[chain.recognised, *chain.cur]))
    with pytest.raises(
        LoadError, match="basis cur values at 2025-12-31, but the stage at 2026-12-31"
    ):
        run_aggregation(
            compiled(Recognise),
            groups=group_row(),
            bases={"cur": chain.rec},
            settings=RunSettings(valuation_date=V_NOW, runs_root=chain.root),
            run_dir=out,
        )
    longer = chain.member(tmp_path / "longer", B, scale=1.5, rf="rf_cur", cpi="cpi_cur", horizon=13)
    with pytest.raises(LoadError, match=r"give every run of a basis the same RunSettings\.horizon"):
        chain.roll(out, bases=chain.bases(cur=[chain.cur[0], longer]))
    with pytest.raises(LoadError, match="is a projection; run_aggregation runs an aggregation"):
        run_aggregation(
            compiled_member(),
            groups=group_row(),
            bases={"cur": chain.rec},
            settings=RunSettings(valuation_date=V_REC),
            run_dir=out,
        )
    with pytest.raises(LoadError, match="already holds a run"):
        chain.roll(chain.in_force.run_dir)


class RollRelative(Roll):
    """Reads the old basis relatively, so the old basis must reach the stage's horizon."""

    outputs = (*ROLL_OUTPUTS, "old_path")

    @var(domain="state")
    def old_path(self, t: Int) -> Money:
        return self.series.fcf_old(t)


class RollLate(Roll):
    """Reads the old basis at month 9 only, so the old basis must reach month 9."""

    outputs = (*ROLL_OUTPUTS, "late")

    @var
    def late(self) -> Money:
        return self.series.fcf_old.at(9)


def test_the_stage_horizon_rule(chain: Chain, tmp_path: Path) -> None:
    """The stage runs to its cur basis's horizon or a shorter RunSettings.horizon; a series
    read relatively comes from a basis that reaches it and is cut to it, a series read at month
    k from a basis that reaches k; otherwise a LoadError names the basis."""
    with pytest.raises(
        LoadError, match=r"basis cur covers months 0\.\.12 but the stage runs to H = 13"
    ):
        chain.roll(tmp_path / "long", horizon=13)
    short = chain.roll(tmp_path / "short", horizon=6)
    assert short.H == 6
    series = short.manifest["inputs"]["series"]
    assert series["cu_g"]["months"] == [0, 6]
    assert series["fcf_g"]["months"] == [0, 6]
    assert series["fcf_old"]["months"] == [0, 12]  # read only at month 0: kept whole
    copy = pq.read_table(short.run_dir / series["cu_g"]["copy"])
    assert copy.num_rows == 7
    old6 = [
        chain.member(tmp_path / "old6-a", A, expense="expense_v1", horizon=6),
        chain.member(tmp_path / "old6-b", B, scale=1.5, expense="expense_v1", horizon=6),
    ]
    with pytest.raises(
        LoadError, match=r"basis old covers months 0\.\.6 but the stage runs to H = 12"
    ):
        chain.roll(tmp_path / "relative", stage=RollRelative, bases=chain.bases(old=old6))
    with pytest.raises(
        LoadError, match=r"series.fcf_old.at\(9\), but basis old covers months 0..6"
    ):
        chain.roll(tmp_path / "late", stage=RollLate, bases=chain.bases(old=old6))
    fine = chain.roll(tmp_path / "late-ok", stage=RollLate)
    assert fine.lane("late", GID)[0] == _ascending_sum(chain.old, "fcf", 9)


# ----------------------------------------------------------------------- the membership


class Paired(Recognise):
    """A stage with two bases and no previous run."""

    bases = (Basis("cur"), Basis("old"))
    series = (*Release.series, GroupSum("fcf_old", "old", "fcf", "Money", start=0))
    outputs = (*RECOGNISE_OUTPUTS, "old_0")

    @var
    def old_0(self) -> Money:
        return self.series.fcf_old.at(0)


def test_the_membership_comes_from_the_model_point_copies(chain: Chain, tmp_path: Path) -> None:
    """Each run's model point copy gives each member's group key; every member maps to exactly
    one group row, is issued on its recognition date and has one key in every basis; a copy
    without the key is a LoadError naming it."""
    odd = chain.rec_basis(tmp_path / "odd", cohorts={2: 2024})
    with pytest.raises(
        LoadError,
        match=r"model points \(.*model_points.parquet\): row 1, policy 2 is in group "
        r"IA-2024-OTHER, which has no row in groups",
    ):
        chain.recognise(tmp_path / "stage", bases={"cur": odd})
    late = chain.rec_basis(tmp_path / "late", issues={3: dt.date(2025, 11, 30)})
    with pytest.raises(
        LoadError,
        match="row 2, policy 3: issue_date 2025-11-30 is not the recognition_date 2025-12-31 of "
        "group IA-2025-OTHER",
    ):
        chain.recognise(tmp_path / "stage", bases={"cur": late})
    covered = run_covered(tmp_path, "covered", covered_points("C"), valuation_date=V_REC)
    with pytest.raises(LoadError, match="column portfolio is missing; a basis run's model points"):
        chain.recognise(tmp_path / "stage", bases={"cur": [covered]})
    other = chain.rec_basis(tmp_path / "other", cohorts={1: 2024})
    with pytest.raises(
        LoadError,
        match=r"row 0, policy 1 is in group IA-2024-OTHER, but .* is in group IA-2025-OTHER; a "
        r"policy's group key is the same in every run \(G4\)",
    ):
        chain.recognise(tmp_path / "stage", stage=Paired, bases={"cur": chain.rec, "old": other})
    paired = chain.recognise(
        tmp_path / "paired", stage=Paired, bases={"cur": chain.rec, "old": chain.rec}
    )
    assert paired.lane("old_0", GID)[0] == chain.recognised.lane("fcf_g", GID)[0]


def test_the_groups_file(chain: Chain, tmp_path: Path) -> None:
    """The groups file holds the group key, not group_id, and no derived field; one row per
    group (a duplicate key names both rows); one group per run (G1)."""
    out = tmp_path / "stage"
    with pytest.raises(LoadError, match="carries a column group_id, which the loader derives"):
        chain.recognise(out, groups=group_row().append_column("group_id", pa.array([GID])))
    with pytest.raises(LoadError, match="carries f_prev_0, which the loader derives"):
        chain.roll(out, groups=chain.groups(f_prev_0=1.0))
    with pytest.raises(LoadError, match="column profit_bucket is missing; the groups carry"):
        chain.recognise(out, groups=group_row(profit_bucket=None))
    two = group_row(
        portfolio=["IA", "IA"],
        cohort=[2025, 2025],
        profit_bucket=["OTHER", "OTHER"],
        recognition_date=[V_REC, V_REC],
        j=[0.002, 0.003],
    )
    path = _write(two, tmp_path / "groups.parquet")
    with pytest.raises(LoadError, match="rows 0 and 1 are both group IA-2025-OTHER") as info:
        chain.recognise(out, groups=path)
    assert str(path) in str(info.value)
    other = group_row(
        portfolio=["IA", "IA"],
        cohort=[2025, 2025],
        profit_bucket=["OTHER", "ONEROUS"],
        recognition_date=[V_REC, V_REC],
        j=[0.002, 0.003],
    )
    with pytest.raises(
        LoadError, match=r"holds 2 groups \(IA-2025-OTHER, IA-2025-ONEROUS\).*\(G1\)"
    ):
        chain.recognise(out, groups=other)


def test_the_groups_rules(chain: Chain, tmp_path: Path) -> None:
    """In-force 3.3's rules on the groups file, each a LoadError naming the file and the group's
    row: month-end dates, prev_valuation_date at least a month before the valuation date and
    no earlier than the recognition date, at most one of csm_prev and lc_prev positive."""
    out = tmp_path / "stage"
    path = _write(chain.groups(recognition_date=dt.date(2025, 12, 30)), tmp_path / "g1.parquet")
    with pytest.raises(LoadError, match="recognition_date 2025-12-30 is not the last day") as info:
        chain.roll(out, groups=path)
    assert f"groups ({path}): group IA-2025-OTHER (row 0)" in str(info.value)
    with pytest.raises(LoadError, match="prev_valuation_date 2026-12-31 is not at least a month"):
        chain.roll(out, groups=chain.groups(prev_valuation_date=V_NOW))
    with pytest.raises(LoadError, match="prev_valuation_date 2026-11-15 is not the last day"):
        chain.roll(out, groups=chain.groups(prev_valuation_date=dt.date(2026, 11, 15)))
    with pytest.raises(LoadError, match="is before recognition_date 2025-12-31, so tau = 13"):
        chain.roll(out, groups=chain.groups(prev_valuation_date=dt.date(2025, 11, 30)))
    with pytest.raises(LoadError, match="recognition_date 2027-01-31 is after the valuation date"):
        chain.roll(out, groups=chain.groups(recognition_date=dt.date(2027, 1, 31)))
    with pytest.raises(LoadError, match=r"csm_prev = .* and lc_prev = 1\.0 are both positive"):
        chain.roll(out, groups=chain.groups(lc_prev=1.0))


# ------------------------------------------------------------------------ the previous run


def test_the_previous_run_is_addressed_by_its_run_id(chain: Chain, tmp_path: Path) -> None:
    """``prev_run_id`` is resolved by scanning runs_root/*/manifest.json: no match, or several,
    is a LoadError; ``previous=`` overrides the search but must hold that run id, and gives the
    same run id (the identity has no directories); the run named must be an aggregation run."""
    out = tmp_path / "stage"
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(
        LoadError, match=f"no run under runs_root .* has run_id {chain.recognised.run_id}"
    ):
        chain.roll(out, runs_root=empty)
    twice = tmp_path / "twice"
    shutil.copytree(chain.recognised.run_dir, twice / "one")
    shutil.copytree(chain.recognised.run_dir, twice / "two")
    with pytest.raises(
        LoadError, match=f"2 runs under runs_root .* have run_id {chain.recognised.run_id}"
    ):
        chain.roll(out, runs_root=twice)
    assert find_run(chain.recognised.run_id, chain.root) == chain.recognised.run_dir
    with pytest.raises(LoadError, match=r"set RunSettings\.runs_root"):
        chain.roll(out, runs_root=None)
    direct = chain.roll(tmp_path / "direct", runs_root=None, previous=chain.recognised.run_dir)
    assert direct.run_id == chain.in_force.run_id
    assert direct.manifest["previous"]["dir"] == str(chain.recognised.run_dir.resolve())
    with pytest.raises(
        LoadError, match=f"holds run {chain.in_force.run_id}, but .* names prev_run_id"
    ):
        chain.roll(out, previous=chain.in_force.run_dir)
    projection = chain.groups(prev_run_id=chain.rec[0].run_id)
    with pytest.raises(LoadError, match="a projection run; it names the previous aggregation run"):
        chain.roll(out, groups=projection)
    with pytest.raises(LoadError, match="previous= was given, but the stage's groups name no"):
        run_aggregation(
            compiled(Recognise),
            groups=group_row(),
            bases={"cur": chain.rec},
            previous=chain.recognised.run_dir,
            settings=RunSettings(valuation_date=V_REC),
            run_dir=out,
        )


def test_g4_holds_against_the_previous_members(chain: Chain, tmp_path: Path) -> None:
    """G4 per basis: the old basis holds every member of the previous run; every policy at V
    was a member, in the same group; the new and cur bases hold the same policies."""
    out = tmp_path / "stage"
    short_old = [
        chain.member(tmp_path / "old-a", A_NOW, expense="expense_v1"),
        chain.old[1],
    ]
    with pytest.raises(
        LoadError, match="basis old holds no run with policies 2 of the previous run"
    ):
        chain.roll(out, bases=chain.bases(old=short_old))
    joined_new = [chain.member(tmp_path / "new-a", [1, 3, 99]), chain.new[1]]
    joined_cur = [
        chain.member(tmp_path / "cur-a", [1, 3, 99], rf="rf_cur", cpi="cpi_cur"),
        chain.cur[1],
    ]
    with pytest.raises(
        LoadError,
        match=r"model_points.parquet\): row 2, policy 99 is not a member of the previous run "
        r".*\(G4, para 28\)",
    ):
        chain.roll(out, bases=chain.bases(new=joined_new, cur=joined_cur))
    moved = {3: 2024}
    bases = chain.bases(
        old=[
            chain.member(tmp_path / "old-m", A, expense="expense_v1", cohorts=moved),
            chain.old[1],
        ],
        new=[chain.member(tmp_path / "new-m", A_NOW, cohorts=moved), chain.new[1]],
        cur=[
            chain.member(tmp_path / "cur-m", A_NOW, rf="rf_cur", cpi="cpi_cur", cohorts=moved),
            chain.cur[1],
        ],
    )
    with pytest.raises(
        LoadError,
        match="row 2, policy 3 is in group IA-2024-OTHER but was in group IA-2025-OTHER at the "
        "previous run",
    ):
        chain.roll(out, bases=bases)
    wide_new = [chain.member(tmp_path / "new-w", A), chain.new[1]]
    with pytest.raises(
        LoadError,
        match=r"bases new and cur hold other policies \(only in new: 2; only in cur: none\)",
    ):
        chain.roll(out, bases=chain.bases(new=wide_new))


def test_the_previous_run_supplies_the_derived_fields(chain: Chain, tmp_path: Path) -> None:
    """The previous run values at prev_valuation_date, holds the group, stores what the derived
    fields read (an output, or failing one its group series) over enough months, and closed at
    csm_prev and lc_prev within 1e-12 * P_g."""
    out = tmp_path / "stage"
    with pytest.raises(
        LoadError,
        match=f"prev_valuation_date is 2026-06-30, but the previous run {chain.recognised.run_id} "
        "values at 2025-12-31",
    ):
        chain.roll(out, groups=chain.groups(prev_valuation_date=dt.date(2026, 6, 30)))
    csm = float(chain.recognised.lane("csm", GID)[0])
    with pytest.raises(
        LoadError, match=r"csm_prev = .*, but the previous run .* closed at csm\(0\)"
    ):
        chain.roll(out, groups=chain.groups(csm_prev=csm * (1 + 1e-9)))
    within = chain.roll(tmp_path / "within", groups=chain.groups(csm_prev=csm + csm * 1e-14))
    assert within.lane("months", GID)[0] == 12

    root = tmp_path / "root"
    rec = chain.rec_basis(root)
    as_series = chain.recognise(
        root / "series-only",
        bases={"cur": rec},
        runs_root=root,
        outputs=("csm", "lc", "p_g", "csm_0"),
    )
    assert "fcf_g" not in as_series.manifest["outputs"]["variables"]
    from_series = chain.roll(
        tmp_path / "from-series", groups=chain.groups(previous=as_series), runs_root=root
    )
    derived = pq.read_table(from_series.run_dir / "inputs" / "derived_groups.parquet").to_pylist()
    mine = pq.read_table(chain.in_force.run_dir / "inputs" / "derived_groups.parquet").to_pylist()
    assert derived == mine  # fcf_g read from the previous run's series copy, the same bits

    no_lc = chain.recognise(
        root / "no-lc", bases={"cur": rec}, runs_root=root, outputs=("csm", "fcf_g", "p_g")
    )
    with pytest.raises(
        LoadError, match=f"lc_prev_tau: the previous run {no_lc.run_id} stored no lc"
    ):
        chain.roll(out, groups=chain.groups(previous=no_lc, lc_prev=0.0), runs_root=root)
    short = chain.recognise(root / "short", bases={"cur": rec}, runs_root=root, horizon=6)
    with pytest.raises(
        LoadError,
        match=r"f_prev_tau: the previous run .* covers months 0\.\.6, but the loader reads "
        r"fcf_g\(12\)",
    ):
        chain.roll(out, groups=chain.groups(previous=short), runs_root=root)


# ------------------------------------------------------------------- hashes and provenance


def test_the_structural_hashes(chain: Chain, tmp_path: Path) -> None:
    """The old and new bases share every financial input (rf_df, cpi, ilp); the cur and new
    bases every non-financial table; otherwise a LoadError names both runs and the input."""
    out = tmp_path / "stage"
    other = {"source_sha256": chain.hashes["rf_rec"], "ilp_sha256": chain.hashes["ilp"]}
    p_l2 = _write(curve(0.0411, meta=other), tmp_path / "p_l2.parquet")
    new = [
        chain.member(tmp_path / "new-a", A_NOW, rf=str(p_l2)),
        chain.member(tmp_path / "new-b", B, scale=1.5, rf=str(p_l2)),
    ]
    with pytest.raises(
        LoadError,
        match=f"basis old run {chain.old[0].run_id} and basis new run {new[0].run_id} read "
        "scenario input rf_df from different files",
    ):
        chain.roll(out, bases=chain.bases(new=new))
    ilp2 = _write(keyless("ilp", 0.006), tmp_path / "ilp2.parquet")
    new = [chain.member(tmp_path / "new-i", A_NOW, ilp=str(ilp2)), chain.new[1]]
    with pytest.raises(LoadError, match=r"read table ilp from different files.*financial input"):
        chain.roll(out, bases=chain.bases(new=new))
    expense3 = _write(keyless("maint", 0.61), tmp_path / "expense3.parquet")
    on3: dict[str, Any] = {"rf": "rf_cur", "cpi": "cpi_cur", "expense": str(expense3)}
    cur = [
        chain.member(tmp_path / "cur-e", A_NOW, **on3),
        chain.member(tmp_path / "cur-f", B, scale=1.5, **on3),
    ]
    with pytest.raises(
        LoadError,
        match=f"basis cur run {cur[0].run_id} and basis new run {chain.new[0].run_id} read table "
        "expense from different files.*non-financial table",
    ):
        chain.roll(out, bases=chain.bases(cur=cur))


def test_the_locked_in_basis(chain: Chain, tmp_path: Path) -> None:
    """The groups file's three locked-in hashes are verified against the metadata of the
    stage's locked-in curve (source_sha256, ilp_sha256) and of the locked-in runs' CPI path
    (rec_sha256), and recorded in the manifest."""
    out = tmp_path / "stage"
    with pytest.raises(
        LoadError,
        match=f"halley.source_sha256 is {chain.hashes['rf_rec']}, but groups: group "
        f"IA-2025-OTHER \\(row 0\\) locks in discount_rec_sha256 {'0' * 64}",
    ) as info:
        chain.roll(out, groups=chain.groups(discount_rec_sha256="0" * 64))
    assert str(chain.files["p_l"]) in str(info.value)
    with pytest.raises(LoadError, match=f"halley.ilp_sha256 is {chain.hashes['ilp']}, but"):
        chain.roll(out, groups=chain.groups(ilp_sha256="1" * 64))
    bare = _write(curve(0.041), tmp_path / "bare.parquet")
    with pytest.raises(LoadError, match=r"carries no metadata halley\.source_sha256"):
        chain.roll(out, scenarios={"p_l": bare})
    with pytest.raises(
        LoadError,
        match=f"basis old run {chain.old[0].run_id}: scenario input cpi "
        r"\(.*cpi.parquet\): halley.rec_sha256 is .*, but .* locks in cpi_rec_sha256 2222",
    ) as info:
        chain.roll(out, groups=chain.groups(cpi_rec_sha256="2" * 64))
    assert str(chain.old[0].run_dir / "inputs" / "scenarios" / "cpi.parquet") in str(info.value)


def test_provenance_is_recomputed_bit_for_bit(
    chain: Chain, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file that names its source (halley.rebased_from, halley.derived_from) is recomputed by
    the helper registered for its key and must equal it bit for bit; with no helper it cannot
    be verified, a LoadError too. The helper finds the source by hash in the previous run's
    bases and a run by id; the manifest records each file verified."""
    monkeypatch.setattr(stage_module, "PROVENANCE_HELPERS", {})
    source = chain.hashes["rf_rec"]
    meta = {"source_sha256": source, "ilp_sha256": chain.hashes["ilp"], "rebased_from": source}
    p_l = _write(curve(0.041, meta=meta), tmp_path / "p_l.parquet")
    out = tmp_path / "stage"
    with pytest.raises(LoadError, match=f"carries halley.rebased_from = {source}, but no helper"):
        chain.roll(out, scenarios={"p_l": p_l})
    found: list[str] = []

    def rebase(check: ProvenanceCheck) -> Any:
        src = check.find_file(check.value)
        assert src.sha256 == check.value
        found.append(src.where)
        return curve(0.041, meta=check.file.metadata)

    register_provenance("rebased_from", rebase)
    verified = chain.roll(tmp_path / "verified", scenarios={"p_l": p_l})
    assert verified.manifest["provenance"] == [
        {"file": "scenario input p_l", "sha256": _sha(p_l), "key": "rebased_from", "value": source}
    ]
    # found in the directory of a basis run of the previous run
    assert str(chain.rec[0].run_dir / "inputs" / "scenarios" / "rf_df.parquet") in found[0]
    register_provenance("rebased_from", lambda check: curve(0.0412, meta=check.file.metadata))
    with pytest.raises(
        LoadError,
        match=r"is not what its source \(halley.rebased_from = .*\) gives: column rf_df, row 1",
    ):
        chain.roll(out, scenarios={"p_l": p_l})
    register_provenance("rebased_from", lambda check: 1 / 0)
    with pytest.raises(LoadError, match="raised ZeroDivisionError"):
        chain.roll(out, scenarios={"p_l": p_l})
    with pytest.raises(ValueError, match="a provenance key is one of"):
        register_provenance("copied_from", rebase)

    marked = member_points(A, rem_m=12).replace_schema_metadata(
        {"halley.derived_from": chain.rec[0].run_id}
    )
    old = [chain.member(tmp_path / "old-d", A, expense="expense_v1", points=marked), chain.old[1]]
    with pytest.raises(
        LoadError,
        match=f"basis old run {old[0].run_id}: model points .* carries halley.derived_from",
    ):
        chain.roll(out, bases=chain.bases(old=old))
    runs: list[str] = []

    def derived(check: ProvenanceCheck) -> Any:
        runs.append(check.find_run(check.value).run_id)
        return table_of(check.file)

    register_provenance("derived_from", derived)
    ok = chain.roll(tmp_path / "derived", bases=chain.bases(old=old))
    assert runs == [chain.rec[0].run_id]
    assert ok.manifest["provenance"][0]["file"] == f"basis old run {old[0].run_id}: model points"


def test_a_stage_reading_a_mean_runs_at_one_scenario(chain: Chain, tmp_path: Path) -> None:
    """A stage that reads a mean series runs at S = 1: a two-scenario scenario input of its own
    is a LoadError naming require_deterministic (D4)."""
    meta = {"source_sha256": chain.hashes["rf_rec"], "ilp_sha256": chain.hashes["ilp"]}
    two = _write(curve(0.041, n_scen=2, meta=meta), tmp_path / "p_l2.parquet")
    with pytest.raises(LoadError, match=r"scenario input p_l has J = 2.*require_deterministic"):
        chain.roll(tmp_path / "stage", scenarios={"p_l": two})


def test_a_mean_over_a_two_scenario_basis_keeps_its_pathwise_values(
    chain: Chain, tmp_path: Path
) -> None:
    """A mean over a J = 2 basis is copied with its per-scenario group totals beside it,
    ``<name>.pathwise.parquet``, labelled pathwise in the manifest; the mean is their
    sequential sum of halves; the run replays from its directory."""
    root = tmp_path / "root"
    rf2 = _write(curve(0.04, n_scen=2), tmp_path / "rf2.parquet")
    cpi2 = _write(cpi_path(0.025, n_scen=2), tmp_path / "cpi2.parquet")
    rec = chain.rec_basis(root, rf=str(rf2), cpi=str(cpi2))
    r = chain.recognise(root / "stage", bases={"cur": rec}, runs_root=root)
    entry = r.manifest["inputs"]["series"]["cu_g"]
    assert entry["basis_scenarios"] == 2
    assert entry["pathwise"]["scenarios"] == 2
    table = pq.read_table(r.run_dir / entry["pathwise"]["copy"])
    assert table.schema.metadata[b"halley.scenarios"] == b"pathwise"
    paths = np.asarray(table.column("value")).reshape(1, 25, 2).transpose(1, 0, 2)
    lanes = sorted((int(p), x) for x in rec for p in x.ids.tolist())
    for s in range(2):
        for t in (1, 12, 24):
            total = 0.0
            for pid, x in lanes:
                total += float(x.lane("cu", pid, s)[t])
            assert paths[t, 0, s] == total
    mean = np.asarray(pq.read_table(r.run_dir / entry["copy"]).column("value"))
    assert np.array_equal(mean, (0.5 * paths[:, 0, 0]) + (0.5 * paths[:, 0, 1]))
    replay = replay_aggregation(compiled(Recognise), r.run_dir, run_dir=tmp_path / "replay")
    assert replay.run_id == r.run_id


class Disclosing(Recognise):
    """A stage that declares its disclosures and reads a para 120 curve."""

    disclosures: ClassVar[dict[str, str | tuple[str, ...]]] = {
        "discount_rates": "bottom-up",
        "notes": ("monthly allocation", "C17"),
    }
    scenario_inputs = (ScenarioInput("df_cur", "Factor", gt=0),)
    outputs = (*RECOGNISE_OUTPUTS, "para_120")

    @var(domain="state")
    def para_120(self, t: Int) -> Factor:
        return self.scen.df_cur(t)


def test_the_manifest_discloses_the_confidence_level_and_the_curve(
    chain: Chain, tmp_path: Path
) -> None:
    """The disclosures block: alpha of the cur basis's ra_param copies, the df_cur file's
    sha256 and what the stage declares; cur runs on different confidence levels are a
    LoadError."""
    df_cur = _write(curve(0.045, column="df_cur"), tmp_path / "df_cur.parquet")
    r = run_aggregation(
        compiled(Disclosing),
        groups=group_row(),
        bases={"cur": chain.rec},
        scenarios={"df_cur": df_cur},
        settings=RunSettings(valuation_date=V_REC),
        run_dir=tmp_path / "stage",
    )
    assert r.manifest["disclosures"] == {
        "confidence_level": 0.75,
        "para_120_curve_sha256": _sha(df_cur),
        "discount_rates": "bottom-up",
        "notes": ["monthly allocation", "C17"],
    }
    assert r.manifest["bases"]["cur"][0]["dir"] == str(chain.rec[0].run_dir.resolve())
    ra = _write(keyless("alpha", 0.8), tmp_path / "ra.parquet")
    rec_b = chain.member(
        tmp_path / "rec-b",
        B,
        scale=1.5,
        at=V_REC,
        rf="rf_rec",
        cpi="cpi_rec",
        expense="expense_v1",
        ra=str(ra),
    )
    with pytest.raises(LoadError, match="the runs of basis cur read different confidence levels"):
        chain.recognise(tmp_path / "other", bases={"cur": [chain.rec[0], rec_b]})


def test_the_settings_round_trip_through_the_manifest(chain: Chain) -> None:
    """``RunSettings.from_dict`` reads back what a manifest records; runs_root, workers and the
    label are recorded but not in the identity."""
    settings = RunSettings.from_dict(chain.in_force.manifest["settings"])
    assert settings == RunSettings(valuation_date=V_NOW, runs_root=chain.root)
    assert settings.identity() == {
        "valuation_date": "2026-12-31",
        "horizon": None,
        "batch_size": 10_000,
        "outputs": None,
    }
    with pytest.raises(ValueError, match="workers is a positive int"):
        RunSettings(valuation_date=V_NOW, workers=0)
    assert json.loads((chain.in_force.run_dir / "manifest.json").read_text())["run_id"] == (
        chain.in_force.run_id
    )


# ----------------------------------------------- the previous run's values, units and domains


class RecogniseLate(Recognise):
    """Stores ``fcf_g`` from t = 1, a plain ``@var``: its month 0 holds the zeros the generated
    code fills a month before the domain with, not a value."""

    @var
    def fcf_g(self, t: Int) -> Money:
        return self.series.fcf_g(t)


class RecogniseRatio(Recognise):
    """Stores ``fcf_g`` as a Factor, the FCF over P_g."""

    @var(domain="state")
    def fcf_g(self, t: Int) -> Factor:
        return self.series.fcf_g(t) / self.series.p_g()


class RecogniseScalar(Recognise):
    """Stores ``fcf_g`` without a time dimension."""

    @var
    def fcf_g(self) -> Money:
        return self.series.fcf_g.at(0)


class RecogniseEach(Recognise):
    """Keeps its basis's scenarios in every series, so it runs at the basis's J."""

    series = tuple(dataclasses.replace(s, scenarios="each") for s in Release.series)


def test_a_derived_field_is_read_inside_the_previous_runs_domain_d11(
    chain: Chain, tmp_path: Path
) -> None:
    """D11: the previous run's fcf_g is read at month 0 only when its recorded domain starts
    there; a run that stores fcf_g from t = 1 holds a zero at month 0, which the loader would
    otherwise take for F_prev. The LoadError names the groups file, the row and the field."""
    root = tmp_path / "root"
    late = chain.recognise(root / "late", stage=RecogniseLate, runs_root=root)
    assert late.variable_info("fcf_g").start == 1
    assert late.lane("fcf_g", GID)[0] == 0.0
    path = _write(chain.groups(previous=late), tmp_path / "groups.parquet")
    with pytest.raises(LoadError) as info:
        chain.roll(tmp_path / "stage", groups=path, runs_root=root)
    assert str(info.value) == (
        f"groups ({path}): group {GID} (row 0): f_prev_0: the previous run {late.run_id} "
        "records fcf_g from t = 1, but the loader reads fcf_g(0) (D11)"
    )


def test_the_previous_runs_values_have_the_fields_units_and_dimensions_d11(
    chain: Chain, tmp_path: Path
) -> None:
    """D11 across runs: a previous run that stores fcf_g as a Factor, or without a time
    dimension, cannot fill f_prev_0, a Money field read at month 0; the LoadError names the
    groups row, the field and the previous run."""
    root = tmp_path / "root"
    ratio = chain.recognise(root / "ratio", stage=RecogniseRatio, runs_root=root)
    with pytest.raises(LoadError) as info:
        chain.roll(tmp_path / "stage", groups=chain.groups(previous=ratio), runs_root=root)
    assert str(info.value) == (
        f"groups: group {GID} (row 0): f_prev_0: the previous run {ratio.run_id} records fcf_g "
        "as Factor, not Money (D11 across runs)"
    )
    scalar = chain.recognise(root / "scalar", stage=RecogniseScalar, runs_root=root)
    with pytest.raises(LoadError) as info:
        chain.roll(tmp_path / "stage", groups=chain.groups(previous=scalar), runs_root=root)
    assert str(info.value) == (
        f"groups: group {GID} (row 0): f_prev_0: the previous run {scalar.run_id} records fcf_g "
        "without a time dimension"
    )


def test_the_previous_run_is_read_at_one_scenario_d4(chain: Chain, tmp_path: Path) -> None:
    """D4: the previous run's group outputs are read at one scenario, through the one check,
    require_deterministic; an aggregation run that kept a J = 2 basis's scenarios ("each"
    series) is not a previous run."""
    root = tmp_path / "root"
    rf2 = _write(curve(0.04, n_scen=2), tmp_path / "rf2.parquet")
    cpi2 = _write(cpi_path(0.025, n_scen=2), tmp_path / "cpi2.parquet")
    rec2 = chain.rec_basis(root, rf=str(rf2), cpi=str(cpi2))
    each = chain.recognise(root / "each", stage=RecogniseEach, bases={"cur": rec2}, runs_root=root)
    assert each.n_scenarios == 2
    with pytest.raises(
        LoadError,
        match=r"the loader reads the previous run's group outputs at one scenario; the previous "
        f"run {each.run_id} has J = 2 scenarios.*\\(require_deterministic\\)",
    ):
        chain.roll(tmp_path / "stage", groups=chain.groups(previous=each), runs_root=root)


def test_the_previous_run_recorded_its_membership_and_the_group(
    chain: Chain, tmp_path: Path
) -> None:
    """The previous run was made by run_aggregation, so it recorded its membership: a stage run
    through run() with group_series did not. It measured the group the groups row names: a
    membership that names the group over outputs of another (a manifest edited, which no hash
    covers) is a LoadError naming the row."""
    root = tmp_path / "root"
    series = group_series(Recognise, {"cur": chain.rec}, {GID: [1, 2, 3, 11, 12]})
    bare = run(
        compiled(Recognise),
        model_points=group_row().append_column("group_id", pa.array([GID])),
        series=series,
        settings=RunSettings(valuation_date=V_REC),
        run_dir=root / "bare",
    )
    assert bare.kind == "aggregation"
    assert "membership" not in bare.manifest
    with pytest.raises(LoadError) as info:
        chain.roll(tmp_path / "stage", groups=chain.groups(previous=bare), runs_root=root)
    assert str(info.value) == (
        f"the previous run {bare.run_id} records no membership; it was not made by run_aggregation"
    )

    elsewhere = tmp_path / "elsewhere"
    onerous = chain.recognise(
        elsewhere / "onerous",
        bases={"cur": chain.rec_basis(elsewhere, bucket="ONEROUS")},
        groups=group_row(profit_bucket="ONEROUS"),
        runs_root=elsewhere,
    )
    assert onerous.ids.tolist() == ["IA-2025-ONEROUS"]

    def rename(m: dict[str, Any]) -> None:
        groups = m["membership"]["groups"]
        groups[GID] = groups.pop("IA-2025-ONEROUS")

    _edited(onerous.run_dir, root / "onerous", rename)
    with pytest.raises(LoadError) as info:
        chain.roll(
            tmp_path / "stage", groups=chain.groups(prev_run_id=onerous.run_id), runs_root=root
        )
    assert str(info.value) == (
        f"groups: group {GID} (row 0): the previous run {onerous.run_id} measured no group {GID}; "
        "a group's key does not change (G4)"
    )


def test_a_previous_runs_series_copy_holds_the_group(chain: Chain, tmp_path: Path) -> None:
    """A previous run that stores fcf_g only as a group series is read from its copy, which
    holds the group the row names: a copy rewritten for another group (and its hash with it in
    the manifest, which no hash covers) is a LoadError naming the row and the field."""
    elsewhere = tmp_path / "elsewhere"
    as_series = chain.recognise(
        elsewhere / "series-only", runs_root=elsewhere, outputs=("csm", "lc", "p_g", "csm_0")
    )
    copy = tmp_path / "copy"
    shutil.copytree(as_series.run_dir, copy)

    def regroup(path: Path) -> str:
        table = pq.read_table(path)
        other = pa.array(["IA-2025-ONEROUS"] * table.num_rows)
        pq.write_table(table.set_column(0, "group_id", other), path)
        return _sha(path)

    entry = as_series.manifest["inputs"]["series"]["fcf_g"]
    sha, pathwise = regroup(copy / entry["copy"]), regroup(copy / entry["pathwise"]["copy"])

    def rehash(m: dict[str, Any]) -> None:
        m["inputs"]["series"]["fcf_g"]["sha256"] = sha
        m["inputs"]["series"]["fcf_g"]["pathwise"]["sha256"] = pathwise

    root = tmp_path / "root"
    _edited(copy, root / "series-only", rehash)
    with pytest.raises(LoadError) as info:
        chain.roll(tmp_path / "stage", groups=chain.groups(previous=as_series), runs_root=root)
    assert str(info.value) == (
        f"groups: group {GID} (row 0): f_prev_0: the previous run {as_series.run_id} holds no "
        f"fcf_g of group {GID}"
    )


# ----------------------------------------------------------- what a basis run's copies hold


def test_a_basis_run_is_the_run_its_directory_holds(chain: Chain, tmp_path: Path) -> None:
    """A RunResult is read through its directory, which must hold that run; a model point copy
    must hold the policies of the run's outputs, even when the manifest records its hash."""
    out = tmp_path / "stage"
    first, second = chain.cur
    wrong = dataclasses.replace(first, run_dir=second.run_dir)
    with pytest.raises(LoadError) as info:
        chain.roll(out, bases=chain.bases(cur=[wrong, second]))
    assert str(info.value) == (
        f"basis cur: run {first.run_id} was given, but its directory {second.run_dir} holds run "
        f"{second.run_id}"
    )
    copy = tmp_path / "cur-a"
    shutil.copytree(first.run_dir, copy)
    copied = copy / "inputs" / "model_points.parquet"
    pq.write_table(member_points([1, 4], rem_m=12), copied)
    sha = _sha(copied)
    edited = _edited(
        copy, tmp_path / "cur-a-2", lambda m: m["inputs"]["model_points"].update(sha256=sha)
    )
    with pytest.raises(LoadError) as info:
        chain.roll(out, bases=chain.bases(cur=[edited, second]))
    points = edited / "inputs" / "model_points.parquet"
    assert str(info.value) == (
        f"basis cur run {first.run_id}: model points ({points}): it holds other policies than "
        "the run's outputs"
    )


def test_the_loader_reads_what_a_copy_carries_not_what_the_manifest_says(
    chain: Chain, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Which inputs name a source, and the locked-in metadata, are read from the verified
    copies: a manifest whose metadata entry is dropped neither skips a provenance check nor
    fails a locked-in one."""
    monkeypatch.setattr(stage_module, "PROVENANCE_HELPERS", {})
    marked = member_points(A, rem_m=12).replace_schema_metadata(
        {"halley.derived_from": chain.rec[0].run_id}
    )
    old = chain.member(tmp_path / "old-d", A, expense="expense_v1", points=marked)
    hidden = _edited(
        old.run_dir, tmp_path / "old-hidden", lambda m: m["inputs"]["model_points"].pop("metadata")
    )
    with pytest.raises(
        LoadError,
        match=f"basis old run {old.run_id}: model points .* carries halley.derived_from = "
        f"{chain.rec[0].run_id}, but no helper",
    ):
        chain.roll(tmp_path / "stage", bases=chain.bases(old=[hidden, chain.old[1]]))
    dropped = _edited(
        chain.old[0].run_dir,
        tmp_path / "old-a",
        lambda m: m["inputs"]["scenarios"]["cpi"].pop("metadata"),
    )
    r = chain.roll(tmp_path / "ok", bases=chain.bases(old=[dropped, chain.old[1]]))
    assert r.run_id == chain.in_force.run_id
    by = r.manifest["locked_in"][GID]["verified_by"]["cpi_rec_sha256"]
    assert by[0]["file"] == f"basis old run {chain.old[0].run_id}: scenario input cpi"


def test_the_run_uses_the_stage_files_it_checked(
    chain: Chain, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stage's own files are read once: a file replaced on disk after the locked-in,
    provenance and disclosure checks does not reach the run, its copy or its hash."""
    p_l = tmp_path / "p_l.parquet"
    shutil.copyfile(chain.files["p_l"], p_l)
    checked = _sha(p_l)
    real_run = stage_module.run

    def replacing(*args: Any, **kw: Any) -> RunResult:
        _write(curve(0.05), p_l)  # after the checks, before the run reads its inputs
        return real_run(*args, **kw)

    monkeypatch.setattr(stage_module, "run", replacing)
    r = chain.roll(tmp_path / "stage", scenarios={"p_l": p_l})
    assert _sha(p_l) != checked
    entry = r.manifest["inputs"]["scenarios"]["p_l"]
    assert entry["sha256"] == checked
    assert _sha(r.run_dir / entry["copy"]) == checked
    assert r.run_id == chain.in_force.run_id
    assert np.array_equal(r.arrays["lock"], chain.in_force.arrays["lock"])


# ----------------------------------------------------------------- the locked-in basis, whole


class Locked(Recognise):
    """Locks in a basis but reads no input built from it."""

    model_point = group_schema(Field("j", "number", "Rate[Monthly]", ge=0, le=0.1), locked_in=True)


class LockedCurve(Locked):
    """Reads the locked-in curve p_l, but has no locked-in runs to read the CPI path of."""

    scenario_inputs = (ScenarioInput("p_l", "Factor", gt=0, column="rf_df"),)
    outputs = (*RECOGNISE_OUTPUTS, "lock")

    @var(domain="state")
    def lock(self, t: Int) -> Factor:
        return self.scen.p_l(t)


def test_every_locked_in_hash_is_verified_b72(chain: Chain, tmp_path: Path) -> None:
    """B72(b): each of the groups file's three hashes is verified against at least one input,
    never only recorded; the curve the old and new runs discount with (their rf_df, p_l) is
    held to the group's discount and ILP hashes, and their CPI path must name its recognition
    path."""
    out = tmp_path / "stage"
    hashes = {
        "discount_rec_sha256": "0" * 64,
        "ilp_sha256": chain.hashes["ilp"],
        "cpi_rec_sha256": chain.hashes["cpi_rec"],
    }
    with pytest.raises(LoadError) as info:
        chain.recognise(out, stage=Locked, groups=group_row(**hashes))
    assert str(info.value) == (
        f"groups: group {GID} (row 0) locks in discount_rec_sha256 {'0' * 64}, but the stage reads "
        "no input that can verify it (the stage's scenario inputs p_l and p_rec and scenario "
        "input rf_df of the runs of bases old and new); a locked-in hash is verified, not only "
        "recorded (B72(b))"
    )
    with pytest.raises(LoadError) as info:
        run_aggregation(
            compiled(LockedCurve),
            groups=group_row(**{**hashes, "discount_rec_sha256": chain.hashes["rf_rec"]}),
            bases={"cur": chain.rec},
            scenarios={"p_l": chain.files["p_l"]},
            settings=RunSettings(valuation_date=V_REC),
            run_dir=out,
        )
    assert str(info.value) == (
        f"groups: group {GID} (row 0) locks in cpi_rec_sha256 {chain.hashes['cpi_rec']}, but the "
        "stage reads no input that can verify it (scenario input cpi of the runs of bases old and "
        "new); a locked-in hash is verified, not only recorded (B72(b))"
    )

    unlocked: dict[str, Any] = {"rf": "rf_cur"}
    old = [
        chain.member(tmp_path / "old-u-a", A, expense="expense_v1", **unlocked),
        chain.member(tmp_path / "old-u-b", B, scale=1.5, expense="expense_v1", **unlocked),
    ]
    new = [
        chain.member(tmp_path / "new-u-a", A_NOW, **unlocked),
        chain.member(tmp_path / "new-u-b", B, scale=1.5, **unlocked),
    ]
    with pytest.raises(LoadError) as info:
        chain.roll(out, bases=chain.bases(old=old, new=new))
    copy = old[0].run_dir / "inputs" / "scenarios" / "rf_df.parquet"
    assert str(info.value) == (
        f"basis old run {old[0].run_id}: scenario input rf_df ({copy}): carries no metadata "
        "halley.source_sha256; the runs of basis old discount at the locked-in rates, with p_l "
        "as their rf_df, which names the files it was built from (design section 10)"
    )
    current = {"source_sha256": chain.hashes["rf_cur"], "ilp_sha256": chain.hashes["ilp"]}
    rf_named = _write(curve(0.045, meta=current), tmp_path / "rf_named.parquet")
    named: dict[str, Any] = {"rf": str(rf_named)}
    old = [
        chain.member(tmp_path / "old-n-a", A, expense="expense_v1", **named),
        chain.member(tmp_path / "old-n-b", B, scale=1.5, expense="expense_v1", **named),
    ]
    new = [
        chain.member(tmp_path / "new-n-a", A_NOW, **named),
        chain.member(tmp_path / "new-n-b", B, scale=1.5, **named),
    ]
    with pytest.raises(
        LoadError,
        match=f"basis old run {old[0].run_id}: scenario input rf_df .*: halley.source_sha256 is "
        f"{chain.hashes['rf_cur']}, but groups: group {GID} \\(row 0\\) locks in "
        f"discount_rec_sha256 {chain.hashes['rf_rec']}",
    ):
        chain.roll(out, bases=chain.bases(old=old, new=new))

    plain = _write(cpi_path(0.03), tmp_path / "cpi.parquet")
    spliced: dict[str, Any] = {"cpi": str(plain)}
    old = [
        chain.member(tmp_path / "old-c-a", A, expense="expense_v1", **spliced),
        chain.member(tmp_path / "old-c-b", B, scale=1.5, expense="expense_v1", **spliced),
    ]
    new = [
        chain.member(tmp_path / "new-c-a", A_NOW, **spliced),
        chain.member(tmp_path / "new-c-b", B, scale=1.5, **spliced),
    ]
    with pytest.raises(LoadError) as info:
        chain.roll(out, bases=chain.bases(old=old, new=new))
    copy = old[0].run_dir / "inputs" / "scenarios" / "cpi.parquet"
    assert str(info.value) == (
        f"basis old run {old[0].run_id}: scenario input cpi ({copy}): carries no metadata "
        "halley.rec_sha256; the locked-in CPI path names the recognition path it splices "
        "(in-force 5.2)"
    )


# ------------------------------------------------------------- the groups file's column types


def test_the_groups_rules_read_typed_columns(chain: Chain, tmp_path: Path) -> None:
    """The groups rules run before the schema's field checks, so they check the type of what
    they read: a number column holding strings, or a run id that is a number, is a LoadError
    naming the file and column, not a bare ValueError."""
    path = _write(chain.groups(csm_prev="abc"), tmp_path / "groups.parquet")
    with pytest.raises(LoadError) as info:
        chain.roll(tmp_path / "stage", groups=path)
    assert str(info.value) == f"groups ({path}): column csm_prev is a number column, not string"
    with pytest.raises(LoadError) as info:
        chain.roll(tmp_path / "stage", groups=chain.groups(prev_run_id=5))
    assert str(info.value) == "groups: column prev_run_id is a string column, not int64"


# -------------------------------------------------------- the group schema and disclosures


J_FIELD = Field("j", "number", "Rate[Monthly]", ge=0, le=0.1)


def _with_field(schema: Schema, replacement: Any) -> Schema:
    """The schema with the field or derived field of ``replacement``'s name replaced by it."""
    fields = tuple(f for f in schema.fields if f.name != replacement.name)
    if isinstance(replacement, Derived):
        return dataclasses.replace(schema, fields=fields, derived=(replacement,))
    return dataclasses.replace(schema, fields=(*fields, replacement))


class Dated(Recognise):
    """Declares tau0, a field the loader derives from the previous run's dates, without the
    prev_run_id that names the previous run."""

    model_point = group_schema(J_FIELD, int_field("tau0", ge=0))


class MoneyTau(Recognise):
    """Declares tau, a month count, as Money."""

    model_point = _with_field(group_schema(J_FIELD, previous=True), Field("tau", "number", "Money"))


class CountedPrev(Recognise):
    """Declares f_prev_0, the previous run's FCF, as an Int."""

    model_point = _with_field(group_schema(J_FIELD, previous=True), int_field("f_prev_0"))


class DerivedTau(Recognise):
    """Derives tau0 in its schema, which the loader fills too."""

    model_point = _with_field(
        group_schema(J_FIELD, previous=True),
        Derived("tau0", "Int", lambda ctx: np.zeros(1, dtype=np.int64)),
    )


def test_the_fields_the_loader_fills_are_declared_as_it_fills_them_d11(tmp_path: Path) -> None:
    """D11 at build: a group field the loader fills is declared with the kind and unit it
    fills, never derived by the schema, and a field derived from the previous run comes with
    prev_run_id; otherwise a BuildError at the stage class."""
    with pytest.raises(BuildError, match="declares tau0, which the loader derives from") as info:
        build(Dated)
    assert "no prev_run_id to name that run" in str(info.value)
    assert info.value.loc is not None
    assert info.value.loc.function == "Dated"
    with pytest.raises(UnitError) as info:
        build(MoneyTau)
    assert info.value.message == (
        "MoneyTau.model_point declares tau as number Money, but the loader fills it with a month "
        "count from the dates: declare it int Int (D11)"
    )
    assert info.value.loc is not None
    assert info.value.loc.function == "MoneyTau"
    with pytest.raises(UnitError) as info:
        build(CountedPrev)
    assert info.value.message == (
        "CountedPrev.model_point declares f_prev_0 as int Int, but the loader fills it with the "
        "previous run's fcf_g, an amount: declare it number Money (D11)"
    )
    with pytest.raises(BuildError, match=r"DerivedTau\.model_point derives tau0, which the"):
        build(DerivedTau)


class Claiming(Recognise):
    """Declares the confidence level, which the data gives."""

    disclosures: ClassVar[dict[str, str | tuple[str, ...]]] = {"confidence_level": "0.99"}


class Mistyped(Recognise):
    """Declares a note that is not a string."""

    disclosures: ClassVar[Any] = {"notes": 1}


def test_the_disclosures_a_stage_declares_are_checked_at_build() -> None:
    """A stage's disclosures name neither data-derived entry (which they would replace) and
    hold strings or tuples of strings (which the manifest writes); a BuildError at the class,
    before anything runs."""
    with pytest.raises(BuildError) as info:
        build(Claiming)
    assert info.value.message == (
        "Claiming.disclosures declares confidence_level, which the loader reads from the data "
        "(the cur basis's alpha, the df_cur file); a stage declares its approach strings and notes"
    )
    assert info.value.loc is not None
    assert info.value.loc.function == "Claiming"
    with pytest.raises(BuildError, match=r"Mistyped.disclosures: notes is a string or a tuple"):
        build(Mistyped)


RA_LEVEL = Table("ra_param", keys=(), columns=(Column("level", "Prob"),))


class MemberLevel(Member):
    """A ``Member`` whose ``ra_param`` table holds its confidence level under another name."""

    assumption_tables = (EXPENSE, ILP, RA_LEVEL)


def test_the_confidence_level_is_disclosed_or_the_run_stops(chain: Chain, tmp_path: Path) -> None:
    """A cur run that reads ra_param must hold one numeric alpha in it: the disclosure does not
    vanish silently. The LoadError names the copy."""
    compiled_level = build(MemberLevel)

    def rec_b(name: str, ra_param: Any) -> RunResult:
        return run(
            compiled_level,
            model_points=member_points(B, rem_m=24, scale=1.5),
            tables={
                "expense": chain.files["expense_v1"],
                "ilp": chain.files["ilp"],
                "ra_param": ra_param,
            },
            scenarios={"rf_df": chain.files["rf_rec"], "cpi": chain.files["cpi_rec"]},
            settings=RunSettings(valuation_date=V_REC, horizon=24),
            run_dir=tmp_path / name,
        )

    level = rec_b("level", keyless("level", 0.75))
    with pytest.raises(LoadError) as info:
        chain.recognise(tmp_path / "stage", bases={"cur": [chain.rec[0], level]})
    copy = level.run_dir / "inputs" / "tables" / "ra_param.parquet"
    assert str(info.value) == (
        f"basis cur run {level.run_id}: table ra_param ({copy}): column alpha is missing; the "
        "cur basis's ra_param holds the confidence level the run discloses (para 119)"
    )
    worded = keyless("level", 0.75).append_column("alpha", pa.array(["high"]))
    named = rec_b("named", worded)
    with pytest.raises(LoadError, match="column alpha is a number column; the cur basis's"):
        chain.recognise(tmp_path / "stage", bases={"cur": [chain.rec[0], named]})


# --------------------------------------------------------------------- a replay reproduces


def test_a_replay_that_does_not_reproduce_its_run_is_an_error_d21(
    chain: Chain, tmp_path: Path
) -> None:
    """D21: a replay compares itself with the run it replays. Outputs that differ with nothing
    in the backend block changed, or a run id that differs with the same backend and NumPy, is
    a RunError; with another environment the difference is recorded, not raised."""
    source = tmp_path / "run"
    shutil.copytree(chain.in_force.run_dir, source)
    monthly = source / "outputs" / "monthly.parquet"
    table = pq.read_table(monthly)
    csm = table.column("csm").to_pylist()
    csm[3] += 1.0
    index = table.column_names.index("csm")
    pq.write_table(table.set_column(index, "csm", pa.array(csm)), monthly)
    rewritten = _edited(
        source,
        tmp_path / "rewritten",
        lambda m: m["outputs"].update(monthly_sha256=_sha(monthly)),
    )
    with pytest.raises(
        RunError,
        match="does not reproduce it with the same backend, NumPy, pyarrow, Python and platform: "
        "monthly outputs sha256",
    ):
        replay_aggregation(compiled(Roll), rewritten, run_dir=tmp_path / "replay")
    written = json.loads((tmp_path / "replay" / "manifest.json").read_text(encoding="utf-8"))
    assert written["replayed_from"]["matches"] is False
    assert written["replayed_from"]["environment"] == {}

    elsewhere = _edited(
        rewritten, tmp_path / "elsewhere", lambda m: m["backend"].update(pyarrow="0.0")
    )
    replay = replay_aggregation(compiled(Roll), elsewhere, run_dir=tmp_path / "replay-2")
    record = replay.manifest["replayed_from"]
    assert record["matches"] is False
    assert record["environment"] == {"pyarrow": {"run": "0.0", "replay": pa.__version__}}
    assert record["differences"][0].startswith("monthly outputs sha256")
    assert replay.run_id == chain.in_force.run_id

    renamed = _edited(
        chain.in_force.run_dir, tmp_path / "renamed", lambda m: m.update(run_id="0" * 16)
    )
    with pytest.raises(
        RunError,
        match=f"with the same backend and NumPy, which the run id covers: run id "
        f"{chain.in_force.run_id} against {'0' * 16}",
    ):
        replay_aggregation(compiled(Roll), renamed, run_dir=tmp_path / "replay-3")
