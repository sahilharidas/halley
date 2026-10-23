# pyright: basic
"""Running a compiled model: batches, outputs, the run directory and its manifest (D21).

A run directory holds everything needed to reproduce the run and to read it back later without
the model: ``manifest.json``, a copy of every input under ``inputs/``, the generated backend
source under ``generated/``, the outputs under ``outputs/`` keyed on the schema's id field with a
versioned schema, and the invariant results. The manifest records the model source hash and git
commit, the IR hash, every input's hash and ``halley.*`` metadata, the backend and its versions,
the settings, and per stored variable its unit, domain start and role. The run id is a hash of
the IR, the inputs' hashes (not their paths, so the same bytes anywhere give the same id), the
settings and the backend. An aggregation stage runs through the same function: its model points
are groups, its run directory also keeps a copy of every group series it read under
``inputs/series/`` with the series' provenance in the manifest, its run id covers each series by
the hash of its values, and its outputs are marked ``outputs.kind = "aggregation"``. A stage run
made from run directories (:func:`~halley.aggregation.stage.run_aggregation`) passes a
:class:`StageRecord` too: its groups file, derived group fields and membership are copied, its
manifest gains the record's blocks, and its run id is the stage's identity; a stage whose groups
name a previous run or lock in a basis runs only that way. :func:`verified_run` reads a run
directory once and checks every file its manifest records a hash for, which a later run does
before it reads a run, and then takes what it reads from the bytes it checked.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.metadata
import io
import json
import platform
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray

from halley import ir
from halley.authoring.contracts import FROM_PREVIOUS, GROUP_KEY, LOCKED_IN_FIELDS
from halley.codegen import CompiledModel, backend_named
from halley.errors import LoadError
from halley.runtime.invariants import InvariantResult, merge_results
from halley.runtime.loader import (
    Inputs,
    LoadedSeries,
    ReadFile,
    Source,
    copy_entry,
    load_inputs,
    parse_copy,
    verify_copy,
)
from halley.runtime.workers import run_batches

OUTPUT_SCHEMA_VERSION = 1
"""Bumped when the layout of the output files changes. A later aggregation stage reads a
previous run's outputs by this version and the hashes in its manifest."""


def _halley_version() -> str:
    try:
        return importlib.metadata.version("halley")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - not installed
        return "0"


IDENTITY_VERSIONS = ("numpy", "numba", "llvmlite")
"""The entries of a backend's versions (``Backend.versions``) a run id covers beside its name:
NumPy for every backend, as before backends were chosen, so a NumPy run's id does not move, and
Numba and llvmlite for the Numba backend, which compile its kernels. The manifest's ``backend``
block records every entry."""


def backend_identity(name: str, versions: Mapping[str, str]) -> dict[str, str]:
    """The ``backend`` entry of a run identity: the backend's name and the versions of
    :data:`IDENTITY_VERSIONS` it reports (``{"name": "numpy", "numpy": ...}`` for NumPy)."""
    return {"name": name, **{k: versions[k] for k in IDENTITY_VERSIONS if k in versions}}


NOT_IN_IDENTITY = ("label", "runs_root", "workers")
"""Settings a manifest records but a run id does not cover: they say where a run is filed and
how it was computed, not what it computed (design sections 9 and 12)."""


@dataclass(frozen=True, slots=True)
class RunSettings:
    valuation_date: dt.date
    horizon: int | None = None
    """Months to project; default the largest value of the schema's horizon field."""
    batch_size: int = 10_000
    outputs: tuple[str, ...] | None = None
    """Variables to store; default the model's ``outputs``, or every variable."""
    label: str = ""
    runs_root: Path | None = None
    """The directory whose run directories an aggregation stage searches for the previous run
    its groups name (``prev_run_id``), and against which its manifest records the directory of
    every run it read (design section 9). Recorded, not in the run id."""
    workers: int = 1
    """Processes to spread the batches over (design section 12, :mod:`halley.runtime.workers`):
    at 1, or with a single batch, every batch runs in the calling process; above, in
    ``min(workers, batches)`` worker processes started with the spawn method, each of which
    imports the model's class from its module (so it must be defined at the top level of an
    importable module, and a script must guard its entry point with
    ``if __name__ == "__main__":``). The outputs, invariant results and errors are the same bit
    for bit at the same ``batch_size`` (D5, D19), so it is recorded but not in the run id. An
    aggregation stage's batches are its groups: a stage of one group (G1) runs in one process."""

    def __post_init__(self) -> None:
        if isinstance(self.workers, bool) or not isinstance(self.workers, int) or self.workers < 1:
            raise ValueError(f"workers is a positive int, not {self.workers!r}")
        if self.runs_root is not None and not isinstance(self.runs_root, Path):
            object.__setattr__(self, "runs_root", Path(self.runs_root))

    def as_dict(self) -> dict[str, Any]:
        return {
            "valuation_date": self.valuation_date.isoformat(),
            "horizon": self.horizon,
            "batch_size": self.batch_size,
            "outputs": list(self.outputs) if self.outputs is not None else None,
            "label": self.label,
            "runs_root": None if self.runs_root is None else str(self.runs_root),
            "workers": self.workers,
        }

    def identity(self) -> dict[str, Any]:
        """The settings a run id covers: all but :data:`NOT_IN_IDENTITY`."""
        return {k: v for k, v in self.as_dict().items() if k not in NOT_IN_IDENTITY}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunSettings:
        """The settings a manifest records (:meth:`as_dict`), as settings again; a manifest
        written before ``runs_root`` and ``workers`` were recorded has neither."""
        outputs = d.get("outputs")
        root = d.get("runs_root")
        return cls(
            valuation_date=dt.date.fromisoformat(str(d["valuation_date"])),
            horizon=None if d.get("horizon") is None else int(d["horizon"]),
            batch_size=int(d.get("batch_size", 10_000)),
            outputs=None if outputs is None else tuple(str(o) for o in outputs),
            label=str(d.get("label", "")),
            runs_root=None if root is None else Path(str(root)),
            workers=int(d.get("workers", 1)),
        )


@dataclass(frozen=True, slots=True)
class VariableInfo:
    """What a run's manifest records about a stored variable (``outputs.variables.<name>``):
    whether it has a time dimension, its unit's name, the first month of its domain and its
    role. An aggregation stage checks a group series' declared unit and takes its domain from
    here, for a run of this session and a loaded one alike."""

    time: bool
    unit: str
    start: int
    role: str | None


@dataclass(slots=True)
class RunResult:
    run_dir: Path
    run_id: str
    manifest: dict[str, Any]
    ids: NDArray[Any]
    H: int
    n_scenarios: int
    arrays: dict[str, NDArray[Any]]
    """Every kept variable, shaped ``(time or 1, policy or 1, scenario or 1)``."""
    has_time: dict[str, bool]
    graph: ir.Graph | None = None
    compiled: CompiledModel | None = None
    inputs: Inputs | None = None
    invariants: list[InvariantResult] = field(default_factory=list)
    """The built-in invariants then the product's, each merged over every batch."""
    manifest_sha256: str | None = None
    """The sha256 of the manifest's bytes: as :func:`run` wrote them, or as :func:`load_run`
    read (and, with ``verify``, checked) them. A later stage records it for the run it read
    (design section 9), so the hash recorded is that of the manifest that was used."""

    @property
    def n_policies(self) -> int:
        return len(self.ids)

    @property
    def kind(self) -> str:
        """``"projection"`` or ``"aggregation"``, as the manifest records it."""
        return str(self.manifest.get("outputs", {}).get("kind", "projection"))

    @property
    def variables(self) -> list[str]:
        return sorted(self.arrays)

    def index(self, policy_id: object) -> int:
        matches = np.flatnonzero(self.ids == policy_id)
        if matches.size == 0:
            raise KeyError(f"no policy {policy_id!r} in this run")
        return int(matches[0])

    def var(self, name: str) -> NDArray[Any]:
        """A variable broadcast to ``(time or 1, policies, scenarios)``."""
        a = self.arrays[name]
        return np.broadcast_to(a, (a.shape[0], self.n_policies, self.n_scenarios))

    def lane(self, name: str, policy_id: object, scenario: int = 0) -> NDArray[Any]:
        """One policy's values over time (or its single value for a scalar variable)."""
        return self.var(name)[:, self.index(policy_id), scenario]

    def variable_info(self, name: str) -> VariableInfo:
        """What the manifest records about a stored variable; a LoadError for one the run did
        not store, or for a manifest written before units and domains were recorded."""
        info = self.manifest.get("outputs", {}).get("variables", {}).get(name)
        if info is None:
            raise LoadError(f"run {self.run_id} did not store {name}; add it to outputs")
        if "unit" not in info or "start" not in info:
            raise LoadError(
                f"run {self.run_id} records no unit or domain for {name}: its manifest predates "
                "them; rerun it"
            )
        role = info.get("role")
        return VariableInfo(
            bool(info["time"]),
            str(info["unit"]),
            int(info["start"]),
            None if role is None else str(role),
        )


def _git_commit(path: Path) -> str | None:
    git = shutil.which("git")
    if git is None:
        return None
    try:
        proc = subprocess.run(
            [git, "-C", str(path.parent), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def _sha256_file(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _json_safe(value: object) -> object:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return repr(value)


def _copy_input(f: ReadFile, dest: Path) -> None:
    """Copy an input into the run directory: the bytes it was read from, which its recorded
    sha256 hashes, not whatever its path holds by now."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if f.raw is not None:
        dest.write_bytes(f.raw)
    elif f.path is not None:
        shutil.copyfile(f.path, dest)


def _input_entry(f: ReadFile, copy_to: str, **extra: Any) -> dict[str, Any]:
    return {
        "path": str(f.path) if f.path is not None else None,
        "copy": copy_to,
        "sha256": f.sha256,
        "metadata": dict(f.metadata),
        **extra,
    }


def _without_paths(inputs: dict[str, Any]) -> dict[str, Any]:
    """The manifest's inputs block without each input's own ``path`` string: a run is addressed
    by what it read, not where the files were. Only the entry level is stripped
    (``model_points``, ``tables.<name>``, ``scenarios.<name>``), never anything below or beside
    it, so a table or scenario input named ``path`` stays in the identity, hash and all."""

    def entry(e: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in e.items() if k != "path"}

    out = {
        "model_points": entry(inputs["model_points"]),
        "tables": {name: entry(e) for name, e in inputs["tables"].items()},
        "scenarios": {name: entry(e) for name, e in inputs["scenarios"].items()},
    }
    if "series" in inputs:
        # an aggregation stage: each series by the hash of its values, not its copy's bytes
        # (which depend on the parquet writer) nor its source runs' directories; a mean's
        # pathwise values are a record beside it, which the stage does not read
        out["series"] = {
            name: {k: v for k, v in e.items() if k not in ("source_runs", "sha256", "pathwise")}
            for name, e in inputs["series"].items()
        }
    return out


@dataclass(frozen=True, slots=True)
class StageRecord:
    """What an aggregation run made from run directories records beside what every run does
    (design section 9; :func:`halley.aggregation.stage.run_aggregation` makes it, and
    :func:`~halley.aggregation.stage.replay_aggregation` takes it from the run it replays).

    ``groups`` is the groups file as given, copied to ``inputs/groups.parquet``; ``derived``
    the table of the group fields the loader derived (``group_id`` and the previous-run
    quantities), copied to ``inputs/derived_groups.parquet``; ``membership`` every member of
    every basis (``policy_id``, ``basis``, ``group_id``), copied to
    ``inputs/membership.parquet`` and hashed like every copy, so that a later run reads the
    members it checks G4 against from a verified file; ``blocks`` the manifest's top-level
    blocks: ``previous`` (the previous run's ``run_id``, its directory and its manifest's
    sha256, or ``None``), ``bases`` (the same for every run of every basis, with its outputs'
    hashes), ``locked_in``, ``provenance`` and ``disclosures`` (and, in a replay,
    ``replayed_from``). Such a run's id covers the IR, the groups file, every series' values,
    the membership's rows, every scenario and table file of the stage, the previous run's id and
    manifest hash, the disclosures, the settings but those of :data:`NOT_IN_IDENTITY`, the
    backend and the horizon: not the bases' directories, which may move, nor the copies' bytes,
    which name their writer (:func:`_stage_identity`)."""

    groups: ReadFile
    derived: Any
    """A ``pyarrow.Table`` keyed on ``group_id``, in the groups file's row order."""
    membership: ReadFile
    """The members by basis, as read (its bytes are the copy's)."""
    blocks: dict[str, Any]

    @property
    def previous(self) -> dict[str, Any] | None:
        return self.blocks.get("previous")


def _require_no_previous_run(graph: ir.Graph) -> None:
    """A stage whose groups name a previous run, carry fields derived from one or lock in a
    basis by hash runs through :func:`~halley.aggregation.stage.run_aggregation`, which reads
    that run, derives those fields from it and verifies the hashes (design section 9:
    previous-run quantities are "never retyped"). :func:`run` would take them as given in the
    model points, with no previous run read and no check made, so it refuses such a stage. A
    stage without them (one that sums its bases alone) runs through :func:`run` with
    ``series=group_series(...)``."""
    previous = [name for name in ("prev_run_id", *FROM_PREVIOUS) if name in graph.fields]
    locked = [name for name in LOCKED_IN_FIELDS if name in graph.fields]
    if not previous and not locked:
        return
    what = []
    if previous:
        what.append(f"reads a previous run ({', '.join(previous)})")
    if locked:
        what.append(f"locks in a basis ({', '.join(locked)})")
    raise LoadError(
        f"{graph.model} {' and '.join(what)}; run it with "
        "halley.aggregation.stage.run_aggregation, which reads the previous run, derives these "
        "fields from it and verifies the hashes, not with run(), which would take them as typed "
        "(design section 9)"
    )


def _require_no_member_rules(model: type[Any]) -> None:
    """A stage that declares member rules (the annuity's G3 and G5, design section 9) checks
    every member of a basis on the basis's runs, which only
    :func:`~halley.aggregation.stage.run_aggregation` reads; :func:`run` sees the group series
    alone, so it refuses such a stage rather than skip the rules."""
    rules = getattr(model, "member_rules", ())
    if not rules:
        return
    names = ", ".join(str(getattr(rule, "name", rule)) for rule in rules)
    raise LoadError(
        f"{model.__qualname__} checks every member of its bases ({names}); run it with "
        "halley.aggregation.stage.run_aggregation, which reads the bases' runs the rules check, "
        "not with run(), which sees only the group series (design section 9)"
    )


def _require_no_recognised_curve(model: type[Any]) -> None:
    """A stage of groups recognised at the valuation date (``recognised_at_valuation``, as
    ``NewGroup`` declares) locks in the composed curve of its cur basis's runs: P_L = P (annuity
    spec, "Locked-in rates"). Only :func:`~halley.aggregation.stage.run_aggregation` verifies
    its ``p_l`` against the files those runs read; :func:`run` sees the group series alone, and
    no invariant can see a wrong ``p_l`` (the accretion is on both sides of I11 and M1), so it
    refuses such a stage rather than skip the check."""
    if getattr(model, "recognised_at_valuation", False) is not True:
        return
    raise LoadError(
        f"{model.__qualname__} locks in the composed curve of its cur basis's runs (P_L = P), "
        "which only run_aggregation can verify; run it with "
        "halley.aggregation.stage.run_aggregation, not with run(), which sees only the group "
        "series (annuity spec, 'Locked-in rates'; design section 9)"
    )


def _require_no_group_schema(graph: ir.Graph) -> None:
    """A stage whose groups carry the group key or the recognition date (a group schema of
    in-force 3.3, :func:`~halley.authoring.contracts.group_schema`) runs through
    :func:`~halley.aggregation.stage.run_aggregation`, whose loader derives each group's id from
    its key, checks G1 (one group per run), the 3.3 rules on the groups file and every member's
    group and issue date against the bases' runs, and records the membership a later period
    checks G4 against. :func:`run` sees the group series alone and would take the groups as
    typed, so it refuses such a stage. A stage keyed on ``group_id`` alone runs through
    :func:`run` with ``series=group_series(...)``."""
    fields = [name for name in (*GROUP_KEY, "recognition_date") if name in graph.fields]
    if not fields:
        return
    raise LoadError(
        f"{graph.model} measures groups of in-force 3.3 ({', '.join(fields)}); run it with "
        "halley.aggregation.stage.run_aggregation, which derives each group's id from its key, "
        "checks G1, the groups file's rules and every member against the bases' runs and records "
        "the membership, not with run(), which sees only the group series (design section 9)"
    )


def _stage_identity(
    manifest: dict[str, Any],
    record: StageRecord,
    settings: RunSettings,
    compiled: CompiledModel,
    horizon: int,
) -> dict[str, Any]:
    """The run identity of a stage run made from run directories (design section 9): the IR,
    the groups file, every series by the hash of its values, the membership by the hash of its
    rows, every scenario and table file the stage read, the previous run's id and manifest hash,
    the disclosures (what the stage declares, the confidence level and the para 120 curve's
    hash), the settings a run id covers, the backend and the horizon. Nothing in it moves when
    the bases' directories do or when a replay reads the copies instead of the files: the
    disclosures' ``para_120_curve_verified`` names the copies it compared, and its hashes are
    ``df_cur``'s metadata, which ``df_cur``'s own hash covers."""
    inputs = manifest["inputs"]
    previous = record.previous
    disclosed: object = record.blocks.get("disclosures") or {}
    disclosures = (
        {k: v for k, v in disclosed.items() if k != "para_120_curve_verified"}
        if isinstance(disclosed, dict)
        else disclosed
    )
    return {
        "kind": "aggregation",
        "ir_sha256": manifest["ir_sha256"],
        "groups": record.groups.sha256,
        "series": {name: e["values_sha256"] for name, e in inputs["series"].items()},
        "membership": _membership_sha256(record.membership),
        "scenarios": {name: e["sha256"] for name, e in inputs["scenarios"].items()},
        "tables": {name: e["sha256"] for name, e in inputs["tables"].items()},
        "previous": None
        if previous is None
        else {"run_id": previous["run_id"], "manifest_sha256": previous["manifest_sha256"]},
        "disclosures": _json_safe(disclosures),
        "settings": settings.identity(),
        "backend": backend_identity(compiled.backend, manifest["backend"]),
        "horizon": horizon,
    }


def _membership_sha256(membership: ReadFile) -> str:
    """The run identity's hash of a stage's membership (``policy_id``, ``basis``, ``group_id``):
    its columns and its rows in the file's order, by value (a policy id's type included), not
    the copy's bytes, which name the parquet writer. Two runs whose groups hold other members
    measured other groups, though their group totals may agree."""
    table = membership.table
    columns = [str(c) for c in table.column_names]
    rows = [[row[c] for c in columns] for row in table.to_pylist()]
    body = json.dumps({"columns": columns, "rows": _json_safe(rows)}, sort_keys=True)
    return hashlib.sha256(body.encode()).hexdigest()


def _write_table(table: pa.Table, dest: Path) -> str:
    """Write a table the run made (not one it was given) and return its bytes' sha256."""
    buf = io.BytesIO()
    pq.write_table(table, buf)
    raw = buf.getvalue()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _series_table(s: LoadedSeries, id_field: str) -> pa.Table:
    """A group series in long form for the run directory (design section 9): the group id,
    ``scenario``, ``t`` and ``value``, group-major, then month, then scenario."""
    n_t, n_g, n_s = s.values.shape
    meta = {
        b"halley.series": s.name.encode(),
        b"halley.unit": s.unit.name.encode(),
        b"halley.start": str(s.start).encode(),
        b"halley.scenarios": s.scenarios.encode(),
    }
    return pa.table(
        {
            id_field: np.repeat(np.asarray(s.group_ids, dtype=object), n_t * n_s),
            "scenario": np.tile(np.arange(n_s, dtype=np.int32), n_g * n_t),
            "t": np.tile(np.repeat(np.arange(n_t, dtype=np.int32), n_s), n_g),
            "value": np.ascontiguousarray(s.values.transpose(1, 0, 2)).reshape(-1),
        }
    ).replace_schema_metadata(meta)


def _write_series(s: LoadedSeries, id_field: str, dest: Path) -> str:
    """Write a group series' copy and return the sha256 of its bytes, which the manifest
    records for the copy's integrity."""
    buf = io.BytesIO()
    pq.write_table(_series_table(s, id_field), buf)
    raw = buf.getvalue()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _series_values_sha256(s: LoadedSeries) -> str:
    """The run identity's hash of a group series: what it holds, not the bytes of its copy
    (D21). The parquet file embeds its writer's version and follows pyarrow's defaults, so a
    hash of the file would move the run id with a pyarrow upgrade, and a replay that rewrites
    the copy could too. This one covers the name, the unit, the domain start, the scenario
    treatment, the group ids in order, the shape and the values as C-order little-endian
    float64 bytes (so ``-0.0`` and ``0.0`` differ)."""
    values = np.ascontiguousarray(s.values, dtype="<f8")
    head = {
        "name": s.name,
        "unit": s.unit.name,
        "start": s.start,
        "scenarios": s.scenarios,
        "group_ids": _json_safe(np.asarray(s.group_ids, dtype=object).tolist()),
        "dtype": "<f8",
        "shape": list(values.shape),
    }
    digest = hashlib.sha256(json.dumps(head, sort_keys=True).encode())
    digest.update(b"\n")
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def run(
    compiled: CompiledModel,
    *,
    model_points: Source,
    tables: dict[str, Source] | None = None,
    scenarios: dict[str, Source] | None = None,
    settings: RunSettings,
    run_dir: str | Path,
    series: Mapping[str, LoadedSeries] | None = None,
    stage_record: StageRecord | None = None,
) -> RunResult:
    """Load and validate the inputs, project every policy in batches, check the built-in and
    the product's invariants on every batch, and write the run directory.

    An aggregation stage runs the same way with its groups as the model points and its group
    series in ``series`` (:func:`~halley.aggregation.group_series`); its run directory keeps a
    copy of every series (and of a mean's pathwise values) and its outputs are keyed on the
    groups' id field, with ``outputs.kind`` ``"aggregation"``. A stage run made from run
    directories (:func:`~halley.aggregation.stage.run_aggregation`) also passes its
    ``stage_record``: the groups file and the fields the loader derived are copied too, the
    record's blocks enter the manifest, and the run id is the stage's identity
    (:class:`StageRecord`). A stage whose groups name a previous run or lock in a basis, that
    declares member rules, whose groups are recognised at the valuation date, or whose groups
    carry the group key or the recognition date of in-force 3.3, runs only that way
    (:func:`_require_no_previous_run`, :func:`_require_no_member_rules`,
    :func:`_require_no_recognised_curve`, :func:`_require_no_group_schema`)."""
    graph = compiled.graph
    model = compiled.model
    run_path = Path(run_dir)
    if (run_path / "manifest.json").exists():
        raise LoadError(f"{run_path} already holds a run; use a new directory for each run")
    stage = model.series_declarations() is not None
    if series and not stage:
        raise LoadError(f"{graph.model} is a projection; group series are read by a stage")
    if stage_record is not None and not stage:
        raise LoadError(f"{graph.model} is a projection; a stage record belongs to a stage")
    if stage and stage_record is None:
        _require_no_previous_run(graph)
        _require_no_member_rules(model)
        _require_no_recognised_curve(model)
        _require_no_group_schema(graph)
    lanes = "groups" if stage else "policies"  # what the messages call the model points
    inputs = load_inputs(
        graph,
        compiled.schema,
        model_points=model_points,
        tables=tables,
        scenarios=scenarios,
        valuation_date=settings.valuation_date,
        horizon=settings.horizon,
        table_contracts={t.name: t for t in model.assumption_tables},
        scenario_contracts={s.name: s for s in model.scenario_inputs},
        input_rules=model.input_rules,
        series=series,
        lanes=lanes,
        model_points_label="model points" if stage_record is None else stage_record.groups.where,
        run_off=getattr(model, "reads_to_run_off", False) is True,  # Aggregation's (D5)
    )
    horizon = inputs.H
    n_pol = len(inputs.ids)
    n_scen = inputs.n_scenarios
    needed = sorted(compiled.schedule.needed)
    backend = backend_named(compiled.backend)
    # in this process or over worker processes; either way in ascending batch order, the first
    # failing batch's error raised (halley.runtime.workers)
    done = run_batches(
        compiled, inputs, tuple(needed), lanes, settings.batch_size, settings.workers
    )
    invariants = merge_results([results for _arrays, results in done])
    arrays: dict[str, NDArray[Any]] = {}
    for name in needed:
        v = graph.variables[name]
        dims = v.dims or frozenset()
        if ir.POLICY in dims:
            arrays[name] = np.concatenate([out[name] for out, _results in done], axis=1)
        else:
            arrays[name] = done[0][0][name]
    has_time = {name: graph.variables[name].has_time for name in needed}

    stored = list(settings.outputs or graph.outputs or needed)
    for name in stored:
        if name not in arrays:
            raise LoadError(f"output {name!r} is not a variable of the model")

    # ------------------------------------------------------------------ run directory
    run_path.mkdir(parents=True, exist_ok=True)
    (run_path / "generated").mkdir(exist_ok=True)
    (run_path / "generated" / backend.artefact).write_text(compiled.source, encoding="utf-8")
    _copy_input(inputs.model_points.file, run_path / "inputs" / "model_points.parquet")
    for name, t in inputs.tables.items():
        _copy_input(t.file, run_path / "inputs" / "tables" / f"{name}.parquet")
    for name, s in inputs.scenarios.items():
        _copy_input(s.file, run_path / "inputs" / "scenarios" / f"{name}.parquet")
    id_field = compiled.schema.id_field
    series_inputs: dict[str, Any] = {}
    for name, y in inputs.series.items():
        copy = f"inputs/series/{name}.parquet"
        series_inputs[name] = {
            **y.provenance,  # basis, variables, valuation date, source runs, members
            "copy": copy,
            "sha256": _write_series(y, id_field, run_path / copy),  # the copy's bytes
            "values_sha256": _series_values_sha256(y),  # what the run id covers
            "unit": y.unit.name,
            "time": y.has_time,
            "start": y.start,
            "scenarios": y.scenarios,
            "months": [0, y.horizon],
        }
        if y.pathwise is not None:
            # the per-scenario group totals the mean averaged, labelled pathwise: a record
            # beside the series, not read by the stage and not in the run id
            pathwise = LoadedSeries(
                name, y.pathwise, y.unit, y.has_time, y.start, y.group_ids, "pathwise"
            )
            pcopy = f"inputs/series/{name}.pathwise.parquet"
            series_inputs[name]["pathwise"] = {
                "copy": pcopy,
                "sha256": _write_series(pathwise, id_field, run_path / pcopy),
                "values_sha256": _series_values_sha256(pathwise),
                "scenarios": pathwise.n_scenarios,
            }
    stage_inputs: dict[str, Any] = {}
    if stage_record is not None:
        _copy_input(stage_record.groups, run_path / "inputs" / "groups.parquet")
        stage_inputs["groups"] = _input_entry(
            stage_record.groups, "inputs/groups.parquet", rows=stage_record.groups.table.num_rows
        )
        stage_inputs["derived_groups"] = {
            "copy": "inputs/derived_groups.parquet",
            "sha256": _write_table(
                stage_record.derived, run_path / "inputs" / "derived_groups.parquet"
            ),
            "columns": list(stage_record.derived.column_names),
        }
        members = stage_record.membership
        _copy_input(members, run_path / "inputs" / "membership.parquet")
        stage_inputs["membership"] = {
            "copy": "inputs/membership.parquet",
            "sha256": members.sha256,
            "rows": members.table.num_rows,
            "columns": list(members.table.column_names),
            "key": list(GROUP_KEY),  # what the members were grouped on
        }

    source_path = Path(graph.source_file)
    manifest: dict[str, Any] = {
        "halley": _halley_version(),
        "output_schema": OUTPUT_SCHEMA_VERSION,
        "model": {
            "name": graph.model,
            "source_file": str(source_path),
            "source_sha256": _sha256_file(source_path),
            "git_commit": _git_commit(source_path),
            "statics": _json_safe(graph.statics),
        },
        "ir_sha256": compiled.ir_hash,
        "backend": {
            "name": compiled.backend,
            **backend.versions(),
            "pyarrow": pa.__version__,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "settings": settings.as_dict(),
        "dimensions": {
            "policies": n_pol,
            "scenarios": n_scen,
            "months": horizon + 1,
            "horizon": horizon,
        },
        "inputs": {
            "model_points": _input_entry(
                inputs.model_points.file, "inputs/model_points.parquet", rows=n_pol
            ),
            "tables": {
                name: _input_entry(t.file, f"inputs/tables/{name}.parquet", version=t.version)
                for name, t in inputs.tables.items()
            },
            "scenarios": {
                name: _input_entry(
                    s.file,
                    f"inputs/scenarios/{name}.parquet",
                    scenarios=s.n_scenarios,
                    months=[s.u0, s.u_max],
                )
                for name, s in inputs.scenarios.items()
            },
            **({"series": series_inputs} if stage else {}),
            **stage_inputs,
        },
        "outputs": {
            "kind": "aggregation" if stage else "projection",
            "id_field": id_field,
            "monthly": "outputs/monthly.parquet",
            "scalars": "outputs/scalars.parquet",
            "variables": {
                name: {
                    "time": has_time[name],
                    "unit": graph.variables[name].declared_unit.name,
                    "start": graph.variables[name].start,
                    "role": graph.variables[name].role,
                }
                for name in stored
            },
        },
        "invariants": "invariants.json",
    }
    if stage_record is not None:
        manifest.update(stage_record.blocks)
        identity = _stage_identity(manifest, stage_record, settings, compiled, horizon)
    else:
        identity = {
            "ir_sha256": manifest["ir_sha256"],
            "inputs": _without_paths(manifest["inputs"]),
            "settings": settings.identity(),
            "backend": backend_identity(compiled.backend, manifest["backend"]),
            "horizon": horizon,
        }

    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    manifest["run_id"] = run_id
    manifest["created_at"] = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")

    _write_outputs(
        run_path, manifest, id_field, inputs.ids, arrays, has_time, stored, horizon, n_scen
    )
    manifest["outputs"]["monthly_sha256"] = _sha256_file(run_path / "outputs" / "monthly.parquet")
    manifest["outputs"]["scalars_sha256"] = _sha256_file(run_path / "outputs" / "scalars.parquet")
    (run_path / "invariants.json").write_text(
        json.dumps([r.as_dict() for r in invariants], indent=2), encoding="utf-8"
    )
    manifest["invariants_sha256"] = _sha256_file(run_path / "invariants.json")
    manifest_sha256 = write_manifest(run_path, manifest)
    return RunResult(
        run_path,
        run_id,
        manifest,
        inputs.ids,
        horizon,
        n_scen,
        arrays,
        has_time,
        graph,
        compiled,
        inputs,
        invariants,
        manifest_sha256,
    )


def _write_outputs(
    run_path: Path,
    manifest: dict[str, Any],
    id_field: str,
    ids: NDArray[Any],
    arrays: dict[str, NDArray[Any]],
    has_time: dict[str, bool],
    stored: list[str],
    horizon: int,
    n_scen: int,
) -> None:
    out_dir = run_path / "outputs"
    out_dir.mkdir(exist_ok=True)
    n_pol = len(ids)
    n_months = horizon + 1
    meta = {
        b"halley.output_schema": str(OUTPUT_SCHEMA_VERSION).encode(),
        b"halley.run_id": str(manifest["run_id"]).encode(),
        b"halley.model": str(manifest["model"]["name"]).encode(),
        b"halley.ir_sha256": str(manifest["ir_sha256"]).encode(),
    }
    monthly: dict[str, Any] = {
        id_field: np.repeat(ids, n_months * n_scen),
        "scenario": np.tile(np.arange(n_scen, dtype=np.int32), n_pol * n_months),
        "t": np.tile(np.repeat(np.arange(n_months, dtype=np.int32), n_scen), n_pol),
    }
    scalars: dict[str, Any] = {
        id_field: np.repeat(ids, n_scen),
        "scenario": np.tile(np.arange(n_scen, dtype=np.int32), n_pol),
    }
    for name in stored:
        a = arrays[name]
        if has_time[name]:
            full = np.broadcast_to(a, (n_months, n_pol, n_scen))
            monthly[name] = np.ascontiguousarray(full.transpose(1, 0, 2)).reshape(-1)
        else:
            scalars[name] = np.broadcast_to(a[0], (n_pol, n_scen)).reshape(-1)
    pq.write_table(pa.table(monthly).replace_schema_metadata(meta), out_dir / "monthly.parquet")
    pq.write_table(pa.table(scalars).replace_schema_metadata(meta), out_dir / "scalars.parquet")


def write_manifest(run_dir: str | Path, manifest: dict[str, Any]) -> str:
    """Write a run's ``manifest.json`` (indented, keys sorted) and return the sha256 of the bytes
    written: :func:`run` does, and so does a replay when it records whether it reproduced the
    run it replays."""
    raw = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    (Path(run_dir) / "manifest.json").write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _parse_manifest(run_path: Path, raw: bytes) -> dict[str, Any]:
    manifest_path = run_path / "manifest.json"
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LoadError(f"{manifest_path} is not a run manifest: {exc}") from None
    if not isinstance(manifest, dict):
        raise LoadError(f"{manifest_path} is not a run manifest")
    if manifest.get("output_schema") != OUTPUT_SCHEMA_VERSION:
        raise LoadError(
            f"{run_path} was written with output schema {manifest.get('output_schema')}; this "
            f"version reads {OUTPUT_SCHEMA_VERSION}"
        )
    return manifest


def _manifest_bytes(run_path: Path) -> bytes:
    try:
        return (run_path / "manifest.json").read_bytes()
    except OSError:
        raise LoadError(f"{run_path} holds no manifest.json") from None


def read_manifest(run_dir: str | Path) -> tuple[dict[str, Any], str]:
    """A run directory's manifest and the sha256 of its bytes, which another run that reads
    this one records (design section 9); a LoadError when there is none or when it was written
    with another output schema."""
    run_path = Path(run_dir)
    raw = _manifest_bytes(run_path)
    return _parse_manifest(run_path, raw), hashlib.sha256(raw).hexdigest()


def run_files(manifest: dict[str, Any]) -> list[tuple[str, str, str]]:
    """Every file of a run directory whose sha256 its manifest records, as ``(what, relative
    path, sha256)``: the copied inputs (the model points, every table and scenario file and, for
    a stage, its groups file, its derived groups, its membership and every series with a mean's
    pathwise sibling), the outputs (``outputs.monthly_sha256``, ``outputs.scalars_sha256``) and
    the invariant results (``invariants_sha256``)."""
    inputs = manifest.get("inputs", {})
    entries: list[tuple[str, dict[str, Any]]] = [("model points", inputs.get("model_points", {}))]
    entries += [(f"table {n}", e) for n, e in inputs.get("tables", {}).items()]
    entries += [(f"scenario input {n}", e) for n, e in inputs.get("scenarios", {}).items()]
    for key, what in STAGE_COPIES:
        if key in inputs:
            entries.append((what, inputs[key]))
    for n, e in inputs.get("series", {}).items():
        entries.append((f"series {n}", e))
        if "pathwise" in e:
            entries.append((f"series {n} (pathwise)", e["pathwise"]))
    out = [(what, str(e.get("copy")), str(e.get("sha256"))) for what, e in entries]
    outputs = manifest.get("outputs", {})
    for key in ("monthly", "scalars"):
        out.append((f"{key} outputs", str(outputs.get(key)), str(outputs.get(f"{key}_sha256"))))
    out.append(
        (
            "invariant results",
            str(manifest.get("invariants")),
            str(manifest.get("invariants_sha256")),
        )
    )
    return out


STAGE_COPIES = (
    ("groups", "groups"),
    ("derived_groups", "derived group fields"),
    ("membership", "membership"),
)
"""The copies an aggregation run made by run_aggregation keeps beside every run's, by key of
the manifest's ``inputs`` block and as messages name them."""


@dataclass(frozen=True, slots=True)
class VerifiedRun:
    """A run directory read once, every file its manifest records a hash for verified
    (:func:`verified_run`): the manifest, the sha256 of its bytes and each verified file's
    bytes. What a later run takes from the directory, its outputs (:meth:`result`) and its
    copies (:meth:`read`), it takes from these bytes, so the bytes checked are the bytes used
    and the manifest hash recorded is that of the manifest checked (D21)."""

    path: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    who: str
    """The run as messages name it, ``"<what> <run_id>"``."""
    files: dict[str, tuple[str, bytes]]
    """By relative path, the sha256 the manifest records and the bytes that hash to it."""

    def read(self, entry: Mapping[str, Any], label: str) -> ReadFile:
        """A copy as its manifest ``entry`` records it (``copy``, ``sha256``), parsed from the
        verified bytes; an entry beside the files :func:`run_files` lists is verified now."""
        rel, sha = copy_entry(entry, label)
        known = self.files.get(rel)
        data = known[1] if known is not None and known[0] == sha else None
        if data is None:
            data = verify_copy(self.path, rel, sha, label)
        return parse_copy(self.path / rel, sha, data, label)

    def result(self) -> RunResult:
        """The run's outputs and invariant results read back from the verified bytes."""
        return _result(self.path, self.manifest, self.manifest_sha256, self._bytes)

    def _bytes(self, rel: str) -> bytes | None:
        known = self.files.get(rel)
        return None if known is None else known[1]


def verified_run(run_dir: str | Path, what: str = "run") -> VerifiedRun:
    """Read a run directory once and check every file whose sha256 its manifest records
    (:func:`run_files`), D21. A LoadError names the first file that is missing or whose bytes
    hash otherwise, or a manifest written before the outputs' hashes were recorded. Messages
    name the run as ``"<what> <run_id>"`` (``"basis cur run 1f..."``)."""
    run_path = Path(run_dir)
    raw = _manifest_bytes(run_path)
    manifest = _parse_manifest(run_path, raw)
    who = f"{what} {manifest.get('run_id')}"
    files: dict[str, tuple[str, bytes]] = {}
    for name, rel, sha in run_files(manifest):
        if rel == "None" or len(sha) != 64:
            raise LoadError(
                f"{who}: its manifest records no sha256 for its {name} ({run_path}); it predates "
                "the hashes a later run verifies, so rerun it"
            )
        files[rel] = (sha, verify_copy(run_path, rel, sha, f"{who}: {name}"))
    return VerifiedRun(run_path, manifest, hashlib.sha256(raw).hexdigest(), who, files)


def verify_run(run_dir: str | Path, what: str = "run") -> dict[str, Any]:
    """Check every file of a run directory whose sha256 its manifest records and return the
    manifest (:func:`verified_run`, which also keeps the bytes it checked)."""
    return verified_run(run_dir, what).manifest


def load_run(run_dir: str | Path, *, verify: bool = False) -> RunResult:
    """Read a run directory back: manifest, outputs and invariants. The model is not needed.
    With ``verify``, every file the manifest records a hash for is checked first and the run is
    read from the bytes checked (:func:`verified_run`)."""
    run_path = Path(run_dir)
    if verify:
        return verified_run(run_path).result()
    manifest, sha = read_manifest(run_path)

    def from_disk(rel: str) -> bytes | None:
        path = run_path / rel
        return path.read_bytes() if path.is_file() else None

    return _result(run_path, manifest, sha, from_disk)


def _result(
    run_path: Path,
    manifest: dict[str, Any],
    manifest_sha256: str,
    read: Callable[[str], bytes | None],
) -> RunResult:
    """A :class:`RunResult` from a manifest and ``read(relative path)``, which returns a file's
    bytes or ``None`` for a missing file: an output file is required, the invariant results
    (which a run made before they were written lacks) are not."""

    def required(rel: str) -> bytes:
        raw = read(rel)
        if raw is None:
            raise LoadError(f"{run_path / rel} is missing, though the run's manifest names it")
        return raw

    dims = manifest["dimensions"]
    n_pol, n_scen = int(dims["policies"]), int(dims["scenarios"])
    horizon = int(dims["horizon"])
    n_months = horizon + 1
    outputs = manifest["outputs"]
    id_field = outputs.get("id_field", "policy_id")
    monthly = pq.read_table(io.BytesIO(required(str(outputs["monthly"]))))
    scalars = pq.read_table(io.BytesIO(required(str(outputs["scalars"]))))
    ids = np.asarray(scalars.column(id_field).to_numpy(zero_copy_only=False))[::n_scen]
    arrays: dict[str, NDArray[Any]] = {}
    has_time: dict[str, bool] = {}
    for name, info in outputs["variables"].items():
        if info["time"]:
            col = np.asarray(monthly.column(name).to_numpy(zero_copy_only=False))
            arrays[name] = np.ascontiguousarray(
                col.reshape(n_pol, n_months, n_scen).transpose(1, 0, 2)
            )
            has_time[name] = True
        else:
            col = np.asarray(scalars.column(name).to_numpy(zero_copy_only=False))
            arrays[name] = col.reshape(1, n_pol, n_scen)
            has_time[name] = False
    raw = read(str(manifest.get("invariants", "invariants.json")))
    invariants: list[InvariantResult] = []
    if raw is not None:
        invariants = [InvariantResult(**r) for r in json.loads(raw.decode("utf-8"))]
    return RunResult(
        run_path,
        str(manifest["run_id"]),
        manifest,
        ids,
        horizon,
        n_scen,
        arrays,
        has_time,
        None,
        None,
        None,
        invariants,
        manifest_sha256,
    )
