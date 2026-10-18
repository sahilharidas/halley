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
the hash of its values, and its outputs are marked ``outputs.kind = "aggregation"``.
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
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray

from halley import ir
from halley.codegen import CompiledModel
from halley.codegen.numpy_backend import run_batch
from halley.errors import LoadError
from halley.runtime.invariants import (
    BatchView,
    InvariantResult,
    check_builtin,
    check_declared,
    merge_results,
)
from halley.runtime.loader import Inputs, LoadedSeries, ReadFile, Source, load_inputs

OUTPUT_SCHEMA_VERSION = 1
"""Bumped when the layout of the output files changes. A later aggregation stage reads a
previous run's outputs by this version and the hashes in its manifest."""


def _halley_version() -> str:
    try:
        return importlib.metadata.version("halley")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - not installed
        return "0"


@dataclass(frozen=True, slots=True)
class RunSettings:
    valuation_date: dt.date
    horizon: int | None = None
    """Months to project; default the largest value of the schema's horizon field."""
    batch_size: int = 10_000
    outputs: tuple[str, ...] | None = None
    """Variables to store; default the model's ``outputs``, or every variable."""
    label: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "valuation_date": self.valuation_date.isoformat(),
            "horizon": self.horizon,
            "batch_size": self.batch_size,
            "outputs": list(self.outputs) if self.outputs is not None else None,
            "label": self.label,
        }


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

    @property
    def n_policies(self) -> int:
        return len(self.ids)

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
    dest.parent.mkdir(parents=True, exist_ok=True)
    if f.path is not None:
        shutil.copyfile(f.path, dest)
    elif f.raw is not None:
        dest.write_bytes(f.raw)


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
        # (which depend on the parquet writer) nor its source runs' directories
        out["series"] = {
            name: {k: v for k, v in e.items() if k not in ("source_runs", "sha256")}
            for name, e in inputs["series"].items()
        }
    return out


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
) -> RunResult:
    """Load and validate the inputs, project every policy in batches, check the built-in and
    the product's invariants on every batch, and write the run directory.

    An aggregation stage runs the same way with its groups as the model points and its group
    series in ``series`` (:func:`~halley.aggregation.group_series`); its run directory keeps a
    copy of every series and its outputs are keyed on the groups' id field, with
    ``outputs.kind`` ``"aggregation"``."""
    graph = compiled.graph
    model = compiled.model
    run_path = Path(run_dir)
    if (run_path / "manifest.json").exists():
        raise LoadError(f"{run_path} already holds a run; use a new directory for each run")
    stage = model.series_declarations() is not None
    if series and not stage:
        raise LoadError(f"{graph.model} is a projection; group series are read by a stage")
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
    )
    horizon = inputs.H
    n_pol = len(inputs.ids)
    n_scen = inputs.n_scenarios
    scales = inputs.scales(graph)
    needed = sorted(compiled.schedule.needed)
    parts: dict[str, list[NDArray[Any]]] = {name: [] for name in needed}
    batches: list[list[InvariantResult]] = []
    bs = max(1, settings.batch_size)
    for lo in range(0, n_pol, bs):
        hi = min(lo + bs, n_pol)
        inp = inputs.batch(graph, lo, hi)
        out = run_batch(compiled.fn, graph, inp, horizon, "groups" if stage else "policies")
        batch_scales = {u: s[lo:hi] for u, s in scales.items()}
        results = check_builtin(graph, out, inputs.ids[lo:hi], horizon, batch_scales)
        view = BatchView(graph, out, inputs, lo, hi, batch_scales)
        results.extend(check_declared(model.invariants, view))
        batches.append(results)
        for name in needed:
            parts[name].append(out[name])
    invariants = merge_results(batches)
    arrays: dict[str, NDArray[Any]] = {}
    for name in needed:
        v = graph.variables[name]
        dims = v.dims or frozenset()
        if ir.POLICY in dims:
            arrays[name] = np.concatenate(parts[name], axis=1)
        else:
            arrays[name] = parts[name][0]
    has_time = {name: graph.variables[name].has_time for name in needed}

    stored = list(settings.outputs or graph.outputs or needed)
    for name in stored:
        if name not in arrays:
            raise LoadError(f"output {name!r} is not a variable of the model")

    # ------------------------------------------------------------------ run directory
    run_path.mkdir(parents=True, exist_ok=True)
    (run_path / "generated").mkdir(exist_ok=True)
    (run_path / "generated" / "numpy_backend.py").write_text(compiled.source, encoding="utf-8")
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
            "numpy": np.__version__,
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
    identity = {
        "ir_sha256": manifest["ir_sha256"],
        "inputs": _without_paths(manifest["inputs"]),
        "settings": {k: v for k, v in settings.as_dict().items() if k != "label"},
        "backend": {"name": compiled.backend, "numpy": np.__version__},
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
    (run_path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
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


def load_run(run_dir: str | Path) -> RunResult:
    """Read a run directory back: manifest, outputs and invariants. The model is not needed."""
    run_path = Path(run_dir)
    manifest_path = run_path / "manifest.json"
    if not manifest_path.exists():
        raise LoadError(f"{run_path} holds no manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("output_schema") != OUTPUT_SCHEMA_VERSION:
        raise LoadError(
            f"{run_path} was written with output schema {manifest.get('output_schema')}; this "
            f"version reads {OUTPUT_SCHEMA_VERSION}"
        )
    dims = manifest["dimensions"]
    n_pol, n_scen = int(dims["policies"]), int(dims["scenarios"])
    horizon = int(dims["horizon"])
    n_months = horizon + 1
    outputs = manifest["outputs"]
    id_field = outputs.get("id_field", "policy_id")
    monthly = pq.read_table(run_path / outputs["monthly"])
    scalars = pq.read_table(run_path / outputs["scalars"])
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
    inv_path = run_path / manifest.get("invariants", "invariants.json")
    invariants: list[InvariantResult] = []
    if inv_path.exists():
        invariants = [
            InvariantResult(**r) for r in json.loads(inv_path.read_text(encoding="utf-8"))
        ]
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
    )
