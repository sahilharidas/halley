# pyright: basic
"""Running a compiled model: batches, outputs, the run directory and its manifest (D21).

A run directory holds everything needed to reproduce the run and to read it back later without
the model: ``manifest.json``, a copy of every input under ``inputs/``, the generated backend
source under ``generated/``, the outputs under ``outputs/`` keyed on the schema's id field with a
versioned schema, and the invariant results. The manifest records the model source hash and git
commit, the IR hash, every input's hash and ``halley.*`` metadata, the backend and its versions,
the settings, and per stored variable its unit, domain start and role. The run id is a hash of
the IR, the inputs' hashes (not their paths, so the same bytes anywhere give the same id), the
settings and the backend.
"""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
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
from halley.runtime.loader import Inputs, ReadFile, Source, load_inputs

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
    """The manifest's inputs block without the original ``path`` strings: a run is addressed by
    what it read, not where the files were."""
    block = copy.deepcopy(inputs)

    def strip(node: Any) -> None:
        if isinstance(node, dict):
            node.pop("path", None)
            for v in node.values():
                strip(v)

    strip(block)
    return block


def run(
    compiled: CompiledModel,
    *,
    model_points: Source,
    tables: dict[str, Source] | None = None,
    scenarios: dict[str, Source] | None = None,
    settings: RunSettings,
    run_dir: str | Path,
) -> RunResult:
    """Load and validate the inputs, project every policy in batches, check the built-in and
    the product's invariants on every batch, and write the run directory."""
    graph = compiled.graph
    model = compiled.model
    run_path = Path(run_dir)
    if (run_path / "manifest.json").exists():
        raise LoadError(f"{run_path} already holds a run; use a new directory for each run")
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
        out = run_batch(compiled.fn, graph, inp, horizon)
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

    source_path = Path(graph.source_file)
    id_field = compiled.schema.id_field
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
        },
        "outputs": {
            "kind": "projection",
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
