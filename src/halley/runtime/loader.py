# pyright: basic
"""Loader: read and validate model points, assumption tables, scenario files and the group
series of an aggregation stage.

Everything is checked against the contracts the model declares before a batch runs: field types
and ranges, cross-field rules, derived fields, table key coverage, value ranges, metadata and
rules, scenario coverage of every month the model reads, value ranges and rules, each group
series' unit, domain, scenarios and groups, and the product's input rules once everything is
loaded. A failure is a
:class:`~halley.errors.LoadError` that names the file, the row or policy and the rule. Nothing
is adjusted silently.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray

from halley import ir
from halley.authoring.contracts import (
    InputRule,
    InputView,
    LoadContext,
    ScenarioInput,
    ScenarioView,
    Schema,
    Table,
    TableView,
    resolve_month,
)
from halley.errors import LoadError
from halley.ir import units

Source = str | Path | Any
"""A Parquet file path, an in-memory ``pyarrow.Table`` or a :class:`ReadFile` already read (an
aggregation stage hands the run the files it checked, so that the run uses those bytes)."""

METADATA_PREFIX = "halley."


def lane_noun(lanes: str) -> str:
    """One of a run's lanes, for a message that names one: ``"policy"`` for ``"policies"``,
    ``"group"`` for ``"groups"`` (an aggregation stage's model points are groups)."""
    return "policy" if lanes == "policies" else lanes.removesuffix("s")


@dataclass(slots=True)
class ReadFile:
    """An input as read: the Arrow table, its bytes' hash, its ``halley.*`` schema metadata (and
    a bare ``version``) as strings without the prefix and, when given as a file, its path."""

    table: pa.Table
    sha256: str
    path: Path | None
    raw: bytes | None
    """The Parquet bytes the table was read from and ``sha256`` hashes (for an in-memory table,
    the bytes it was written to), which the run directory's copy holds: a file replaced on disk
    after it was read does not reach the copy."""
    label: str
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def where(self) -> str:
        """The label with the file's path, for messages."""
        return f"{self.label} ({self.path})" if self.path is not None else self.label


def _schema_metadata(table: pa.Table) -> dict[str, str]:
    meta = table.schema.metadata or {}
    out: dict[str, str] = {}
    for k, v in meta.items():
        if k.decode(errors="replace") == "version":
            out["version"] = v.decode(errors="replace")
    for k, v in meta.items():
        key = k.decode(errors="replace")
        if key.startswith(METADATA_PREFIX):
            out[key[len(METADATA_PREFIX) :]] = v.decode(errors="replace")
    return out


def read_source(source: Source, label: str) -> ReadFile:
    """An input as read: a file is read once, and its table parsed from the very bytes its
    sha256 hashes and the run copies; a :class:`ReadFile` read before is taken as it is, under
    ``label``."""
    if isinstance(source, ReadFile):
        return ReadFile(
            source.table, source.sha256, source.path, source.raw, label, dict(source.metadata)
        )
    if isinstance(source, pa.Table):
        buf = io.BytesIO()
        pq.write_table(source, buf)
        raw = buf.getvalue()
        return ReadFile(
            source, hashlib.sha256(raw).hexdigest(), None, raw, label, _schema_metadata(source)
        )
    path = Path(source)
    if not path.exists():
        raise LoadError(f"{label}: no such file {path}")
    data = path.read_bytes()
    try:
        table = pq.read_table(io.BytesIO(data))
    except (pa.ArrowException, OSError) as exc:
        raise LoadError(f"{label}: {path} is not a Parquet file: {exc}") from exc
    return ReadFile(
        table, hashlib.sha256(data).hexdigest(), path, data, label, _schema_metadata(table)
    )


# ------------------------------------------------------------------------ a run's own copies


def verify_copy(run_dir: Path, rel: str, sha256: str, label: str) -> bytes:
    """The bytes of ``run_dir / rel``, a file a run's manifest records with ``sha256`` (D21): a
    LoadError naming the file when it is missing or when its bytes hash to anything else."""
    path = run_dir / rel
    try:
        data = path.read_bytes()
    except OSError:
        raise LoadError(
            f"{label}: {path} is missing, though its run's manifest records it"
        ) from None
    got = hashlib.sha256(data).hexdigest()
    if got != sha256:
        raise LoadError(
            f"{label}: {path} does not match the sha256 its run's manifest records ({got[:12]}... "
            f"against {sha256[:12]}...); the file was changed after the run"
        )
    return data


def copy_entry(entry: Mapping[str, Any], label: str) -> tuple[str, str]:
    """The relative path and sha256 a run's manifest entry records for one of its copies."""
    rel, sha = entry.get("copy"), entry.get("sha256")
    if not isinstance(rel, str) or not isinstance(sha, str):
        raise LoadError(f"{label}: its run's manifest records no copy with a sha256")
    return rel, sha


def parse_copy(path: Path, sha256: str, data: bytes, label: str) -> ReadFile:
    """A run's copy parsed from ``data``, the bytes already verified against ``sha256``: the
    :class:`ReadFile` holds those bytes, so what is checked, used and copied again is what was
    hashed."""
    try:
        table = pq.read_table(io.BytesIO(data))
    except (pa.ArrowException, OSError) as exc:
        raise LoadError(f"{label}: {path} is not a Parquet file: {exc}") from exc
    return ReadFile(table, sha256, path, data, label, _schema_metadata(table))


def read_copy(run_dir: Path, entry: Mapping[str, Any], label: str) -> ReadFile:
    """A run's copy of one of its inputs, as its manifest's entry records it (``copy`` and
    ``sha256``), verified (:func:`verify_copy`) and read. The :class:`ReadFile`'s path is the
    copy's, so a later run that reads it copies those bytes."""
    rel, sha = copy_entry(entry, label)
    data = verify_copy(run_dir, rel, sha, label)
    return parse_copy(run_dir / rel, sha, data, label)


def table_columns(f: ReadFile) -> dict[str, NDArray[Any]]:
    """Every column of a file as a NumPy array, for the aggregation loader (design section 9):
    strings, dictionary-encoded or not, as ``str`` objects; integers as int64; floats as
    float64; booleans as bool; dates and timestamps as ``datetime.date`` objects. A column with
    nulls, or of another type, is a LoadError naming the file and the column."""
    out: dict[str, NDArray[Any]] = {}
    for name in f.table.column_names:
        col = f.table.column(name)
        if col.null_count:
            raise LoadError(f"{f.where}: column {name} has {col.null_count} null value(s)")
        t = col.type
        if pa.types.is_dictionary(t):
            col = col.cast(pa.string())
            t = col.type
        if pa.types.is_string(t) or pa.types.is_large_string(t):
            out[name] = np.asarray([str(v) for v in col.to_pylist()], dtype=object)
        elif pa.types.is_integer(t):
            out[name] = np.asarray(col.to_numpy(), dtype=np.int64)
        elif pa.types.is_floating(t):
            out[name] = np.asarray(col.to_numpy(), dtype=np.float64)
        elif pa.types.is_boolean(t):
            out[name] = np.asarray(col.to_pylist(), dtype=np.bool_)
        elif pa.types.is_date(t) or pa.types.is_timestamp(t):
            out[name] = np.asarray(col.cast(pa.date32()).to_pylist(), dtype=object)
        else:
            raise LoadError(f"{f.where}: column {name} is {t}, which the loader does not read")
    return out


def with_columns(f: ReadFile | None, columns: Mapping[str, NDArray[Any]]) -> Any:
    """A ``pyarrow.Table``: the file's table with ``columns`` appended (strings as string, ints
    as int64, floats as float64), or, for a file that is ``None``, ``columns`` alone. The stage
    loader's model points are the groups file with the fields it derived; its
    ``derived_groups`` copy is those fields alone."""
    arrays = {name: _arrow_array(values) for name, values in columns.items()}
    if f is None:
        return pa.table(arrays)
    table = f.table
    for name, arr in arrays.items():
        table = table.append_column(name, arr)
    return table


def _arrow_array(values: NDArray[Any]) -> Any:
    if values.dtype == object:
        return pa.array([str(v) for v in values], pa.string())
    if np.issubdtype(values.dtype, np.integer):
        return pa.array(values.astype(np.int64), pa.int64())
    return pa.array(values.astype(np.float64), pa.float64())


def table_difference(a: Any, b: Any) -> str | None:
    """``None`` when two ``pyarrow.Table`` objects are equal bit for bit (column names and
    types, every value, floats compared as their 64-bit patterns so ``-0.0`` and ``0.0``
    differ, and the ``halley.*`` and ``version`` schema metadata); otherwise the first
    difference, for a message."""
    if a.column_names != b.column_names:
        return f"columns {a.column_names} against {b.column_names}"
    if a.num_rows != b.num_rows:
        return f"{a.num_rows} rows against {b.num_rows}"
    for name in a.column_names:
        x, y = a.column(name), b.column(name)
        if x.type != y.type:
            return f"column {name} is {x.type} against {y.type}"
        if pa.types.is_floating(x.type):
            xs = np.asarray(x.to_numpy(), dtype=np.float64).view(np.uint64)
            ys = np.asarray(y.to_numpy(), dtype=np.float64).view(np.uint64)
            bad = np.flatnonzero(xs != ys)
            if bad.size:
                i = int(bad[0])
                return f"column {name}, row {i}: {x[i].as_py()!r} against {y[i].as_py()!r}"
        elif not x.equals(y):
            return f"column {name} differs"
    if _schema_metadata(a) != _schema_metadata(b):
        return f"metadata {_schema_metadata(a)} against {_schema_metadata(b)}"
    return None


def file_difference(f: ReadFile, table: Any) -> str | None:
    """:func:`table_difference` between a file as read and a ``pyarrow.Table``: ``None`` when
    the table holds the file's values bit for bit."""
    if not isinstance(table, pa.Table):
        return f"a {type(table).__name__}, not a pyarrow.Table"
    return table_difference(f.table, table)


def table_of(f: ReadFile) -> Any:
    """The ``pyarrow.Table`` of a file as read."""
    return f.table


def column_type(f: ReadFile, name: str) -> str:
    """The Arrow type of a file's column, for a message (``string``, ``double``)."""
    return str(f.table.schema.field(name).type)


def _ids(values: NDArray[Any], mask: NDArray[np.bool_], limit: int = 5) -> str:
    bad = [str(v) for v in values[mask]]
    shown = ", ".join(bad[:limit])
    more = f" and {len(bad) - limit} more" if len(bad) > limit else ""
    return f"{shown}{more}"


def _run_check(check: Callable[[Any], Any], view: Any, what: str) -> NDArray[np.bool_]:
    """A rule's result as a boolean array; the rule's own exception is a LoadError."""
    try:
        result = check(view)
    except Exception as exc:  # the rule's own failure
        raise LoadError(f"{what} raised {type(exc).__name__}: {exc}") from exc
    arr = np.asarray(result)
    if arr.dtype != np.bool_:
        raise LoadError(f"{what} must return booleans, not {arr.dtype}")
    return arr


def _doc(doc: str) -> str:
    return f" ({doc})" if doc else ""


# ------------------------------------------------------------------------------ model points


@dataclass(slots=True)
class ModelPoints:
    schema: Schema
    ids: NDArray[Any]
    columns: dict[str, NDArray[Any]]
    """Every field and derived field, sorted by id: numbers float64, ints int64, enums as their
    strings, dates datetime64[D]."""
    codes: dict[str, NDArray[np.int64]]
    """Enum fields as integer codes in the order the schema lists the values."""
    file: ReadFile

    def __len__(self) -> int:
        return len(self.ids)


def _as_numpy(col: pa.ChunkedArray, decl: ir.FieldDecl, label: str) -> NDArray[Any]:
    if col.null_count:
        raise LoadError(f"{label}: field {decl.name} has {col.null_count} null value(s)")
    t = col.type
    if decl.kind == "number":
        if not (pa.types.is_floating(t) or pa.types.is_integer(t)):
            raise LoadError(f"{label}: field {decl.name} must be float64, not {t}")
        return np.asarray(col.to_numpy(), dtype=np.float64)
    if decl.kind == "int":
        if not pa.types.is_integer(t):
            raise LoadError(f"{label}: field {decl.name} must be an integer type, not {t}")
        return np.asarray(col.to_numpy(), dtype=np.int64)
    if decl.kind == "enum":
        if pa.types.is_dictionary(t):
            col = col.cast(pa.string())
        elif not (pa.types.is_string(t) or pa.types.is_large_string(t)):
            raise LoadError(f"{label}: enum field {decl.name} must be a string type, not {t}")
        return np.asarray(col.to_pylist(), dtype=object)
    if decl.kind == "date":
        if pa.types.is_date(t) or pa.types.is_timestamp(t):
            return np.asarray(col.cast(pa.date32()).to_numpy(), dtype="datetime64[D]")
        raise LoadError(f"{label}: date field {decl.name} must be date32, not {t}")
    if pa.types.is_dictionary(t):  # a dictionary-encoded string column reads as its strings
        col = col.cast(pa.string())
    elif not (pa.types.is_string(t) or pa.types.is_large_string(t)):
        raise LoadError(f"{label}: field {decl.name} must be a string type, not {t}")
    return np.asarray(col.to_pylist(), dtype=object)


def _check_range(
    values: NDArray[Any], decl: ir.FieldDecl, ids: NDArray[Any], label: str, lanes: str
) -> None:
    if decl.kind not in ("number", "int"):
        return
    checks = (
        ("gt", decl.gt, values > decl.gt if decl.gt is not None else None),
        ("ge", decl.ge, values >= decl.ge if decl.ge is not None else None),
        ("lt", decl.lt, values < decl.lt if decl.lt is not None else None),
        ("le", decl.le, values <= decl.le if decl.le is not None else None),
    )
    for op, bound, ok in checks:
        if ok is None:
            continue
        bad = ~np.asarray(ok)
        if bad.any():
            symbol = {"gt": ">", "ge": ">=", "lt": "<", "le": "<="}[op]
            raise LoadError(
                f"{label}: field {decl.name} must be {symbol} {bound}; {lanes} "
                f"{_ids(ids, bad)} break this"
            )
    if decl.kind == "number" and not np.isfinite(values).all():
        raise LoadError(f"{label}: field {decl.name} has non-finite values")


def load_model_points(
    graph: ir.Graph,
    schema: Schema,
    source: Source,
    valuation_date: dt.date,
    lanes: str = "policies",
    label: str = "model points",
) -> ModelPoints:
    """Read and check the model points; a failure names the rows by their ids under ``lanes``
    (``"groups"`` for an aggregation stage) and the file by ``label`` (an aggregation run names
    its groups file, from which the loader built the table)."""
    f = read_source(source, label)
    label = f.where
    if schema.valuation_date_rule is not None:
        message = schema.valuation_date_rule(valuation_date)
        if message:
            raise LoadError(f"valuation date {valuation_date}: {message}")
    table = f.table
    columns: dict[str, NDArray[Any]] = {}
    for fld in schema.fields:
        if fld.name not in table.column_names:
            raise LoadError(f"{label}: field {fld.name} is missing")
        columns[fld.name] = _as_numpy(table.column(fld.name), graph.fields[fld.name], label)
    ids = columns[schema.id_field]
    if len(ids) == 0:
        raise LoadError(f"{label}: no rows")
    for fld in schema.fields:
        decl = graph.fields[fld.name]
        values = columns[fld.name]
        if decl.enum is not None:
            allowed = set(decl.enum)
            bad = np.array([v not in allowed for v in values], dtype=bool)
            if bad.any():
                raise LoadError(
                    f"{label}: field {fld.name} takes exactly {', '.join(decl.enum)}; {lanes} "
                    f"{_ids(ids, bad)} have other values"
                )
        if decl.unique:
            _, counts = np.unique(values.astype(str), return_counts=True)
            if (counts > 1).any():
                uniq = np.unique(values.astype(str))
                dup = {v for v, c in zip(uniq, counts, strict=True) if c > 1}
                raise LoadError(
                    f"{label}: field {fld.name} is not unique: {', '.join(sorted(dup))}"
                )
        _check_range(values, decl, ids, label, lanes)
    ctx = LoadContext(columns, valuation_date)
    for d in schema.derived:
        try:
            values = np.asarray(d.fn(ctx))
        except Exception as exc:  # the derived field's own failure
            raise LoadError(f"{label}: derived field {d.name} failed: {exc}") from exc
        if values.shape != ids.shape:
            raise LoadError(f"{label}: derived field {d.name} must give one value per row")
        decl = graph.fields[d.name]
        values = values.astype(np.int64) if decl.kind == "int" else values.astype(np.float64)
        _check_range(values, decl, ids, label, lanes)
        columns[d.name] = values
    for rule in schema.rules:
        ok = np.asarray(rule.check(ctx), dtype=bool)
        if ok.shape != ids.shape:
            raise LoadError(f"{label}: rule {rule.name} must give one boolean per row")
        if not ok.all():
            raise LoadError(
                f"{label}: rule {rule.name} fails for {lanes} {_ids(ids, ~ok)}{_doc(rule.doc)}"
            )
    order = np.argsort(ids, kind="stable")
    columns = {k: v[order] for k, v in columns.items()}
    ids = columns[schema.id_field]
    codes: dict[str, NDArray[np.int64]] = {}
    for name, decl in graph.fields.items():
        if decl.enum is not None:
            lookup = {v: i for i, v in enumerate(decl.enum)}
            codes[name] = np.array([lookup[v] for v in columns[name]], dtype=np.int64)
    return ModelPoints(schema, ids, columns, codes, f)


# ----------------------------------------------------------------------------------- tables


@dataclass(slots=True)
class LoadedTable:
    decl: ir.TableDecl
    values: dict[str, NDArray[Any]]
    """One dense array per value column, one axis per key in declared order; keyless tables
    give 0-d arrays."""
    version: str | None
    file: ReadFile

    def view(self) -> TableView:
        return TableView(self.decl.name, self.decl.keys, self.values, self.file.metadata)


def _first_failing_key(view: TableView, ok: NDArray[np.bool_]) -> str:
    """``at sex = F, age = 120`` for the first False of a rule's result, when the result lines
    up with the table's dense shape (an axis of length 1 names no key); otherwise its position."""
    shape = view.shape
    if ok.ndim == 0:
        return ""
    if ok.ndim == len(shape) and all(
        n in (1, full) for n, full in zip(ok.shape, shape, strict=True)
    ):
        idx = np.argwhere(~ok)[0]
        parts = [
            f"{name} = {view.key_values(name)[int(i)]}"
            for name, i, n in zip(view.keys, idx, ok.shape, strict=True)
            if n > 1
        ]
        return f" at {', '.join(parts)}" if parts else ""
    return f" at element {int(np.flatnonzero(~ok.reshape(-1))[0])} of the rule's result"


def load_table(decl: ir.TableDecl, source: Source, contract: Table | None = None) -> LoadedTable:
    """Read and validate one assumption table against its IR declaration and, when given, the
    authoring contract's required metadata and rules."""
    f = read_source(source, f"table {decl.name}")
    label = f.where
    table = f.table
    n = table.num_rows
    sizes: list[int] = []
    flat = np.zeros(n, dtype=np.int64)
    for key in decl.keys:
        if key.name not in table.column_names:
            raise LoadError(f"{label}: key column {key.name} is missing")
        col = table.column(key.name)
        if col.null_count:
            raise LoadError(f"{label}: key {key.name} has null values")
        if key.enum is not None:
            if pa.types.is_dictionary(col.type):
                col = col.cast(pa.string())
            raw = col.to_pylist()
            lookup = {v: i for i, v in enumerate(key.enum)}
            unknown = sorted({str(v) for v in raw if v not in lookup})
            if unknown:
                raise LoadError(
                    f"{label}: key {key.name} has values outside its enum "
                    f"({', '.join(key.enum)}): {', '.join(unknown)}"
                )
            idx = np.array([lookup[v] for v in raw], dtype=np.int64)
            size = len(key.enum)
        else:
            if not pa.types.is_integer(col.type):
                raise LoadError(f"{label}: key {key.name} must be an integer column")
            vals = np.asarray(col.to_numpy(), dtype=np.int64)
            lo, hi = int(key.lo or 0), int(key.hi or 0)
            out = (vals < lo) | (vals > hi)
            if out.any():
                raise LoadError(
                    f"{label}: key {key.name} has rows outside its declared range [{lo}, {hi}]: "
                    f"{', '.join(str(v) for v in np.unique(vals[out])[:5])}"
                )
            idx = vals - lo
            size = hi - lo + 1
        flat = flat * size + idx
        sizes.append(size)
    expected = int(np.prod(sizes)) if sizes else 1
    if n != expected or len(np.unique(flat)) != n:
        missing = expected - len(np.unique(flat))
        raise LoadError(
            f"{label}: the rows must cover every key combination exactly once; "
            f"declared grid has {expected} rows, the file {n} rows"
            + (f", {missing} combination(s) missing" if missing > 0 else ", with duplicates")
        )
    values: dict[str, NDArray[Any]] = {}
    for colspec in decl.columns:
        if colspec.name not in table.column_names:
            raise LoadError(f"{label}: value column {colspec.name} is missing")
        col = table.column(colspec.name)
        if col.null_count:
            raise LoadError(f"{label}: column {colspec.name} has null values")
        if not (pa.types.is_floating(col.type) or pa.types.is_integer(col.type)):
            raise LoadError(f"{label}: column {colspec.name} must be numeric")
        dtype = np.int64 if colspec.unit == units.INT else np.float64
        vals = np.asarray(col.to_numpy(), dtype=dtype)
        if dtype is np.float64 and not np.isfinite(vals).all():
            raise LoadError(f"{label}: column {colspec.name} has non-finite values")
        ge, le = colspec.ge, colspec.le
        if colspec.unit in units.PROBS:
            ge = 0.0 if ge is None else max(ge, 0.0)
            le = 1.0 if le is None else min(le, 1.0)
        if ge is not None and (vals < ge).any():
            raise LoadError(f"{label}: column {colspec.name} has values below {ge}")
        if le is not None and (vals > le).any():
            raise LoadError(f"{label}: column {colspec.name} has values above {le}")
        dense = np.zeros(sizes, dtype=dtype) if sizes else np.zeros((), dtype=dtype)
        dense.reshape(-1)[flat] = vals
        values[colspec.name] = dense
    loaded = LoadedTable(decl, values, f.metadata.get("version"), f)
    if contract is not None:
        for key in contract.metadata:
            if key not in f.metadata:
                raise LoadError(
                    f"{label}: the file carries no schema metadata {METADATA_PREFIX}{key}"
                )
        view = loaded.view()
        for rule in contract.rules:
            ok = _run_check(rule.check, view, f"{label}: rule {rule.name}")
            if not ok.all():
                raise LoadError(
                    f"{label}: rule {rule.name} fails{_first_failing_key(view, ok)}{_doc(rule.doc)}"
                )
    return loaded


# -------------------------------------------------------------------------------- scenarios


@dataclass(slots=True)
class LoadedScenario:
    decl: ir.ScenarioDecl
    values: NDArray[np.float64]
    """Shape ``(months, 1, scenarios)``; row ``i`` is month ``u0 + i``."""
    u0: int
    file: ReadFile

    @property
    def n_scenarios(self) -> int:
        return int(self.values.shape[2])

    @property
    def u_max(self) -> int:
        return self.u0 + int(self.values.shape[0]) - 1

    def view(self) -> ScenarioView:
        return ScenarioView(self.decl.name, self.values[:, 0, :], self.u0)


def scenario_requirements(graph: ir.Graph, horizon: int) -> dict[str, tuple[int, int]]:
    """For each scenario input, the first and last month the model reads at ``horizon``."""
    req: dict[str, tuple[int, int]] = {}

    def need(name: str, lo: int, hi: int) -> None:
        cur = req.get(name)
        req[name] = (lo, hi) if cur is None else (min(cur[0], lo), max(cur[1], hi))

    for v in graph.variables.values():
        first, last = v.formula_range
        last_month = horizon if last == "H" else horizon - 1
        for expr in (v.expr, v.init, v.terminal):
            if expr is None:
                continue
            for node in ir.walk(expr):
                if isinstance(node, ir.ScenRead):
                    if v.has_time:
                        need(node.name, first + node.offset, last_month + node.offset)
                elif isinstance(node, ir.ScenAt):
                    if isinstance(node.index, ir.Const) and not isinstance(node.index.value, bool):
                        k = int(node.index.value)
                        need(node.name, k, k)
                    else:
                        need(node.name, 0, horizon)
    for name in graph.scenarios:
        req.setdefault(name, (0, horizon))
    return req


def _first_failing_row(view: ScenarioView, ok: NDArray[np.bool_]) -> str:
    """``for scenario 2, month -3`` for the first False of a rule's result, when its shape is the
    view's ``(months, scenarios)``, ``(scenarios,)`` or ``(months,)``."""
    n_months, n_scen = view.values.shape
    if ok.shape == (n_months, n_scen):
        i, s = (int(x) for x in np.argwhere(~ok)[0])
        return f" for scenario {s}, month {view.u0 + i}"
    if ok.shape == (n_scen,):
        return f" for scenario {int(np.flatnonzero(~ok)[0])}"
    if ok.shape == (n_months,):
        return f" for month {view.u0 + int(np.flatnonzero(~ok)[0])}"
    return ""


def _check_scenario_range(
    label: str, contract: ScenarioInput, view: ScenarioView, dense: NDArray[np.float64]
) -> None:
    checks = (
        (">", contract.gt, dense > contract.gt if contract.gt is not None else None),
        (">=", contract.ge, dense >= contract.ge if contract.ge is not None else None),
        ("<", contract.lt, dense < contract.lt if contract.lt is not None else None),
        ("<=", contract.le, dense <= contract.le if contract.le is not None else None),
    )
    for symbol, bound, ok in checks:
        if ok is None:
            continue
        ok = np.asarray(ok)
        if not ok.all():
            i, s = (int(x) for x in np.argwhere(~ok)[0])
            raise LoadError(
                f"{label}: values must be {symbol} {bound}; scenario {s}, month {view.u0 + i} "
                f"has {dense[i, s]}"
            )


def load_scenario(
    decl: ir.ScenarioDecl, source: Source, lo: int, hi: int, contract: ScenarioInput | None = None
) -> LoadedScenario:
    """Read and validate one scenario file, which must cover months ``lo..hi`` for every
    scenario, against its IR declaration and, when given, the authoring contract's value column
    name, bounds and rules."""
    f = read_source(source, f"scenario input {decl.name}")
    label = f.where
    table = f.table
    value_column = contract.file_column if contract is not None else decl.name
    month_columns = [c for c in ("t", "u") if c in table.column_names]
    if len(month_columns) == 2:
        raise LoadError(f"{label}: the month column is named t or u, not both")
    if not month_columns:
        raise LoadError(
            f"{label}: the month column t (or u) is missing (needs scenario, t, {value_column})"
        )
    month_column = month_columns[0]
    for col in ("scenario", month_column, value_column):
        if col not in table.column_names:
            raise LoadError(
                f"{label}: column {col} is missing (needs scenario, {month_column}, {value_column})"
            )
        if table.column(col).null_count:
            raise LoadError(f"{label}: column {col} has null values")
    s = np.asarray(table.column("scenario").to_numpy(), dtype=np.int64)
    u = np.asarray(table.column(month_column).to_numpy(), dtype=np.int64)
    x = np.asarray(table.column(value_column).to_numpy(), dtype=np.float64)
    if len(s) == 0:
        raise LoadError(f"{label}: no rows")
    n_scen = int(s.max()) + 1
    if s.min() != 0 or len(np.unique(s)) != n_scen:
        raise LoadError(f"{label}: scenarios must be numbered 0 to J-1 without gaps")
    u0, u1 = int(u.min()), int(u.max())
    nu = u1 - u0 + 1
    flat = (u - u0) * n_scen + s
    if len(np.unique(flat)) != len(flat) or len(flat) != nu * n_scen:
        raise LoadError(
            f"{label}: every (scenario, {month_column}) in the file's range must appear once"
        )
    if u0 > lo or u1 < hi:
        raise LoadError(
            f"{label}: the model reads months {lo} to {hi} but the file covers {u0} to {u1}"
        )
    if not np.isfinite(x).all():
        raise LoadError(f"{label}: non-finite values")
    dense = np.zeros((nu, 1, n_scen), dtype=np.float64)
    dense.reshape(-1)[flat] = x
    loaded = LoadedScenario(decl, dense, u0, f)
    if contract is not None:
        view = loaded.view()
        _check_scenario_range(label, contract, view, view.values)
        for rule in contract.rules:
            ok = _run_check(rule.check, view, f"{label}: rule {rule.name}")
            if not ok.all():
                raise LoadError(
                    f"{label}: rule {rule.name} fails{_first_failing_row(view, ok)}{_doc(rule.doc)}"
                )
    return loaded


# ----------------------------------------------------------------------------------- series


@dataclass(slots=True)
class LoadedSeries:
    """A group series an aggregation stage reads (design section 9): each group's members
    summed over one or more outputs of a basis's projection runs, sequentially in ascending
    ``policy_id`` (D19).

    ``values`` is dense, shaped ``(Tn | 1, groups, scenarios)``: months ``0..H`` of the basis
    (one row for a series without a time dimension), the groups in ascending ``group_ids``, and
    the basis's scenarios, or one after a mean (``scenarios == "mean"``). ``unit`` and
    ``start`` are what the source runs recorded (the latest start of the variables summed);
    ``provenance`` is what the run manifest records about where the values came from; a run
    requires every key of :data:`SERIES_PROVENANCE` in it (D21) and float64 ``values``."""

    name: str
    values: NDArray[np.float64]
    unit: units.Unit
    has_time: bool
    start: int
    group_ids: NDArray[Any]
    scenarios: str
    """``"mean"`` (one scenario, the mean over the basis's) or ``"each"``."""
    provenance: dict[str, Any] = field(default_factory=dict)
    pathwise: NDArray[np.float64] | None = None
    """For a mean, the per-scenario group totals it averaged, ``(Tn | 1, groups, J)``, kept
    beside it in the run directory as ``<name>.pathwise.parquet`` (design section 9); ``None``
    for a series that keeps its scenarios."""

    @property
    def n_scenarios(self) -> int:
        return int(self.values.shape[2])

    @property
    def horizon(self) -> int:
        """The last month the series covers (0 without a time dimension)."""
        return int(self.values.shape[0]) - 1

    @property
    def basis(self) -> str:
        """The basis the series sums (its provenance's ``basis``)."""
        return str(self.provenance["basis"])


SERIES_PROVENANCE = (
    "basis",
    "variables",
    "basis_scenarios",
    "valuation_date",
    "basis_horizon",
    "source_runs",
    "members",
)
"""What a group series' provenance records, every key required when a stage runs on it: the
basis and the variables summed (which must be what the stage's declaration sums), the basis's
number of scenarios, valuation date and horizon (recorded for a series without a time dimension
too, so a ``cur`` basis that supplies only scalars still sets the stage's horizon), the source
runs (``run_id``, ``dir``, ``manifest_sha256``) and the membership the sums used (``[group id,
member ids ascending]`` per group, groups ascending: the members held by the runs of the
series' own basis, which are the policies its sums ran over).
:func:`~halley.aggregation.group_series` fills it."""

CUR_BASIS = "cur"
"""The basis whose horizon is the stage's (design section 9, "Horizon")."""


def _read(graph: ir.Graph) -> set[str]:
    """The series some formula or boundary reads, relatively or at a fixed month."""
    return {node.name for _v, node in graph.series_reads()}


def _relatively_read(graph: ir.Graph) -> set[str]:
    """The series some formula reads at a relative month, ``series.x(t + k)`` (a series without
    a time dimension, ``series.x()``, has no month to be relative to)."""
    return {
        node.name
        for _v, node in graph.series_reads()
        if isinstance(node, ir.SeriesRead) and node.offset is not None
    }


def _fixed_months(graph: ir.Graph) -> dict[str, set[int]]:
    """The constant months some formula or boundary reads a series at, ``series.x.at(k)``."""
    months: dict[str, set[int]] = {}
    for _v, n in graph.series_reads():
        if (
            isinstance(n, ir.SeriesAt)
            and isinstance(n.index, ir.Const)
            and not isinstance(n.index.value, bool)
        ):
            months.setdefault(n.name, set()).add(int(n.index.value))
    return months


def _summing(variables: Sequence[object]) -> str:
    """The variables a series sums as it adds their totals: ``fcf``, ``bel + ra``."""
    return " + ".join(str(v) for v in variables)


def _provenance_variables(s: LoadedSeries) -> tuple[object, ...]:
    got: object = s.provenance["variables"]
    if isinstance(got, (list, tuple)):
        return tuple(got)
    return (got,)


def _basis_horizon(s: LoadedSeries) -> int:
    """The horizon of the runs a series was summed from (checked by :func:`_check_given`)."""
    return int(s.provenance["basis_horizon"])


def _check_given(graph: ir.Graph, series: Mapping[str, LoadedSeries]) -> None:
    """One series per declaration and no other, each with float64 values and the provenance a
    run directory records (:data:`SERIES_PROVENANCE`), summed from what its declaration sums:
    the same variables, in the same order, of the same basis. The declaration is the stage's
    input contract, and its basis and variables are in the graph hash, so a stage edited to sum
    other outputs is another stage and runs only on series summed for it (D21)."""
    for name in graph.series:
        if name not in series:
            raise LoadError(f"the stage declares series {name!r}, which was not given")
    for name in series:
        if name not in graph.series:
            raise LoadError(f"series {name!r} was given but the stage declares no such series")
    for name, s in series.items():
        missing = [key for key in SERIES_PROVENANCE if key not in s.provenance]
        if missing:
            raise LoadError(
                f"series {name}: its provenance records no {', '.join(missing)}; a stage runs on "
                "series that say where they came from (halley.aggregation.group_series, D21)"
            )
        decl = graph.series[name]
        got_basis = s.provenance["basis"]
        got_variables = _provenance_variables(s)
        if got_basis != decl.basis or got_variables != decl.variables:
            raise LoadError(
                f"series {name}: the stage sums {_summing(decl.variables)} of basis {decl.basis} "
                f"but was given {_summing(got_variables)} of basis {got_basis}"
            )
        reach = s.provenance["basis_horizon"]
        if not isinstance(reach, int) or isinstance(reach, bool) or reach < 1:
            raise LoadError(
                f"series {name}: its provenance records basis_horizon {reach!r}, not the horizon "
                "of its basis's runs (an int of at least 1)"
            )
        if not isinstance(s.values, np.ndarray) or s.values.dtype != np.float64:
            kind = s.values.dtype if isinstance(s.values, np.ndarray) else type(s.values).__name__
            raise LoadError(f"series {name}: its values are {kind}, not float64")


def _stage_horizon(graph: ir.Graph, series: Mapping[str, LoadedSeries]) -> int | None:
    """A stage's default horizon (design section 9, "Horizon"): its ``cur`` basis's, which
    every series the stage declares of that basis records as ``basis_horizon``, whatever its
    time dimension (a ``cur`` basis that supplies only a scalar series sets it too). A stage
    without a ``cur`` series takes the horizon its series read relatively agree on (its series
    with a time dimension, when none is read relatively); series that disagree are a LoadError
    naming each one's basis and asking for ``RunSettings.horizon``. ``None`` without a ``cur``
    series or a series with a time dimension. The basis is the declaration's, which the series
    given must sum (:func:`_check_given`)."""
    _check_given(graph, series)
    cur = {n: _basis_horizon(series[n]) for n, d in graph.series.items() if d.basis == CUR_BASIS}
    if len(set(cur.values())) > 1:
        listing = "; ".join(f"series {n} records H = {h}" for n, h in cur.items())
        raise LoadError(
            f"the stage's series of basis cur disagree on its horizon: {listing}; the series of "
            "a basis are summed from the same runs"
        )
    if cur:
        return next(iter(cur.values()))
    timed = [n for n, d in graph.series.items() if d.has_time]
    relative = _relatively_read(graph)
    chosen = [n for n in timed if n in relative] or timed
    horizons = sorted({series[n].horizon for n in chosen})
    if len(horizons) > 1:
        listing = "; ".join(
            f"series {n} of basis {graph.series[n].basis} covers months 0..{series[n].horizon}"
            for n in chosen
        )
        raise LoadError(
            f"the stage's series disagree on its horizon: {listing}; a stage takes its cur "
            "basis's horizon, so without one set RunSettings.horizon"
        )
    return horizons[0] if horizons else None


def load_series(
    graph: ir.Graph,
    series: Mapping[str, LoadedSeries],
    ids: NDArray[Any],
    horizon: int,
    valuation_date: dt.date,
) -> dict[str, LoadedSeries]:
    """Check every group series the stage declares against what was supplied: one per
    declaration and no other, with float64 values and its provenance, summed from the declared
    variables of the declared basis, the stage's valuation date (every basis's must equal it),
    the declared unit (D11 across runs), time dimension, domain and scenario treatment, the
    stage's groups in its order, finite values. A series read relatively is then cut to months
    ``0..H``, which its basis must cover; the ``cur`` basis's horizon, as every series of it
    records, bounds ``H`` whatever the series' time dimension. A series read only at fixed
    months keeps every month of its basis, and each constant ``at(k)`` must be one of the
    months the series then covers (a per-group month is checked when the batch runs)."""
    _check_given(graph, series)
    relative = _relatively_read(graph)
    fixed = _fixed_months(graph)
    out: dict[str, LoadedSeries] = {}
    for name, decl in graph.series.items():
        s = series[name]
        label = f"series {name}"
        basis = decl.basis  # the series' own, by _check_given
        when = str(s.provenance["valuation_date"])
        if when != valuation_date.isoformat():
            raise LoadError(
                f"{label}: basis {basis} values at {when} but the stage values at "
                f"{valuation_date.isoformat()}; a stage and its bases share one valuation date"
            )
        if s.unit != decl.unit:
            raise LoadError(
                f"{label}: the stage declares it {decl.unit} but its basis records {s.unit} "
                "(D11 across runs)"
            )
        if s.has_time != decl.has_time:
            had = "has" if s.has_time else "has no"
            raise LoadError(
                f"{label}: the stage declares it {'with' if decl.has_time else 'without'} a time "
                f"dimension but it {had} one"
            )
        if s.start > decl.start:
            raise LoadError(
                f"{label}: the stage reads it from t = {decl.start} but it is defined from "
                f"t = {s.start}"
            )
        want = "each" if decl.scenario else "mean"
        if s.scenarios != want:
            raise LoadError(f"{label}: the stage declares scenarios={want!r}, not {s.scenarios!r}")
        if not np.array_equal(np.asarray(s.group_ids, dtype=object), np.asarray(ids, dtype=object)):
            raise LoadError(
                f"{label}: it covers groups {', '.join(map(str, s.group_ids))} but the model "
                f"points are {', '.join(map(str, ids))}"
            )
        values = s.values
        if values.ndim != 3 or values.shape[1] != len(ids):
            raise LoadError(
                f"{label}: values must be shaped (months, {len(ids)} groups, scenarios), not "
                f"{values.shape}"
            )
        if not decl.has_time and values.shape[0] != 1:
            raise LoadError(f"{label}: a series without a time dimension has one row")
        if not decl.scenario and values.shape[2] != 1:
            raise LoadError(f"{label}: a mean over scenarios has one scenario")
        if not np.isfinite(values).all():
            raise LoadError(f"{label}: non-finite values")
        recorded = [m[0] for m in s.provenance["members"]]
        if recorded != np.asarray(s.group_ids, dtype=object).tolist():
            raise LoadError(
                f"{label}: its provenance records the members of groups "
                f"{', '.join(map(str, recorded))}, not of the groups it covers"
            )
        if decl.has_time and name in relative and s.horizon < horizon:
            raise LoadError(
                f"{label}: basis {basis} covers months 0..{s.horizon} but the stage runs to "
                f"H = {horizon}; a series read relatively comes from a basis whose horizon is at "
                "least the stage's"
            )
        if basis == CUR_BASIS and _basis_horizon(s) < horizon:
            raise LoadError(
                f"{label}: basis cur covers months 0..{_basis_horizon(s)} but the stage runs to "
                f"H = {horizon}; the stage's horizon is at most its cur basis's"
            )
        pathwise = s.pathwise
        if pathwise is not None:
            _check_pathwise(label, decl, values, pathwise)
        if decl.has_time and name in relative:
            values = values[: horizon + 1]
            pathwise = None if pathwise is None else pathwise[: horizon + 1]
        last = int(values.shape[0]) - 1
        late = [k for k in sorted(fixed.get(name, ())) if k > last] if decl.has_time else []
        if late:
            why = (
                f"it is also read relatively, so it is cut to the stage's horizon, 0..{last}"
                if last < s.horizon
                else f"basis {basis} covers months 0..{last}"
            )
            raise LoadError(
                f"{label}: the stage reads series.{name}.at({late[0]}), but {why}; a series read "
                "at month k comes from a basis whose horizon is at least k"
            )
        out[name] = LoadedSeries(
            name,
            values,
            s.unit,
            s.has_time,
            s.start,
            s.group_ids,
            s.scenarios,
            s.provenance,
            pathwise,
        )
    return out


def _check_pathwise(
    label: str, decl: ir.SeriesDecl, values: NDArray[np.float64], pathwise: NDArray[Any]
) -> None:
    """A mean's pathwise values are the basis's per-scenario group totals it averaged: finite
    float64, shaped like the mean but for the scenarios. They are a record beside the mean,
    which is what the stage reads and what the run id covers."""
    if decl.scenario:
        raise LoadError(f"{label}: keeps its scenarios, so it has no pathwise values")
    if (
        not isinstance(pathwise, np.ndarray)
        or pathwise.dtype != np.float64
        or pathwise.ndim != 3
        or pathwise.shape[:2] != values.shape[:2]
        or not np.isfinite(pathwise).all()
    ):
        raise LoadError(
            f"{label}: its pathwise values must be finite float64 shaped like the mean, "
            f"{values.shape[:2]}, with the basis's scenarios"
        )


def read_series_copy(
    run_dir: Path,
    name: str,
    entry: Mapping[str, Any],
    id_field: str,
    label: str,
    read: Callable[[Mapping[str, Any], str], ReadFile] | None = None,
) -> LoadedSeries:
    """A group series as a stage's run directory keeps it: ``inputs/series/<name>.parquet`` in
    long form (the group id, ``scenario``, ``t``, ``value``, group-major, then month, then
    scenario) and, for a mean, its ``.pathwise`` sibling, each verified against the sha256 the
    manifest's ``entry`` records and read back to the dense ``(Tn | 1, groups, scenarios)``
    array, with the unit, domain, scenario treatment and provenance the entry records. A
    LoadError names the file that is missing, changed or laid out otherwise. ``read`` reads a
    copy from its entry (:func:`read_copy` by default; a run verified as a whole passes the
    reader of the bytes it verified)."""

    def from_disk(e: Mapping[str, Any], what: str) -> ReadFile:
        return read_copy(run_dir, e, what)

    reader = from_disk if read is None else read
    f = reader(entry, label)
    values, group_ids = _dense_series(f, id_field)
    pathwise: NDArray[np.float64] | None = None
    sibling = entry.get("pathwise")
    if sibling is not None:
        pf = reader(sibling, f"{label} (pathwise)")
        pathwise, pids = _dense_series(pf, id_field)
        if pids.tolist() != group_ids.tolist():
            raise LoadError(f"{pf.where}: it covers other groups than the series it sits beside")
    try:
        unit = units.unit_from_name(str(entry["unit"]))
        has_time, start = bool(entry["time"]), int(entry["start"])
        scenarios = str(entry["scenarios"])
    except (KeyError, TypeError, ValueError) as exc:
        raise LoadError(f"{label}: its manifest entry is incomplete: {exc}") from None
    provenance = {key: entry[key] for key in SERIES_PROVENANCE if key in entry}
    return LoadedSeries(
        name, values, unit, has_time, start, group_ids, scenarios, provenance, pathwise
    )


def _dense_series(f: ReadFile, id_field: str) -> tuple[NDArray[np.float64], NDArray[Any]]:
    """A series copy's long form back to ``(Tn, groups, scenarios)`` and its group ids."""
    table = f.table
    for col in (id_field, "scenario", "t", "value"):
        if col not in table.column_names:
            raise LoadError(f"{f.where}: column {col} is missing")
    ids = np.asarray(table.column(id_field).to_pylist(), dtype=object)
    scen = np.asarray(table.column("scenario").to_numpy(), dtype=np.int64)
    month = np.asarray(table.column("t").to_numpy(), dtype=np.int64)
    value = np.asarray(table.column("value").to_numpy(), dtype=np.float64)
    n = len(value)
    if n == 0:
        raise LoadError(f"{f.where}: no rows")
    n_s, n_t = int(scen.max()) + 1, int(month.max()) + 1
    n_g = n // (n_t * n_s)
    if (
        n_g * n_t * n_s != n
        or not np.array_equal(scen, np.tile(np.arange(n_s), n_g * n_t))
        or not np.array_equal(month, np.tile(np.repeat(np.arange(n_t), n_s), n_g))
    ):
        raise LoadError(f"{f.where}: the rows are not group-major, then month, then scenario")
    group_ids = ids[:: n_t * n_s]
    if not np.array_equal(ids, np.repeat(group_ids, n_t * n_s)):
        raise LoadError(f"{f.where}: the rows are not group-major, then month, then scenario")
    values = np.ascontiguousarray(value.reshape(n_g, n_t, n_s).transpose(1, 0, 2))
    return values, group_ids


def _stage_scenarios(
    graph: ir.Graph, scenarios: Mapping[str, LoadedScenario], series: Mapping[str, LoadedSeries]
) -> int:
    """The number of scenarios a run has: the scenario inputs' and the per-scenario series'
    common J, or 1 without either. A stage that reads a mean over scenarios runs at S = 1, so
    then every scenario input and every per-scenario series it reads is deterministic: one
    check, :func:`~halley.aggregation.require_deterministic` (D4). Only the series some formula
    reads count: a declared series no formula reads (a BuildWarning) sets nothing."""
    read = _read(graph)
    means = [name for name, d in graph.series.items() if not d.scenario and name in read]
    each = [series[name] for name, d in graph.series.items() if d.scenario and name in read]
    if means:
        from halley.aggregation import require_deterministic  # the one J = 1 check

        reason = f"the stage reads the mean series {means[0]}, so it runs at S = 1"
        for name, sc in scenarios.items():
            require_deterministic(sc, f"scenario input {name}", reason, LoadError)
        for s in each:
            require_deterministic(s, f"series {s.name}", reason, LoadError)
    counts = {s.n_scenarios for s in scenarios.values()} | {s.n_scenarios for s in each}
    if len(counts) > 1:
        what = "scenario inputs and per-scenario series" if each else "scenario inputs"
        raise LoadError(f"{what} disagree on the number of scenarios: {sorted(counts)}")
    return counts.pop() if counts else 1


# ----------------------------------------------------------------------------------- inputs


def run_constants(valuation_date: dt.date, horizon: int) -> dict[str, int]:
    """The run constants a formula reads as ``self.run.<name>``: ``m_val``, the valuation month
    index ``12 * year + month - 1``, and ``horizon``."""
    return {
        "m_val": 12 * valuation_date.year + valuation_date.month - 1,
        "horizon": horizon,
    }


@dataclass(slots=True)
class Inputs:
    model_points: ModelPoints
    tables: dict[str, LoadedTable]
    scenarios: dict[str, LoadedScenario]
    n_scenarios: int
    H: int
    run: dict[str, int] = field(default_factory=dict)
    """The run constants (:func:`run_constants`), supplied to every batch as ``run.<name>``."""
    extras: dict[str, Any] = field(default_factory=dict)
    series: dict[str, LoadedSeries] = field(default_factory=dict)
    """An aggregation stage's group series (:func:`load_series`), supplied to every batch as
    ``series.<name>`` sliced to the batch's groups."""

    @property
    def ids(self) -> NDArray[Any]:
        return self.model_points.ids

    def scales(self, graph: ir.Graph) -> dict[units.Unit, NDArray[np.float64]]:
        """Per-unit tolerance scales per policy, from the model's ``tolerance_scales``."""
        out: dict[units.Unit, NDArray[np.float64]] = {}
        for unit, name in graph.scale_fields.items():
            out[unit] = np.asarray(self.model_points.columns[name], dtype=np.float64)
        return out

    def view(self, valuation_date: dt.date) -> InputView:
        """Every input as the product's input rules see it."""
        mp = self.model_points
        return InputView(
            tables={name: t.view() for name, t in self.tables.items()},
            model_points=LoadContext(mp.columns, valuation_date),
            scenarios={name: s.view() for name, s in self.scenarios.items()},
            valuation_date=valuation_date,
            valuation_month=self.run["m_val"],
            horizon=self.H,
        )

    def batch(self, graph: ir.Graph, lo: int, hi: int) -> dict[str, Any]:
        """The arrays the generated code reads for policies ``lo:hi``."""
        mp = self.model_points
        p = hi - lo
        inp: dict[str, Any] = {
            "P": p,
            "S": self.n_scenarios,
            "policy_ids": mp.ids[lo:hi],
        }
        for name, decl in graph.fields.items():
            if not decl.readable:
                continue
            if decl.enum is not None:
                arr = mp.codes[name][lo:hi]
            elif decl.kind == "int":
                arr = mp.columns[name][lo:hi].astype(np.int64)
            else:
                arr = mp.columns[name][lo:hi].astype(np.float64)
            inp[f"mp.{name}"] = arr.reshape(1, p, 1)
        for tname, t in self.tables.items():
            for cname, dense in t.values.items():
                inp[f"tbl.{tname}.{cname}"] = dense
        for sname, sc in self.scenarios.items():
            inp[f"scen.{sname}"] = sc.values
            inp[f"scen.{sname}.u0"] = sc.u0
        for rname, value in self.run.items():
            inp[f"run.{rname}"] = value
        for yname, s in self.series.items():
            inp[f"series.{yname}"] = s.values[:, lo:hi, :]  # the groups are the batch's policies
        return inp


def load_inputs(
    graph: ir.Graph,
    schema: Schema,
    *,
    model_points: Source,
    tables: dict[str, Source] | None,
    scenarios: dict[str, Source] | None,
    valuation_date: dt.date,
    horizon: int | None,
    table_contracts: Mapping[str, Table] | None = None,
    scenario_contracts: Mapping[str, ScenarioInput] | None = None,
    input_rules: Sequence[InputRule] = (),
    series: Mapping[str, LoadedSeries] | None = None,
    lanes: str = "policies",
    model_points_label: str = "model points",
) -> Inputs:
    """Load every input the model reads, in order: model points, tables, the horizon, an
    aggregation stage's group series, scenario files, then the product's input rules over all of
    them. The contracts, when given, add the tables' required metadata and rules, the scenario
    inputs' column names, bounds, coverage and rules. A stage's horizon defaults to its cur
    basis's (:func:`_stage_horizon`) when its schema names no horizon field. A failure names the
    model points by their ids under ``lanes``, ``"groups"`` for a stage, and their file by
    ``model_points_label``."""
    table_contracts = table_contracts or {}
    scenario_contracts = scenario_contracts or {}
    series = series or {}
    mp = load_model_points(graph, schema, model_points, valuation_date, lanes, model_points_label)
    tables = tables or {}
    loaded_tables: dict[str, LoadedTable] = {}
    for name, decl in graph.tables.items():
        if name not in tables:
            raise LoadError(f"the model reads assumption table {name!r}, which was not given")
        loaded_tables[name] = load_table(decl, tables[name], table_contracts.get(name))
    for name in tables:
        if name not in graph.tables:
            raise LoadError(f"table {name!r} was given but the model declares no such table")
    if horizon is None:
        if schema.horizon is not None:
            horizon = int(mp.columns[schema.horizon].max())
        else:
            horizon = _stage_horizon(graph, series)
            if horizon is None:
                raise LoadError("the run needs a horizon: the schema names no horizon field")
    if horizon < 1:
        raise LoadError(f"the horizon must be at least 1 month, not {horizon}")
    loaded_series = load_series(graph, series, mp.ids, horizon, valuation_date)
    scenarios = scenarios or {}
    req = scenario_requirements(graph, horizon)
    loaded_scen: dict[str, LoadedScenario] = {}
    for name, decl in graph.scenarios.items():
        if name not in scenarios:
            raise LoadError(f"the model reads scenario input {name!r}, which was not given")
        lo, hi = req[name]
        contract = scenario_contracts.get(name)
        if contract is not None and contract.covers is not None:
            lo = min(lo, resolve_month(contract.covers[0], horizon))
            hi = max(hi, resolve_month(contract.covers[1], horizon))
        loaded_scen[name] = load_scenario(decl, scenarios[name], lo, hi, contract)
    for name in scenarios:
        if name not in graph.scenarios:
            raise LoadError(f"scenario input {name!r} was given but the model declares none")
    n_scenarios = _stage_scenarios(graph, loaded_scen, loaded_series)
    inputs = Inputs(
        mp,
        loaded_tables,
        loaded_scen,
        n_scenarios,
        horizon,
        run_constants(valuation_date, horizon),
        series=loaded_series,
    )
    if input_rules:
        view = inputs.view(valuation_date)
        for rule in input_rules:
            ok = _run_check(rule.check, view, f"input rule {rule.name}")
            if not ok.all():
                where = _first_failing_input(view, inputs.ids, ok, lane_noun(lanes))
                raise LoadError(f"input rule {rule.name} fails{where}{_doc(rule.doc)}")
    return inputs


def _lines_up(ok: NDArray[np.bool_], shape: tuple[int, ...]) -> bool:
    """Whether a rule's result has a table's dense shape, axes of length 1 allowed."""
    return (
        ok.ndim == len(shape)
        and ok.ndim > 0
        and all(n in (1, full) for n, full in zip(ok.shape, shape, strict=True))
    )


def _first_failing_input(
    view: InputView, ids: NDArray[Any], ok: NDArray[np.bool_], lane: str = "policy"
) -> str:
    """Where an input rule first fails, by the shape of its result: ``at sex = M, age = 25 of
    table mort`` when it lines up with exactly one table's dense shape, ``for policy P05`` (``for
    group G0`` in a stage, by ``lane``) when it runs over the model points, ``for scenario 1,
    month -14 of scenario input cpi`` when it lines up with exactly one scenario input, else the
    element's position (also when the shape fits more than one input); with the count of
    failing values when the result has more than one. A one-element result of a one-policy,
    one-scenario run fits the policy axis and every scenario input's scenario axis alike; it
    names the policy."""
    if ok.ndim == 0:
        return ""
    tables: list[str] = []
    for name, table in view.tables.items():
        if _lines_up(ok, table.shape):
            key = _first_failing_key(table, ok).removeprefix(" at ")
            tables.append(f"at {key} of table {name}" if key else f"in table {name}")
    policy = f"for {lane} {ids[int(np.flatnonzero(~ok)[0])]}" if ok.shape == ids.shape else None
    scenarios: list[str] = []
    for name, scen in view.scenarios.items():
        row = _first_failing_row(scen, ok)
        if row:
            scenarios.append(f"{row.strip()} of scenario input {name}")
    candidates = [*tables, *([policy] if policy else []), *scenarios]
    if len(candidates) == 1:
        where = candidates[0]
    elif policy is not None and not tables and ok.size == 1:
        where = policy
    else:
        where = f"at element {int(np.flatnonzero(~ok.reshape(-1))[0])} of the rule's result"
    if ok.size == 1:
        return f" {where}"
    return f" for {int((~ok).sum())} of {ok.size} values, first {where}"
