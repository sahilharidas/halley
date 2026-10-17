# pyright: basic
"""Loader: read and validate model points, assumption tables and scenario files.

Everything is checked against the contracts the model declares before a batch runs: field types
and ranges, cross-field rules, derived fields, table key coverage, value ranges, metadata and
rules, scenario coverage of every month the model reads, value ranges and rules, and the
product's input rules once everything is loaded. A failure is a
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
"""A Parquet file path or an in-memory ``pyarrow.Table``."""

METADATA_PREFIX = "halley."


@dataclass(slots=True)
class ReadFile:
    """An input as read: the Arrow table, its bytes' hash, its ``halley.*`` schema metadata (and
    a bare ``version``) as strings without the prefix and, when given as a file, its path."""

    table: pa.Table
    sha256: str
    path: Path | None
    raw: bytes | None  # the Parquet bytes of an in-memory table, for the run directory copy
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
        table = pq.read_table(path)
    except (pa.ArrowException, OSError) as exc:
        raise LoadError(f"{label}: {path} is not a Parquet file: {exc}") from exc
    return ReadFile(
        table, hashlib.sha256(data).hexdigest(), path, None, label, _schema_metadata(table)
    )


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


def _check_range(values: NDArray[Any], decl: ir.FieldDecl, ids: NDArray[Any], label: str) -> None:
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
                f"{label}: field {decl.name} must be {symbol} {bound}; policies "
                f"{_ids(ids, bad)} break this"
            )
    if decl.kind == "number" and not np.isfinite(values).all():
        raise LoadError(f"{label}: field {decl.name} has non-finite values")


def load_model_points(
    graph: ir.Graph, schema: Schema, source: Source, valuation_date: dt.date
) -> ModelPoints:
    f = read_source(source, "model points")
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
                    f"{label}: field {fld.name} takes exactly {', '.join(decl.enum)}; policies "
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
        _check_range(values, decl, ids, label)
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
        _check_range(values, decl, ids, label)
        columns[d.name] = values
    for rule in schema.rules:
        ok = np.asarray(rule.check(ctx), dtype=bool)
        if ok.shape != ids.shape:
            raise LoadError(f"{label}: rule {rule.name} must give one boolean per row")
        if not ok.all():
            raise LoadError(
                f"{label}: rule {rule.name} fails for policies {_ids(ids, ~ok)}{_doc(rule.doc)}"
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
) -> Inputs:
    """Load every input the model reads, in order: model points, tables, the horizon, scenario
    files, then the product's input rules over all of them. The contracts, when given, add the
    tables' required metadata and rules, the scenario inputs' column names, bounds, coverage and
    rules."""
    table_contracts = table_contracts or {}
    scenario_contracts = scenario_contracts or {}
    mp = load_model_points(graph, schema, model_points, valuation_date)
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
        if schema.horizon is None:
            raise LoadError("the run needs a horizon: the schema names no horizon field")
        horizon = int(mp.columns[schema.horizon].max())
    if horizon < 1:
        raise LoadError(f"the horizon must be at least 1 month, not {horizon}")
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
    counts = {s.n_scenarios for s in loaded_scen.values()}
    if len(counts) > 1:
        raise LoadError(f"scenario inputs disagree on the number of scenarios: {sorted(counts)}")
    n_scenarios = counts.pop() if counts else 1
    inputs = Inputs(
        mp, loaded_tables, loaded_scen, n_scenarios, horizon, run_constants(valuation_date, horizon)
    )
    if input_rules:
        view = inputs.view(valuation_date)
        for rule in input_rules:
            ok = _run_check(rule.check, view, f"input rule {rule.name}")
            if not ok.all():
                where = _first_failing_input(view, inputs.ids, ok)
                raise LoadError(f"input rule {rule.name} fails{where}{_doc(rule.doc)}")
    return inputs


def _lines_up(ok: NDArray[np.bool_], shape: tuple[int, ...]) -> bool:
    """Whether a rule's result has a table's dense shape, axes of length 1 allowed."""
    return (
        ok.ndim == len(shape)
        and ok.ndim > 0
        and all(n in (1, full) for n, full in zip(ok.shape, shape, strict=True))
    )


def _first_failing_input(view: InputView, ids: NDArray[Any], ok: NDArray[np.bool_]) -> str:
    """Where an input rule first fails, by the shape of its result: ``at sex = M, age = 25 of
    table mort`` when it lines up with exactly one table's dense shape, ``for policy P05`` when
    it runs over the model points, ``for scenario 1, month -14 of scenario input cpi`` when it
    lines up with exactly one scenario input, else the element's position (also when the shape
    fits more than one input); with the count of failing values when the result has more than
    one. A one-element result of a one-policy, one-scenario run fits the policy axis and every
    scenario input's scenario axis alike; it names the policy."""
    if ok.ndim == 0:
        return ""
    tables: list[str] = []
    for name, table in view.tables.items():
        if _lines_up(ok, table.shape):
            key = _first_failing_key(table, ok).removeprefix(" at ")
            tables.append(f"at {key} of table {name}" if key else f"in table {name}")
    policy = f"for policy {ids[int(np.flatnonzero(~ok)[0])]}" if ok.shape == ids.shape else None
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
