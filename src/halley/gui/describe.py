# pyright: basic
"""What the GUI's pages show, as JSON: builds, runs, lanes, explanations, comparisons and errors.

Every function here reads Halley's own objects (a :class:`~halley.CompiledModel`, a
:class:`~halley.RunResult`, a run's manifest, an exception) and returns plain data; nothing here
computes a value the engine has not. An explanation is :func:`halley.explain`'s, with each input
it lists given a target the page can explain in turn, found by walking the formula's nodes as
``explain`` does; a comparison is :func:`halley.diff`'s, beside what each run's id covers.
"""

from __future__ import annotations

import hashlib
import json
import linecache
import math
import struct
import sys
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

import halley
from halley import ir
from halley.aggregation import portfolio_totals
from halley.authoring.contracts import resolve_month
from halley.codegen import BACKENDS, CompiledModel
from halley.errors import HalleyError, Loc
from halley.gui.catalog import (
    ModelFileError,
    exception_name,
    exception_text,
    format_quietly,
    is_library_file,
    plain,
    traceback_of,
)
from halley.runtime import RunResult, recompute
from halley.runtime.loader import read_copy
from halley.runtime.run import (
    IDENTITY_VERSIONS,
    NOT_IN_IDENTITY,
    _membership_sha256,  # pyright: ignore[reportPrivateUsage]
)

LEAVES = (
    ir.VarRead,
    ir.VarAt,
    ir.SumT,
    ir.MpField,
    ir.TableLookup,
    ir.ScenRead,
    ir.ScenAt,
    ir.SeriesRead,
    ir.SeriesAt,
    ir.RunConst,
)
"""The nodes ``explain`` lists as a formula's inputs, in the order it walks them."""

MAX_SERIES = 8
"""The most series one chart draws: the categorical palette's eight colours."""


# --------------------------------------------------------------------------------- errors


def loc_json(loc: Loc | None) -> dict[str, object] | None:
    if loc is None:
        return None
    return {"file": loc.file, "line": loc.line, "function": loc.function}


def snippet(file: str, line: int, before: int = 3, after: int = 6) -> list[dict[str, object]]:
    """The lines around ``line`` of ``file`` as the file is now, each marked when it is the one
    meant; empty when the file cannot be read."""
    linecache.checkcache(file)
    lines = linecache.getlines(file)
    if not lines or not 1 <= line <= len(lines):
        return []
    first, last = max(1, line - before), min(len(lines), line + after)
    return [
        {"line": n, "text": lines[n - 1].rstrip("\r\n"), "here": n == line}
        for n in range(first, last + 1)
    ]


def _where_raised(exc: BaseException) -> Loc | None:
    """Where an exception of the user's points in their code: a syntax error's line, as Python
    names it, else the deepest frame of its traceback in the user's code (see
    :func:`_user_frame`); None when neither can be had, whatever its attributes raise."""
    if issubclass(type(exc), SyntaxError):
        try:
            error = cast(SyntaxError, exc)
            file, line = error.filename, error.lineno
            if isinstance(file, str) and isinstance(line, int) and file and line:
                return Loc(str.__str__(file), int(line), "<module>")
        except BaseException:  # a filename or lineno of the user's class that raises
            pass
    try:
        return _user_frame(exc)
    except BaseException:
        return None


def _user_frame(exc: BaseException) -> Loc | None:
    """The deepest frame of an exception's traceback in code that is neither Halley's, Python's
    own nor an installed package's: where in a model file a plain Python error happened."""
    found: Loc | None = None
    for frame in traceback.extract_tb(traceback_of(exc)):
        name = frame.filename
        if name.startswith("<") or frame.lineno is None or is_library_file(name):
            continue
        found = Loc(name, frame.lineno, frame.name)
    return found


def describe_error(exc: BaseException) -> dict[str, object]:
    """An exception as the page shows it: its type, its message, the model source line it points
    at with the lines around it, and, for an exception that is not one of Halley's own, the
    traceback. A Halley error's location is its own (``exc.loc``); a syntax error's is the line
    Python names; any other exception's is the deepest frame in the user's code. An exception
    of the user's is told by its class, and what it holds is read where it cannot fail the
    answer: one whose frames or file cannot be read has no line."""
    mine = issubclass(type(exc), ModelFileError)
    cause: BaseException = exc
    if mine and cast(ModelFileError, exc).error is not None:
        cause = cast(BaseException, cast(ModelFileError, exc).error)
    halley_error = issubclass(type(cause), HalleyError)
    out: dict[str, object] = {"type": exception_name(cause)}
    loc: Loc | None = None
    if mine:
        out["message"] = cast(ModelFileError, exc).message
    elif issubclass(type(exc), HalleyError):
        out["message"] = getattr(exc, "message", None) or str(exc)
        loc = getattr(exc, "loc", None)
        out["kind"] = _kind(cast(HalleyError, exc))
    else:
        out["message"] = f"{exception_name(exc)}: {exception_text(exc)}"
    if loc is None and not halley_error:
        loc = _where_raised(cause)
    if loc is not None:
        out["loc"] = loc_json(loc)
        try:
            out["snippet"] = snippet(loc.file, loc.line)
        except BaseException:  # a file that cannot be read now
            out["snippet"] = []
    notes = _notes(cause)
    if notes:
        out["notes"] = notes
    if not halley_error:
        # a model file's error as written out when it was made, before the clean-up (see
        # format_quietly)
        written = cast(ModelFileError, exc).traceback if mine else None
        out["traceback"] = written or format_quietly(cause)
    return out


def _notes(exc: BaseException) -> list[str]:
    """An exception's notes as Python's traceback writes them: each note's text, or
    ``<note str() failed>`` for one whose text cannot be had, and ``__notes__`` that is not a
    sequence of them (a text, say) as its repr."""

    def safe(write: Callable[[object], str], value: object, what: str) -> str:
        try:
            return write(value)
        except BaseException:
            return f"<{what} {write.__name__}() failed>"

    try:
        notes: object = getattr(exc, "__notes__", None)
    except BaseException as e:
        return [f"Ignored error getting __notes__: {safe(repr, e, '__notes__')}"]
    if notes is None:
        return []
    try:
        if isinstance(notes, Sequence) and not isinstance(notes, (str, bytes)):
            return [safe(str, n, "note") for n in cast(Sequence[object], notes)]
    except BaseException:  # notes of the user's that cannot be read as a sequence
        return []
    return [safe(repr, notes, "__notes__")]


def _kind(exc: HalleyError) -> str:
    for cls, name in (
        (halley.UnitError, "unit"),
        (halley.BuildError, "build"),
        (halley.LoadError, "load"),
        (halley.RunError, "run"),
    ):
        if isinstance(exc, cls):
            return name
    return "halley"


# --------------------------------------------------------------------------------- builds


def _reads(graph: ir.Graph, v: ir.Variable) -> list[dict[str, object]]:
    """The variables a variable reads, each once per offset, in the order its formula reads
    them, boundary values included."""
    seen: dict[tuple[str, int | None], dict[str, object]] = {}
    for _node, target, offset in graph.reads(v):
        key = (target, offset)
        if key in seen:
            continue
        if offset is None:
            label = f"{target} (all months)"
        elif not graph.variables[target].has_time:
            label = f"{target}()"
        else:
            label = f"{target}(t)" if offset == 0 else f"{target}(t{offset:+d})"
        seen[key] = {"var": target, "offset": offset, "label": label}
    return list(seen.values())


def _input_names(v: ir.Variable) -> list[str]:
    """The model point fields, tables, scenario inputs, group series and run constants a
    variable's formula and boundary values read, by name."""
    names: dict[str, None] = {}
    for expr in (v.expr, v.init, v.terminal):
        if expr is None:
            continue
        for node in ir.walk(expr):
            if isinstance(node, ir.MpField):
                names[f"mp.{node.name}"] = None
            elif isinstance(node, ir.TableLookup):
                names[f"tables.{node.table}.{node.column}"] = None
            elif isinstance(node, (ir.ScenRead, ir.ScenAt)):
                names[f"scen.{node.name}"] = None
            elif isinstance(node, (ir.SeriesRead, ir.SeriesAt)):
                names[f"series.{node.name}"] = None
            elif isinstance(node, ir.RunConst):
                names[f"run.{node.name}"] = None
    return list(names)


def describe_compiled(compiled: CompiledModel) -> dict[str, object]:
    """A build as the model page shows it: every variable with its unit, role, timing, domain,
    recursion direction, formula, source and what it reads and is read by; the components in
    the order the schedule evaluates them; and what a run of it needs."""
    graph = compiled.graph
    needed = compiled.schedule.needed
    stored = set(graph.outputs) if graph.outputs is not None else set(needed)
    reads = {name: _reads(graph, v) for name, v in graph.variables.items()}
    read_by: dict[str, set[str]] = {name: set() for name in graph.variables}
    for name, targets in reads.items():
        for r in targets:
            read_by[str(r["var"])].add(name)
    variables = []
    for name in sorted(graph.variables):
        v = graph.variables[name]
        first, last = v.formula_range
        variables.append(
            {
                "name": name,
                "unit": v.declared_unit.name,
                "time": v.has_time,
                "timing": v.timing,
                "role": v.role,
                "domain": v.domain,
                "start": v.start,
                "component": v.component,
                "direction": v.direction,
                "formula_range": [first, last] if v.has_time else None,
                "formula": ir.render(v.expr),
                "init": None if v.init is None else ir.render(v.init),
                "terminal": None if v.terminal is None else ir.render(v.terminal),
                "loc": loc_json(v.loc),
                "source": v.source,
                "doc": v.doc,
                "needed": name in needed,
                "stored": name in stored,
                "reads": reads[name],
                "read_by": sorted(read_by[name]),
                "inputs": _input_names(v),
            }
        )
    components = [
        {
            "index": i,
            "members": list(members),
            "direction": graph.variables[members[0]].direction,
            "needed": any(m in needed for m in members),
        }
        for i, members in enumerate(graph.components)
    ]
    return {
        "name": compiled.name,
        "backend": compiled.backend,
        "artefact": BACKENDS[compiled.backend].artefact,
        "ir_sha256": compiled.ir_hash,
        "statics": plain(graph.statics),
        "source_file": graph.source_file,
        "stage": compiled.model.series_declarations() is not None,
        "variables": variables,
        "components": components,
        "outputs": None if graph.outputs is None else list(graph.outputs),
        "needed": sorted(needed),
        "id_field": compiled.schema.id_field,
        "horizon_field": graph.horizon_field,
        "tables": sorted(graph.tables),
        "scenarios": sorted(graph.scenarios),
        "series": sorted(graph.series),
    }


# ---------------------------------------------------------------------------------- runs


def invariant_counts(raw: object) -> dict[str, int]:
    rows = raw if isinstance(raw, list) else []
    failed = sum(1 for r in rows if isinstance(r, dict) and r.get("passed") is False)
    return {"checked": len(rows), "failed": failed}


def run_summary(
    run_dir: Path, manifest: Mapping[str, Any], invariants: object
) -> dict[str, object]:
    """A run directory as the runs list shows it, from its manifest and invariant results alone
    (its outputs are not read)."""
    model = manifest.get("model", {})
    settings = manifest.get("settings", {})
    dims = manifest.get("dimensions", {})
    outputs = manifest.get("outputs", {})
    return {
        "dir": str(run_dir),
        "name": run_dir.name,
        "run_id": manifest.get("run_id"),
        "model": model.get("name"),
        "kind": outputs.get("kind", "projection"),
        "label": settings.get("label", ""),
        "valuation_date": settings.get("valuation_date"),
        "created_at": manifest.get("created_at"),
        "backend": manifest.get("backend", {}).get("name"),
        "policies": dims.get("policies"),
        "scenarios": dims.get("scenarios"),
        "horizon": dims.get("horizon"),
        "invariants": invariant_counts(invariants),
        "replayed_from": manifest.get("replayed_from"),
    }


def _input_rows(manifest: Mapping[str, Any]) -> list[dict[str, object]]:
    inputs = manifest.get("inputs", {})
    rows: list[dict[str, object]] = []

    def add(kind: str, name: str, entry: Mapping[str, Any]) -> None:
        rows.append(
            {
                "kind": kind,
                "name": name,
                "path": entry.get("path"),
                "copy": entry.get("copy"),
                "sha256": entry.get("sha256"),
                "metadata": entry.get("metadata", {}),
                "rows": entry.get("rows"),
                "version": entry.get("version"),
                "scenarios": entry.get("scenarios"),
                "months": entry.get("months"),
            }
        )

    if "model_points" in inputs:
        add("model_points", "model points", inputs["model_points"])
    for name, entry in inputs.get("tables", {}).items():
        add("table", name, entry)
    for name, entry in inputs.get("scenarios", {}).items():
        add("scenario", name, entry)
    for name, entry in inputs.get("series", {}).items():
        add("series", name, entry)
    for key in ("groups", "derived_groups", "membership"):
        if key in inputs:
            add(key, key.replace("_", " "), inputs[key])
    return rows


def describe_run(result: RunResult, in_session: bool, max_ids: int = 20_000) -> dict[str, object]:
    """A run as its page shows it: the manifest's blocks, the variables it holds with their
    units, the invariant results, the inputs with their hashes and whether ``explain`` can be
    asked of it in this session."""
    manifest = result.manifest
    stored = manifest.get("outputs", {}).get("variables", {})
    graph = result.graph
    variables = []
    for name in result.variables:
        info = stored.get(name)
        if info is not None:
            unit, start, role = info.get("unit"), info.get("start"), info.get("role")
        elif graph is not None and name in graph.variables:
            v = graph.variables[name]
            unit, start, role = v.declared_unit.name, v.start, v.role
        else:
            unit = start = role = None
        known = graph is not None and name in graph.variables
        variables.append(
            {
                "name": name,
                "time": bool(result.has_time.get(name, False)),
                "unit": unit,
                "start": start,
                "role": role,
                "stored": info is not None,
                "doc": graph.variables[name].doc if graph is not None and known else None,
            }
        )
    ids = result.ids[:max_ids]
    stage_blocks = {
        key: manifest[key]
        for key in (
            "previous",
            "bases",
            "member_rules",
            "locked_in",
            "provenance",
            "disclosures",
            "replayed_from",
        )
        if key in manifest
    }
    return {
        "dir": str(result.run_dir),
        "name": result.run_dir.name,
        "run_id": result.run_id,
        "kind": result.kind,
        "lane": "group" if result.kind == "aggregation" else "policy",
        "model": manifest.get("model", {}),
        # as Python writes the recorded settings: a page's JSON would round 2**53 + 1 and drop
        # the .0 of 1.0
        "statics_text": json.dumps(
            manifest.get("model", {}).get("statics") or {}, ensure_ascii=False
        ),
        "created_at": manifest.get("created_at"),
        "halley": manifest.get("halley"),
        "settings": manifest.get("settings", {}),
        "dimensions": manifest.get("dimensions", {}),
        "backend": manifest.get("backend", {}),
        "ir_sha256": manifest.get("ir_sha256"),
        "manifest_sha256": result.manifest_sha256,
        "H": result.H,
        "n_policies": result.n_policies,
        "n_scenarios": result.n_scenarios,
        "ids": ids,
        "ids_truncated": len(result.ids) > max_ids,
        "variables": variables,
        "invariants": [r.as_dict() for r in result.invariants],
        "inputs": _input_rows(manifest),
        "stage": stage_blocks,
        "in_session": in_session,
        "explainable": result.graph is not None and result.inputs is not None,
    }


def _scenarios(result: RunResult, spec: Sequence[int]) -> list[int]:
    for s in spec:
        if not 0 <= s < result.n_scenarios:
            raise ValueError(
                f"scenario {s} is outside this run's scenarios 0..{result.n_scenarios - 1}"
            )
    return list(spec)


def _policies(result: RunResult, spec: Sequence[int]) -> list[int]:
    for p in spec:
        if not 0 <= p < result.n_policies:
            raise ValueError(f"policy index {p} is outside 0..{result.n_policies - 1}")
    return list(spec)


def lanes(
    result: RunResult, name: str, policies: Sequence[int], scenarios: Sequence[int]
) -> dict[str, object]:
    """One variable for the chosen policies (by their index in the run) and scenarios: one
    series per pair, over every month for a monthly variable, one value for a scalar one. At most
    :data:`MAX_SERIES` series; the answer says how many were asked for."""
    if name not in result.arrays:
        raise KeyError(name)
    ps, ss = _policies(result, policies), _scenarios(result, scenarios)
    arr = result.var(name)
    pairs = [(p, s) for p in ps for s in ss]
    series = [
        {"p": p, "id": result.ids[p], "scenario": s, "values": np.asarray(arr[:, p, s])}
        for p, s in pairs[:MAX_SERIES]
    ]
    time = bool(result.has_time.get(name, False))
    return {
        "variable": name,
        "time": time,
        "t": list(range(result.H + 1)) if time else None,
        "series": series,
        "asked": len(pairs),
    }


def totals(result: RunResult, name: str, scenarios: Sequence[int]) -> dict[str, object]:
    """A variable summed over every policy in ascending id, sequentially (the engine's own
    portfolio total, D19), per month and scenario."""
    if name not in result.arrays:
        raise KeyError(name)
    ss = _scenarios(result, scenarios)
    total = portfolio_totals(result, [name])[name]
    time = bool(result.has_time.get(name, False))
    return {
        "variable": name,
        "time": time,
        "t": list(range(result.H + 1)) if time else None,
        "series": [{"scenario": s, "values": total[:, 0, s]} for s in ss[:MAX_SERIES]],
        "asked": len(ss),
    }


# ------------------------------------------------------------------------------- explain


def _targets(graph: ir.Graph, v: ir.Variable, t: int | None) -> dict[str, dict[str, object]]:
    """For each input ``explain`` lists for ``v`` (by its rendered text), what the page can open
    from it: another variable at a month, a variable's whole lane, a model point field, a
    table, a scenario input, a group series or a run constant."""
    out: dict[str, dict[str, object]] = {}
    for node in ir.walk(v.expr):
        if not isinstance(node, LEAVES):
            continue
        text = ir.render(node)
        if text in out:
            continue
        if isinstance(node, ir.VarRead):
            target = graph.variables[node.name]
            month = t + node.offset if target.has_time and t is not None else None
            out[text] = {"kind": "variable", "var": node.name, "t": month}
        elif isinstance(node, ir.VarAt):
            index = node.index
            month = None
            if isinstance(index, ir.Const) and isinstance(index.value, int):
                month = None if isinstance(index.value, bool) else index.value
            out[text] = {"kind": "variable", "var": node.name, "t": month}
        elif isinstance(node, ir.SumT):
            out[text] = {"kind": "lane", "var": node.name}
        elif isinstance(node, ir.MpField):
            out[text] = {"kind": "field", "name": node.name}
        elif isinstance(node, ir.TableLookup):
            out[text] = {"kind": "table", "name": node.table, "column": node.column}
        elif isinstance(node, (ir.ScenRead, ir.ScenAt)):
            out[text] = {"kind": "scenario", "name": node.name}
        elif isinstance(node, (ir.SeriesRead, ir.SeriesAt)):
            out[text] = {"kind": "series", "name": node.name}
        elif isinstance(node, ir.RunConst):
            out[text] = {"kind": "run", "name": node.name}
    return out


def _relative(again: object, stored: object) -> float | None:
    """``|again - stored| / max(|again|, |stored|)``, 0 when both are 0; None when either is not
    a number."""
    numbers = (int, float, np.integer, np.floating)
    if not isinstance(again, numbers) or not isinstance(stored, numbers):
        return None
    x, y = float(again), float(stored)
    scale = max(abs(x), abs(y))
    return 0.0 if scale == 0 else abs(x - y) / scale


def same_value(again: object, stored: object) -> bool:
    """Whether a recomputed value is the stored one, compared in the stored value's type: the
    interpreter may give a Python int where the run stored a float (``terminal=0`` on a Money
    variable), which is the same value. Floats are compared bit for bit, so -0.0 is not 0.0,
    and two NaNs agree."""
    if isinstance(stored, (bool, np.bool_)):
        # before a Bool variable's domain the interpreter gives 0.0 where the run holds False
        numbers = (bool, np.bool_, int, float, np.integer, np.floating)
        return isinstance(again, numbers) and float(again) == float(bool(stored))
    if not isinstance(again, (int, float, np.integer, np.floating)) or isinstance(again, bool):
        return False
    if isinstance(stored, (int, np.integer)):
        return float(again) == float(int(stored)) and float(again).is_integer()
    if isinstance(stored, (float, np.floating)):
        x, y = float(again), float(stored)
        if math.isnan(x) and math.isnan(y):
            return True
        return struct.pack("<d", x) == struct.pack("<d", y)
    return False


def explanation(
    result: RunResult, p: int, name: str, t: int | None, scenario: int
) -> dict[str, object]:
    """:func:`halley.explain` for the policy at index ``p``: the value, its unit, where the
    formula is defined, its source and every input with its value and a target to open; beside
    it the boundary value or domain the month falls under, and the value recomputed from the
    stored inputs by the interpreter, independently of the generated code, with whether the two
    agree bit for bit."""
    graph = result.graph
    if graph is None or result.inputs is None:
        raise HalleyError("explain needs a run made in this session (load_run has no model)")
    if name not in graph.variables:
        raise KeyError(name)
    _policies(result, [p])
    _scenarios(result, [scenario])
    v = graph.variables[name]
    month = t if v.has_time else None
    if v.has_time and (month is None or not 0 <= month <= result.H):
        raise ValueError(f"{name} is monthly: give t in 0..{result.H}")
    policy_id = result.ids[p]
    copies = int(np.count_nonzero(result.ids == policy_id))
    if copies > 1:
        raise HalleyError(
            f"policy id {policy_id!r} appears {copies} times in this run, and explain finds a "
            "policy by its id, so it would explain the first of them; give the model points "
            "unique ids (Field(..., unique=True) makes the loader insist on it)"
        )
    try:
        e = halley.explain(result, policy_id, name, month, scenario)
    except IndexError as exc:
        # explain does not check a scenario input's month against the file's rows; at a
        # boundary month a formula can read one the file does not hold
        raise HalleyError(
            f"explain cannot evaluate the inputs of {name} at t = {month}: one of them reads a "
            f"month its input does not hold ({exc}); explain another month"
        ) from exc
    targets = _targets(graph, v, month)
    boundary: dict[str, object] | None = None
    first, last_spec = v.formula_range
    last = resolve_month(last_spec, result.H)
    if v.has_time and month is not None:
        if month == 0 and v.init is not None:
            boundary = {
                "kind": "init",
                "expr": ir.render(v.init),
                "note": "At t = 0 the variable takes its init value; its formula applies from "
                "t = 1, so the inputs below are what the formula would read a month before the "
                "projection starts.",
            }
        elif month == result.H and v.terminal is not None:
            boundary = {
                "kind": "terminal",
                "expr": ir.render(v.terminal),
                "note": f"At t = {result.H}, the horizon, the variable takes its terminal value; "
                f"its formula applies up to t = {last}.",
            }
        elif month < first:
            boundary = {
                "kind": "before",
                "expr": None,
                "note": f"The formula applies from t = {first}; before that the value is 0.",
            }
    unused = boundary is not None
    inputs = []
    for text, value in e.inputs:
        failed = isinstance(value, str) and value.startswith("<") and value.endswith(">")
        inputs.append(
            {
                "text": text,
                "value": value,
                "error": failed,
                # at a boundary month the value is not the formula's, so neither are its inputs
                "unused": unused,
                "target": targets.get(text),
            }
        )
    try:
        again: object = recompute(
            result, policy_id, name, month if month is not None else 0, scenario
        )
        check: dict[str, object] = {
            "value": again,
            "same": same_value(again, e.value),
            "relative": _relative(again, e.value),
            # the interpreter follows the NumPy backend; another backend agrees within 1e-9
            "backend": result.manifest.get("backend", {}).get("name", "numpy"),
        }
    except (HalleyError, ArithmeticError, ValueError, IndexError, KeyError) as exc:
        check = {"value": None, "same": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "variable": name,
        "policy": policy_id,
        "p": p,
        "t": month,
        "scenario": scenario,
        "lane": e.lane,
        "value": e.value,
        "unit": e.unit,
        "loc": loc_json(e.loc),
        "source": e.source,
        "doc": v.doc,
        "formula": ir.render(v.expr),
        "role": v.role,
        "timing": v.timing,
        "direction": v.direction,
        "start": v.start,
        "time": v.has_time,
        "formula_range": [first, last] if v.has_time else None,
        "boundary": boundary,
        "inputs": inputs,
        "recomputed": check,
        "text": str(e),
    }


# ------------------------------------------------------------------------------ compare


def _digest(value: object) -> str:
    """A short hash of a manifest entry, as the run id's identity holds it."""
    text = json.dumps(plain(value), sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _identity(manifest: Mapping[str, Any], run_dir: Path) -> dict[str, object]:
    """What a run's id covers, entry by entry, so that a comparison can say which of them two
    runs differ in: run.py's identity of a projection, or of a stage run made by
    ``run_aggregation`` (its groups file, its membership, the previous run and the
    disclosures, beside the series, scenario and table files)."""
    inputs = manifest.get("inputs", {})
    stage = "groups" in inputs and "membership" in inputs
    out: dict[str, object] = {"model": manifest.get("model", {}).get("name")}
    out["IR hash"] = manifest.get("ir_sha256")
    backend = manifest.get("backend", {})
    out["backend"] = ", ".join(
        f"{k} {backend[k]}" for k in ("name", *IDENTITY_VERSIONS) if k in backend
    )
    settings = manifest.get("settings", {})
    for key, value in sorted(settings.items()):
        if key not in NOT_IN_IDENTITY:
            out[f"setting {key}"] = plain(value)
    out["horizon"] = manifest.get("dimensions", {}).get("horizon")
    if stage:
        out["groups file"] = inputs["groups"].get("sha256")
        try:
            members = read_copy(run_dir, inputs["membership"], "membership")
            out["membership"] = _membership_sha256(members)
        except HalleyError as exc:
            out["membership"] = f"unreadable: {exc}"
        previous = manifest.get("previous")
        out["previous run"] = (
            None
            if not isinstance(previous, Mapping)
            else f"{previous.get('run_id')} (manifest {str(previous.get('manifest_sha256'))[:16]})"
        )
        disclosed = manifest.get("disclosures") or {}
        if isinstance(disclosed, Mapping):
            disclosed = {k: v for k, v in disclosed.items() if k != "para_120_curve_verified"}
        out["disclosures"] = _digest(disclosed)
    elif "model_points" in inputs:
        out["model points"] = inputs["model_points"].get("sha256")
    for kind, label in (("tables", "table"), ("scenarios", "scenario input")):
        for name, entry in sorted(inputs.get(kind, {}).items()):
            out[f"{label} {name}"] = entry.get("sha256")
    for name, entry in sorted(inputs.get("series", {}).items()):
        if stage:
            out[f"series {name}"] = entry.get("values_sha256")
        else:
            # a stage run by run() covers everything the entry records but its copy's bytes, its
            # source runs' directories and its pathwise record
            ignored = ("source_runs", "sha256", "pathwise")
            out[f"series {name}"] = _digest({k: v for k, v in entry.items() if k not in ignored})
    return out


def compare(a: RunResult, b: RunResult) -> dict[str, object]:
    """Two runs side by side: what their run ids cover, entry by entry, and, when they hold the
    same policies over the same months and scenarios, :func:`halley.diff`'s largest difference
    for every variable both hold, largest first."""
    ia, ib = _identity(a.manifest, Path(a.run_dir)), _identity(b.manifest, Path(b.run_dir))
    identity = [
        {"what": key, "a": ia.get(key), "b": ib.get(key), "same": ia.get(key) == ib.get(key)}
        for key in list(dict.fromkeys([*ia, *ib]))
    ]
    reason: str | None = None
    if a.n_policies != b.n_policies or not np.array_equal(a.ids, b.ids):
        reason = "the two runs do not hold the same policies in the same order"
    elif a.H != b.H:
        reason = f"the runs have different horizons ({a.H} and {b.H} months)"
    elif a.n_scenarios != b.n_scenarios:
        reason = f"the runs have different scenario counts ({a.n_scenarios} and {b.n_scenarios})"
    rows: list[dict[str, object]] = []
    if reason is None:
        common = sorted(set(a.arrays) & set(b.arrays))
        for name, d in halley.diff(a, b, common).items():
            scale = max(abs(d.a), abs(d.b))
            relative = d.max_abs / scale if scale > 0 else 0.0
            rows.append(
                {
                    "variable": name,
                    "max_abs": d.max_abs,
                    "relative": relative,
                    "policy": d.policy_id,
                    "p": _place(a, b, name, d.policy_id),
                    "t": d.t,
                    "scenario": d.scenario,
                    "a": d.a,
                    "b": d.b,
                    "time": bool(a.has_time.get(name, False)),
                    "unit": _unit(a, name),
                }
            )
        rows.sort(key=lambda r: (-_order(r["max_abs"]), str(r["variable"])))
    return {
        "a": {"dir": str(a.run_dir), "name": a.run_dir.name, "run_id": a.run_id},
        "b": {"dir": str(b.run_dir), "name": b.run_dir.name, "run_id": b.run_id},
        "same_run_id": a.run_id == b.run_id,
        "scenarios": a.n_scenarios,
        "identity": identity,
        "comparable": reason is None,
        "reason": reason,
        "variables": rows,
        "only_a": sorted(set(a.arrays) - set(b.arrays)),
        "only_b": sorted(set(b.arrays) - set(a.arrays)),
    }


def _place(a: RunResult, b: RunResult, name: str, policy_id: object) -> int:
    """The index of the policy where :func:`halley.diff` found a variable's largest difference,
    found as it finds it: diff gives the policy's id, which a run whose ids repeat holds at
    several places."""
    x = np.asarray(a.var(name), dtype=np.float64)
    y = np.asarray(b.var(name), dtype=np.float64)
    _t, p, _s = np.unravel_index(int(np.argmax(np.abs(x - y))), x.shape)
    if a.ids[int(p)] == policy_id:
        return int(p)
    return int(np.flatnonzero(a.ids == policy_id)[0])


def _unit(result: RunResult, name: str) -> str | None:
    """A variable's unit, from the manifest of a run that stored it or the model of a run made in
    this session."""
    info = result.manifest.get("outputs", {}).get("variables", {}).get(name)
    if info is not None and "unit" in info:
        return str(info["unit"])
    if result.graph is not None and name in result.graph.variables:
        return result.graph.variables[name].declared_unit.name
    return None


def _order(value: object) -> float:
    """A difference's place in the ranking: NaN first, as the worst there is."""
    number = float(value) if isinstance(value, (int, float)) else 0.0
    return math.inf if math.isnan(number) else number


def same_outputs(a: RunResult, b: RunResult) -> dict[str, object]:
    """Whether two runs hold the same values bit for bit, variable by variable over the
    variables both store, and whether their output files are the same bytes."""
    names = sorted(set(a.arrays) & set(b.arrays))
    differs: list[str] = []
    for name in names:
        x, y = np.asarray(a.var(name)), np.asarray(b.var(name))
        if x.shape != y.shape or x.dtype != y.dtype:
            differs.append(name)
        elif x.dtype.kind == "f":
            # bit patterns, so that NaN agrees with NaN and -0.0 differs from 0.0
            bits = np.dtype(f"u{x.itemsize}")
            if not np.array_equal(x.view(bits), y.view(bits)):
                differs.append(name)
        elif not np.array_equal(x, y):
            differs.append(name)
    oa, ob = a.manifest.get("outputs", {}), b.manifest.get("outputs", {})
    files = all(oa.get(k) == ob.get(k) for k in ("monthly_sha256", "scalars_sha256"))
    return {"compared": len(names), "differ": differs, "bit_for_bit": not differs, "files": files}


def text_of(path: Path, limit: int = 5_000_000) -> str:
    raw = path.read_bytes()[:limit]
    return raw.decode("utf-8", errors="replace")


def platform_info() -> dict[str, object]:
    return {"python": sys.version.split()[0], "platform": sys.platform}
