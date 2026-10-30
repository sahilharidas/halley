# pyright: basic
"""The GUI's application: what each route of the API does.

:class:`App` holds a session: the catalogue of models, the builds made in it, the runs made in it
(a run's result keeps its model and inputs, so ``explain`` works on it), a few runs read back
from their directories, and the queue of jobs. A route turns a request into calls of Halley's
public API and answers with what :mod:`halley.gui.describe` makes of the result. Builds and runs
are jobs (:mod:`halley.gui.jobs`); everything else answers at once.

A run the GUI makes is the run the same call makes in Python, with the same run id: the inputs
are passed to :func:`halley.run` as files (a CSV file is read into an Arrow table first, as the
tutorial reads one), the settings as :class:`~halley.RunSettings`. A rerun reads the inputs from
the run directory's own copies, verified against their hashes, and takes the settings the
manifest records, so a rerun that is the same computation has the same run id (D21).

A route that a page's address can trigger without a click (every GET) reads only run
directories the session knows: those under the runs folder, those the session made, and those
the user opened with **Open a run directory**, a POST. So an address another site sends the
GUI's tab to cannot make the server open a path of that site's choosing.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import importlib.util
import itertools
import json
import linecache
import os
import re
import sys
import threading
import time
import warnings
from collections import OrderedDict
from collections.abc import Callable, Generator, Mapping
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.csv as pa_csv

import halley
from halley import BuildWarning, LoadError, RunResult, RunSettings, build, ir, load_run, run
from halley.aggregation import replay_aggregation
from halley.codegen import BACKENDS, CompiledModel
from halley.errors import HalleyError
from halley.gui import describe, examples
from halley.gui.catalog import (
    MAX_RERUN_TRIES,
    Catalog,
    ModelEntry,
    ModelFileError,
    carried_statics,
    default_valuation_date,
    describe_class,
    exception_name,
    exception_text,
    form_texts,
    format_quietly,
    plain,
    rerun_misses,
    rerun_tries,
    settings_texts,
    statics_of,
    typed_statics,
    unsorted_dict,
)
from halley.gui.jobs import Jobs
from halley.gui.server import ApiError, Request, Route
from halley.runtime.loader import load_inputs, read_copy, read_series_copy, table_columns
from halley.runtime.run import read_manifest, verify_run
from halley.runtime.workers import batch_ranges
from halley.tracing.tracer import trace

RERUNS = "reruns"
"""The folder under the runs folder that holds reruns and replays. A replay of an aggregation run
has the run's id, and a stage finds a previous run by its id among the run directories directly
under its runs root, so two of them there would be ambiguous."""

NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}")
"""A run directory's name: one folder under the runs folder, no path."""

RESERVED_NAMES = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"{d}{n}" for d in ("com", "lpt") for n in range(10)),
}
"""Names Windows keeps for devices, with or without an extension."""

MAX_BUILDS = 40
MAX_SESSION_RUNS = 8
MAX_LOADED = 6
MAX_RESULT_BYTES = 1_500_000_000
"""The memory the runs the session holds may take, made in it and read back together: a run made
in the session keeps every computed variable and its inputs, so beyond this the least recently
used are let go (they stay on disk, and one made in the session can be rerun). The most recent
is always kept."""


@dataclass(slots=True)
class Build:
    """A model built in this session, with the build warnings it gave."""

    id: str
    entry: ModelEntry
    compiled: CompiledModel
    warnings: list[str]
    seconds: float
    built_at: float

    def summary(self, current: ModelEntry | None) -> dict[str, object]:
        values = self.compiled.graph.statics
        return {
            "id": self.id,
            "model": self.entry.id,
            "name": self.entry.name,
            "backend": self.compiled.backend,
            "ir_sha256": self.compiled.ir_hash,
            "statics": plain(values),
            # each setting written from the value itself, so that the page tells 1 from 1.0 and
            # shows an enum's member by its name; "form" is what the build form shows for the
            # class as it is now, which a build of the file as it was may differ from
            "statics_text": settings_texts(self.entry.cls, values),
            "form": form_texts(
                self.entry.cls, values, self.entry.cls if current is None else current.cls
            ),
            "warnings": list(self.warnings),
            "seconds": round(self.seconds, 3),
            "built_at": self.built_at,
            "stage": self.compiled.model.series_declarations() is not None,
            # a build of a class the file has since replaced: built from the file as it was
            "stale": current is None or current.cls is not self.entry.cls,
        }


@dataclass(frozen=True, slots=True)
class Sources:
    """The inputs a run or a check reads: an example's, or files by declared name."""

    example: str | None
    model_points: Path | None
    tables: dict[str, Path]
    scenarios: dict[str, Path]

    def materialise(self) -> tuple[Any, dict[str, Any], dict[str, Any]]:
        if self.example is not None:
            found = examples.example(self.example)
            if found is None:  # checked when the request came in
                raise LoadError(f"there is no example {self.example!r}")
            return found.make()
        assert self.model_points is not None
        return (
            _read(self.model_points),
            {name: _read(path) for name, path in self.tables.items()},
            {name: _read(path) for name, path in self.scenarios.items()},
        )


def _read(path: Path) -> Any:
    """A Parquet input as its path, which the run reads and hashes; a CSV file as the Arrow
    table pyarrow's CSV reader makes of it, which the run copies as Parquet."""
    if path.suffix.lower() != ".csv":
        return path
    try:
        return pa_csv.read_csv(path)
    except (pa.ArrowInvalid, OSError) as exc:
        raise LoadError(f"{path}: the CSV file cannot be read: {exc}") from exc


def _guarded(fn: Callable[[Request], object]) -> Route:
    """A route whose engine errors answer 422 with the error described, whose bad requests
    answer 400, whose unknown names answer 404, and whose faults answer 500 with their
    traceback."""

    def route(request: Request) -> object:
        try:
            return fn(request)
        except ApiError:
            raise
        except (HalleyError, ModelFileError) as exc:
            detail = describe.describe_error(exc)
            raise ApiError(HTTPStatus.UNPROCESSABLE_ENTITY, str(detail["message"]), detail) from exc
        except KeyError as exc:
            name = exc.args[0] if exc.args else ""
            raise ApiError(HTTPStatus.NOT_FOUND, f"there is no {name!r} here") from exc
        except ValueError as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
        except Exception as exc:  # a fault in the GUI or the engine: the page shows all of it
            # written as a model file's error is, since code of the user's may trip Python's
            # own writing (see format_quietly)
            kind = exception_name(exc)
            raise ApiError(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                f"{kind}: {exception_text(exc)}",
                {"type": kind, "traceback": format_quietly(exc)},
            ) from exc

    return route


def _mapping(value: object, what: str) -> Mapping[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ApiError(HTTPStatus.BAD_REQUEST, f"{what} is a JSON object")
    return value


def _text(value: object, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ApiError(HTTPStatus.BAD_REQUEST, f"the request gives no {what}")
    return value.strip()


def _whole(value: object, what: str, minimum: int) -> int | None:
    """An optional whole number of at least ``minimum`` from a form: None when it is blank."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ApiError(HTTPStatus.BAD_REQUEST, f"{what} is a whole number, not {value!r}")
    whole_float = isinstance(value, float) and value.is_integer()
    digits = isinstance(value, str) and re.fullmatch(r"\s*-?\d+\s*", value) is not None
    if not (isinstance(value, int) or whole_float or digits):
        raise ApiError(HTTPStatus.BAD_REQUEST, f"{what} is a whole number, not {value!r}")
    number = int(value)  # pyright: ignore[reportArgumentType]
    if number < minimum:
        raise ApiError(HTTPStatus.BAD_REQUEST, f"{what} is at least {minimum}, not {number}")
    return number


def _file(value: object, what: str) -> Path:
    """An input file named in a form: a path, quoted or not (Windows' Copy as path quotes it),
    relative to the folder the GUI was started in."""
    raw = value.get("path") if isinstance(value, Mapping) else value
    if not isinstance(raw, str) or not raw.strip().strip('"'):
        raise ApiError(HTTPStatus.BAD_REQUEST, f"give a file for {what}")
    path = Path(raw.strip().strip('"')).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.is_file():
        raise ApiError(HTTPStatus.BAD_REQUEST, f"{what}: there is no file {path}")
    return path


def _ints(text: str, what: str) -> list[int]:
    try:
        return [int(part) for part in text.split(",") if part.strip()]
    except ValueError:
        raise ApiError(HTTPStatus.BAD_REQUEST, f"{what} is a list of whole numbers") from None


def _sha256(path: Path | None) -> str | None:
    try:
        return None if path is None else hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _cell(value: object) -> object:
    """An input table's cell as JSON holds it; a type JSON has no form for as its text."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    return str(value)


def _replaced(cls: type) -> bool:
    """Whether the class's module now holds another class under its name, as after a reload of
    the model files: a run's worker processes import the class by its module and name, and
    pickle refuses this one. A class its name does not reach (one a function made) is left to
    ``run()``, which says what is wrong with it, and runs it in this process when there is one
    batch, as Python does."""
    found: object = sys.modules.get(cls.__module__)
    for part in cls.__qualname__.split("."):
        try:
            found = getattr(found, part, None)
        except BaseException:  # a module's own __getattr__, the user's code, which raised
            return False
        if found is None:
            return False
    return found is not cls


_ENGINE = threading.Lock()
"""The engine of every session in this process: sys.modules is the process's, so a reload in one
session replaces the modules another's job uses."""


def _rerun_warnings(cls: type, statics: Mapping[str, Any]) -> list[str]:
    """A build's warnings for the settings a rerun from its run's directory would not give back
    (see :func:`rerun_misses`), each with why, and what would rerun. A setting whose reading
    back runs code of the user's that raises (an enum's own ``_missing_``) is warned of, never
    failing the build, which Python's own ``build()`` made."""
    try:
        missed = rerun_misses(cls, statics)  # pyright: ignore[reportArgumentType]
    except BaseException as exc:
        return [
            "whether a run of this build can be rerun from its directory cannot be told: reading "
            f"its build-time settings back as a run's manifest records them raised "
            f"{exception_name(exc)}: {exception_text(exc)}"
        ]
    found: list[str] = []
    if missed.beyond:
        found.append(
            f"build-time settings {', '.join(missed.beyond)} are dicts given sorted, which a "
            "run's manifest records as their defaults are, mixed with others left at their "
            f"defaults: a rerun from a run's directory tries {MAX_RERUN_TRIES} mixes of such "
            "dicts at their defaults and sorted, which do not reach this one, so a run of this "
            "build cannot be rerun from its directory; give every such dict sorted, or every "
            "one as its default"
        )
    for name in missed.names:
        value = statics[name]
        where = unsorted_dict(value)
        if name in missed.alike:
            found.append(
                f"build-time setting {name} holds an enum's member whose class's own __repr__ "
                "writes it as it writes another member, and a run's manifest records the member "
                "as its repr, so a rerun from a run's directory gives back the other member: a "
                "run of this build cannot be rerun from its directory unless the member makes no "
                "difference to the model"
            )
            continue
        if where is None:
            found.append(
                f"build-time setting {name} does not read back from a run's manifest as the "
                "value built, since the manifest records it as JSON (a tuple as a list, a NumPy "
                "number as a plain one, any other object as its repr), so a run of this build "
                "cannot be rerun from its directory"
            )
            continue
        at, order = where
        what = "is a dict" if at == "" else f"holds a dict, at {at},"
        shown = ", ".join(order[:8] + (["..."] if len(order) > 8 else []))
        found.append(
            f"build-time setting {name} {what} whose keys are not in sorted order as a run's "
            f"manifest sorts them, as text ({shown}), and the setting is not its class's own "
            "default: the manifest records the keys in that order while the IR keeps the order "
            "given, so a run of this build cannot be rerun from its directory; give the keys in "
            "that order"
        )
    return found


def _lexical(path: Path) -> str:
    """A path made absolute and normalised by its text alone, without touching the file system,
    in the case the system compares paths in."""
    return os.path.normcase(os.path.abspath(path))


def _size(result: RunResult) -> int:
    """The bytes a run's arrays hold, and its model points' columns when it keeps its inputs."""
    total = sum(int(a.nbytes) for a in result.arrays.values())
    if result.inputs is not None:
        total += sum(int(v.nbytes) for v in result.inputs.model_points.columns.values())
    return total


class App:
    """A session of the GUI: its models, builds, runs and jobs, and the API's routes."""

    def __init__(self, runs_root: str | Path) -> None:
        self.runs_root = Path(runs_root).expanduser().resolve()
        self.catalog = Catalog()
        self.jobs = Jobs()
        self._lock = threading.RLock()
        self._builds: OrderedDict[str, Build] = OrderedDict()
        self._build_ids = itertools.count(1)
        self._session: OrderedDict[str, RunResult] = OrderedDict()
        self._loaded: OrderedDict[str, RunResult] = OrderedDict()
        self._used: OrderedDict[str, None] = OrderedDict()
        """Every run held, made or read back, least recently used first."""
        self._points: dict[str, tuple[str | None, list[str], dict[str, Any]]] = {}
        self._opened: list[str] = []
        self._reserved: set[str] = set()
        self._let_go: dict[str, str | None] = {}
        self._engine = _ENGINE
        """Held by a job while it builds, checks or runs, and by a reload of the model files: a
        reload replaces the modules a job's model comes from, which its trace and its worker
        processes import by name, so the two never overlap."""
        """Runs the session made and then let go of to save memory: by directory, the sha256 of
        the run's manifest, which tells that run from another written there since."""

    # ------------------------------------------------------------------------- the session

    def load_model_file(self, path: str | Path) -> None:
        self.catalog.load_file(path)

    def close(self) -> None:
        self.jobs.close()

    def routes(self) -> dict[tuple[str, str], Route]:
        table: dict[tuple[str, str], Callable[[Request], object]] = {
            ("GET", "/api/session"): self.session_info,
            ("GET", "/api/models"): self.models,
            ("POST", "/api/models/load"): self.load_models,
            ("POST", "/api/models/reload"): self.reload_models,
            ("GET", "/api/model"): self.model,
            ("POST", "/api/build"): self.start_build,
            ("GET", "/api/build"): self.build_info,
            ("GET", "/api/build/source"): self.build_source,
            ("POST", "/api/inputs/check"): self.start_check,
            ("POST", "/api/run"): self.start_run,
            ("GET", "/api/jobs"): self.job_list,
            ("GET", "/api/job"): self.job,
            ("POST", "/api/job/cancel"): self.cancel_job,
            ("GET", "/api/runs"): self.runs,
            ("POST", "/api/runs/open"): self.open_run,
            ("GET", "/api/run"): self.run_info,
            ("GET", "/api/run/find"): self.find_policy,
            ("GET", "/api/run/lanes"): self.lanes,
            ("GET", "/api/run/totals"): self.totals,
            ("GET", "/api/run/policies"): self.policies,
            ("GET", "/api/run/explain"): self.explain,
            ("GET", "/api/run/file"): self.run_file,
            ("GET", "/api/run/input"): self.run_input,
            ("POST", "/api/run/rerun"): self.start_rerun,
            ("GET", "/api/compare"): self.compare,
        }
        return {key: _guarded(fn) for key, fn in table.items()}

    def session_info(self, request: Request) -> object:
        numba = importlib.util.find_spec("numba") is not None
        return {
            "halley": halley.__version__,
            **describe.platform_info(),
            "runs_root": str(self.runs_root),
            "cwd": str(Path.cwd()),
            "backends": [
                {
                    "name": name,
                    "artefact": backend.artefact,
                    "available": name != "numba" or numba,
                    "missing": None
                    if name != "numba" or numba
                    else "the Numba backend needs numba, which is not installed: pip install "
                    "halley[numba]",
                }
                for name, backend in BACKENDS.items()
            ],
        }

    # -------------------------------------------------------------------------------- models

    def _entry(self, model_id: object) -> ModelEntry:
        entry = self.catalog.get(_text(model_id, "model"))
        if entry is None:
            raise ApiError(HTTPStatus.NOT_FOUND, f"there is no model {model_id!r}")
        return entry

    def _builds_of(self, entry: ModelEntry) -> list[Build]:
        with self._lock:
            builds = [b for b in self._builds.values() if b.entry.id == entry.id]
        return sorted(builds, key=lambda b: b.built_at, reverse=True)

    def models(self, request: Request) -> object:
        out = []
        for entry in self.catalog.entries():
            out.append(
                {
                    **entry.summary(),
                    "builds": [b.id for b in self._builds_of(entry)],
                    "examples": len(examples.examples_for(entry.id)),
                }
            )
        changed = self.catalog.changed()
        return {
            "models": out,
            "files": [f.as_dict(changed) for f in self.catalog.files()],
            "changed": changed,
        }

    @contextlib.contextmanager
    def _reloading(self) -> Generator[None]:
        """The engine, for a reload of the model files: refused, not waited for, while a job
        holds it, since a run may take hours."""
        if not self._engine.acquire(blocking=False):
            busy = self.jobs.running()
            # with no job running, the engine's holder is another reload, which takes seconds
            if busy is not None or not self._engine.acquire(timeout=30):
                what = busy.label if busy else "A build, a run or another reload of the files"
                raise ApiError(
                    HTTPStatus.CONFLICT,
                    f"{what} is running on the models as they are loaded now; reload the files "
                    "when it has finished",
                )
        try:
            yield
        finally:
            self._engine.release()

    def _held(self, entry: ModelEntry | None, work: Callable[[], object]) -> Callable[[], object]:
        """A job's work, run holding the engine, its imports of the user's modules (an input
        rule's, an invariant's) recorded as helpers of the model's file, as a build's are."""

        def held() -> object:
            with self._engine, self.catalog.watch_imports() as imported:
                try:
                    return work()
                finally:
                    if entry is not None:
                        self.catalog.note_imports(entry, imported)

        return held

    def load_models(self, request: Request) -> object:
        path = _text(request.body.get("path"), "file").strip('"')
        dropped: list[str] = []
        known = self.catalog.known(path)
        if known is not None:  # a loaded file again: a reload of every file
            with self._reloading():
                _files, dropped = self.catalog.reload()
            again = self.catalog.known(path)
            if again is None:  # the file asked for is itself gone, and was let go
                raise ApiError(
                    HTTPStatus.NOT_FOUND,
                    f"{known.path} is no longer there, so the GUI let it go with its models; "
                    "load it again from where it is now",
                    {"dropped": dropped, "file": str(known.path)},
                )
            loaded = again
        else:
            loaded = self.catalog.load_file(path)
        models = [self.catalog.get(i) for i in loaded.models]
        return {
            "file": loaded.as_dict(self.catalog.changed()),
            "models": [m.summary() for m in models if m is not None],
            "dropped": dropped,
        }

    def reload_models(self, request: Request) -> object:
        with self._reloading():
            files, dropped = self.catalog.reload()
        changed = self.catalog.changed()
        return {"files": [f.as_dict(changed) for f in files], "dropped": dropped}

    def model(self, request: Request) -> object:
        entry = self._entry(request.arg("id"))
        loaded = next((f for f in self.catalog.files() if entry.id in f.models), None)
        try:
            described = describe_class(entry)
        except Exception as exc:
            # a declaration the class gets wrong (a one-element tuple without its comma, say):
            # the page still shows the class, and its build gives the tracer's own message
            try:
                statics = [p.as_dict() for p in statics_of(entry.cls)]
            except Exception:
                statics = []
            described = {
                **entry.summary(),
                "statics": statics,
                "schema": None,
                "tables": [],
                "scenarios": [],
                "input_rules": [],
                "invariants": [],
                "outputs": None,
                "tolerance_scales": {},
                "valuation_date": default_valuation_date(entry.cls),
                "error": describe.describe_error(exc),
            }
        return {
            **described,
            "examples": [e.as_dict() for e in examples.examples_for(entry.id)],
            "builds": [b.summary(entry) for b in self._builds_of(entry)],
            "loaded_file": None if loaded is None else loaded.as_dict(self.catalog.changed()),
        }

    # -------------------------------------------------------------------------------- builds

    def _build(self, build_id: object) -> Build:
        key = _text(build_id, "build")
        with self._lock:
            found = self._builds.get(key)
        if found is None:
            raise ApiError(HTTPStatus.NOT_FOUND, f"there is no build {key!r} in this session")
        return found

    def build_now(self, entry: ModelEntry, backend: str, statics: Mapping[str, object]) -> Build:
        """Build a model in this thread, capturing its build warnings, and keep the build."""
        return self._keep(self._compile(entry, backend, statics))

    def _compile(self, entry: ModelEntry, backend: str, statics: Mapping[str, object]) -> Build:
        """A build not yet kept in the session. The modules of the user's own the build imports
        (a formula's import, say) become helpers of the model's file."""
        if entry.file is not None:
            linecache.checkcache(str(entry.file))
        started = time.perf_counter()
        with self.catalog.watch_imports() as imported:  # this thread's imports, not a load's
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    compiled = build(entry.cls, backend=backend, **statics)
            finally:
                self.catalog.note_imports(entry, imported)
        seconds = time.perf_counter() - started
        found = [str(w.message) for w in caught if issubclass(w.category, BuildWarning)]
        found.extend(_rerun_warnings(entry.cls, compiled.graph.statics))
        return Build("", entry, compiled, found, seconds, time.time())

    def _keep(self, made: Build) -> Build:
        with self._lock:
            made.id = f"build-{next(self._build_ids)}"
            self._builds[made.id] = made
            while len(self._builds) > MAX_BUILDS:
                self._builds.popitem(last=False)
        return made

    def start_build(self, request: Request) -> object:
        entry = self._entry(request.body.get("model"))
        backend = str(request.body.get("backend") or "numpy")
        if backend not in BACKENDS:
            known = ", ".join(sorted(BACKENDS))
            raise ApiError(
                HTTPStatus.BAD_REQUEST, f"there is no backend {backend!r}; the backends are {known}"
            )
        # the form sends the fields changed from what it showed: the defaults, or the settings
        # of the build it showed, which this build starts from
        base: Mapping[str, object] | None = None
        changed = dict(_mapping(request.body.get("statics"), "statics"))
        shown = request.body.get("from_build")
        if shown:
            started_from = self._build(shown)
            if started_from.entry.id != entry.id:
                raise ApiError(
                    HTTPStatus.BAD_REQUEST,
                    f"{started_from.id} is a build of {started_from.entry.id}, not {entry.id}",
                )
            values = started_from.compiled.graph.statics
            if started_from.entry.cls is entry.cls:
                base = values
            else:
                # a build of the file as it was: its settings carried to the class as it is now,
                # as its form shows them
                base = carried_statics(started_from.entry.cls, values, entry.cls)
        statics = typed_statics(entry.cls, changed, base)
        job = self.jobs.submit(
            "build",
            f"Build {entry.name} ({backend})",
            self._held(None, lambda: {"build": self.build_now(entry, backend, statics).id}),
            describe.describe_error,
        )
        return {"job": job.as_dict()}

    def build_info(self, request: Request) -> object:
        made = self._build(request.arg("id"))
        current = self.catalog.get(made.entry.id)
        return {**made.summary(current), **describe.describe_compiled(made.compiled)}

    def build_source(self, request: Request) -> object:
        made = self._build(request.arg("id"))
        return {
            "source": made.compiled.source,
            "artefact": BACKENDS[made.compiled.backend].artefact,
        }

    # ---------------------------------------------------------------------------------- runs

    def _sources(self, made: Build, raw: object) -> Sources:
        spec = _mapping(raw, "inputs")
        example_id = spec.get("example")
        if example_id:
            found = examples.example(str(example_id))
            if found is None:
                raise ApiError(HTTPStatus.NOT_FOUND, f"there is no example {example_id!r}")
            if found.model != made.entry.id:
                raise ApiError(
                    HTTPStatus.BAD_REQUEST,
                    f"example {found.id} is for {found.model}, not {made.entry.id}",
                )
            missing = found.missing()
            if missing is not None:
                raise ApiError(HTTPStatus.CONFLICT, missing)
            return Sources(found.id, None, {}, {})
        graph = made.compiled.graph
        tables = _mapping(spec.get("tables"), "tables")
        scenarios = _mapping(spec.get("scenarios"), "scenarios")
        for given, declared, what in (
            (tables, graph.tables, "assumption table"),
            (scenarios, graph.scenarios, "scenario input"),
        ):
            extra = sorted(set(given) - set(declared))
            if extra:
                raise ApiError(
                    HTTPStatus.BAD_REQUEST,
                    f"the model declares no {what} {', '.join(extra)}; it declares "
                    f"{', '.join(sorted(declared)) or 'none'}",
                )
        return Sources(
            None,
            _file(spec.get("model_points"), "the model points"),
            {n: _file(tables.get(n), f"table {n}") for n in sorted(graph.tables)},
            {n: _file(scenarios.get(n), f"scenario input {n}") for n in sorted(graph.scenarios)},
        )

    def _settings(self, made: Build, raw: object) -> RunSettings:
        spec = _mapping(raw, "settings")
        text = spec.get("valuation_date")
        if not isinstance(text, str) or not text.strip():
            raise ApiError(HTTPStatus.BAD_REQUEST, "give the valuation date")
        try:
            valuation_date = dt.date.fromisoformat(text.strip())
        except ValueError:
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                f"the valuation date is a date such as 2027-01-01, not {text!r}",
            ) from None
        outputs: tuple[str, ...] | None = None
        asked = spec.get("outputs")
        if asked:
            if not isinstance(asked, list) or not all(isinstance(n, str) for n in asked):
                raise ApiError(HTTPStatus.BAD_REQUEST, "outputs is a list of variable names")
            needed = made.compiled.schedule.needed
            unknown = [n for n in asked if n not in needed]
            if unknown:
                raise ApiError(
                    HTTPStatus.BAD_REQUEST,
                    "a run stores only variables its build computes; "
                    f"{', '.join(unknown)} {'is' if len(unknown) == 1 else 'are'} not among them",
                )
            outputs = tuple(dict.fromkeys(asked))
        label = spec.get("label") or ""
        if not isinstance(label, str):
            raise ApiError(HTTPStatus.BAD_REQUEST, "the label is text")
        return RunSettings(
            valuation_date=valuation_date,
            horizon=_whole(spec.get("horizon"), "the horizon", 1),
            batch_size=_whole(spec.get("batch_size"), "the batch size", 1) or 10_000,
            outputs=outputs,
            label=label,
            workers=_whole(spec.get("workers"), "the number of worker processes", 1) or 1,
        )

    def _new_run_dir(self, name: object) -> Path:
        text = _text(name, "name for the run")
        stem = text.split(".")[0].lower()
        if (
            not NAME.fullmatch(text)
            or text.endswith(".")
            or text.lower() == RERUNS
            or stem in RESERVED_NAMES
        ):
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                "a run's name is up to 100 letters, digits, dots, dashes and underscores, starting "
                "with a letter or a digit and not ending with a dot (and not reruns, nor a name "
                f"Windows keeps for a device, such as nul), not {text!r}",
            )
        path = self.runs_root / text
        with self._lock:
            taken = {_lexical(Path(p)) for p in self._reserved}
            if path.exists() or _lexical(path) in taken:
                raise ApiError(
                    HTTPStatus.CONFLICT, f"{path} exists already; give the run another name"
                )
            self._reserved.add(str(path))
        return path

    def _release(self, path: Path) -> None:
        with self._lock:
            self._reserved.discard(str(path))

    def _projection(self, made: Build) -> None:
        if made.compiled.model.series_declarations() is not None:
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                f"{made.entry.name} is an aggregation stage: it runs on projection run "
                "directories through halley.aggregation.run_aggregation, not on model points; "
                "the GUI reads its runs and replays them",
            )

    def start_check(self, request: Request) -> object:
        made = self._build(request.body.get("build"))
        self._projection(made)
        sources = self._sources(made, request.body.get("inputs"))
        settings = self._settings(made, request.body.get("settings"))

        def work() -> object:
            points, tables, scenarios = sources.materialise()
            model = made.compiled.model
            inputs = load_inputs(
                made.compiled.graph,
                made.compiled.schema,
                model_points=points,
                tables=tables,
                scenarios=scenarios,
                valuation_date=settings.valuation_date,
                horizon=settings.horizon,
                table_contracts={t.name: t for t in model.assumption_tables},
                scenario_contracts={s.name: s for s in model.scenario_inputs},
                input_rules=model.input_rules,
            )
            batches = len(batch_ranges(len(inputs.ids), settings.batch_size))
            return {
                "policies": len(inputs.ids),
                "scenarios": inputs.n_scenarios,
                "horizon": inputs.H,
                "batches": batches,
                "processes": max(1, min(settings.workers, batches)),
            }

        job = self.jobs.submit(
            "check",
            f"Check the inputs of {made.entry.name}",
            self._held(made.entry, work),
            describe.describe_error,
        )
        return {"job": job.as_dict()}

    def start_run(self, request: Request) -> object:
        made = self._build(request.body.get("build"))
        self._projection(made)
        sources = self._sources(made, request.body.get("inputs"))
        settings = self._settings(made, request.body.get("settings"))
        if settings.workers > 1 and _replaced(made.entry.cls):
            raise ApiError(
                HTTPStatus.CONFLICT,
                "the model's file was reloaded after this build, and worker processes import the "
                "class as the file defines it now; build the model again to run with worker "
                "processes",
            )
        run_dir = self._new_run_dir(request.body.get("name"))

        def work() -> object:
            try:
                if settings.workers > 1 and _replaced(made.entry.cls):
                    raise HalleyError(
                        "the model's file was reloaded while this run waited, and worker "
                        "processes import the class as the file defines it now; build the model "
                        "again to run with worker processes"
                    )
                points, tables, scenarios = sources.materialise()
                result = run(
                    made.compiled,
                    model_points=points,
                    tables=tables,
                    scenarios=scenarios,
                    settings=settings,
                    run_dir=run_dir,
                )
            finally:
                self._release(run_dir)
            self._remember(result)
            return {"run": str(result.run_dir), "run_id": result.run_id}

        job = self.jobs.submit(
            "run",
            f"Run {made.entry.name} into {run_dir.name}",
            self._held(made.entry, work),
            describe.describe_error,
            on_cancel=lambda: self._release(run_dir),
        )
        return {"job": job.as_dict()}

    # ---------------------------------------------------------------------------------- jobs

    def job_list(self, request: Request) -> object:
        return {"jobs": [j.as_dict() for j in self.jobs.recent()]}

    def job(self, request: Request) -> object:
        found = self.jobs.get(request.arg("id"))
        if found is None:
            raise ApiError(HTTPStatus.NOT_FOUND, f"there is no job {request.arg('id')!r}")
        return found.as_dict()

    def cancel_job(self, request: Request) -> object:
        return {"cancelled": self.jobs.cancel(_text(request.body.get("id"), "job"))}

    # ------------------------------------------------------------------- the runs held in memory

    def _touch(self, key: str) -> None:
        self._used[key] = None
        self._used.move_to_end(key)

    def _forget(self, key: str) -> None:
        self._session.pop(key, None)
        self._loaded.pop(key, None)
        self._points.pop(key, None)
        self._used.pop(key, None)

    def _trim(self) -> None:
        """Let go of runs, least recently used first, beyond the counts and the bytes the session
        may hold; the most recent stays whatever its size."""
        for held, limit in ((self._session, MAX_SESSION_RUNS), (self._loaded, MAX_LOADED)):
            while len(held) > limit:
                oldest = next((k for k in self._used if k in held), None)
                self._let_go_of(oldest if oldest is not None else next(iter(held)))
        total = sum(_size(r) for r in (*self._session.values(), *self._loaded.values()))
        for key in list(self._used)[:-1]:
            if total <= MAX_RESULT_BYTES:
                break
            held = self._session.get(key) or self._loaded.get(key)
            if held is not None:
                total -= _size(held)
            self._let_go_of(key)

    def _let_go_of(self, key: str) -> None:
        held = self._session.get(key)
        if held is not None:
            self._let_go[key] = held.manifest_sha256
        self._forget(key)

    def _remember(self, result: RunResult) -> None:
        key = str(Path(result.run_dir).resolve())
        with self._lock:
            self._forget(key)  # whatever the directory held before is gone
            self._let_go.pop(key, None)
            self._session[key] = result
            self._touch(key)
            self._trim()

    # ----------------------------------------------------------------------- run directories

    def _known_path(self, raw: object) -> Path:
        """A run directory a GET may read: one under the runs folder, one the session made, or
        one the user opened. The check is made on the path's text before the file system is
        touched, so an address that names a network share does not make the server connect to
        it."""
        path = Path(_text(raw, "run directory").strip('"')).expanduser()
        if not path.is_absolute():
            path = self.runs_root / path
        where = _lexical(path)
        # a runs folder at a drive's or a share's root already ends in a separator
        root = _lexical(self.runs_root).rstrip(os.sep) + os.sep
        with self._lock:
            known = {_lexical(Path(p)) for p in (*self._session, *self._opened)}
        if not (where.startswith(root) or where in known):
            raise ApiError(
                HTTPStatus.NOT_FOUND,
                f"{path} is not a run directory of this session: open it on the Runs page first",
            )
        resolved = path.resolve()
        if not (resolved / "manifest.json").is_file():
            raise ApiError(
                HTTPStatus.NOT_FOUND, f"{resolved} holds no run: it has no manifest.json"
            )
        return resolved

    def _open_path(self, raw: object) -> Path:
        """A run directory the user names: a path, quoted or not, relative to the folder the GUI
        was started in (the other path fields' base) or, failing that, to the runs folder, so a
        bare run name works too."""
        path = Path(_text(raw, "run directory").strip('"')).expanduser()
        places = [path] if path.is_absolute() else [Path.cwd() / path, self.runs_root / path]
        for place in places:
            if (place / "manifest.json").is_file():
                return place.resolve()
        raise ApiError(HTTPStatus.NOT_FOUND, f"{places[0]} holds no run: it has no manifest.json")

    def _result(self, raw: object) -> tuple[RunResult, bool]:
        """A run by its directory: the session's own result when the session made it, else the
        run read back from its directory (and kept for the next request). A result held for a
        directory whose manifest has changed since, because the run was deleted and the
        directory used again, is dropped first."""
        path = self._known_path(raw)
        key = str(path)
        _manifest, sha = read_manifest(path)
        with self._lock:
            held = self._session.get(key) or self._loaded.get(key)
            if held is not None and held.manifest_sha256 != sha:
                self._forget(key)
            if key in self._session:
                self._touch(key)
                return self._session[key], True
            if key in self._loaded:
                self._touch(key)
                return self._loaded[key], False
        try:
            result = load_run(path)
        except HalleyError:
            raise
        except (OSError, KeyError, TypeError, ValueError, AttributeError, pa.ArrowException) as exc:
            # a damaged directory: a missing output column, a corrupt file, unreadable JSON
            raise LoadError(
                f"{path} cannot be read back as a run: {type(exc).__name__}: {exc}"
            ) from exc
        with self._lock:
            self._loaded[key] = result
            self._touch(key)
            self._trim()
        return result, False

    def runs(self, request: Request) -> object:
        found: dict[str, Path] = {}
        for folder in (self.runs_root, self.runs_root / RERUNS):
            if folder.is_dir():
                for child in sorted(folder.iterdir()):
                    if child.is_dir() and (child / "manifest.json").is_file():
                        found.setdefault(str(child.resolve()), child.resolve())
        with self._lock:
            known = [*self._session, *self._opened]
            session = set(self._session)
        for key in known:
            if (Path(key) / "manifest.json").is_file():
                found.setdefault(key, Path(key))
        rows: list[dict[str, object]] = []
        for key, path in found.items():
            row: dict[str, object]
            try:
                manifest, _sha = read_manifest(path)
                file = path / str(manifest.get("invariants", "invariants.json"))
                invariants = json.loads(file.read_text(encoding="utf-8")) if file.is_file() else []
                row = describe.run_summary(path, manifest, invariants)
            except Exception as exc:  # one damaged directory must not hide the others
                row = {"dir": key, "name": path.name, "error": f"{type(exc).__name__}: {exc}"}
            row["in_session"] = key in session
            row["rerun"] = path.parent == self.runs_root / RERUNS
            rows.append(row)
        rows.sort(key=lambda r: str(r.get("created_at") or ""), reverse=True)
        return {"root": str(self.runs_root), "runs": rows}

    def open_run(self, request: Request) -> object:
        path = self._open_path(request.body.get("path"))
        with self._lock:
            if str(path) not in self._opened:
                self._opened.append(str(path))
        manifest, _sha = read_manifest(path)
        return {"dir": str(path), "run_id": manifest.get("run_id")}

    def run_info(self, request: Request) -> object:
        verify = request.query.get("verify") == "1"
        if verify:
            # every file the manifest records a hash for is checked; the run shown stays the
            # session's own when the session made it
            verify_run(self._known_path(request.arg("dir")))
        result, in_session = self._result(request.arg("dir"))
        return {**describe.describe_run(result, in_session), "verified": verify}

    def find_policy(self, request: Request) -> object:
        """The places of a policy id in a run, for a run with more policies than its page lists:
        every index whose id reads as the text given."""
        result, _ = self._result(request.arg("dir"))
        wanted = request.arg("id")
        places = [int(i) for i in np.flatnonzero([str(i) == wanted for i in result.ids.tolist()])]
        return {"id": wanted, "p": places[:100], "count": len(places)}

    def lanes(self, request: Request) -> object:
        result, _ = self._result(request.arg("dir"))
        policies = _ints(request.query.get("p", "0"), "p")
        scenarios = (
            list(range(result.n_scenarios))
            if request.query.get("s") == "all"
            else _ints(request.query.get("s", "0"), "s")
        )
        return describe.lanes(result, request.arg("var"), policies, scenarios)

    def totals(self, request: Request) -> object:
        result, _ = self._result(request.arg("dir"))
        scenarios = (
            list(range(result.n_scenarios))
            if request.query.get("s", "all") == "all"
            else _ints(request.query.get("s", "0"), "s")
        )
        return describe.totals(result, request.arg("var"), scenarios)

    def _model_points(self, result: RunResult) -> tuple[list[str], dict[str, Any]]:
        """The model point fields of a run, each a column in the run's policy order: the loaded
        fields and derived fields of a run made in this session, or the fields of the model
        point file the run directory copied, ordered by id as the loader orders them."""
        key = str(Path(result.run_dir).resolve())
        with self._lock:
            held = self._points.get(key)
            if held is not None and held[0] == result.manifest_sha256:
                return held[1], held[2]
        columns: dict[str, Any] = {}
        id_field = result.manifest.get("outputs", {}).get("id_field", "policy_id")
        if result.inputs is not None:
            columns = dict(result.inputs.model_points.columns)
        else:
            entry = result.manifest.get("inputs", {}).get("model_points")
            if entry is not None:
                raw: dict[str, Any]
                try:
                    raw = dict(
                        table_columns(read_copy(Path(result.run_dir), entry, "model points"))
                    )
                except LoadError:
                    raw = {}
                if id_field in raw:
                    order = np.argsort(raw[id_field], kind="stable")
                    if np.array_equal(raw[id_field][order], result.ids):
                        columns = {name: values[order] for name, values in raw.items()}
        # the id has a column of its own in the table; the rest are the model point's fields
        names = [name for name in columns if name != id_field]
        with self._lock:
            self._points[key] = (result.manifest_sha256, names, columns)
        return names, columns

    def policies(self, request: Request) -> object:
        result, _ = self._result(request.arg("dir"))
        name = request.query.get("var") or None
        t = request.int_arg("t", 0)
        s = request.int_arg("s", 0)
        offset = max(0, request.int_arg("offset", 0))
        limit = max(1, min(request.int_arg("limit", 100), 500))
        fields, columns = self._model_points(result)
        values: Any = None
        time_var = False
        if name is not None:
            if name not in result.arrays:
                raise ApiError(HTTPStatus.NOT_FOUND, f"the run holds no variable {name!r}")
            if not 0 <= s < result.n_scenarios:
                raise ApiError(
                    HTTPStatus.BAD_REQUEST, f"scenario {s} is outside 0..{result.n_scenarios - 1}"
                )
            time_var = bool(result.has_time.get(name, False))
            if time_var and not 0 <= t <= result.H:
                raise ApiError(HTTPStatus.BAD_REQUEST, f"t = {t} is outside 0..{result.H}")
            values = np.asarray(result.var(name)[t if time_var else 0, :, s])
        order = np.arange(result.n_policies)
        sort = request.query.get("sort") or "id"
        if sort == "value" and values is not None:
            keys = np.asarray(values, dtype=np.float64)
            order = np.argsort(np.where(np.isnan(keys), np.inf, keys), kind="stable")
        elif sort in columns:
            order = np.argsort(np.asarray(columns[sort]), kind="stable")
        if request.query.get("order") == "desc":
            order = order[::-1]
        page = order[offset : offset + limit]
        rows = [
            {
                "p": int(i),
                "id": result.ids[i],
                "fields": [columns[f][i] for f in fields],
                "value": None if values is None else values[i],
            }
            for i in page
        ]
        return {
            "total": result.n_policies,
            "offset": offset,
            "limit": limit,
            "fields": fields,
            "rows": rows,
            "variable": name,
            "t": t if time_var else None,
            "scenario": s,
        }

    def explain(self, request: Request) -> object:
        result, _ = self._result(request.arg("dir"))
        if result.graph is None or result.inputs is None:
            with self._lock:
                key = str(Path(result.run_dir).resolve())
                let_go = key in self._let_go and self._let_go[key] == result.manifest_sha256
            raise ApiError(
                HTTPStatus.CONFLICT,
                "this run was made in this session, which then let go of its model and inputs "
                "to make room for later runs (it is on disk as it was); rerun it from its own "
                "inputs and explain the rerun"
                if let_go
                else "explain needs the model and the inputs a run was made with, and a run read "
                "back from its directory holds neither; rerun it from its own inputs in this "
                "session and explain the rerun",
                {"rerun": True},
            )
        raw_t = request.query.get("t", "")
        t = None if raw_t == "" else request.int_arg("t")
        return describe.explanation(
            result, request.int_arg("p", 0), request.arg("var"), t, request.int_arg("s", 0)
        )

    def run_file(self, request: Request) -> object:
        path = self._known_path(request.arg("dir"))
        which = request.arg("name")
        manifest, _sha = read_manifest(path)
        if which == "manifest":
            file = path / "manifest.json"
        elif which == "invariants":
            file = path / str(manifest.get("invariants", "invariants.json"))
        elif which == "generated":
            folder = path / "generated"
            found = sorted(folder.iterdir()) if folder.is_dir() else []
            if not found:
                raise ApiError(HTTPStatus.NOT_FOUND, f"{path} holds no generated code")
            file = found[0]
        else:
            raise ApiError(HTTPStatus.BAD_REQUEST, "name is manifest, invariants or generated")
        if not file.is_file():
            raise ApiError(HTTPStatus.NOT_FOUND, f"there is no file {file}")
        return {"name": which, "path": str(file), "text": describe.text_of(file)}

    def run_input(self, request: Request) -> object:
        path = self._known_path(request.arg("dir"))
        manifest, _sha = read_manifest(path)
        inputs = manifest.get("inputs", {})
        kind = request.arg("kind")
        name = request.query.get("name", "")
        entry: Any
        if kind in ("model_points", "groups", "derived_groups", "membership"):
            entry = inputs.get(kind)
            label = kind.replace("_", " ")
        elif kind in ("table", "scenario", "series"):
            entry = inputs.get(f"{kind}s" if kind != "series" else "series", {}).get(name)
            label = f"{'scenario input' if kind == 'scenario' else kind} {name}"
        else:
            raise ApiError(HTTPStatus.BAD_REQUEST, f"there is no kind of input {kind!r}")
        if entry is None:
            raise ApiError(HTTPStatus.NOT_FOUND, f"the run's manifest records no {label}")
        copy = read_copy(path, entry, label)  # verified against the sha256 the manifest records
        table = copy.table
        limit = max(1, min(request.int_arg("limit", 500), 2000))
        head = table.slice(0, limit)
        names = list(head.column_names)
        rows = [[_cell(row[c]) for c in names] for row in head.to_pylist()]
        return {
            "kind": kind,
            "name": name or label,
            "copy": entry.get("copy"),
            "sha256": copy.sha256,
            "metadata": dict(copy.metadata),
            "columns": [{"name": f.name, "type": str(f.type)} for f in table.schema],
            "rows": rows,
            "total": table.num_rows,
        }

    # -------------------------------------------------------------------------------- reruns

    def _rerun_dir(self, name: str, word: str) -> Path:
        """A new folder under the reruns folder for a rerun of ``name``: ``<name>-rerun``, then
        ``<name>-rerun-2`` and so on, each cut to 100 characters from the name's end, never the
        number's, so every try is another name."""
        folder = self.runs_root / RERUNS
        with self._lock:
            taken = {_lexical(Path(p)) for p in self._reserved}
            for n in range(1, 10_000):
                suffix = f"-{word}" if n == 1 else f"-{word}-{n}"
                candidate = folder / f"{name[: 100 - len(suffix)]}{suffix}"
                if not candidate.exists() and _lexical(candidate) not in taken:
                    self._reserved.add(str(candidate))
                    return candidate
        raise ApiError(
            HTTPStatus.CONFLICT, f"{folder} holds too many reruns of {name}; delete some of them"
        )

    def _build_for(self, manifest: Mapping[str, Any]) -> Build:
        """A build of the model a run's manifest names, with its recorded build-time settings
        and backend: one of this session's when one has the run's IR hash, else a new one."""
        model = manifest.get("model", {})
        entry = self.catalog.entry_for(str(model.get("name")), model.get("source_file"))
        backend = str(manifest.get("backend", {}).get("name", "numpy"))
        wanted = manifest.get("ir_sha256")
        current = self.catalog.get(entry.id)
        for made in self._builds_of(entry):
            same = made.compiled.backend == backend and made.compiled.ir_hash == wanted
            if same and current is not None and current.cls is made.entry.cls:
                return made
        recorded = model.get("statics") or {}
        # each try is traced, which gives its IR hash, and only the run's is compiled: a
        # manifest writes a dict's keys sorted, so a setting recorded as its default is may
        # still be another value (see rerun_tries)
        first: str | None = None
        refused: Exception | None = None
        tries, cut_short = rerun_tries(entry.cls, recorded)
        with self.catalog.watch_imports() as imported:
            try:
                for statics in tries:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")  # the build that is kept says them
                            traced = ir.graph_hash(trace(entry.cls, statics))
                    except Exception as exc:  # a try the class refuses, check_statics say
                        refused = refused or exc
                        continue
                    if traced == wanted:
                        return self._keep(self._compile(entry, backend, statics))
                    first = first or traced
            finally:
                self.catalog.note_imports(entry, imported)
        if first is None and refused is not None:
            raise refused
        cut = (
            f", or the run gave some of the dict settings recorded as their defaults are at "
            f"their defaults and some sorted, a mix the {MAX_RERUN_TRIES} the rerun tries do "
            "not reach"
            if cut_short
            else ""
        )
        raise HalleyError(
            f"{entry.full_name} as it is now, built with the run's recorded build-time settings, "
            f"has IR hash {str(first)[:12]}, but the run was made with {str(wanted)[:12]}: its "
            "formulas, units, contracts or build-time settings have changed since, or a setting "
            "was a value its manifest cannot record exactly (a list given for a tuple, a NumPy "
            "number, a dict whose keys were not in sorted order and that was not its class's own "
            f"default){cut}, so a rerun would not be the same computation"
        )

    def start_rerun(self, request: Request) -> object:
        source = self._open_path(request.body.get("dir"))
        manifest, _sha = read_manifest(source)
        inputs = manifest.get("inputs", {})
        # a stage run made by run_aggregation keeps its groups and membership and is replayed;
        # any other run, a stage run made by run() with its series included, is run again
        replay = "groups" in inputs and "membership" in inputs
        target = self._rerun_dir(source.name, "replay" if replay else "rerun")

        def work() -> object:
            try:
                made = self._build_for(manifest)
                with self.catalog.watch_imports() as imported:
                    try:
                        if replay:
                            result = replay_aggregation(made.compiled, source, run_dir=target)
                        else:
                            result = self._rerun(source, manifest, made, target)
                    finally:
                        self.catalog.note_imports(made.entry, imported)
            finally:
                self._release(target)
            self._remember(result)
            notes: list[str] = []
            recorded = manifest.get("model", {}).get("source_sha256")
            if recorded and _sha256(made.entry.file) not in (None, recorded):
                notes.append(
                    "the model's source file has changed since the run, but not what it computes: "
                    "the rerun's IR hash is the run's"
                )
            original = load_run(source)
            return {
                "run": str(result.run_dir),
                "run_id": result.run_id,
                "same_run_id": result.run_id == manifest.get("run_id"),
                "outputs": describe.same_outputs(original, result),
                "replayed_from": result.manifest.get("replayed_from"),
                "notes": notes,
            }

        verb = "Replay" if replay else "Rerun"
        job = self.jobs.submit(
            "rerun",
            f"{verb} {source.name}",
            self._held(None, work),
            describe.describe_error,
            on_cancel=lambda: self._release(target),
        )
        return {"job": job.as_dict(), "target": str(target)}

    def _rerun(
        self, source: Path, manifest: Mapping[str, Any], made: Build, target: Path
    ) -> RunResult:
        """A run again from its directory alone: its copies of its inputs, each verified against
        the hash its manifest records (a stage's group series too), and its recorded settings."""
        inputs = manifest.get("inputs", {})
        points = read_copy(source, inputs["model_points"], "model points")
        tables = {
            n: read_copy(source, e, f"table {n}") for n, e in inputs.get("tables", {}).items()
        }
        scenarios = {
            n: read_copy(source, e, f"scenario input {n}")
            for n, e in inputs.get("scenarios", {}).items()
        }
        id_field = manifest.get("outputs", {}).get("id_field", "policy_id")
        series = {
            n: read_series_copy(source, n, e, id_field, f"series {n}")
            for n, e in inputs.get("series", {}).items()
        }
        return run(
            made.compiled,
            model_points=points,
            tables=tables,
            scenarios=scenarios,
            settings=RunSettings.from_dict(manifest["settings"]),
            run_dir=target,
            series=series or None,
        )

    # ------------------------------------------------------------------------------- compare

    def compare(self, request: Request) -> object:
        a, _ = self._result(request.arg("a"))
        b, _ = self._result(request.arg("b"))
        return describe.compare(a, b)
