"""The GUI's models: the products and stages that ship with Halley, the models of the files the
user loads, and what each class declares, read off the class without building it.

A model file is imported as a top-level module named after the file (``term.py`` as ``term``),
with its folder on ``sys.path``, the way ``import term`` would import it from there. Three things
depend on that. A model's name in the IR is ``<module>.<class>``, so a stable module name keeps
its IR hash, and with it its run ids, the same across reloads. Its source is the file, so a build
error and ``explain`` point at the user's own lines. And a run with ``RunSettings.workers`` above 1
starts worker processes that import the class by that name from that folder: they inherit
``sys.path`` but not the modules already imported. A file whose name another importable module
already has is refused, since the workers would import the other one.
"""

from __future__ import annotations

import ast
import collections.abc
import contextlib
import dataclasses
import datetime as dt
import enum
import functools
import importlib
import importlib.machinery
import importlib.util
import inspect
import itertools
import json
import linecache
import os
import re
import site
import sys
import sysconfig
import threading
import time
import traceback
import types
import typing
import zipimport
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType

import halley
from halley.aggregation.stage import Aggregation
from halley.authoring.product import Product, Static
from halley.errors import HalleyError
from halley.runtime.run import _json_safe  # pyright: ignore[reportPrivateUsage]

SHIPPED_MODULES = (
    "halley.products.term_assurance",
    "halley.products.lifelib_basic_term",
    "halley.products.annuity",
    "halley.aggregation.csm",
)
"""The modules whose models the GUI lists without being asked: the reference products and the
IFRS 17 stages."""


class ModelFileError(Exception):
    """A model file the GUI cannot import: not a Python file, a clash of module names, or an
    exception while it ran. ``error`` is the exception the file raised, if any, and
    ``traceback`` its traceback, written out when the error is made, before the clean-up that
    follows a failed import (see :func:`format_quietly`)."""

    def __init__(self, message: str, error: BaseException | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.error = error
        self.traceback = None if error is None else format_quietly(error)


def format_quietly(exc: BaseException) -> str:
    """An exception's traceback as Python writes it, without importing anything, and without
    failing: for an ``ImportError`` of a name a module lacks (``from h import LOADING``), Python
    3.12 suggests a near name by importing the module again, which, for one no longer imported
    (taken out after a failed load, say), would bring it back unseen; such an error is written
    without the suggestion (one of the user's own class whose ``name_from`` cannot be set is
    written with it, as Python writes it). A suggestion that runs code of the user's that raises (a
    ``NameError``'s looks the name up on ``self``, whose ``__getattr__`` may raise a
    ``KeyError``, or ``SystemExit``) has every error written without one, and an exception
    Python cannot write at all (one whose ``__notes__`` cannot be read, say, or whose ``name``
    cannot be hushed) is written as its frames, type and text. Whatever the user's code raises
    meanwhile, ``SystemExit`` too, is caught, as Python's own writing of a text catches it; an
    exception's type is told by its class, never by what it says its ``__class__`` is."""
    chain: list[BaseException] = []
    todo: list[object] = [exc]
    while todo:
        current = todo.pop()
        if not _is(current, BaseException) or any(current is c for c in chain):
            continue  # nothing, or what a chain of the user's holds that is no exception
        error = typing.cast(BaseException, current)
        chain.append(error)
        # each link as Python keeps it, past a property of the user's class that raises or lies
        todo += [_own(error, BaseException, "__cause__"), _own(error, BaseException, "__context__")]
        if _is(error, BaseExceptionGroup):
            members = _own(error, BaseExceptionGroup, "exceptions")
            with contextlib.suppress(BaseException):
                todo += list(typing.cast(Sequence[object], members or ()))
    hushed: list[tuple[BaseException, str, object]] = []

    def hush(e: BaseException, attr: str) -> None:
        try:
            value = getattr(e, attr)
        except BaseException:  # a name of the user's that cannot be read
            return
        hushed.append((e, attr, value))  # put back whether or not the setting below holds
        with contextlib.suppress(BaseException):  # one that cannot be set
            setattr(e, attr, None)

    def written() -> str | None:
        try:
            return "".join(traceback.format_exception(exc))
        except BaseException:  # a suggestion ran code of the user's, which raised
            return None

    try:
        for e in chain:
            with contextlib.suppress(BaseException):  # a name of the user's that is no text
                if _is(e, ImportError):
                    error = typing.cast(ImportError, e)
                    if error.name_from is not None and error.name not in sys.modules:
                        hush(e, "name_from")
        text = written()
        if text is not None:
            return text
        if any(_unhushed(e) for e in chain):
            # its suggestion imports a module Python's writing imports once: not twice
            return _fallback(exc)
        for e in chain:
            with contextlib.suppress(BaseException):
                if _is(e, ImportError):
                    if typing.cast(ImportError, e).name_from is not None:
                        hush(e, "name_from")
                elif (
                    _is(e, (NameError, AttributeError))
                    and typing.cast(NameError, e).name is not None
                ):
                    hush(e, "name")
        text = written()
        return text if text is not None else _fallback(exc)
    finally:
        for e, attr, value in reversed(hushed):
            with contextlib.suppress(BaseException):
                setattr(e, attr, value)


def _unhushed(e: BaseException) -> bool:
    """Whether an exception is an ``ImportError`` whose suggestion would still import a module no
    longer imported, its ``name_from`` being one the hush could not set."""
    try:
        if not _is(e, ImportError):
            return False
        error = typing.cast(ImportError, e)
        return error.name_from is not None and error.name not in sys.modules
    except BaseException:
        return False


def _fallback(exc: BaseException) -> str:
    """An exception written as its frames, type and text, as Python's writing of it could not."""
    try:
        frames = "".join(traceback.format_tb(traceback_of(exc)))
    except BaseException:
        frames = ""
    return (
        f"Traceback (most recent call last):\n{frames}"
        f"{exception_name(exc)}: {exception_text(exc)}\n"
    )


def exception_text(exc: BaseException) -> str:
    """An exception's text, as Python's traceback writes one whose ``__str__`` raises, whatever
    it raises; a plain ``str`` even when its ``__str__`` gives a subclass of one, whose own
    methods may raise in turn."""
    try:
        return str.__str__(str(exc))
    except BaseException:
        return "<exception str() failed>"


def exception_name(exc: BaseException) -> str:
    """An exception's type's name, as a plain ``str``, or a stand-in for a class of the user's
    whose name cannot be read."""
    try:
        return str.__str__(type(exc).__name__)
    except BaseException:
        return "<exception>"


def _own(obj: object, cls: type, name: str) -> object:
    """``obj.<name>`` as ``cls`` itself keeps it, past a property of the user's subclass that
    raises or gives something else; None when it cannot be had."""
    try:
        return cls.__dict__[name].__get__(obj, type(obj))
    except BaseException:
        return None


def traceback_of(exc: BaseException) -> types.TracebackType | None:
    """An exception's traceback as Python keeps it, past a ``__traceback__`` of the user's
    class."""
    found = _own(exc, BaseException, "__traceback__")
    return found if isinstance(found, types.TracebackType) else None


def _built_in(module: object) -> bool:
    """Whether a module is one Python itself holds, built in or frozen, which its importers find
    before the path finder: no file of the user's is ever it."""
    if not _is(module, ModuleType):
        return False
    spec = _attr(module, "__spec__")
    origin = _attr(spec, "origin")
    if _is(origin, str) and str.__str__(typing.cast(str, origin)) in ("built-in", "frozen"):
        return True
    loader = _attr(spec, "loader")
    machinery = importlib.machinery
    return loader is machinery.BuiltinImporter or loader is machinery.FrozenImporter


def _name_of(module: object) -> str | None:
    """A module's ``__name__`` as plain text, or None for one that is not text."""
    own = _attr(module, "__name__")
    return str.__str__(typing.cast(str, own)) if _is(own, str) else None


def _made_by_import(module: object, file: Path) -> bool:
    """Whether a module object is as an import of ``file`` makes one: code ran in it (its
    namespace holds ``__builtins__``), with a loader and a spec whose origin is the file, which a
    module object of the file's own making (``types.ModuleType(__name__)``) is not."""
    space = _namespace(module)
    origin = _attr(space.get("__spec__"), "origin")
    if "__builtins__" not in space or space.get("__loader__") is None or not _is(origin, str):
        return False
    try:
        return Path(str.__str__(typing.cast(str, origin))).resolve() == file
    except (OSError, ValueError):
        return False


def _namespace(module: object) -> dict[str, object]:
    """A module's namespace as Python keeps it, past a ``__getattribute__`` of its class (a
    module whose ``__class__`` its code changed); empty for anything else."""
    found = _own(module, ModuleType, "__dict__") if _is(module, ModuleType) else None
    return typing.cast(dict[str, object], found) if isinstance(found, dict) else {}


def _is(obj: object, kind: type | tuple[type, ...]) -> bool:
    """Whether an object is of a kind by its own class, as Python's traceback tells an exception's
    type: ``isinstance`` asks an object of another kind for its ``__class__``, which an object of
    the user's may answer by raising."""
    return issubclass(type(obj), kind)


_MISSING = object()


@dataclass(frozen=True, slots=True)
class StaticParam:
    """A build-time parameter as a form shows it: its name, the type it is declared with
    (``Static[bool]`` is ``"bool"``, and ``hint`` the type itself), its default if the class
    gives one, and whether the type is a tuple (``Static[tuple[float, ...]]`` too), which JSON
    writes as a list."""

    name: str
    kind: str
    default: object = None
    has_default: bool = False
    tuple_like: bool = False
    hint: object = field(default=None, compare=False, repr=False)

    @property
    def input(self) -> str:
        """The field the parameter takes: ``bool`` (a check box), ``int``, ``float``, ``str``,
        ``choice`` (one of an enum's members), ``date``, ``datetime``, or ``json`` for any other
        type."""
        return _input_of(self.hint)

    @property
    def choices(self) -> list[str] | None:
        """An enum's members by name, for a parameter declared with an enum."""
        hint = _unwrap(self.hint)
        if isinstance(hint, type) and issubclass(hint, enum.Enum):
            return [m.name for m in hint]
        return None

    @property
    def text(self) -> str:
        """The default as its form field shows it (:func:`form_text`)."""
        return form_text(self.hint, self.default) if self.has_default else ""

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "input": self.input,
            "choices": self.choices,
            "default": plain(self.default) if self.has_default else None,
            "has_default": self.has_default,
            "text": self.text,
        }


@dataclass(slots=True)
class ModelEntry:
    """A model class the GUI can show and build, by its id ``<module>:<qualname>``."""

    id: str
    cls: type[Product]
    origin: str
    """``"shipped"`` for Halley's own models, ``"file"`` for a model file's."""
    file: Path | None

    @property
    def name(self) -> str:
        return self.cls.__qualname__

    @property
    def full_name(self) -> str:
        """The model's name in the IR and a run's manifest: ``<module>.<qualname>``."""
        return f"{self.cls.__module__}.{self.cls.__qualname__}"

    @property
    def abstract(self) -> bool:
        # the flag is read off the class itself, as build() reads it: it is not inherited
        return vars(self.cls).get("abstract") is True

    @property
    def stage(self) -> bool:
        return issubclass(self.cls, Aggregation)

    def summary(self) -> dict[str, object]:
        line = _class_line(self.cls)
        return {
            "id": self.id,
            "name": self.name,
            "module": self.cls.__module__,
            "full_name": self.full_name,
            "origin": self.origin,
            "file": None if self.file is None else str(self.file),
            "line": line,
            "kind": "aggregation" if self.stage else "projection",
            "abstract": self.abstract,
            # the class's own docstring: inspect.getdoc would give Product's to a class without one
            "doc": _first_paragraph(inspect.cleandoc(vars(self.cls).get("__doc__") or "")),
        }


@dataclass(slots=True)
class LoadedFile:
    """A model file and what importing it gave: its module's name, the ids of its models, the
    modules of the user's own it uses, directly or through one another (a shared assumptions
    module, say, or another model file), and each file's modification time when the module it
    uses was imported from it."""

    path: Path
    module: str
    mtime_ns: int
    models: list[str] = field(default_factory=lambda: [])
    helpers: dict[str, tuple[str, int]] = field(default_factory=lambda: {})
    """By module name, the file and its modification time."""

    @property
    def changed(self) -> bool:
        """Whether the file, or a module of the user's own it imports, has changed since."""
        files = [(str(self.path), self.mtime_ns), *self.helpers.values()]
        for name, mtime_ns in files:
            try:
                if Path(name).stat().st_mtime_ns != mtime_ns:
                    return True
            except OSError:
                return True
        return False

    def as_dict(self, changed: bool | None = None) -> dict[str, object]:
        """The record as the page reads it; ``changed`` is the catalogue's flag for all the
        files (:meth:`Catalog.changed`), this file's own when it is not given."""
        return {
            "path": str(self.path),
            "module": self.module,
            "models": list(self.models),
            "helpers": sorted(self.helpers),
            "changed": self.changed if changed is None else changed,
        }


class _ImportWatch:
    """A finder that finds nothing: first in ``sys.meta_path``, it notes, for the thread that
    asks, each module the import system is about to import for the first time, so that a load
    on one thread and a build on another each learn what they imported, and nothing else. It
    also tells whether another thread noted its imports while this one did something (see
    :meth:`beside`)."""

    def __init__(self) -> None:
        self._local = threading.local()
        self._guard = threading.Lock()
        self._open: dict[int, int] = {}
        """By thread, how many noting contexts it has open."""
        self._opened = 0
        """How many noting contexts every thread has opened so far."""

    def find_spec(
        self, fullname: str, path: Sequence[str] | None, target: ModuleType | None = None
    ) -> ModuleSpec | None:
        noted: list[str] | None = getattr(self._local, "noted", None)
        if noted is not None:
            noted.append(fullname)
            _IMPORTED_AT[fullname] = time.time_ns()
            _NOTED_BY[fullname] = threading.get_ident()
            _FOUND_AT[fullname] = _found(fullname, path)
        return None

    @contextlib.contextmanager
    def noting(self) -> Generator[list[str]]:
        me = threading.get_ident()
        with self._guard:
            self._open[me] = self._open.get(me, 0) + 1
            self._opened += 1
        self._local.opened = getattr(self._local, "opened", 0) + 1
        outer: list[str] | None = getattr(self._local, "noted", None)
        noted: list[str] = []
        self._local.noted = noted
        try:
            yield noted
        finally:
            self._local.noted = outer
            if outer is not None:
                outer.extend(noted)
            with self._guard:
                left = self._open[me] - 1
                if left:
                    self._open[me] = left
                else:
                    del self._open[me]

    def mark(self) -> tuple[int, int, bool]:
        """Where the threads' noting stands, for :meth:`beside`: how many noting contexts every
        thread and this one have opened, and whether another thread has one open."""
        me = threading.get_ident()
        with self._guard:
            return self._opened, getattr(self._local, "opened", 0), any(t != me for t in self._open)

    def beside(self, mark: tuple[int, int, bool]) -> bool:
        """Whether another thread noted its imports at some time since ``mark``: it had a noting
        context open then, has one open now, or opened one meanwhile. A build, a check and a run
        note theirs from start to end, so this tells whether one ran beside what this thread did
        since (a load, say)."""
        opened, mine, was_open = mark
        now_opened, now_mine, now_open = self.mark()
        return was_open or now_open or now_opened - opened > now_mine - mine


_WATCH = _ImportWatch()

_IMPORTED_AT: dict[str, int] = {}
"""By module name, when a noted import of it began (``time.time_ns()``): a file modified since
was not the file the import ran."""

_NOTED_BY: dict[str, int] = {}
"""By module name, the thread whose noted import of it began last (``threading.get_ident()``)."""

_FOUND_AT: dict[str, tuple[str, int] | None] = {}
"""By module name, the file of the user's own the path finder found for its last noted import,
and the file's modification time as that import began."""


def _found(fullname: str, path: Sequence[str] | None) -> tuple[str, int] | None:
    """The file of the user's own that Python's path finder finds for a module about to be
    imported, and its modification time now, as the import begins; None for any other module.
    A finder of the user's ahead of the path finder may import another file, which the record
    tells by its name."""
    try:
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
    except Exception:  # a broken path entry: the import itself says what is wrong
        return None
    origin = None if spec is None or not spec.has_location else spec.origin
    if not origin or is_library_file(origin):
        return None
    try:
        return str(Path(origin).resolve()), Path(origin).stat().st_mtime_ns
    except OSError:
        return None


def _drop_stale_bytecode(file: str, mtime: int | None = None) -> None:
    """Take out a source file's cached bytecode (its ``__pycache__`` file) that Python would take
    for the file as it is now, though the file changed since its import (``mtime``, its time
    then): saved again within the second the cache records, at the same size, which a cache that
    records whole seconds cannot tell from the file it was made from, so the next import would
    run the code as it was. That import writes the cache again, as any import does. With no
    ``mtime`` (a module a failed import brought in, which no record keeps), the cache goes
    whatever the file's time, so that next import runs the file as it is then."""
    try:
        now = os.stat(file)
        if now.st_mtime_ns == mtime:
            return  # as it was imported: the cache is of this file
        cached = importlib.util.cache_from_source(file)
        with open(cached, "rb") as f:
            head = f.read(16)
    except (OSError, ValueError, NotImplementedError):
        return  # no cache, or one Python does not keep for it
    if len(head) < 16 or int.from_bytes(head[4:8], "little") != 0:
        return  # a cache Python checks by the source's hash, which tells any edit
    seconds = int.from_bytes(head[8:12], "little")
    size = int.from_bytes(head[12:16], "little")
    if seconds == int(now.st_mtime) & 0xFFFFFFFF and size == now.st_size & 0xFFFFFFFF:
        with contextlib.suppress(OSError):
            os.remove(cached)


def _drop_failed_caches(names: Iterable[str], file: Path | None) -> None:
    """After a failed load or reload: the cached bytecode of each file of the user's whose import
    it began and that is not imported now (its code raised, and its import took it out), and of
    the model file it was running, which no record keeps; so the next import runs each file as
    it is then (see :func:`_drop_stale_bytecode`)."""
    if file is not None:
        _drop_stale_bytecode(str(file))
    for name in sorted(set(names)):
        found = _FOUND_AT.get(name)
        if name not in sys.modules and found is not None:
            _drop_stale_bytecode(found[0])


def _saved_since_import(name: str, file: str, mtime: int) -> bool:
    """Whether a module's file (as :meth:`Catalog._user_file` gives it) was modified after its
    noted import began: its time then and ``mtime`` differ. For a file the import watch could
    not read as the import began (one a finder of the user's found), whether ``mtime`` is after
    the import's start and not after now, since a file dated ahead of this computer's clock (one
    from an archive made where the clock reads later, say) was not saved while the import ran."""
    found = _FOUND_AT.get(name)
    if found is not None and found[0] == file:
        return found[1] != mtime
    began = _IMPORTED_AT.get(name)
    return began is not None and began < mtime <= time.time_ns()


def _theirs(name: str, since: int) -> bool:
    """Whether another thread's watch saw a module's import begin at ``since`` or after: a build's,
    say, on its own thread while a file loads on this one."""
    me = threading.get_ident()
    return _NOTED_BY.get(name, me) != me and _IMPORTED_AT.get(name, 0) >= since


def _namespace_places(module: object) -> list[str] | None:
    """The folders of a namespace package (a package without a file of its own: a folder without
    an ``__init__.py``); None for any other module, and for one whose folders cannot be read,
    which reads its parent's in sys.modules, where a failed import may have left it without one."""
    if not _is(module, ModuleType) or _attr(module, "__file__") is not None:
        return None
    locations = _attr(_attr(module, "__spec__"), "submodule_search_locations")
    if locations is None:
        return None
    try:
        return [str(p) for p in typing.cast(Sequence[object], locations)]
    except Exception:
        return None


def _user_namespace(module: object) -> bool:
    """Whether a module is a namespace package of the user's, outside the folders of Halley,
    Python and installed packages; one without a parent in sys.modules, whose folders cannot be
    read, is taken as the user's, since only a failed import of the user's leaves one so."""
    if not _is(module, ModuleType) or _attr(module, "__file__") is not None:
        return False
    places = _namespace_places(module)
    if places is None:
        name = _attr(module, "__name__")
        parent = name.rpartition(".")[0] if isinstance(name, str) else ""
        has_path = _attr(_attr(module, "__spec__"), "submodule_search_locations")
        return bool(parent) and has_path is not None and parent not in sys.modules
    return bool(places) and not any(is_library_file(os.path.join(p, "__init__.py")) for p in places)


def _attr(obj: object, name: str) -> object:
    """An attribute of what ``sys.modules`` holds, or None: a module may put an object of its own
    in its place there (``sys.modules[__name__] = _Basis()``), whose ``__getattr__`` may raise
    anything for a name it lacks (a ``KeyError`` from the dict it reads, or ``SystemExit``)."""
    try:
        return getattr(obj, name, None)
    except BaseException:
        return None


def _watch() -> _ImportWatch:
    """The process's import watch, kept first in ``sys.meta_path``: a finder ahead of it that
    found a module would hide that module's import from it."""
    if not sys.meta_path or sys.meta_path[0] is not _WATCH:
        with contextlib.suppress(ValueError):
            sys.meta_path.remove(_WATCH)
        sys.meta_path.insert(0, _WATCH)
    return _WATCH


def _drop(name: str) -> None:
    """Take a module out of ``sys.modules``, and out of the package it belongs to when that
    package stays there, which would otherwise hand ``from pkg import name`` the old module."""
    module = sys.modules.pop(name, None)
    parent, _, child = name.rpartition(".")
    holder = sys.modules.get(parent) if parent else None
    if module is not None and _is(holder, ModuleType) and _attr(holder, child) is module:
        with contextlib.suppress(AttributeError, TypeError):
            delattr(holder, child)


class Catalog:
    """Every model the GUI knows: the shipped ones, then those of each file loaded, by id."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entries: dict[str, ModelEntry] = {}
        self._files: dict[str, LoadedFile] = {}
        self._versions: dict[str, tuple[object, str, int]] = {}
        """By module name, the module of the user's own in ``sys.modules`` as the catalogue last
        saw it, its file, and the file's modification time when the module was imported."""
        self._namespaces: set[str] = set()
        """The namespace packages of the user's the catalogue saw imported, which have no file to
        record: a reload takes them out, and a folder of theirs made a package shows as a change."""
        self._placed: dict[str, tuple[str, int]] = {}
        """By module name, for a module recorded with another file than the one whose import put
        it in ``sys.modules`` (a helper that runs another file as itself), that file and its
        modification time as the import found it, so an edit to either shows."""
        for name in SHIPPED_MODULES:
            module = importlib.import_module(name)
            for cls in models_in(module):
                entry = ModelEntry(_model_id(cls), cls, "shipped", _source_path(cls))
                self._entries[entry.id] = entry

    def entries(self) -> list[ModelEntry]:
        with self._lock:
            return list(self._entries.values())

    def files(self) -> list[LoadedFile]:
        with self._lock:
            return list(self._files.values())

    def get(self, model_id: str) -> ModelEntry | None:
        with self._lock:
            return self._entries.get(model_id)

    def load_file(self, path: str | Path) -> LoadedFile:
        """Import a model file, or, when it was loaded before, reload: run every loaded file
        again (:meth:`reload`). Importing runs the file's top-level code. A file that fails to
        import leaves everything as it was."""
        file = _model_path(path)
        if file.suffix != ".py":
            raise ModelFileError(f"{file} is not a Python file: a model file ends in .py")
        if not file.is_file():
            raise ModelFileError(f"there is no file {file}")
        name = file.stem
        if not name.isidentifier():
            raise ModelFileError(
                f"{file.name}: the GUI imports a model file as the module named after it, so the "
                f"name before .py is a Python identifier (letters, digits and underscores, not "
                f"starting with a digit), not {name!r}"
            )
        with self._lock:
            known = self._files.get(str(file))
            if known is not None:
                # loaded already, as a load on another thread may have just done: running it
                # again is a reload of every file, which App.reload_models asks for, holding the
                # engine
                return known
            # a first load takes no engine: a build may run on another thread meanwhile, and what
            # it imports is its own
            mark = _watch().mark()
            since = time.time_ns()
            before = set(sys.modules)
            existing = sys.modules.get(name)
            if _attr(_attr(existing, "__spec__"), "_initializing"):
                # another thread is importing it (a build's formula, say): wait for that import to
                # end, as Python's own import of it would, then take what it gave
                with _watch().noting(), contextlib.suppress(Exception):
                    importlib.import_module(name)
                existing = sys.modules.get(name)
            # a module with a file of its own is adopted, whatever made it (an import, or a file
            # that runs another as a module of its own making), and refused when that file is
            # another; anything else holding the name is refused
            if self._own_module(name, existing):
                return self._adopt(file, name, typing.cast(ModuleType, existing))
            if existing is not None:
                raise ModelFileError(self._not_a_module(file, name, existing))
            folder = str(file.parent)
            if folder not in sys.path:
                sys.path.append(folder)
            found = importlib.util.find_spec(name)
            origin = None if found is None else found.origin
            if origin is not None and Path(origin).resolve() != file:
                raise ModelFileError(
                    f"import {name} finds {origin} before {file}; rename {file.name}, so that "
                    "the GUI and the worker processes of a run import yours"
                )
            with _watch().noting() as noted:
                try:
                    module, ran = self._exec(name, file)
                except BaseException:
                    # what the import brought in of the user's own goes, so the next import runs
                    # it again from its files, unless a build or a run on another thread may use
                    # it (see _drop_new); and so do the caches of the files whose code raised
                    self._drop_new(before, set(), beside=_watch().beside(mark))
                    _drop_failed_caches(noted, file)
                    raise
            added = self._register(noted, {name})
            # a module the watch did not see imported (a finder of the file's own put ahead of
            # it) is still recorded, so an edit to it shows and a reload takes it out
            unseen = set(sys.modules) - before - set(noted) - {name}
            self._register(sorted(n for n in unseen if not _theirs(n, since)), set())
            return self._record(file, name, module, added, None, ran)

    def known(self, path: str | Path) -> LoadedFile | None:
        """The record of a model file loaded before, by its path as :meth:`load_file` takes it."""
        with self._lock:
            return self._files.get(str(_model_path(path)))

    def reload(self) -> tuple[list[LoadedFile], list[str]]:
        """Run every loaded file again, in the order they were loaded, with every module of the
        user's own the catalogue has seen them use or import (their helpers, a formula's
        imports included): each is taken out of ``sys.modules`` first, its submodules with it,
        so the files import them afresh from disk, one copy of each, as a new Python process
        importing the files would. Every model is then a new class, and every earlier build of
        one is a build of the file as it was. A loaded file that is no longer there is let go,
        with its models. On an exception every module, file and model is as it was before. The
        files, and the paths of those let go."""
        with self._lock:
            dropped = self._reload()
            return list(self._files.values()), dropped

    def _reload(self) -> list[str]:
        state = (dict(self._files), dict(self._entries), dict(self._versions), dict(self._placed))
        dropped: list[str] = []
        for key, loaded in list(self._files.items()):
            if not loaded.path.is_file():  # renamed, moved or deleted: nothing to run again
                dropped.append(key)
                del self._files[key]
                for model_id in loaded.models:
                    self._entries.pop(model_id, None)
        files = list(self._files.values())
        gone = self._held()
        # a file saved again since its import, within the second its cached bytecode records and
        # at the same size, would run as it was: its cache goes first (see _drop_stale_bytecode)
        recorded: list[tuple[str, int]] = [(str(f.path), f.mtime_ns) for f in files]
        for name in gone:
            version = self._versions.get(name)
            if version is not None:
                recorded.append((version[1], version[2]))
            placed = self._placed.get(name)
            if placed is not None:
                recorded.append(placed)
        for file, mtime in recorded:
            _drop_stale_bytecode(file, mtime)
        for name in gone:
            sys.modules.pop(name, None)
        detached: list[tuple[ModuleType, str, object]] = []
        for name, module in gone.items():
            # a package left in sys.modules (a namespace package, which has no file of its own)
            # would hand ``from pkg import rates`` the old module it holds as an attribute
            parent, _, child = name.rpartition(".")
            holder = sys.modules.get(parent) if parent else None
            if _is(holder, ModuleType) and _attr(holder, child) is module:
                holder = typing.cast(ModuleType, holder)
                with contextlib.suppress(AttributeError, TypeError):
                    delattr(holder, child)
                    detached.append((holder, child, module))
        start = set(sys.modules)
        done: dict[str, tuple[ModuleType, list[str], int | None]] = {}
        imported: set[str] = set()  # what the files run so far imported
        noted: list[str] = []
        running: Path | None = None
        try:
            for loaded in files:
                running = loaded.path
                with _watch().noting() as noted:
                    module = sys.modules.get(loaded.module)
                    ran = None
                    found = _FOUND_AT.get(loaded.module)
                    put_here = found is not None and found[0] == str(loaded.path)
                    if module is not None and self._own_module(loaded.module, module):
                        # imported by a file run before, as a module with its file: that import
                        # is the file's, as a load after it takes it, unless such a load refuses
                        why = self._refusal(loaded.path, loaded.module, module, models_in(module))
                        if why is not None:
                            raise ModelFileError(why)
                    elif module is not None and loaded.module in imported and put_here:
                        # a file run before imported it, and its import put something else in
                        # its own place: refused, as a load after such an import is, rather than
                        # run a second time
                        raise ModelFileError(self._not_a_module(loaded.path, loaded.module, module))
                    else:
                        module, ran = self._exec(loaded.module, loaded.path)
                imported.update(noted)
                done[loaded.module] = (module, noted, ran)
            # everything this reload imported, recorded as imported now before any file's
            # record looks a module up: one file can use a module a later file's run imported
            self._register(sorted(set(sys.modules) - start), set())
            for loaded in files:
                module, noted, ran = done[loaded.module]
                added = self._register(noted, set(done))
                self._record(loaded.path, loaded.module, module, added, loaded, ran)
        except BaseException as exc:  # a file saved part-way through an edit may raise anything
            failure: BaseException = exc
            try:
                # the error written out before the modules it names are put back or taken out
                # (see format_quietly); one that is not an Exception (a KeyboardInterrupt) goes
                # on as it is, once everything is put back
                if _is(exc, Exception) and not _is(exc, ModelFileError):
                    failure = ModelFileError(
                        f"reloading the model files: {exception_text(exc)}", exc
                    )
            finally:
                self._files, self._entries, self._versions, self._placed = state
                # an object a module put in its own place there goes back too
                sys.modules.update(typing.cast(dict[str, ModuleType], gone))
                for holder, child, module in detached:
                    setattr(holder, child, module)
                self._drop_new(start, set(gone))
                _drop_failed_caches([*imported, *noted], running)
            if failure is exc:
                raise
            raise failure from exc
        return dropped

    def _drop_new(self, before: set[str], keep: set[str], *, beside: bool = False) -> None:
        """After a failed import or reload: take out what it brought into ``sys.modules`` of the
        user's own (but ``keep``, which is put back): every module with a file of the user's,
        every namespace package of the user's (a folder without an __init__.py, which has no file
        of its own), and everything under those names, until nothing more goes; a new Python
        process that imported the files would have none of them. After a load that another
        thread's work ran beside (``beside``: a build's, a check's or a run's), nothing goes:
        that work may use what the load imported first, through its own modules (``import h``),
        a package's code (``from . import h``) or a value it took (``from h import RATE``), and
        may still be importing a package the load imported a module of. What the load imported
        is recorded as imported now instead, so an edit to it shows and a reload takes it out."""
        new = set(sys.modules) - before - keep
        if beside:
            self._register(sorted(new), set())
            return
        dropped = {
            n
            for n in new
            if self._user_file(n, sys.modules.get(n)) is not None
            or _user_namespace(sys.modules.get(n))
        }
        while True:
            prefixes = tuple(f"{n}." for n in dropped | keep)
            more = {n for n in new if prefixes and n.startswith(prefixes)} - dropped
            if not more:
                break
            dropped |= more
        for name in sorted(dropped, key=len, reverse=True):  # submodules before their packages
            file = self._user_file(name, sys.modules.get(name))
            found = _FOUND_AT.get(name)
            # its next import runs the file as it is then, and the file whose import put it there
            for place in {file, None if found is None else found[0]} - {None}:
                _drop_stale_bytecode(typing.cast(str, place))
            _drop(name)

    def _held(self) -> dict[str, object]:
        """The modules a reload takes out of ``sys.modules``: every loaded file's, every module
        of the user's own the catalogue records (a file's helpers, and whatever its loads and
        builds imported, an object one put in its own place there too), and each one's
        submodules."""
        names = {f.module for f in self._files.values()} | set(self._versions)
        for f in self._files.values():
            names |= set(f.helpers)
        out: dict[str, object] = {}
        for name in names:
            module = sys.modules.get(name)
            if module is not None and self._user_file(name, module) is not None:
                out[name] = module
        # a namespace package of the user's (a folder without an __init__.py, which has no file
        # of its own) that the catalogue saw imported, or above one of them: left behind, it
        # would stay a namespace package after the folder is made a package, which a new Python
        # process would import as one
        for name in self._namespaces:
            module = sys.modules.get(name)
            if _user_namespace(module):
                out[name] = module
        for name in list(out):
            parent = name.rpartition(".")[0]
            while parent:
                module = sys.modules.get(parent)
                if parent not in out and _user_namespace(module):
                    out[parent] = module
                parent = parent.rpartition(".")[0]
        # a held package's every submodule, a namespace subpackage (a folder without an
        # __init__.py, which has no file of its own) too: left behind, the old one would still
        # look its parent up in sys.modules
        packages = tuple(f"{n}." for n in out)
        for name, module in list(sys.modules.items()):
            if packages and name.startswith(packages) and name not in out:
                out[name] = module
        return out

    def changed(self) -> bool:
        """Whether a loaded file, or a module of the user's own the catalogue imported or
        records, has changed (or gone) since it was imported, so that a reload would build
        something else. One flag for all the files, since a reload runs them all again."""
        with self._lock:
            # each file's own record, with the modules of the user's own it uses
            for f in self._files.values():
                if f.changed:
                    return True
            for f in self._files.values():
                # another session in this process may have run the file again since
                known = self._versions.get(f.module)
                if known is not None and sys.modules.get(f.module) is not known[0]:
                    return True
            # the catalogue's record of a module counts while that module is the one imported:
            # one imported again since (by another session, or by a build after a reload took it
            # out) counts through the records of the files whose models use it, above
            for name, (module, file, mtime) in self._versions.items():
                if sys.modules.get(name) is module:
                    # its file, and the file whose import put it there when that is another (one
                    # that runs another file as a module of its own name: spec_from_file_location)
                    found = self._placed.get(name)
                    places = [(file, mtime)] if found is None else [(file, mtime), found]
                    for place, stamp in places:
                        try:
                            if Path(place).stat().st_mtime_ns != stamp:
                                return True
                        except OSError:
                            return True
            # a folder imported as a namespace package, since given an __init__.py: a new Python
            # process imports it as a package
            for name in self._namespaces:
                places = _namespace_places(sys.modules.get(name)) or []
                if any(os.path.isfile(os.path.join(p, "__init__.py")) for p in places):
                    return True
            return False

    def _adopt(self, file: Path, name: str, existing: ModuleType) -> LoadedFile:
        """A model file imported before it was loaded: by another loaded file or a build, whose
        import the catalogue saw, or by the program that started the GUI, whose import it did
        not. That import is the file's, as the file was when it ran (see :meth:`_version`): one
        the catalogue did not see is taken as changed, which a reload settles."""
        classes = models_in(existing)  # read before anything is recorded
        why = self._refusal(file, name, existing, classes)
        if why is not None:
            raise ModelFileError(why)
        version = self._version(name, existing)
        mtime = 0 if version is None else version[1]
        uses = self._dependencies(existing)
        ids = [self._add(cls, file) for cls in classes]
        loaded = LoadedFile(file, name, mtime, ids, uses)
        self._files[str(file)] = loaded
        return loaded

    @staticmethod
    def _refusal(
        file: Path, name: str, existing: ModuleType, classes: list[type[Product]]
    ) -> str | None:
        """Why a module with a file of its own, held under a model file's name, is not that file's
        own import, which a load (or a reload) after it adopts; None when it is. It is not when
        it is another file's: imported from elsewhere, or run as this module by the file's own
        import (importlib's recipe); nor when it is a module object the file's own import put in
        its place, holding none of its models, that is not as an import of the file makes one
        (``types.ModuleType(__name__)`` given its ``__file__``, code run in it or not)."""
        where = _attr(existing, "__file__")
        found = _FOUND_AT.get(name)
        put_here = found is not None and found[0] == str(file)  # by this file's own import
        if not isinstance(where, str) or Path(where).resolve() != file:
            if put_here:  # it ran another file as its own module (spec_from_file_location)
                return (
                    f"importing {file.name} puts a module of another file, {where}, in its own "
                    "place in sys.modules, so the GUI cannot load it as a model file; put the "
                    "models in a module of their own, which imports this one"
                )
            return (
                f"a module named {name} is already imported, from "
                f"{where or 'Python itself'}; rename {file.name}, so that the GUI and "
                "the worker processes of a run import yours"
            )
        if not classes and put_here and not _made_by_import(existing, file):
            return (
                f"importing {file.name} puts a module object in its own place in sys.modules "
                "(sys.modules[__name__] = ...), so the GUI cannot load it as a model file; put "
                "the models in a module of their own, which imports this one"
            )
        return None

    def _record(
        self,
        file: Path,
        name: str,
        module: ModuleType,
        added: Mapping[str, tuple[str, int]],
        previous: LoadedFile | None,
        ran: int | None = None,
    ) -> LoadedFile:
        """A file's record after its import ran: its models (those it no longer defines gone)
        and the modules of the user's own it uses: those its import brought in, those its source
        and namespace name, and those its last record had that are still imported (one its top
        level reaches through ``importlib.import_module`` without binding it, say, which another
        file imported first). A file saved while it ran is not what ran, and is recorded as 0,
        which no file has, so it shows as changed: for a file the catalogue ran itself, its
        modification time then (``ran``) and now differ; for one another file's run imported,
        it is later than that import's start."""
        mtime = file.stat().st_mtime_ns
        saved = mtime != ran if ran is not None else _saved_since_import(name, str(file), mtime)
        if saved:
            mtime = 0
        # what it defines and uses, read before anything is recorded
        classes = models_in(module)
        uses = self._dependencies(module)
        # what sys.modules holds for it, which a file may have replaced with an object of its own
        # (sys.modules[__name__] = ...), as a change is told by
        self._versions[name] = (sys.modules.get(name, module), str(file), mtime)
        self._placed.pop(name, None)
        ids = [self._add(cls, file) for cls in classes]
        for model_id in [] if previous is None else previous.models:
            if model_id not in ids:
                self._entries.pop(model_id, None)
        kept: dict[str, tuple[str, int]] = {}
        earlier: list[str] = [] if previous is None else list(previous.helpers)
        for helper in earlier:
            still = sys.modules.get(helper)
            version = None if still is None else self._version(helper, still)
            if version is not None:
                kept[helper] = version
        helpers = {**kept, **added, **uses}
        helpers.pop(name, None)
        loaded = LoadedFile(file, name, mtime, ids, helpers)
        self._files[str(file)] = loaded
        return loaded

    def _register(self, names: list[str], skip: set[str]) -> dict[str, tuple[str, int]]:
        """The modules of the user's own among ``names``, which this thread imported just now
        (but those in ``skip``), each recorded as imported from its file as it is now; a
        namespace package of the user's among them is noted, having no file to record."""
        found: dict[str, tuple[str, int]] = {}
        for name in sorted(set(names) - skip):
            module = sys.modules.get(name)
            version = None if module is None else self._version(name, module, fresh=True)
            if version is not None:
                found[name] = version
            elif _user_namespace(module):
                self._namespaces.add(name)
        return found

    def watch_imports(self) -> contextlib.AbstractContextManager[list[str]]:
        """A context in which the modules this thread imports for the first time are noted,
        another thread's imports apart: a build's, say, while a file loads on another thread."""
        return _watch().noting()

    def note_imports(self, entry: ModelEntry, names: list[str]) -> None:
        """The modules a build imported (a formula that imports one, say): those of the user's
        own are helpers of the file that defines the model, as if its import had brought them
        in, so an edit to one shows as a change and a reload runs it again."""
        if entry.file is None:
            return
        with self._lock:
            found = self._register(names, set())
            loaded = self._files.get(str(entry.file))
            if loaded is None:
                return
            for name, version in found.items():
                if name != loaded.module:
                    loaded.helpers[name] = version
                    module = sys.modules.get(name)
                    if _is(module, ModuleType):
                        loaded.helpers.update(self._dependencies(typing.cast(ModuleType, module)))
            loaded.helpers.pop(loaded.module, None)

    def _version(self, name: str, module: object, *, fresh: bool = False) -> tuple[str, int] | None:
        """The file of a module of the user's own and the file's modification time when the
        module in ``sys.modules`` was imported from it; None for any other module. A module
        imported just now (``fresh``), or one whose import an import watch saw begin (a build's
        on another thread, say, whose own record of it comes when the build ends), is recorded
        as its file is now, unless the file was modified after its import began. One the
        catalogue first sees here, imported before it looked by an import no watch saw, is
        recorded as 0, which no file has, so it shows as changed until a reload imports it
        afresh: nothing records which version of the file such an import ran (a cached bytecode
        file is shared with every other process)."""
        file = self._user_file(name, module)
        if file is None:
            return None
        known = self._versions.get(name)
        if known is not None and known[0] is module and known[1] == file and not fresh:
            return file, known[2]
        try:
            mtime = Path(file).stat().st_mtime_ns
        except OSError:
            return None
        if (not fresh and name not in _IMPORTED_AT) or _saved_since_import(name, file, mtime):
            mtime = 0
        self._versions[name] = (module, file, mtime)
        found = _FOUND_AT.get(name)
        if found is not None and found[0] != file:
            self._placed[name] = found  # as this import found it, whatever a later one finds
        else:
            self._placed.pop(name, None)
        return file, mtime

    def _dependencies(self, module: ModuleType) -> dict[str, tuple[str, int]]:
        """The modules of the user's own a module uses, directly or through one another (see
        :func:`_names_used`), each with its file and the file's modification time when it was
        imported."""
        found: dict[str, tuple[str, int]] = {}
        own = _attr(module, "__name__")
        seen: set[str] = {str.__str__(typing.cast(str, own))} if _is(own, str) else set()
        todo = [module]
        while todo:
            current = todo.pop()
            for name in sorted(_names_used(current)):
                if name in seen:
                    continue
                seen.add(name)
                used = sys.modules.get(name)
                version = None if used is None else self._version(name, used)
                if version is None:
                    continue
                found[name] = version
                if _is(used, ModuleType):  # not an object a module put in its place
                    todo.append(typing.cast(ModuleType, used))
        return found

    def _user_file(self, name: str, module: object) -> str | None:
        """The source file of a module of the user's own; None for one of Halley's (the Numba
        backend's generated modules too), Python's or an installed package's, one without a
        file, or one imported from a zip archive on ``sys.path``, whose file is a member of the
        archive, which has no modification time of its own to compare. For an object a module of
        the user's put in its own place in ``sys.modules`` (``sys.modules[__name__] =
        _Basis()``), which ``import`` gives, the file the import watch saw that import find: the
        object's own attributes are not read, since its ``__getattr__`` may raise anything. So
        too for a module object such a file made itself (``types.ModuleType(__name__)``, given
        the import's spec or not), which has no file of its own, and for another module it put
        there (``sys.modules[__name__] = importlib.import_module("prudent")``), which keeps its
        own name too. Any other module is its own file's, whatever its ``__name__`` says."""
        if module is None or name.startswith("halley_numba_"):
            return None
        if not self._own_module(name, module):
            # an object, a module object of the file's making, or another module kept under its
            # own name (one built into Python too), put in this name's place by the file whose
            # import the watch saw find it; a module built into Python under its own name is
            # not the file's, which never ran: Python imports its own first
            found = _FOUND_AT.get(name)
            if found is not None and not (_built_in(module) and _name_of(module) == name):
                return found[0]
            if not _is(module, ModuleType) or not _is(_attr(module, "__file__"), str):
                return None
            # another module's alias the watch did not see made: that module's file
        file = _attr(module, "__file__")
        if _is(_attr(module, "__loader__"), zipimport.zipimporter):
            return None
        text = str.__str__(typing.cast(str, file))
        if is_library_file(text):
            return None
        return str(Path(text).resolve())

    @staticmethod
    def _own_module(name: str, module: object) -> bool:
        """Whether what ``sys.modules`` holds under ``name`` is a module with a file of its own,
        and not another module kept under its own name too, which a file put in this name's
        place (``sys.modules[__name__] = importlib.import_module("prudent")``)."""
        if not _is(module, ModuleType) or not _is(_attr(module, "__file__"), str):
            return False
        own = _name_of(module)
        return own is None or own == name or sys.modules.get(own) is not module

    def _exec(self, name: str, file: Path) -> tuple[ModuleType, int]:
        """Run a model file as the module ``name``: the module, and the file's modification time
        as its run began, which a save while it runs changes (a time, not the clock's: a file
        may be dated ahead of this computer's clock)."""
        spec = importlib.util.spec_from_file_location(name, file)
        if spec is None or spec.loader is None:
            raise ModelFileError(f"Python cannot import {file}")
        try:
            ran = file.stat().st_mtime_ns
        except OSError as exc:
            raise ModelFileError(f"cannot read {file}: {exc}", exc) from exc
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module  # before it runs: the tracer finds the class's source by it
        linecache.checkcache(str(file))  # so build errors and explain show the file as saved
        # when this run of the file began, for a module that imports it (the import watch stamps
        # the modules the import system imports; this one it does not see)
        _IMPORTED_AT[name] = time.time_ns()
        _NOTED_BY[name] = threading.get_ident()
        _FOUND_AT[name] = (str(file.resolve()), ran)
        try:
            spec.loader.exec_module(module)
        except BaseException as exc:  # a model file's code may raise anything, sys.exit included
            try:
                # written out while what the file imported is still imported (see
                # format_quietly)
                failure = ModelFileError(
                    f"importing {file.name} raised {exception_name(exc)}: {exception_text(exc)}",
                    exc,
                )
            finally:
                sys.modules.pop(name, None)
            raise failure from exc
        return module, ran

    def _not_a_module(self, file: Path, name: str, existing: object) -> str:
        """Why a model file whose module name ``sys.modules`` holds an object for, which is not a
        module its import made, cannot be loaded."""
        try:
            kind = str.__str__(type(existing).__name__)
        except BaseException:  # a class of the user's whose name cannot be read
            kind = "another"
        what = "a module object" if _is(existing, ModuleType) else f"a {kind} object"
        if self._user_file(name, existing) == str(file):
            return (
                f"importing {file.name} puts {what} in its own place in sys.modules "
                "(sys.modules[__name__] = ...), so the GUI cannot load it as a model file; put "
                "the models in a module of their own, which imports this one"
            )
        if _built_in(existing) and _name_of(existing) == name:
            return (  # Python's own, which Python imports before any file
                f"a module named {name} is already imported, from Python itself; rename "
                f"{file.name}, so that the GUI and the worker processes of a run import yours"
            )
        return (
            f"{name} is already in sys.modules, as {what}, not the module an import of "
            f"{file.name} makes; if {file.name} puts it there itself (sys.modules[__name__] = "
            "...), put its models in a module of their own, which imports it; otherwise rename "
            f"{file.name}, so that the GUI and the worker processes of a run import yours"
        )

    def _add(self, cls: type[Product], file: Path) -> str:
        entry = ModelEntry(_model_id(cls), cls, "file", file)
        self._entries[entry.id] = entry
        return entry.id

    def entry_for(self, full_name: str, source_file: str | None) -> ModelEntry:
        """The model a run's manifest names (``model.name``, ``<module>.<qualname>``), from the
        catalogue or, for a model file not loaded yet, by loading the file the manifest records.
        A model defined in a script run as ``python file.py`` is named ``__main__.<class>``,
        which no import of the file can give it, so it is refused without running the script."""
        for entry in self.entries():
            if entry.full_name == full_name:
                return entry
        if full_name.startswith("__main__."):
            script = Path(source_file).name if source_file else "a script"
            raise ModelFileError(
                f"the run's model is {full_name}: it was defined in {script} run as a script, "
                "whose module is __main__, and a model's module name is part of its IR hash, so "
                "no import of the file can build the same model; put the class in a module of "
                "its own, import it in the script, and run that"
            )
        source = None if source_file is None else Path(source_file)
        # a file loaded already is not run again here: that is a reload of every file, which
        # the user asks for, and the file as it is does not define the run's model
        if (
            source is not None
            and source.suffix == ".py"
            and source.is_file()
            and not self.known(source)
        ):
            loaded = self.load_file(source)
            for model_id in loaded.models:
                entry = self.get(model_id)
                if entry is not None and entry.full_name == full_name:
                    return entry
        raise ModelFileError(
            f"the run's model {full_name} is not a model the GUI has loaded, and its source "
            f"{source_file} does not define it; load the file that defines it"
        )


def _model_path(path: str | Path) -> Path:
    """A model file's path as the catalogue keeps it: absolute, a relative one from the folder the
    GUI was started in, with links resolved."""
    file = Path(path).expanduser()
    if not file.is_absolute():
        file = Path.cwd() / file
    return file.resolve()


def models_in(module: ModuleType) -> list[type[Product]]:
    """The model classes a module defines (not those it imports), in the order of their source
    lines."""
    found: list[type[Product]] = []
    own = _attr(module, "__name__")
    for value in list(_namespace(module).values()):
        if (
            _is(value, type)  # not what an object of the user's in the module says it is
            and issubclass(typing.cast(type, value), Product)
            and value is not Product
            and value is not Aggregation
            and _attr(value, "__module__") == own
        ):
            found.append(typing.cast(type[Product], value))
    return sorted(found, key=lambda c: (_class_line(c) or 0, c.__qualname__))


@functools.cache
def _library_roots() -> tuple[str, ...]:
    """The folders whose code is not the user's: Halley's package, Python's standard library and
    compiled modules (the base installation's folders as well as the environment's), and the
    folders installed packages go to; each in the case the system compares paths in, as
    written and as the file system resolves it, and ending in a separator. Neither an
    environment's own folder nor a whole installation prefix is one of them, since a project
    may hold its environment and its model files in one folder, and a prefix may be ``/usr``."""
    places = {str(Path(halley.__file__).parent)}
    base = {"base": sys.base_prefix, "platbase": sys.base_exec_prefix}
    for paths in (sysconfig.get_paths(), sysconfig.get_paths(vars=base)):
        places.update(
            paths[k] for k in ("stdlib", "platstdlib", "purelib", "platlib") if paths.get(k)
        )
    prefixes = {
        os.path.normcase(os.path.abspath(p)).rstrip(os.sep)
        for p in (sys.prefix, sys.exec_prefix, sys.base_prefix, sys.base_exec_prefix)
    }
    getsitepackages = getattr(site, "getsitepackages", None)
    if getsitepackages is not None:
        for place in typing.cast(list[str], getsitepackages()):
            # Windows lists each prefix itself among its site folders
            if os.path.normcase(os.path.abspath(place)).rstrip(os.sep) not in prefixes:
                places.add(place)
    getusersitepackages = getattr(site, "getusersitepackages", None)
    user_site = None if getusersitepackages is None else getusersitepackages()
    if isinstance(user_site, str):
        places.add(user_site)
    if sys.platform == "win32":  # where Windows keeps the standard library's compiled modules
        places.update(os.path.join(p, "DLLs") for p in (sys.base_prefix, sys.base_exec_prefix))
    roots: set[str] = set()
    for place in places:
        for form in (os.path.abspath(place), os.path.realpath(place)):
            roots.add(os.path.normcase(form).rstrip(os.sep) + os.sep)
    return tuple(sorted(roots))


def is_library_file(file: str) -> bool:
    """Whether a source file is Halley's own, Python's or an installed package's rather than the
    user's. The file is compared as written and as the file system resolves it, so a link or a
    folder the system redirects does not make a library file look like the user's. A relative
    name is never the user's: Python imports a model file by its full path, and a relative name
    is what compiled code of an installed package gives its frames (``pyarrow/error.pxi``)."""
    if not os.path.isabs(file):
        return True
    roots = _library_roots()
    for form in (os.path.abspath(file), os.path.realpath(file)):
        if os.path.normcase(form).startswith(roots):
            return True
    return False


def _names_used(module: ModuleType) -> set[str]:
    """The modules a module uses by name: those its source imports anywhere in the file
    (``import x``, ``from x import y``, which a constant comes through too), those its namespace
    holds, and those its classes and functions come from; each with the packages above it,
    which are imported with it."""
    names: set[str] = set()
    # a module object of the user's may raise for any attribute it lacks: each is read so as
    # never to fail
    file = _attr(module, "__file__")
    if _is(file, str) and str.__str__(typing.cast(str, file)).endswith(".py"):
        package = _attr(module, "__package__")
        place = str.__str__(typing.cast(str, package)) if _is(package, str) else ""
        names |= _source_imports(str.__str__(typing.cast(str, file)), place)
    for value in list(_namespace(module).values()):
        # told by each value's own class: an object of the user's the module holds (one a helper
        # put in its own place in sys.modules) may raise for any attribute, __class__ included
        if _is(value, ModuleType):
            name = _attr(value, "__name__")
            if _is(name, str):
                names.add(str.__str__(typing.cast(str, name)))
        elif _is(value, (type, types.FunctionType)):
            owner = _attr(value, "__module__")
            if _is(owner, str):
                names.add(str.__str__(typing.cast(str, owner)))
    out: set[str] = set()
    for name in names:
        parts = name.split(".")
        out.update(".".join(parts[: i + 1]) for i in range(len(parts)))
    return out


def _source_imports(file: str, package: str = "") -> set[str]:
    """The modules a source file's import statements name, a relative one resolved against the
    module's ``package``; a submodule a ``from`` names is one of them when it is a module."""
    try:
        tree = ast.parse(Path(file).read_bytes(), filename=file)
    except (OSError, SyntaxError, ValueError):
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                try:
                    base = importlib.util.resolve_name("." * node.level + base, package)
                except (ImportError, ValueError):
                    continue
            if not base:
                continue
            names.add(base)
            names.update(f"{base}.{a.name}" for a in node.names if a.name != "*")
    return names


def _type_text(hint: object) -> str:
    """A declared type as Python writes it; where the user's own ``__repr__`` of an enum's member
    in it raises, with that member by its class and name (``Literal[RB.A, RB.B] | float``)."""
    if isinstance(hint, type):  # as typing writes one inside another: float, rates.Rate
        name = _shown(str, _attr(hint, "__qualname__"))
        module = _shown(str, _attr(hint, "__module__"))
        return f"{name}" if module == "builtins" else f"{module}.{name}"
    text = _shown(repr, hint)
    if text is not None:
        return text
    if _is(hint, enum.Enum):
        return f"{type(hint).__name__}.{_attr(hint, '_name_')}"
    origin = typing.get_origin(hint)
    parts = [_type_text(a) for a in typing.get_args(hint)]
    if origin in (typing.Union, types.UnionType):
        return " | ".join(parts)
    name = "Literal" if origin is typing.Literal else _attr(origin, "__name__")
    return f"{name}[{', '.join(parts)}]" if _is(name, str) and parts else "a type"


def statics_of(cls: type[Product]) -> list[StaticParam]:
    """The build-time parameters of a model, found as the tracer finds them: every annotation
    ``Static[T]`` of the class and its bases, with the class's value as the default."""
    declared: dict[str, StaticParam] = {}
    for klass in reversed(cls.__mro__):
        if klass is object or klass is Product:
            continue
        try:
            hints = typing.get_type_hints(klass)
        except BaseException:  # the build says what is wrong with the annotations, or a module's
            continue
        for name, hint in hints.items():
            if getattr(hint, "__origin__", None) is not Static:
                continue
            args: tuple[object, ...] = getattr(hint, "__args__", ())
            kind = args[0] if len(args) == 1 else None
            default = getattr(cls, name, _MISSING)
            declared[name] = StaticParam(
                name,
                kind.__name__ if isinstance(kind, type) else _type_text(kind),
                None if default is _MISSING else default,
                default is not _MISSING,
                (typing.get_origin(kind) or kind) is tuple,
                kind,
            )
    return list(declared.values())


def typed_statics(
    cls: type[Product],
    values: Mapping[str, object],
    base: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build-time settings from the build form, which sends only the fields changed from what it
    showed: the class's defaults, or the settings of the build it showed (``base``, that build's
    own values, carried over as they are). So an untouched form builds what ``build(cls)``
    builds, or the build it showed, with the same IR hash. A field changed back to the default's
    own text takes the default itself; any other is read by the type its parameter is declared
    with (:func:`read_static`)."""
    params = {p.name: p for p in statics_of(cls)}
    out: dict[str, object] = {}
    for name, value in (base or {}).items():
        if name in params:  # a setting the class no longer declares is not carried over
            out[name] = value
    for name, value in values.items():
        param = params.get(name)
        if param is None:
            out[name] = value  # build() names the parameters the class has
        elif param.has_default and _is_default_text(param, value):
            out.pop(name, None)
        else:
            out[name] = read_static(param, value)
    return out


def _is_default_text(param: StaticParam, value: object) -> bool:
    if param.input == "bool":
        flag = value if isinstance(value, bool) else {"true": True, "false": False}.get(str(value))
        return flag is not None and flag is (param.default is True)
    return isinstance(value, str) and value == param.text


def carried_statics(
    then: type[Product], values: Mapping[str, object], now: type[Product]
) -> dict[str, object]:
    """A build's settings for a build of the class as it is now, when the file has replaced
    the build's class (``then``) since: each as it was, an enum's member of an enum the file has
    defined again as the new enum's member of that name (in a tuple or a union too). A setting
    that was its class's default takes the default the class has now, so a default edited in
    the file takes effect, and so does one that holds another object of the file as it was (an
    instance of a class the file defines), which no setting of the new class can be."""
    out: dict[str, object] = {}
    for param in statics_of(now):
        if param.name not in values:
            continue
        value = values[param.name]
        # the old default as the old class holds it: its annotations, written as strings, may
        # no longer resolve in the module as it is now, so statics_of(then) cannot be asked
        old = getattr(then, param.name, _MISSING)
        if old is not _MISSING and _same(value, old):
            continue
        moved = _moved(value, param.hint)
        if moved is not _NO:
            out[param.name] = moved
    return out


def _same(a: object, b: object) -> bool:
    """Whether two settings are one value by the IR hash's own measure, which writes each as its
    repr: ``1`` and ``1.0`` apart, a tuple and a list apart, a dict's keys in their order; and
    each enum's member in them the same member, which a formula may test (``self.basis is
    Basis.BEST``) where its class writes every member alike."""
    if repr(a) != repr(b):
        return False
    first, second = _members_in(a), _members_in(b)
    return len(first) == len(second) and all(x is y for x, y in zip(first, second, strict=True))


def _members_in(value: object) -> list[object]:
    """The enum's members a setting holds, depth first: itself, or those in its lists, tuples
    and dicts, their keys too."""
    if _is(value, enum.Enum):
        return [value]
    found: list[object] = []
    if _is(value, Mapping):
        for k, v in typing.cast(Mapping[object, object], value).items():
            found += _members_in(k) + _members_in(v)
    elif _is(value, (list, tuple)):
        for v in typing.cast(Sequence[object], value):
            found += _members_in(v)
    return found


def _moved(value: object, hint: object) -> object:
    """A value of a build of a class since replaced, for the class as it is now (see
    :func:`carried_statics`), or ``_NO``."""
    hint = _unwrap(hint)
    if typing.get_origin(hint) in (typing.Union, types.UnionType):
        if value is None:
            return None
        for option in typing.get_args(hint):
            if option is not type(None):
                found = _moved(value, option)
                if found is not _NO:
                    return found
        return _NO
    if value is None or isinstance(value, (bool, int, float, str, dt.date)):
        if isinstance(value, enum.Enum):
            return _renamed(value, hint)
        return value
    if isinstance(value, enum.Enum):
        return _renamed(value, hint)
    if isinstance(value, (tuple, list)):
        items = list(typing.cast(tuple[object, ...] | list[object], value))
        args = typing.get_args(hint)
        if (typing.get_origin(hint) or hint) is tuple and len(args) == 2 and args[1] is Ellipsis:
            hints = [args[0]] * len(items)
        elif (typing.get_origin(hint) or hint) is tuple and len(args) == len(items):
            hints = list(args)
        elif typing.get_origin(hint) in _LISTS and len(args) == 1:
            hints = [args[0]] * len(items)
        else:
            hints = [object] * len(items)
        parts = [_moved(v, h) for v, h in zip(items, hints, strict=True)]
        if any(p is _NO for p in parts):
            return _NO
        return tuple(parts) if isinstance(value, tuple) else parts
    if isinstance(value, Mapping):
        items = typing.cast(Mapping[object, object], value).items()
        args = typing.get_args(hint)
        key_hint, value_hint = args if len(args) == 2 else (object, object)
        pairs = [(_moved(k, key_hint), _moved(v, value_hint)) for k, v in items]
        return _NO if any(k is _NO or v is _NO for k, v in pairs) else dict(pairs)
    module = sys.modules.get(type(value).__module__)
    if module is not None and getattr(module, type(value).__qualname__, None) is type(value):
        return value  # its class is still the one its module holds: a library's, say
    return _NO


def _renamed(member: enum.Enum, hint: object) -> object:
    """An enum's member as the member of that name of the enum the parameter is declared with
    now (directly, in a union or as an element type), whatever that enum is called now, since
    the form shows the member by its name; or itself when its enum is still that enum."""
    options = [hint, *typing.get_args(hint)]
    for option in options:
        if isinstance(option, enum.Enum) and option.name == member.name:
            return option  # a Literal's member of that name
        if isinstance(option, type) and issubclass(option, enum.Enum):
            if isinstance(member, option):
                return member
            if member.name is not None and member.name in option.__members__:
                return option.__members__[member.name]
            if isinstance(member, enum.Flag) and issubclass(option, enum.Flag):
                names = [m.name for m in type(member) if m in member and m.name is not None]
                if all(n in option.__members__ for n in names):
                    found = _combined(option, names)
                    if found is not _NO:
                        return found
    return _NO


def settings_texts(cls: type[Product], values: Mapping[str, object]) -> dict[str, str]:
    """A build's settings as text, each written by its declared type, or as JSON (or its repr)
    for one whose type the class's annotations no longer give: annotations written as strings
    resolve in the module as it is now, which may no longer define the types they name."""
    hints = {p.name: p.hint for p in statics_of(cls)}
    return {name: form_text(hints.get(name), value) for name, value in values.items()}


def form_texts(
    then: type[Product], values: Mapping[str, object], now: type[Product]
) -> dict[str, str]:
    """A build's settings as the build form of the class as it is now shows them: its values'
    texts, and for a build of a class the file has since replaced, the texts of
    :func:`carried_statics`, with the class's defaults for the settings that do not carry."""
    if then is now:
        return {
            p.name: form_text(p.hint, values[p.name]) for p in statics_of(now) if p.name in values
        }
    carried = carried_statics(then, values, now)
    return {
        p.name: form_text(p.hint, carried[p.name]) if p.name in carried else p.text
        for p in statics_of(now)
    }


def recorded_statics(
    cls: type[Product], values: Mapping[str, object], *, keep_defaults: bool = False
) -> dict[str, object]:
    """Build-time settings as a run's manifest records them, for building its model again. The
    manifest records every setting, the defaults too, as JSON holds it (``1`` and ``1.0``
    apart), and a value JSON cannot hold as its repr. Each is read back by its declared type
    where it fits (:func:`_converted`: an enum's member, a date, through tuples and unions, a
    list as a tuple where the type is a tuple), and used as recorded where it does not, since
    ``build()`` took it and the IR hash, not the type, says whether it builds the run's model.
    A value recorded exactly as the class's default is recorded is left out, so the default
    itself is used, unless ``keep_defaults``. A setting declared an enum, a date or another
    class whose record does not read back (a member the enum no longer has, the repr of an
    object) cannot be what ``build()`` took, and is refused before anything is built."""
    params = {p.name: p for p in statics_of(cls)}
    out: dict[str, object] = {}
    for name, value in values.items():
        param = params.get(name)
        if param is None:
            out[name] = value  # build() names the parameters the class has
            continue
        found = _converted(value, param.hint, "record")
        if param.has_default and _record(param.default) == _record(value):
            if keep_defaults and found is not _NO:
                out[name] = found
            continue
        if found is _NO:
            hint = param.hint
            strict = isinstance(hint, type) and hint not in (
                bool,
                int,
                float,
                str,
                tuple,
                list,
                dict,
            )
            if strict:
                raise HalleyError(_unreadable(cls, param, value))
            found = _as_recorded(value, hint)
        out[name] = found
    return out


MAX_RERUN_TRIES = 64
"""How many sets of build-time settings a rerun tries (see :func:`rerun_tries`)."""


@dataclass(frozen=True, slots=True)
class RerunMisses:
    """What :func:`rerun_misses` finds: the settings no reading of a run's manifest gives back,
    and the dicts given sorted, recorded as their defaults are, in a mix with others at their
    defaults that a rerun's tries, cut short at ``MAX_RERUN_TRIES``, do not reach."""

    names: list[str]
    beyond: list[str] = field(default_factory=lambda: [])
    alike: list[str] = field(default_factory=lambda: [])
    """Of ``names``, those a rerun gives back as built but for an enum's member its class writes
    as it writes another, which a run misses only where a formula tells them apart."""


def _rerun_space(
    cls: type[Product], recorded: Mapping[str, object]
) -> tuple[dict[str, object], dict[str, object], list[str]]:
    """The settings as a rerun reads them back with the defaults left to the class, as it reads
    them back every one, and the settings it tries both ways: recorded exactly as their defaults
    are recorded, but reading back as another value (a dict whose default's keys are not sorted,
    which a manifest writes sorted)."""
    base = recorded_statics(cls, recorded)
    every = recorded_statics(cls, recorded, keep_defaults=True)
    defaults = {p.name: p.default for p in statics_of(cls) if p.has_default}
    either = [n for n in every if n not in base and not _same(every[n], defaults.get(n))]
    return base, every, either


def _read_back(names: list[str]) -> list[tuple[str, ...]]:
    """The sets of ``names`` a rerun reads back (the others left at their defaults), in the
    order it tries them: none, then all (as given in a key order a manifest writes), then all
    but one, one, all but two, two and so on, up to ``MAX_RERUN_TRIES``."""
    k = len(names)
    out: list[tuple[str, ...]] = []
    for d in range(k // 2 + 1):
        for size in dict.fromkeys((d, k - d)):
            for chosen in itertools.combinations(names, size):
                out.append(chosen)
                if len(out) == MAX_RERUN_TRIES:
                    return out
    return out


def rerun_tries(
    cls: type[Product], recorded: Mapping[str, object]
) -> tuple[list[dict[str, object]], bool]:
    """The build-time settings a rerun tries, in order, to build a run's model again from what
    its manifest records (:func:`recorded_statics`), the one with the run's IR hash being the
    run's, and whether they were cut short at ``MAX_RERUN_TRIES``. A setting recorded exactly as
    its default is recorded may still be another value, since a manifest writes a dict's keys
    sorted and the default's may not be: each such setting whose record reads back as another
    value than the default is tried both ways (see :func:`_read_back`)."""
    base, every, either = _rerun_space(cls, recorded)
    sets = _read_back(either)
    tries = [{**base, **{n: every[n] for n in chosen}} for chosen in sets]
    return tries, len(sets) < 2 ** len(either)


def rerun_misses(cls: type[Product], statics: Mapping[str, object]) -> RerunMisses:
    """The settings of a build (``statics``: every one, as its graph holds them) that a rerun
    from a run's directory would not give back, measured as the IR hash measures them (each
    value's repr), with what the run's manifest would record. A rerun finds the mix the build
    needs, each dict recorded as its default is read back or left at its default, when its tries
    reach it (:func:`rerun_tries`); a value a manifest cannot record exactly is not given back in
    any: a dict typed in a key order that is not sorted and is not its class's own default, a
    NumPy number, an object recorded as its repr; nor is a setting whose record does not read
    back as its declared type at all, which a rerun refuses, while the others are measured as a
    rerun reads them once that is put right."""
    recorded = json.loads(json.dumps(_json_safe(dict(statics)), sort_keys=True))
    unread: set[str] = set()
    try:
        recorded_statics(cls, recorded)
    except HalleyError:  # a setting whose record does not read back as its type at all
        for name, value in recorded.items():
            try:
                recorded_statics(cls, {name: value})
            except HalleyError:
                unread.add(name)
    # the others as a rerun reads them once those are put right
    rest = {n: v for n, v in recorded.items() if n not in unread}
    base, every, either = _rerun_space(cls, rest)
    defaults = {p.name: p.default for p in statics_of(cls) if p.has_default}
    # the mix the build needs: each dict read back that was given as it reads back
    needed = tuple(n for n in either if _same(every[n], statics.get(n)))
    given = {**defaults, **base, **{n: every[n] for n in needed}}
    names = [
        n for n, v in statics.items() if n in unread or n not in given or not _same(given[n], v)
    ]
    alike = [
        n
        for n in names
        if n not in unread and n in given and _shown(repr, given[n]) == _shown(repr, statics[n])
    ]
    reached = needed in _read_back(either)
    return RerunMisses(names, [] if reached else list(needed), alike)


def unsorted_dict(value: object, at: str = "") -> tuple[str, list[str]] | None:
    """The first dict in a value (the value itself, or one it holds, depth first) whose keys are
    not in the order a run's manifest writes them, sorted as text (``str()`` of each: ``10``
    before ``2``, an int or str enum's member by its value): where it is, as the form writes a
    path into JSON (``""`` for the value itself, ``["x"][0]`` below it), and its keys in that
    order, each as the form writes it; None when every dict's keys are in that order."""
    if isinstance(value, Mapping):
        items = list(typing.cast(Mapping[object, object], value).items())
        keys = [k for k, _ in items]
        if [str(k) for k in keys] != sorted(str(k) for k in keys):
            return at, [_key_text(k) for k in sorted(keys, key=str)]
        for k, v in items:
            found = unsorted_dict(v, f"{at}[{json.dumps(_key_text(k), ensure_ascii=False)}]")
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(typing.cast(Sequence[object], value)):
            found = unsorted_dict(v, f"{at}[{i}]")
            if found is not None:
                return found
    return None


def _as_recorded(
    value: object, hint: object, mode: str = "record", *, inner: bool = False
) -> object:
    """A value that does not fit its type item by item, read as far as it fits (by ``mode``'s
    rules, a manifest's or a form's) and as given beyond: a list as a tuple where the type is a
    tuple, element by element, and a list inside a tuple that its type does not take as a
    tuple too (a manifest writes every tuple as a list); a list's items by its item type, and a
    dict's keys by its key type and its values by its value type, as far as each fits."""
    found = _converted(value, hint, mode)
    if found is not _NO:
        return found
    hint = _unwrap(hint)
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)
    if origin in (typing.Union, types.UnionType):
        shaped = [a for a in args if _shaped(value, a) and _read_by_type(a)]
        # the member type that reads part of it, else the first of the value's shape
        reading = [a for a in shaped if _reads_some(value, a, mode)] + shaped
        if reading:
            return _as_recorded(value, reading[0], mode, inner=inner)
        return _tupled(value) if inner else value
    if (origin or hint) is tuple and isinstance(value, list):
        items = typing.cast(list[object], value)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_as_recorded(v, args[0], mode, inner=True) for v in items)
        if args and len(args) == len(items):
            return tuple(
                _as_recorded(v, a, mode, inner=True) for v, a in zip(items, args, strict=True)
            )
        return tuple(_tupled(v) for v in items)
    if origin in _MAPPINGS and isinstance(value, Mapping):
        key_hint, value_hint = (*args, object, object)[:2]
        out: dict[object, object] = {}
        for k, v in typing.cast(Mapping[object, object], value).items():
            key = _key(k, key_hint, mode)
            out[k if key is _NO else key] = _as_recorded(v, value_hint, mode)
        return out
    if origin in _LISTS and len(args) == 1 and isinstance(value, list):
        return [_as_recorded(v, args[0], mode) for v in typing.cast(list[object], value)]
    listy = (origin or hint) in _LISTS
    return _tupled(value) if inner and not listy else value


def _unreadable(cls: type[Product], param: StaticParam, value: object) -> str:
    hint = param.hint
    if isinstance(hint, type) and issubclass(hint, enum.Enum):
        names = ", ".join(hint.__members__)
        why = f"is none of its members as the class is now ({names})"
    else:
        why = f"does not read back as a {param.kind}"
    return (
        f"the run's manifest records {param.name}, a {param.kind}, as {value!r}, which {why}, so "
        f"{cls.__qualname__} cannot be built again as the run built it"
    )


def _record(value: object) -> str:
    """A value as a manifest records it, as text: JSON, which tells ``1``, ``1.0`` and ``true``
    apart where ``==`` does not."""
    return json.dumps(_json_safe(value), sort_keys=True)


_NO = object()
"""What :func:`_converted` gives for a value that does not fit the type."""

_ENUM_REPR = re.compile(r"<[\w.|]+: (.*)>", re.DOTALL)


def _converted(value: object, hint: object, mode: str) -> object:
    """A value from JSON as the declared type ``hint`` takes it, or ``_NO`` when it does not
    fit: through unions (the first member type that takes it), ``Literal``, tuples (element by
    element; a bare ``tuple``'s nested lists are tuples too), lists and dicts (item by item), an
    enum's member, a date, and the plain types. ``mode`` says where the value comes from:
    ``"form"``, a form field's JSON, where a whole number is a float's 2.0, an enum's member is
    its name and a date its ISO text; or ``"record"``, a manifest's, where numbers stay as
    recorded, a member is recorded as the manifest records it (its repr, or its value for an int
    or str enum) and a date as its repr. A ``NewType`` is read as its supertype."""
    hint = _unwrap(hint)
    if hint is typing.Any or hint is object or hint is None:
        return value
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)
    if origin in (typing.Union, types.UnionType):
        if value is None:
            return None if type(None) in args else _NO
        # the first member type that reads it, before one that takes it only as JSON gives it (a
        # set's): the list option after it reads its items; and a list or a dict that a member
        # type reads, but not whole, is not taken as JSON gives it by one that cannot hold it,
        # but read by that type as far as it fits (see _as_recorded)
        options = [a for a in args if a is not type(None)]
        read = [a for a in options if _read_by_type(a)]
        for option in read:
            found = _converted(value, option, mode)
            if found is not _NO:
                return found
        # then a member type that holds the value as JSON gives it (an Iterable holds a list),
        # and only after the one that reads it in part, one that cannot hold it (a set)
        rest = [a for a in options if a not in read]
        for option in rest:
            if _holds(value, option):
                return value
        if any(_reads_some(value, a, mode) for a in read):
            return _NO
        for option in rest:
            found = _converted(value, option, mode)
            if found is not _NO:
                return found
        return _NO
    if origin is typing.Literal:
        # the literal itself: an enum's member is typed as the enum's own field takes it (see
        # _literal_member), and recorded as the manifest records one
        if mode == "form":
            return _literal_member(args, value, lambda a: _record(value) == _recorded(a))
        return next((a for a in args if _record(value) == _recorded(a)), _NO)
    if (origin or hint) is tuple:
        if not isinstance(value, (list, tuple)):
            return _NO
        items = list(typing.cast(list[object] | tuple[object, ...], value))
        if len(args) == 2 and args[1] is Ellipsis:
            parts = [_converted(v, args[0], mode) for v in items]
        elif not args:
            parts = [_tupled(v) for v in items]
        elif len(args) == len(items):
            parts = [_converted(v, a, mode) for v, a in zip(items, args, strict=True)]
        else:
            return _NO
        return _NO if any(p is _NO for p in parts) else tuple(parts)
    if origin in _MAPPINGS:
        # a dict's keys by its key type (JSON's keys are texts: see _key), its values by its
        # value type (an enum's member, a date, a tuple inside it)
        if not isinstance(value, Mapping):
            return _NO
        key_hint, value_hint = (*args, object, object)[:2]
        out: dict[object, object] = {}
        for k, v in typing.cast(Mapping[object, object], value).items():
            key = _key(k, key_hint, mode)
            item = _converted(v, value_hint, mode)
            if key is _NO or item is _NO:
                return _NO
            try:
                out[key] = item
            except TypeError:  # a key that is no key
                return _NO
        return out
    if origin in _LISTS and len(args) == 1:
        # a list's items by its item type, as a tuple's are (an enum's member, a date, a tuple
        # inside it); a tuple a Sequence takes (in a tuple key's record) stays a tuple
        if not isinstance(value, (list, tuple)):
            return _NO
        parts = [_converted(v, args[0], mode) for v in typing.cast(Sequence[object], value)]
        if any(p is _NO for p in parts):
            return _NO
        return parts if isinstance(value, list) else tuple(parts)
    if origin is not None or not isinstance(hint, type):
        return value  # a set[...] or other form: as JSON gives it
    if issubclass(hint, enum.Enum):
        return _member(hint, value, mode)
    if hint is dt.datetime or hint is dt.date:
        return _date(value, hint, mode)
    if hint is bool:
        return value if isinstance(value, bool) else _NO
    if isinstance(value, bool):
        return _NO  # True is an int to Python, never a count or a rate
    if hint is int:
        return value if isinstance(value, int) else _NO
    if hint is float:
        if not isinstance(value, (int, float)):
            return _NO
        return float(value) if mode == "form" else value
    if hint is str:
        return value if isinstance(value, str) else _NO
    if not _checkable(typing.cast(object, hint)):
        return value  # a TypedDict's, say: as JSON gives it
    try:
        return value if isinstance(value, hint) else _NO
    except BaseException:  # a class's own __instancecheck__, for this value: as JSON gives it
        return value


_LISTS = (list, collections.abc.Sequence, collections.abc.MutableSequence)
"""The types a list[...] is declared with whose items :func:`_converted` reads by the item
type."""

_MAPPINGS = (dict, collections.abc.Mapping, collections.abc.MutableMapping)
"""The types a dict[...] is declared with whose keys and values :func:`_converted` reads."""


def _unwrap(hint: object) -> object:
    """A type as a setting's value is read: a ``NewType`` as its supertype, which its values are
    at run time (``Rate(2.0)`` is 2.0), a ``type`` alias as its value (a generic one with its
    arguments given: ``Pair[float]`` of ``type Pair[T] = tuple[T, T]`` is ``tuple[float,
    float]``), and ``Annotated[T, ...]`` as ``T``. An alias whose value cannot be had (one naming
    what its module does not define) stays as it is, a type taken as JSON gives it."""
    for _ in range(64):
        try:
            origin = typing.get_origin(hint)
            if isinstance(hint, typing.NewType):
                hint = hint.__supertype__
            elif isinstance(hint, typing.TypeAliasType):  # type Weights = list[float]
                hint = hint.__value__
            elif isinstance(origin, typing.TypeAliasType):  # Pair[float]
                hint = _given(origin, typing.get_args(hint))
            elif origin is typing.Annotated:
                hint = typing.get_args(hint)[0]
            else:
                break
        except BaseException:  # an alias evaluated lazily, whose value may raise anything
            break
    return hint


def _given(alias: typing.TypeAliasType, args: tuple[object, ...]) -> object:
    """A generic ``type`` alias's value with its type parameters given ``args``, each where the
    value names it (``type Swap[A, B] = dict[B, A]``: ``Swap[str, int]`` is ``dict[int,
    str]``; ``type Id[T] = T``: ``Id[float]`` is ``float``), a ``TypeVarTuple`` the arguments
    the others leave (``type Row[*Ts] = tuple[*Ts]``: ``Row[float, str]`` is ``tuple[float,
    str]``)."""
    value: typing.Any = alias.__value__
    params = alias.__type_params__
    many = [i for i, p in enumerate(params) if isinstance(p, typing.TypeVarTuple)]
    spare = len(args) - len(params) + 1
    if many and spare >= 0:
        i = many[0]
        parts = [(a,) for a in args[:i]] + [args[i : i + spare]] + [(a,) for a in args[i + spare :]]
    else:
        parts = [(a,) for a in args]
    given: dict[object, tuple[object, ...]] = dict(zip(params, parts, strict=True))
    for param, part in given.items():
        if value is param and len(part) == 1:  # type Id[T] = T
            return part[0]
    free: tuple[object, ...] = getattr(value, "__parameters__", ())
    if not free:
        return value
    return value[tuple(x for p in free for x in given[p])]


def _shaped(value: object, hint: object) -> bool:
    """Whether a value from JSON has the shape a member type reads: a list for a tuple or a
    list, an object for a dict; for a union (an alias of one too), one of its member types'."""
    hint = _unwrap(hint)
    if typing.get_origin(hint) in (typing.Union, types.UnionType):
        return any(_shaped(value, a) for a in typing.get_args(hint))
    shape = typing.get_origin(hint) or hint
    if isinstance(value, list):
        return shape is tuple or shape in _LISTS
    return isinstance(value, Mapping) and shape in _MAPPINGS


def _holds(value: object, hint: object) -> bool:
    """Whether a type the GUI takes as JSON gives it holds a JSON value as it is: the value is an
    instance of its class (an ``Iterable`` or a ``Collection`` holds a list; a set holds none)."""
    hint = _unwrap(hint)
    shape = typing.get_origin(hint) or hint
    if not isinstance(shape, type):
        return False
    try:
        return isinstance(value, shape)
    except BaseException:  # a class isinstance cannot check (a TypedDict, a Protocol): it may
        return True


def _reads_part(value: object, hint: object, mode: str) -> bool:
    """Whether a member type of a union reads some of a list or a dict from JSON though not the
    whole: an item, or a key with its value, that fits, whole or in part; one that reads none of
    it leaves the value to a member type that takes it as JSON gives it."""
    hint = _unwrap(hint)
    shape = typing.get_origin(hint) or hint
    args = typing.get_args(hint)
    if isinstance(value, list):
        items = typing.cast(list[object], value)
        if not args:
            return True
        if shape is tuple and len(args) == 2 and args[1] is Ellipsis:
            hints = [args[0]] * len(items)
        elif shape is tuple:
            if len(args) != len(items):
                return False
            hints = list(args)
        else:
            hints = [args[0]] * len(items)
        return any(_part(v, h, mode) for v, h in zip(items, hints, strict=True))
    if isinstance(value, Mapping):
        key_hint, value_hint = (*args, object, object)[:2]
        return any(
            _key(k, key_hint, mode) is not _NO and _part(v, value_hint, mode)
            for k, v in typing.cast(Mapping[object, object], value).items()
        )
    return False


def _part(value: object, hint: object, mode: str) -> bool:
    """Whether an item of a list or a dict fits its type whole or, a list or a dict, in part."""
    return _converted(value, hint, mode) is not _NO or _reads_some(value, hint, mode)


def _reads_some(value: object, hint: object, mode: str) -> bool:
    """Whether a type reads part of a list or a dict from JSON, though not the whole: by its own
    shape, or by a member type's of a union (an alias of one too) that has the value's shape."""
    hint = _unwrap(hint)
    if typing.get_origin(hint) in (typing.Union, types.UnionType):
        return any(_reads_some(value, a, mode) for a in typing.get_args(hint))
    return _shaped(value, hint) and _reads_part(value, hint, mode)


def _read_by_type(hint: object) -> bool:
    """Whether :func:`_converted` reads a value by ``hint`` rather than taking it as JSON gives
    it, as it takes a ``set[...]`` or a type it has no reading for."""
    hint = _unwrap(hint)
    origin = typing.get_origin(hint)
    if origin is not None:
        return origin in (typing.Union, types.UnionType, typing.Literal, tuple, *_MAPPINGS, *_LISTS)
    if hint is typing.Any or hint is None:
        return True
    return isinstance(hint, type) and _checkable(hint)


def _checkable(cls: object) -> bool:
    """Whether ``isinstance`` can check a class: not a ``TypedDict``'s, nor a ``Protocol``'s
    that is not ``runtime_checkable``, whose values the GUI takes as JSON gives them."""
    try:
        isinstance(None, typing.cast(type, cls))
    except BaseException:  # a class's own __instancecheck__, the user's, may raise anything
        return False
    return True


def _hashed(value: object) -> object:
    """A value read for a dict's key, its lists made tuples at any depth, as a key holds a
    sequence (a ``Sequence[int]`` in a key is a tuple); ``_NO`` for one that cannot be a key."""
    if value is _NO:
        return _NO

    def keyed(v: object) -> object:
        if isinstance(v, list) or type(v) is tuple:
            return tuple(keyed(x) for x in typing.cast(Sequence[object], v))
        return v

    key = keyed(value)
    try:
        hash(key)
    except Exception:
        return _NO
    return key


def _tupled(value: object) -> object:
    if isinstance(value, list):
        return tuple(_tupled(v) for v in typing.cast(list[object], value))
    return value


def _key(text: object, hint: object, mode: str) -> object:
    """A dict's key from JSON, whose keys are texts, read by the dict's key type, or ``_NO``: in
    a form's JSON as the form writes it (:func:`_key_text`: an enum's member by its name, a date
    in ISO form, a number as JSON writes it), and in a manifest's as the manifest records it,
    ``str()`` of the key, which must give the same text back (``Basis.BEST``, ``2027-01-01``,
    ``(1, 2)``)."""
    if not isinstance(text, str):
        return _converted(text, hint, mode)
    hint = _unwrap(hint)
    if hint is typing.Any or hint is object or hint is None or hint is str:
        return text
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)
    if origin in (typing.Union, types.UnionType):
        for option in args:
            found = _key(text, option, mode)
            if found is not _NO:
                return found
        return _NO
    shown = _key_text if mode == "form" else str
    if hint is type(None):
        return None if text == shown(None) else _NO
    found: object = _NO
    if origin is typing.Literal:
        if mode == "form":  # an enum's member as the enum's own key takes it
            found = _literal_member(
                args, text, lambda a: _shown(_key_text, a) == text, _scalar(text)
            )
        else:
            found = next((a for a in args if _shown(shown, a) == text), _NO)
    elif hint is bool:
        found = {"true": True, "false": False, "True": True, "False": False}.get(text, _NO)
    elif hint is int or hint is float:
        with contextlib.suppress(ValueError):
            found = int(text) if hint is int else float(text)
    elif isinstance(hint, type) and issubclass(hint, enum.Enum):
        if mode == "form":  # by its name, a form Python's Enum writes, or its value (2: HIGH)
            found = _member(hint, text, "form")
            number = _scalar(text)
            return _member(hint, number, "form") if found is _NO and number is not _NO else found
        found = _stated_member(hint, text)
    elif hint is dt.date or hint is dt.datetime:
        kinds = (dt.date, dt.datetime) if hint is dt.date else (dt.datetime,)
        for kind in kinds:
            with contextlib.suppress(ValueError):
                found = kind.fromisoformat(text)
                break
    elif (origin or hint) is tuple or (origin or hint) in _LISTS:
        # a tuple, or a sequence, which a key holds as a tuple; written as a list in a form, and
        # as str() of the tuple in a manifest
        if mode == "form":
            with contextlib.suppress(ValueError, TypeError, RecursionError):
                found = _hashed(_converted(json.loads(text), typing.cast(object, hint), mode))
        else:
            if (origin or hint) is tuple:
                found = _stated_tuple(text, typing.cast(object, hint))
            else:
                item = typing.cast(object, args[0] if args else object)
                found = _stated_tuple(text, typing.cast(object, tuple[item, ...]))
    if found is _NO or mode == "form":  # a form's key as typed: 1.50 is 1.5
        return found
    return found if _shown(shown, found) == text else _NO  # a manifest's: the same text back


def _literal_member(
    args: tuple[object, ...],
    value: object,
    written: typing.Callable[[object], bool],
    also: object = _NO,
) -> object:
    """The ``Literal`` value a form's ``value`` is, as an enum's own field reads its members:
    first one ``written`` as it is, or an enum's member by its name or a form Python's Enum
    writes, whichever comes first; then an enum's member that the enum's field reads ``value``
    (or ``also``) as, by its value or, a flag's, by its names in any order. ``_NO`` for none."""
    for a in args:
        if value in _member_forms(a) if isinstance(a, enum.Enum) else written(a):
            return a
    for a in args:
        if isinstance(a, enum.Enum):
            for typed in (value, also):
                if typed is not _NO and _member(type(a), typed, "form") is a:
                    return a
    return _NO


def _scalar(text: str) -> object:
    """A dict's key typed as a number, ``true``, ``false`` or ``null``, as JSON reads it (an int
    enum's member by its value, as ``str()`` writes it): ``_NO`` for any other text."""
    try:
        found = json.loads(text)
    except (ValueError, RecursionError):
        return _NO
    return found if found is None or isinstance(found, (bool, int, float)) else _NO


def _member(enum_cls: type[enum.Enum], value: object, mode: str) -> object:
    """An enum's member from a form's text (its name, repr, str or value) or from a manifest's
    record of it (whatever ``_record`` gives for it: the repr of a plain enum's member, the value
    of an int or str enum's, a combined flag's repr or value)."""
    members = list(enum_cls.__members__.values())
    if mode == "form":
        # by its name, or by the forms Python's Enum itself writes (``Basis.BEST``,
        # ``<Basis.BEST: 'best'>``), or by its value (below), never a text of the class's own
        # __str__ or __repr__ (a label), which another member type of a union would read as the
        # text it is
        for member in members:
            if value in _member_forms(member):
                return member
        if issubclass(enum_cls, enum.Flag) and isinstance(value, str):
            # R|W, or Perm.R|W as str() writes it
            text = value.removeprefix(f"{enum_cls.__name__}.")
            names = [n.strip() for n in text.split("|") if n.strip()]
            if all(n in enum_cls.__members__ for n in names):
                found = _combined(enum_cls, names)
                if found is not _NO:
                    return found
    want = _record(value)
    if mode == "record":
        for member in members:
            if _recorded(member) == want:
                return member
    tries: list[object] = [value]
    if isinstance(value, str):
        match = _ENUM_REPR.fullmatch(value)
        if match is not None:
            found = _literal(match.group(1))
            if found is not _NO:
                tries.append(found)
    for attempt in tries:
        try:
            member = enum_cls(attempt)
        except BaseException:  # the enum's own _missing_, the user's code, may raise anything
            continue
        # a member by its value, not one the enum's own _missing_ gives for anything (a default
        # member), which another option of a union would read as the value it is
        if mode == "form" and _recorded(member.value) == _record(attempt):
            return member
        if mode == "record" and _recorded(member) == want:
            return member
    return _NO


def _member_forms(member: enum.Enum) -> tuple[str, ...]:
    """The texts a form takes for an enum's member: its name (:func:`_member_text`), and the
    forms Python's Enum itself writes for it (``Basis.BEST``, ``<Basis.BEST: 'best'>``)."""
    cls_name = type(member).__name__
    name = _member_text(member)
    return (name, f"{cls_name}.{name}", f"<{cls_name}.{name}: {_shown(repr, member.value)}>")


def _recorded(value: object) -> str | None:
    """:func:`_record` of a member or its value, or None where the class's own ``__repr__``
    raises (a member's, even one no setting uses)."""
    try:
        return _record(value)
    except BaseException:
        return None


def _stated_member(
    enum_cls: type[enum.Enum], text: str, shown: typing.Callable[[object], str] = str
) -> object:
    """An enum's member, or a flag's members combined, whose ``shown`` text is ``text``: its
    ``str()``, as a manifest records a dict's key (``Basis.BEST``, ``Perm.R|W``, ``Perm(0)``, an
    int enum's ``1``), or its ``repr()``, as ``str()`` of a tuple writes its items; or ``_NO``."""
    for member in enum_cls.__members__.values():
        if _shown(shown, member) == text:
            return member
    tries: list[object] = []
    prefix = f"{enum_cls.__name__}."
    names = text[len(prefix) :].split("|") if text.startswith(prefix) else []
    if issubclass(enum_cls, enum.Flag) and names and all(n in enum_cls.__members__ for n in names):
        tries.append(_combined(enum_cls, names))
    own = re.compile(rf"{re.escape(enum_cls.__name__)}\((.*)\)")
    for pattern in (_ENUM_REPR, own, _WHOLE_NUMBER):
        match = pattern.fullmatch(text)
        if match is not None:
            # the enum's own _missing_, the user's code, may raise anything
            with contextlib.suppress(BaseException):
                tries.append(enum_cls(ast.literal_eval(match.group(1))))
    for found in tries:
        if found is not _NO and _shown(shown, found) == text:
            return found
    return _NO


def _combined(flag: type[enum.Flag], names: list[str]) -> object:
    """A flag's members of those names, combined as ``|`` combines them, which a flag's own
    ``_missing_`` may take where it refuses 0, the value a combination could start from; its
    empty value for no name; ``_NO`` when the flag refuses either."""
    try:
        if not names:
            return flag(0)
        members = [flag.__members__[n] for n in names]
        return functools.reduce(lambda a, b: a | b, members)
    except BaseException:  # the flag's own _missing_, the user's code
        return _NO


def _shown(write: typing.Callable[[object], str], value: object) -> str | None:
    """``write(value)`` (``repr`` or ``str`` of a member), or None where the user's own
    ``__repr__`` or ``__str__`` raises."""
    try:
        return str.__str__(write(value))
    except BaseException:
        return None


_WHOLE_NUMBER = re.compile(r"(-?[0-9]+)")


def _stated_tuple(text: str, hint: object) -> object:
    """A tuple from its ``str()``, as a manifest records a dict's key (``(<Sex.F: 'female'>,
    40)``), each item read from its repr by the tuple's item types; ``_NO`` when it does not
    fit."""
    items = _split_items(text)
    args = typing.get_args(hint)
    if items is None:
        return _NO
    if len(args) == 2 and args[1] is Ellipsis:
        hints: list[object] = [args[0]] * len(items)
    elif not args:
        hints = [object] * len(items)
    elif len(args) == len(items):
        hints = list(args)
    else:
        return _NO
    parts = [_from_repr(t, h) for t, h in zip(items, hints, strict=True)]
    return _NO if any(p is _NO for p in parts) else tuple(parts)


def _split_items(text: str) -> list[str] | None:
    """The items of a tuple's ``str()``, split at its top-level commas; None for another text."""
    if not (text.startswith("(") and text.endswith(")")):
        return None
    inner = text[1:-1]
    items: list[str] = []
    depth, start, quote, i = 0, 0, "", 0
    while i < len(inner):
        c = inner[i]
        if quote:
            if c == "\\":
                i += 1  # an escaped character, a quote too
            elif c == quote:
                quote = ""
        elif c in "'\"":
            quote = c
        elif c in "([{<":
            depth += 1
        elif c in ")]}>":
            depth -= 1
        elif c == "," and depth == 0:
            items.append(inner[start:i].strip())
            start = i + 1
        i += 1
    tail = inner[start:].strip()
    if tail:
        items.append(tail)
    return items


def _from_repr(text: str, hint: object) -> object:
    """A value from its repr, as ``str()`` of a tuple writes its items, read by its type: an
    enum's member, a date, a tuple (a sequence too, which a key holds as a tuple), a
    ``Literal``'s member, or a literal; ``_NO`` when it does not fit."""
    hint = _unwrap(hint)
    origin = typing.get_origin(hint)
    if origin in (typing.Union, types.UnionType):
        for option in typing.get_args(hint):
            found = _from_repr(text, option)
            if found is not _NO:
                return found
        return _NO
    if origin is typing.Literal:
        return next((a for a in typing.get_args(hint) if _shown(repr, a) == text), _NO)
    if isinstance(hint, type) and issubclass(hint, enum.Enum):
        return _stated_member(hint, text, repr)
    if hint is dt.date or hint is dt.datetime:
        return _date(text, hint, "record")
    if (origin or hint) is tuple:
        return _stated_tuple(text, typing.cast(object, hint))
    if (origin or hint) in _LISTS:
        args = typing.get_args(hint)
        item = typing.cast(object, args[0] if args else object)
        return _stated_tuple(text, typing.cast(object, tuple[item, ...]))
    value = _literal(text)
    return _NO if value is _NO else _converted(value, typing.cast(object, hint), "record")


class _NonFinite(ast.NodeTransformer):
    """``inf`` and ``nan`` read as the floats whose repr they are, which no literal writes."""

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in ("inf", "nan"):
            return ast.copy_location(ast.Constant(float(node.id)), node)
        return node


def _literal(text: str) -> object:
    """A value from its repr, as ``ast.literal_eval`` reads one, with a float that is not finite
    (``inf``, ``-inf``, ``nan``) read at any depth (``((inf, 1), 2)``); ``_NO`` for another
    text."""
    try:
        tree = _NonFinite().visit(ast.parse(text.lstrip(" \t"), mode="eval"))
        return ast.literal_eval(tree)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return _NO


def _date(value: object, hint: type, mode: str) -> object:
    """A date (or a date and time) from ISO text, and from a manifest also from its repr. A
    ``Static[date]`` takes a date and time too, as ``build()`` does."""
    if isinstance(value, dt.date):
        return value if isinstance(value, hint) else _NO
    if not isinstance(value, str):
        return _NO
    text = value.strip()
    if mode == "record":
        # a manifest records a date by its repr; a text in its place was a text when the run
        # was made, which build() took inside a union or a tuple
        found = _date_repr(text)
        return found if found is not _NO and isinstance(found, hint) else _NO
    if hint is dt.date:
        try:
            return dt.date.fromisoformat(text)
        except ValueError:
            pass
    try:
        return dt.datetime.fromisoformat(text)
    except ValueError:
        return _NO


def _date_repr(text: str) -> object:
    """A date, or a date and time, from its repr as a manifest records one, with its time zone
    when it has one of the ``datetime`` module's own (``datetime.timezone.utc``, or an offset,
    ``datetime.timezone(datetime.timedelta(seconds=3600))``, as ``fromisoformat`` makes one from
    ``+01:00``); ``_NO`` for any other text."""
    try:
        found = _made(ast.parse(text, mode="eval").body)
    except (ValueError, TypeError, SyntaxError, OverflowError, MemoryError, RecursionError):
        return _NO
    return found if type(found) in (dt.date, dt.datetime) else _NO


_DATE_MAKERS: dict[str, Callable[..., object]] = {
    "datetime.date": dt.date,
    "datetime.datetime": dt.datetime,
    "datetime.timedelta": dt.timedelta,
    "datetime.timezone": dt.timezone,
}


def _made(node: ast.expr) -> object:
    """What a repr's call of the ``datetime`` module's classes makes (a date, a time, a time
    zone, an offset), its arguments whole numbers, texts and such calls; ValueError for anything
    else."""
    if isinstance(node, ast.Constant) and type(node.value) in (int, str):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _made(node.operand)
        if type(value) is int:
            return -value
    if _dotted(node) == "datetime.timezone.utc":
        return dt.UTC
    if isinstance(node, ast.Call):
        maker = _DATE_MAKERS.get(_dotted(node.func) or "")
        if maker is not None and all(k.arg is not None for k in node.keywords):
            args = [_made(a) for a in node.args]
            return maker(*args, **{str(k.arg): _made(k.value) for k in node.keywords})
    raise ValueError("not a repr of the datetime module's")


def _dotted(node: ast.expr) -> str | None:
    """A name with its dots (``datetime.timezone.utc``); None for another expression."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        head = _dotted(node.value)
        return None if head is None else f"{head}.{node.attr}"
    return None


def _input_of(hint: object) -> str:
    hint = _unwrap(hint)
    if hint is bool:
        return "bool"
    if hint is int:
        return "int"
    if hint is float:
        return "float"
    if hint is str:
        return "str"
    if isinstance(hint, type) and issubclass(hint, enum.Flag):
        return "flags"
    if isinstance(hint, type) and issubclass(hint, enum.Enum):
        return "choice"
    if hint is dt.datetime:
        return "datetime"
    if hint is dt.date:
        return "date"
    return "json"


def form_text(hint: object, value: object) -> str:
    """A build-time value as its form field shows it, in the form :func:`read_static` reads back
    as the same value: a number as Python writes it (``1`` and ``1.0`` apart), an enum's member
    by its name, a date in ISO form, a check box's ``true`` or ``false``, and any other value as
    JSON, a tuple as a list, with an enum's member by its name and a date in ISO form inside it.
    A value JSON cannot hold shows as its repr, which reads back only as text."""
    kind = _input_of(hint)
    if kind == "bool":
        return "true" if value is True else "false"
    if value is None:
        return "null" if kind == "json" else ""
    if kind in ("choice", "flags") and isinstance(value, enum.Enum):
        return _member_text(value)
    if kind in ("date", "datetime") and isinstance(value, dt.date):
        return value.isoformat()
    if kind == "str" and isinstance(value, str):
        return value
    if kind in ("int", "float") and isinstance(value, (int, float)) and not isinstance(value, bool):
        return repr(value)
    try:
        return json.dumps(_listed(value), ensure_ascii=False)
    except (TypeError, ValueError):
        return repr(value)


def _member_text(member: enum.Enum) -> str:
    """An enum's member by its name; a flag's combined members by their names joined by ``|``,
    and its empty value as an empty text."""
    name = typing.cast(str | None, member.name)  # None for a flag's empty or combined value
    if name is not None:
        return name
    flag = typing.cast(enum.Flag, member)
    found = [m.name for m in type(flag) if m in flag]
    return "|".join(n for n in found if n is not None)


def _listed(value: object) -> object:
    if isinstance(value, enum.Enum):
        return _member_text(value)
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [_listed(v) for v in typing.cast(list[object] | tuple[object, ...], value)]
    if isinstance(value, Mapping):
        items = typing.cast(Mapping[object, object], value).items()
        return {_key_text(k): _listed(v) for k, v in items}
    return value


def _key_text(key: object) -> str:
    """A dict's key as the form writes it, a JSON object's keys being texts: a text as it is, an
    enum's member by its name, a date in ISO form, and any other value as JSON writes it (``1``,
    ``1.5``, ``true``, ``null``, a tuple as a list), or as ``str()`` gives it when JSON cannot."""
    shown = _listed(key)
    if isinstance(shown, str):
        return shown
    try:
        return json.dumps(shown, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(key)


def read_static(param: StaticParam, value: object) -> object:
    """A build-time setting as the build form sends it, read by the type its parameter is
    declared with: a ``Static[float]`` typed ``2`` is 2.0, as the field is a float's; an enum's
    member by its name; a date in ISO form; any other type as JSON, read through the type
    (:func:`_converted`): a list as a tuple where the type is one, an enum's member by its name
    and a date in ISO form inside it. A ValueError says what the field holds instead; a value
    that does not fit a JSON field's type is left for ``build()`` to refuse, at the class."""
    name, kind = param.name, param.input
    if kind == "bool":
        if isinstance(value, bool):
            return value
        if value in ("true", "false"):
            return value == "true"
        raise ValueError(f"{name} is true or false, not {value!r}")
    if not isinstance(value, str):
        return value
    text = value.strip()
    if kind == "int":
        try:
            return int(text)
        except ValueError:
            raise ValueError(f"{name} is a whole number, not {value!r}") from None
    if kind == "float":
        try:
            return float(text)
        except ValueError:
            raise ValueError(f"{name} is a number, not {value!r}") from None
    if kind == "str":
        return value
    if kind in ("choice", "flags", "date", "datetime"):
        found = _converted(text, param.hint, "form")
        if found is not _NO:
            return found
        if kind in ("choice", "flags"):
            names = ", ".join(typing.cast(type[enum.Enum], _unwrap(param.hint)).__members__)
            if kind == "flags":
                raise ValueError(f"{name} is some of {names}, joined by |, not {value!r}")
            raise ValueError(f"{name} is one of {names}, not {value!r}")
        example = "2027-01-01T09:00" if kind == "datetime" else "2027-01-01"
        raise ValueError(f"{name} is a date such as {example}, not {value!r}")
    if text == "":
        return None
    try:
        parsed: object = json.loads(text)
    except json.JSONDecodeError:
        return value  # a bare word: build() says what the parameter takes, at the class
    found = _converted(parsed, param.hint, "form")
    # a value that does not fit its type item by item is still a tuple where the type is one:
    # the form writes a tuple as a list, and a rerun reads the record back so
    return _as_recorded(parsed, param.hint, "form") if found is _NO else found


def describe_class(entry: ModelEntry) -> dict[str, object]:
    """What a model class declares, for the page that shows it before any build: build-time
    parameters, the model point schema, the tables, scenario inputs and rules it reads, its
    invariants and outputs and, for a stage, its bases and series."""
    cls = entry.cls
    schema = cls.model_point
    out: dict[str, object] = {
        **entry.summary(),
        "statics": [p.as_dict() for p in statics_of(cls)],
        "schema": None if schema is None else _schema(schema),
        "tables": [_table(t) for t in cls.assumption_tables],
        "scenarios": [_scenario(s) for s in cls.scenario_inputs],
        "input_rules": [{"name": r.name, "doc": r.doc} for r in cls.input_rules],
        "invariants": [_invariant(i) for i in cls.invariants],
        "outputs": None if cls.outputs is None else list(cls.outputs),
        "tolerance_scales": dict(cls.tolerance_scales),
        "valuation_date": default_valuation_date(cls),
    }
    if entry.stage:
        out["bases"] = [_declared(b) for b in getattr(cls, "bases", ())]
        out["series"] = [_declared(s) for s in getattr(cls, "series", ())]
        out["member_rules"] = [_declared(r) for r in getattr(cls, "member_rules", ())]
    return out


def default_valuation_date(cls: type[Product]) -> str | None:
    """The valuation date the model's module names as ``VALUATION_DATE``, the reference
    products' convention, as an ISO date."""
    module = sys.modules.get(cls.__module__)
    value = _attr(module, "VALUATION_DATE")  # an object in the module's place may raise
    return typing.cast(dt.date, value).isoformat() if _is(value, dt.date) else None


def plain(value: object) -> object:
    """A build-time setting or contract value as JSON can hold it: numbers, strings, booleans
    and None as they are, lists and tuples item by item, anything else as its repr."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in typing.cast(list[object] | tuple[object, ...], value)]
    if isinstance(value, Mapping):
        items = typing.cast(Mapping[object, object], value).items()
        return {str(k): plain(v) for k, v in items}
    if isinstance(value, dt.date):
        return value.isoformat()
    return repr(value)


def _schema(schema: object) -> dict[str, object]:
    s: typing.Any = schema
    return {
        "id_field": s.id_field,
        "horizon": s.horizon,
        "valuation_date_rule": s.valuation_date_rule is not None,
        "fields": [
            {
                "name": f.name,
                "kind": f.kind,
                "unit": f.unit,
                "enum": None if f.enum is None else list(f.enum),
                "gt": f.gt,
                "ge": f.ge,
                "lt": f.lt,
                "le": f.le,
                "unique": f.unique,
                "doc": f.doc,
            }
            for f in s.fields
        ],
        "derived": [
            {"name": d.name, "kind": d.kind, "unit": d.unit, "ge": d.ge, "le": d.le, "doc": d.doc}
            for d in s.derived
        ],
        "rules": [{"name": r.name, "doc": r.doc} for r in s.rules],
    }


def _table(t: typing.Any) -> dict[str, object]:
    return {
        "name": t.name,
        "doc": t.doc,
        "keys": [
            {
                "name": k.name,
                "enum": None if k.enum is None else list(k.enum),
                "lo": k.lo,
                "hi": k.hi,
            }
            for k in t.keys
        ],
        "columns": [{"name": c.name, "unit": c.unit, "ge": c.ge, "le": c.le} for c in t.columns],
        "metadata": list(t.metadata),
        "rules": [{"name": r.name, "doc": r.doc} for r in t.rules],
    }


def _scenario(s: typing.Any) -> dict[str, object]:
    return {
        "name": s.name,
        "unit": s.unit,
        "doc": s.doc,
        "column": s.file_column,
        "covers": None if s.covers is None else list(s.covers),
        "gt": s.gt,
        "ge": s.ge,
        "lt": s.lt,
        "le": s.le,
        "rules": [{"name": r.name, "doc": r.doc} for r in s.rules],
    }


def _invariant(i: typing.Any) -> dict[str, object]:
    loc = i.loc
    return {
        "name": i.name,
        "kind": i.kind,
        "tol": i.tol,
        "scale": i.scale if isinstance(i.scale, (int, float)) else "computed",
        "months": None if i.months is None else list(i.months),
        "reads": list(i.reads),
        "doc": i.doc,
        "loc": None
        if loc is None
        else {"file": loc.file, "line": loc.line, "function": loc.function},
    }


def _declared(value: object) -> dict[str, object]:
    """A stage's declaration (a basis, a series, a member rule) by its plain attributes."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        items = {f.name: getattr(value, f.name) for f in dataclasses.fields(value)}
    else:
        items = {k: v for k, v in vars(value).items() if not k.startswith("_")}
    return {k: plain(v) for k, v in items.items() if not callable(v) and k != "loc"}


def _model_id(cls: type) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


def _source_path(cls: type) -> Path | None:
    try:
        file = inspect.getsourcefile(cls)
    except BaseException:  # an odd class, a file being edited, or an object in its module's place
        return None
    return None if file is None else Path(file)


def _class_line(cls: type) -> int | None:
    try:
        return inspect.getsourcelines(cls)[1]
    except BaseException:  # Python 3.12's inspect parses the file, which may be saved part-way
        return None


def _first_paragraph(text: str) -> str:
    return text.strip().split("\n\n", 1)[0].replace("\n", " ").strip()
