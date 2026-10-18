"""The aggregation stage's language: ``Aggregation``, ``Basis`` and ``GroupSum`` (design 9).

An aggregation stage is written in the same ``@var`` language as a projection and checked by the
same checker (open question 15): an :class:`Aggregation` is a ``Product`` whose model points are
groups and whose extra inputs are group series. Each series is a :class:`GroupSum`: one or more
output variables of the projection runs of a :class:`Basis`, summed over each group's members in
ascending ``policy_id`` across the basis's runs, sequentially (D19), then added left to right
per scenario, then either averaged over the scenarios, ``cumsum_s((1.0 / J) * x)[-1]`` in
ascending ``s`` (para 33(a)), or kept per scenario. A formula reads a series as
``self.series.<name>(t + k)``, ``self.series.<name>.at(k)`` or, without a time dimension,
``self.series.<name>()``; ``self.group.<field>`` is the group's own field.

:func:`group_series` builds the dense series from projection runs of this session, so a stage
can be run end to end through :func:`~halley.runtime.run.run`. The loader that reads the bases
from run directories, with their copies, previous runs and membership checks, comes later
(design section 9, "Loader and runs").
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field
from typing import Any, ClassVar, TypeGuard

import numpy as np
from numpy.typing import NDArray

from halley import ir
from halley.authoring.product import ModelPoint, Product, VarDef
from halley.errors import BuildError, LoadError, Loc
from halley.ir.units import Unit, unit_from_name
from halley.runtime.loader import LoadedSeries
from halley.runtime.run import RunResult

SCENARIO_TREATMENTS = ("mean", "each")
"""What a series does with its basis's scenarios: their mean (one scenario) or each of them."""

_STAGE_NAMES = {
    "group": "the group's fields (self.group)",
    "series": "the group series accessor (self.series)",
    "bases": "a stage class attribute",
}
"""Names a stage's variable may not take, beside the projection's own reserved names."""


def sequential_sum(values: NDArray[Any], axis: int) -> NDArray[Any]:
    """Sum along an axis sequentially, in index order, keeping the axis with length 1 (D19:
    a fixed-order reduction, not pairwise)."""
    return np.cumsum(values, axis=axis).take([-1], axis=axis)


_NOT_MODEL_SOURCE = frozenset({__file__, dataclasses.__file__})
"""This module and the one whose ``replace()`` re-runs a declaration's checks."""


def _declaration_loc() -> Loc | None:
    """The model source line that constructs a declaration: the nearest frame outside this
    module, outside ``dataclasses`` (a ``replace()`` of a declaration) and outside generated
    code (a dataclass's ``__init__``)."""
    frame = inspect.currentframe()
    while frame is not None:
        name = frame.f_code.co_filename
        if name not in _NOT_MODEL_SOURCE and not name.startswith("<"):
            return Loc(name, frame.f_lineno, frame.f_code.co_qualname)
        frame = frame.f_back
    return None


def _class_loc(cls: type) -> Loc | None:
    try:
        file = inspect.getsourcefile(cls)
        _lines, line = inspect.getsourcelines(cls)
    except (OSError, TypeError):
        return None
    return None if file is None else Loc(file, line, cls.__qualname__)


# ------------------------------------------------------------------------------ declarations


@dataclass(frozen=True, slots=True)
class Basis:
    """A basis of a stage: one or more projection runs with the same valuation date, horizon and
    number of scenarios J, and pairwise disjoint policy ids (one run per product class), whose
    outputs the stage's :class:`GroupSum` series sum."""

    name: str
    doc: str = ""
    _: KW_ONLY
    loc: Loc | None = field(default=None, compare=False)
    """Where the basis is declared; filled in when left out."""

    def __post_init__(self) -> None:
        if self.loc is None:
            object.__setattr__(self, "loc", _declaration_loc())
        if not self.name.isidentifier():
            raise BuildError(f"a basis is named like a variable, not {self.name!r}", self.loc)

    def check(self, runs: Sequence[RunResult]) -> None:
        """The runs given for this basis agree on the valuation date, the horizon and J, and
        hold disjoint policies. A LoadError otherwise."""
        if not runs:
            raise LoadError(f"basis {self.name} has no runs")
        first = runs[0]
        when = _valuation_date(first)
        for r in runs[1:]:
            if _valuation_date(r) != when:
                raise LoadError(
                    f"basis {self.name}: run {r.run_id} values at {_valuation_date(r)} but run "
                    f"{first.run_id} at {when}; the runs of a basis share one valuation date"
                )
            if r.H != first.H:
                raise LoadError(
                    f"basis {self.name}: run {r.run_id} has horizon {r.H} but run "
                    f"{first.run_id} has {first.H}; give every run of a basis the same "
                    "RunSettings.horizon"
                )
            if r.n_scenarios != first.n_scenarios:
                raise LoadError(
                    f"basis {self.name}: run {r.run_id} has J = {r.n_scenarios} scenarios but "
                    f"run {first.run_id} has J = {first.n_scenarios}; the runs of a basis share "
                    "their scenarios"
                )
        seen: dict[object, str] = {}
        for r in runs:
            for pid in r.ids.tolist():
                if pid in seen:
                    raise LoadError(
                        f"basis {self.name}: policy {pid} is in run {seen[pid]} and in run "
                        f"{r.run_id}; the runs of a basis hold disjoint policies (one run per "
                        "product class)"
                    )
                seen[pid] = r.run_id


def _valuation_date(r: RunResult) -> str:
    return str(r.manifest["settings"]["valuation_date"])


@dataclass(frozen=True, slots=True)
class GroupSum:
    """A group series: each named output of every run of ``basis`` summed over the group's
    members in ascending ``policy_id`` across the runs, sequentially (D19); several variables'
    group totals added left to right per scenario (``("bel", "ra")`` gives BEL_g + RA_g); then
    ``scenarios="mean"`` reduces over the scenarios as ``cumsum_s((1.0 / J) * x)[-1]`` in
    ascending ``s``, giving one scenario, and ``"each"`` keeps them.

    ``unit`` must be the unit every source run recorded for every variable (D11 across runs).
    ``time`` says whether the variables have a time dimension; ``start`` is the first month the
    stage reads, 1 for a sum of month quantities and 0 for a sum of states (a series read at
    ``t - 1`` from month 1, or at month 0); the source runs' domains must start no later."""

    name: str
    basis: str
    variables: str | tuple[str, ...]
    unit: str
    scenarios: str = "mean"
    _: KW_ONLY
    time: bool = True
    start: int = 1
    doc: str = ""
    loc: Loc | None = field(default=None, compare=False)
    """Where the series is declared; filled in when left out."""

    def __post_init__(self) -> None:
        if self.loc is None:
            object.__setattr__(self, "loc", _declaration_loc())
        what = f"series {self.name}"
        if not self.name.isidentifier():
            raise BuildError(f"a series is named like a variable, not {self.name!r}", self.loc)
        if self.scenarios not in SCENARIO_TREATMENTS:
            raise BuildError(
                f"{what}: scenarios is 'mean' or 'each', not {self.scenarios!r}", self.loc
            )
        names: object = (self.variables,) if isinstance(self.variables, str) else self.variables
        if not _is_tuple(names) or not names or not all(isinstance(n, str) and n for n in names):
            raise BuildError(f"{what} names its variables, as 'bel' or ('bel', 'ra')", self.loc)
        if self.start not in (0, 1):
            raise BuildError(
                f"{what}: start is 0 (a sum of states) or 1 (a sum of month quantities), not "
                f"{self.start!r}",
                self.loc,
            )
        try:
            unit_from_name(self.unit)
        except ValueError as exc:
            raise BuildError(f"{what}: {exc}", self.loc) from exc

    @property
    def names(self) -> tuple[str, ...]:
        """The variables summed, in the order their totals are added."""
        return (self.variables,) if isinstance(self.variables, str) else self.variables

    @property
    def declared_unit(self) -> Unit:
        return unit_from_name(self.unit)

    def decl(self) -> ir.SeriesDecl:
        """The IR declaration: dimensions and domain fixed here, at build time, never from J."""
        start = self.start if self.time else 0
        return ir.SeriesDecl(
            self.name, self.declared_unit, self.time, start, self.scenarios == "each", self.doc
        )


# ------------------------------------------------------------------------------------ stage


class Aggregation(Product):
    """Base class of an aggregation stage (design section 9).

    Beside a ``Product``'s class attributes a stage sets:

    * ``model_point``: the schema of its groups, one row per group (``self.group`` reads a
      field, the same as ``self.mp``);
    * ``bases``: the :class:`Basis` declarations whose runs it reads;
    * ``series``: the :class:`GroupSum` declarations, read in formulas as
      ``self.series.<name>(t + k)``, ``self.series.<name>.at(k)`` or ``self.series.<name>()``;
    * ``scenario_inputs``: its own curves, such as a locked-in discount curve.

    The class attribute ``series`` holds the declarations; inside a formula ``self.series`` is
    the accessor that turns them into reads. A series' unit, time dimension, domain and
    scenario treatment are declared, so the checker checks every read at build time: a read of
    an undeclared series, or a series of an undeclared basis, is a BuildError at the model line.
    """

    abstract = True
    bases: ClassVar[tuple[Basis, ...]] = ()
    series: ClassVar[Any] = ()
    """A tuple of :class:`GroupSum`. Typed ``Any`` because a formula reads ``self.series`` as
    the accessor of the same name, which a tuple's type would refuse."""

    @property
    def group(self) -> ModelPoint:
        """``self.group``: the group's fields, rendered ``mp.<field>`` as ``self.mp`` is."""
        return self.mp

    @classmethod
    def series_declarations(cls) -> dict[str, ir.SeriesDecl]:
        """The stage's group series by name, each checked against the bases: a series of an
        undeclared basis, a name declared twice, a malformed tuple or a variable taking one of
        the stage's names is a BuildError at the model source."""
        for klass in cls.__mro__:
            for name, what in _STAGE_NAMES.items():
                value = vars(klass).get(name)
                if isinstance(value, VarDef):
                    raise BuildError(f"{name} is {what} in a stage; rename the variable", value.loc)
        bases = _bases(cls)
        declared: dict[str, Basis] = {}
        for b in bases:
            if b.name in declared:
                raise BuildError(f"{cls.__qualname__} declares basis {b.name} twice", b.loc)
            declared[b.name] = b
        out: dict[str, ir.SeriesDecl] = {}
        for s in _sums(cls):
            if s.name in out:
                raise BuildError(f"{cls.__qualname__} declares series {s.name} twice", s.loc)
            if s.basis not in declared:
                known = ", ".join(declared) or "none"
                raise BuildError(
                    f"series {s.name} sums basis {s.basis!r}, which {cls.__qualname__} does not "
                    f"declare; it declares: {known}",
                    s.loc,
                )
            out[s.name] = s.decl()
        return out


def _is_tuple(value: object) -> TypeGuard[tuple[object, ...]]:
    return isinstance(value, tuple)


def _tuple_of[T](cls: type[Aggregation], name: str, kind: type[T]) -> tuple[T, ...]:
    """``cls.<name>`` as a tuple of ``kind``; the realistic slip is a one-element tuple without
    its comma, which arrives as the element itself."""
    value: object = getattr(cls, name)
    if _is_tuple(value):
        out = tuple(item for item in value if isinstance(item, kind))
        if len(out) == len(value):
            return out
    raise BuildError(
        f"{cls.__qualname__}.{name} must be a tuple of {kind.__name__} (a one-element tuple "
        "needs a trailing comma)",
        _class_loc(cls),
    )


def _bases(cls: type[Aggregation]) -> tuple[Basis, ...]:
    return _tuple_of(cls, "bases", Basis)


def _sums(cls: type[Aggregation]) -> tuple[GroupSum, ...]:
    return _tuple_of(cls, "series", GroupSum)


# ---------------------------------------------------------------------- in-session builder


def group_series(
    stage: type[Aggregation],
    bases: Mapping[str, Sequence[RunResult]],
    members: Mapping[Any, Sequence[Any]],
) -> dict[str, LoadedSeries]:
    """Every series ``stage`` declares, summed from projection runs of this session.

    ``bases`` gives each declared basis's runs; ``members`` each group's member policy ids, by
    group id (the stage's model point ids). Every policy of every run belongs to exactly one
    group, and every member to some run. Each series is a dense ``(Tn | 1, groups, scenarios)``
    array with the groups in ascending id, as the loader orders the stage's model points, and
    months ``0..H`` of its basis. A variable a series names must be stored by every run of its
    basis with the series' unit; its recorded domain start is the series' (the latest over the
    variables), which the loader requires to be no later than the stage's declared start when
    the stage runs. A LoadError otherwise."""
    stage.series_declarations()  # the declarations are sound before anything is summed
    declared = {b.name: b for b in _bases(stage)}
    for name in declared:
        if name not in bases:
            raise LoadError(f"the stage reads basis {name}, which was not given")
    for name in bases:
        if name not in declared:
            raise LoadError(
                f"basis {name!r} was given but {stage.__qualname__} declares no such basis"
            )
    for name, b in declared.items():
        b.check(bases[name])
    group_ids = _sorted_ids(list(members))
    owner: dict[Any, Any] = {}
    for g in group_ids:
        for pid in members[g]:
            if pid in owner:
                raise LoadError(
                    f"policy {pid} is a member of group {owner[pid]} and of group {g}; a policy "
                    "belongs to one group"
                )
            owner[pid] = g
    found: set[Any] = set()
    for name, runs in bases.items():
        for r in runs:
            for pid in r.ids.tolist():
                if pid not in owner:
                    raise LoadError(f"basis {name} run {r.run_id}: policy {pid} is in no group")
                found.add(pid)
    for pid, g in owner.items():
        if pid not in found:
            raise LoadError(f"group {g}: member {pid} is in no run of the stage's bases")
    return {s.name: _summed(s, bases[s.basis], group_ids, members) for s in _sums(stage)}


def _sorted_ids(ids: list[Any]) -> NDArray[Any]:
    """Ids in ascending order, as the loader sorts model points (a stable argsort)."""
    arr = np.asarray(ids, dtype=object)
    return arr[np.argsort(arr, kind="stable")]


def _summed(
    s: GroupSum, runs: Sequence[RunResult], group_ids: NDArray[Any], members: Mapping[Any, Any]
) -> LoadedSeries:
    start = 0
    for var in s.names:
        for r in runs:
            try:
                info = r.variable_info(var)
            except LoadError as exc:
                raise LoadError(f"basis {s.basis} {exc}") from None
            if unit_from_name(info.unit) != s.declared_unit:
                raise LoadError(
                    f"series {s.name} is declared {s.unit} but basis {s.basis} run {r.run_id} "
                    f"records {var} as {info.unit} (D11 across runs)"
                )
            if info.time != s.time:
                has = "has" if info.time else "has no"
                raise LoadError(
                    f"series {s.name} is declared {'with' if s.time else 'without'} a time "
                    f"dimension but {var} of basis {s.basis} run {r.run_id} {has} one"
                )
            start = max(start, info.start)
    totals = [_group_totals(var, runs, group_ids, members) for var in s.names]
    x = totals[0]
    for other in totals[1:]:
        x = x + other  # the group totals, added left to right per scenario
    n_scen = int(x.shape[2])
    if s.scenarios == "mean":
        x = sequential_sum((1.0 / n_scen) * x, axis=2)  # ascending s; x * 1.0 when J = 1
    provenance: dict[str, Any] = {
        "basis": s.basis,
        "variables": list(s.names),
        "basis_scenarios": n_scen,
        "source_runs": [_source_run(s, r) for r in runs],
    }
    return LoadedSeries(
        s.name,
        np.ascontiguousarray(x),
        s.declared_unit,
        s.time,
        start if s.time else 0,
        group_ids,
        s.scenarios,
        provenance,
    )


def _source_run(s: GroupSum, r: RunResult) -> dict[str, str]:
    """Where a series' values came from: the run, its directory and its manifest's hash."""
    manifest = r.run_dir / "manifest.json"
    try:
        raw = manifest.read_bytes()
    except OSError:
        raise LoadError(
            f"series {s.name}: basis {s.basis} run {r.run_id} has no manifest at {manifest}"
        ) from None
    return {
        "run_id": r.run_id,
        "dir": str(r.run_dir),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _group_totals(
    var: str, runs: Sequence[RunResult], group_ids: NDArray[Any], members: Mapping[Any, Any]
) -> NDArray[np.float64]:
    """``var`` summed over each group's members in ascending ``policy_id`` across the runs,
    sequentially: ``(Tn | 1, groups, scenarios)``. A group with no member in these runs sums to
    zero."""
    arrays = [np.asarray(r.var(var), dtype=np.float64) for r in runs]
    where: dict[Any, tuple[int, int]] = {}
    for i, r in enumerate(runs):
        for j, pid in enumerate(r.ids.tolist()):
            where[pid] = (i, j)
    n_t, _, n_s = arrays[0].shape
    columns: list[NDArray[np.float64]] = []
    for g in group_ids:
        lanes = [
            arrays[where[pid][0]][:, where[pid][1], :]
            for pid in _sorted_ids(list(members[g]))
            if pid in where
        ]
        if lanes:
            columns.append(sequential_sum(np.stack(lanes, axis=1), axis=1))
        else:
            columns.append(np.zeros((n_t, 1, n_s), dtype=np.float64))
    return np.concatenate(columns, axis=1)
