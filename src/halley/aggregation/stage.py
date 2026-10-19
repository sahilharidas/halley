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

:func:`group_series` builds the dense series from projection runs, so a stage can be run end to
end through :func:`~halley.runtime.run.run`. :func:`run_aggregation` is the stage's own entry
point (design section 9, "Loader and runs"): it reads every basis from run directories (or runs
of the session, through their directories), each verified against the hashes its manifest
records; takes the membership from each run's model point copy, grouped on (``portfolio``,
``cohort``, ``profit_bucket``); resolves the previous aggregation run a group names by its
``prev_run_id``; checks G1, G4, the ``groups`` rules, the structural and locked-in hashes and
the provenance of every file that names one; fills the group fields derived from the previous
run; and writes a run directory that :func:`replay_aggregation` re-runs from its own copies.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import inspect
import json
import numbers
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Final, NamedTuple, TypeGuard

import numpy as np
from numpy.typing import NDArray

from halley import ir
from halley.authoring.contracts import (
    FROM_DATES,
    FROM_PREVIOUS,
    FROM_PREVIOUS_RUN,
    GROUP_ID,
    GROUP_KEY,
    LOCKED_IN_FIELDS,
)
from halley.authoring.product import ModelPoint, Product, VarDef
from halley.codegen import CompiledModel
from halley.errors import BuildError, LoadError, Loc, RunError, UnitError
from halley.ir import units
from halley.ir.units import Unit, unit_from_name
from halley.runtime.loader import (
    CUR_BASIS,
    LoadedSeries,
    ReadFile,
    Source,
    column_type,
    file_difference,
    read_copy,
    read_series_copy,
    read_source,
    table_columns,
    table_of,
    with_columns,
)
from halley.runtime.run import (
    RunResult,
    RunSettings,
    StageRecord,
    load_run,
    read_manifest,
    run,
    run_files,
    verify_run,
    write_manifest,
)

SCENARIO_TREATMENTS = ("mean", "each")
"""What a series does with its basis's scenarios: their mean (one scenario) or each of them."""

GROUP_ADDITIVE: Final = frozenset({units.MONEY, units.COUNT, units.COVERAGE_UNITS})
"""Units a group sum adds over a group's members (D11): amounts, numbers of policies and
coverage units. A probability, rate, factor, index level, flag or Int of each member does not
add up to the group's, and a MoneyPerPolicy is per policy by definition. Narrower than
:data:`~halley.ir.units.SUMMABLE`, the units ``sum_t()`` adds over the months of one policy."""

_STAGE_NAMES = {
    "group": "the group's fields (self.group)",
    "series": "the group series accessor (self.series)",
    "bases": "a stage class attribute",
    "disclosures": "a stage class attribute",
}
"""Names a stage's variable may not take, beside the projection's own reserved names."""


def sequential_sum(values: NDArray[Any], axis: int) -> NDArray[Any]:
    """Sum along an axis sequentially, in index order, keeping the axis with length 1 (D19:
    a fixed-order reduction, not pairwise)."""
    return np.cumsum(values, axis=axis).take([-1], axis=axis)


_NOT_MODEL_SOURCE = frozenset({__file__, dataclasses.__file__})
"""This module and the one whose ``replace()`` constructs a declaration from another."""


def _declaration_loc() -> Loc | None:
    """The model source line that constructs a declaration: the nearest frame outside this
    module, outside ``dataclasses`` (a ``replace()`` of a declaration) and outside generated
    code (a dataclass's ``__init__``). A declaration's ``loc`` is not an argument, so a
    ``replace()`` does not copy it: the copy is declared where it is made, which is where a
    later error about it belongs (a subclass that replaces its parent's ``GroupSum``)."""
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


class BasisInfo(NamedTuple):
    """What the runs of a basis share (:meth:`Basis.check`)."""

    valuation_date: dt.date
    """Which the stage's valuation date must equal (design section 9)."""
    horizon: int
    """The runs' ``H``: a ``cur`` basis's is the stage's default horizon and its bound."""


@dataclass(frozen=True, slots=True)
class Basis:
    """A basis of a stage: one or more projection runs with the same valuation date, horizon,
    number of scenarios J and scenario files, and pairwise disjoint policy ids (one run per
    product class), whose outputs the stage's :class:`GroupSum` series sum."""

    name: str
    doc: str = ""
    loc: Loc | None = field(default=None, init=False, compare=False)
    """Where the basis is declared: the line that constructs it (:func:`_declaration_loc`)."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "loc", _declaration_loc())
        if not _is_name(self.name):
            raise BuildError(f"a basis is named like a variable, not {self.name!r}", self.loc)

    def check(self, runs: Sequence[RunResult]) -> BasisInfo:
        """The runs given for this basis agree on the valuation date, the horizon and J, read
        every scenario input they share from the same file, and hold disjoint policies; returns
        their valuation date, which the stage's must equal, and their horizon (design section
        9). A LoadError otherwise.

        The scenario files are compared by the sha256 each run's manifest records: a group
        total adds the members' values of scenario ``s`` across the runs, and the mean averages
        those totals, which means something only when ``s`` is the same path in every run (two
        class runs given the same J = 2 paths in the other order would otherwise be summed path
        by path)."""
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
        files: dict[str, tuple[str, str]] = {}  # scenario input -> (first run, file sha256)
        for r in runs:
            for name, sha in _scenario_files(r).items():
                if name in files and files[name][1] != sha:
                    other, known = files[name]
                    raise LoadError(
                        f"basis {self.name}: run {r.run_id} read scenario input {name} from "
                        f"another file than run {other} (sha256 {sha[:12]}... and "
                        f"{known[:12]}...); the runs of a basis share their scenarios, so that "
                        "scenario s is one path in every run"
                    )
                files.setdefault(name, (r.run_id, sha))
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
        return BasisInfo(dt.date.fromisoformat(when), first.H)


def _valuation_date(r: RunResult) -> str:
    return str(r.manifest["settings"]["valuation_date"])


def _scenario_files(r: RunResult) -> dict[str, str]:
    """The sha256 of every scenario file a run read, by scenario input, from its manifest."""
    entries: dict[str, Any] = r.manifest["inputs"]["scenarios"]
    return {name: str(e["sha256"]) for name, e in entries.items()}


def _is_name(value: object) -> bool:
    return isinstance(value, str) and value.isidentifier()


def _is_str(value: object) -> bool:
    return isinstance(value, str)


def _is_bool(value: object) -> bool:
    return isinstance(value, bool)


def _is_start(value: object) -> bool:
    """0 or 1 as an int: ``True == 1`` and ``1.0 == 1``, but neither is a month."""
    return isinstance(value, int) and not isinstance(value, bool) and value in (0, 1)


@dataclass(frozen=True, slots=True)
class GroupSum:
    """A group series: each named output of every run of ``basis`` summed over the group's
    members in ascending ``policy_id`` across the runs, sequentially (D19); several variables'
    group totals added left to right per scenario (``("bel", "ra")`` gives BEL_g + RA_g); then
    ``scenarios="mean"`` reduces over the scenarios as ``cumsum_s((1.0 / J) * x)[-1]`` in
    ascending ``s``, giving one scenario, and ``"each"`` keeps them.

    ``unit`` must be the unit every source run recorded for every variable (D11 across runs),
    and one that adds across policies (:data:`GROUP_ADDITIVE`). ``time`` says whether the
    variables have a time dimension; ``start`` is the first month the stage reads, 1 (the
    default) for a sum of month quantities and 0 for a sum of states (a series read at ``t - 1``
    from month 1, or at month 0); the source runs' domains must start no later. A series
    without a time dimension has no start to give."""

    name: str
    basis: str
    variables: str | tuple[str, ...]
    unit: str
    scenarios: str = "mean"
    _: KW_ONLY
    time: bool = True
    start: int | None = None
    """1 when left out, for a series with a time dimension."""
    doc: str = ""
    loc: Loc | None = field(default=None, init=False, compare=False)
    """Where the series is declared: the line that constructs it, or that makes it with
    ``dataclasses.replace()`` from another (:func:`_declaration_loc`)."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "loc", _declaration_loc())
        if not _is_name(self.name):
            raise BuildError(f"a series is named like a variable, not {self.name!r}", self.loc)
        what = f"series {self.name}"
        if not _is_name(self.basis):
            raise BuildError(f"{what}: its basis is a Basis name, not {self.basis!r}", self.loc)
        if self.scenarios not in SCENARIO_TREATMENTS:
            raise BuildError(
                f"{what}: scenarios is 'mean' or 'each', not {self.scenarios!r}", self.loc
            )
        names: object = (self.variables,) if isinstance(self.variables, str) else self.variables
        if not _is_tuple(names) or not names or not all(isinstance(n, str) and n for n in names):
            raise BuildError(f"{what} names its variables, as 'bel' or ('bel', 'ra')", self.loc)
        seen: set[object] = set()
        for n in names:
            # a name no run can have stored would fail only when the stage loads, as "did not
            # store", and a name given twice would add that output's group total twice
            if not _is_name(n):
                raise BuildError(f"{what}: {n!r} is not the name of a variable", self.loc)
            if n in seen:
                raise BuildError(f"{what} names {n} twice; each output is summed once", self.loc)
            seen.add(n)
        # the annotations do not bind a caller; a flag or a month of another type would reach
        # the declaration's repr, and so the graph hash, as a different series
        if not _is_bool(self.time):
            raise BuildError(f"{what}: time is True or False, not {self.time!r}", self.loc)
        if self.start is not None and not self.time:
            raise BuildError(
                f"{what} has no time dimension (time=False), so no start; leave start out",
                self.loc,
            )
        if self.start is not None and not _is_start(self.start):
            raise BuildError(
                f"{what}: start is 0 (a sum of states) or 1 (a sum of month quantities), not "
                f"{self.start!r}",
                self.loc,
            )
        if not _is_str(self.unit):
            raise BuildError(
                f"{what}: unit is a unit's name, as 'Money', not {self.unit!r}", self.loc
            )
        try:
            unit = unit_from_name(self.unit)
        except ValueError as exc:
            raise BuildError(f"{what}: {exc}", self.loc) from exc
        if unit not in GROUP_ADDITIVE:
            raise UnitError(
                f"{what} sums {unit} over a group's members; a group sum adds Money, Count or "
                "CoverageUnits",
                self.loc,
            )

    @property
    def names(self) -> tuple[str, ...]:
        """The variables summed, in the order their totals are added."""
        return (self.variables,) if isinstance(self.variables, str) else self.variables

    @property
    def declared_unit(self) -> Unit:
        return unit_from_name(self.unit)

    @property
    def first_month(self) -> int:
        """The first month of the series' domain: ``start``, 1 when left out, and 0 for a
        series without a time dimension."""
        if not self.time:
            return 0
        return 1 if self.start is None else self.start

    def decl(self) -> ir.SeriesDecl:
        """The IR declaration: dimensions and domain fixed here, at build time, never from J."""
        return ir.SeriesDecl(
            self.name,
            self.declared_unit,
            self.time,
            self.first_month,
            self.scenarios == "each",
            self.basis,
            self.names,
            self.doc,
            self.loc,
        )


# ------------------------------------------------------------------------------------ stage


class Aggregation(Product):
    """Base class of an aggregation stage (design section 9).

    Beside a ``Product``'s class attributes a stage sets:

    * ``model_point``: the schema of its groups, one row per group, keyed on ``group_id``, as
      its outputs are (``self.group`` reads a field, the same as ``self.mp``);
    * ``bases``: the :class:`Basis` declarations whose runs it reads;
    * ``series``: the :class:`GroupSum` declarations, read in formulas as
      ``self.series.<name>(t + k)``, ``self.series.<name>.at(k)`` or ``self.series.<name>()``;
    * ``scenario_inputs``: its own curves, such as a locked-in discount curve;
    * ``disclosures``: the approach strings and notes its run manifest discloses beside the
      confidence level and the para 120 curve (:func:`run_aggregation`).

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
    disclosures: ClassVar[dict[str, str | tuple[str, ...]]] = {}
    """What the manifest's ``disclosures`` block adds to the data-derived entries: by name, a
    string (the approach to discount rates, ``discount_rates``, or to the risk adjustment,
    ``risk_adjustment``) or a tuple of strings (``notes``)."""

    @property
    def group(self) -> ModelPoint:
        """``self.group``: the group's fields, rendered ``mp.<field>`` as ``self.mp`` is."""
        return self.mp

    @classmethod
    def series_declarations(cls) -> dict[str, ir.SeriesDecl]:
        """The stage's group series by name, each checked against the bases: a series of an
        undeclared basis, a name declared twice, a malformed tuple, a group schema keyed on
        another field than ``group_id``, a group field the loader fills declared with another
        kind or unit than it fills (:func:`_check_group_fields`), malformed disclosures
        (:func:`_check_disclosures`) or a variable taking one of the stage's names is a
        BuildError at the model source."""
        schema = cls.model_point
        if schema is not None and schema.id_field != GROUP_ID:
            raise BuildError(
                f"{cls.__qualname__}.model_point keys its rows on {schema.id_field!r}; a stage's "
                f"model points are groups, keyed on {GROUP_ID!r}, which its outputs are keyed on "
                "(design section 9)",
                _class_loc(cls),
            )
        if schema is not None:
            _check_group_fields(cls, schema.decls())
        _check_disclosures(cls)
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


def _is_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def _is_dict(value: object) -> TypeGuard[dict[object, object]]:
    return isinstance(value, dict)


def _is_mapping(value: object) -> TypeGuard[Mapping[str, Any]]:
    """A JSON object of a manifest."""
    return isinstance(value, dict)


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


_LOADER_FILLS: Final = {
    **dict.fromkeys(FROM_DATES, ("int", units.INT, "a month count from the dates")),
    **{
        name: ("number", units.MONEY, f"the previous run's {variable}, an amount")
        for name, (variable, _month) in FROM_PREVIOUS_RUN.items()
    },
    "csm_prev": ("number", units.MONEY, "compared with the previous run's csm(0), an amount"),
    "lc_prev": ("number", units.MONEY, "compared with the previous run's lc(0), an amount"),
}
"""The group fields the loader fills or reads as a given kind and unit, by name: the kind and
unit a stage that declares one must give it, and what the loader puts there."""


def _check_group_fields(cls: type[Aggregation], fields: Mapping[str, ir.FieldDecl]) -> None:
    """A group field the loader fills (:data:`FROM_PREVIOUS`) or compares with the previous
    run (``csm_prev``, ``lc_prev``) is declared as the loader fills it: the month counts as
    ``int`` fields of unit ``Int``, the amounts as ``number`` fields of unit ``Money``, none as
    a derived field of the schema, which the loader's would collide with (D11: a month count
    read as Money is a unit mistake, found at build). A schema that declares a field derived
    from the previous run declares ``prev_run_id``, which names that run."""
    where = f"{cls.__qualname__}.model_point"
    for name, (kind, unit, what) in _LOADER_FILLS.items():
        decl = fields.get(name)
        if decl is None:
            continue
        if decl.derived:
            raise BuildError(
                f"{where} derives {name}, which the loader fills ({what}); declare it as a field",
                _class_loc(cls),
            )
        if decl.kind != kind or decl.unit != unit:
            got = decl.kind if decl.unit is None else f"{decl.kind} {decl.unit}"
            raise UnitError(
                f"{where} declares {name} as {got}, but the loader fills it with {what}: declare "
                f"it {kind} {unit} (D11)",
                _class_loc(cls),
            )
    derived = [name for name in FROM_PREVIOUS if name in fields]
    if derived and "prev_run_id" not in fields:
        raise BuildError(
            f"{where} declares {', '.join(derived)}, which the loader derives from the previous "
            "run the groups name and the dates, but no prev_run_id to name that run; build the "
            "schema with group_schema(previous=True)",
            _class_loc(cls),
        )


DATA_DISCLOSURES: Final = ("confidence_level", "para_120_curve_sha256")
"""The disclosures the loader reads from the data (the cur basis's ``alpha``, the ``df_cur``
file's hash), which a stage's declared :attr:`Aggregation.disclosures` may not name."""


def _check_disclosures(cls: type[Aggregation]) -> None:
    """``disclosures`` is a dict from a name to a string or a tuple of strings, and names none
    of :data:`DATA_DISCLOSURES`: a declared value would replace what the data says, and a value
    of another type would fail only when the manifest is written, after the run."""
    value: object = getattr(cls, "disclosures", {})
    where = f"{cls.__qualname__}.disclosures"
    if not _is_dict(value):
        raise BuildError(
            f"{where} is a dict of strings or tuples of strings, not {type(value).__name__}",
            _class_loc(cls),
        )
    for key, entry in value.items():
        if not _is_name(key):
            raise BuildError(
                f"{where}: a disclosure is named like a variable, not {key!r}", _class_loc(cls)
            )
        if key in DATA_DISCLOSURES:
            raise BuildError(
                f"{where} declares {key}, which the loader reads from the data (the cur basis's "
                "alpha, the df_cur file); a stage declares its approach strings and notes",
                _class_loc(cls),
            )
        if not isinstance(entry, str) and not (
            _is_tuple(entry) and all(isinstance(x, str) for x in entry)
        ):
            raise BuildError(
                f"{where}: {key} is a string or a tuple of strings, not {entry!r}",
                _class_loc(cls),
            )


# ---------------------------------------------------------------------- in-session builder


def group_series(
    stage: type[Aggregation],
    bases: Mapping[str, Sequence[RunResult]],
    members: Mapping[Any, Sequence[Any]],
    *,
    runs_root: Path | None = None,
) -> dict[str, LoadedSeries]:
    """Every series ``stage`` declares, summed from projection runs.

    ``bases`` gives each declared basis's runs; ``members`` each group's member policy ids, by
    group id (the stage's model point ids). Every group has members, every policy of every run
    belongs to exactly one group, every member is in some run, and every group has a member in
    the runs of each basis a series sums. Each series is a dense ``(Tn | 1, groups,
    scenarios)`` array with the groups in ascending id, as the loader orders the stage's model
    points, and months ``0..H`` of its basis. A variable a series names must be stored by every
    run of its basis with the series' unit; its recorded domain start is the series' (the latest
    over the variables), which the loader requires to be no later than the stage's declared
    start when the stage runs. The provenance records the basis's valuation date, which the
    stage's must equal, its horizon, which bounds the stage's when the basis is ``cur``, the
    membership the sums used and each source run's id, directory (relative to ``runs_root``
    when it lies under it), manifest sha256 and outputs' sha256s. A mean keeps the
    per-scenario group totals it averaged as its ``pathwise`` values. A LoadError otherwise."""
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
    shared = {name: b.check(bases[name]) for name, b in declared.items()}
    group_ids = _sorted_ids(list(members))
    owner: dict[Any, Any] = {}
    for g in group_ids:
        if len(members[g]) == 0:
            raise LoadError(f"group {g} has no members; a group holds at least one policy")
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
    sums = _sums(stage)
    for name in dict.fromkeys(s.basis for s in sums):  # each basis a series sums, in order
        held = {pid for r in bases[name] for pid in r.ids.tolist()}
        for g in group_ids:
            if not any(pid in held for pid in members[g]):
                raise LoadError(
                    f"group {g} has no member in the runs of basis {name}, which a series of "
                    "the stage sums; its sums there would be zero by omission"
                )
    membership = [
        [_plain(g), [_plain(p) for p in _sorted_ids(list(members[g]))]] for g in group_ids
    ]
    return {
        s.name: _summed(
            s, bases[s.basis], group_ids, members, shared[s.basis], membership, runs_root
        )
        for s in sums
    }


def _sorted_ids(ids: list[Any]) -> NDArray[Any]:
    """Ids in ascending order, as the loader sorts model points (a stable argsort)."""
    arr = np.asarray(ids, dtype=object)
    return arr[np.argsort(arr, kind="stable")]


def _plain(value: object) -> object:
    """An id as a plain Python value, for the manifest's JSON: a NumPy integer (an integer id
    field's value) as an int, a NumPy string as a str."""
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, str):
        return str(value)
    return value


def _summed(
    s: GroupSum,
    runs: Sequence[RunResult],
    group_ids: NDArray[Any],
    members: Mapping[Any, Any],
    basis: BasisInfo,
    membership: list[list[Any]],
    runs_root: Path | None,
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
    pathwise: NDArray[np.float64] | None = None
    if s.scenarios == "mean":
        pathwise = np.ascontiguousarray(x)  # the per-scenario totals, kept beside the mean
        x = sequential_sum((1.0 / n_scen) * x, axis=2)  # ascending s; x * 1.0 when J = 1
    provenance: dict[str, Any] = {
        "basis": s.basis,
        "variables": list(s.names),
        "basis_scenarios": n_scen,
        "valuation_date": basis.valuation_date.isoformat(),
        "basis_horizon": basis.horizon,  # recorded for a series without a time dimension too
        "source_runs": [_source_run(s, r, runs_root) for r in runs],
        "members": membership,  # [group id, its member ids ascending], groups ascending
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
        pathwise,
    )


def _source_run(s: GroupSum, r: RunResult, runs_root: Path | None) -> dict[str, Any]:
    """Where a series' values came from: the run, its directory, its manifest's hash and the
    hashes of its outputs (design section 9)."""
    manifest = r.run_dir / "manifest.json"
    try:
        raw = manifest.read_bytes()
    except OSError:
        raise LoadError(
            f"series {s.name}: basis {s.basis} run {r.run_id} has no manifest at {manifest}"
        ) from None
    return {
        "run_id": r.run_id,
        "dir": _dir_label(r.run_dir, runs_root),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "outputs_sha256": _outputs_sha256(r.manifest),
    }


def _outputs_sha256(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """The sha256 of a run's two output files, as its manifest records them."""
    outputs: Mapping[str, Any] = manifest.get("outputs", {})
    return {"monthly": outputs.get("monthly_sha256"), "scalars": outputs.get("scalars_sha256")}


def _dir_label(path: Path, runs_root: Path | None) -> str:
    """A run directory as a manifest records it: relative to ``runs_root`` (``/``-separated)
    when it lies under it, so that a tree of runs can move as a whole, else its absolute
    path."""
    full = Path(path).resolve()
    if runs_root is not None:
        try:
            return full.relative_to(Path(runs_root).resolve()).as_posix()
        except ValueError:
            pass
    return str(full)


def _group_totals(
    var: str, runs: Sequence[RunResult], group_ids: NDArray[Any], members: Mapping[Any, Any]
) -> NDArray[np.float64]:
    """``var`` summed over each group's members in ascending ``policy_id`` across the runs,
    sequentially, whatever the order of the runs and of the rows in them: ``(Tn | 1, groups,
    scenarios)``. Every group has a member in these runs (:func:`group_series`)."""
    arrays = [np.asarray(r.var(var), dtype=np.float64) for r in runs]
    where: dict[Any, tuple[int, int]] = {}
    for i, r in enumerate(runs):
        for j, pid in enumerate(r.ids.tolist()):
            where[pid] = (i, j)
    columns: list[NDArray[np.float64]] = []
    for g in group_ids:
        lanes = [
            arrays[where[pid][0]][:, where[pid][1], :]
            for pid in _sorted_ids(list(members[g]))
            if pid in where
        ]
        columns.append(sequential_sum(np.stack(lanes, axis=1), axis=1))
    return np.concatenate(columns, axis=1)


# --------------------------------------------------------------------------- loader and runs

OLD_BASIS: Final = "old"
"""The basis on the previous non-financial tables and the expected in-force (in-force 5.5): its
policies are exactly the previous run's members (G4)."""

NEW_BASIS: Final = "new"
"""The basis on the current non-financial tables at the locked-in rates (in-force 5.5): its
policies are the cur basis's (G4)."""

FINANCIAL_INPUTS: Final = ("rf_df", "cpi", "ilp")
"""The inputs of a projection run that carry its financial basis: the discount curve and the CPI
path (scenario inputs) and the illiquidity premium (a table). The old and new bases share every
one of them; the cur and new bases share every other table (in-force 5.5)."""

LOCKED_IN_CURVES: Final = ("p_l", "p_rec")
"""A stage's scenario inputs built from the locked-in discount file and ILP table (design
section 10, ``locked_in_curve``): their metadata ``source_sha256`` and ``ilp_sha256`` must be
the group's ``discount_rec_sha256`` and ``ilp_sha256``."""

LOCKED_IN_RUN_CURVE: Final = "rf_df"
"""The scenario input through which the locked-in runs (bases old and new) read ``p_l``, their
discount curve (design section 10): its metadata ``source_sha256`` and ``ilp_sha256`` must be
the group's too."""

LOCKED_IN_CPI: Final = "cpi"
"""The scenario input through which the locked-in runs (bases old and new) read the spliced CPI
path (``locked_in_cpi``): its metadata ``rec_sha256`` must be the group's ``cpi_rec_sha256``."""

CONFIDENCE_LEVEL: Final = ("ra_param", "alpha")
"""The table and column of the cur basis's runs that hold the disclosed confidence level (para
119)."""

PARA_120_CURVE: Final = "df_cur"
"""The stage's scenario input that holds the disclosed composed current curve (para 120)."""

PREVIOUS_TOLERANCE: Final = 1e-12
"""``csm_prev`` and ``lc_prev`` agree with the previous run's ``csm(0)`` and ``lc(0)`` within this
many times ``P_g`` (design section 9)."""

STAGE_BLOCKS: Final = (
    "previous",
    "bases",
    "membership",
    "locked_in",
    "provenance",
    "disclosures",
)
"""The manifest blocks a run made by :func:`run_aggregation` adds beside every run's, which
:func:`replay_aggregation` carries over."""

PROVENANCE_KEYS: Final = ("rebased_from", "derived_from")
"""The metadata keys (``halley.<key>``) of a file that a section 10 helper made from another:
the source's sha256 (``rebased_from``) or run id (``derived_from``)."""


def require_one_group(group_ids: Sequence[str], where: str) -> None:
    """G1 (in-force 12.4): one group per aggregation run in this version. This is the rule's one
    place: :func:`load_aggregation_inputs` calls it once, and the rest of the loader (the
    membership, the derived fields, the locked-in hashes, the manifest blocks) works per group,
    so lifting the rule is deleting that call."""
    if len(group_ids) != 1:
        raise LoadError(
            f"{where}: holds {len(group_ids)} groups ({', '.join(group_ids)}); an aggregation run "
            "measures one group in this version (G1)"
        )


def _month_index(d: dt.date) -> int:
    """``mi(d) = 12 * year(d) + month(d) - 1``, the run constant ``m_val`` of a date."""
    return 12 * d.year + d.month - 1


def _is_month_end(d: dt.date) -> bool:
    return (d + dt.timedelta(days=1)).day == 1


def _listing(values: Sequence[object], limit: int = 5) -> str:
    shown = ", ".join(str(v) for v in values[:limit])
    return shown + (f" and {len(values) - limit} more" if len(values) > limit else "")


def _group_ids(cols: Mapping[str, NDArray[Any]], where: str, what: str) -> list[str]:
    """Each row's group id, ``f"{portfolio}-{cohort}-{profit_bucket}"`` (design section 9); a
    key column that is missing or of another type is a LoadError naming the file."""
    for name in GROUP_KEY:
        if name not in cols:
            raise LoadError(
                f"{where}: column {name} is missing; {what} carry the group key "
                f"({', '.join(GROUP_KEY)})"
            )
    portfolio, cohort, bucket = (cols[name] for name in GROUP_KEY)
    if portfolio.dtype != object or bucket.dtype != object:
        raise LoadError(f"{where}: portfolio and profit_bucket are string columns")
    if not np.issubdtype(cohort.dtype, np.integer):
        raise LoadError(f"{where}: cohort is an integer column, not {cohort.dtype}")
    return [
        f"{p}-{int(c)}-{b}"
        for p, c, b in zip(portfolio.tolist(), cohort.tolist(), bucket.tolist(), strict=True)
    ]


# ------------------------------------------------------------------------ the bases' runs


@dataclass(frozen=True, slots=True)
class _Member:
    """One row of a basis run's model point copy."""

    pid: Any
    group: str
    issue_date: dt.date
    row: int


MODEL_POINTS: Final = ("model points", "")
"""The key of a run's model point copy among :attr:`_BasisRun.inputs`."""


@dataclass(frozen=True, slots=True)
class _BasisRun:
    """A projection run of a basis, read from its directory with every file verified: the run
    (its outputs loaded), its manifest's sha256, the rows of its model point copy and every
    copied input as read."""

    basis: str
    result: RunResult
    manifest_sha256: str
    copy: str
    """The model point copy, as messages name it."""
    members: tuple[_Member, ...]
    inputs: dict[tuple[str, str], ReadFile]
    """Every copied input by (kind, name): :data:`MODEL_POINTS`, ``("table", <name>)`` and
    ``("scenario input", <name>)``, each read from its copy and verified against the sha256 the
    manifest records. What the loader asks of an input (its metadata, its hash, its values) it
    asks of these, never of the manifest's description of them, which no hash covers."""

    @property
    def run_id(self) -> str:
        return self.result.run_id

    def at(self, m: _Member) -> str:
        """A member's row, for a message: the copy, the row and the policy."""
        return f"{self.copy}: row {m.row}, policy {m.pid}"

    def scenario(self, name: str) -> ReadFile | None:
        """The copy of scenario input ``name``, ``None`` when the run read no such input."""
        return self.inputs.get(("scenario input", name))


def _input_copies(
    path: Path, manifest: Mapping[str, Any], who: str
) -> dict[tuple[str, str], ReadFile]:
    """Every input a run copied (its model points, tables and scenario files), each read from
    its copy and verified against the sha256 its manifest records (:func:`read_copy`)."""
    inputs: Mapping[str, Any] = manifest["inputs"]
    entries: list[tuple[tuple[str, str], Any]] = [(MODEL_POINTS, inputs["model_points"])]
    entries += [(("table", n), e) for n, e in inputs["tables"].items()]
    entries += [(("scenario input", n), e) for n, e in inputs["scenarios"].items()]
    return {key: read_copy(path, e, f"{who}: {key[0]} {key[1]}".rstrip()) for key, e in entries}


def _basis_run(basis: str, given: object) -> _BasisRun:
    """A run of a basis given as a :class:`RunResult` or a run directory, read through its
    directory: every file its manifest records a hash for is verified (a tampered file is a
    LoadError naming it), the run must be a projection, its copied inputs are read from the
    copies (:func:`_input_copies`), and its model point copy gives the membership (each row's
    group key and issue date)."""
    if isinstance(given, RunResult):
        path, expected = given.run_dir, given.run_id
    elif isinstance(given, (str, Path)):
        path, expected = Path(given), None
    else:
        raise LoadError(
            f"basis {basis}: a run is a RunResult or a run directory, not {type(given).__name__}"
        )
    manifest = verify_run(path, f"basis {basis} run")
    run_id = str(manifest.get("run_id"))
    if expected is not None and run_id != expected:
        raise LoadError(
            f"basis {basis}: run {expected} was given, but its directory {path} holds run {run_id}"
        )
    kind = manifest.get("outputs", {}).get("kind", "projection")
    if kind != "projection":
        raise LoadError(f"basis {basis}: {path} holds an {kind} run; a basis reads projection runs")
    _manifest, sha = read_manifest(path)
    result = load_run(path)
    copies = _input_copies(path, manifest, f"basis {basis} run {run_id}")
    f = copies[MODEL_POINTS]
    cols = table_columns(f)
    groups = _group_ids(cols, f.where, "a basis run's model points")
    id_field = str(manifest["outputs"].get("id_field", "policy_id"))
    for name in (id_field, "issue_date"):
        if name not in cols:
            raise LoadError(f"{f.where}: column {name} is missing")
    ids: list[Any] = cols[id_field].tolist()
    dates: list[Any] = cols["issue_date"].tolist()
    if not all(isinstance(d, dt.date) for d in dates):
        raise LoadError(f"{f.where}: issue_date is a date column")
    if sorted(map(repr, ids)) != sorted(map(repr, result.ids.tolist())):
        raise LoadError(f"{f.where}: it holds other policies than the run's outputs")
    members = tuple(
        _Member(pid, g, d, row)
        for row, (pid, g, d) in enumerate(zip(ids, groups, dates, strict=True))
    )
    return _BasisRun(basis, result, sha, f.where, members, copies)


def _runs_of(basis: str, given: object) -> list[object]:
    if _is_tuple(given) or _is_list(given):
        return list(given)
    raise LoadError(
        f"basis {basis} takes a list of runs (RunResults or run directories), not "
        f"{type(given).__name__}"
    )


def _check_keys_agree(runs: Mapping[str, Sequence[_BasisRun]]) -> None:
    """A policy in several bases has one group key in all of them (G4 without a previous
    run)."""
    seen: dict[Any, tuple[_BasisRun, _Member]] = {}
    for rs in runs.values():
        for r in rs:
            for m in r.members:
                if m.pid not in seen:
                    seen[m.pid] = (r, m)
                    continue
                r0, m0 = seen[m.pid]
                if m0.group != m.group:
                    raise LoadError(
                        f"{r.at(m)} is in group {m.group}, but {r0.at(m0)} is in group "
                        f"{m0.group}; a policy's group key is the same in every run (G4)"
                    )


# ---------------------------------------------------------------------------- the groups


@dataclass(frozen=True, slots=True)
class _Groups:
    """The groups file as read: every column, and each row's derived group id."""

    file: ReadFile
    columns: dict[str, NDArray[Any]]
    ids: list[str]

    def at(self, row: int) -> str:
        """A row, for a message: the file, the group and the row."""
        return f"{self.file.where}: group {self.ids[row]} (row {row})"

    def date(self, name: str, row: int) -> dt.date:
        value: object = self._column(name)[row]
        if not isinstance(value, dt.date):
            raise LoadError(f"{self.at(row)}: {name} is a date, not {value!r}")
        return value

    def _column(self, name: str) -> NDArray[Any]:
        if name not in self.columns:
            raise LoadError(f"{self.file.where}: column {name} is missing")
        return self.columns[name]

    def number(self, name: str, row: int) -> float:
        """A number column's value. The rules read these before the schema's field checks
        run, so a column of another type is a LoadError here, naming the file and column."""
        values = self._column(name)
        if values.dtype.kind not in "iuf":
            got = column_type(self.file, name)
            raise LoadError(f"{self.file.where}: column {name} is a number column, not {got}")
        return float(values[row])

    def string(self, name: str, row: int) -> str:
        """A string column's value (a run id, a hash), checked before the schema's are."""
        value: object = self._column(name)[row]
        if not isinstance(value, str):
            got = column_type(self.file, name)
            raise LoadError(f"{self.file.where}: column {name} is a string column, not {got}")
        return value


def _read_groups(source: Source, valuation_date: dt.date) -> _Groups:
    """The groups file (in-force 3.3): the group key, the recognition date and the stage's
    fields, one row per group. ``group_id`` is derived, never a column, and so are the fields
    the loader fills from the previous run and the dates; a duplicate key is a LoadError naming
    both rows; then the rules that need no run (:func:`_check_group_rules`)."""
    f = read_source(source, "groups")
    cols = table_columns(f)
    if GROUP_ID in cols:
        raise LoadError(
            f"{f.where}: carries a column {GROUP_ID}, which the loader derives as "
            "portfolio-cohort-profit_bucket (design section 9)"
        )
    retyped = [name for name in FROM_PREVIOUS if name in cols]
    if retyped:
        raise LoadError(
            f"{f.where}: carries {', '.join(retyped)}, which the loader derives from the "
            "previous run and the dates; they are never typed into the groups file (design "
            "section 9)"
        )
    ids = _group_ids(cols, f.where, "the groups")
    if not ids:
        raise LoadError(f"{f.where}: no rows")
    seen: dict[str, int] = {}
    for row, g in enumerate(ids):
        if g in seen:
            raise LoadError(
                f"{f.where}: rows {seen[g]} and {row} are both group {g}; a group has one row, "
                "and every member maps to exactly one"
            )
        seen[g] = row
    groups = _Groups(f, cols, ids)
    _check_group_rules(groups, valuation_date)
    return groups


def _check_group_rules(groups: _Groups, valuation_date: dt.date) -> None:
    """The ``groups`` rules of in-force 3.3 that need no run: the dates are month-ends; the
    recognition date is no later than the valuation date; ``prev_valuation_date`` is at least a
    month before it and no earlier than the recognition date (``0 <= tau <= tau0``, with ``tau``
    at least 1); at most one of ``csm_prev`` and ``lc_prev`` is positive."""
    cols = groups.columns
    v = valuation_date
    for row in range(len(groups.ids)):
        at = groups.at(row)
        rec = groups.date("recognition_date", row)
        if not _is_month_end(rec):
            raise LoadError(f"{at}: recognition_date {rec} is not the last day of a month")
        if rec > v:
            raise LoadError(f"{at}: recognition_date {rec} is after the valuation date {v}")
        if "prev_valuation_date" in cols:
            prev = groups.date("prev_valuation_date", row)
            if not _is_month_end(prev):
                raise LoadError(f"{at}: prev_valuation_date {prev} is not the last day of a month")
            if _month_index(prev) > _month_index(v) - 1:
                raise LoadError(
                    f"{at}: prev_valuation_date {prev} is not at least a month before the "
                    f"valuation date {v}"
                )
            if prev < rec:
                raise LoadError(
                    f"{at}: prev_valuation_date {prev} is before recognition_date {rec}, so tau "
                    f"= {_month_index(v) - _month_index(prev)} would exceed tau0 = "
                    f"{_month_index(v) - _month_index(rec)}"
                )
        if "csm_prev" in cols and "lc_prev" in cols:
            csm, lc = groups.number("csm_prev", row), groups.number("lc_prev", row)
            if csm > 0 and lc > 0:
                raise LoadError(
                    f"{at}: csm_prev = {csm!r} and lc_prev = {lc!r} are both positive; at most "
                    "one of them is"
                )


def _members_by_group(
    runs: Mapping[str, Sequence[_BasisRun]], groups: _Groups
) -> dict[str, list[Any]]:
    """Every member of every basis run maps to exactly one group row and was issued on its
    group's recognition date; every group has a member. The members of each group, over every
    basis."""
    rows = {g: row for row, g in enumerate(groups.ids)}
    members: dict[str, dict[Any, None]] = {g: {} for g in groups.ids}
    for rs in runs.values():
        for r in rs:
            for m in r.members:
                row = rows.get(m.group)
                if row is None:
                    raise LoadError(
                        f"{r.at(m)} is in group {m.group}, which has no row in "
                        f"{groups.file.where}; every member maps to exactly one group row"
                    )
                rec = groups.date("recognition_date", row)
                if m.issue_date != rec:
                    raise LoadError(
                        f"{r.at(m)}: issue_date {m.issue_date} is not the recognition_date {rec} "
                        f"of group {m.group}; every member is issued on it (in-force 3.3)"
                    )
                members[m.group][m.pid] = None
    for row, g in enumerate(groups.ids):
        if not members[g]:
            raise LoadError(f"{groups.at(row)} has no member in any run of the stage's bases")
    return {g: list(pids) for g, pids in members.items()}


# ------------------------------------------------------------------------ the previous run


@dataclass(frozen=True, slots=True)
class _Previous:
    """The previous aggregation run a group names, read from its directory with every file
    verified: its outputs, its manifest's sha256 and its members at its valuation date (its cur
    basis's), by policy."""

    path: Path
    result: RunResult
    manifest_sha256: str
    members: dict[Any, str]

    @property
    def run_id(self) -> str:
        return self.result.run_id

    @property
    def who(self) -> str:
        return f"the previous run {self.run_id}"


def find_run(run_id: str, runs_root: Path) -> Path:
    """The directory under ``runs_root`` whose manifest carries ``run_id`` (design section 9,
    "Run addressing"): every ``runs_root/*/manifest.json`` is read; no match, or several (a run
    and its copy, a replay filed beside the run it replays), is a LoadError."""
    root = Path(runs_root)
    if not root.is_dir():
        raise LoadError(f"runs_root {root} is not a directory")
    found: list[Path] = []
    for manifest in sorted(root.glob("*/manifest.json")):
        try:
            data: Any = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LoadError(f"{manifest} is not a run manifest: {exc}") from None
        if _is_mapping(data) and data.get("run_id") == run_id:
            found.append(manifest.parent)
    if not found:
        raise LoadError(
            f"no run under runs_root {root} has run_id {run_id} ({root}/*/manifest.json)"
        )
    if len(found) > 1:
        raise LoadError(
            f"{len(found)} runs under runs_root {root} have run_id {run_id}: "
            f"{', '.join(str(p) for p in found)}; a run id addresses one run, so file each run "
            "once under runs_root"
        )
    return found[0]


def _needs_previous(graph: ir.Graph) -> bool:
    """Whether the stage's groups name a previous run (``prev_run_id``) or read a field derived
    from one."""
    return "prev_run_id" in graph.fields or any(n in graph.fields for n in FROM_PREVIOUS_RUN)


def _resolve_previous(
    graph: ir.Graph, groups: _Groups, previous: Path | None, settings: RunSettings
) -> _Previous | None:
    """The previous aggregation run the groups name by ``prev_run_id``: found under
    ``settings.runs_root`` (:func:`find_run`), or ``previous=``, a test override whose manifest
    must carry that run id; every file of it verified; an aggregation run at one scenario that
    recorded its membership."""
    if not _needs_previous(graph):
        if previous is not None:
            raise LoadError(
                "previous= was given, but the stage's groups name no previous run (prev_run_id)"
            )
        return None
    if "prev_run_id" not in groups.columns:
        raise LoadError(
            f"{groups.file.where}: column prev_run_id is missing; the stage reads the previous "
            "run of each group"
        )
    wanted = sorted({groups.string("prev_run_id", row) for row in range(len(groups.ids))})
    if len(wanted) != 1:
        raise LoadError(
            f"{groups.file.where}: the groups name several previous runs ({', '.join(wanted)})"
        )
    want = wanted[0]
    if previous is not None:
        path = Path(previous)
        manifest, _sha = read_manifest(path)
        if manifest.get("run_id") != want:
            raise LoadError(
                f"previous= {path} holds run {manifest.get('run_id')}, but {groups.at(0)} names "
                f"prev_run_id {want}"
            )
    elif settings.runs_root is None:
        raise LoadError(
            f"{groups.at(0)} names prev_run_id {want}; set RunSettings.runs_root to the directory "
            "the runs are filed in (or pass previous=)"
        )
    else:
        path = find_run(want, settings.runs_root)
    manifest = verify_run(path, "previous run")
    kind = manifest.get("outputs", {}).get("kind", "projection")
    if kind != "aggregation":
        raise LoadError(
            f"prev_run_id {want} names {path}, a {kind} run; it names the previous aggregation run"
        )
    _manifest, sha = read_manifest(path)
    result = load_run(path)
    from halley.aggregation import require_deterministic  # the one J = 1 check

    require_deterministic(
        result,
        f"the previous run {want}",
        "the loader reads the previous run's group outputs at one scenario",
        LoadError,
    )
    membership: Any = manifest.get("membership")
    groups_block: Any = membership.get("groups") if _is_mapping(membership) else None
    if not _is_mapping(groups_block):
        raise LoadError(
            f"the previous run {want} records no membership; it was not made by run_aggregation"
        )
    members: dict[Any, str] = {}
    for gid, by_basis in groups_block.items():
        cur: Any = by_basis.get(CUR_BASIS) if _is_mapping(by_basis) else None
        if not _is_list(cur):
            raise LoadError(
                f"the previous run {want} records no members of basis {CUR_BASIS} for group {gid}"
            )
        for pid in cur:
            members[pid] = str(gid)
    return _Previous(path, result, sha, members)


def _check_g4(runs: Mapping[str, Sequence[_BasisRun]], prev: _Previous) -> None:
    """G4 (design section 9; in-force 3.2, 12.4) per basis against the previous run's members:
    every policy at V was a member, in the same group (its key unchanged); the old basis holds
    every member (``ids(old) == ids(previous)``); the new and cur bases hold the same policies
    (``ids(new) == ids(cur) ⊆ ids(previous)``)."""
    for rs in runs.values():
        for r in rs:
            for m in r.members:
                was = prev.members.get(m.pid)
                if was is None:
                    raise LoadError(
                        f"{r.at(m)} is not a member of {prev.who}; contracts do not join a group "
                        "after recognition (G4, para 28)"
                    )
                if was != m.group:
                    raise LoadError(
                        f"{r.at(m)} is in group {m.group} but was in group {was} at {prev.who}; "
                        "a policy's group key does not change (G4)"
                    )
    ids = {b: {m.pid for r in rs for m in r.members} for b, rs in runs.items()}
    if OLD_BASIS in ids:
        missing = [p for p in _sorted_ids(list(prev.members)).tolist() if p not in ids[OLD_BASIS]]
        if missing:
            raise LoadError(
                f"basis {OLD_BASIS} holds no run with policies {_listing(missing)} of "
                f"{prev.who}'s members; the old basis projects every member of the previous run "
                "(G4)"
            )
    if NEW_BASIS in ids and CUR_BASIS in ids and ids[NEW_BASIS] != ids[CUR_BASIS]:
        only_new = _sorted_ids(list(ids[NEW_BASIS] - ids[CUR_BASIS])).tolist()
        only_cur = _sorted_ids(list(ids[CUR_BASIS] - ids[NEW_BASIS])).tolist()
        raise LoadError(
            f"bases {NEW_BASIS} and {CUR_BASIS} hold other policies (only in {NEW_BASIS}: "
            f"{_listing(only_new) or 'none'}; only in {CUR_BASIS}: "
            f"{_listing(only_cur) or 'none'}); both project the members in force at the valuation "
            "date (G4)"
        )


def _previous_value(
    prev: _Previous, name: str, gid: str, month: int | None, unit: Unit | None, what: str
) -> float:
    """The previous run's ``name`` for group ``gid`` at ``month`` (``None`` for a quantity
    without a time dimension): its stored output of that name or, failing one, its copy of the
    group series of that name. The unit must be the derived field's (D11 across runs), and the
    month inside the domain the run records for it, ``start..H`` (D11: a month before the start
    holds the zeros the generated code fills it with, not a value)."""
    r = prev.result
    variables: Mapping[str, Any] = r.manifest["outputs"]["variables"]
    if name in variables:
        info = r.variable_info(name)
        got_unit, has_time, start, last = info.unit, info.time, info.start, r.H
        values = r.var(name)
        ids: list[Any] = r.ids.tolist()
    else:
        series: Mapping[str, Any] = r.manifest["inputs"].get("series", {})
        if name not in series:
            raise LoadError(
                f"{what}: {prev.who} stored no {name}, as an output or a group series; the stage "
                f"that made it stores {name}"
            )
        s = read_series_copy(r.run_dir, name, series[name], GROUP_ID, f"{prev.who}: series {name}")
        got_unit, has_time, start, last = s.unit.name, s.has_time, s.start, s.horizon
        values = s.values
        ids = s.group_ids.tolist()
    if unit is not None and got_unit != unit.name:
        raise LoadError(
            f"{what}: {prev.who} records {name} as {got_unit}, not {unit.name} (D11 across runs)"
        )
    if has_time != (month is not None):
        raise LoadError(
            f"{what}: {prev.who} records {name} {'with' if has_time else 'without'} a time "
            "dimension"
        )
    k = 0 if month is None else month
    if has_time and k < start:
        raise LoadError(
            f"{what}: {prev.who} records {name} from t = {start}, but the loader reads "
            f"{name}({k}) (D11)"
        )
    if k > last:
        raise LoadError(
            f"{what}: {prev.who} covers months 0..{last}, but the loader reads {name}({k})"
        )
    if gid not in ids:
        raise LoadError(f"{what}: {prev.who} holds no {name} of group {gid}")
    return float(values[k, ids.index(gid), 0])


def _from_previous(
    graph: ir.Graph, groups: _Groups, row: int, prev: _Previous, valuation_date: dt.date
) -> dict[str, float | int]:
    """The derived fields of one group (design section 9): from the previous run its
    ``fcf_g(0)``, ``fcf_g(tau)``, ``lc(tau)`` and ``p_g`` (:data:`FROM_PREVIOUS_RUN`), and from
    the dates ``tau0``, ``tau``, ``i_acc``, ``i_rec`` (:data:`FROM_DATES`), each that the stage
    declares. Checked first: the previous run values at ``prev_valuation_date`` and holds the
    group; after, ``csm_prev`` and ``lc_prev`` equal its ``csm(0)`` and ``lc(0)`` within
    ``1e-12 * P_g``."""
    gid, at = groups.ids[row], groups.at(row)
    prev_date = groups.date("prev_valuation_date", row)
    valued = str(prev.result.manifest["settings"]["valuation_date"])
    if valued != prev_date.isoformat():
        raise LoadError(
            f"{at}: prev_valuation_date is {prev_date}, but {prev.who} values at {valued}"
        )
    if gid not in prev.result.ids.tolist():
        raise LoadError(
            f"{at}: {prev.who} measured no group {gid}; a group's key does not change (G4)"
        )
    rec = groups.date("recognition_date", row)
    tau0 = _month_index(valuation_date) - _month_index(rec)
    tau = _month_index(valuation_date) - _month_index(prev_date)
    out: dict[str, float | int] = {}
    for name, (variable, month) in FROM_PREVIOUS_RUN.items():
        if name in graph.fields:
            k = None if month is None else (0 if month == "0" else tau)
            unit = graph.fields[name].unit
            out[name] = _previous_value(prev, variable, gid, k, unit, f"{at}: {name}")
    dates = {"tau0": tau0, "tau": tau, "i_acc": tau0 - tau, "i_rec": tau0}
    for name in FROM_DATES:
        if name in graph.fields:
            out[name] = dates[name]
    closing = [(f, v) for f, v in (("csm_prev", "csm"), ("lc_prev", "lc")) if f in groups.columns]
    if closing:
        p_g = out.get("p_g_prev")
        if p_g is None:
            p_g = _previous_value(prev, "p_g", gid, None, units.MONEY, f"{at}: P_g")
        for fieldname, variable in closing:
            given = groups.number(fieldname, row)
            closed = _previous_value(prev, variable, gid, 0, units.MONEY, f"{at}: {fieldname}")
            if abs(given - closed) > PREVIOUS_TOLERANCE * abs(p_g):
                raise LoadError(
                    f"{at}: {fieldname} = {given!r}, but {prev.who} closed at {variable}(0) = "
                    f"{closed!r}; they agree within 1e-12 * P_g = {PREVIOUS_TOLERANCE * abs(p_g)!r}"
                )
    return out


# ---------------------------------------------------------------- structure and locked-in basis


def _input_hashes(r: _BasisRun) -> dict[tuple[str, str], str]:
    """The sha256 of every scenario file and table a run read, by (kind, name), from its
    verified copies."""
    return {key: f.sha256 for key, f in r.inputs.items() if key != MODEL_POINTS}


def _share(runs: Sequence[_BasisRun], keep: Callable[[str, str], bool], why: str) -> None:
    """Every input ``keep`` selects is read from one file by every run in ``runs``."""
    seen: dict[tuple[str, str], tuple[str, _BasisRun]] = {}
    for r in runs:
        for key, sha in _input_hashes(r).items():
            if not keep(*key):
                continue
            if key in seen and seen[key][0] != sha:
                known, other = seen[key]
                raise LoadError(
                    f"basis {other.basis} run {other.run_id} and basis {r.basis} run {r.run_id} "
                    f"read {key[0]} {key[1]} from different files (sha256 {known[:12]}... and "
                    f"{sha[:12]}...); {why}"
                )
            seen.setdefault(key, (sha, r))


def _check_structure(runs: Mapping[str, Sequence[_BasisRun]]) -> None:
    """The structural checks of design section 9: the old and new bases share every financial
    input (:data:`FINANCIAL_INPUTS`), and the cur and new bases every other table."""
    if OLD_BASIS in runs and NEW_BASIS in runs:
        _share(
            [*runs[OLD_BASIS], *runs[NEW_BASIS]],
            lambda _kind, name: name in FINANCIAL_INPUTS,
            "the old and new bases share every financial input (rf_df, cpi, ilp), so that their "
            "difference is the change in the non-financial tables and the in-force (in-force 5.5)",
        )
    if CUR_BASIS in runs and NEW_BASIS in runs:
        _share(
            [*runs[CUR_BASIS], *runs[NEW_BASIS]],
            lambda kind, name: kind == "table" and name not in FINANCIAL_INPUTS,
            "the cur and new bases share every non-financial table, so that their difference is "
            "a financial effect (in-force 5.5)",
        )


_CURVE_METADATA: Final = (("source_sha256", "discount_rec_sha256"), ("ilp_sha256", "ilp_sha256"))
"""The metadata of a locked-in curve (``locked_in_curve``, design section 10) and the group's
hash each must equal."""

_CPI_METADATA: Final = (("rec_sha256", "cpi_rec_sha256"),)
"""The metadata of the locked-in CPI path (``locked_in_cpi``) and the group's hash it equals."""

_LOCKED_IN_RUNS: Final = f"of the runs of bases {OLD_BASIS} and {NEW_BASIS}"
_CURVE_SOURCES: Final = (
    f"the stage's scenario inputs {' and '.join(LOCKED_IN_CURVES)} and scenario input "
    f"{LOCKED_IN_RUN_CURVE} {_LOCKED_IN_RUNS}"
)
_LOCKED_IN_SOURCES: Final = {
    "discount_rec_sha256": _CURVE_SOURCES,
    "ilp_sha256": _CURVE_SOURCES,
    "cpi_rec_sha256": f"scenario input {LOCKED_IN_CPI} {_LOCKED_IN_RUNS}",
}
"""By hash of the groups file, the inputs whose metadata can verify it, for a message."""


def _check_locked_in(
    groups: _Groups,
    row: int,
    stage_files: Mapping[str, ReadFile],
    runs: Mapping[str, Sequence[_BasisRun]],
) -> dict[str, Any]:
    """A group's locked-in basis (design section 9; in-force 3.3, B72(b)): the three hashes of
    the groups file against the metadata of every input built from the locked-in basis: the
    stage's locked-in curves (``p_l``, ``p_rec``: ``source_sha256``, ``ilp_sha256``), the curve
    each run of the old and new bases discounts with (``rf_df``, which is ``p_l``: the same two)
    and their CPI path (``cpi``, the ``locked_in_cpi`` splice: ``rec_sha256``), each read from
    its verified copy. Such an input that lacks one of its metadata, or holds another value, is
    a LoadError naming it; so is a hash that no input the stage reads can verify, since a hash
    recorded unverified would say the group was measured on a basis nobody checked. Returns the
    three hashes and, under ``verified_by``, the files that verified each (label, sha256 and
    metadata key), which the manifest records."""
    at = groups.at(row)
    hashes = {name: groups.string(name, row) for name in LOCKED_IN_FIELDS}
    verified: dict[str, list[dict[str, str]]] = {name: [] for name in LOCKED_IN_FIELDS}

    def check(f: ReadFile, pairs: tuple[tuple[str, str], ...], why: str) -> None:
        for meta, name in pairs:
            got = f.metadata.get(meta)
            if got is None:
                raise LoadError(f"{f.where}: carries no metadata halley.{meta}; {why}")
            if got != hashes[name]:
                raise LoadError(
                    f"{f.where}: halley.{meta} is {got}, but {at} locks in {name} {hashes[name]}"
                )
            verified[name].append({"file": f.label, "sha256": f.sha256, "metadata": meta})

    for curve in LOCKED_IN_CURVES:
        f = stage_files.get(curve)
        if f is not None:
            check(
                f,
                _CURVE_METADATA,
                "a locked-in curve names the files it was built from (design section 10)",
            )
    for basis in (OLD_BASIS, NEW_BASIS):
        for r in runs.get(basis, ()):
            f = r.scenario(LOCKED_IN_RUN_CURVE)
            if f is not None:
                check(
                    f,
                    _CURVE_METADATA,
                    f"the runs of basis {basis} discount at the locked-in rates, with p_l as "
                    f"their {LOCKED_IN_RUN_CURVE}, which names the files it was built from "
                    "(design section 10)",
                )
            f = r.scenario(LOCKED_IN_CPI)
            if f is not None:
                check(
                    f,
                    _CPI_METADATA,
                    "the locked-in CPI path names the recognition path it splices (in-force 5.2)",
                )
    for name in LOCKED_IN_FIELDS:
        if not verified[name]:
            raise LoadError(
                f"{at} locks in {name} {hashes[name]}, but the stage reads no input that can "
                f"verify it ({_LOCKED_IN_SOURCES[name]}); a locked-in hash is verified, not "
                "only recorded (B72(b))"
            )
    return {**hashes, "verified_by": verified}


# ------------------------------------------------------------------------------ provenance


@dataclass(frozen=True, slots=True)
class ProvenanceCheck:
    """What a provenance helper receives (design sections 9 and 10).

    ``file`` is an input that names its source in ``halley.<key> = value``
    (:data:`PROVENANCE_KEYS`): a hash for ``rebased_from``, a run id for ``derived_from``. The
    helper recomputes the file from that source and returns the ``pyarrow.Table`` it must equal
    bit for bit. It finds a file by sha256 with ``find_file`` (among the stage's own inputs and
    in the directories of the previous run, the runs it read and the stage's bases, verified on
    read) and a run by id with ``find_run``; ``bases`` are the stage's runs by basis,
    ``previous`` the previous aggregation run (``None`` at recognition)."""

    key: str
    value: str
    file: ReadFile
    valuation_date: dt.date
    bases: Mapping[str, Sequence[RunResult]]
    previous: RunResult | None
    find_file: Callable[[str], ReadFile]
    find_run: Callable[[str], RunResult]


type ProvenanceHelper = Callable[[ProvenanceCheck], Any]
"""Recomputes a file from the source it names, as a ``pyarrow.Table``."""

PROVENANCE_HELPERS: dict[str, ProvenanceHelper] = {}
"""The registry: by metadata key, the helper that recomputes a file carrying it. The section 10
helpers (stage B3a) register themselves here with :func:`register_provenance`."""


def register_provenance(key: str, helper: ProvenanceHelper) -> None:
    """Register the helper that recomputes a file carrying ``halley.<key>``, replacing any."""
    if key not in PROVENANCE_KEYS:
        raise ValueError(f"a provenance key is one of {', '.join(PROVENANCE_KEYS)}, not {key!r}")
    PROVENANCE_HELPERS[key] = helper


class _Finder:
    """Where a provenance helper looks for a source: the stage's own input files, then the run
    directories of the previous run, the runs it read and the stage's bases."""

    def __init__(
        self,
        files: Sequence[ReadFile],
        prev: _Previous | None,
        runs: Mapping[str, Sequence[_BasisRun]],
        runs_root: Path | None,
    ) -> None:
        self._files = list(files)
        self._dirs: list[tuple[str, Path]] = []
        self._runs_root = runs_root
        if prev is not None:
            self._dirs.append((prev.who, prev.path))
            recorded: Any = prev.result.manifest.get("bases") or {}
            for basis, entries in recorded.items():
                for e in entries:
                    d = Path(str(e["dir"]))
                    if not d.is_absolute() and runs_root is not None:
                        d = Path(runs_root) / d
                    self._dirs.append((f"{prev.who}'s basis {basis} run {e['run_id']}", d))
        for rs in runs.values():
            for r in rs:
                self._dirs.append((f"basis {r.basis} run {r.run_id}", r.result.run_dir))

    def _manifests(self) -> Iterator[tuple[str, Path, dict[str, Any]]]:
        for what, d in self._dirs:
            if (d / "manifest.json").is_file():
                manifest, _sha = read_manifest(d)
                yield what, d, manifest

    def file(self, sha256: str) -> ReadFile:
        for f in self._files:
            if f.sha256 == sha256:
                return f
        for what, d, manifest in self._manifests():
            for name, rel, sha in run_files(manifest):
                if sha == sha256:
                    return read_copy(d, {"copy": rel, "sha256": sha}, f"{what}: {name}")
        raise LoadError(
            f"no file with sha256 {sha256} in the stage's inputs, the previous run, the runs it "
            "read or the stage's bases"
        )

    def run(self, run_id: str) -> RunResult:
        for _what, d, manifest in self._manifests():
            if manifest.get("run_id") == run_id:
                verify_run(d)
                return load_run(d)
        if self._runs_root is not None:
            d = find_run(run_id, self._runs_root)
            verify_run(d)
            return load_run(d)
        raise LoadError(f"no run {run_id} among the previous run, the runs it read or the bases")


def _provenance_files(
    runs: Mapping[str, Sequence[_BasisRun]], own: Sequence[ReadFile]
) -> Iterator[ReadFile]:
    """Every input that names its source: each copied input of each basis run whose copy, as
    read and verified, carries a provenance key (never the manifest's description of it, which
    no hash covers), then the stage's own files."""
    for rs in runs.values():
        for r in rs:
            for f in r.inputs.values():
                if any(key in f.metadata for key in PROVENANCE_KEYS):
                    yield f
    for f in own:
        if any(key in f.metadata for key in PROVENANCE_KEYS):
            yield f


def _verify_provenance(
    files: Iterator[ReadFile],
    finder: _Finder,
    valuation_date: dt.date,
    runs: Mapping[str, Sequence[_BasisRun]],
    prev: _Previous | None,
) -> list[dict[str, str]]:
    """Each file that names its source is recomputed by the helper registered for its key and
    must equal what it gives bit for bit (design section 9, "Provenance verification"); a file
    whose key has no helper is a LoadError too, since it cannot be verified. Returns what the
    manifest records of each."""
    bases = {b: [r.result for r in rs] for b, rs in runs.items()}
    previous = None if prev is None else prev.result
    record: list[dict[str, str]] = []
    for f in files:
        for key in PROVENANCE_KEYS:
            value = f.metadata.get(key)
            if value is None:
                continue
            helper = PROVENANCE_HELPERS.get(key)
            if helper is None:
                raise LoadError(
                    f"{f.where}: carries halley.{key} = {value}, but no helper is registered to "
                    "recompute it (register_provenance); a file that names its source is "
                    "verified against it (design section 9)"
                )
            check = ProvenanceCheck(
                key, value, f, valuation_date, bases, previous, finder.file, finder.run
            )
            try:
                table = helper(check)
            except LoadError:
                raise
            except Exception as exc:  # the helper's own failure
                raise LoadError(
                    f"{f.where}: recomputing it from halley.{key} = {value} raised "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            difference = file_difference(f, table)
            if difference is not None:
                raise LoadError(
                    f"{f.where}: it is not what its source (halley.{key} = {value}) gives: "
                    f"{difference}; a file that names its source equals the recomputed file bit "
                    "for bit"
                )
            record.append({"file": f.label, "sha256": f.sha256, "key": key, "value": value})
    return record


# ----------------------------------------------------------------------------- disclosures


def _disclosures(
    stage: type[Aggregation],
    runs: Mapping[str, Sequence[_BasisRun]],
    stage_files: Mapping[str, ReadFile],
) -> dict[str, Any]:
    """The manifest's ``disclosures`` block (design section 9, "Outputs"): the confidence level
    (para 119), ``alpha`` of the cur basis's ``ra_param`` copies; the sha256 of the para 120
    curve, the stage's ``df_cur`` file; and what the stage declares (:attr:`Aggregation.
    disclosures`, checked at build not to name either). ``None`` where the stage reads no such
    input; a cur run's ``ra_param`` that holds no single ``alpha`` is a LoadError naming the
    copy, since the disclosure would otherwise vanish."""
    table, column = CONFIDENCE_LEVEL
    levels: dict[float, str] = {}
    for r in runs.get(CUR_BASIS, ()):
        f = r.inputs.get(("table", table))
        if f is None:
            continue
        cols = table_columns(f)
        why = f"the cur basis's {table} holds the confidence level the run discloses (para 119)"
        if column not in cols:
            raise LoadError(f"{f.where}: column {column} is missing; {why}")
        values = cols[column]
        if values.dtype.kind not in "iuf":
            raise LoadError(f"{f.where}: column {column} is a number column; {why}")
        if len(values) != 1:
            raise LoadError(f"{f.where}: {len(values)} rows, not one; {why}")
        levels.setdefault(float(values[0]), r.run_id)
    if len(levels) > 1:
        listing = "; ".join(f"run {rid}: {level!r}" for level, rid in levels.items())
        raise LoadError(
            f"the runs of basis {CUR_BASIS} read different confidence levels: {listing}"
        )
    curve = stage_files.get(PARA_120_CURVE)
    declared = {k: list(v) if isinstance(v, tuple) else v for k, v in stage.disclosures.items()}
    return {
        **declared,
        "confidence_level": next(iter(levels), None),
        "para_120_curve_sha256": None if curve is None else curve.sha256,
    }


# ------------------------------------------------------------------------ the entry points


@dataclass(frozen=True, slots=True)
class StageInputs:
    """What :func:`load_aggregation_inputs` hands :func:`~halley.runtime.run.run`: the stage's
    model points, its group series and the record of what it read."""

    model_points: Any
    """A ``pyarrow.Table``: the groups file's rows with ``group_id`` and the derived fields."""
    series: dict[str, LoadedSeries]
    record: StageRecord
    scenarios: dict[str, ReadFile]
    """The stage's own scenario files as read and checked (the locked-in, provenance and
    disclosure checks), which the run reads and copies: the bytes checked are the bytes used."""
    tables: dict[str, ReadFile]
    """The stage's assumption tables, likewise."""


def _stage_of(compiled: CompiledModel) -> type[Aggregation]:
    model = compiled.model
    if not issubclass(model, Aggregation):
        raise LoadError(
            f"{compiled.name} is a projection; run_aggregation runs an aggregation stage"
        )
    return model


def load_aggregation_inputs(
    compiled: CompiledModel,
    *,
    groups: Source,
    bases: Mapping[str, Sequence[RunResult | Path | str]],
    scenarios: Mapping[str, Source] | None = None,
    tables: Mapping[str, Source] | None = None,
    previous: Path | None = None,
    settings: RunSettings,
) -> StageInputs:
    """The loader of design section 9, "Loader and runs", in order: each basis's runs from their
    directories, every file verified (:func:`_basis_run`) and checked as a :class:`Basis`, at
    the stage's valuation date; the groups file and its rules (:func:`_read_groups`); G1
    (:func:`require_one_group`); the previous run by ``prev_run_id`` (:func:`_resolve_previous`)
    and G4 against its members, or, without one, one group key per policy across the bases;
    every member mapped to exactly one group row, issued on its recognition date; the
    structural hashes; the locked-in hashes; the provenance of every file that names a source;
    the derived group fields; the group series, summed in ascending policy id across each
    basis's runs (:func:`group_series`); and the manifest blocks. A LoadError names the file and
    the row or key of the first failure."""
    stage = _stage_of(compiled)
    graph = compiled.graph
    valuation_date = settings.valuation_date
    declared = _bases(stage)
    for b in declared:
        if b.name not in bases:
            raise LoadError(f"{stage.__qualname__} reads basis {b.name}, which was not given")
    names = {b.name for b in declared}
    for name in bases:
        if name not in names:
            raise LoadError(
                f"basis {name!r} was given but {stage.__qualname__} declares no such basis"
            )
    runs: dict[str, list[_BasisRun]] = {
        b.name: [_basis_run(b.name, given) for given in _runs_of(b.name, bases[b.name])]
        for b in declared
    }
    for b in declared:
        info = b.check([r.result for r in runs[b.name]])
        if info.valuation_date != valuation_date:
            raise LoadError(
                f"basis {b.name} values at {info.valuation_date}, but the stage at "
                f"{valuation_date} (RunSettings.valuation_date); a stage and its bases share one "
                "valuation date"
            )
    group_rows = _read_groups(groups, valuation_date)
    require_one_group(group_rows.ids, group_rows.file.where)  # G1, the rule's one place
    prev = _resolve_previous(graph, group_rows, previous, settings)
    if prev is None:
        _check_keys_agree(runs)
    else:
        _check_g4(runs, prev)
    members = _members_by_group(runs, group_rows)
    _check_structure(runs)
    stage_files = {n: read_source(s, f"scenario input {n}") for n, s in (scenarios or {}).items()}
    table_files = {n: read_source(s, f"table {n}") for n, s in (tables or {}).items()}
    locked_in: dict[str, dict[str, Any]] = {}
    if any(name in graph.fields for name in LOCKED_IN_FIELDS):
        for row, gid in enumerate(group_rows.ids):
            locked_in[gid] = _check_locked_in(group_rows, row, stage_files, runs)
    derived: dict[str, list[Any]] = {GROUP_ID: list(group_rows.ids)}
    # a schema that declares one of these declares prev_run_id (a BuildError otherwise), so the
    # previous run is resolved whenever one is wanted
    wanted = [name for name in FROM_PREVIOUS if name in graph.fields]
    if prev is not None:
        rows = [
            _from_previous(graph, group_rows, row, prev, valuation_date)
            for row in range(len(group_rows.ids))
        ]
        for name in wanted:
            derived[name] = [values[name] for values in rows]
    own = [group_rows.file, *stage_files.values(), *table_files.values()]
    finder = _Finder(own, prev, runs, settings.runs_root)
    provenance = _verify_provenance(
        _provenance_files(runs, own), finder, valuation_date, runs, prev
    )
    series = group_series(
        stage,
        {name: [r.result for r in rs] for name, rs in runs.items()},
        members,
        runs_root=settings.runs_root,
    )
    root = settings.runs_root
    blocks: dict[str, Any] = {
        "previous": None
        if prev is None
        else {
            "run_id": prev.run_id,
            "dir": _dir_label(prev.path, root),
            "manifest_sha256": prev.manifest_sha256,
        },
        "bases": {
            name: [
                {
                    "run_id": r.run_id,
                    "dir": _dir_label(r.result.run_dir, root),
                    "manifest_sha256": r.manifest_sha256,
                    "outputs_sha256": _outputs_sha256(r.result.manifest),
                }
                for r in rs
            ]
            for name, rs in runs.items()
        },
        "membership": {
            "key": list(GROUP_KEY),
            "groups": {
                gid: {
                    name: [
                        _plain(p)
                        for p in _sorted_ids(
                            [m.pid for r in rs for m in r.members if m.group == gid]
                        ).tolist()
                    ]
                    for name, rs in runs.items()
                }
                for gid in group_rows.ids
            },
        },
        "locked_in": locked_in or None,
        "provenance": provenance,
        "disclosures": _disclosures(stage, runs, stage_files),
    }
    columns: dict[str, NDArray[Any]] = {
        name: np.asarray(values, dtype=_dtype(values)) for name, values in derived.items()
    }
    return StageInputs(
        with_columns(group_rows.file, columns),
        series,
        StageRecord(group_rows.file, with_columns(None, columns), blocks),
        stage_files,
        table_files,
    )


def _dtype(values: Sequence[object]) -> Any:
    """The NumPy dtype of a derived column: the group ids as objects (strings), the month counts
    as int64, the amounts as float64."""
    if all(isinstance(v, str) for v in values):
        return object
    if all(isinstance(v, int) and not isinstance(v, bool) for v in values):
        return np.int64
    return np.float64


def run_aggregation(
    compiled: CompiledModel,
    *,
    groups: Source,
    bases: Mapping[str, Sequence[RunResult | Path | str]],
    scenarios: Mapping[str, Source] | None = None,
    tables: Mapping[str, Source] | None = None,
    previous: Path | None = None,
    settings: RunSettings,
    run_dir: str | Path,
) -> RunResult:
    """Run an aggregation stage on its groups file and bases (design section 9): the inputs are
    loaded and checked by :func:`load_aggregation_inputs`, then the stage runs as any model does
    (:func:`~halley.runtime.run.run`), its outputs keyed on ``group_id`` with ``outputs.kind =
    "aggregation"``. Its run directory copies everything the stage read (the groups file,
    ``inputs/derived_groups.parquet``, every series with a mean's ``.pathwise`` sibling, the
    scenario and table files) and its manifest records the previous run, every run of every
    basis, the membership, the locked-in hashes, the provenance checks and the disclosures, so
    that :func:`replay_aggregation` re-runs it from the directory alone. Its run id covers the
    IR, the groups file, every series' values, every scenario and table file, the previous
    run's id and manifest hash, the settings but ``label``, ``runs_root`` and ``workers``, the
    backend and the horizon: not where any run lives."""
    if (Path(run_dir) / "manifest.json").exists():
        raise LoadError(f"{run_dir} already holds a run; use a new directory for each run")
    loaded = load_aggregation_inputs(
        compiled,
        groups=groups,
        bases=bases,
        scenarios=scenarios,
        tables=tables,
        previous=previous,
        settings=settings,
    )
    return run(
        compiled,
        model_points=loaded.model_points,
        tables=dict(loaded.tables) if tables is not None else None,
        scenarios=dict(loaded.scenarios) if scenarios is not None else None,
        settings=settings,
        run_dir=run_dir,
        series=loaded.series,
        stage_record=loaded.record,
    )


def replay_aggregation(
    compiled: CompiledModel, from_run: str | Path, *, run_dir: str | Path
) -> RunResult:
    """Re-run an aggregation run from its own directory alone (D21): every file the manifest
    records a hash for is verified, then the stage runs on the copies as read (the groups file
    and the derived group fields, every series, the scenario and table files) with the recorded
    settings, into ``run_dir``. The bases and the previous run are not read, so they may have
    moved or gone; the manifest blocks that record them are carried over, with
    ``replayed_from``. ``compiled`` is the stage that made the run: a run directory holds the
    generated code, not the model, and a stage with another IR is a LoadError.

    The replay is then compared with the run (:func:`_compare_replay`):
    ``replayed_from.matches`` records whether it has the run's id and outputs bit for bit, and
    a difference the environment does not explain is a RunError."""
    source = Path(from_run)
    manifest = verify_run(source, "replayed run")
    inputs: Mapping[str, Any] = manifest["inputs"]
    if manifest.get("outputs", {}).get("kind") != "aggregation" or "groups" not in inputs:
        raise LoadError(f"{source} holds no run made by run_aggregation; nothing to replay")
    if manifest.get("ir_sha256") != compiled.ir_hash:
        raise LoadError(
            f"{source} was made by a stage with ir_sha256 {manifest.get('ir_sha256')}, but "
            f"{compiled.name} has {compiled.ir_hash}; replay it with the stage that made it"
        )
    groups = read_copy(source, inputs["groups"], "groups")
    derived = read_copy(source, inputs["derived_groups"], "derived group fields")
    series = {
        name: read_series_copy(source, name, entry, GROUP_ID, f"series {name}")
        for name, entry in inputs["series"].items()
    }
    scenarios: dict[str, Source] = {
        name: read_copy(source, e, f"scenario input {name}")
        for name, e in inputs["scenarios"].items()
    }
    tables: dict[str, Source] = {
        name: read_copy(source, e, f"table {name}") for name, e in inputs["tables"].items()
    }
    blocks = {key: manifest[key] for key in STAGE_BLOCKS if key in manifest}
    blocks["replayed_from"] = {"run_id": manifest["run_id"], "dir": str(source.resolve())}
    record = StageRecord(groups, table_of(derived), blocks)
    result = run(
        compiled,
        model_points=with_columns(groups, table_columns(derived)),
        tables=tables,
        scenarios=scenarios,
        settings=RunSettings.from_dict(manifest["settings"]),
        run_dir=run_dir,
        series=series,
        stage_record=record,
    )
    _compare_replay(source, manifest, result)
    return result


RUN_ID_ENVIRONMENT: Final = ("name", "numpy")
"""The entries of a manifest's ``backend`` block that a run id covers (the backend's name and
the NumPy version); the output files' bytes depend on every entry (pyarrow writes its version
into them, and the platform's libm computes the powers)."""


def _compare_replay(source: Path, original: Mapping[str, Any], replay: RunResult) -> None:
    """Record in the replay's manifest whether it reproduced the run it replays: under
    ``replayed_from``, ``matches`` (the same run id and the same sha256 of both output files)
    and, when not, ``differences`` and the ``environment`` that changed (the entries of the
    manifest's ``backend`` block that differ). A run id that differs while the backend and
    NumPy are unchanged, or outputs that differ while nothing in the backend block changed, is a
    RunError (D21): the run is not reproducible from its own directory."""
    mine = replay.manifest
    differences: list[str] = []
    id_differs = replay.run_id != original["run_id"]
    if id_differs:
        differences.append(f"run id {replay.run_id} against {original['run_id']}")
    for key in ("monthly", "scalars"):
        a = str(mine["outputs"][f"{key}_sha256"])
        b = str(original["outputs"].get(f"{key}_sha256"))
        if a != b:
            differences.append(f"{key} outputs sha256 {a[:12]}... against {b[:12]}...")
    then: Mapping[str, Any] = original.get("backend", {})
    now: Mapping[str, Any] = mine["backend"]
    changed = {
        k: {"run": then.get(k), "replay": now.get(k)}
        for k in sorted(set(then) | set(now))
        if then.get(k) != now.get(k)
    }
    record: dict[str, Any] = mine["replayed_from"]
    record["matches"] = not differences
    if differences:
        record["differences"] = differences
        record["environment"] = changed
    write_manifest(replay.run_dir, mine)
    if id_differs and not any(k in changed for k in RUN_ID_ENVIRONMENT):
        same = "the same backend and NumPy, which the run id covers"
    elif differences and not changed:
        same = "the same backend, NumPy, pyarrow, Python and platform"
    else:
        return
    raise RunError(
        f"the replay of {source} (run {original['run_id']}) in {replay.run_dir} does not "
        f"reproduce it with {same}: {'; '.join(differences)}; every run is reproducible from "
        "its own directory (D21)"
    )
