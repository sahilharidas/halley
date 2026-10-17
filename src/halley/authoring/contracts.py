"""Data contracts a model declares: the model point schema, assumption tables, scenario inputs,
the rules on them, and the invariants checked on every run.

The loader validates every input file against these before a run (Spec, Data). A row outside a
range, a missing enum value, a lookup key outside a table's declared range, a scenario file that
does not cover the months the model reads, or a failing rule is a
:class:`~halley.errors.LoadError`, never a silent adjustment. An :class:`Invariant` is checked on
every batch after the projection; a failure is a :class:`~halley.errors.RunError`.
"""

from __future__ import annotations

import datetime as dt
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from halley import ir
from halley.errors import Loc
from halley.ir.units import ENUM, INT, Unit, unit_from_name

KINDS = ("number", "int", "enum", "date", "string")

INVARIANT_KINDS = ("eq", "ge", "le")
"""``eq``: ``|lhs - rhs| <= tol * scale``; ``ge``: ``lhs >= rhs - tol * scale``; ``le``:
``lhs <= rhs + tol * scale``."""


class LoadContext:
    """What a derived field or rule sees: every field loaded so far as a NumPy array, by
    attribute, plus the valuation date."""

    def __init__(self, columns: dict[str, NDArray[Any]], valuation_date: dt.date) -> None:
        self._columns = columns
        self.valuation_date = valuation_date
        self.valuation = np.datetime64(valuation_date, "D")

    def __getattr__(self, name: str) -> NDArray[Any]:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._columns[name]
        except KeyError:
            raise AttributeError(f"no field {name!r} has been loaded yet") from None

    def __getitem__(self, name: str) -> NDArray[Any]:
        return self._columns[name]


# ------------------------------------------------------------------------------ month specs


def resolve_month(spec: int | str, horizon: int) -> int:
    """A month written as an int, ``"H"`` or ``"H-k"``, resolved at the run's horizon."""
    if isinstance(spec, int):
        return spec
    if spec == "H":
        return horizon
    if spec.startswith("H-") and spec[2:].isdigit():
        return horizon - int(spec[2:])
    raise ValueError(f"a month is an int, 'H' or 'H-k', not {spec!r}")


def _check_month_spec(spec: object, what: str) -> None:
    if isinstance(spec, bool) or not isinstance(spec, (int, str)):
        raise ValueError(f"{what}: a month is an int, 'H' or 'H-k', not {spec!r}")
    if isinstance(spec, str):
        try:
            resolve_month(spec, 0)
        except ValueError as exc:
            raise ValueError(f"{what}: {exc}") from None


# ----------------------------------------------------------------------------- model points


@dataclass(frozen=True, slots=True)
class Field:
    """A model point input field.

    ``kind`` is ``number``, ``int``, ``enum``, ``date`` or ``string``. Numbers and ints carry a
    unit a formula can use; enums are table keys; dates and strings are not readable in formulas
    (derive integers from them at load). A ``string`` column may be dictionary-encoded in the
    file.
    """

    name: str
    kind: str
    unit: str | None = None
    enum: tuple[str, ...] | None = None
    gt: float | None = None
    ge: float | None = None
    lt: float | None = None
    le: float | None = None
    unique: bool = False
    doc: str = ""

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"field {self.name}: kind must be one of {KINDS}")
        if self.kind == "enum" and not self.enum:
            raise ValueError(f"field {self.name}: an enum field lists its values")
        if self.kind in ("number", "int") and self.unit is None:
            raise ValueError(f"field {self.name}: a {self.kind} field declares its unit")

    def decl(self, derived: bool = False) -> ir.FieldDecl:
        unit: Unit | None
        if self.kind == "enum":
            unit = ENUM
        elif self.unit is not None:
            unit = unit_from_name(self.unit)
        else:
            unit = None
        return ir.FieldDecl(
            self.name,
            self.kind,
            unit,
            self.enum,
            self.gt,
            self.ge,
            self.lt,
            self.le,
            self.unique,
            derived,
            self.doc,
        )


@dataclass(frozen=True, slots=True)
class Derived:
    """A field computed once per model point at load, from fields and earlier derived fields."""

    name: str
    unit: str
    fn: Callable[[LoadContext], NDArray[Any]]
    kind: str = "int"
    ge: float | None = None
    le: float | None = None
    doc: str = ""

    def field(self) -> Field:
        return Field(self.name, self.kind, self.unit, ge=self.ge, le=self.le, doc=self.doc)


@dataclass(frozen=True, slots=True)
class Rule:
    """A cross-field rule, true for every valid model point."""

    name: str
    check: Callable[[LoadContext], NDArray[np.bool_]]
    doc: str = ""


@dataclass(frozen=True, slots=True)
class Schema:
    """The model point schema: fields, derived fields, cross-field rules, the field that sets
    the run horizon, and a rule on the valuation date."""

    fields: tuple[Field, ...]
    derived: tuple[Derived, ...] = ()
    rules: tuple[Rule, ...] = ()
    horizon: str | None = None
    valuation_date_rule: Callable[[dt.date], str | None] | None = None
    id_field: str = "policy_id"

    def __post_init__(self) -> None:
        names = [f.name for f in self.fields] + [d.name for d in self.derived]
        if len(set(names)) != len(names):
            raise ValueError("schema field names must be unique")
        if self.id_field not in names:
            raise ValueError(f"schema needs the identifier field {self.id_field!r}")
        if self.horizon is not None and self.horizon not in names:
            raise ValueError(f"horizon field {self.horizon!r} is not a schema field")

    def decls(self) -> dict[str, ir.FieldDecl]:
        out = {f.name: f.decl() for f in self.fields}
        out.update({d.name: d.field().decl(derived=True) for d in self.derived})
        return out


# ----------------------------------------------------------------------------------- tables


@dataclass(frozen=True, slots=True)
class Key:
    """A table key: an integer in ``[lo, hi]`` or an enum with the listed values."""

    name: str
    enum: tuple[str, ...] | None = None
    lo: int | None = None
    hi: int | None = None

    def __post_init__(self) -> None:
        if self.enum is None and (self.lo is None or self.hi is None):
            raise ValueError(f"key {self.name}: an integer key declares lo and hi")

    def decl(self) -> ir.KeyDecl:
        return ir.KeyDecl(self.name, self.enum, self.lo, self.hi)


@dataclass(frozen=True, slots=True)
class Column:
    name: str
    unit: str
    ge: float | None = None
    le: float | None = None

    def decl(self) -> ir.ColumnDecl:
        return ir.ColumnDecl(self.name, unit_from_name(self.unit), self.ge, self.le)


class TableView:
    """What a :class:`TableRule` sees: the loaded table's dense value arrays by column (one axis
    per key, in declared order; 0-d for a keyless table), its key names and ranges, and the
    file's ``halley.*`` metadata as strings, keys without the prefix.

    ``view.q`` or ``view["q"]`` is the dense array of column ``q``; ``view.at("q", age=120)``
    fixes keys by value and keeps every axis, so a comparison on it still names the failing key.
    """

    def __init__(
        self,
        name: str,
        keys: tuple[ir.KeyDecl, ...],
        values: dict[str, NDArray[Any]],
        metadata: dict[str, str],
    ) -> None:
        self.name = name
        self._keys = keys
        self._values = values
        self.metadata = metadata

    @property
    def keys(self) -> tuple[str, ...]:
        """The key names, in axis order."""
        return tuple(k.name for k in self._keys)

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(self._values)

    @property
    def shape(self) -> tuple[int, ...]:
        """The dense shape: one axis per key."""
        return tuple(len(self.key_values(k.name)) for k in self._keys)

    def _key(self, name: str) -> tuple[int, ir.KeyDecl]:
        for i, k in enumerate(self._keys):
            if k.name == name:
                return i, k
        raise KeyError(f"table {self.name} has no key {name!r}; its keys are {self.keys}")

    def key_values(self, name: str) -> NDArray[Any]:
        """The values along a key's axis: the enum's strings, or ``lo..hi``."""
        _, k = self._key(name)
        if k.enum is not None:
            return np.array(k.enum, dtype=object)
        return np.arange(int(k.lo or 0), int(k.hi or 0) + 1, dtype=np.int64)

    def key_range(self, name: str) -> tuple[int, int]:
        """``(lo, hi)`` of an integer key."""
        _, k = self._key(name)
        if k.enum is not None:
            raise KeyError(f"key {name} of table {self.name} is an enum, not a range")
        return int(k.lo or 0), int(k.hi or 0)

    def index(self, name: str, value: object) -> int:
        """The axis index of a key value."""
        _, k = self._key(name)
        if k.enum is not None:
            if value not in k.enum:
                raise KeyError(f"{value!r} is not a value of key {name} of table {self.name}")
            return k.enum.index(str(value))
        lo, hi = self.key_range(name)
        i = int(value)  # pyright: ignore[reportArgumentType]
        if i < lo or i > hi:
            raise KeyError(f"{i} is outside key {name} of table {self.name}, [{lo}, {hi}]")
        return i - lo

    def at(self, column: str, **fixed: object) -> NDArray[Any]:
        """Column ``column`` with the named keys fixed at the given values; the fixed axes are
        kept with length 1 so the result still lines up with the dense shape."""
        arr = self[column]
        for name, value in fixed.items():
            axis, _ = self._key(name)
            arr = np.take(arr, [self.index(name, value)], axis=axis)
        return arr

    def __getitem__(self, column: str) -> NDArray[Any]:
        try:
            return self._values[column]
        except KeyError:
            raise KeyError(
                f"table {self.name} has no column {column!r}; its columns are {self.columns}"
            ) from None

    def __getattr__(self, column: str) -> NDArray[Any]:
        if column.startswith("_"):
            raise AttributeError(column)
        try:
            return self._values[column]
        except KeyError:
            raise AttributeError(
                f"table {self.name} has no column {column!r}; its columns are {self.columns}"
            ) from None


@dataclass(frozen=True, slots=True)
class TableRule:
    """A rule on one loaded table: ``check(view)`` returns a bool, or a boolean array over the
    table's dense shape (or one with fixed axes of length 1, as ``view.at`` gives), true where
    the rule holds. A failure names the table, the rule and the first failing key."""

    name: str
    check: Callable[[TableView], Any]
    doc: str = ""


@dataclass(frozen=True, slots=True)
class Table:
    """An assumption table: typed keys with declared ranges, value columns with units. The
    file's version and hash enter the run manifest. ``metadata`` names the ``halley.<key>``
    schema-metadata keys the file must carry; ``rules`` are checked on the loaded table."""

    name: str
    keys: tuple[Key, ...]
    columns: tuple[Column, ...]
    doc: str = ""
    metadata: tuple[str, ...] = ()
    rules: tuple[TableRule, ...] = ()

    def decl(self) -> ir.TableDecl:
        return ir.TableDecl(
            self.name, tuple(k.decl() for k in self.keys), tuple(c.decl() for c in self.columns)
        )


# -------------------------------------------------------------------------------- scenarios


class ScenarioView:
    """What a :class:`ScenarioRule` sees: the dense ``(months, scenarios)`` array of a loaded
    scenario input, where row ``i`` is month ``u0 + i``."""

    def __init__(self, name: str, values: NDArray[np.float64], u0: int) -> None:
        self.name = name
        self.values = values
        self.u0 = u0

    @property
    def n_scenarios(self) -> int:
        return int(self.values.shape[1])

    @property
    def u_max(self) -> int:
        return self.u0 + int(self.values.shape[0]) - 1

    @property
    def months(self) -> NDArray[np.int64]:
        """The month of every row."""
        return np.arange(self.u0, self.u_max + 1, dtype=np.int64)

    def at(self, u: int) -> NDArray[np.float64]:
        """Month ``u`` across scenarios, shape ``(scenarios,)``."""
        if u < self.u0 or u > self.u_max:
            raise KeyError(
                f"scenario input {self.name} has no month {u}; it covers {self.u0} to {self.u_max}"
            )
        return self.values[u - self.u0]

    def rows(self, lo: int, hi: int) -> NDArray[np.float64]:
        """Months ``lo..hi`` across scenarios, shape ``(hi - lo + 1, scenarios)``."""
        if lo < self.u0 or hi > self.u_max or lo > hi:
            raise KeyError(
                f"scenario input {self.name} does not cover months {lo} to {hi}; it covers "
                f"{self.u0} to {self.u_max}"
            )
        return self.values[lo - self.u0 : hi - self.u0 + 1]


@dataclass(frozen=True, slots=True)
class ScenarioRule:
    """A rule on one loaded scenario input: ``check(view)`` returns a bool, or a boolean array
    shaped like the view's values, ``(scenarios,)`` or ``(months,)``, true where the rule
    holds."""

    name: str
    check: Callable[[ScenarioView], Any]
    doc: str = ""


@dataclass(frozen=True, slots=True)
class ScenarioInput:
    """A scenario input: one value per scenario and month, read from a scenario file (D4).

    The file holds ``scenario``, a month column named ``t`` or ``u`` (not both), and the value
    column, named ``column`` when that differs from the input's name. ``gt``, ``ge``, ``lt``,
    ``le`` bound every value; ``covers=(lo, hi)`` demands that the file cover at least months
    ``lo..hi`` (``hi`` may be ``"H"`` or ``"H-k"``) whatever months the model reads; ``rules``
    are checked on the loaded input.
    """

    name: str
    unit: str
    doc: str = ""
    gt: float | None = None
    ge: float | None = None
    lt: float | None = None
    le: float | None = None
    rules: tuple[ScenarioRule, ...] = ()
    column: str | None = None
    covers: tuple[int, int | str] | None = None

    def __post_init__(self) -> None:
        if self.covers is not None:
            lo, hi = self.covers
            _check_month_spec(lo, f"scenario input {self.name}: covers")
            _check_month_spec(hi, f"scenario input {self.name}: covers")

    @property
    def file_column(self) -> str:
        return self.column or self.name

    def decl(self) -> ir.ScenarioDecl:
        return ir.ScenarioDecl(self.name, unit_from_name(self.unit), self.doc)


# ------------------------------------------------------------------------------ input rules


@dataclass(frozen=True, slots=True)
class InputView:
    """What an :class:`InputRule` sees, after every input is loaded: the tables by name as
    :class:`TableView`, the model points as a :class:`LoadContext` (every field and derived
    field, by attribute), the scenario inputs by name as :class:`ScenarioView`, the valuation
    date, the valuation month index ``12 * year + month - 1`` (the run constant ``m_val``) and
    the horizon ``H``."""

    tables: dict[str, TableView]
    model_points: LoadContext
    scenarios: dict[str, ScenarioView]
    valuation_date: dt.date
    valuation_month: int
    horizon: int


@dataclass(frozen=True, slots=True)
class InputRule:
    """A rule across inputs, checked once every input is loaded: ``check(view)`` returns a bool
    or a boolean array, true where the rule holds."""

    name: str
    check: Callable[[InputView], Any]
    doc: str = ""


# -------------------------------------------------------------------------------- invariants


class BatchView(Protocol):
    """What an :class:`Invariant`'s callables see: one batch of policies after projection.

    Arrays are ``(time, policy, scenario)``: ``a.var(name)`` is the variable broadcast to
    ``(H + 1, P, S)`` (``(1, P, S)`` for a variable without a time dimension), ``a.mp.<field>``
    a model point field shaped ``(1, P, 1)`` (enums as their strings, dates as
    ``datetime64[D]``), ``a.t`` the month index shaped ``(H + 1, 1, 1)``, ``a.H`` the horizon,
    ``a.run.m_val`` a run constant, ``a.scen(name, u)`` a scenario month shaped ``(1, 1, S)``,
    ``a.scen_range(name, lo, hi)`` months ``lo..hi`` shaped ``(hi - lo + 1, 1, S)``,
    ``a.table(name, column)`` a dense table column, ``a.at(name, index)`` a variable at a fixed
    month, an int or a per-policy ``(1, P, 1)`` index, shaped ``(1, P, S)``, and
    ``a.scale("Money")`` the per-policy tolerance scale of a unit shaped ``(1, P, 1)``.
    """

    H: int
    t: NDArray[np.int64]
    mp: Any
    run: Any

    def var(self, name: str) -> NDArray[Any]: ...
    def at(self, name: str, index: Any) -> NDArray[Any]: ...
    def scen(self, name: str, u: int) -> NDArray[np.float64]: ...
    def scen_range(self, name: str, lo: int, hi: int) -> NDArray[np.float64]: ...
    def table(self, name: str, column: str) -> NDArray[Any]: ...
    def scale(self, unit: str) -> NDArray[np.float64]: ...


def _declaration_loc() -> Loc | None:
    """The model source line that constructs a contract: the nearest frame outside this module
    and outside generated code."""
    frame = sys._getframe(1)  # pyright: ignore[reportPrivateUsage]
    while frame is not None:
        name = frame.f_code.co_filename
        if name != __file__ and not name.startswith("<"):
            return Loc(name, frame.f_lineno, frame.f_code.co_qualname)
        frame = frame.f_back
    return None


@dataclass(frozen=True, slots=True)
class Invariant:
    """A product-declared invariant, checked on every batch after the projection (term H4, H5;
    annuity I3 to I10).

    ``lhs``, ``rhs`` and ``where`` are callables over a :class:`BatchView` (or numbers); ``kind``
    is ``eq`` (``|lhs - rhs| <= tol * scale``), ``ge`` (``lhs >= rhs - tol * scale``) or ``le``
    (``lhs <= rhs + tol * scale``); ``tol = 0`` means exact. ``scale`` is a number or a callable
    ``(a, lhs, rhs) -> array``, so the spec's ``max(|lhs|, |rhs|, S)`` is one line. The arrays
    the callables return cover the whole time axis ``0..H`` or, with ``months=(first, last)``
    (``last`` may be ``"H"`` or ``"H-1"``), that window: a full-length result is sliced to the
    window, a result of the window's length is taken as is, and the report names the failing
    month accordingly; a result with one time row is a single month. ``where`` masks the months
    and policies the check applies to. ``reads`` names every variable the callables read, so the
    build keeps them generated even when nothing else needs them.
    """

    name: str
    lhs: Callable[[BatchView], Any] | float
    rhs: Callable[[BatchView], Any] | float = 0.0
    kind: str = "eq"
    scale: Callable[[BatchView, Any, Any], Any] | float = 1.0
    tol: float = 1e-12
    months: tuple[int | str, int | str] | None = None
    where: Callable[[BatchView], Any] | None = None
    reads: tuple[str, ...] = ()
    doc: str = ""
    loc: Loc | None = field(default=None, compare=False)
    """Where the invariant is declared; filled in when left out."""

    def __post_init__(self) -> None:
        if self.kind not in INVARIANT_KINDS:
            raise ValueError(
                f"invariant {self.name}: kind must be one of {INVARIANT_KINDS}, not {self.kind!r}"
            )
        if self.tol < 0:
            raise ValueError(f"invariant {self.name}: tol must be at least 0")
        if self.months is not None:
            first, last = self.months
            _check_month_spec(first, f"invariant {self.name}: months")
            _check_month_spec(last, f"invariant {self.name}: months")
        if self.loc is None:
            object.__setattr__(self, "loc", _declaration_loc())

    def window(self, horizon: int) -> tuple[int, int]:
        """``(first, last)`` of the months the invariant covers at ``horizon``."""
        if self.months is None:
            return 0, horizon
        return resolve_month(self.months[0], horizon), resolve_month(self.months[1], horizon)


def int_field(name: str, **kw: Any) -> Field:
    return Field(name, "int", INT.name, **kw)
