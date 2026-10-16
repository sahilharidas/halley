"""Data contracts a model declares: the model point schema, assumption tables and scenario inputs.

The loader validates every input file against these before a run (Spec, Data). A row outside a
range, a missing enum value, a lookup key outside a table's declared range or a scenario file
that does not cover the months the model reads is a :class:`~halley.errors.LoadError`, never a
silent adjustment.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from halley import ir
from halley.ir.units import ENUM, INT, Unit, unit_from_name

KINDS = ("number", "int", "enum", "date", "string")


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


@dataclass(frozen=True, slots=True)
class Field:
    """A model point input field.

    ``kind`` is ``number``, ``int``, ``enum``, ``date`` or ``string``. Numbers and ints carry a
    unit a formula can use; enums are table keys; dates and strings are not readable in formulas
    (derive integers from them at load).
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


@dataclass(frozen=True, slots=True)
class Table:
    """An assumption table: typed keys with declared ranges, value columns with units. The
    file's version and hash enter the run manifest."""

    name: str
    keys: tuple[Key, ...]
    columns: tuple[Column, ...]
    doc: str = ""

    def decl(self) -> ir.TableDecl:
        return ir.TableDecl(
            self.name, tuple(k.decl() for k in self.keys), tuple(c.decl() for c in self.columns)
        )


@dataclass(frozen=True, slots=True)
class ScenarioInput:
    """A scenario input: one value per scenario and month, read from a scenario file (D4)."""

    name: str
    unit: str
    doc: str = ""

    def decl(self) -> ir.ScenarioDecl:
        return ir.ScenarioDecl(self.name, unit_from_name(self.unit), self.doc)


def int_field(name: str, **kw: Any) -> Field:
    return Field(name, "int", INT.name, **kw)
