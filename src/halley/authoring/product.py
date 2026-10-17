"""The ``Product`` base class, ``@var`` and ``Static``.

A model is a subclass of :class:`Product`. Each ``@var`` method is one variable: ``def x(self,
t: Int) -> Money`` for a monthly variable, ``def x(self) -> Money`` for one without a time
dimension. Build-time configuration is a ``Static[...]`` class attribute, plain Python during the
build (D8). Variants are subclasses; two parents defining the same variable differently is an
error until the subclass resolves it (D9).
"""

from __future__ import annotations

import inspect
import typing
from collections.abc import Callable
from typing import Any, ClassVar, cast, overload

from halley import ir
from halley.authoring.contracts import ScenarioInput, Schema, Table
from halley.authoring.units import Int, unit_from_annotation
from halley.errors import BuildError, Loc
from halley.ir.units import Unit
from halley.tracing.symbolic import Sym, as_node, current_loc, offset_of

type Static[T] = T
"""A build-time parameter: ``has_waiver: Static[bool] = False``. Evaluated during the build, so
an ``if self.has_waiver:`` is ordinary Python and the unused branch never enters the graph."""

TIMINGS = ("start", "mid", "end")
ROLES = ("in_force", "decrement", "reserve")
DOMAINS = ("state", "month")
RUN_CONSTANTS = ("m_val", "horizon")
"""The run constants a formula reads as ``self.run.<name>``: ``m_val`` is the valuation month
index ``12 * year + month - 1`` and ``horizon`` the run's ``H``. Both are scalar ``Int``."""


class VarDef:
    """What ``@var`` leaves on the class: the method and its declarations."""

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        init: object,
        terminal: object,
        timing: str | None,
        role: str | None,
        domain: str | None,
    ) -> None:
        self.fn = fn
        self.init = init
        self.terminal = terminal
        self.timing = timing
        self.role = role
        self.domain = domain
        self.name = fn.__name__
        self.has_time = _takes_time(fn)
        if timing is not None and timing not in TIMINGS:
            raise BuildError(f"timing must be one of {TIMINGS}, not {timing!r}", self.loc)
        if role is not None and role not in ROLES:
            raise BuildError(f"role must be one of {ROLES}, not {role!r}", self.loc)
        if domain is not None and domain not in DOMAINS:
            raise BuildError(f"domain must be one of {DOMAINS}, not {domain!r}", self.loc)
        if init is not None and terminal is not None:
            raise BuildError("a variable has an init or a terminal value, not both", self.loc)
        if not self.has_time and (init is not None or terminal is not None or domain):
            raise BuildError(
                "init, terminal and domain belong to variables with a time dimension (self, t)",
                self.loc,
            )

    @property
    def loc(self) -> Loc:
        code = self.fn.__code__
        return Loc(code.co_filename, code.co_firstlineno, self.fn.__qualname__)

    def declared_unit(self) -> Unit:
        """The unit of the method's return annotation, resolved when asked for (the annotation
        may be a string under ``from __future__ import annotations``)."""
        try:
            hints = typing.get_type_hints(self.fn)
        except Exception as exc:
            raise BuildError(
                f"cannot resolve the annotations of {self.name}: {exc}", self.loc
            ) from exc
        if "return" not in hints:
            raise BuildError(
                f"{self.name} declares no unit; write def {self.name}(self, t: Int) -> Money (or "
                "another unit)",
                self.loc,
            )
        return unit_from_annotation(hints["return"], self.loc)

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    def __get__(self, obj: object, objtype: type | None = None) -> Any:
        if obj is None:
            return self
        return BoundVar(self)


class BoundVar:
    """``self.x`` inside a formula: callable as ``self.x(t-1)`` or ``self.x()``."""

    __slots__ = ("var",)

    def __init__(self, var: VarDef) -> None:
        self.var = var

    def __call__(self, *args: object) -> Any:
        loc = current_loc()
        name = self.var.name
        if self.var.has_time:
            if len(args) != 1:
                raise BuildError(f"{name}(t) takes the time index with a constant offset", loc)
            return Sym(ir.VarRead(loc, name, offset_of(args[0], loc, name)))
        if args:
            raise BuildError(f"{name} has no time dimension; read it as self.{name}()", loc)
        return Sym(ir.VarRead(loc, name, 0))


def _loc_of(fn: Callable[..., Any]) -> Loc:
    code = fn.__code__
    return Loc(code.co_filename, code.co_firstlineno, fn.__qualname__)


def _takes_time(fn: Callable[..., Any]) -> bool:
    params = list(inspect.signature(fn).parameters)
    if params[:1] != ["self"]:
        raise BuildError("a @var method's first parameter is self", _loc_of(fn))
    rest = params[1:]
    if rest == []:
        return False
    if rest == ["t"]:
        return True
    raise BuildError(
        "a @var method is written def x(self, t: Int) or def x(self); other parameters are "
        "not allowed",
        _loc_of(fn),
    )


@overload
def var[F: Callable[..., Any]](fn: F, /) -> F: ...
@overload
def var[F: Callable[..., Any]](
    *,
    init: Callable[[Any], Any] | float | None = None,
    terminal: Callable[[Any], Any] | float | None = None,
    timing: str | None = None,
    role: str | None = None,
    domain: str | None = None,
) -> Callable[[F], F]: ...
def var[F: Callable[..., Any]](
    fn: F | None = None,
    /,
    *,
    init: Callable[[Any], Any] | float | None = None,
    terminal: Callable[[Any], Any] | float | None = None,
    timing: str | None = None,
    role: str | None = None,
    domain: str | None = None,
) -> F | Callable[[F], F]:
    """Declare a model variable.

    ``init`` gives the value at ``t = 0`` of a forward-recursive variable (a number, or a
    function of the model point); ``terminal`` the value at the horizon of a backward-recursive
    one. ``timing`` says when within month ``t`` a cash flow is paid: ``start``, ``mid`` or
    ``end``. ``role`` names the in-force count, the decrements and the reserve for the built-in
    invariants. ``domain="state"`` declares a value defined from ``t = 0`` (a variable with an
    init or terminal value, or one another formula needs at ``t = 0``, is a state without
    saying so); every other variable is a month quantity, defined from ``t = 1``.
    """

    def wrap(f: F) -> F:
        return cast(
            F, VarDef(f, init=init, terminal=terminal, timing=timing, role=role, domain=domain)
        )

    return wrap(fn) if fn is not None else wrap


# ------------------------------------------------------------------------------ accessors


class ModelPoint:
    """``self.mp``: the model point's fields and load-time derived fields."""

    def __init__(self, fields: dict[str, ir.FieldDecl]) -> None:
        self._fields = fields

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        loc = current_loc()
        decl = self._fields.get(name)
        if decl is None:
            known = ", ".join(sorted(self._fields))
            raise BuildError(f"the model point schema has no field {name!r}; it has {known}", loc)
        if not decl.readable:
            raise BuildError(
                f"mp.{name} is a {decl.kind} field; formulas read numbers, so derive an integer "
                "from it at load (a Derived field)",
                loc,
            )
        return Sym(ir.MpField(loc, name))


class TableColumn:
    def __init__(self, decl: ir.TableDecl, column: str) -> None:
        self._decl = decl
        self._column = column

    def __getitem__(self, keys: object) -> Any:
        loc = current_loc()
        key_tuple: tuple[object, ...] = (
            cast(tuple[object, ...], keys) if isinstance(keys, tuple) else (keys,)
        )
        if len(key_tuple) != len(self._decl.keys):
            raise BuildError(
                f"table {self._decl.name} has {len(self._decl.keys)} key(s): "
                f"{', '.join(k.name for k in self._decl.keys)}",
                loc,
            )
        nodes = tuple(as_node(k, loc) for k in key_tuple)
        return Sym(ir.TableLookup(loc, self._decl.name, self._column, nodes))


class TableAccess:
    """``self.tables.<name>``: ``[keys]`` looks a value up; ``.column`` names a value column."""

    def __init__(self, decl: ir.TableDecl) -> None:
        self._decl = decl

    def __getitem__(self, keys: object) -> Any:
        if len(self._decl.columns) != 1:
            names = ", ".join(c.name for c in self._decl.columns)
            raise BuildError(
                f"table {self._decl.name} has several value columns ({names}); write "
                f"tables.{self._decl.name}.<column>[keys]",
                current_loc(),
            )
        return TableColumn(self._decl, self._decl.columns[0].name)[keys]

    def __getattr__(self, column: str) -> Any:
        if column.startswith("_"):
            raise AttributeError(column)
        loc = current_loc()
        if column not in {c.name for c in self._decl.columns}:
            raise BuildError(f"table {self._decl.name} has no column {column!r}", loc)
        if not self._decl.keys:
            return Sym(ir.TableLookup(loc, self._decl.name, column, ()))
        return TableColumn(self._decl, column)


class Tables:
    def __init__(self, tables: dict[str, ir.TableDecl]) -> None:
        self._tables = tables

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        decl = self._tables.get(name)
        if decl is None:
            known = ", ".join(sorted(self._tables)) or "none"
            raise BuildError(
                f"the model declares no assumption table {name!r}; it declares: {known}",
                current_loc(),
            )
        return TableAccess(decl)


class ScenarioAccess:
    """``self.scen.<name>``: ``(t + k)`` reads a relative month, ``.at(k)`` an absolute one."""

    def __init__(self, decl: ir.ScenarioDecl) -> None:
        self._decl = decl

    def __call__(self, t: object) -> Any:
        loc = current_loc()
        return Sym(ir.ScenRead(loc, self._decl.name, offset_of(t, loc, f"scen.{self._decl.name}")))

    def at(self, index: object) -> Any:
        loc = current_loc()
        return Sym(ir.ScenAt(loc, self._decl.name, as_node(index, loc)))


class Scenarios:
    def __init__(self, scenarios: dict[str, ir.ScenarioDecl]) -> None:
        self._scenarios = scenarios

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        decl = self._scenarios.get(name)
        if decl is None:
            known = ", ".join(sorted(self._scenarios)) or "none"
            raise BuildError(
                f"the model declares no scenario input {name!r}; it declares: {known}",
                current_loc(),
            )
        return ScenarioAccess(decl)


class RunConstants:
    """``self.run``: the run's constants (:data:`RUN_CONSTANTS`), scalar ``Int`` values the
    loader fills from the valuation date and the horizon. ``self.run.m_val`` is the valuation
    month index, ``self.run.horizon`` the run's ``H``; boundary values may read them."""

    m_val: Int
    horizon: Int

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        loc = current_loc()
        if name not in RUN_CONSTANTS:
            raise BuildError(
                f"the run has no constant {name!r}; it has: {', '.join(RUN_CONSTANTS)}", loc
            )
        return Sym(ir.RunConst(loc, name))


# -------------------------------------------------------------------------------- Product


class Product:
    """Base class of every model.

    Class attributes a model sets:

    * ``model_point``: the :class:`~halley.authoring.contracts.Schema` of its model points;
    * ``assumption_tables``: the :class:`~halley.authoring.contracts.Table` contracts it reads;
    * ``scenario_inputs``: the :class:`~halley.authoring.contracts.ScenarioInput` contracts;
    * ``outputs``: the variables a run stores (``None`` stores them all);
    * ``tolerance_scales``: per unit, the model point field that scales invariant tolerances;
    * ``abstract``: ``True`` on a class that only exists to be subclassed; ``build()`` refuses
      it. The flag is not inherited: a class is abstract only when it sets it itself.
    """

    model_point: ClassVar[Schema | None] = None
    assumption_tables: ClassVar[tuple[Table, ...]] = ()
    scenario_inputs: ClassVar[tuple[ScenarioInput, ...]] = ()
    outputs: ClassVar[tuple[str, ...] | None] = None
    tolerance_scales: ClassVar[dict[str, str]] = {}
    abstract: ClassVar[bool] = False

    mp: ModelPoint
    tables: Tables
    scen: Scenarios
    run: RunConstants

    def __init__(self) -> None:
        raise TypeError(
            "a Product is not instantiated directly; halley.build(Model, **statics) traces it"
        )
