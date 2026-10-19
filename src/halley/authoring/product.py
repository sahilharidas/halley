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
from halley.authoring.contracts import InputRule, Invariant, ScenarioInput, Schema, Table
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

    ``init`` gives the value at ``t = 0`` of a forward-recursive variable; ``terminal`` the
    value at the horizon of a backward-recursive one. Each is a number or a function of the
    :class:`BoundaryScope`: model point fields (``lambda mp: mp.pols_if_init``) and the model's
    variables, read as a scalar (``lambda g: g.csm_0()``) or at a fixed month (``lambda g:
    at(g.fcf_g, 0)``), never relative to ``t``; a boundary may not read a fixed month of a
    member of its own recursive component. ``timing`` says when within month ``t`` a cash flow
    is paid: ``start``, ``mid`` or ``end``. ``role`` names the in-force count, the decrements
    and the reserve for the built-in invariants. ``domain="state"`` declares a value defined
    from ``t = 0`` (a variable with an init or terminal value, or one another formula needs at
    ``t = 0``, is a state without saying so); every other variable is a month quantity, defined
    from ``t = 1``.
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


class BoundaryScope(ModelPoint):
    """The object an ``init`` or ``terminal`` callable receives. A name is a model point field
    first (``lambda mp: mp.pols_if_init``) and otherwise a variable of the model, as a
    :class:`BoundVar` (``lambda g: g.csm_0()`` for a scalar, ``at(g.fcf_g, 0)`` for a fixed
    month); a name in neither namespace is an error that lists both. A name that is both a field
    and a variable is the field, so a model whose boundaries read variables keeps the two
    namespaces apart where they meet."""

    def __init__(self, fields: dict[str, ir.FieldDecl], variables: dict[str, VarDef]) -> None:
        super().__init__(fields)
        self._variables = variables

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._fields:
            return super().__getattr__(name)
        var = self._variables.get(name)
        if var is not None:
            return BoundVar(var)
        fields = ", ".join(sorted(self._fields))
        variables = ", ".join(self._variables)
        raise BuildError(
            f"a boundary value reads {name!r}, which is neither a model point field ({fields}) "
            f"nor a variable of the model ({variables})",
            current_loc(),
        )


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


class GroupSeries:
    """``self.series.<name>`` inside a stage's formula: ``(t + k)`` reads a relative month,
    ``.at(k)`` a fixed one, and ``()`` a series without a time dimension."""

    def __init__(self, decl: ir.SeriesDecl) -> None:
        self._decl = decl

    def __call__(self, *args: object) -> Any:
        loc = current_loc()
        name = self._decl.name
        if not self._decl.has_time:
            if args:
                raise BuildError(
                    f"series.{name} has no time dimension; read it as self.series.{name}()", loc
                )
            return Sym(ir.SeriesRead(loc, name, None))  # no month to be relative to
        if len(args) != 1:
            raise BuildError(
                f"series.{name}(t) takes the time index with a constant offset, as "
                f"self.series.{name}(t - 1)",
                loc,
            )
        arg = args[0]
        if isinstance(arg, int) and not isinstance(arg, bool):
            raise BuildError(
                f"series.{name}({arg}) reads a fixed month; write self.series.{name}.at({arg}) "
                f"for that, or self.series.{name}(t) for the current month",
                loc,
            )
        return Sym(ir.SeriesRead(loc, name, offset_of(arg, loc, f"series.{name}")))

    def at(self, index: object) -> Any:
        loc = current_loc()
        name = self._decl.name
        if not self._decl.has_time:
            raise BuildError(
                f"series.{name} has no time dimension; read it as self.series.{name}()", loc
            )
        return Sym(ir.SeriesAt(loc, name, as_node(index, loc)))


class SeriesAccess:
    """``self.series`` inside a stage's formula: the group series the stage declares, by name
    (:meth:`Product.series_declarations`)."""

    def __init__(self, series: dict[str, ir.SeriesDecl]) -> None:
        self._series = series

    def __getattr__(self, name: str) -> GroupSeries:
        if name.startswith("_"):
            raise AttributeError(name)
        decl = self._series.get(name)
        if decl is None:
            known = ", ".join(sorted(self._series)) or "none"
            raise BuildError(
                f"the stage declares no series {name!r}; it declares: {known}", current_loc()
            )
        return GroupSeries(decl)


class RunConstants:
    """``self.run``: the run's constants (:data:`RUN_CONSTANTS`), scalar ``Int`` values the
    loader fills from the valuation date and the horizon. ``self.run.m_val`` is the valuation
    month index, ``self.run.horizon`` the run's ``H``. Formulas read them; an ``init`` or
    ``terminal`` callable reaches them through a variable without a time dimension."""

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
    * ``input_rules``: the :class:`~halley.authoring.contracts.InputRule` checks across inputs,
      evaluated once every input is loaded;
    * ``outputs``: the variables a run stores (``None`` stores them all);
    * ``tolerance_scales``: per unit, the model point field that scales invariant tolerances;
    * ``invariants``: the :class:`~halley.authoring.contracts.Invariant` checks run on every
      batch after the built-in ones, in declaration order;
    * ``abstract``: ``True`` on a class that only exists to be subclassed; ``build()`` refuses
      it. The flag is not inherited: a class is abstract only when it sets it itself.

    A projection reads no group series; an aggregation stage
    (:class:`~halley.aggregation.stage.Aggregation`) does, through ``self.series``, and says
    which in :meth:`series_declarations`.
    """

    model_point: ClassVar[Schema | None] = None
    assumption_tables: ClassVar[tuple[Table, ...]] = ()
    scenario_inputs: ClassVar[tuple[ScenarioInput, ...]] = ()
    input_rules: ClassVar[tuple[InputRule, ...]] = ()
    outputs: ClassVar[tuple[str, ...] | None] = None
    tolerance_scales: ClassVar[dict[str, str]] = {}
    invariants: ClassVar[tuple[Invariant, ...]] = ()
    abstract: ClassVar[bool] = False

    mp: ModelPoint
    tables: Tables
    scen: Scenarios
    run: RunConstants

    def __init__(self) -> None:
        raise TypeError(
            "a Product is not instantiated directly; halley.build(Model, **statics) traces it"
        )

    @classmethod
    def series_declarations(cls) -> dict[str, ir.SeriesDecl] | None:
        """The group series the model reads, by name, validated; ``None`` for a projection,
        which has no ``self.series`` (reads across policies belong to the aggregation stage).
        The tracer calls it once, before it traces the formulas.

        A projection that sets ``bases`` or ``series`` as a stage does is a BuildError at the
        class: nothing would read the declarations, so the class was meant to be a stage. A
        ``@var`` or a build-time parameter of either name is the projection's own."""
        for name in _STAGE_DECLARATIONS:
            value: object = getattr(cls, name, None)
            if _is_stage_declaration(value):
                raise BuildError(
                    f"{cls.__qualname__} sets {name}, which an aggregation stage declares; a "
                    "projection reads no group series, so derive the class from "
                    "halley.Aggregation",
                    _class_loc(cls),
                )
        return None


_STAGE_DECLARATIONS = ("bases", "series")
"""The class attributes of an aggregation stage that a projection does not have."""


def _is_stage_declaration(value: object) -> bool:
    """A non-empty tuple or list (of ``Basis`` or ``GroupSum``), or one declaration written
    without the tuple around it (an object of :mod:`halley.aggregation`)."""
    if isinstance(value, (tuple, list)):
        return len(cast(tuple[object, ...], value)) > 0
    return type(value).__module__.startswith("halley.aggregation")


def _class_loc(cls: type) -> Loc | None:
    """Where a class is defined, for an error about the class rather than a method."""
    try:
        file = inspect.getsourcefile(cls)
        _lines, line = inspect.getsourcelines(cls)
    except (OSError, TypeError):
        return None
    return None if file is None else Loc(file, line, cls.__qualname__)
