"""Tracing: run each ``@var`` method once with a symbolic ``t`` to capture the graph (D8).

Build-time configuration (``Static[...]``) is plain Python here, so an ``if`` on it runs and the
unused branch never enters the graph. Anything that depends on data or time comes back as a
symbolic value, and a Python ``if`` on it is an error that names ``where()``.
"""

from __future__ import annotations

import inspect
import textwrap
import typing
from collections.abc import Callable
from typing import Any, cast

from halley import ir
from halley.authoring.contracts import InputRule, Invariant, ScenarioInput, Table
from halley.authoring.product import (
    BoundaryScope,
    ModelPoint,
    Product,
    RunConstants,
    Scenarios,
    SeriesAccess,
    Static,
    Tables,
    VarDef,
)
from halley.errors import BuildError, HalleyError, Loc
from halley.ir.units import Unit, unit_from_name
from halley.tracing.symbolic import Sym, TraceContext, as_node, set_context


def trace(model: type[Product], statics: dict[str, object] | None = None) -> ir.Graph:
    """Trace a model class into a graph, with the given build-time settings. An abstract class
    (one that sets ``abstract = True`` itself) is refused: only its subclasses are models."""
    if not (
        isinstance(model, type) and issubclass(model, Product)  # pyright: ignore[reportUnnecessaryIsInstance]
    ):
        raise BuildError(f"{model!r} is not a Product subclass")
    # ``is True``: the flag is a bool; a @var named abstract is a VarDef, which the reserved-name
    # check below reports at its definition
    if vars(model).get("abstract") is True:
        raise BuildError(
            f"{model.__qualname__} is abstract (it sets abstract = True); build one of its "
            "subclasses",
            _class_loc(model),
        )
    resolved = _resolve_statics(model, statics or {})
    var_defs = _collect_vars(model)
    if not var_defs:
        raise BuildError(f"{model.__qualname__} declares no @var")
    schema = model.model_point
    if schema is None:
        raise BuildError(f"{model.__qualname__} declares no model_point schema")
    fields = schema.decls()
    _require_tuple_of(model, "assumption_tables", Table)
    _require_tuple_of(model, "scenario_inputs", ScenarioInput)
    tables = {t.name: t.decl() for t in model.assumption_tables}
    scenarios = {s.name: s.decl() for s in model.scenario_inputs}
    series = model.series_declarations()  # None for a projection: it has no self.series
    model.check_statics(resolved)  # a stage's disclosures that read the statics, say

    inst: Any = model.__new__(model)
    for name, value in resolved.items():
        setattr(inst, name, value)
    inst.mp = ModelPoint(fields)
    inst.tables = Tables(tables)
    inst.scen = Scenarios(scenarios)
    inst.run = RunConstants()
    if series is not None:
        inst.series = SeriesAccess(series)  # shadows the class's tuple of declarations
    scope = BoundaryScope(fields, var_defs)

    variables: dict[str, ir.Variable] = {}
    for vd in var_defs.values():
        variables[vd.name] = _trace_var(vd, inst, scope)

    outputs = model.outputs
    if outputs is not None:
        for name in outputs:
            if name not in variables:
                raise BuildError(f"{model.__qualname__}.outputs names no variable {name!r}")
    scales: dict[Unit, str] = {}
    for unit_name, field_name in model.tolerance_scales.items():
        if field_name not in fields:
            raise BuildError(
                f"tolerance_scales names {field_name!r}, which is not a model point field"
            )
        scales[unit_from_name(unit_name)] = field_name
    _require_tuple_of(model, "invariants", Invariant)
    _require_tuple_of(model, "input_rules", InputRule)
    invariant_reads: set[str] = set()
    for inv in model.invariants:
        for name in inv.reads:
            if name not in variables:
                raise BuildError(
                    f"invariant {inv.name} reads {name!r}, which is not a variable of "
                    f"{model.__qualname__}",
                    inv.loc,
                )
            invariant_reads.add(name)

    source_file = inspect.getsourcefile(model) or "<unknown>"
    return ir.Graph(
        model=f"{model.__module__}.{model.__qualname__}",
        source_file=source_file,
        statics=resolved,
        variables=variables,
        fields=fields,
        tables=tables,
        scenarios=scenarios,
        horizon_field=schema.horizon,
        outputs=outputs,
        scale_fields=scales,
        invariant_reads=frozenset(invariant_reads),
        series=dict(series or {}),
    )


def _class_loc(model: type) -> Loc | None:
    """Where the class itself is defined, for an error about the class rather than a method."""
    try:
        file = inspect.getsourcefile(model)
        _lines, line = inspect.getsourcelines(model)
    except (OSError, TypeError):
        return None
    return None if file is None else Loc(file, line, model.__qualname__)


def _require_tuple_of(model: type[Product], name: str, cls: type) -> None:
    """``model.<name>`` is a tuple of ``cls``. The realistic slip is a one-element tuple written
    without its comma, ``invariants = (Invariant(...))``, which arrives as the element itself
    and would otherwise surface as a TypeError far from the class."""
    value: object = getattr(model, name)
    if isinstance(value, tuple):
        items = cast(tuple[object, ...], value)  # the elements are checked one by one below
        if all(isinstance(item, cls) for item in items):
            return
    raise BuildError(
        f"{model.__qualname__}.{name} must be a tuple of {cls.__name__} (a one-element tuple "
        "needs a trailing comma)",
        _class_loc(model),
    )


def _resolve_statics(model: type[Product], overrides: dict[str, object]) -> dict[str, object]:
    declared: dict[str, object] = {}
    kinds: dict[str, object] = {}
    for klass in reversed(model.__mro__):
        if klass is object or klass is Product:
            continue
        try:
            hints = typing.get_type_hints(klass)
        except Exception as exc:
            raise BuildError(
                f"cannot resolve the annotations of {klass.__qualname__}: {exc}"
            ) from exc
        for name, hint in hints.items():
            if getattr(hint, "__origin__", None) is Static:
                if name in _RESERVED_STATICS:
                    raise BuildError(
                        f"{klass.__qualname__}.{name} is a build-time parameter, but {name} is "
                        f"{_RESERVED_STATICS[name]}; rename the parameter",
                        _class_loc(klass),
                    )
                declared[name] = getattr(model, name, _MISSING)
                args: tuple[object, ...] = getattr(hint, "__args__", ())
                kinds[name] = args[0] if len(args) == 1 else None
    for name in overrides:
        if name not in declared:
            known = ", ".join(sorted(declared)) or "none"
            raise BuildError(
                f"{model.__qualname__} has no build-time parameter {name!r}; it has: {known}"
            )
    resolved = dict(declared)
    resolved.update(overrides)
    missing = [n for n, v in resolved.items() if v is _MISSING]
    if missing:
        raise BuildError(
            f"build-time parameter(s) without a value: {', '.join(missing)}; "
            "give a default or pass them to build()"
        )
    for name, value in resolved.items():
        _check_static(model, name, kinds.get(name), value, given=name in overrides)
    return resolved


def _check_static(
    model: type[Product], name: str, kind: object, value: object, *, given: bool
) -> None:
    """A build-time parameter declared ``Static[T]`` holds a ``T``, as given to ``build()`` or as
    the class's default: ``Static[bool]`` True or False (the string ``"False"`` is truthy and
    would build the other branch), ``Static[int]`` an integer that is not a bool,
    ``Static[float]`` a number that is not a bool, each a Python number, as a formula takes it (a
    NumPy scalar is not a constant of the graph). A ``T`` that is not a class (a union, a
    ``Literal``) is not checked. A BuildError at the class otherwise."""
    if not isinstance(kind, type):
        return
    if kind is bool:
        ok, wanted = value is True or value is False, "True or False"
    elif kind is int:
        ok, wanted = isinstance(value, int), "an integer"
    elif kind is float:
        ok, wanted = isinstance(value, (int, float)), "a number"
    else:
        ok, wanted = isinstance(value, kind), f"a {kind.__name__}"
    if isinstance(value, bool) and kind is not bool:
        ok = False  # True is an int to Python, never a month count or a rate
    if ok:
        return
    where = f"build({model.__qualname__}, {name}=...)" if given else f"{model.__qualname__}.{name}"
    raise BuildError(
        f"{where}: {name} is a build-time parameter declared Static[{kind.__name__}], so its "
        f"value is {wanted}, not {value!r} ({type(value).__name__})",
        _class_loc(model),
    )


_MISSING = object()

_RESERVED_NAMES = {
    "mp": "the model point accessor",
    "tables": "the assumption-table accessor",
    "scen": "the scenario accessor",
    "run": "the run-constants accessor",
    "model_point": "a Product class attribute",
    "assumption_tables": "a Product class attribute",
    "scenario_inputs": "a Product class attribute",
    "input_rules": "a Product class attribute",
    "outputs": "a Product class attribute",
    "tolerance_scales": "a Product class attribute",
    "invariants": "a Product class attribute",
    "abstract": "a Product class attribute",
    "series_declarations": "a Product class method",
    "check_statics": "a Product class method",
}
"""Names a variable may not take: the tracer sets the accessors on the instance, where they would
shadow a ``@var`` of the same name, and ``Product`` reads its class attributes by name."""

_RESERVED_STATICS = {
    "backend": "build()'s own argument, the backend's name (build(model, backend='numba'))",
}
"""Names a build-time parameter may not take: ``build()`` takes them itself, so a parameter of
that name could never be set."""


def _collect_vars(model: type[Product]) -> dict[str, VarDef]:
    """Every ``@var`` visible on the class, with the D9 conflict rule."""
    names: list[str] = []
    for klass in reversed(model.__mro__):
        for name, value in vars(klass).items():
            if isinstance(value, VarDef) and name not in names:
                names.append(name)
    out: dict[str, VarDef] = {}
    for name in names:
        definers = [k for k in model.__mro__ if isinstance(vars(k).get(name), VarDef)]
        effective = definers[0]
        reserved = _RESERVED_NAMES.get(name)
        if reserved is not None:
            raise BuildError(
                f"{name} is {reserved} (self.{name}); rename the variable",
                vars(effective)[name].loc,
            )
        if effective is not model:
            for i, d1 in enumerate(definers):
                for d2 in definers[i + 1 :]:
                    if not issubclass(d1, d2) and not issubclass(d2, d1):
                        raise BuildError(
                            f"{model.__qualname__} inherits two definitions of {name}, from "
                            f"{d1.__qualname__} and {d2.__qualname__}; override {name} in "
                            f"{model.__qualname__} to resolve them (D9)",
                            vars(effective)[name].loc,
                        )
        out[name] = vars(effective)[name]
    return out


def _trace_var(vd: VarDef, inst: Product, scope: BoundaryScope) -> ir.Variable:
    fn = vd.fn
    loc = vd.loc
    declared = vd.declared_unit()
    ctx = TraceContext(loc.file, fn.__qualname__, loc.line)
    set_context(ctx)
    try:
        result = fn(inst, Sym(ir.TimeIndex(loc))) if vd.has_time else fn(inst)
        expr = as_node(result, loc)
        init = _boundary(vd.init, scope, loc, "init")
        terminal = _boundary(vd.terminal, scope, loc, "terminal")
    except HalleyError:
        raise
    except Exception as exc:
        raise BuildError(
            f"{vd.name} raised {type(exc).__name__}: {exc}", _loc_from_exception(exc, loc)
        ) from exc
    finally:
        set_context(None)
    try:
        source = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError):
        source = ""
    return ir.Variable(
        name=vd.name,
        declared_unit=declared,
        has_time=vd.has_time,
        timing=vd.timing,
        init=init,
        terminal=terminal,
        role=vd.role,
        domain=vd.domain,
        expr=expr,
        loc=loc,
        source=source,
        doc=inspect.getdoc(fn),
    )


def _boundary(value: object, scope: BoundaryScope, loc: Loc, what: str) -> ir.Node | None:
    """The ``init`` or ``terminal`` node: a number, or the callable applied to the boundary
    scope (model point fields first, the model's variables second)."""
    if value is None:
        return None
    if callable(value):
        fn: Callable[[Any], Any] = value
        return as_node(fn(scope), loc)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BuildError(
            f"{what} is a number or a function of the model point and the model's variables",
            loc,
        )
    return ir.Const(loc, value)


def _loc_from_exception(exc: BaseException, default: Loc) -> Loc:
    tb = exc.__traceback__
    found = default
    while tb is not None:
        code = tb.tb_frame.f_code
        if code.co_filename == default.file:
            found = Loc(default.file, tb.tb_lineno, code.co_qualname)
        tb = tb.tb_next
    return found
