"""Halley: an actuarial projection engine.

Models are written as Python classes, checked before they run (units, time references,
calculation order, data contracts) and compiled when they do.

Typical use::

    from halley import build, run, RunSettings
    compiled = build(TermAssurance)
    result = run(compiled, model_points=..., tables={...}, settings=RunSettings(...), run_dir=...)
"""

from __future__ import annotations

from halley.authoring import (
    Annual,
    Bool,
    Column,
    Count,
    Derived,
    Factor,
    Field,
    Fraction,
    Int,
    Key,
    Money,
    MoneyPerPolicy,
    Monthly,
    PerPolicy,
    Prob,
    Product,
    Rate,
    Rule,
    ScenarioInput,
    Schema,
    Static,
    Table,
    at,
    sum_t,
    to_monthly,
    to_monthly_prob,
    var,
    where,
)
from halley.authoring.product import Product as _Product
from halley.checker import check
from halley.codegen import CompiledModel, compile_model
from halley.errors import BuildError, BuildWarning, HalleyError, LoadError, RunError, UnitError
from halley.runtime import RunResult, RunSettings, diff, explain, load_run, run
from halley.tracing.tracer import trace

__version__ = "0.1.0"


def build(model: type[_Product], **statics: object) -> CompiledModel:
    """Trace, check, schedule and compile a model with the given build-time settings."""
    graph = trace(model, statics)
    check(graph)
    return compile_model(graph, model)


__all__ = [
    "Annual",
    "Bool",
    "BuildError",
    "BuildWarning",
    "Column",
    "CompiledModel",
    "Count",
    "Derived",
    "Factor",
    "Field",
    "Fraction",
    "HalleyError",
    "Int",
    "Key",
    "LoadError",
    "Money",
    "MoneyPerPolicy",
    "Monthly",
    "PerPolicy",
    "Prob",
    "Product",
    "Rate",
    "Rule",
    "RunError",
    "RunResult",
    "RunSettings",
    "ScenarioInput",
    "Schema",
    "Static",
    "Table",
    "UnitError",
    "at",
    "build",
    "check",
    "diff",
    "explain",
    "load_run",
    "run",
    "sum_t",
    "to_monthly",
    "to_monthly_prob",
    "trace",
    "var",
    "where",
]
