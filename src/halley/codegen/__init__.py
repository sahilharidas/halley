"""Code generation: the NumPy reference backend, then Numba.

:func:`compile_model` takes a checked graph to a :class:`CompiledModel`: the schedule, the
generated source and the callable that runs one batch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from halley import ir
from halley.codegen.numpy_backend import BatchFn, compile_source, generate
from halley.errors import BuildError
from halley.scheduler import Schedule, schedule

if TYPE_CHECKING:
    from halley.authoring.contracts import Schema
    from halley.authoring.product import Product

BACKEND = "numpy"


@dataclass(frozen=True, slots=True)
class CompiledModel:
    graph: ir.Graph
    schedule: Schedule
    backend: str
    source: str
    fn: BatchFn
    ir_hash: str
    schema: Schema
    model: type[Product]

    @property
    def name(self) -> str:
        return self.graph.model


def compile_model(graph: ir.Graph, model: type[Product]) -> CompiledModel:
    """Schedule a checked graph and generate the NumPy backend for it."""
    if model.model_point is None:
        raise BuildError(f"{model.__qualname__} declares no model_point schema")
    sched = schedule(graph)
    source = generate(graph, sched)
    fn = compile_source(source, graph.model)
    return CompiledModel(
        graph, sched, BACKEND, source, fn, ir.graph_hash(graph), model.model_point, model
    )
