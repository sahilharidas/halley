"""Code generation: the NumPy reference backend and the Numba backend.

:func:`compile_model` takes a checked graph to a :class:`CompiledModel`: the schedule, the
generated source and the callable that runs one batch. Every backend is generated from the same
IR and schedule (vault Spec, "Backends") and implements :class:`Backend`; :data:`BACKENDS` is
the registry ``build(model, backend=...)`` chooses from. The NumPy backend is the executable
spec (D16): every other backend is tested against it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Protocol

from numpy.typing import NDArray

from halley import ir
from halley.codegen.numba_backend import NumbaBackend
from halley.codegen.numpy_backend import BatchFn, NumpyBackend
from halley.errors import BuildError
from halley.scheduler import Schedule, schedule

if TYPE_CHECKING:
    from halley.authoring.contracts import Schema
    from halley.authoring.product import Product

DEFAULT_BACKEND: Final = "numpy"


class Backend(Protocol):
    """A code generator and the runtime of what it generates.

    ``name`` is what ``build(model, backend=...)`` takes and the manifest records;
    ``artefact`` the generated file's name under the run directory's ``generated/``;
    ``generate`` the source for a checked graph and its schedule; ``compile`` that source as the
    function that runs one batch (``run_batch(inp, H)``, the batch ABI of
    :mod:`halley.codegen.numpy_backend`); ``run_batch`` calls it and turns a failure into a
    :class:`~halley.errors.RunError` at the model's source line; ``versions`` the versions of
    what computes the values, for the manifest's ``backend`` block."""

    @property
    def name(self) -> str: ...

    @property
    def artefact(self) -> str: ...

    def generate(self, graph: ir.Graph, schedule: Schedule) -> str: ...

    def compile(self, source: str, model: str) -> BatchFn: ...

    def run_batch(
        self,
        fn: BatchFn,
        graph: ir.Graph,
        inp: dict[str, Any],
        horizon: int,
        lanes: str = "policies",
    ) -> dict[str, NDArray[Any]]: ...

    def versions(self) -> dict[str, str]: ...


BACKENDS: Final[Mapping[str, Backend]] = {
    "numpy": NumpyBackend(),
    "numba": NumbaBackend(),
}
"""Every backend by name."""


def backend_named(name: str) -> Backend:
    """The backend registered under ``name``; a BuildError listing the registry otherwise."""
    try:
        return BACKENDS[name]
    except (KeyError, TypeError):  # TypeError: a name that is not even hashable
        known = ", ".join(sorted(BACKENDS))
        raise BuildError(f"there is no backend {name!r}; the backends are: {known}") from None


@dataclass(frozen=True, slots=True)
class CompiledModel:
    graph: ir.Graph
    schedule: Schedule
    backend: str
    """The name of the backend that generated ``source`` (:data:`BACKENDS`)."""
    source: str
    fn: BatchFn
    ir_hash: str
    schema: Schema
    model: type[Product]

    @property
    def name(self) -> str:
        return self.graph.model


def compile_model(
    graph: ir.Graph, model: type[Product], backend: str = DEFAULT_BACKEND
) -> CompiledModel:
    """Schedule a checked graph and generate and compile the named backend for it."""
    chosen = backend_named(backend)
    if model.model_point is None:
        raise BuildError(f"{model.__qualname__} declares no model_point schema")
    sched = schedule(graph)
    source = chosen.generate(graph, sched)
    fn = chosen.compile(source, graph.model)
    return CompiledModel(
        graph, sched, chosen.name, source, fn, ir.graph_hash(graph), model.model_point, model
    )
