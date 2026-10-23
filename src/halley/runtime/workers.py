"""A run's batches, in the calling process or spread over worker processes (design section 12).

A run splits its model points into batches of ``RunSettings.batch_size`` (:func:`batch_ranges`).
Each batch is projected, checked against the built-in invariants and the product's, and gives
the arrays of the variables the run keeps and its invariant results (:class:`BatchRunner`).
With ``RunSettings.workers`` at 1, or a single batch, every batch runs in the calling process in
ascending order. With more, the batches are spread over a ``ProcessPoolExecutor`` of
``min(workers, batches)`` processes started with the spawn method, the only one Windows has:
a worker shares nothing with the caller but what it is sent. Each worker is sent, once, the
generated source, the backend's name, the graph, the model's class (by reference: the worker
imports it from its module, so it must be defined at the top level of an importable module)
and the inputs without their files (:meth:`~halley.runtime.loader.Inputs.for_workers`), and
compiles the batch function itself; a task is a batch's ``(lo, hi)``.

The caller takes the results in ascending batch order whatever order they finish in: the arrays
are assembled and the invariants merged as one process would, and the error of the first
failing batch in that order is raised, a :class:`~halley.errors.RunError` with the message and
the model source location the worker gave it. A batch computes the same bits in any process of
the machine (its inputs reach a worker with their dtypes, shapes and C-contiguous layout, so
NumPy takes the same loops), so a run's outputs, invariant results and errors do not depend on
the number of workers (D5, D19), which is why ``workers`` is recorded in the manifest and not in
the run id.
"""

from __future__ import annotations

import inspect
import multiprocessing
import pickle
import sys
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from halley import ir
from halley.codegen import Backend, backend_named
from halley.codegen.numpy_backend import BatchFn
from halley.errors import HalleyError, Loc, RunError
from halley.ir import units
from halley.runtime.invariants import BatchView, InvariantResult, check_builtin, check_declared
from halley.runtime.loader import Inputs

if TYPE_CHECKING:
    from halley.authoring.contracts import Invariant
    from halley.authoring.product import Product
    from halley.codegen import CompiledModel

Arrays = dict[str, NDArray[Any]]
BatchResult = tuple[Arrays, list[InvariantResult]]
"""What a batch gives: the arrays of the variables the run keeps, by name, and the results of
the built-in invariants then the product's."""


WINDOWS_MAX_PROCESSES = 61
"""The most worker processes a run starts on Windows, where a process pool waits on at most 61
(``concurrent.futures``' own limit); elsewhere there is none."""


def batch_ranges(n: int, batch_size: int) -> list[tuple[int, int]]:
    """The batches of ``n`` model points as ``(lo, hi)``, ``batch_size`` each (at least one),
    the last one shorter."""
    size = max(1, batch_size)
    return [(lo, min(lo + size, n)) for lo in range(0, n, size)]


@dataclass(slots=True)
class BatchRunner:
    """Runs one batch of a run (:meth:`__call__`), in the calling process or in a worker."""

    graph: ir.Graph
    invariants: Sequence[Invariant]
    """The product's invariants (``model.invariants``), checked after the built-in ones."""
    backend: Backend
    fn: BatchFn
    inputs: Inputs
    needed: tuple[str, ...]
    """The variables the run keeps, which a batch returns."""
    lanes: str
    """What the model points are called in messages: ``"policies"``, or ``"groups"``."""
    scales: dict[units.Unit, NDArray[np.float64]] = field(init=False)

    def __post_init__(self) -> None:
        self.scales = self.inputs.scales(self.graph)

    def __call__(self, lo: int, hi: int) -> BatchResult:
        """Project the model points ``lo:hi`` and check the invariants on them; a failure is a
        RunError at the model source."""
        graph, inputs = self.graph, self.inputs
        inp = inputs.batch(graph, lo, hi)
        out = self.backend.run_batch(self.fn, graph, inp, inputs.H, self.lanes)
        scales = {u: s[lo:hi] for u, s in self.scales.items()}
        results = check_builtin(graph, out, inputs.ids[lo:hi], inputs.H, scales, self.lanes)
        view = BatchView(graph, out, inputs, lo, hi, scales, self.lanes)
        results.extend(check_declared(self.invariants, view))
        return {name: out[name] for name in self.needed}, results


def run_batches(
    compiled: CompiledModel,
    inputs: Inputs,
    needed: tuple[str, ...],
    lanes: str,
    batch_size: int,
    workers: int,
) -> list[BatchResult]:
    """Every batch's result, in ascending batch order: in this process when ``workers`` is 1
    or there is a single batch, else over ``min(workers, batches)`` worker processes
    (:func:`_in_workers`). The first failing batch's error is raised, in batch order."""
    ranges = batch_ranges(len(inputs.ids), batch_size)
    processes = min(workers, len(ranges))
    if sys.platform == "win32":
        processes = min(processes, WINDOWS_MAX_PROCESSES)
    if processes > 1:
        return _in_workers(compiled, inputs, needed, lanes, ranges, processes)
    backend = backend_named(compiled.backend)
    runner = BatchRunner(
        compiled.graph, compiled.model.invariants, backend, compiled.fn, inputs, needed, lanes
    )
    return [runner(lo, hi) for lo, hi in ranges]


# ------------------------------------------------------------------------- the worker side


@dataclass(frozen=True, slots=True)
class _Setup:
    """What each worker is sent once, at its start (pickled by the caller)."""

    source: str
    backend: str
    graph: ir.Graph
    model: type[Product]
    """Pickled by reference: the worker imports it from its module."""
    inputs: Inputs
    needed: tuple[str, ...]
    lanes: str
    errstate: dict[str, Any]
    """The caller's NumPy floating-point error handling (``np.geterr()``), which the checks
    after a batch run under: a spawned process starts with NumPy's default."""


_WORKER: list[BatchRunner | BaseException] = []
"""This worker's runner, or the error that stopped it from starting (one entry once started)."""


def _start_worker(payload: bytes, model: str) -> None:
    """A worker's initializer: unpickle what it was sent, import the model's class, compile
    the batch function. A failure is kept and raised by every task, so that the caller gets it
    as a batch's error rather than as a broken pool."""
    try:
        setup: _Setup = pickle.loads(payload)
        np.seterr(**setup.errstate)
        backend = backend_named(setup.backend)
        fn = backend.compile(setup.source, setup.graph.model)
        runner: BatchRunner | BaseException = BatchRunner(
            setup.graph,
            setup.model.invariants,
            backend,
            fn,
            setup.inputs,
            setup.needed,
            setup.lanes,
        )
    except HalleyError as exc:
        runner = exc
    except Exception as exc:  # the model's module failing to import in the worker, say
        runner = RunError(
            f"a worker process could not start the run of {model}: {type(exc).__name__}: "
            f"{exc}; with RunSettings.workers above 1 each worker imports the model's class "
            "from its module: define it at the top level of a module the worker can import, "
            "or run with workers=1"
        )
    _WORKER[:] = [runner]


def _run_task(lo: int, hi: int) -> BatchResult:
    """A worker's task: the batch ``lo:hi``."""
    runner = _WORKER[0]
    if isinstance(runner, BaseException):
        raise runner
    return runner(lo, hi)


# ------------------------------------------------------------------------- the caller side


def _class_loc(cls: type) -> Loc | None:
    """Where a class is defined, for an error about the class."""
    try:
        file = inspect.getsourcefile(cls)
        _lines, line = inspect.getsourcelines(cls)
    except (OSError, TypeError):
        return None
    return None if file is None else Loc(file, line, cls.__qualname__)


def _require_importable(model: type[Product]) -> None:
    """A worker imports the model's class from its module, so the class must be found there
    under its name: a class defined inside a function, or in an interactive session, is not."""
    reason: str | None = None
    try:
        pickle.dumps(model)
    except (pickle.PicklingError, AttributeError, TypeError) as exc:
        reason = str(exc)
    main = sys.modules.get("__main__")
    if reason is None and model.__module__ == "__main__" and not hasattr(main, "__file__"):
        reason = "it is defined in an interactive session, which a worker process cannot import"
    if reason is None:
        return
    raise RunError(
        "with RunSettings.workers above 1 the batches run in worker processes, each of which "
        f"imports {model.__qualname__} from its module {model.__module__}, and it cannot "
        f"({reason}): define the model at the top level of a module, or run with workers=1",
        _class_loc(model),
    )


def _in_workers(
    compiled: CompiledModel,
    inputs: Inputs,
    needed: tuple[str, ...],
    lanes: str,
    ranges: list[tuple[int, int]],
    processes: int,
) -> list[BatchResult]:
    """The batches over a pool of ``processes`` spawned workers, taken in ascending batch
    order; the first failing batch's error is raised once the batches before it have passed,
    the batches not yet started are cancelled, and the pool is shut down either way."""
    model = compiled.model
    _require_importable(model)
    setup = _Setup(
        compiled.source,
        compiled.backend,
        compiled.graph,
        model,
        inputs.for_workers(),
        needed,
        lanes,
        dict(np.geterr()),
    )
    try:
        payload = pickle.dumps(setup, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:  # a build-time setting that does not pickle, say
        raise RunError(
            f"the run of {compiled.name} cannot be sent to worker processes: "
            f"{type(exc).__name__}: {exc}; run with workers=1"
        ) from exc
    pool = ProcessPoolExecutor(
        processes,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_start_worker,
        initargs=(payload, compiled.name),
    )
    try:
        futures = [pool.submit(_run_task, lo, hi) for lo, hi in ranges]
        return [future.result() for future in futures]
    except BrokenProcessPool as exc:
        raise RunError(
            f"a worker process of the run of {compiled.name} stopped without a result "
            f"({exc}): it may have run out of memory (lower RunSettings.batch_size or workers), "
            "or the script that started the run does not guard its entry point with "
            "'if __name__ == \"__main__\":', which the spawn start method needs"
        ) from exc
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
