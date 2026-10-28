# pyright: basic
"""Process parallelism (design section 12, brief M6 D2): ``RunSettings(workers=n)`` spreads a
run's batches over spawned worker processes (:mod:`halley.runtime.workers`).

At the same ``batch_size`` a run with two workers is the run with one, bit for bit, on both
backends: the same run id (``workers`` is recorded in the manifest, not in the identity), the
same output files and arrays and the same invariant results. The runs are TermAssurance on the
reference inputs, a CPI-linked annuity of five policies over two scenarios in three batches,
and an aggregation stage over two groups, a batch each. A RunError raised in a worker reaches
the caller with its message and its model location, the first failing batch's in batch order,
as with one process. A worker imports the model's class from its module, so the models these
tests run with workers are defined at the top level of this module or of ``models``.
"""

from __future__ import annotations

import datetime as dt
import multiprocessing
import os
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from halley import (
    BuildError,
    CompiledModel,
    Count,
    Int,
    Invariant,
    LoadError,
    Monthly,
    Prob,
    Product,
    RunError,
    RunResult,
    RunSettings,
    UnitError,
    build,
    run,
    var,
    where,
)
from halley.aggregation import group_series
from halley.errors import Loc
from halley.products import annuity_data as data
from halley.products.annuity import VALUATION_DATE as ANNUITY_DATE
from halley.products.annuity import CPILinkedAnnuity
from halley.products.term_assurance import VALUATION_DATE, TermAssurance
from halley.products.term_assurance_data import reference_model_points, reference_tables
from halley.runtime import workers
from models import (
    MEMBERS,
    SIMPLE_SCHEMA,
    VAL,
    Growth,
    Release,
    release_bases,
    simple_model_points,
    stage_groups,
)

BACKENDS = ("numpy", pytest.param("numba", marks=pytest.mark.numba))

ANNUITY = Path(__file__).resolve().parent.parent / "docs" / "spec" / "annuity"
ANNUITY_TABLES = {
    name: ANNUITY / f"{name}.parquet"
    for name in ("mort_base", "mort_cif", "expense", "inv_expense", "ilp", "ra_param")
}

_BUILT: dict[tuple[type[Product], str], CompiledModel] = {}


def built(model: type[Product], backend: str) -> CompiledModel:
    """One build per model and backend in the session."""
    if backend == "numba":
        pytest.importorskip("numba", reason="the Numba backend needs numba")
    key = (model, backend)
    if key not in _BUILT:
        _BUILT[key] = build(model, backend=backend)
    return _BUILT[key]


@pytest.fixture
def spread(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """The number of worker processes of every run this test spreads over a pool, as the
    calling process starts them."""
    pools: list[int] = []
    in_workers = workers._in_workers

    def spy(*args: Any) -> Any:
        pools.append(args[-1])
        return in_workers(*args)

    monkeypatch.setattr(workers, "_in_workers", spy)
    return pools


def assert_bit_identical(one: RunResult, two: RunResult) -> None:
    """``two`` (several workers) is ``one`` (one process) bit for bit: the run id, the output
    files' hashes, every array's dtype, shape and bytes, the invariant results; the manifests
    differ only in the workers recorded and the time the run was made."""
    assert two.run_id == one.run_id
    for key in ("monthly_sha256", "scalars_sha256"):
        assert two.manifest["outputs"][key] == one.manifest["outputs"][key], key
    assert two.manifest["invariants_sha256"] == one.manifest["invariants_sha256"]
    assert set(two.arrays) == set(one.arrays)
    for name, a in one.arrays.items():
        b = two.arrays[name]
        assert (b.dtype, b.shape) == (a.dtype, a.shape), name
        assert np.ascontiguousarray(b).tobytes() == np.ascontiguousarray(a).tobytes(), name
    assert [r.as_dict() for r in two.invariants] == [r.as_dict() for r in one.invariants]
    assert (one.manifest["settings"]["workers"], two.manifest["settings"]["workers"]) == (1, 2)

    def rest(m: dict[str, Any]) -> dict[str, Any]:
        settings = {k: v for k, v in m["settings"].items() if k != "workers"}
        return {**{k: v for k, v in m.items() if k != "created_at"}, "settings": settings}

    assert rest(two.manifest) == rest(one.manifest)


def one_and_two(go: Callable[[int], RunResult]) -> tuple[RunResult, RunResult]:
    """``go(workers)`` with one worker and with two, held to each other bit for bit."""
    one, two = go(1), go(2)
    assert_bit_identical(one, two)
    return one, two


# ------------------------------------------------------------------- the same bits (D19, D5)


@pytest.mark.parametrize("backend", BACKENDS)
def test_term_assurance_with_two_workers(backend: str, spread: list[int], tmp_path: Path) -> None:
    """The reference case's ten policies in batches of three: four batches over two workers."""

    def go(n: int) -> RunResult:
        return run(
            built(TermAssurance, backend),
            model_points=reference_model_points(),
            tables=reference_tables(),
            settings=RunSettings(valuation_date=VALUATION_DATE, batch_size=3, workers=n),
            run_dir=tmp_path / f"workers-{n}",
        )

    one, _two = one_and_two(go)
    assert spread == [2]
    assert one.n_policies == 10
    assert one.lane("reserve", "P05")[0] == pytest.approx(537.9608658387098, rel=1e-12)


def cpi_policies(n: int) -> Any:
    """``n`` CPI-linked lives built from the annuity spec's policy 3: born a year apart, of
    alternating sex, with other instalments and counts."""
    base = data.example_policy(3)
    rows = [
        {
            **base,
            "policy_id": 101 + i,
            "sex": "MF"[i % 2],
            "date_of_birth": dt.date(1950 + i, 1 + 2 * i, 10),
            "instalment_pm": 600.0 + 150.0 * i,
            "pols_if_init": 1.0 + 0.5 * i,
        }
        for i in range(n)
    ]
    return data.model_points_table(rows)


def two_cpi_paths(s: int, u: int) -> float:
    """Scenario 0 the spec's CPI path; scenario 1 the same history and 3.5% a year after."""
    if u <= 0 or s == 0:
        return data.cpi_path(u)
    return data.CPI_BASE * 1.035 ** (u / 12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_an_annuity_run_in_several_batches(backend: str, spread: list[int], tmp_path: Path) -> None:
    """Five CPI-linked lives over two scenarios (D4) in batches of two: three batches over two
    workers, every product invariant checked in the workers."""
    points = cpi_policies(5)
    scenarios = {
        "rf_df": data.discount_table(n_scenarios=2),
        "cpi": data.cpi_table(two_cpi_paths, n_scenarios=2),
    }

    def go(n: int) -> RunResult:
        return run(
            built(CPILinkedAnnuity, backend),
            model_points=points,
            tables=ANNUITY_TABLES,
            scenarios=scenarios,
            settings=RunSettings(valuation_date=ANNUITY_DATE, batch_size=2, workers=n),
            run_dir=tmp_path / f"workers-{n}",
        )

    one, _two = one_and_two(go)
    assert spread == [2]
    assert (one.n_policies, one.n_scenarios) == (5, 2)
    assert all(r.passed for r in one.invariants)
    assert any(r.name.startswith("I9") for r in one.invariants)


@pytest.mark.parametrize("backend", BACKENDS)
def test_an_aggregation_stage_batches_its_groups_over_workers(
    backend: str, spread: list[int], tmp_path: Path
) -> None:
    """``Release`` over two groups in batches of one: an aggregation stage's batches are its
    groups, and two groups are spread over two workers."""
    series = group_series(Release, release_bases(tmp_path / "bases"), MEMBERS)

    def go(n: int) -> RunResult:
        return run(
            built(Release, backend),
            model_points=stage_groups(),
            series=series,
            settings=RunSettings(valuation_date=VAL, batch_size=1, workers=n),
            run_dir=tmp_path / f"stage-{n}",
        )

    one, _two = one_and_two(go)
    assert spread == [2]
    assert one.kind == "aggregation"
    assert list(one.ids) == ["G0", "G1"]


def test_more_workers_than_batches(spread: list[int], tmp_path: Path) -> None:
    """Four workers asked for and three batches: three processes, the same bits; one batch runs
    in the calling process whatever the workers, recorded as asked."""
    settings = {"valuation_date": VAL, "batch_size": 1}
    compiled = built(Growth, "numpy")
    points = simple_model_points(3)
    runs = {
        n: run(
            compiled,
            model_points=points,
            settings=RunSettings(**settings, workers=n),
            run_dir=tmp_path / f"workers-{n}",
        )
        for n in (1, 4)
    }
    assert spread == [3]
    assert runs[4].run_id == runs[1].run_id
    assert (
        runs[4].manifest["outputs"]["monthly_sha256"]
        == runs[1].manifest["outputs"]["monthly_sha256"]
    )
    single = run(
        compiled,
        model_points=points,
        settings=RunSettings(valuation_date=VAL, workers=4),
        run_dir=tmp_path / "one-batch",
    )
    assert spread == [3]  # no pool for one batch
    assert single.manifest["settings"]["workers"] == 4


# ----------------------------------------------------------------- errors across processes


class OutOfRange(Product):
    """A probability that leaves [0, 1] at t = 3 for the policies with k = 5 and k = 7, S03 and
    S05 of ``simple_model_points(6)``: in batches of two the second and third batches fail."""

    model_point = SIMPLE_SCHEMA

    @var
    def q(self, t: Int) -> Prob[Monthly]:
        bad = (t == 3) & ((self.mp.k == 5) | (self.mp.k == 7))
        return where(bad, 1.5, 0.01)  # type: ignore[return-value] - literals take the unit

    @var
    def x(self, t: Int) -> Count:
        return self.mp.pols_if_init * self.q(t)


class Capped(Growth):
    """``Growth`` with an invariant that S03, S04 and S05 of ``simple_model_points(6)`` fail
    (their ``x(0)``, the initial count, is 4, 5 and 6)."""

    invariants = (Invariant("x(0) at most 3.5", lambda a: a.at("x", 0), 3.5, kind="le", tol=0),)


def _remote(err: BaseException) -> bool:
    """Whether ``err`` was raised in a worker: ``concurrent.futures`` chains the worker's
    formatted traceback to it."""
    return type(err.__cause__).__name__ == "_RemoteTraceback"


def errors_of(
    compiled: CompiledModel, tmp_path: Path, n_policies: int = 6
) -> tuple[RunError, RunError]:
    """The RunError of a run in batches of two with one worker and with two."""
    errors = []
    for n in (1, 2):
        with pytest.raises(RunError) as info:
            run(
                compiled,
                model_points=simple_model_points(n_policies),
                settings=RunSettings(valuation_date=VAL, batch_size=2, workers=n),
                run_dir=tmp_path / f"workers-{n}",
            )
        errors.append(info.value)
    return errors[0], errors[1]


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_run_error_in_a_worker_keeps_its_message_and_location(
    backend: str, spread: list[int], tmp_path: Path
) -> None:
    """The second and third batches fail the probability check: the caller raises the second
    batch's error, as one process does, with the message and the model line the worker gave
    it (the variable's, never the generated code's), and no run directory is written."""
    one, two = errors_of(built(OutOfRange, backend), tmp_path)
    assert spread == [2]
    assert _remote(two)
    assert not _remote(one)
    assert two.message == one.message
    assert two.message == (
        "q is a probability but takes values outside [0, 1]: 0.01 to 1.5; policies S03"
    )
    assert two.loc == one.loc
    assert two.loc is not None
    assert two.loc.function.endswith("OutOfRange.q")
    assert Path(two.loc.file).name == "test_workers.py"
    assert str(two) == str(one)
    assert not (tmp_path / "workers-2" / "manifest.json").exists()


def test_a_failing_invariant_in_a_worker(spread: list[int], tmp_path: Path) -> None:
    """A product invariant fails in the second and third batches: the second's failure, at the
    invariant's declaration."""
    one, two = errors_of(built(Capped, "numpy"), tmp_path)
    assert spread == [2]
    assert _remote(two)
    assert two.message == one.message
    assert two.message.startswith("invariant x(0) at most 3.5 failed: ")
    assert "S03" in two.message
    assert "S04" not in two.message
    assert two.loc == one.loc
    assert two.loc is not None
    assert two.loc == Capped.invariants[0].loc


@pytest.mark.parametrize("cls", [BuildError, UnitError, LoadError, RunError])
def test_an_error_pickles_with_its_message_and_location(cls: Any) -> None:
    """What a worker sends the caller: the class, the message, the location, its notes."""
    loc = Loc("model.py", 12, "Model.q")
    err = cls("q is a probability but takes values outside [0, 1]", loc)
    err.add_note("a note")
    back = pickle.loads(pickle.dumps(err))
    assert type(back) is cls
    assert (back.message, back.loc, str(back)) == (err.message, loc, str(err))
    assert back.__notes__ == ["a note"]
    plain = pickle.loads(pickle.dumps(cls("no location")))
    assert (plain.message, plain.loc, str(plain)) == ("no location", None, "no location")


def test_a_model_defined_in_a_function_runs_in_one_process(
    spread: list[int], tmp_path: Path
) -> None:
    """A worker imports the model's class from its module, which cannot find a class defined
    in a function: a RunError at the class before any process starts; one process runs it."""

    class Local(Growth):
        pass

    compiled = build(Local)
    with pytest.raises(RunError, match=r"imports .*Local from its module test_workers") as info:
        run(
            compiled,
            model_points=simple_model_points(4),
            settings=RunSettings(valuation_date=VAL, batch_size=2, workers=2),
            run_dir=tmp_path / "two",
        )
    assert "define the model at the top level of a module, or run with workers=1" in str(info.value)
    assert info.value.loc is not None
    assert info.value.loc.function.endswith("Local")
    run(
        compiled,
        model_points=simple_model_points(4),
        settings=RunSettings(valuation_date=VAL, batch_size=2, workers=1),
        run_dir=tmp_path / "one",
    )


def test_a_worker_that_cannot_start_fails_its_tasks() -> None:
    """A failure while a worker starts (unpickling what it was sent, importing the model,
    compiling) is kept and raised by its tasks, so the caller gets it as a batch's error."""
    workers._start_worker(b"not a pickle", "Model")
    try:
        with pytest.raises(RunError, match="a worker process could not start the run of Model"):
            workers._run_task(0, 1)
    finally:
        workers._WORKER.clear()


def _end_the_worker(a: Any) -> Any:
    """End the worker process it runs in, as the system ends one that runs out of memory."""
    if multiprocessing.parent_process() is None:
        raise AssertionError("this invariant runs in worker processes only")
    os._exit(3)


class Ended(Growth):
    """``Growth`` with an invariant that ends the worker that checks it."""

    invariants = (Invariant("ends the worker", _end_the_worker),)


def test_a_worker_that_ends_without_a_result(spread: list[int], tmp_path: Path) -> None:
    """A worker that ends without a result breaks the pool: a RunError saying so and what to
    try, not the pool's own error."""
    with pytest.raises(
        RunError, match=r"a worker process of the run of test_workers\.Ended stopped"
    ) as info:
        run(
            built(Ended, "numpy"),
            model_points=simple_model_points(4),
            settings=RunSettings(valuation_date=VAL, batch_size=2, workers=2),
            run_dir=tmp_path / "ended",
        )
    assert spread == [2]
    assert "lower RunSettings.batch_size or workers" in info.value.message


def test_the_inputs_a_worker_is_sent(term_run: RunResult) -> None:
    """``Inputs.for_workers()``: the same arrays, every file without its table and bytes (its
    hash, path and metadata kept), and after pickling a batch of the same values, dtypes,
    shapes and strides as the caller's."""
    inputs, graph = term_run.inputs, term_run.graph
    assert inputs is not None
    assert graph is not None
    sent = inputs.for_workers()
    files = [sent.model_points.file, *(t.file for t in sent.tables.values())]
    originals = [inputs.model_points.file, *(t.file for t in inputs.tables.values())]
    for f, o in zip(files, originals, strict=True):
        assert (f.raw, f.table.num_rows) == (None, 0)
        assert (f.sha256, f.path, f.metadata, f.label) == (o.sha256, o.path, o.metadata, o.label)
        assert o.raw is not None
    assert sent.model_points.columns is inputs.model_points.columns
    payload = pickle.dumps(sent, protocol=pickle.HIGHEST_PROTOCOL)
    assert len(payload) < len(pickle.dumps(inputs, protocol=pickle.HIGHEST_PROTOCOL))
    back = pickle.loads(payload)
    want, got = inputs.batch(graph, 3, 7), back.batch(graph, 3, 7)
    assert set(got) == set(want)
    for key, a in want.items():
        b = got[key]
        if isinstance(a, np.ndarray):
            assert (b.dtype, b.shape, b.strides) == (a.dtype, a.shape, a.strides), key
            if a.dtype.kind == "O":  # the policy ids: compared as objects, not as pointers
                assert b.tolist() == a.tolist(), key
            else:
                assert np.ascontiguousarray(b).tobytes() == np.ascontiguousarray(a).tobytes()
        else:
            assert b == a, key


def test_batch_ranges() -> None:
    assert workers.batch_ranges(10, 3) == [(0, 3), (3, 6), (6, 9), (9, 10)]
    assert workers.batch_ranges(4, 10_000) == [(0, 4)]
    assert workers.batch_ranges(2, 0) == [(0, 1), (1, 2)]
