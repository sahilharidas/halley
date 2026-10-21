# pyright: basic
"""The Numba backend against the NumPy reference (D16; design section 12, brief M6 D1).

Every differential runs one model on both backends from the same inputs and holds the Numba
run to the NumPy one: the same policies, months and scenarios, every stored array within 1e-9
relative under the products' tolerance scales (D19's rule across backends), exact zeros in the
same places, Int and Bool arrays equal, and the same invariant results. The models are the term
assurance reference case, lifelib's BasicTerm, the annuity examples and a two-scenario annuity
run, the in-force chain's eight projection runs and both aggregation stages for each variant,
the engine's small test models and one that calls every formula function. Then the Numba
backend's own promises: reruns bit-identical (D19), no FMA contraction, every run-time check
with the NumPy backend's text, model line and lanes, the run directory and manifest, the module
file and Numba's cache beside it, and a BuildError for a node it cannot emit.

The compiled models are session fixtures. A first build compiles every kernel, which takes a
while; Numba's ``cache=True`` keeps them under ``halley.codegen.numba_backend.cache_dir()`` for
later sessions. The module is marked ``numba``.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest

pytest.importorskip("numba", reason="the Numba backend needs numba: pip install halley[numba]")

from halley import (
    Annual,
    Bool,
    BuildError,
    Column,
    CompiledModel,
    Count,
    CoverageUnits,
    Factor,
    Field,
    Int,
    Key,
    Money,
    Monthly,
    Prob,
    Product,
    Rate,
    RunError,
    RunResult,
    RunSettings,
    Schema,
    Table,
    ann_fwd,
    ann_spot,
    annual_step,
    as_cu,
    at,
    build,
    cc_discount,
    fwd_rate,
    ir,
    load_run,
    run,
    to_monthly,
    to_monthly_cf,
    to_monthly_prob,
    var,
    where,
)
from halley.aggregation import (
    InForceGroup,
    NewGroup,
    group_series,
    replay_aggregation,
)
from halley.authoring import exp, expm1, int_field, log, log1p, maximum, minimum, sqrt
from halley.codegen import BACKENDS, numba_backend
from halley.products import annuity_data as data
from halley.products.annuity import VALUATION_DATE as ANNUITY_DATE
from halley.products.annuity import (
    CPILinkedAnnuity,
    FixedEscalationAnnuity,
    LevelAnnuity,
)
from halley.products.lifelib_basic_term import VALUATION_DATE as LIFELIB_DATE
from halley.products.lifelib_basic_term import BasicTerm
from halley.products.term_assurance import VALUATION_DATE, TermAssurance
from halley.runtime import within
from models import (
    MEMBERS,
    SIMPLE_SCHEMA,
    VAL,
    Expensed,
    Growth,
    Indexed,
    Margin,
    Powered,
    Release,
    margin_table,
    powered_model_points,
    release_bases,
    scenario_file,
    simple_model_points,
    stage_groups,
)
from test_annuity import SCENARIOS, SPEC, T_EXAMPLE, TABLES, TRACE_OUTPUTS
from test_in_force_group import Chain
from test_lifelib_basic_term import TABLES as LIFELIB_TABLES
from test_lifelib_basic_term import model_points_with_premium_rate
from test_series import AtAll, Keyed, _keyed_groups

pytestmark = pytest.mark.numba

REL = 1e-9
"""The tolerance between backends (D19)."""

BACKEND_NAMES = ("numpy", "numba")

_BUILT: dict[tuple[type[Product], str, tuple[tuple[str, object], ...]], CompiledModel] = {}


def built(model: type[Product], backend: str, **statics: object) -> CompiledModel:
    """One build per model, backend and build-time settings in the session."""
    key = (model, backend, tuple(sorted(statics.items())))
    if key not in _BUILT:
        _BUILT[key] = build(model, backend=backend, **statics)
    return _BUILT[key]


def _scale(name: str, res: RunResult, shape: tuple[int, ...]) -> Any:
    """The tolerance scale of a stored variable: the product's per-policy scale of its unit
    (a total's is the sum over policies), else 1 (term spec section 1)."""
    graph = res.graph
    assert graph is not None
    assert res.inputs is not None
    v = graph.variables[name]
    scales = res.inputs.scales(graph)
    s = scales.get(v.declared_unit)
    if s is None:
        return 1.0
    if ir.POLICY in (v.dims or frozenset()) and shape[1] == len(s):
        return np.asarray(s, dtype=np.float64).reshape(1, -1, 1)
    return float(np.sum(s))


def assert_same_run(want: RunResult, got: RunResult) -> None:
    """``got`` (Numba) holds ``want`` (NumPy): ids, months, scenarios, every stored array within
    1e-9 relative under the product's scales with exact zeros in the same places, Int and Bool
    arrays equal, and the same invariant results."""
    assert got.manifest["backend"]["name"] == "numba"
    assert want.manifest["backend"]["name"] == "numpy"
    assert list(got.ids) == list(want.ids)
    assert (got.H, got.n_scenarios) == (want.H, want.n_scenarios)
    assert set(got.arrays) == set(want.arrays)
    for name in sorted(want.arrays):
        x, y = want.arrays[name], got.arrays[name]
        assert (y.shape, y.dtype) == (x.shape, x.dtype), name
        if x.dtype.kind in "biu":
            assert np.array_equal(x, y), name
            continue
        zeros = (x == 0) != (y == 0)
        assert not zeros.any(), f"{name}: an exact zero differs at {np.argwhere(zeros)[0]}"
        ok = within(y, x, _scale(name, want, x.shape), rel=REL)
        if not ok.all():
            i = tuple(int(j) for j in np.argwhere(~ok)[0])
            raise AssertionError(f"{name}{list(i)}: numba {y[i]!r} against numpy {x[i]!r}")
    assert [(r.name, r.passed) for r in got.invariants] == [
        (r.name, r.passed) for r in want.invariants
    ]


def on_both(go: Callable[[str], RunResult]) -> RunResult:
    """Run ``go(backend)`` on both backends, hold the Numba run to the NumPy one and return
    the Numba run."""
    want, got = go("numpy"), go("numba")
    assert_same_run(want, got)
    return got


def same_error(go: Callable[[str], Any], pattern: str) -> RunError:
    """``go(backend)`` raises a RunError on both backends with the same text and location;
    the NumPy one's text matches ``pattern``."""
    errors: dict[str, RunError] = {}
    for backend in BACKEND_NAMES:
        with pytest.raises(RunError) as info:
            go(backend)
        errors[backend] = info.value
    want, got = errors["numpy"], errors["numba"]
    assert re.search(pattern, want.message), want.message
    assert got.message == want.message
    assert got.loc == want.loc
    return got


# ------------------------------------------------------------------------- compiled models


@pytest.fixture(scope="session")
def numba_annuity() -> dict[str, CompiledModel]:
    classes = {"level": LevelAnnuity, "fixed": FixedEscalationAnnuity, "cpi": CPILinkedAnnuity}
    return {name: built(cls, "numba") for name, cls in classes.items()}


@pytest.fixture(scope="session")
def numba_new_group() -> CompiledModel:
    return built(NewGroup, "numba")


@pytest.fixture(scope="session")
def numba_in_force_group() -> CompiledModel:
    return built(InForceGroup, "numba")


# --------------------------------------------------------------------------- the products


def test_term_assurance_on_the_reference_inputs(
    term_run: RunResult, reference_sources: dict[str, Any], tmp_path: Path
) -> None:
    inputs = Path(__file__).resolve().parent.parent / "docs" / "spec" / "term-assurance"
    got = run(
        built(TermAssurance, "numba"),
        model_points=inputs / "model_points.parquet",
        tables={name: inputs / f"{name}.parquet" for name in reference_sources["tables"]},
        settings=RunSettings(valuation_date=VALUATION_DATE, label="reference case 1"),
        run_dir=tmp_path / "term-numba",
    )
    assert_same_run(term_run, got)
    assert got.lane("reserve", "P05")[0] == pytest.approx(537.9608658387098, rel=1e-12)


def test_basic_term_on_the_lifelib_inputs(tmp_path: Path) -> None:
    points = model_points_with_premium_rate()

    def go(backend: str) -> RunResult:
        return run(
            built(BasicTerm, backend),
            model_points=points,
            tables=LIFELIB_TABLES,
            settings=RunSettings(valuation_date=LIFELIB_DATE, batch_size=2000),
            run_dir=tmp_path / backend,
        )

    got = on_both(go)
    assert got.n_policies == 10_000


@pytest.mark.parametrize(
    ("cls", "points", "horizon", "outputs"),
    [
        ("level", "mp_level", None, None),
        ("fixed", "mp_fixed", T_EXAMPLE, TRACE_OUTPUTS),
        ("cpi", "mp_cpi", T_EXAMPLE, TRACE_OUTPUTS),
        ("cpi", "mp_cpi", None, TRACE_OUTPUTS),
    ],
    ids=["level", "fixed", "cpi", "cpi-alone"],
)
def test_the_annuity_class_runs(
    cls: str,
    points: str,
    horizon: int | None,
    outputs: tuple[str, ...] | None,
    compiled_annuity: dict[str, CompiledModel],
    numba_annuity: dict[str, CompiledModel],
    tmp_path: Path,
) -> None:
    """Example 3's three class runs at T = 655 with the tracing outputs (which include the Int
    and Bool columns ``age``, ``cal_year`` and ``anniv``), and the horizon test's policy 3."""
    compiled = {"numpy": compiled_annuity, "numba": numba_annuity}

    def go(backend: str) -> RunResult:
        return run(
            compiled[backend][cls],
            model_points=SPEC / f"{points}.parquet",
            tables=TABLES,
            scenarios=SCENARIOS,
            settings=RunSettings(valuation_date=ANNUITY_DATE, horizon=horizon, outputs=outputs),
            run_dir=tmp_path / backend,
        )

    got = on_both(go)
    if outputs is not None:
        assert got.arrays["age"].dtype == np.int64
        assert got.arrays["anniv"].dtype == np.bool_


@pytest.mark.parametrize("pid", [1, 4])
def test_the_annuity_examples_1_and_2(
    pid: int,
    compiled_annuity: dict[str, CompiledModel],
    numba_annuity: dict[str, CompiledModel],
    tmp_path: Path,
) -> None:
    compiled = {"numpy": compiled_annuity, "numba": numba_annuity}
    table = data.model_points_table([data.example_policy(pid)])

    def go(backend: str) -> RunResult:
        return run(
            compiled[backend]["level"],
            model_points=table,
            tables=TABLES,
            scenarios=SCENARIOS,
            settings=RunSettings(valuation_date=ANNUITY_DATE),
            run_dir=tmp_path / backend,
        )

    on_both(go)


def test_a_two_scenario_annuity_run(
    compiled_annuity: dict[str, CompiledModel],
    numba_annuity: dict[str, CompiledModel],
    tmp_path: Path,
) -> None:
    """The CPI-linked class with two CPI paths (identical history, 2.5% and 3.5% a year): the
    scenario axis is the kernels' inner loop."""

    def cpi(s: int, u: int) -> float:
        if u <= 0 or s == 0:
            return data.cpi_path(u)
        return data.CPI_BASE * 1.035 ** (u / 12)

    compiled = {"numpy": compiled_annuity, "numba": numba_annuity}

    def go(backend: str) -> RunResult:
        return run(
            compiled[backend]["cpi"],
            model_points=SPEC / "mp_cpi.parquet",
            tables=TABLES,
            scenarios={
                "rf_df": data.discount_table(n_scenarios=2),
                "cpi": data.cpi_table(cpi, n_scenarios=2),
            },
            settings=RunSettings(valuation_date=ANNUITY_DATE),
            run_dir=tmp_path / backend,
        )

    got = on_both(go)
    assert got.n_scenarios == 2
    bel = got.var("bel")
    assert bel[0, 0, 1] > bel[0, 0, 0]


@pytest.mark.parametrize("k", ["P", "O"])
def test_the_in_force_chain(
    k: str,
    compiled_annuity: dict[str, CompiledModel],
    compiled_new_group: dict[bool, CompiledModel],
    compiled_in_force_group: CompiledModel,
    numba_annuity: dict[str, CompiledModel],
    numba_new_group: CompiledModel,
    numba_in_force_group: CompiledModel,
    tmp_path: Path,
) -> None:
    """One variant's measurement on each backend: the eight projection runs (REC, OLD, NEW and
    CUR of the level and CPI-linked classes), the recognition stage ``NewGroup`` and the
    in-force stage ``InForceGroup``, each held to the NumPy one. Every run of a chain is on one
    backend, as the stage's one-build check requires."""
    want = Chain(
        k,
        compiled_annuity,
        compiled_new_group[True],
        compiled_in_force_group,
        tmp_path / "numpy" / "runs",
    )
    got = Chain(
        k, numba_annuity, numba_new_group, numba_in_force_group, tmp_path / "numba" / "runs"
    )
    for name in ("REC", "OLD", "NEW", "CUR"):
        for c in ("level", "cpi"):
            assert_same_run(want.runs(name)[c], got.runs(name)[c])
    assert_same_run(want.recognised, got.recognised)
    assert_same_run(want.stage, got.stage)
    assert got.stage.manifest["backend"]["name"] == "numba"
    # D21 on the Numba backend: the stage replayed from its own directory is the same run
    replay = replay_aggregation(numba_in_force_group, got.stage.run_dir, run_dir=tmp_path / "re")
    assert replay.run_id == got.stage.run_id
    assert replay.manifest["replayed_from"]["matches"] is True


def test_an_aggregation_stage_over_group_series(tmp_path: Path) -> None:
    """``Release``: series read at t, at a fixed month and without a time dimension, a
    backward recursion feeding forward ones, groups as the policies of a batch; on a
    deterministic basis and on one with two scenarios."""
    for shocks in ((1.0,), (1.0, 1.5)):
        bases = release_bases(tmp_path / f"bases-{len(shocks)}", shocks=shocks)
        series = group_series(Release, bases, MEMBERS)

        def go(backend: str, series: Any = series, n: int = len(shocks)) -> RunResult:
            return run(
                built(Release, backend),
                model_points=stage_groups(),
                series=series,
                settings=RunSettings(valuation_date=VAL),
                run_dir=tmp_path / f"stage-{n}-{backend}",
            )

        on_both(go)


def test_a_series_read_at_each_groups_month(tmp_path: Path) -> None:
    """``Keyed``: ``series.x.at(group.k)`` in a scalar formula, in whole mode and in a forward
    recursion."""
    series = group_series(Keyed, release_bases(tmp_path / "bases"), MEMBERS)

    def go(backend: str) -> RunResult:
        return run(
            built(Keyed, backend),
            model_points=_keyed_groups((5, 3)),
            series=series,
            settings=RunSettings(valuation_date=VAL),
            run_dir=tmp_path / backend,
        )

    on_both(go)


# ---------------------------------------------------------------------- the small models


def test_the_engine_test_models(tmp_path: Path) -> None:
    """A forward and a backward recursion, a masked division whose discarded branch divides by
    zero (``Growth``: ``where(t > 2, 1.0 / (t - 2), 0.0)`` and ``where(x > 0, r / x, 0)``),
    ``sum_t`` and ``at``; a backward component of two members; boundary values from other
    variables; scenario inputs at t, t - 1, t - 3 and at a fixed month over two scenarios, with
    variables that have no policy dimension."""
    horizon = 12
    df = scenario_file("df", 0, horizon, 2, lambda s, u: (0.99 if s == 0 else 0.98) ** u)
    cpi = scenario_file("cpi", -3, horizon, 2, lambda s, u: 100.0 * (1.002 + 0.001 * s) ** u)
    cases: dict[str, tuple[type[Product], dict[str, Any]]] = {
        "growth": (Growth, {}),
        "expensed": (Expensed, {}),
        "margin": (Margin, {"tables": {"margin": margin_table(0.5)}}),
        "indexed": (Indexed, {"scenarios": {"df": df, "cpi": cpi}}),
    }
    for case, (model, inputs) in cases.items():

        def go(
            backend: str, model: type[Product] = model, inputs: Any = inputs, case: str = case
        ) -> RunResult:
            return run(
                built(model, backend),
                model_points=simple_model_points(rem_m=horizon),
                settings=RunSettings(valuation_date=VAL),
                run_dir=tmp_path / f"{case}-{backend}",
                **inputs,
            )

        got = on_both(go)
        if case == "growth":
            assert list(got.var("masked")[:5, 0, 0]) == [0.0, 0.0, 0.0, 1.0, 0.5]
            assert np.isfinite(got.arrays["ratio"]).all()


def test_a_non_literal_exponent(tmp_path: Path) -> None:
    """``x(t) ** self.mp.k`` for k = -1, 2 and 3, outside a recursive component and inside
    one: LLVM's pow against NumPy's (fast path or not), within 1e-9. A kernel raises each lane
    with the same scalar code whatever the batch, so on the Numba backend a batch of one policy
    and a batch of three agree bit for bit, where NumPy's fast path depends on the batch's
    shapes (D5; ``value_kind``)."""
    points = powered_model_points([60_000, 70_000, 80_000], [-1, 2, 3])

    def go(backend: str, batch_size: int = 10_000) -> RunResult:
        return run(
            built(Powered, backend),
            model_points=points,
            settings=RunSettings(valuation_date=VAL, batch_size=batch_size),
            run_dir=tmp_path / f"{backend}-{batch_size}",
        )

    three = on_both(go)
    one = go("numba", batch_size=1)
    for name in three.arrays:
        assert np.array_equal(one.var(name), three.var(name)), name


CALLS_SCHEMA = Schema(
    fields=(
        Field("policy_id", "string", unique=True),
        Field("f", "number", "Factor", gt=0),
        Field("i", "number", "Rate[Annual]", ge=0),
        Field("q", "number", "Prob[Annual]", ge=0, le=1),
        Field("s", "number", "Rate[Annual, Continuous]", ge=0),
        Field("amt", "number", "Money", ge=0),
        int_field("k", ge=0, le=5),
        int_field("rem_m", ge=1, le=1000),
    ),
    horizon="rem_m",
)


class Calls(Product):
    """Every formula function the NumPy backend emits, with integer arithmetic, booleans, a
    negative literal exponent of zero in a discarded branch and an Int that takes a float."""

    model_point = CALLS_SCHEMA

    @var
    def base(self, t: Int) -> Factor:
        return self.mp.f / (t + 1)

    @var
    def e(self, t: Int) -> Factor:
        return exp(self.base(t))

    @var
    def lg(self, t: Int) -> Factor:
        return log(self.base(t))

    @var
    def em(self, t: Int) -> Factor:
        return expm1(self.base(t))

    @var
    def lp(self, t: Int) -> Factor:
        return log1p(self.base(t))

    @var
    def sq(self, t: Int) -> Factor:
        return sqrt(self.base(t))

    @var
    def mn(self, t: Int) -> Factor:
        return minimum(self.e(t), self.lg(t))

    @var
    def mx(self, t: Int) -> Factor:
        return maximum(self.e(t), self.lg(t))

    @var
    def ab(self, t: Int) -> Factor:
        return abs(self.lg(t))

    @var
    def tm(self, t: Int) -> Rate[Monthly]:
        return to_monthly(self.mp.i)

    @var
    def tmp(self, t: Int) -> Prob[Monthly]:
        return to_monthly_prob(self.mp.q)

    @var
    def tmc(self, t: Int) -> Prob[Monthly]:
        return to_monthly_cf(self.mp.q)

    @var
    def step(self, t: Int) -> Factor:
        return annual_step(self.mp.i)

    @var
    def ccd(self, t: Int) -> Factor:
        return cc_discount(self.mp.s, months=t)

    @var
    def fr(self, t: Int) -> Rate[Monthly]:
        return fwd_rate(self.ccd(t - 1), self.ccd(t))

    @var
    def af(self, t: Int) -> Rate[Annual]:
        return ann_fwd(self.ccd(t - 1), self.ccd(t))

    @var
    def spot(self, t: Int) -> Rate[Annual]:
        return ann_spot(self.ccd(t), t)  # -12 / 0 at t = 0: 1 ** -inf = 1

    @var
    def cu(self, t: Int) -> CoverageUnits:
        return as_cu(self.mp.amt * self.e(t))

    @var
    def pw(self, t: Int) -> Factor:
        return self.e(t) ** self.mp.k

    @var
    def pw2(self, t: Int) -> Factor:
        return 2 ** (t // 12)  # type: ignore[operator] - an int literal to an Int power

    @var
    def zero(self, t: Int) -> Factor:
        return self.base(t) * 0

    @var
    def recip(self, t: Int) -> Factor:
        return where(t > 1000, self.zero(t) ** -1, 1.0)  # 0.0 ** -1 in the discarded branch

    @var
    def iq(self, t: Int) -> Int:
        return ((t - 7) // 3) + ((t - 7) % 3) + ((t * 7) // 5)

    @var
    def b(self, t: Int) -> Bool:
        return ((t > 3) & ~(t > 9)) | (t == 1)

    @var
    def wb(self, t: Int) -> Factor:
        return where(self.b(t), 1.5, -self.e(t))

    @var
    def wi(self, t: Int) -> Int:
        return where(self.b(t), 2, 0.5)  # type: ignore[return-value] - truncated to 0

    @var
    def total(self) -> Factor:
        return at(self.e, 3) + at(self.sq, self.mp.k)


def calls_points() -> Any:
    n = 4
    return pa.table(
        {
            "policy_id": pa.array([f"C{i}" for i in range(n)], pa.string()),
            "f": pa.array([0.5, 1.0, 2.0, 1e-9], pa.float64()),
            "i": pa.array([0.0, 0.03, 0.05, 0.5], pa.float64()),
            "q": pa.array([0.0, 0.01, 0.5, 1.0], pa.float64()),
            "s": pa.array([0.0, 0.02, 0.04, 0.1], pa.float64()),
            "amt": pa.array([0.0, 100.0, 2500.0, 1e6], pa.float64()),
            "k": pa.array([1, 1, 2, 5], pa.int32()),
            "rem_m": pa.array([30] * n, pa.int32()),
        }
    )


def test_every_formula_function(tmp_path: Path) -> None:
    def go(backend: str) -> RunResult:
        return run(
            built(Calls, backend),
            model_points=calls_points(),
            settings=RunSettings(valuation_date=VAL),
            run_dir=tmp_path / backend,
        )

    got = on_both(go)
    assert got.graph is not None
    used = {
        n.fn
        for v in got.graph.variables.values()
        for n in ir.walk(v.expr)
        if isinstance(n, ir.Call)
    }
    emitted = {"exp", "log", "expm1", "log1p", "sqrt", "minimum", "maximum", "abs"} | {
        "to_monthly",
        "to_monthly_prob",
        "to_monthly_cf",
        "annual_step",
        "cc_discount",
        "fwd_rate",
        "ann_fwd",
        "ann_spot",
        "as_cu",
    }
    assert used == emitted  # every Call of the NumPy backend's table
    assert list(got.var("wi")[:6, 0, 0]) == [0, 2, 0, 0, 2, 2]
    assert np.array_equal(got.var("recip")[1:, :, 0], np.ones((30, 4)))  # a month quantity


def test_the_policy_loop_in_parallel(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """With ``PARALLEL`` set, a kernel with a policy dimension runs its blocks with
    ``numba.prange``; no reduction crosses policies, so the results are the serial ones."""

    class ParallelExpensed(Expensed):
        pass

    monkeypatch.setattr(numba_backend, "PARALLEL", True)
    compiled = build(ParallelExpensed, backend="numba")
    assert "for _b in numba.prange((P + 63) // 64):" in compiled.source
    assert "for _b in range((1 + 63) // 64):" in compiled.source  # v_m has no policy axis
    monkeypatch.undo()
    got = run(
        compiled,
        model_points=simple_model_points(70),  # two blocks, the second partial
        settings=RunSettings(valuation_date=VAL),
        run_dir=tmp_path / "parallel",
    )
    serial = run(
        built(Expensed, "numba"),
        model_points=simple_model_points(70),
        settings=RunSettings(valuation_date=VAL),
        run_dir=tmp_path / "serial",
    )
    for name in serial.arrays:
        assert np.array_equal(got.arrays[name], serial.arrays[name]), name


# ------------------------------------------------------------------------- determinism


def test_reruns_are_bit_identical_d19(tmp_path: Path) -> None:
    """Two Numba runs of the same inputs and batching: the same run id, the same output files
    and the same arrays, bit for bit."""
    inputs = Path(__file__).resolve().parent.parent / "docs" / "spec" / "term-assurance"
    names = ("mort_table", "lapse_table", "comm_table", "rates", "expenses")
    results = [
        run(
            built(TermAssurance, "numba"),
            model_points=inputs / "model_points.parquet",
            tables={name: inputs / f"{name}.parquet" for name in names},
            settings=RunSettings(valuation_date=VALUATION_DATE, batch_size=3),
            run_dir=tmp_path / f"run-{i}",
        )
        for i in range(2)
    ]
    a, b = results
    assert a.run_id == b.run_id
    for key in ("monthly_sha256", "scalars_sha256"):
        assert a.manifest["outputs"][key] == b.manifest["outputs"][key]
    for name in a.arrays:
        assert np.array_equal(a.arrays[name], b.arrays[name]), name


FMA_SCHEMA = Schema(
    fields=(
        Field("policy_id", "string", unique=True),
        Field("a", "number", "Factor"),
        int_field("rem_m", ge=1, le=12),
    ),
    horizon="rem_m",
)


class Fma(Product):
    """``a * a - 1`` with ``a = 1 + 2**-27``: rounded after the multiplication it is 2**-26;
    fused into one FMA it would be 2**-26 + 2**-54. Once in whole mode, once inside a forward
    recursion."""

    model_point = FMA_SCHEMA

    @var
    def whole(self, t: Int) -> Factor:
        return self.mp.a * self.mp.a - 1

    @var(init=0.0)
    def stepped(self, t: Int) -> Factor:
        return self.stepped(t - 1) * 0 + (self.mp.a * self.mp.a - 1)


def test_no_fma_contraction(tmp_path: Path) -> None:
    a = 1 + 2**-27
    assert a * a - 1 == 2.0**-26  # Python rounds after the multiplication
    points = pa.table(
        {
            "policy_id": pa.array(["F0"], pa.string()),
            "a": pa.array([a], pa.float64()),
            "rem_m": pa.array([3], pa.int32()),
        }
    )
    values: dict[str, RunResult] = {}
    for backend in BACKEND_NAMES:
        values[backend] = run(
            built(Fma, backend),
            model_points=points,
            settings=RunSettings(valuation_date=VAL),
            run_dir=tmp_path / backend,
        )
    for name in ("whole", "stepped"):
        want = values["numpy"].lane(name, "F0")
        got = values["numba"].lane(name, "F0")
        assert np.array_equal(got, want), name
        assert got[1] == 2.0**-26, name


# --------------------------------------------------------------------- run-time checks


def test_a_probability_out_of_range(tmp_path: Path) -> None:
    class BadProb(Product):
        model_point = SIMPLE_SCHEMA

        @var
        def q(self, t: Int) -> Prob[Monthly]:
            return -0.5 + 0.0  # type: ignore[return-value]

        @var
        def x(self, t: Int) -> Count:
            return self.mp.pols_if_init * self.q(t)

    err = same_error(
        lambda backend: run(
            build(BadProb, backend=backend),
            model_points=simple_model_points(),
            settings=RunSettings(valuation_date=VAL),
            run_dir=tmp_path / backend,
        ),
        r"q is a probability but takes values outside \[0, 1\]: -0\.5 to -0\.5; policies S00, "
        r"S01, S02",
    )
    assert err.loc is not None
    assert err.loc.function.endswith("BadProb.q")


def test_a_lookup_key_outside_its_table(reference_sources: dict[str, Any], tmp_path: Path) -> None:
    class Old(TermAssurance):
        @var
        def age(self, t: Int) -> Int:
            return self.mp.age_at_entry + self.dur_y(t) + 100

    err = same_error(
        lambda backend: run(
            build(Old, backend=backend),
            model_points=reference_sources["model_points"],
            tables=reference_sources["tables"],
            settings=RunSettings(valuation_date=VALUATION_DATE),
            run_dir=tmp_path / backend,
        ),
        r"key age of table mort_table .* P01",
    )
    assert err.loc is not None
    assert err.loc.function.endswith("q_ann")


BUMP = Table("bump", keys=(Key("k", lo=0, hi=10),), columns=(Column("v", "Factor"),))


def bump_table() -> Any:
    return pa.table(
        {"k": pa.array(range(11), pa.int64()), "v": pa.array([0.1 * i for i in range(11)])}
    )


class Bumped(Product):
    """A table read whose key ``k + t`` leaves the table at t = 11 - k, inside a forward
    recursion (``acc``), in whole mode (``whole``) and with a key that has no policy axis
    (``shared``)."""

    model_point = SIMPLE_SCHEMA
    assumption_tables = (BUMP,)

    @var(init=0.0)
    def acc(self, t: Int) -> Factor:
        return self.acc(t - 1) + self.tables.bump[self.mp.k + t]


class BumpedWhole(Product):
    model_point = SIMPLE_SCHEMA
    assumption_tables = (BUMP,)

    @var
    def whole(self, t: Int) -> Factor:
        return self.tables.bump[self.mp.k + t]


class BumpedShared(Product):
    model_point = SIMPLE_SCHEMA
    assumption_tables = (BUMP,)

    @var
    def shared(self, t: Int) -> Count:
        return self.tables.bump[t + 5] * self.mp.pols_if_init


class BumpedStep(Product):
    """A key built from ``t`` alone inside a recursion: the NumPy backend checks a scalar
    there, so its message names the value and no policies."""

    model_point = SIMPLE_SCHEMA
    assumption_tables = (BUMP,)

    @var(init=0.0)
    def acc(self, t: Int) -> Factor:
        return self.acc(t - 1) + self.tables.bump[t + 5]


@pytest.mark.parametrize(
    ("model", "variable", "pattern"),
    [
        # policies k = 2, 3, 4: the recursion fails first at t = 7, for S02 alone
        (
            Bumped,
            "acc",
            r"^lookup key k of table bump is outside its declared range \[0, 10\]: value 11 to "
            r"11; no silent extrapolation; policies S02$",
        ),
        # every month at once: k + t from 11 to 16 over the three policies
        (
            BumpedWhole,
            "whole",
            r"value 11 to 16; no silent extrapolation; policies S00, S01, S02$",
        ),
        # a key without a policy axis fails for every policy: t + 5 from 11 to 17
        (
            BumpedShared,
            "shared",
            r"value 11 to 17; no silent extrapolation; policies S00, S01, S02$",
        ),
        # a scalar key, at t = 6, names no policies
        (BumpedStep, "acc", r"value 11 to 11; no silent extrapolation$"),
    ],
    ids=["step", "whole", "shared", "scalar"],
)
def test_a_key_that_leaves_its_table(
    model: type[Product], variable: str, pattern: str, tmp_path: Path
) -> None:
    err = same_error(
        lambda backend: run(
            build(model, backend=backend),
            model_points=simple_model_points(),
            tables={"bump": bump_table()},
            settings=RunSettings(valuation_date=VAL),
            run_dir=tmp_path / backend,
        ),
        pattern,
    )
    assert err.loc is not None
    assert err.loc.function == f"{model.__qualname__}.{variable}"
    assert Path(err.loc.file) == Path(__file__)


def test_a_non_finite_result(tmp_path: Path) -> None:
    class Inf(Product):
        model_point = SIMPLE_SCHEMA

        @var
        def x(self, t: Int) -> Factor:
            return 1.0 / (t - t)

    same_error(
        lambda backend: run(
            build(Inf, backend=backend),
            model_points=simple_model_points(),
            settings=RunSettings(valuation_date=VAL),
            run_dir=tmp_path / backend,
        ),
        r"^x is not finite \(NaN or infinity\) at some month; policies S00, S01, S02$",
    )


def test_a_fixed_month_outside_the_horizon(tmp_path: Path) -> None:
    class Late(Growth):
        @var
        def late(self) -> Money:
            return at(self.cf, 50)

    same_error(
        lambda backend: run(
            build(Late, backend=backend),
            model_points=simple_model_points(),
            settings=RunSettings(valuation_date=VAL),
            run_dir=tmp_path / backend,
        ),
        r"^at\(cf, 50\) is outside the domain of cf, months 1..12$",
    )


class Ahead(Growth):
    """``at(x, t + 1)`` reaches month H + 1 at t = H for every policy."""

    @var
    def ahead(self, t: Int) -> Count:
        return at(self.x, t + 1)


def test_an_at_that_fails_for_every_lane(tmp_path: Path) -> None:
    for batch_size, named in ((1, "S00"), (2, "S00, S01"), (3, "S00, S01, S02")):
        same_error(
            lambda backend, batch_size=batch_size: run(
                build(Ahead, backend=backend),
                model_points=simple_model_points(),
                settings=RunSettings(valuation_date=VAL, batch_size=batch_size),
                run_dir=tmp_path / f"{backend}-{batch_size}",
            ),
            rf"at\(x, \.\.\.\) reads a month outside the domain of x, months 0\.\.12; "
            rf"policies {named}$",
        )


class AheadStep(Growth):
    """``at(x, t + 1)`` inside a forward recursion: a scalar month, named in the message."""

    @var(init=0.0)
    def peek(self, t: Int) -> Count:
        return self.peek(t - 1) * 0 + at(self.x, t + 1) * 0


def test_an_at_with_a_scalar_month_in_a_recursion(tmp_path: Path) -> None:
    same_error(
        lambda backend: run(
            build(AheadStep, backend=backend),
            model_points=simple_model_points(),
            settings=RunSettings(valuation_date=VAL),
            run_dir=tmp_path / backend,
        ),
        r"^at\(x, 13\) is outside the domain of x, months 0\.\.12$",
    )


class ScenAhead(Indexed):
    @var
    def ahead(self, t: Int) -> Factor:
        return self.scen.df.at(t + 1) / self.scen.df.at(0)


def test_a_scenario_month_outside_the_file(tmp_path: Path) -> None:
    horizon = 12
    df = scenario_file("df", 0, horizon, 2, lambda s, u: 0.99**u)
    cpi = scenario_file("cpi", -3, horizon, 2, lambda s, u: 100.0)
    same_error(
        lambda backend: run(
            build(ScenAhead, backend=backend),
            model_points=simple_model_points(rem_m=horizon),
            scenarios={"df": df, "cpi": cpi},
            settings=RunSettings(valuation_date=VAL),
            run_dir=tmp_path / backend,
        ),
        r"^scen\.df\.at\(\.\.\.\) reads a month outside the scenario file; policies S00, S01, "
        r"S02$",
    )


def test_a_series_month_outside_the_series(tmp_path: Path) -> None:
    bases = release_bases(tmp_path / "bases")
    keyed = group_series(Keyed, bases, MEMBERS)
    same_error(
        lambda backend: run(
            build(Keyed, backend=backend),
            model_points=_keyed_groups((5, 0)),
            series=keyed,
            settings=RunSettings(valuation_date=VAL),
            run_dir=tmp_path / f"keyed-{backend}",
        ),
        r"^series.cu_g.at\(...\) reads a month outside the months of series cu_g, 1..12; "
        r"groups G0$",
    )
    every = group_series(AtAll, bases, MEMBERS)
    for batch_size, named in ((1, "G0"), (2, "G0, G1")):
        same_error(
            lambda backend, batch_size=batch_size: run(
                build(AtAll, backend=backend),
                model_points=stage_groups(),
                series=every,
                settings=RunSettings(valuation_date=VAL, batch_size=batch_size),
                run_dir=tmp_path / f"all-{backend}-{batch_size}",
            ),
            r"^series\.cu_g\.at\(\.\.\.\) reads a month outside the months of series cu_g, "
            rf"1\.\.12; groups {named}$",
        )


# ------------------------------------------------------------- the run directory and build


def test_the_run_directory_and_manifest(tmp_path: Path) -> None:
    compiled = built(Growth, "numba")
    assert compiled.backend == "numba"
    res = run(
        compiled,
        model_points=simple_model_points(),
        settings=RunSettings(valuation_date=VAL),
        run_dir=tmp_path / "numba",
    )
    numpy_run = run(
        built(Growth, "numpy"),
        model_points=simple_model_points(),
        settings=RunSettings(valuation_date=VAL),
        run_dir=tmp_path / "numpy",
    )
    d = res.run_dir
    generated = d / "generated" / "numba_backend.py"
    assert generated.read_text(encoding="utf-8") == compiled.source
    assert not (d / "generated" / "numpy_backend.py").exists()
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    llvmlite: Any = importlib.import_module("llvmlite")
    numba: Any = importlib.import_module("numba")

    block = manifest["backend"]
    assert block["name"] == "numba"
    assert block["numba"] == numba.__version__
    assert block["llvmlite"] == llvmlite.__version__
    assert block["numpy"] == np.__version__
    assert block["using_svml"] == str(bool(numba.config.USING_SVML))
    assert "processor" in block
    assert "platform" in block
    assert "pyarrow" in block
    assert set(numpy_run.manifest["backend"]) == {"name", "numpy", "pyarrow", "python", "platform"}
    # the identity adds the Numba and llvmlite versions: another backend is another run
    assert res.run_id != numpy_run.run_id
    back = load_run(d, verify=True)
    assert back.run_id == res.run_id
    for name in res.arrays:
        assert np.array_equal(back.var(name), res.var(name)), name


def test_the_generated_module_and_its_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The module is written to ``HALLEY_NUMBA_CACHE / <ir_hash>.py`` (the source exactly) and
    compiled there; Numba's cache files land beside it, and compiling the same source again
    neither rewrites the file nor recompiles."""

    class Cached(Growth):
        pass

    monkeypatch.setenv(numba_backend.CACHE_ENV, str(tmp_path / "cache"))
    compiled = build(Cached, backend="numba")
    path = tmp_path / "cache" / f"{compiled.ir_hash}.py"
    assert path.read_text(encoding="utf-8") == compiled.source
    assert f"IR_SHA256 = {compiled.ir_hash!r}" in compiled.source
    cached = list((tmp_path / "cache" / "__pycache__").glob(f"{compiled.ir_hash}.k_*.nbi"))
    assert len(cached) == len(compiled.schedule.components)
    stamp = path.stat().st_mtime_ns
    numba_backend._LOADED.clear()  # as a fresh process would: the file is there, the kernels too
    again = numba_backend.compile_source(compiled.source, compiled.name)
    assert path.stat().st_mtime_ns == stamp
    module = numba_backend._LOADED[hashlib.sha256(compiled.source.encode()).hexdigest()]
    for entry in module.KERNELS:
        assert getattr(module, entry[0]).stats.cache_hits, entry[0]
    assert again is module.run_batch
    # the generated code keeps the model's grouping and compiles with fast-math off
    assert 'fastmath=False, error_model="numpy"' in compiled.source
    assert "V_r[t, p, s] = ((V_r[t+1, p, 0] + V_cf[t+1, p, 0]) * 0.5)" in compiled.source


def test_a_kernel_numba_cannot_compile_is_a_build_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A kernel that fails Numba's typing (here a module edited so that ``cf`` stores a
    string) is a BuildError at the model line of its component's first variable, not a
    traceback into the generated module."""
    compiled = built(Growth, "numba")
    statement = "V_cf[t, p, s] = (V_x[t-1, p, 0] * M_premium[0, p, 0])"
    assert statement in compiled.source
    edited = compiled.source.replace(statement, "V_cf[t, p, s] = 'cf'").replace(
        compiled.ir_hash, "f" * 64
    )
    monkeypatch.setenv(numba_backend.CACHE_ENV, str(tmp_path / "cache"))
    with pytest.raises(
        BuildError, match=r"^the Numba backend could not compile the kernel of cf: Typing"
    ) as info:
        numba_backend.compile_source(edited, compiled.name)
    assert info.value.loc == compiled.graph.variables["cf"].loc
    assert f"halley_numba_{'f' * 64}" not in sys.modules


def test_a_node_the_generator_cannot_emit_is_a_build_error() -> None:
    compiled = built(Growth, "numpy")
    graph = copy.deepcopy(compiled.graph)
    loc = graph.variables["cf"].loc

    class Mystery(ir.Node):
        pass

    graph.variables["cf"].expr = ir.Binary(loc, "mul", Mystery(loc), ir.Const(loc, 2.0))
    with pytest.raises(BuildError, match="the Numba backend cannot emit a Mystery node") as info:
        numba_backend.generate(graph, compiled.schedule)
    assert info.value.loc == loc
    graph.variables["cf"].expr = ir.Call(loc, "erf", (ir.Const(loc, 0.5),))
    with pytest.raises(BuildError, match=r"the Numba backend has no form for erf\(\)") as info:
        numba_backend.generate(graph, compiled.schedule)
    assert info.value.loc == loc


def test_the_registry_names_both_backends() -> None:
    assert sorted(BACKENDS) == ["numba", "numpy"]
    assert BACKENDS["numba"].artefact == "numba_backend.py"
    assert BACKENDS["numpy"].artefact == "numpy_backend.py"
