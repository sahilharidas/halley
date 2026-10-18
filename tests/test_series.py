# pyright: basic
"""The aggregation stage's language and IR (design section 9, "Language and IR"; the addendum's
section 9 clarifications): ``Aggregation``, ``Basis`` and ``GroupSum`` declarations, the
``self.series`` reads and their IR nodes, units, dimensions and domains in the checker, the
generated code in all three contexts, ``explain``, the in-session series builder, and a toy
stage run end to end with outputs keyed on ``group_id``."""

from __future__ import annotations

import dataclasses
import datetime as dt
import functools
import hashlib
import importlib
import inspect
import json
import operator
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from halley import (
    Aggregation,
    Basis,
    BuildError,
    BuildWarning,
    CoverageUnits,
    Factor,
    GroupSum,
    Int,
    LoadError,
    Money,
    Product,
    RunError,
    RunResult,
    RunSettings,
    ScenarioInput,
    Schema,
    UnitError,
    build,
    check,
    explain,
    ir,
    load_run,
    run,
    trace,
    var,
)
from halley.aggregation import group_series
from halley.authoring import int_field
from halley.errors import Loc
from halley.ir import units as u
from halley.runtime.explain import recompute
from halley.runtime.loader import LoadedSeries
from halley.runtime.run import VariableInfo
from models import (
    MEMBERS,
    SIMPLE_SCHEMA,
    STAGE_SCHEMA,
    VAL,
    Growth,
    Release,
    closed_form_release,
    covered_points,
    release_bases,
    run_covered,
    run_stage,
    scenario_file,
    simple_model_points,
    stage_groups,
)

run_module = importlib.import_module("halley.runtime.run")  # the package's run is the function

CU_G = GroupSum("cu_g", "cur", "cu", "CoverageUnits")
FCF_G = GroupSum("fcf_g", "cur", "fcf", "Money", start=0)
P_G = GroupSum("p_g", "cur", "premium_total", "Money", time=False)


@pytest.fixture(scope="module")
def cur(tmp_path_factory: Any) -> dict[str, list[RunResult]]:
    """The deterministic ``cur`` basis: two class runs, A.. and B.., one scenario."""
    return release_bases(tmp_path_factory.mktemp("cur"))


@pytest.fixture(scope="module")
def cur_j2(tmp_path_factory: Any) -> dict[str, list[RunResult]]:
    """The same basis with two scenarios: claims at 1.0 and 1.5 times the deterministic ones,
    the second onerous (its fulfilment cash flows exceed the premium), their mean not."""
    return release_bases(tmp_path_factory.mktemp("cur-j2"), shocks=(1.0, 1.5))


@pytest.fixture(scope="module")
def stage(cur: dict[str, list[RunResult]], tmp_path_factory: Any) -> RunResult:
    return run_stage(tmp_path_factory.mktemp("stage"), Release, group_series(Release, cur, MEMBERS))


def _inside(loc: Loc | None, model: type[Product], variable: str) -> None:
    """The error points at a line of ``variable``'s method, in the class of ``model``'s MRO
    that defines it."""
    assert loc is not None, "every build error carries a model source location"
    owner = next(k for k in model.__mro__ if variable in vars(k))
    fn = vars(owner)[variable].fn
    lines, first = inspect.getsourcelines(fn)
    assert Path(loc.file) == Path(fn.__code__.co_filename), loc
    assert loc.function.endswith(f"{owner.__name__}.{variable}"), loc
    assert first <= loc.line < first + len(lines), (loc, first, len(lines))


def _fails(model: type[Product], variable: str, match: str, error: type = BuildError) -> Any:
    with pytest.raises(error, match=match) as info:
        check(trace(model))
    _inside(info.value.loc, model, variable)
    return info.value


def _at_class(loc: Loc | None, model: type) -> None:
    """The error points at a line of ``model``'s class body."""
    assert loc is not None
    lines, first = inspect.getsourcelines(model)
    assert Path(loc.file) == Path(__file__), loc
    assert first <= loc.line < first + len(lines), (loc, first, len(lines))


def _line_of(marker: str) -> int:
    """The line of this file that ends with the comment ``# <marker>``."""
    lines = Path(__file__).read_text(encoding="utf-8").splitlines()
    found = [i + 1 for i, text in enumerate(lines) if text.rstrip().endswith(f"# {marker}")]
    assert len(found) == 1, (marker, found)
    return found[0]


def _summed(runs: list[RunResult], name: str, members: list[str], s: int = 0) -> list[float]:
    """A group total written independently: the members' lanes, wherever their run, added one
    by one in ascending policy id as Python floats."""
    lanes = [_run_of(runs, pid).lane(name, pid, s).tolist() for pid in sorted(members)]
    return [functools.reduce(operator.add, column) for column in zip(*lanes, strict=True)]


def _run_of(runs: list[RunResult], pid: str) -> RunResult:
    return next(r for r in runs if pid in r.ids.tolist())


def _series(res: RunResult) -> dict[str, LoadedSeries]:
    """The group series a stage run of this session read, as the batches saw them."""
    assert res.inputs is not None
    return res.inputs.series


# ------------------------------------------------------------------------- the IR and checker


def test_a_stage_declares_its_series_in_the_ir() -> None:
    graph = trace(Release)
    assert graph.series == {
        "cu_g": ir.SeriesDecl("cu_g", u.COVERAGE_UNITS, True, 1, False),
        "fcf_g": ir.SeriesDecl("fcf_g", u.MONEY, True, 0, False),
        "p_g": ir.SeriesDecl("p_g", u.MONEY, False, 0, False),
    }
    each = dataclasses.replace(FCF_G, scenarios="each")
    assert each.decl().scenario  # "each" keeps the basis's scenarios; a mean has one
    assert Growth.series_declarations() is None  # a projection reads no series
    assert trace(Growth).series == {}
    reads = [n for v in graph.variables.values() for n in ir.walk(v.expr)]
    assert {type(n).__name__ for n in reads} >= {"SeriesRead", "SeriesAt"}


def test_render_writes_series_reads_as_the_model_does() -> None:
    """A scalar series read is written ``self.series.p_g()`` in the model and has no month, so
    its node has no offset and renders without a ``t``."""
    assert ir.render(ir.SeriesRead(None, "cu_g", -1)) == "series.cu_g(t-1)"
    assert ir.render(ir.SeriesRead(None, "cu_g", 1)) == "series.cu_g(t+1)"
    assert ir.render(ir.SeriesRead(None, "cu_g", 0)) == "series.cu_g(t)"
    assert ir.render(ir.SeriesRead(None, "p_g", None)) == "series.p_g()"
    assert ir.render(ir.SeriesAt(None, "fcf_g", ir.Const(None, 0))) == "series.fcf_g.at(0)"
    assert ir.render(ir.SeriesAt(None, "fcf_g", ir.MpField(None, "k"))) == "series.fcf_g.at(mp.k)"
    csm_0 = trace(Release).variables["csm_0"]
    assert ir.render(csm_0.expr) == (
        "where(((series.p_g() - series.fcf_g.at(0)) > 0), (series.p_g() - series.fcf_g.at(0)), 0.0)"
    )
    p_g = next(n for n in ir.walk(csm_0.expr) if isinstance(n, ir.SeriesRead))
    assert p_g.offset is None


def test_the_graph_hash_covers_the_series_declarations() -> None:
    """Series declarations are hashed like scenario declarations: the same stage gives the same
    hash, and a series' domain or scenario treatment changes it even where no formula does. A
    projection declares none, so its hash is unchanged (the pinned hashes in test_tracing)."""
    base = ir.graph_hash(trace(Release))
    assert ir.graph_hash(trace(Release)) == base
    for change in ({"scenario": True}, {"start": 1}, {"unit": u.FACTOR}):
        g = trace(Release)
        g.series["fcf_g"] = dataclasses.replace(g.series["fcf_g"], **change)
        assert ir.graph_hash(g) != base, change


def test_series_units_come_from_the_declaration() -> None:
    graph = check(trace(Release))
    units = {ir.render(n): n.unit for v in graph.variables.values() for n in ir.walk(v.expr)}
    assert units["series.cu_g(t+1)"] == u.COVERAGE_UNITS
    assert units["series.fcf_g.at(0)"] == u.MONEY
    assert units["series.p_g()"] == u.MONEY

    class Mixed(Release):
        @var
        def bad(self, t: Int) -> Money:
            return self.series.cu_g(t) + self.series.fcf_g(t)

    _fails(Mixed, "bad", r"unit mismatch: CoverageUnits \+ Money", UnitError)

    class IndexUnit(Release):
        @var
        def bad(self) -> Money:
            return self.series.fcf_g.at(self.group.j)

    _fails(IndexUnit, "bad", r"the month in series.fcf_g.at\(...\) is an Int", UnitError)


def test_series_dimensions_are_fixed_by_the_declaration() -> None:
    """A time series is (time, group, scenario), a scalar one (group, scenario): the scenario
    axis is there whatever J the basis has (D4), and a scalar variable that reads a series at a
    relative month names it as the culprit."""
    graph = check(trace(Release))
    every = frozenset({ir.TIME, ir.POLICY, ir.SCENARIO})
    assert graph.variables["cu_next"].dims == every
    assert graph.variables["csm_0"].dims == frozenset({ir.POLICY, ir.SCENARIO})
    assert graph.variables["d"].dims == frozenset({ir.POLICY})
    dims = {ir.render(n): n.dims for v in graph.variables.values() for n in ir.walk(v.expr)}
    assert dims["series.cu_g(t+1)"] == every
    assert dims["series.p_g()"] == frozenset({ir.POLICY, ir.SCENARIO})
    assert dims["series.fcf_g.at(0)"] == frozenset({ir.POLICY, ir.SCENARIO})

    g = trace(Release)  # the authoring API has no t in a scalar formula; the IR can say it
    g.variables["csm_0"].expr = ir.SeriesRead(None, "fcf_g", 0)
    with pytest.raises(BuildError, match=r"csm_0 has no time dimension .* reads series.fcf_g\(t\)"):
        check(g)
    # the node's offset says whether it reads a month, and must agree with the declaration
    for node, match in (
        (ir.SeriesRead(None, "p_g", 0), r"series p_g has no time dimension; read it as"),
        (ir.SeriesRead(None, "fcf_g", None), r"series fcf_g has a time dimension; read it at a"),
    ):
        g = trace(Release)
        g.variables["csm_0"].expr = node
        with pytest.raises(BuildError, match=match):
            check(g)


def test_series_reads_stay_inside_the_series_domain() -> None:
    """The offsets are checked like variable reads, from the reader's formula range: no month
    before the series' start, no month after H (or H - 1 from a backward recursion)."""

    class EarlyAt(Release):
        series = (CU_G, GroupSum("fcf_g", "cur", "fcf", "Money"), P_G)  # start 1, not 0

    _fails(
        EarlyAt,
        "csm_0",
        r"csm_0 reads series.fcf_g.at\(0\), which is before the start of series fcf_g at t = 1; "
        r"a sum of states declares GroupSum\(..., start=0\)",
    )

    class Lagged(Release):
        @var(init=0.0)
        def lagged(self, t: Int) -> CoverageUnits:
            return self.lagged(t - 1) + self.series.cu_g(t - 1)

    _fails(
        Lagged,
        "lagged",
        r"lagged reads series.cu_g\(t-1\), which is outside the domain of series cu_g at t = 1: "
        r"the series starts at t = 1",
    )

    class TwoBack(Release):
        @var(init=0.0)
        def lagged(self, t: Int) -> Money:
            return self.lagged(t - 1) + self.series.fcf_g(t - 2)

    _fails(TwoBack, "lagged", r"reads series.fcf_g\(t-2\), which is before t = 0 at t = 1")

    class Peek(Release):
        @var
        def peek(self, t: Int) -> CoverageUnits:
            return self.series.cu_g(t + 1)

    _fails(
        Peek,
        "peek",
        r"peek reads series.cu_g\(t\+1\), which is outside the domain of series cu_g at t = H; "
        r"a variable that reads a later month is backward recursive and declares a terminal",
    )

    class PeekTwo(Release):
        @var(terminal=0)
        def ahead(self, t: Int) -> CoverageUnits:
            return self.ahead(t + 1) + self.series.cu_g(t + 2)

    _fails(PeekTwo, "ahead", r"outside the domain of series cu_g at t = H-1")

    class Before(Release):
        @var
        def first(self) -> Money:
            return self.series.fcf_g.at(-1)

    _fails(Before, "first", r"first reads series.fcf_g.at\(-1\), which is before t = 0")


def test_a_boundary_reads_a_series_only_at_a_fixed_month() -> None:
    """The boundary allow-list admits ``series.x.at(k)`` with a constant ``k`` and rejects a
    relative series read and a non-constant month; the authoring API offers no series in a
    boundary at all (it reaches one through a scalar variable, as ``csm_0``)."""
    fixed = "a boundary value is fixed before the months run"
    g = trace(Release)
    g.variables["csm"].init = ir.SeriesAt(None, "fcf_g", ir.Const(None, 0))
    check(g)
    for node, text in (
        (ir.SeriesRead(None, "p_g", None), r"series.p_g\(\)"),
        (ir.SeriesRead(None, "fcf_g", 0), r"series.fcf_g\(t\)"),
        (
            ir.SeriesAt(
                None, "fcf_g", ir.Binary(None, "add", ir.Const(None, 0), ir.RunConst(None, "m_val"))
            ),
            r"series.fcf_g.at\(\(0 \+ run.m_val\)\)",
        ),
    ):
        g = trace(Release)
        g.variables["csm"].init = node
        with pytest.raises(BuildError, match=rf"the init value of csm reads {text}; {fixed}"):
            check(g)

    class Direct(Release):
        @var(init=lambda g: g.series.p_g())
        def csm(self, t: Int) -> Money:
            return (self.csm(t - 1) + self.acc(t)) - self.release(t)

    with pytest.raises(BuildError, match="a boundary value reads 'series', which is neither"):
        trace(Direct)


def test_series_reads_are_checked_where_they_are_written() -> None:
    """A read of an undeclared series, the wrong call for a series' dimensions, a fixed month
    written as a call: BuildErrors at the formula's line."""

    class Unknown(Release):
        @var
        def x(self, t: Int) -> Money:
            return self.series.bel_g(t)

    exc = _fails(
        Unknown, "x", "the stage declares no series 'bel_g'; it declares: cu_g, fcf_g, p_g"
    )
    assert exc.loc.line == inspect.getsourcelines(vars(Unknown)["x"].fn)[1] + 2  # the return

    class ScalarAtT(Release):
        @var
        def x(self, t: Int) -> Money:
            return self.series.p_g(t)

    _fails(ScalarAtT, "x", r"series.p_g has no time dimension; read it as self.series.p_g\(\)")

    class ScalarAt(Release):
        @var
        def x(self) -> Money:
            return self.series.p_g.at(0)

    _fails(ScalarAt, "x", r"series.p_g has no time dimension")

    class NoMonth(Release):
        @var
        def x(self, t: Int) -> CoverageUnits:
            return self.series.cu_g()

    _fails(NoMonth, "x", r"series.cu_g\(t\) takes the time index with a constant offset")

    class FixedMonth(Release):
        @var
        def x(self) -> CoverageUnits:
            return self.series.cu_g(3)

    _fails(FixedMonth, "x", r"series.cu_g\(3\) reads a fixed month; write self.series.cu_g.at\(3\)")

    class DataOffset(Release):
        @var
        def x(self, t: Int) -> CoverageUnits:
            return self.series.cu_g(t - self.mp.j)

    _fails(DataOffset, "x", r"the time offset of series.cu_g must be a constant")


def test_the_declarations_are_validated_at_trace() -> None:
    """A series of an undeclared basis is a BuildError at the GroupSum's line; a malformed
    tuple, a name declared twice and a variable taking one of the stage's names are errors at
    the class or the variable; a GroupSum that cannot be right fails where it is written."""

    class NoBasis(Release):
        series = (CU_G, GroupSum("bel_g", "old", "bel", "Money"))

    with pytest.raises(
        BuildError,
        match=r"series bel_g sums basis 'old', which .*NoBasis does not declare; it declares: cur",
    ) as info:
        trace(NoBasis)
    _at_class(info.value.loc, NoBasis)

    class Bare(Release):
        series = CU_G  # the comma of (CU_G,) forgotten

    with pytest.raises(BuildError, match=r"Bare.series must be a tuple of GroupSum \(a one-ele"):
        trace(Bare)

    class Twice(Release):
        series = (CU_G, FCF_G, P_G, GroupSum("cu_g", "cur", "cu", "CoverageUnits"))

    with pytest.raises(BuildError, match="Twice declares series cu_g twice") as info:
        trace(Twice)
    _at_class(info.value.loc, Twice)

    class Grouped(Release):
        @var
        def group(self) -> Money:
            return self.series.p_g()

    _fails(Grouped, "group", r"group is the group's fields \(self.group\) in a stage; rename")

    class Hooked(Product):
        model_point = SIMPLE_SCHEMA

        @var
        def series_declarations(self, t: Int) -> Int:
            return t

    with pytest.raises(BuildError, match="series_declarations is a Product class method"):
        trace(Hooked)

    for kwargs, match in (
        ({"scenarios": "median"}, "scenarios is 'mean' or 'each', not 'median'"),
        ({"start": 2}, "start is 0 .* or 1 .*, not 2"),
        ({"start": True}, "start is 0 .* or 1 .*, not True"),  # True == 1, but not a month
        ({"start": 1.0}, r"start is 0 .* or 1 .*, not 1\.0"),
        ({"time": "yes"}, "series s: time is True or False, not 'yes'"),
        ({"time": False, "start": 1}, r"series s has no time dimension \(time=False\), so no st"),
        ({"time": False, "start": 0}, r"series s has no time dimension \(time=False\), so no st"),
        ({"unit": "Euro"}, "series s: unknown unit 'Euro'"),
        ({"unit": 3}, "series s: unit is a unit's name, as 'Money', not 3"),
        ({"variables": ()}, "series s names its variables"),
        ({"variables": ["cu"]}, "series s names its variables"),
        ({"variables": ("cu", 3)}, "series s names its variables"),
        ({"name": 3}, "a series is named like a variable, not 3"),
        ({"name": "two words"}, "a series is named like a variable, not 'two words'"),
        ({"basis": 3}, "series s: its basis is a Basis name, not 3"),
    ):
        args: dict[str, Any] = {"name": "s", "basis": "cur", "variables": "cu", "unit": "Money"}
        with pytest.raises(BuildError, match=match) as info:
            GroupSum(**{**args, **kwargs})
        assert info.value.loc is not None
        assert Path(info.value.loc.file) == Path(__file__)
    names: list[Any] = [3, "3cur", None]
    for name in names:
        with pytest.raises(BuildError, match="a basis is named like a variable") as info:
            Basis(name)
        assert info.value.loc is not None
        assert Path(info.value.loc.file) == Path(__file__)
    # one declaration, and one hash, whatever the spelling: start=1 is the default
    spelled = GroupSum("s", "cur", "cu", "Money", start=1).decl()
    assert repr(spelled) == repr(GroupSum("s", "cur", "cu", "Money").decl())
    assert (CU_G.decl().start, FCF_G.decl().start, P_G.decl().start) == (1, 0, 0)


def test_a_group_sum_adds_only_what_adds_across_policies() -> None:
    """D11: a group sum keeps the unit of what it adds, so only amounts, numbers of policies and
    coverage units are summed over a group's members. Three members' monthly probabilities of
    0.6 are not a probability of 1.8, nor three flags a flag. A UnitError at the declaration."""
    with pytest.raises(
        UnitError, match=r"^series q_g sums Prob\[Monthly\] over a group's members; a group sum "
    ) as info:
        GroupSum("q_g", "cur", "q", "Prob[Monthly]")  # decl: q_g
    loc = info.value.loc
    assert loc is not None
    assert Path(loc.file) == Path(__file__), loc
    assert loc.line == _line_of("decl: q_g"), loc
    assert "a group sum adds Money, Count or CoverageUnits" in info.value.message
    for unit in (
        "Prob[Monthly]",
        "Prob[Annual]",
        "Rate[Monthly]",
        "Rate[Annual]",
        "Factor",
        "Fraction",
        "Index",
        "Int",
        "Bool",
        "MoneyPerPolicy",
    ):
        with pytest.raises(UnitError, match="over a group's members; a group sum adds Money"):
            GroupSum("q_g", "cur", "q", unit, time=unit != "Bool")
    for unit in ("Money", "Count", "CoverageUnits"):
        assert GroupSum("ok", "cur", "x", unit).declared_unit.name == unit

    with pytest.raises(UnitError, match="series big_g sums Bool over a group's members") as info:

        class SumsFlags(Release):  # a stage's class body, where a declaration is written
            big_g = GroupSum("big_g", "cur", "big", "Bool", time=False)  # decl: big_g
            series = (*Release.series, big_g)

    loc = info.value.loc
    assert loc is not None
    assert loc.line == _line_of("decl: big_g"), loc
    assert loc.function.endswith("SumsFlags"), loc
    with pytest.raises(BuildError, match="scenarios is 'mean' or 'each', not 'all'") as info:
        dataclasses.replace(CU_G, scenarios="all")
    assert info.value.loc is not None  # the replace() line here, not inside dataclasses
    assert Path(info.value.loc.file) == Path(__file__)
    with pytest.raises(BuildError, match="build one of its subclasses"):
        build(Aggregation)


# --------------------------------------------------------------------------- the toy stage


def test_a_toy_stage_runs_end_to_end(stage: RunResult, cur: dict[str, list[RunResult]]) -> None:
    """Release, a backward recursion over a group series feeding forward ones, built from two
    class runs, run through run() and written keyed on group_id with outputs.kind
    "aggregation"; every value against the closed form from independently summed series."""
    runs = cur["cur"]
    assert stage.ids.tolist() == ["G0", "G1"]  # the loader's order, as the series'
    assert (stage.H, stage.n_scenarios) == (12, 1)
    outputs = stage.manifest["outputs"]
    assert (outputs["kind"], outputs["id_field"]) == ("aggregation", "group_id")
    monthly = pq.read_table(stage.run_dir / "outputs" / "monthly.parquet")
    assert monthly.column_names[:3] == ["group_id", "scenario", "t"]
    assert monthly.column("group_id").to_pylist()[:13] == ["G0"] * 13

    series = _series(stage)
    assert series["cu_g"].values.shape == (13, 2, 1)
    assert series["p_g"].values.shape == (1, 2, 1)
    j = {"G0": 0.002, "G1": 0.004}
    for gi, g in enumerate(("G0", "G1")):
        cu_g = _summed(runs, "cu", MEMBERS[g])
        fcf_g = _summed(runs, "fcf", MEMBERS[g])
        p_g = _summed(runs, "premium_total", MEMBERS[g])[0]
        assert series["cu_g"].values[:, gi, 0].tolist() == cu_g
        assert series["fcf_g"].values[:, gi, 0].tolist() == fcf_g
        assert series["p_g"].values[0, gi, 0] == p_g
        expected = closed_form_release(cu_g, fcf_g, p_g, j[g])
        for name, values in expected.items():
            assert stage.lane(name, g).tolist() == values, (g, name)
    assert stage.lane("csm", "G0")[0] > 0  # a profitable group: the CSM runs off to zero
    assert stage.lane("csm", "G0")[-1] == 0.0
    assert all(r.passed for r in stage.invariants)  # H3: cu_next ends at its terminal

    assert stage.compiled is not None
    source = stage.compiled.source
    assert "Y_cu_g = inp['series.cu_g']" in source  # the series path, never Z_/u0
    assert "Y_cu_g[t+1]" in source  # a backward recursion
    assert "Y_cu_g[t]" in source  # a forward one
    assert "Y_cu_g[1:H+1]" in source  # a whole-mode formula
    assert "_series_at(Y_fcf_g, 0, 0, 'fcf_g', 'csm_0'" in source  # a scalar one
    assert "Z_" not in source


def test_the_stage_run_directory_keeps_its_series(
    stage: RunResult, cur: dict[str, list[RunResult]]
) -> None:
    """D21 for the series: a copy of each in the design's long form, hashed in the manifest
    with its provenance; the run id covers the series' hashes, and load_run round-trips."""
    entry = stage.manifest["inputs"]["series"]["cu_g"]
    copy = stage.run_dir / entry["copy"]
    assert entry["sha256"] == hashlib.sha256(copy.read_bytes()).hexdigest()
    assert (entry["basis"], entry["variables"], entry["scenarios"]) == ("cur", ["cu"], "mean")
    assert (entry["unit"], entry["start"], entry["time"], entry["months"]) == (
        "CoverageUnits",
        1,
        True,
        [0, 12],
    )
    assert [r["run_id"] for r in entry["source_runs"]] == [r.run_id for r in cur["cur"]]
    assert [r["manifest_sha256"] for r in entry["source_runs"]] == [
        hashlib.sha256((r.run_dir / "manifest.json").read_bytes()).hexdigest() for r in cur["cur"]
    ]
    assert (
        entry["valuation_date"] == VAL.isoformat() == stage.manifest["settings"]["valuation_date"]
    )
    assert entry["members"] == [[g, sorted(m)] for g, m in sorted(MEMBERS.items())]
    assert len(entry["values_sha256"]) == 64
    assert entry["values_sha256"] != entry["sha256"]  # what the run id covers, not the file
    table = pq.read_table(copy)
    assert table.column_names == ["group_id", "scenario", "t", "value"]
    values = np.asarray(table.column("value")).reshape(2, 13, 1).transpose(1, 0, 2)
    assert np.array_equal(values, _series(stage)["cu_g"].values)
    assert table.schema.metadata[b"halley.unit"] == b"CoverageUnits"

    loaded = load_run(stage.run_dir)
    assert loaded.ids.tolist() == ["G0", "G1"]
    assert loaded.manifest["outputs"]["kind"] == "aggregation"
    for name in ("csm", "lc", "cu_next", "csm_0"):
        assert np.array_equal(loaded.var(name), stage.var(name)), name


def test_reruns_are_bit_identical_and_batches_agree(
    stage: RunResult, cur: dict[str, list[RunResult]], tmp_path: Path
) -> None:
    """D19: the same inputs give the same run id and the same bits; D5: a batch of one group
    agrees within 1e-12 relative (here exactly, nothing in the stage depends on the batch)."""
    series = group_series(Release, cur, MEMBERS)
    again = run_stage(tmp_path, Release, series, name="again")
    assert again.run_id == stage.run_id
    one = run_stage(tmp_path, Release, series, name="one", batch_size=1)
    for name in stage.variables:
        assert np.array_equal(again.arrays[name], stage.arrays[name]), name
        np.testing.assert_allclose(one.var(name), stage.var(name), rtol=1e-12, atol=0)
    other = run_stage(
        tmp_path, Release, series, name="other", groups=stage_groups(j=(0.005, 0.002))
    )
    assert other.run_id != stage.run_id
    p_g = series["p_g"]
    bumped = {**series, "p_g": dataclasses.replace(p_g, values=p_g.values * 1.01)}
    assert run_stage(tmp_path, Release, bumped, name="bumped").run_id != stage.run_id


def test_the_run_id_covers_the_series_values_not_the_bytes_of_their_copies(
    stage: RunResult,
    cur: dict[str, list[RunResult]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D21: the run id covers each series by a hash of what it holds (``values_sha256``), not of
    its parquet copy, whose bytes name the writer's version and follow pyarrow's defaults. A
    copy laid out otherwise (another pyarrow, a replay that rewrites it) keeps the run id; the
    copy's own sha256 stays in the manifest for its integrity; one bit of the values (-0.0 for
    0.0 at a month the stage never reads) moves the id."""
    series = group_series(Release, cur, MEMBERS)

    def gzip_copy(s: LoadedSeries, id_field: str, dest: Path) -> str:
        dest.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(run_module._series_table(s, id_field), dest, compression="gzip")
        return hashlib.sha256(dest.read_bytes()).hexdigest()

    with monkeypatch.context() as patch:
        patch.setattr(run_module, "_write_series", gzip_copy)
        other = run_stage(tmp_path, Release, series, name="gzip")
    assert other.run_id == stage.run_id
    for name, entry in other.manifest["inputs"]["series"].items():
        mine = stage.manifest["inputs"]["series"][name]
        assert entry["sha256"] != mine["sha256"], name
        assert (
            entry["sha256"]
            == hashlib.sha256((other.run_dir / entry["copy"]).read_bytes()).hexdigest()
        )
        assert entry["values_sha256"] == mine["values_sha256"], name
    negative = series["cu_g"].values.copy()
    assert negative[0].tolist() == [[0.0], [0.0]]  # a sum of month quantities at t = 0
    negative[0] = -0.0
    signed = {**series, "cu_g": dataclasses.replace(series["cu_g"], values=negative)}
    moved = run_stage(tmp_path, Release, signed, name="signed")
    assert moved.run_id != stage.run_id
    for name in stage.variables:
        assert np.array_equal(moved.arrays[name], stage.arrays[name]), name


def test_the_series_do_not_depend_on_how_the_basis_is_split_or_batched(
    cur: dict[str, list[RunResult]], tmp_path: Path
) -> None:
    """A group total runs over the members in ascending policy id across the basis's runs, so
    the same policies in one run instead of two, in the runs given the other way round, split
    across the runs differently, or projected in batches of one, give the same series (D19: one
    fixed order; D5). Run order is not ascending id here: with the B run first, G0's members
    come B01, A00, A02, and that order gives other bits for cu and fcf, so a sum in run order
    fails this test."""
    a, b = cur["cur"]
    two = group_series(Release, cur, MEMBERS)
    both = pa.concat_tables([covered_points("A"), covered_points("B", scale=1.5)])
    one = {"cur": [run_covered(tmp_path, "both", both)]}
    batched = release_bases(tmp_path / "batched", batch_size=1)
    ids = both.column("policy_id").to_pylist()
    x = run_covered(tmp_path, "x", both.take([ids.index(p) for p in ("B01", "A00", "B02")]))
    y = run_covered(tmp_path, "y", both.take([ids.index(p) for p in ("A02", "B00", "A01")]))
    for other in (
        group_series(Release, {"cur": [b, a]}, MEMBERS),
        group_series(Release, {"cur": [x, y]}, MEMBERS),
        group_series(Release, {"cur": [y, x]}, MEMBERS),
        group_series(Release, one, MEMBERS),
        group_series(Release, batched, MEMBERS),
    ):
        for name, s in two.items():
            assert np.array_equal(other[name].values, s.values), name
    for gi, g in enumerate(("G0", "G1")):  # the reference, sorted independently
        assert two["cu_g"].values[:, gi, 0].tolist() == _summed([b, a], "cu", MEMBERS[g])
        assert two["fcf_g"].values[:, gi, 0].tolist() == _summed([b, a], "fcf", MEMBERS[g])
    in_run_order = functools.reduce(
        operator.add, [b.var("cu")[:, 1, 0], *a.var("cu")[:, [0, 2], 0].T]
    )
    assert not np.array_equal(in_run_order, two["cu_g"].values[:, 0, 0])  # B01, A00, A02


def test_recompute_and_explain_agree_with_the_generated_code(
    stage: RunResult, cur: dict[str, list[RunResult]]
) -> None:
    """The interpreter reads the same series and evaluates every variable of every group at
    every month to the same bits as the generated code."""
    graph = stage.graph
    assert graph is not None
    for name, v in graph.variables.items():
        for g in stage.ids:
            lane = stage.lane(name, g)
            for t in range(stage.H + 1 if v.has_time else 1):
                assert recompute(stage, g, name, t) == lane[t], (name, g, t)
    e = explain(stage, "G1", "cu_next", 3)
    inputs = dict(e.inputs)
    assert inputs["series.cu_g(t+1)"] == _series(stage)["cu_g"].values[4, 1, 0]
    assert inputs["cu_next(t+1)"] == stage.lane("cu_next", "G1")[4]
    assert e.value == stage.lane("cu_next", "G1")[3]
    e0 = explain(stage, "G0", "csm_0")
    assert dict(e0.inputs)["series.fcf_g.at(0)"] == _series(stage)["fcf_g"].values[0, 0, 0]
    assert dict(e0.inputs)["series.p_g()"] == _series(stage)["p_g"].values[0, 0, 0]
    text = str(e0)  # a stage's lanes are groups, and the series read as the model writes it
    assert text.startswith(f"csm_0 for group G0 = {e0.value!r} [Money]"), text
    assert f"series.p_g() = {float(_series(stage)['p_g'].values[0, 0, 0])!r}" in text
    projection = explain(cur["cur"][0], "A00", "cu", 2)
    assert str(projection).startswith("cu at t = 2 for policy A00")


# ------------------------------------------------------------------------------ fixed months

KEYED_SCHEMA = Schema(
    fields=(*STAGE_SCHEMA.fields, int_field("k", ge=0, le=30)),
    id_field="group_id",
)


class Keyed(Release):
    """Release with a month each group names, read in the three contexts of the generated code:
    a scalar formula, a whole-mode one and a forward recursion stepped month by month, as the
    in-force close reads ``scen.p_rec.at(mp.i_acc)``."""

    model_point = KEYED_SCHEMA

    @var
    def fcf_k(self) -> Money:
        return self.series.fcf_g.at(self.group.k)

    @var
    def cu_k(self, t: Int) -> CoverageUnits:
        return self.series.cu_g(t) - self.series.cu_g.at(self.group.k)

    @var(init=0.0)
    def cu_acc(self, t: Int) -> CoverageUnits:
        return self.cu_acc(t - 1) + self.series.cu_g.at(self.group.k)


def _keyed_groups(k: tuple[int, int]) -> Any:
    """Keyed's groups G1 and G0, listed out of order as stage_groups lists them."""
    return pa.table(
        {
            "group_id": pa.array(["G1", "G0"], pa.string()),
            "j": pa.array([0.004, 0.002], pa.float64()),
            "k": pa.array(list(k), pa.int32()),
        }
    )


def test_a_series_read_at_a_month_each_group_names(
    cur: dict[str, list[RunResult]], tmp_path: Path
) -> None:
    """``series.x.at(mp.k)`` takes each group's own month in every context, the interpreter
    agrees bit for bit, and a month outside the series' domain is a RunError at the model line
    naming the group."""
    series = group_series(Keyed, cur, MEMBERS)
    res = run_stage(tmp_path, Keyed, series, groups=_keyed_groups((5, 3)))
    assert res.compiled is not None
    source = res.compiled.source
    assert "_series_at(Y_fcf_g, M_k, 0, 'fcf_g', 'fcf_k'" in source  # a scalar formula
    assert "_series_at(Y_cu_g, M_k, 1, 'cu_g', 'cu_k'" in source  # whole mode
    assert "_series_at(Y_cu_g, m_k, 1, 'cu_g', 'cu_acc'" in source  # step mode
    cu_g, fcf_g = series["cu_g"].values, series["fcf_g"].values
    for gi, (g, k) in enumerate((("G0", 3), ("G1", 5))):
        assert res.lane("fcf_k", g)[0] == fcf_g[k, gi, 0]
        expected = [0.0] + [cu_g[t, gi, 0] - cu_g[k, gi, 0] for t in range(1, 13)]
        assert res.lane("cu_k", g).tolist() == expected
        acc = [0.0]
        for _ in range(12):
            acc.append(acc[-1] + cu_g[k, gi, 0])
        assert res.lane("cu_acc", g).tolist() == acc
    graph = res.graph
    assert graph is not None
    for name, v in graph.variables.items():
        for g in res.ids:
            lane = res.lane(name, g)
            for t in range(res.H + 1 if v.has_time else 1):
                assert recompute(res, g, name, t) == lane[t], (name, g, t)
    assert dict(explain(res, "G1", "cu_acc", 2).inputs)["series.cu_g.at(mp.k)"] == cu_g[5, 1, 0]

    with pytest.raises(
        RunError,
        match=r"series.cu_g.at\(...\) reads a month outside the months of series cu_g, 1..12; "
        r"groups G0\n",
    ) as info:  # month 0 of a sum of month quantities
        run_stage(tmp_path, Keyed, series, name="early", groups=_keyed_groups((5, 0)))
    loc = info.value.loc
    assert loc is not None
    assert loc.function.endswith(("Keyed.cu_k", "Keyed.cu_acc")), loc
    assert Path(loc.file) == Path(__file__)
    assert "polic" not in info.value.message  # a stage's lanes are groups


def test_a_fixed_month_must_be_a_month_of_the_series(
    cur: dict[str, list[RunResult]], tmp_path: Path
) -> None:
    """A constant ``series.x.at(k)`` beyond the months a series covers is a LoadError naming its
    basis when the run loads. A series read only at fixed months keeps every month of its basis
    whatever the stage's horizon; one also read relatively is cut to the horizon first."""

    class Late(Release):
        @var
        def late(self) -> Money:
            return self.series.fcf_g.at(20)

    with pytest.raises(
        LoadError,
        match=r"series fcf_g: the stage reads series.fcf_g.at\(20\), but basis cur covers months "
        r"0..12; a series read at month k comes from a basis whose horizon is at least k",
    ):
        run_stage(tmp_path, Late, group_series(Late, cur, MEMBERS), name="late")

    class Nine(Release):
        @var
        def nine(self) -> Money:
            return self.series.fcf_g.at(9)

    nine = group_series(Nine, cur, MEMBERS)
    assert (
        run_stage(tmp_path, Nine, nine, name="nine").lane("nine", "G1")[0]
        == (nine["fcf_g"].values[9, 1, 0])
    )
    with pytest.raises(
        LoadError,
        match=r"series fcf_g: the stage reads series.fcf_g.at\(9\), but it is also read "
        r"relatively, so it is cut to the stage's horizon, 0..6",
    ):
        run_stage(tmp_path, Nine, nine, name="nine-short", horizon=6)

    class Ahead(Release):
        series = (*Release.series, GroupSum("fcf_at", "cur", "fcf", "Money", start=0))

        @var
        def at_12(self) -> Money:
            return self.series.fcf_at.at(12)

    ahead = group_series(Ahead, cur, MEMBERS)
    res = run_stage(tmp_path, Ahead, ahead, name="ahead", horizon=6)
    assert _series(res)["fcf_at"].values.shape == (13, 2, 1)  # every month of the basis
    assert _series(res)["fcf_g"].values.shape == (7, 2, 1)  # read relatively: 0..H
    for gi, g in enumerate(("G0", "G1")):
        assert res.lane("at_12", g)[0] == ahead["fcf_at"].values[12, gi, 0]
        assert recompute(res, g, "at_12", 0) == res.lane("at_12", g)[0]


# --------------------------------------------------------------------- scenarios and J = 1


class TwoBases(Release):
    """Release beside an ``old`` basis read relatively, as the in-force close reads its
    ``fcf_old``."""

    bases = (Basis("cur"), Basis("old"))
    series = (*Release.series, GroupSum("cu_old", "old", "cu", "CoverageUnits"))

    @var
    def cu_moved(self, t: Int) -> CoverageUnits:
        return self.series.cu_old(t) - self.series.cu_g(t)


class ReleaseEach(Release):
    """Release over every scenario of its basis: no mean, so the stage runs at S = J."""

    series = (
        dataclasses.replace(CU_G, scenarios="each"),
        dataclasses.replace(FCF_G, scenarios="each"),
        dataclasses.replace(P_G, scenarios="each"),
    )


def test_the_mean_over_a_two_scenario_basis(
    cur: dict[str, list[RunResult]], cur_j2: dict[str, list[RunResult]], tmp_path: Path
) -> None:
    """``scenarios="mean"`` reduces as ``cumsum_s((1.0 / J) * x)[-1]`` in ascending s, giving
    one scenario; with J = 1 it is ``x * 1.0``, exact. ``"each"`` keeps the scenarios, and its
    scenario 0 is the deterministic stage bit for bit (D4)."""
    mean = group_series(Release, cur_j2, MEMBERS)
    each = group_series(ReleaseEach, cur_j2, MEMBERS)
    for name in ("cu_g", "fcf_g", "p_g"):
        x = each[name].values
        assert x.shape[2] == 2
        assert mean[name].values.shape[2] == 1
        assert np.array_equal(mean[name].values[:, :, 0], (0.5 * x[:, :, 0]) + (0.5 * x[:, :, 1]))
    assert not np.array_equal(each["fcf_g"].values[:, :, 0], each["fcf_g"].values[:, :, 1])
    deterministic = group_series(Release, cur, MEMBERS)
    for name in ("cu_g", "fcf_g", "p_g"):
        assert np.array_equal(deterministic[name].values, each[name].values[:, :, :1])

    on_mean = run_stage(tmp_path, Release, mean, name="mean")
    assert on_mean.n_scenarios == 1
    assert on_mean.manifest["inputs"]["series"]["fcf_g"]["basis_scenarios"] == 2
    expected = closed_form_release(
        mean["cu_g"].values[:, 0, 0].tolist(),
        mean["fcf_g"].values[:, 0, 0].tolist(),
        float(mean["p_g"].values[0, 0, 0]),
        0.002,
    )
    assert on_mean.lane("csm", "G0").tolist() == expected["csm"]

    per_scenario = run_stage(tmp_path, ReleaseEach, each, name="each")
    assert per_scenario.n_scenarios == 2
    alone = run_stage(tmp_path, Release, deterministic, name="alone")
    for name in alone.variables:
        assert np.array_equal(per_scenario.var(name)[:, :, 0], alone.var(name)[:, :, 0]), name
    assert on_mean.lane("lc", "G0").tolist() == [0.0] * 13  # the mean group is profitable
    for g in ("G0", "G1"):  # the second scenario is onerous: a loss component and no CSM
        assert per_scenario.lane("csm", g, 0)[0] > 0
        assert per_scenario.lane("lc", g, 0)[0] == 0.0
        assert per_scenario.lane("lc", g, 1)[0] > 0
        assert per_scenario.lane("csm", g, 1)[0] == 0.0


def test_the_mean_over_three_scenarios_is_a_sequential_sum_of_thirds(tmp_path: Path) -> None:
    """With J = 3 the mean is ``((1/3) x0 + (1/3) x1) + (1/3) x2`` bit for bit (para 33(a):
    ``cumsum_s((1.0 / J) * x)[-1]`` in ascending s). At J = 2 every order gives the same bits
    (``0.5 * x`` is exact and there is one addition); here a sum then a division, or NumPy's
    mean, gives other bits, so a mean reduced in another order fails this test."""
    cur_j3 = release_bases(tmp_path, shocks=(1.0, 1.5, 0.7))
    mean = group_series(Release, cur_j3, MEMBERS)
    each = group_series(ReleaseEach, cur_j3, MEMBERS)
    third = 1.0 / 3
    for name in ("cu_g", "fcf_g", "p_g"):
        x = each[name].values
        assert x.shape[2] == 3
        thirds = ((third * x[:, :, 0]) + (third * x[:, :, 1])) + (third * x[:, :, 2])
        assert np.array_equal(mean[name].values[:, :, 0], thirds), name
        assert mean[name].provenance["basis_scenarios"] == 3
    x = each["fcf_g"].values
    got = mean["fcf_g"].values[:, :, 0]
    assert not np.array_equal(got, ((x[:, :, 0] + x[:, :, 1]) + x[:, :, 2]) / 3)
    assert not np.array_equal(got, np.mean(x, axis=2))
    assert not np.array_equal(
        got, (third * x[:, :, 0]) + ((third * x[:, :, 1]) + (third * x[:, :, 2]))
    )


class LockedIn(Release):
    """Release with its own scenario input, a locked-in discount curve for the units."""

    scenario_inputs = (ScenarioInput("p_l", "Factor"),)

    @var(domain="state")
    def p(self, t: Int) -> Factor:
        return self.scen.p_l(t)

    @var
    def d_l(self, t: Int) -> Factor:
        return self.p(t) / self.p(t - 1)

    @var(terminal=0)
    def cu_next(self, t: Int) -> CoverageUnits:
        return self.series.cu_g(t + 1) + self.d_l(t + 1) * self.cu_next(t + 1)


def test_a_stage_that_reads_a_mean_runs_at_one_scenario(
    cur: dict[str, list[RunResult]], cur_j2: dict[str, list[RunResult]], tmp_path: Path
) -> None:
    """The J = 1 check, in one place (require_deterministic): beside a mean series, a
    two-scenario scenario input of the stage, or a per-scenario series of a J = 2 basis, is a
    LoadError naming the check."""
    p_l = scenario_file("p_l", 0, 12, 1, lambda s, u: 0.997**u)
    res = run_stage(
        tmp_path, LockedIn, group_series(LockedIn, cur, MEMBERS), scenarios={"p_l": p_l}
    )
    assert res.n_scenarios == 1
    p_l2 = scenario_file("p_l", 0, 12, 2, lambda s, u: (0.997 - 0.001 * s) ** u)
    with pytest.raises(
        LoadError,
        match=r"the stage reads the mean series cu_g, so it runs at S = 1; scenario input p_l "
        r"has J = 2 scenarios.*\(require_deterministic\)",
    ):
        run_stage(
            tmp_path,
            LockedIn,
            group_series(LockedIn, cur, MEMBERS),
            name="two",
            scenarios={"p_l": p_l2},
        )

    class Mixed(Release):
        series = (CU_G, dataclasses.replace(FCF_G, scenarios="each"), P_G)

    with pytest.raises(
        LoadError, match=r"series fcf_g has J = 2 scenarios.*\(require_deterministic\)"
    ):
        run_stage(tmp_path, Mixed, group_series(Mixed, cur_j2, MEMBERS), name="mixed")
    # a deterministic basis passes the same check
    run_stage(tmp_path, Mixed, group_series(Mixed, cur, MEMBERS), name="mixed-j1")


# ------------------------------------------------------------------------ the series builder


def test_several_variables_add_their_group_totals_left_to_right(
    cur_j2: dict[str, list[RunResult]],
) -> None:
    """``("fcf", "cf")`` is the group total of fcf plus that of cf, per scenario, and only then
    the mean; its domain starts where the latest of the two does. Three totals are added left
    to right, ``(fcf_g + cf_g) + fee_g``, which gives other bits than ``fcf_g + (cf_g + fee_g)``
    on these values, so another grouping fails this test."""

    class Parts(Aggregation):
        model_point = STAGE_SCHEMA
        bases = (Basis("cur"),)
        series = (
            GroupSum("both", "cur", ("fcf", "cf"), "Money"),
            GroupSum("both_each", "cur", ("fcf", "cf"), "Money", "each"),
            GroupSum("three_each", "cur", ("fcf", "cf", "fee"), "Money", "each"),
            GroupSum("fcf_each", "cur", "fcf", "Money", "each", start=0),
            GroupSum("cf_each", "cur", "cf", "Money", "each"),
            GroupSum("fee_each", "cur", "fee", "Money", "each"),
        )

        @var
        def total(self, t: Int) -> Money:
            return self.series.both(t)

    s = group_series(Parts, cur_j2, MEMBERS)
    f, c, e = s["fcf_each"].values, s["cf_each"].values, s["fee_each"].values
    assert np.array_equal(s["both_each"].values, f + c)
    added = f + c
    assert np.array_equal(
        s["both"].values[:, :, 0], (0.5 * added[:, :, 0]) + (0.5 * added[:, :, 1])
    )
    assert np.array_equal(s["three_each"].values, (f + c) + e)
    assert not np.array_equal(s["three_each"].values, f + (c + e))
    assert (s["both"].start, s["fcf_each"].start, s["three_each"].start) == (1, 0, 1)
    assert s["both"].provenance["variables"] == ["fcf", "cf"]
    assert s["three_each"].provenance["variables"] == ["fcf", "cf", "fee"]


def test_the_series_builder_checks_runs_members_and_units(
    cur: dict[str, list[RunResult]], tmp_path: Path
) -> None:
    runs = cur["cur"]

    class NotStored(Release):
        series = (CU_G, FCF_G, P_G, GroupSum("bel_g", "cur", "bel", "Money"))

    with pytest.raises(
        LoadError, match=f"basis cur run {runs[0].run_id} did not store bel; add it to outputs"
    ):
        group_series(NotStored, cur, MEMBERS)

    class WrongUnit(Release):
        series = (GroupSum("cu_g", "cur", "cf", "CoverageUnits"), FCF_G, P_G)

    with pytest.raises(
        LoadError,
        match=rf"series cu_g is declared CoverageUnits but basis cur run {runs[0].run_id} "
        r"records cf as Money \(D11 across runs\)",
    ):
        group_series(WrongUnit, cur, MEMBERS)

    class NoTime(Release):
        series = (CU_G, FCF_G, GroupSum("p_g", "cur", "cf", "Money", time=False))

    with pytest.raises(LoadError, match=r"declared without a time dimension but cf .* has one"):
        group_series(NoTime, cur, MEMBERS)

    with pytest.raises(LoadError, match="the stage reads basis cur, which was not given"):
        group_series(Release, {}, MEMBERS)
    with pytest.raises(LoadError, match="basis 'old' was given but Release declares no such"):
        group_series(Release, {**cur, "old": runs}, MEMBERS)
    with pytest.raises(LoadError, match="basis cur: policy A00 is in run"):
        group_series(Release, {"cur": [runs[0], runs[0]]}, MEMBERS)

    short = {"G0": MEMBERS["G0"], "G1": MEMBERS["G1"][:2]}
    with pytest.raises(LoadError, match=f"basis cur run {runs[1].run_id}: policy B02 is in no"):
        group_series(Release, cur, short)
    twice = {"G0": MEMBERS["G0"], "G1": [*MEMBERS["G1"], "A00"]}
    with pytest.raises(LoadError, match="policy A00 is a member of group G0 and of group G1"):
        group_series(Release, cur, twice)
    extra = {"G0": [*MEMBERS["G0"], "C00"], "G1": MEMBERS["G1"]}
    with pytest.raises(LoadError, match="group G0: member C00 is in no run of the stage's bases"):
        group_series(Release, cur, extra)

    # a group whose sums would be zero by omission: no members, or none in one basis's runs
    with pytest.raises(LoadError, match="group G2 has no members; a group holds at least one"):
        group_series(Release, cur, {**MEMBERS, "G2": []})
    by_class = {"G0": ["A00", "A01", "A02"], "G1": ["B00", "B01", "B02"]}
    group_series(TwoBases, {**cur, "old": runs}, by_class)  # every group in both bases
    with pytest.raises(
        LoadError,
        match="group G1 has no member in the runs of basis old, which a series of the stage sums",
    ):
        group_series(TwoBases, {**cur, "old": [runs[0]]}, by_class)  # old holds the A run only


def test_the_loader_checks_each_series_against_the_stage(
    cur: dict[str, list[RunResult]], tmp_path: Path
) -> None:
    """At run time every declared series is given and no other, for the stage's groups, with a
    domain that covers what the stage reads; a series read relatively is cut to the stage's
    horizon, which its basis must reach."""
    series = group_series(Release, cur, MEMBERS)

    class FromZero(Release):
        series = (dataclasses.replace(CU_G, start=0), FCF_G, P_G)

    with pytest.raises(
        LoadError, match="series cu_g: the stage reads it from t = 0 but it is defined from t = 1"
    ):
        run_stage(tmp_path, FromZero, group_series(FromZero, cur, MEMBERS), name="zero")
    missing = {k: v for k, v in series.items() if k != "p_g"}
    with pytest.raises(LoadError, match="the stage declares series 'p_g', which was not given"):
        run_stage(tmp_path, Release, missing, name="missing")
    with pytest.raises(LoadError, match="series 'q_g' was given but the stage declares no such"):
        run_stage(tmp_path, Release, {**series, "q_g": series["p_g"]}, name="extra")

    # D21: a series says where it came from, and holds float64 values the stage computes on
    cu_g, p_g = series["cu_g"], series["p_g"]
    for provenance, missing in (
        ({}, "basis, variables, basis_scenarios, valuation_date, source_runs, members"),
        ({k: v for k, v in cu_g.provenance.items() if k != "members"}, "members"),
    ):
        bare = {**series, "cu_g": dataclasses.replace(cu_g, provenance=provenance)}
        with pytest.raises(
            LoadError,
            match=rf"series cu_g: its provenance records no {missing}; a stage runs on series "
            r"that say where they came from \(halley.aggregation.group_series, D21\)",
        ):
            run_stage(tmp_path, Release, bare, name="bare")
    for name, values, kind in (
        ("cu_g", cu_g.values.astype(np.float32), "float32"),
        ("p_g", np.array([[[3], [4]]], dtype=np.int64), "int64"),
    ):
        typed = {**series, name: dataclasses.replace(series[name], values=values)}
        with pytest.raises(LoadError, match=f"series {name}: its values are {kind}, not float64"):
            run_stage(tmp_path, Release, typed, name="typed")
    swapped = {**p_g.provenance, "members": list(reversed(p_g.provenance["members"]))}
    other = {**series, "p_g": dataclasses.replace(p_g, provenance=swapped)}
    with pytest.raises(
        LoadError, match="series p_g: its provenance records the members of groups G1, G0, not of"
    ):
        run_stage(tmp_path, Release, other, name="members")
    with pytest.raises(LoadError, match="covers groups G0, G1 but the model points are G0, G1, G2"):
        run_stage(
            tmp_path,
            Release,
            series,
            name="groups",
            groups=stage_groups(("G0", "G1", "G2"), (0.002, 0.004, 0.001)),
        )
    with pytest.raises(
        LoadError,
        match=r"series cu_g: basis cur covers months 0\.\.12 but the stage runs to H = 24; a "
        r"series read relatively comes from a basis whose horizon is at least the stage's",
    ):
        run_stage(tmp_path, Release, series, name="long", horizon=24)
    shorter = run_stage(tmp_path, Release, series, name="short", horizon=6)
    assert shorter.H == 6
    assert _series(shorter)["cu_g"].values.shape == (7, 2, 1)
    assert np.array_equal(_series(shorter)["cu_g"].values, series["cu_g"].values[:7])
    with pytest.raises(LoadError, match="is a projection; group series are read by a stage"):
        run(
            build(Growth),
            model_points=simple_model_points(),
            series=series,
            settings=RunSettings(valuation_date=VAL),
            run_dir=tmp_path / "projection",
        )


def test_a_run_records_what_it_stored_for_the_stage(cur: dict[str, list[RunResult]]) -> None:
    """RunResult.variable_info reads the manifest, so it works for a run of this session and a
    loaded one alike."""
    a = cur["cur"][0]
    assert a.variable_info("cu") == VariableInfo(True, "CoverageUnits", 1, None)
    assert a.variable_info("fcf") == VariableInfo(True, "Money", 0, None)
    assert a.variable_info("x") == VariableInfo(True, "Count", 0, "in_force")
    assert a.variable_info("premium_total") == VariableInfo(False, "Money", 0, None)
    assert load_run(a.run_dir).variable_info("cu") == a.variable_info("cu")
    with pytest.raises(LoadError, match=f"run {a.run_id} did not store bel; add it to outputs"):
        a.variable_info("bel")
    old = dict(a.manifest)
    old["outputs"] = {**a.manifest["outputs"], "variables": {"cu": {"time": True}}}
    legacy = dataclasses.replace(a, manifest=old)
    with pytest.raises(LoadError, match="records no unit or domain for cu"):
        legacy.variable_info("cu")
    assert json.loads(json.dumps(a.manifest["outputs"]["variables"]["cu"]))["start"] == 1


# ------------------------------------------------------------- the valuation date and horizon


def test_the_stage_values_at_its_bases_valuation_date(tmp_path: Path) -> None:
    """Design section 9: the stage's valuation date equals every basis's. Basis.check returns
    the date its runs share, the series record it, and a stage valued at another date is a
    LoadError naming the series and its basis (D11: a time-reference mistake)."""
    june = dt.date(2027, 6, 1)
    later = release_bases(tmp_path / "later", valuation_date=june)
    assert Basis("cur").check(later["cur"]) == june
    series = group_series(Release, later, MEMBERS)
    assert {s.provenance["valuation_date"] for s in series.values()} == {"2027-06-01"}
    with pytest.raises(
        LoadError,
        match=r"series cu_g: basis cur values at 2027-06-01 but the stage values at 2027-01-01; "
        r"a stage and its bases share one valuation date",
    ):
        run_stage(tmp_path, Release, series)  # the stage at VAL, 2027-01-01
    res = run_stage(tmp_path, Release, series, name="june", valuation_date=june)
    assert res.manifest["inputs"]["series"]["cu_g"]["valuation_date"] == "2027-06-01"


def _old_basis(tmp_path: Path, horizon: int) -> dict[str, list[RunResult]]:
    """The cur basis's policies projected to another horizon."""
    runs = [
        run_covered(tmp_path, f"old-{p}-{horizon}", covered_points(p, scale=k), horizon=horizon)
        for p, k in (("A", 1.0), ("B", 1.5))
    ]
    return {"old": runs}


def test_the_stage_runs_to_its_cur_basis_horizon(
    cur: dict[str, list[RunResult]], tmp_path: Path
) -> None:
    """Design section 9, "Horizon": the stage's H is its cur basis's, or RunSettings.horizon,
    which may not exceed it; a series read relatively comes from a basis that reaches H, or the
    LoadError names that basis. A shorter basis beside cur does not cut the stage short, and a
    stage without a cur basis whose series disagree asks for RunSettings.horizon."""
    old = _old_basis(tmp_path, 6)
    series = group_series(TwoBases, {**cur, **old}, MEMBERS)
    with pytest.raises(
        LoadError,
        match=r"series cu_old: basis old covers months 0\.\.6 but the stage runs to H = 12; a "
        r"series read relatively comes from a basis whose horizon is at least the stage's",
    ):
        run_stage(tmp_path, TwoBases, series, name="two")
    assert run_stage(tmp_path, TwoBases, series, name="two-6", horizon=6).H == 6
    longer = group_series(TwoBases, {**cur, **_old_basis(tmp_path, 24)}, MEMBERS)
    assert run_stage(tmp_path, TwoBases, longer, name="two-24").H == 12  # cur's, not old's

    class AtOnly(Aggregation):  # the cur basis read at a fixed month only
        model_point = STAGE_SCHEMA
        bases = (Basis("cur"),)
        series = (GroupSum("fcf_g", "cur", "fcf", "Money", start=0),)

        @var
        def first(self) -> Money:
            return self.series.fcf_g.at(0)

    at_only = group_series(AtOnly, cur, MEMBERS)
    assert run_stage(tmp_path, AtOnly, at_only, name="at-only").H == 12
    with pytest.raises(
        LoadError,
        match=r"series fcf_g: basis cur covers months 0\.\.12 but the stage runs to H = 24; the "
        r"stage's horizon is at most its cur basis's",
    ):
        run_stage(tmp_path, AtOnly, at_only, name="at-only-24", horizon=24)

    class NoCur(Aggregation):
        model_point = STAGE_SCHEMA
        bases = (Basis("new"), Basis("old"))
        series = (
            GroupSum("cu_new", "new", "cu", "CoverageUnits"),
            GroupSum("cu_old", "old", "cu", "CoverageUnits"),
        )

        @var
        def moved(self, t: Int) -> CoverageUnits:
            return self.series.cu_new(t) - self.series.cu_old(t)

    no_cur = group_series(NoCur, {"new": cur["cur"], **old}, MEMBERS)
    with pytest.raises(
        LoadError,
        match=r"the stage's series disagree on its horizon: series cu_new of basis new covers "
        r"months 0\.\.12; series cu_old of basis old covers months 0\.\.6; a stage takes its "
        r"cur basis's horizon, so without one set RunSettings.horizon",
    ):
        run_stage(tmp_path, NoCur, no_cur, name="no-cur")
    assert run_stage(tmp_path, NoCur, no_cur, name="no-cur-6", horizon=6).H == 6


def test_a_series_no_formula_reads_warns_and_sets_no_scenarios(
    cur_j2: dict[str, list[RunResult]], tmp_path: Path
) -> None:
    """A declared series no formula reads is a BuildWarning naming its declaration, as an
    unused variable is. The run still asks for it (the declarations are the stage's input
    contract), but an unread mean does not make the stage run at one scenario."""

    class EachAndAnUnreadMean(ReleaseEach):
        p_mean = GroupSum("p_mean", "cur", "premium_total", "Money", time=False)  # decl: p_mean
        series = (*ReleaseEach.series, p_mean)

    with pytest.warns(BuildWarning, match="series p_mean is read by no formula; a run of") as seen:
        compiled = build(EachAndAnUnreadMean)
    where = f"{Path(__file__).name}:{_line_of('decl: p_mean')})"
    assert any(where in str(w.message) for w in seen), [str(w.message) for w in seen]
    series = group_series(EachAndAnUnreadMean, cur_j2, MEMBERS)
    settings = RunSettings(valuation_date=VAL)
    groups = stage_groups()
    res = run(
        compiled, model_points=groups, series=series, settings=settings, run_dir=tmp_path / "a"
    )
    assert res.n_scenarios == 2  # the "each" series it reads set S, not the unread mean
    with pytest.raises(LoadError, match="the stage declares series 'p_mean', which was not given"):
        run(
            compiled,
            model_points=groups,
            series={k: v for k, v in series.items() if k != "p_mean"},
            settings=settings,
            run_dir=tmp_path / "b",
        )
