# pyright: basic
"""Product-declared invariants (design section 7): the kinds, tolerances, windows and masks,
the batch view, evaluation per batch merged over every batch, the RunError on a failure, the
listing H1-H3 then the product's, and ``reads`` keeping variables generated."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

from halley import (
    BuildError,
    BuildWarning,
    Int,
    Invariant,
    Money,
    RunError,
    RunSettings,
    build,
    run,
    var,
)
from halley.authoring import BatchView
from models import (
    VAL,
    Growth,
    Indexed,
    Margin,
    margin_table,
    scenario_file,
    simple_model_points,
)


def _money(a: BatchView, lhs: Any, rhs: Any) -> Any:
    return np.maximum(np.maximum(np.abs(lhs), np.abs(rhs)), a.scale("Money"))


class Checked(Growth):
    """Growth with one invariant per view accessor and per kind; every one holds."""

    tolerance_scales: ClassVar[dict[str, str]] = {"Money": "premium", "Count": "pols_if_init"}
    invariants = (
        Invariant("x never negative", lambda a: a.var("x"), 0.0, kind="ge", tol=0),
        Invariant(
            "x never above its start",
            lambda a: a.var("x"),
            lambda a: a.mp.pols_if_init,
            kind="le",
            tol=0,
        ),
        Invariant(
            "cf = x(t-1) premium",
            lambda a: a.var("cf")[1:],
            lambda a: a.var("x")[:-1] * a.mp.premium,
            tol=0,
            months=(1, "H"),
            doc="exactly, the same operations as the generated code",
        ),
        Invariant("r0 = at(r, 0)", lambda a: a.var("r0"), lambda a: a.at("r", 0), tol=0),
        Invariant(
            "cf_at_k = at(cf, k)", lambda a: a.var("cf_at_k"), lambda a: a.at("cf", a.mp.k), tol=0
        ),
        Invariant(
            "total = sum of cf",
            lambda a: a.var("total"),
            lambda a: np.cumsum(a.var("cf")[1:], axis=0)[-1:],
            tol=1e-9,
            scale=_money,
        ),
        Invariant(
            "ratio = r / x where x > 0",
            lambda a: a.var("ratio")[1:],
            lambda a: a.var("r")[1:] / a.var("x")[1:],
            tol=0,
            months=(1, "H"),
            where=lambda a: a.var("x")[1:] > 0,
        ),
        Invariant(
            "masked = 1 / (t - 2) for t > 2",
            lambda a: a.var("masked"),
            lambda a: np.where(a.t > 2, 1.0 / np.where(a.t > 2, a.t - 2, 1), 0.0),
            tol=0,
        ),
        Invariant("H is the horizon", lambda a: float(a.H), lambda a: float(a.run.horizon), tol=0),
        Invariant("m_val is the valuation month", lambda a: a.run.m_val, 12 * 2027, tol=0),
        Invariant(
            "x(H) is small",
            lambda a: a.at("x", a.H),
            lambda a: a.mp.pols_if_init * 0.99**a.H,
            tol=1e-12,
            scale=lambda a, lhs, rhs: a.scale("Count"),
            months=("H", "H"),
        ),
    )


def _run(model: type[Any], tmp_path: Path, name: str = "run", **settings: Any) -> Any:
    return run(
        build(model),
        model_points=simple_model_points(),
        settings=RunSettings(valuation_date=VAL, **settings),
        run_dir=tmp_path / name,
    )


def test_every_kind_and_view_accessor(tmp_path: Path) -> None:
    res = _run(Checked, tmp_path)
    names = [r.name for r in res.invariants]
    assert names[:3] == [
        "H1 in force never negative, never increasing",
        "H2 probabilities in [0, 1]",
        "H3 r reaches its terminal value at t = 12",
    ]
    assert names[3:] == [inv.name for inv in Checked.invariants]
    assert all(r.passed for r in res.invariants)
    by_name = {r.name: r for r in res.invariants}
    assert by_name["cf = x(t-1) premium"].worst == 0.0
    assert by_name["total = sum of cf"].worst == 0.0  # the same fixed-order sum
    assert by_name["x(H) is small"].worst < 1e-12


def test_results_cover_every_batch_merged_in_ascending_order(tmp_path: Path) -> None:
    class Residual(Growth):
        """A passing invariant whose residual is largest for the first policy, so a result that
        kept only the last batch would under-report it."""

        invariants = (
            Invariant(
                "x within a thousandth",
                lambda a: a.var("x"),
                lambda a: a.var("x") + 1e-3 * (4 - a.mp.pols_if_init),
                tol=1.0,
            ),
        )

    whole = _run(Residual, tmp_path, "whole")
    batched = _run(Residual, tmp_path, "batched", batch_size=2)
    for res in (whole, batched):
        r = res.invariants[-1]
        assert r.passed
        assert r.worst == pytest.approx(3e-3)  # policy S00, pols_if_init = 1
        on_disk = json.loads((res.run_dir / "invariants.json").read_text(encoding="utf-8"))
        assert on_disk == [x.as_dict() for x in res.invariants]
        assert [x["name"] for x in on_disk][:3] == [
            "H1 in force never negative, never increasing",
            "H2 probabilities in [0, 1]",
            "H3 r reaches its terminal value at t = 12",
        ]
        assert on_disk[-1]["name"] == "x within a thousandth"


def test_a_failure_names_the_invariant_first_policy_and_month(tmp_path: Path) -> None:
    class Failing(Growth):
        invariants = (
            Invariant(
                "premium below 15",
                lambda a: a.mp.premium,
                15.0,
                kind="le",
                tol=0,
                doc="a deliberate failure for S01 and S02",
            ),
        )

    # two batches: S00, S01 then S02; the first batch fails, at S01
    with pytest.raises(RunError, match="invariant premium below 15 failed: policy S01:") as info:
        _run(Failing, tmp_path, batch_size=2)
    assert "lhs = 20.0, rhs = 15.0" in str(info.value)
    assert "(a deliberate failure for S01 and S02)" in str(info.value)
    assert info.value.loc is not None
    assert Path(info.value.loc.file) == Path(__file__)
    assert not (tmp_path / "run" / "manifest.json").exists()


def test_a_failure_in_time_names_the_month_of_the_window(tmp_path: Path) -> None:
    class LateFail(Growth):
        invariants = (
            Invariant(
                "cf stays above 5",
                lambda a: a.var("cf")[1:],
                5.0,
                kind="ge",
                tol=0,
                months=(1, "H"),
            ),
        )

    # S00: cf(t) = 0.99 ** (t-1) * 10 > 5 for every t <= 12; so S00 passes and nothing fails...
    res = _run(LateFail, tmp_path)
    assert res.invariants[-1].passed

    class EarlyFail(Growth):
        invariants = (
            Invariant(
                "cf below 25",
                lambda a: a.var("cf"),
                25.0,
                kind="le",
                tol=0,
                months=(1, "H"),
            ),
        )

    # S02: cf(1) = 3 * 30 = 90; the full-length array is sliced to the window, so t = 1
    with pytest.raises(RunError, match=r"policy S01, t = 1: lhs = 40\.0, rhs = 25\.0"):
        _run(EarlyFail, tmp_path, "early")


def test_scenario_and_table_accessors(tmp_path: Path) -> None:
    class CheckedIndexed(Indexed):
        invariants = (
            Invariant(
                "df = scen df",
                lambda a: a.var("df"),
                lambda a: a.scen_range("df", 0, a.H),
                tol=0,
            ),
            Invariant(
                "esc = cpi(t-3) / cpi(0)",
                lambda a: a.var("esc")[1:],
                lambda a: a.scen_range("cpi", -2, a.H - 3) / a.scen("cpi", 0),
                tol=0,
                months=(1, "H"),
            ),
        )

    horizon = 12
    df = scenario_file("df", 0, horizon, 2, lambda s, u: (0.99 if s == 0 else 0.98) ** u)
    cpi = scenario_file("cpi", -3, horizon, 2, lambda s, u: 100.0 * (1.002 + 0.001 * s) ** u)
    res = run(
        build(CheckedIndexed),
        model_points=simple_model_points(rem_m=horizon),
        scenarios={"df": df, "cpi": cpi},
        settings=RunSettings(valuation_date=VAL),
        run_dir=tmp_path / "indexed",
    )
    assert [r.passed for r in res.invariants[-2:]] == [True, True]

    class CheckedMargin(Margin):
        invariants = (
            Invariant(
                "csm_0 = (premium_total - fcf(0)) share",
                lambda a: a.var("csm_0"),
                lambda a: (a.var("premium_total") - a.at("fcf", 0)) * a.table("margin", "share"),
                tol=0,
            ),
        )

    res = run(
        build(CheckedMargin),
        model_points=simple_model_points(),
        tables={"margin": margin_table(0.5)},
        settings=RunSettings(valuation_date=VAL),
        run_dir=tmp_path / "margin",
    )
    assert res.invariants[-1].passed


def test_reads_keeps_variables_generated(tmp_path: Path) -> None:
    class Narrow(Growth):
        outputs = ("x", "cf")
        invariants = (
            Invariant(
                "r0 = at(r, 0)",
                lambda a: a.var("r0"),
                lambda a: a.at("r", 0),
                tol=0,
                reads=("r0", "r"),
            ),
        )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(BuildWarning, match="ratio is read by no other variable"):
            build(Narrow)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = _run(Narrow, tmp_path)
    assert {"r", "r0"} <= set(res.arrays)
    assert set(res.manifest["outputs"]["variables"]) == {"x", "cf"}
    assert res.invariants[-1].passed

    class Unread(Growth):
        outputs = ("x", "cf")
        invariants = (
            Invariant("r0 = at(r, 0)", lambda a: a.var("r0"), lambda a: a.at("r", 0), tol=0),
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(RunError, match="r0 was not generated for this run; name it in"):
            _run(Unread, tmp_path, "unread")


def test_declaration_errors(tmp_path: Path) -> None:
    class Unknown(Growth):
        invariants = (Invariant("reads nothing", lambda a: a.var("x"), reads=("nope",)),)

    with pytest.raises(BuildError, match="invariant reads nothing reads 'nope'") as info:
        build(Unknown)
    assert info.value.loc is not None
    assert Path(info.value.loc.file) == Path(__file__)
    with pytest.raises(ValueError, match="kind must be one of"):
        Invariant("bad kind", 1.0, kind="lt")
    with pytest.raises(ValueError, match="tol must be at least 0"):
        Invariant("bad tol", 1.0, tol=-1)

    class Misshapen(Growth):
        invariants = (
            Invariant("sliced without a window", lambda a: a.var("cf")[1:], 0.0, kind="ge"),
        )

    with pytest.raises(RunError, match=r"arrays have 12 months, not 13; declare months="):
        _run(Misshapen, tmp_path, "misshapen")

    class Unscaled(Growth):
        invariants = (Invariant("needs a scale", lambda a: a.var("cf"), 0.0, scale=_money),)

    with pytest.raises(RunError, match="declares no tolerance scale for Money"):
        _run(Unscaled, tmp_path, "unscaled")

    class Reserved(Growth):
        @var
        def invariants(self, t: Int) -> Money:  # type: ignore[override]
            return self.cf(t)

    with pytest.raises(BuildError, match="invariants is a Product class attribute"):
        build(Reserved)

    class Wide(Growth):
        invariants = (
            Invariant("too many axes", lambda a: a.var("x")[None], 0.0, kind="ge", tol=0),
        )

    with pytest.raises(RunError, match="has 4 axes"):
        _run(Wide, tmp_path, "wide")

    class Mixed(Growth):
        invariants = (
            Invariant("k below the count", lambda a: a.mp.k, lambda a: a.var("x") * 10, kind="le"),
        )

    res = _run(Mixed, tmp_path, "mixed")  # k = 2, 3, 4 <= 10 x(t) for every t
    assert res.invariants[-1].passed
    assert res.var("x").dtype == np.float64

    class Nothing(Growth):
        invariants = (Invariant("a count check", lambda a: a.var("x"), 0.0, kind="ge", tol=0),)

    res = _run(Nothing, tmp_path, "nothing")
    assert res.invariants[-1].worst == 0.0


def test_an_invariant_records_where_it_was_declared() -> None:
    a = Invariant("same", 1.0, doc="x")
    b = Invariant("same", 1.0, doc="x")
    assert a == b  # the declaration site is not part of equality
    assert a.loc is not None
    assert b.loc is not None
    assert Path(a.loc.file) == Path(__file__)
    assert b.loc.line == a.loc.line + 1


class _Scaled(Growth):
    tolerance_scales: ClassVar[dict[str, str]] = {"Count": "pols_if_init"}
    invariants = (
        Invariant(
            "x keeps a tenth of its start by H",
            lambda a: a.at("x", a.H),
            lambda a: 0.1 * a.mp.pols_if_init,
            kind="ge",
            tol=0,
        ),
    )


def test_a_scalar_per_policy_result_reports_no_month(tmp_path: Path) -> None:
    res = _run(_Scaled, tmp_path)
    assert res.invariants[-1].passed

    class Over(_Scaled):
        invariants = (
            Invariant(
                "x keeps all of its start by H",
                lambda a: a.at("x", a.H),
                lambda a: a.mp.pols_if_init,
                kind="ge",
                tol=0,
            ),
        )

    with pytest.raises(RunError, match=r"failed: policy S00: lhs = ") as info:
        _run(Over, tmp_path, "over")
    assert ", t = " not in str(info.value)


def test_helpers_may_be_annotated_with_the_batch_view() -> None:
    """``Checked`` annotates nothing; a model may annotate its helpers with ``BatchView``."""

    def helper(a: BatchView) -> Any:
        return a.var("x")

    class Annotated(Growth):
        invariants = (Invariant("x >= 0", helper, 0.0, kind="ge", tol=0),)

    assert Annotated.invariants[0].lhs is helper
    assert isinstance(Annotated.invariants[0].rhs, float)
