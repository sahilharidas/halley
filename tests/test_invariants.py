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
import pyarrow.parquet as pq
import pytest

from halley import (
    BuildError,
    BuildWarning,
    Factor,
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
from halley.runtime.invariants import InvariantResult, merge_results
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


def _declared_here(exc: RunError) -> None:
    """A run-time error of an invariant points at its declaration in this file."""
    assert exc.loc is not None
    assert Path(exc.loc.file) == Path(__file__)


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

    with pytest.raises(RunError, match=r"arrays have 12 months, not 13; declare months=") as info:
        _run(Misshapen, tmp_path, "misshapen")
    _declared_here(info.value)

    class Unscaled(Growth):
        invariants = (Invariant("needs a scale", lambda a: a.var("cf"), 0.0, scale=_money),)

    with pytest.raises(RunError, match="declares no tolerance scale for Money") as info:
        _run(Unscaled, tmp_path, "unscaled")
    _declared_here(info.value)

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

    with pytest.raises(RunError, match="has 4 axes") as info:
        _run(Wide, tmp_path, "wide")
    _declared_here(info.value)

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


def test_an_eq_failure_pins_the_residual_against_the_tolerance(tmp_path: Path) -> None:
    class Offset(Growth):
        invariants = (
            Invariant(
                "x = x + a half",
                lambda a: a.var("x"),
                lambda a: a.var("x") + 0.5,
                tol=1e-9,
                doc="a deliberate offset",
            ),
        )

    with pytest.raises(RunError, match="invariant x = x \\+ a half failed: ") as info:
        _run(Offset, tmp_path)
    assert (
        "policy S00, t = 0: lhs = 1.0, rhs = 1.5; the residual 0.5 exceeds the tolerance 1e-09 "
        "(a deliberate offset)" in str(info.value)
    )


def test_a_callable_scale_near_miss_fails_and_a_nearer_one_passes(tmp_path: Path) -> None:
    def relative(a: BatchView, lhs: Any, rhs: Any) -> Any:
        return np.maximum(np.abs(lhs), np.abs(rhs))

    class Near(_Scaled):
        invariants = (
            Invariant(
                "x within a billionth of x",
                lambda a: a.var("x"),
                lambda a: a.var("x") * (1 + 5e-10),
                tol=1e-9,
                scale=relative,
            ),
        )

    res = _run(Near, tmp_path, "near")
    r = res.invariants[-1]
    assert r.passed
    assert 1e-10 < r.worst < 2e-9  # 5e-10 of x(0) = 3 for S02

    class Over(_Scaled):
        invariants = (
            Invariant(
                "x within a billionth of x",
                lambda a: a.var("x"),
                lambda a: a.var("x") * (1 + 2e-9),
                tol=1e-9,
                scale=relative,
            ),
        )

    with pytest.raises(
        RunError, match=r"failed: policy S00, t = 0: lhs = 1\.0, rhs = 1\.000000002"
    ):
        _run(Over, tmp_path, "over")


def test_where_excludes_the_months_and_policies_it_masks(tmp_path: Path) -> None:
    class Unmasked(Growth):
        invariants = (
            Invariant("cf below 85", lambda a: a.var("cf")[1:], 85.0, kind="le", months=(1, "H")),
        )

    # S02: cf(1) = 3 * 30 = 90, falling by 1% a month; below 85 from t = 7
    with pytest.raises(RunError, match=r"failed: policy S02, t = 1: lhs = 90\.0, rhs = 85\.0"):
        _run(Unmasked, tmp_path, "unmasked")

    class LateMonths(Growth):
        invariants = (
            Invariant(
                "cf below 85 from t = 7",
                lambda a: a.var("cf")[1:],
                85.0,
                kind="le",
                months=(1, "H"),
                where=lambda a: a.t[1:] > 6,
            ),
        )

    res = _run(LateMonths, tmp_path, "late")
    assert res.invariants[-1].passed
    assert res.invariants[-1].worst == 0.0  # the masked residuals count for nothing

    class OtherPolicies(Growth):
        invariants = (
            Invariant(
                "cf below 85 but for the largest premium",
                lambda a: a.var("cf")[1:],
                85.0,
                kind="le",
                months=(1, "H"),
                where=lambda a: a.mp.premium < 30,
            ),
        )

    res = _run(OtherPolicies, tmp_path, "others", batch_size=2)
    assert res.invariants[-1].passed
    assert res.invariants[-1].worst == 0.0


def test_a_residual_that_is_not_a_number_fails(tmp_path: Path) -> None:
    class NotANumber(Growth):
        invariants = (
            Invariant("nan lhs", lambda a: a.var("x") * np.nan, lambda a: a.var("x"), tol=0),
        )

    with pytest.raises(
        RunError,
        match=r"invariant nan lhs failed: policy S00, t = 0: lhs = nan, rhs = 1\.0; the residual "
        r"is not a number",
    ):
        _run(NotANumber, tmp_path, "nan")
    assert not (tmp_path / "nan" / "invariants.json").exists()
    assert not (tmp_path / "nan" / "manifest.json").exists()

    class Infinite(Growth):
        invariants = (
            Invariant("inf lhs", lambda a: a.var("x") * np.inf, lambda a: a.var("x"), tol=0),
        )

    with pytest.raises(RunError, match=r"lhs = inf, rhs = 1\.0; the residual inf exceeds the "):
        _run(Infinite, tmp_path, "inf")

    class MaskedOut(Growth):
        """A division that is not a number only where the mask excludes it."""

        invariants = (
            Invariant(
                "nan outside the mask",
                lambda a: np.where(a.t > 0, a.var("x"), np.nan),
                lambda a: a.var("x"),
                tol=0,
                where=lambda a: a.t > 0,
            ),
        )

    res = _run(MaskedOut, tmp_path, "masked")
    assert res.invariants[-1].passed
    assert res.invariants[-1].worst == 0.0
    on_disk = (tmp_path / "masked" / "invariants.json").read_text(encoding="utf-8")
    assert "NaN" not in on_disk
    assert "Infinity" not in on_disk
    assert all(np.isfinite(r["worst"]) for r in json.loads(on_disk))


def test_merging_never_loses_a_residual_that_is_not_a_number() -> None:
    ok = InvariantResult("r", True, 0.5, "")
    nan = InvariantResult("r", True, float("nan"), "")
    assert merge_results([[nan], [ok]])[0].worst == float("inf")
    assert merge_results([[ok], [nan]])[0].worst == float("inf")
    assert merge_results([[ok], [ok]])[0].worst == 0.5


def test_where_must_return_booleans(tmp_path: Path) -> None:
    class Forgotten(Growth):
        invariants = (
            Invariant(
                "a mask without a comparison",
                lambda a: a.var("x"),
                0.0,
                kind="ge",
                tol=0,
                where=lambda a: a.var("x") * 0.0,
            ),
        )

    with pytest.raises(
        RunError, match="invariant a mask without a comparison: where must return booleans, not"
    ) as info:
        _run(Forgotten, tmp_path)
    assert "float64" in str(info.value)
    assert info.value.loc is not None
    assert Path(info.value.loc.file) == Path(__file__)


def test_an_empty_or_misplaced_window_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"invariant bad: months: \(5, 2\) is an empty window"):
        Invariant("bad", 1.0, months=(5, 2))
    with pytest.raises(ValueError, match=r"months: \('H', 'H-1'\) is an empty window"):
        Invariant("bad", 1.0, months=("H", "H-1"))
    assert Invariant("fine", 1.0, months=("H-1", "H")).months == ("H-1", "H")
    assert Invariant("fine", 1.0, months=(3, 3)).months == (3, 3)

    class Reversed(Growth):
        # the order of a mixed pair depends on the horizon, so it is checked at the run
        invariants = (Invariant("reversed", lambda a: a.var("x"), 0.0, months=("H", 3)),)

    with pytest.raises(RunError, match=r"months \(12, 3\) is an empty window at H = 12"):
        _run(Reversed, tmp_path, "reversed")

    class Early(Growth):
        invariants = (Invariant("early", lambda a: a.var("x"), 0.0, months=(0, "H-13")),)

    with pytest.raises(RunError, match=r"months \(0, -1\) is an empty window at H = 12"):
        _run(Early, tmp_path, "early")

    class Outside(Growth):
        invariants = (Invariant("outside", lambda a: a.var("x"), 0.0, months=(-1, "H")),)

    with pytest.raises(RunError, match=r"months \(-1, 12\) is outside 0\.\.12 at H = 12"):
        _run(Outside, tmp_path, "outside")


def test_at_requires_an_integer_index(tmp_path: Path) -> None:
    """The model's ``at()`` takes an Int index (D11); the view is no laxer, so a float index is
    refused instead of being truncated silently."""

    class FloatIndex(Growth):
        invariants = (
            Invariant(
                "x at a float month",
                lambda a: a.at("x", a.mp.premium / 10),
                0.0,
                kind="ge",
                tol=0,
            ),
        )

    with pytest.raises(
        RunError,
        match=r"invariant x at a float month: a\.at\('x', index\): the index is float64, not an "
        "integer",
    ) as info:
        _run(FloatIndex, tmp_path)
    _declared_here(info.value)


def test_the_view_is_read_only(tmp_path: Path) -> None:
    """``a.mp.<field>``, ``a.table``, ``a.scen``, ``a.scale``, ``a.ids`` and ``a.t`` are views of
    the loaded inputs or shared by the batch's invariants (the Money scale here is the premium
    column itself); writing into one raises, and the inputs are untouched for the later
    invariants and batches and for the run's result and its files."""
    refused: list[str] = []
    seen_t: list[int] = []

    def tamper(a: BatchView) -> Any:
        targets = {
            "mp": (a.mp.premium, 999.0),
            "table": (a.table("margin", "share"), 999.0),
            "scen": (a.scen("df", 1), 999.0),
            "scen_range": (a.scen_range("df", 0, 2), 999.0),
            "scale": (a.scale("Money"), 999.0),
            "ids": (a.ids, "ZZZ"),
            "t": (a.t, 999),
        }
        for name, (target, value) in targets.items():
            try:
                target[...] = value
            except ValueError:
                refused.append(name)
        return a.var("x")

    def later(a: BatchView) -> Any:
        seen_t.append(int(a.t[0, 0, 0]))
        return a.var("x")

    class Tampering(Margin):
        scenario_inputs = Indexed.scenario_inputs[:1]
        tolerance_scales: ClassVar[dict[str, str]] = {"Money": "premium"}
        invariants = (
            Invariant("x >= 0", tamper, 0.0, kind="ge", tol=0),
            Invariant("x >= 0 again", later, 0.0, kind="ge", tol=0),
        )

        @var
        def disc(self, t: Int) -> Factor:
            return self.scen.df(t)

    df = scenario_file("df", 0, 12, 1, lambda s, u: 0.99**u)
    res = run(
        build(Tampering),
        model_points=simple_model_points(),
        tables={"margin": margin_table(0.5)},
        scenarios={"df": df},
        settings=RunSettings(valuation_date=VAL, batch_size=1),
        run_dir=tmp_path / "tamper",
    )
    assert refused == ["mp", "table", "scen", "scen_range", "scale", "ids", "t"] * 3
    assert seen_t == [0, 0, 0]
    assert res.invariants[-1].passed
    assert res.inputs is not None
    assert list(res.inputs.model_points.columns["premium"]) == [10.0, 20.0, 30.0]
    assert res.inputs.tables["margin"].values["share"][()] == 0.5
    assert res.inputs.scenarios["df"].values[1, 0, 0] == 0.99
    assert res.inputs.scenarios["df"].values[2, 0, 0] == 0.99**2
    assert list(res.ids) == ["S00", "S01", "S02"]
    assert list(res.inputs.model_points.ids) == ["S00", "S01", "S02"]
    monthly = pq.read_table(res.run_dir / "outputs" / "monthly.parquet")
    assert sorted(set(monthly.column("policy_id").to_pylist())) == ["S00", "S01", "S02"]


def test_a_one_row_result_under_a_window_names_the_window(tmp_path: Path) -> None:
    """A one-row result is one month of its window, which one is not known; the report says so
    instead of naming the window's first month. A one-month window names the month."""

    class Point(Growth):
        invariants = (
            Invariant("x at 5 is zero", lambda a: a.at("x", 5), 0.0, tol=0, months=(1, "H")),
        )

    with pytest.raises(
        RunError, match=r"invariant x at 5 is zero failed: policy S00, one month of 1\.\.12: lhs = "
    ) as info:
        _run(Point, tmp_path)
    assert ", t = " not in str(info.value)

    class Pinned(Growth):
        invariants = (
            Invariant("x at 5 is zero", lambda a: a.at("x", 5), 0.0, tol=0, months=(5, 5)),
        )

    with pytest.raises(RunError, match=r"failed: policy S00, t = 5: lhs = "):
        _run(Pinned, tmp_path, "pinned")


def test_a_failure_is_reported_before_a_later_invariant_is_evaluated(tmp_path: Path) -> None:
    class FailThenRaise(Growth):
        invariants = (
            Invariant("fails", lambda a: a.var("x"), -1.0, kind="le", tol=0),
            Invariant("raises", lambda a: a.var("nope"), 0.0),
        )

    with pytest.raises(RunError, match=r"invariant fails failed: policy S00, t = 0: lhs = 1\.0"):
        _run(FailThenRaise, tmp_path)
