# pyright: basic
"""Trace, explain and diff."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest

from halley import (
    Annual,
    Column,
    CompiledModel,
    Factor,
    Field,
    Int,
    Key,
    Money,
    Product,
    Rate,
    RunResult,
    RunSettings,
    Schema,
    Table,
    ann_spot,
    build,
    diff,
    explain,
    pv_month,
    run,
    var,
    where,
)
from halley.authoring import int_field
from halley.products.term_assurance import VALUATION_DATE
from halley.runtime import recompute, within
from models import (
    SIMPLE_SCHEMA,
    VAL,
    Expensed,
    Indexed,
    Margin,
    margin_table,
    scenario_file,
    simple_model_points,
)
from test_helpers import run_helpers


def test_explain_shows_the_formula_and_every_input(term_run: RunResult) -> None:
    e = explain(term_run, "P05", "reserve", 13)
    assert within(e.value, 46.35078406367477, 200_000)
    assert e.unit == "Money"
    assert e.loc.function.endswith("TermAssurance.reserve")
    inputs = dict(e.inputs)
    assert inputs["reserve(t+1)"] == 0.0
    assert within(inputs["net_outgo(t+1)"], 46.502524469400136, 200_000)
    assert within(inputs["v(t+1)"], 0.9967369426185624, 1)
    text = str(e)
    assert "def reserve(self, t: Int) -> Money:" in text
    assert "reserve(t+1) = 0.0" in text


def test_explain_a_scalar_and_a_lookup(term_run: RunResult) -> None:
    e = explain(term_run, "P05", "pv_net_cf")
    assert e.t is None
    assert within(e.value, -537.9608658387098, 200_000)
    q = explain(term_run, "P05", "q_ann", 3)
    assert dict(q.inputs)["mp.sex"] == "M"
    assert dict(q.inputs)["age(t)"] == 51
    assert within(q.value, 0.0025001, 1, rel=1e-5)


def test_recompute_matches_the_engine_lane_by_lane(term_run: RunResult) -> None:
    for name in ("deaths", "lapses", "pols_if", "net_outgo", "reserve", "reserve_pp", "disc"):
        for t in (0, 1, 2, 3, 13, 14, 15):
            assert recompute(term_run, "P05", name, t) == term_run.lane(name, "P05")[t], (name, t)
    assert recompute(term_run, "P05", "pv_premiums", 0) == term_run.lane("pv_premiums", "P05")[0]


def test_recompute_matches_the_generated_code_for_every_helper(tmp_path: Path) -> None:
    """The interpreter evaluates each helper's emitted form in the same floating-point order as
    the backend (D19), so the two agree bit for bit on a model that uses every helper, the
    three pv_month expansions and both run constants."""
    res = run_helpers(tmp_path)
    assert res.graph is not None
    for name, v in res.graph.variables.items():
        months = range(res.H + 1) if v.has_time else [0]
        for pid in res.ids:
            lane = res.lane(name, pid)
            for t in months:
                assert recompute(res, pid, name, t) == lane[t], (name, pid, t)
    e = explain(res, "H01", "cal_year", 3)
    assert dict(e.inputs) == {"run.m_val": 12 * 2027 + 1 - 1}
    assert e.value == 2027
    assert e.unit == "Int"
    bel = explain(res, "H01", "bel", 5)
    assert set(dict(bel.inputs)) == {
        "bel(t+1)",
        "v(t+1)",
        "ann_out(t+1)",
        "exp_out(t+1)",
        "claims(t+1)",
    }


def test_explain_a_member_without_a_terminal_at_the_horizon(tmp_path: Path) -> None:
    """``inv_exp(H)`` is its formula on the terminal reserve, not a boundary value, so explain
    lists its inputs there and recompute evaluates the formula; the terminal is a year's premium
    per life, so the value is not a trivial zero."""
    res = run(
        build(Expensed),
        model_points=simple_model_points(),
        settings=RunSettings(valuation_date=VAL),
        run_dir=tmp_path / "expensed",
    )
    bel_h = res.lane("bel", "S01")[res.H]
    ie = res.lane("ie_m", "S01")[0]
    assert bel_h == 2.0 * 20.0  # S01: pols_if_init 2, premium 20
    e = explain(res, "S01", "inv_exp", res.H)
    assert e.value == ie * bel_h != 0.0
    assert dict(e.inputs)["bel(t)"] == bel_h
    assert dict(e.inputs)["ie_m(t)"] == ie
    assert recompute(res, "S01", "inv_exp", res.H) == e.value
    e1 = explain(res, "S01", "bel", res.H - 1)
    assert dict(e1.inputs)["inv_exp(t+1)"] == e.value
    assert dict(e1.inputs)["cf(t+1)"] == res.lane("cf", "S01")[res.H]


def test_explain_a_variable_whose_init_reads_another(tmp_path: Path) -> None:
    res = run(
        build(Margin),
        model_points=simple_model_points(),
        tables={"margin": margin_table()},
        settings=RunSettings(valuation_date=VAL),
        run_dir=tmp_path / "margin",
    )
    csm_0 = res.lane("csm_0", "S02")[0]
    e = explain(res, "S02", "csm", 1)
    assert dict(e.inputs) == {"csm(t-1)": csm_0}
    assert e.value == csm_0 * 0.95
    assert "init=lambda g: g.csm_0()" in str(e)
    assert recompute(res, "S02", "csm", 0) == csm_0
    assert recompute(res, "S02", "lc", 0) == res.lane("fcf", "S02")[0]


MID_SCHEMA = Schema(
    fields=(
        Field("policy_id", "string", unique=True),
        Field("disc", "number", "Factor", gt=0, le=1),
        Field("premium", "number", "MoneyPerPolicy", gt=0),
        Field("pols_if_init", "number", "Count", gt=0),
        int_field("rem_m", ge=1, le=1000),
    ),
    horizon="rem_m",
)


class MidTimed(Product):
    """A mid-month cash flow discounted by a constant monthly factor: the backward recursion's
    last step is ``cf(H) * (v(H) ** 0.5)``, which with ``cf = 1`` is ``v ** 0.5`` itself."""

    model_point = MID_SCHEMA

    @var
    def v(self, t: Int) -> Factor:
        return self.mp.disc

    @var(timing="mid")
    def cf(self, t: Int) -> Money:
        return self.mp.pols_if_init * self.mp.premium

    @var(terminal=0)
    def pv(self, t: Int) -> Money:
        return (self.pv(t + 1) * self.v(t + 1)) + pv_month(self.cf, t + 1, self.v)


def _fast_path_values(
    exponent: float, lo: float, hi: float, n: int = 3, *, loop: bool = False
) -> list[float]:
    """Values in ``(lo, hi)`` for which NumPy's ``x ** exponent`` over an array (its fast path
    for 0.5, 2 and -1, which the power loop takes for a scalar exponent too) differs in the last
    bit from pow: by default Python's ``**``, the C library's pow, the path a scalar base takes
    in the generated code; with ``loop``, NumPy's power loop for an array exponent, the path an
    array exponent takes (the C library's pow on this laptop and on Linux without AVX-512,
    NumPy's own kernel on an AVX-512 Linux machine, where the two differ). About one value in a
    thousand does. A machine whose pow agrees with the fast path on every value offers none, in
    which case the mirror cannot be told from pow here and the test is skipped."""
    rng = np.random.default_rng(20261017)
    found: list[float] = []
    for _ in range(50):
        xs = rng.uniform(lo, hi, 20_000)
        fast = xs**exponent
        if loop:
            hits = xs[np.power(xs, np.full_like(xs, exponent)) != fast].tolist()
        else:
            hits = [x for x, f in zip(xs.tolist(), fast.tolist(), strict=True) if x**exponent != f]
        found.extend(hits)
        if len(found) >= n:
            return found[:n]
    where = "NumPy's power loop" if loop else "the C library"
    pytest.skip(
        f"pow(x, {exponent}) in {where} equals NumPy's fast path on every probed value; the "
        "fast-path mirror cannot be distinguished here"
    )


def _sqrt_fast_path_values(n: int = 3) -> list[float]:
    """Discount factors in (0.9, 1) whose ``** 0.5`` differs between sqrt and pow."""
    return _fast_path_values(0.5, 0.9, 1.0, n)


def _reciprocal_fast_path_values(n: int = 3, *, loop: bool = False) -> list[float]:
    """Discount factors in (0.5, 1) whose ``** -1.0`` differs between 1 / x and pow (the C
    library's, or with ``loop`` NumPy's power loop's)."""
    return _fast_path_values(-1.0, 0.5, 1.0, n, loop=loop)


def _loop_pow(x: float, exponent: float) -> float:
    """``x ** exponent`` as NumPy's power loop computes it for an array exponent, the path the
    generated code takes for one: the C library's pow here, NumPy's own kernel on an AVX-512
    Linux machine (a scalar exponent would take the loop's fast path instead)."""
    return float(np.power(np.array([x]), np.array([exponent])).item())


def _points(discs: list[float], rem_m: int = 6) -> Any:
    """``MID_SCHEMA`` model points, one per discount factor, ``D0, D1, ...``."""
    n = len(discs)
    return pa.table(
        {
            "policy_id": pa.array([f"D{i}" for i in range(n)], pa.string()),
            "disc": pa.array(discs, pa.float64()),
            "premium": pa.array([1.0] * n, pa.float64()),
            "pols_if_init": pa.array([1.0] * n, pa.float64()),
            "rem_m": pa.array([rem_m] * n, pa.int32()),
        }
    )


def test_recompute_mirrors_the_power_fast_path_of_the_generated_code(tmp_path: Path) -> None:
    """``pv_month`` of a mid-month flow emits ``cf * (v ** 0.5)`` and the backend evaluates
    ``ndarray ** 0.5`` through sqrt, which differs from libm's pow (Python's ``**``) in the last
    bit for some values. Every model point here is such a value, so the interpreter agrees with
    the backend only when it takes the same path (D19 for the second implementation)."""
    discs = _sqrt_fast_path_values()
    res = run(
        build(MidTimed),
        model_points=_points(discs),
        settings=RunSettings(valuation_date=VAL),
        run_dir=tmp_path / "mid",
    )
    for pid, d in zip(res.ids, discs, strict=True):
        lane = res.lane("pv", pid)
        # the last step is 0 * v + 1.0 * (v ** 0.5): the fast path's value, not pow's
        assert lane[res.H - 1] == float(np.sqrt(d))
        assert lane[res.H - 1] != d**0.5
        for t in range(res.H + 1):
            assert recompute(res, pid, "pv", t) == lane[t], (pid, t)
    e = explain(res, "D0", "pv", res.H - 1)
    assert dict(e.inputs)["v(t+1)"] == discs[0]


RATES = Table(
    "rates",
    keys=(Key("name", enum=("a", "b", "c")),),
    columns=(Column("f", "Factor", ge=0, le=1),),
)
HALF = Table("half", keys=(), columns=(Column("d", "Factor", ge=0, le=1),))


def _keyed_root(key: str) -> type[Product]:
    """A model whose scalar ``root`` is a constant-key lookup raised to 0.5."""

    class Keyed(Product):
        model_point = MID_SCHEMA
        assumption_tables = (RATES,)

        @var
        def root(self) -> Factor:
            return self.tables.rates.f[key] ** 0.5

    return Keyed


class Keyless(Product):
    """A keyless table's value is a 0-d array in the generated code, which does take the fast
    path."""

    model_point = MID_SCHEMA
    assumption_tables = (HALF,)

    @var
    def root(self) -> Factor:
        return self.tables.half.d**0.5


def test_recompute_follows_the_backend_scalar_ness_of_a_lookup_base(tmp_path: Path) -> None:
    """A constant-key lookup is a NumPy scalar in the generated code (``T_rates__f[0]``), so
    ``** 0.5`` there is libm's pow, never the array fast path; a keyless table's value is a 0-d
    array, which takes it. The interpreter decides the path from the backend's own kind of
    value, so it agrees bit for bit with both."""
    discs = _sqrt_fast_path_values()
    rates = pa.table({"name": pa.array(["a", "b", "c"]), "f": pa.array(discs, pa.float64())})
    for key, d in zip(("a", "b", "c"), discs, strict=True):
        res = run(
            build(_keyed_root(key)),
            model_points=_points(discs[:1]),
            tables={"rates": rates},
            settings=RunSettings(valuation_date=VAL),
            run_dir=tmp_path / f"keyed-{key}",
        )
        value = res.lane("root", "D0")[0]
        assert value == d**0.5  # libm's pow
        assert value != float(np.sqrt(d))
        assert recompute(res, "D0", "root", 0) == value
    for i, d in enumerate(discs):
        res = run(
            build(Keyless),
            model_points=_points(discs[:1]),
            tables={"half": pa.table({"d": pa.array([d], pa.float64())})},
            settings=RunSettings(valuation_date=VAL),
            run_dir=tmp_path / f"keyless-{i}",
        )
        value = res.lane("root", "D0")[0]
        assert value == float(np.sqrt(d))  # the 0-d array takes the fast path
        assert value != d**0.5
        assert recompute(res, "D0", "root", 0) == value


POWER = Table("power", keys=(), columns=(Column("e", "Int"),))


class KeylessExponent(Product):
    """A constant-key lookup raised to a keyless table's Int value: the generated code holds a
    NumPy scalar base and a 0-d exponent, ``T_rates__f[0] ** T_power__e``."""

    model_point = MID_SCHEMA
    assumption_tables = (RATES, POWER)

    @var
    def inv(self) -> Factor:
        return self.tables.rates.f["a"] ** self.tables.power.e


def test_recompute_takes_the_fast_path_for_a_zero_d_exponent(tmp_path: Path) -> None:
    """A 0-d exponent takes NumPy's fast path against any base, a scalar one included, so a
    constant-key lookup raised to a keyless Int column holding -1 is stored as the reciprocal,
    not libm's pow, and the interpreter takes the same path."""
    discs = _reciprocal_fast_path_values()
    compiled = build(KeylessExponent)
    assert "(T_rates__f[0] ** T_power__e)" in compiled.source
    power = pa.table({"e": pa.array([-1], pa.int64())})
    for i, d in enumerate(discs):
        rates = pa.table(
            {"name": pa.array(["a", "b", "c"]), "f": pa.array([d, 0.5, 0.5], pa.float64())}
        )
        res = run(
            compiled,
            model_points=_points(discs[:1]),
            tables={"rates": rates, "power": power},
            settings=RunSettings(valuation_date=VAL),
            run_dir=tmp_path / f"inverse-{i}",
        )
        value = res.lane("inv", "D0")[0]
        assert value == 1.0 / d  # the reciprocal fast path
        assert value != d**-1.0  # not libm's pow
        assert recompute(res, "D0", "inv", 0) == value


def _pow_mismatch_month(den: int = 1000, horizon: int = 12) -> tuple[int, int]:
    """``(k, t)`` with ``t`` in ``1..horizon`` such that ``(t + k) / den`` is a value whose
    ``** 0.5`` differs between sqrt and pow; skips when this libm offers none."""
    for k in range(200_000):
        for t in range(1, horizon + 1):
            x = (t + k) / den
            if x**0.5 != float(np.sqrt(x)):
                return k, t
    pytest.skip("this libm's pow(x, 0.5) equals sqrt(x) on every probed month value")


def test_recompute_takes_libm_pow_inside_a_recursive_component(tmp_path: Path) -> None:
    """Inside a recursive component ``t`` is a Python int, so ``((t + k) / 1000) ** 0.5`` is a
    Python float raised by libm's pow; outside one the same expression is the month column
    raised through the sqrt fast path. The interpreter is told the reader's direction and takes
    each path accordingly."""
    k, t_hit = _pow_mismatch_month()

    class Stepped(Product):
        model_point = SIMPLE_SCHEMA

        @var(init=0.0)
        def acc(self, t: Int) -> Factor:
            return self.acc(t - 1) + self.term(t)

        @var
        def term(self, t: Int) -> Factor:
            # a member of acc's component: t is a Python int here
            return where(self.acc(t - 1) >= 0, ((t + k) / 1000) ** 0.5, 0.0)

        @var
        def whole(self, t: Int) -> Factor:
            # the same expression outside the component: t is the month column
            return ((t + k) / 1000) ** 0.5

    compiled = build(Stepped)
    assert f"(((t + {k}) / 1000) ** 0.5)" in compiled.source
    assert f"(((_t[1:H+1] + {k}) / 1000) ** 0.5)" in compiled.source
    res = run(
        compiled,
        model_points=simple_model_points(),
        settings=RunSettings(valuation_date=VAL),
        run_dir=tmp_path / "stepped",
    )
    x = (t_hit + k) / 1000
    for pid in res.ids:
        term, whole = res.lane("term", pid), res.lane("whole", pid)
        assert term[t_hit] == x**0.5  # libm's pow
        assert term[t_hit] != float(np.sqrt(x))
        assert whole[t_hit] == float(np.sqrt(x))  # the sqrt fast path
        assert whole[t_hit] != x**0.5
        for name in ("acc", "term", "whole"):
            lane = res.lane(name, pid)
            for t in range(res.H + 1):
                assert recompute(res, pid, name, t) == lane[t], (name, pid, t)


class Spot(Product):
    """``ann_spot`` with a literal month: the emitted exponent ``(-12 / 12)`` is the Python float
    -1.0, so the backend takes the reciprocal fast path on the discount array."""

    model_point = MID_SCHEMA

    @var(domain="state")
    def df(self, t: Int) -> Factor:
        return self.mp.disc

    @var
    def spot(self, t: Int) -> Rate[Annual]:
        return ann_spot(self.df(t), 12)


def test_recompute_mirrors_the_reciprocal_fast_path_inside_ann_spot(tmp_path: Path) -> None:
    discs = _reciprocal_fast_path_values()
    res = run(
        build(Spot),
        model_points=_points(discs),
        settings=RunSettings(valuation_date=VAL),
        run_dir=tmp_path / "spot",
    )
    for pid, d in zip(res.ids, discs, strict=True):
        lane = res.lane("spot", pid)
        assert lane[1] == (1.0 / d) - 1  # the reciprocal fast path
        assert lane[1] != (d**-1.0) - 1  # not libm's pow
        for t in range(res.H + 1):
            assert recompute(res, pid, "spot", t) == lane[t], (pid, t)


class SpotByMonth(Product):
    """``ann_spot`` at the month itself: the emitted exponent ``-12 / _t[1:H+1]`` is the month
    column, an array, which is exactly -1.0 at t = 12."""

    model_point = MID_SCHEMA

    @var(domain="state")
    def df(self, t: Int) -> Factor:
        return self.mp.disc

    @var
    def spot(self, t: Int) -> Rate[Annual]:
        return ann_spot(self.df(t), t)


def test_recompute_takes_the_power_loop_for_the_month_exponent_of_ann_spot(
    tmp_path: Path,
) -> None:
    """A month-column exponent over a window of several months takes NumPy's power loop at
    every batch size, so ``ann_spot(df(t), t)`` at t = 12 is ``np.power(d, -1.0) - 1``, not the
    reciprocal, and the interpreter, which routes ``ann_spot``'s power by the kinds of both
    operands, agrees bit for bit at every month."""
    discs = _reciprocal_fast_path_values(loop=True)
    compiled = build(SpotByMonth)
    assert "** (-12 / _t[1:H+1])" in compiled.source
    for batch_size in (1, 3):
        res = run(
            compiled,
            model_points=_points(discs, rem_m=12),
            settings=RunSettings(valuation_date=VAL, batch_size=batch_size),
            run_dir=tmp_path / f"spot-{batch_size}",
        )
        for pid, d in zip(res.ids, discs, strict=True):
            lane = res.lane("spot", pid)
            assert lane[12] == _loop_pow(d, -1.0) - 1  # the power loop
            assert lane[12] != (1.0 / d) - 1  # not the reciprocal fast path
            for t in range(res.H + 1):
                assert recompute(res, pid, "spot", t) == lane[t], (batch_size, pid, t)


class InverseByMonth(Product):
    """A model point field raised to the negated month column, ``M_disc ** (-_t[1:H+1])``."""

    model_point = MID_SCHEMA

    @var
    def inv(self, t: Int) -> Factor:
        return self.mp.disc ** (-t)


def test_a_month_exponent_over_one_month_takes_the_fast_path(tmp_path: Path) -> None:
    """The documented exception to the month-column rule: over a one-month window (H = 1 for a
    formula from t = 1) the month column is a size-one array and NumPy takes its reciprocal fast
    path at every batch size, so the stored value is ``1 / d`` while ``recompute`` (the power
    loop) differs from it in the last place. Over two months the column takes the loop and the
    two agree bit for bit. A NumPy that chooses otherwise fails here, which flags the documented
    rule."""
    discs = _reciprocal_fast_path_values(loop=True)
    compiled = build(InverseByMonth)
    assert "(M_disc ** (-_t[1:H+1]))" in compiled.source
    for batch_size in (1, 3):
        one = run(
            compiled,
            model_points=_points(discs, rem_m=1),
            settings=RunSettings(valuation_date=VAL, batch_size=batch_size),
            run_dir=tmp_path / f"one-month-{batch_size}",
        )
        two = run(
            compiled,
            model_points=_points(discs, rem_m=2),
            settings=RunSettings(valuation_date=VAL, batch_size=batch_size),
            run_dir=tmp_path / f"two-months-{batch_size}",
        )
        for pid, d in zip(one.ids, discs, strict=True):
            stored = one.lane("inv", pid)[1]
            assert stored == 1.0 / d  # the reciprocal fast path
            again = recompute(one, pid, "inv", 1)
            assert again == _loop_pow(d, -1.0)  # the power loop
            assert abs(again - stored) <= 1e-12 * abs(stored)
            lane = two.lane("inv", pid)
            assert lane[1] == _loop_pow(d, -1.0)
            assert lane[1] != 1.0 / d
            for t in range(two.H + 1):
                assert recompute(two, pid, "inv", t) == lane[t], (batch_size, pid, t)


def test_diff_locates_the_largest_change(
    compiled_term: CompiledModel,
    reference_sources: dict[str, Any],
    term_run: RunResult,
    tmp_path: Path,
) -> None:
    table = reference_sources["model_points"]
    rows = table.to_pylist()
    rows[4]["premium"] = rows[4]["premium"] * 2  # P05
    changed = run(
        compiled_term,
        model_points=pa.Table.from_pylist(rows, schema=table.schema),
        tables=reference_sources["tables"],
        settings=RunSettings(valuation_date=VALUATION_DATE),
        run_dir=tmp_path / "changed",
    )
    d = diff(term_run, changed)
    assert d["premium_income"].policy_id == "P05"
    assert d["premium_income"].t == 1
    assert d["premium_income"].max_abs == pytest.approx(46.40, abs=1e-9)
    assert d["death_outgo"].max_abs == 0.0
    assert d["reserve_0"].policy_id == "P05"


def test_explain_reads_no_scenario_month_outside_its_file(tmp_path: Path) -> None:
    """A scenario input read at a month its file does not hold is an error of that input alone,
    as a variable read outside 0..H is: explain listed the file's last row for a month before
    its first (a negative row index) and failed whole for a month past its last. Such a month is
    read only at a boundary month, where the formula's value is not used."""
    res = run(
        build(Indexed),
        model_points=simple_model_points(rem_m=6),
        scenarios={
            "df": scenario_file("df", 0, 6, 1, lambda s, u: 0.99**u),
            "cpi": scenario_file("cpi", -2, 6, 1, lambda s, u: 100.0 + u),  # all the loader asks
        },
        settings=RunSettings(valuation_date=VAL),
        run_dir=tmp_path / "indexed",
    )
    pid = res.ids[0]
    before = dict(explain(res, pid, "esc", 0).inputs)  # esc is a month quantity, from t = 1
    assert before["scen.cpi(t-3)"] == "<scen.cpi read at month -3, outside -2..6>"
    assert before["scen.cpi.at(0)"] == 100.0
    inside = dict(explain(res, pid, "esc", 3).inputs)
    assert inside["scen.cpi(t-3)"] == 100.0
    assert recompute(res, pid, "esc", 3) == res.lane("esc", pid)[3]
