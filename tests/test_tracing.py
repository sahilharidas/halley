"""Tracing: capture by running each method once with a symbolic t (D8), and what it rejects."""

from __future__ import annotations

from typing import Any

import pytest

from halley import BuildError, Count, Int, Money, Product, Static, build, trace, var, where
from halley.ir import MpField, RunConst, VarRead, graph_hash, render, walk
from halley.products.lifelib_basic_term import BasicTerm
from halley.products.term_assurance import TermAssurance
from models import SIMPLE_SCHEMA, Growth

TERM_ASSURANCE_HASH = "33e332cf057ec6340f79c6d88e0750a82d3b641608ce4a1c86d83e83d3bb5a14"
BASIC_TERM_HASH = "acc1e97143421b336d55ebb3c10b4a1fc17405a3e5a3992ff71f0929f383e1c0"


def test_reference_model_graph_hashes_are_pinned() -> None:
    """The graphs of the two reference models (D22, D23) do not change under engine work: a
    different hash here means every existing run of them gets a new run id (D21)."""
    assert graph_hash(trace(TermAssurance)) == TERM_ASSURANCE_HASH
    assert graph_hash(trace(BasicTerm)) == BASIC_TERM_HASH


def test_graph_captures_variables_offsets_and_boundaries() -> None:
    g = trace(Growth)
    assert list(g.variables) == ["x", "cf", "r", "ratio", "masked", "total", "r0", "cf_at_k"]
    x = g.variables["x"]
    assert x.has_time
    assert x.role == "in_force"
    assert isinstance(x.init, MpField)
    assert x.init.name == "pols_if_init"
    reads = [n for n in [x.expr, *x.expr.children()] if isinstance(n, VarRead)]
    assert reads
    assert reads[0].offset == -1
    r = g.variables["r"]
    assert render(r.terminal or r.expr) == "0"
    assert render(r.expr) == "((r(t+1) + cf(t+1)) * 0.5)"
    assert not g.variables["total"].has_time
    assert render(g.variables["cf_at_k"].expr) == "at(cf, mp.k)"
    assert g.horizon_field == "rem_m"
    assert g.variables["x"].loc.file.endswith("models.py")


def test_static_branches_disappear_from_the_graph() -> None:
    class Switched(Product):
        model_point = SIMPLE_SCHEMA
        has_loading: Static[bool] = False

        @var
        def premium(self, t: Int) -> Money:
            base = self.mp.pols_if_init * self.mp.premium
            if self.has_loading:
                return base * 1.1
            return base

    plain = trace(Switched)
    loaded = trace(Switched, {"has_loading": True})
    assert render(plain.variables["premium"].expr) == "(mp.pols_if_init * mp.premium)"
    assert render(loaded.variables["premium"].expr) == "((mp.pols_if_init * mp.premium) * 1.1)"
    assert graph_hash(plain) != graph_hash(loaded)
    assert plain.statics == {"has_loading": False}
    with pytest.raises(BuildError, match="no build-time parameter 'has_waiver'"):
        trace(Switched, {"has_waiver": True})


def test_python_if_on_symbolic_value_names_where() -> None:
    class Bad(Product):
        model_point = SIMPLE_SCHEMA

        @var
        def x(self, t: Int) -> Money:
            if t > 3:  # a data-dependent condition
                return self.mp.premium * 1.0
            return self.mp.premium * 0.0

    with pytest.raises(BuildError, match="where") as info:
        trace(Bad)
    assert info.value.loc is not None
    assert info.value.loc.function.endswith("Bad.x")
    assert info.value.loc.file.endswith("test_tracing.py")


def test_min_and_max_on_symbolic_values_name_where() -> None:
    class Bad(Product):
        model_point = SIMPLE_SCHEMA

        @var
        def x(self, t: Int) -> Int:
            return min(t, 5)  # pyright: ignore[reportReturnType, reportArgumentType]

    with pytest.raises(BuildError, match="where"):
        trace(Bad)


def test_time_offsets_must_be_constants() -> None:
    class Bad(Product):
        model_point = SIMPLE_SCHEMA

        @var(init=lambda mp: mp.pols_if_init)
        def x(self, t: Int) -> Count:
            return self.x(t - self.mp.k)

    with pytest.raises(BuildError, match="constant"):
        trace(Bad)

    class Absolute(Product):
        model_point = SIMPLE_SCHEMA

        @var(init=lambda mp: mp.pols_if_init)
        def x(self, t: Int) -> Count:
            return self.x(5)  # pyright: ignore[reportArgumentType]

    with pytest.raises(BuildError, match=r"at\(x, 5\)"):
        trace(Absolute)


def _offsets(model: type[Product], name: str) -> list[int]:
    return [n.offset for n in walk(trace(model).variables[name].expr) if isinstance(n, VarRead)]


def test_time_offsets_fold_add_and_sub_trees() -> None:
    """An add/sub tree over one ``t`` and integer constants is one constant offset."""
    lag_m = 3

    class Folded(Product):
        model_point = SIMPLE_SCHEMA

        @var
        def x(self, t: Int) -> Money:
            return self.mp.premium * self.mp.pols_if_init

        @var
        def y(self, t: Int) -> Money:
            return (
                self.x((t - 3) - 12)
                + self.x(t - (lag_m + 12))
                + self.x(t + 1 - 1)
                + self.x(1 + t)
                + self.x((2 + t) - 1)
                + self.x(t - 3 + 1 - 1)
            )

    assert _offsets(Folded, "y") == [-15, -15, 0, 1, 1, -3]


@pytest.mark.parametrize(
    "case",
    ["t * 1", "t - (t - 1)", "t + 0.5", "t - mp.k", "-t + 2 * t"],
)
def test_time_offsets_that_are_not_one_t_plus_a_constant_are_rejected(case: str) -> None:
    class Bad(Product):
        model_point = SIMPLE_SCHEMA

        @var
        def x(self, t: Int) -> Money:
            return self.mp.premium * self.mp.pols_if_init

        @var
        def y(self, t: Int) -> Money:
            if case == "t * 1":
                return self.x(t * 1)
            if case == "t - (t - 1)":
                return self.x(t - (t - 1))
            if case == "t + 0.5":
                return self.x(t + 0.5)  # type: ignore[operator]
            if case == "t - mp.k":
                return self.x(t - self.mp.k)
            return self.x(-t + 2 * t)

    with pytest.raises(BuildError, match="the time offset of x must be a constant") as info:
        trace(Bad)
    assert "lag()" in str(info.value)
    assert "at(x, k)" in str(info.value)
    assert info.value.loc is not None
    assert info.value.loc.function.endswith("Bad.y")


def test_run_constants_are_read_as_self_run() -> None:
    class Dated(Product):
        model_point = SIMPLE_SCHEMA

        @var
        def cal_year(self, t: Int) -> Int:
            return (self.run.m_val + t) // 12

        @var
        def months(self) -> Int:
            return self.run.horizon

    g = trace(Dated)
    assert render(g.variables["cal_year"].expr) == "((run.m_val + t) // 12)"
    assert isinstance(g.variables["months"].expr, RunConst)
    assert graph_hash(g) != graph_hash(trace(Growth))

    class Unknown(Product):
        model_point = SIMPLE_SCHEMA

        @var
        def x(self) -> Int:
            return self.run.valuation_year

    with pytest.raises(BuildError, match="the run has no constant 'valuation_year'") as info:
        trace(Unknown)
    assert "m_val, horizon" in str(info.value)
    assert info.value.loc is not None
    assert info.value.loc.function.endswith("Unknown.x")


def test_abstract_classes_are_refused_and_the_flag_is_not_inherited() -> None:
    class Base(Product):
        abstract = True
        model_point = SIMPLE_SCHEMA

        @var
        def x(self, t: Int) -> Money:
            return self.mp.premium * self.mp.pols_if_init

    class Concrete(Base):
        pass

    class AbstractAgain(Concrete):
        abstract = True

    with pytest.raises(BuildError, match="Base is abstract"):
        build(Base)
    assert render(build(Concrete).graph.variables["x"].expr) == "(mp.premium * mp.pols_if_init)"
    with pytest.raises(BuildError, match="AbstractAgain is abstract"):
        trace(AbstractAgain)


def test_unknown_and_unreadable_fields() -> None:
    class Unknown(Product):
        model_point = SIMPLE_SCHEMA

        @var
        def x(self, t: Int) -> Money:
            return self.mp.sum_assured * 1.0

    with pytest.raises(BuildError, match="no field 'sum_assured'"):
        trace(Unknown)

    class NoTables(Product):
        model_point = SIMPLE_SCHEMA

        @var
        def x(self, t: Int) -> Money:
            return self.tables.rates["i"] * 1.0

    with pytest.raises(BuildError, match="declares no assumption table"):
        trace(NoTables)


def test_missing_unit_annotation_is_an_error() -> None:
    class NoUnit(Product):
        model_point = SIMPLE_SCHEMA

        @var
        def x(self, t: Int):  # type: ignore[no-untyped-def]
            return self.mp.premium

    with pytest.raises(BuildError, match="declares no unit"):
        trace(NoUnit)


def test_variant_conflicts_follow_d9() -> None:
    class Base(Product):
        model_point = SIMPLE_SCHEMA

        @var
        def x(self, t: Int) -> Money:
            return self.mp.premium * 1.0

    class A(Base):
        @var
        def x(self, t: Int) -> Money:
            return self.mp.premium * 2.0

    class B(Base):
        @var
        def x(self, t: Int) -> Money:
            return self.mp.premium * 3.0

    class Conflict(A, B):
        pass

    class Resolved(A, B):
        @var
        def x(self, t: Int) -> Money:
            return self.mp.premium * 4.0

    with pytest.raises(BuildError, match="two definitions of x"):
        trace(Conflict)
    assert render(trace(Resolved).variables["x"].expr) == "(mp.premium * 4.0)"
    assert render(trace(A).variables["x"].expr) == "(mp.premium * 2.0)"


def test_user_exceptions_point_at_the_model_line() -> None:
    class Broken(Product):
        model_point = SIMPLE_SCHEMA

        @var
        def x(self, t: Int) -> Money:
            helper: dict[str, Any] = {}
            return helper["missing"] * self.mp.premium

    with pytest.raises(BuildError, match="KeyError") as info:
        trace(Broken)
    assert info.value.loc is not None
    assert info.value.loc.file.endswith("test_tracing.py")


def test_where_with_python_bool_condition_is_a_constant() -> None:
    class Const(Product):
        model_point = SIMPLE_SCHEMA

        @var
        def x(self, t: Int) -> Money:
            return where(True, self.mp.premium * self.mp.pols_if_init, 0)

    compiled = build(Const)
    assert "np.where(True" in compiled.source
