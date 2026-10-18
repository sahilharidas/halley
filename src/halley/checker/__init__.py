"""Checker: units, dimensions, time references, domains, recursion direction and cycles.

Every rule here is a build error (Spec, Checker rules), except an unused variable and a group
series no formula reads, which are warnings. Errors point at the model source through the nodes'
recorded locations.

Domains. A variable with a time dimension is a *state*, defined from ``t = 0``, when it has an
init or terminal value, declares ``domain="state"``, or when another formula needs its value at
``t = 0`` (reads it at ``t-1`` from month 1, or at ``t`` from a state, or with ``at(x, 0)``).
Every other variable is a *month quantity*, defined from ``t = 1`` and stored as 0 at ``t = 0``.
A read outside a variable's domain is an error. The group series an aggregation stage reads are
inputs whose domain the declaration fixes; a read outside it is an error in the same way.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

from halley import ir
from halley.errors import BuildError, BuildWarning, Loc, UnitError
from halley.ir import units

Dims = ir.Dims


def _const_unit(value: object) -> units.Unit:
    if isinstance(value, bool):
        return units.BOOL
    if isinstance(value, int):
        return units.INTEGER
    if isinstance(value, float):
        return units.NUMBER
    return units.ENUM


class _Checker:
    def __init__(self, graph: ir.Graph) -> None:
        self.graph = graph
        self.vars = graph.variables

    # ------------------------------------------------------------------ units per node
    def unit_of(self, node: ir.Node) -> units.Unit:
        unit = self._unit_of(node)
        node.unit = unit
        return unit

    def _unit_of(self, node: ir.Node) -> units.Unit:
        loc = node.loc
        if isinstance(node, ir.Const):
            return _const_unit(node.value)
        if isinstance(node, ir.TimeIndex):
            return units.INT
        if isinstance(node, ir.VarRead):
            return self._var(node.name, loc).declared_unit
        if isinstance(node, ir.VarAt):
            self._int_index(node.index, loc, f"at({node.name}, ...)")
            return self._var(node.name, loc).declared_unit
        if isinstance(node, ir.SumT):
            unit = self._var(node.name, loc).declared_unit
            if unit not in units.SUMMABLE:
                raise UnitError(
                    f"sum_t({node.name}) sums a {unit}; only Money, MoneyPerPolicy, Count, "
                    "Factor and CoverageUnits are summed over time",
                    loc,
                )
            return unit
        if isinstance(node, ir.MpField):
            decl = self.graph.fields[node.name]
            if decl.unit is None:
                raise UnitError(f"mp.{node.name} has no unit", loc)
            return decl.unit
        if isinstance(node, ir.TableLookup):
            return self._lookup_unit(node)
        if isinstance(node, ir.ScenRead):
            return self.graph.scenarios[node.name].unit
        if isinstance(node, ir.ScenAt):
            self._int_index(node.index, loc, f"scen.{node.name}.at(...)")
            return self.graph.scenarios[node.name].unit
        if isinstance(node, ir.SeriesRead):
            decl = self._series(node.name, loc)
            if not decl.has_time and node.offset is not None:
                raise BuildError(
                    f"series {node.name} has no time dimension; read it as "
                    f"self.series.{node.name}()",
                    loc,
                )
            if decl.has_time and node.offset is None:
                raise BuildError(
                    f"series {node.name} has a time dimension; read it at a month, as "
                    f"self.series.{node.name}(t)",
                    loc,
                )
            return decl.unit
        if isinstance(node, ir.SeriesAt):
            decl = self._series(node.name, loc)
            if not decl.has_time:
                raise BuildError(
                    f"series {node.name} has no time dimension; read it as "
                    f"self.series.{node.name}(), not at a month",
                    loc,
                )
            self._int_index(node.index, loc, f"series.{node.name}.at(...)")
            return decl.unit
        if isinstance(node, ir.RunConst):
            return units.INT
        if isinstance(node, ir.Unary):
            return units.unary_unit(node.op, self.unit_of(node.a), loc)
        if isinstance(node, ir.Binary):
            return units.binary_unit(node.op, self.unit_of(node.a), self.unit_of(node.b), loc)
        if isinstance(node, ir.Where):
            return units.where_unit(
                self.unit_of(node.cond), self.unit_of(node.a), self.unit_of(node.b), loc
            )
        if isinstance(node, ir.Call):
            args = tuple(self.unit_of(a) for a in node.args)
            return units.call_unit(node.fn, args, loc)
        raise BuildError(f"unknown node {type(node).__name__}", loc)

    def _var(self, name: str, loc: Loc | None) -> ir.Variable:
        try:
            return self.vars[name]
        except KeyError:
            raise BuildError(f"no variable {name!r}", loc) from None

    def _series(self, name: str, loc: Loc | None) -> ir.SeriesDecl:
        try:
            return self.graph.series[name]
        except KeyError:
            raise BuildError(f"no series {name!r}", loc) from None

    def _int_index(self, index: ir.Node, loc: Loc | None, what: str) -> None:
        unit = self.unit_of(index)
        if unit not in units.INT_LIKE:
            raise UnitError(f"the month in {what} is an Int, not a {unit}", loc)

    def _lookup_unit(self, node: ir.TableLookup) -> units.Unit:
        decl = self.graph.tables[node.table]
        loc = node.loc
        for key, key_node in zip(decl.keys, node.keys, strict=True):
            unit = self.unit_of(key_node)
            if key.enum is not None:
                if unit != units.ENUM:
                    raise UnitError(
                        f"key {key.name} of table {node.table} is an enum "
                        f"({', '.join(key.enum)}); the lookup passes a {unit}",
                        loc,
                    )
                if isinstance(key_node, ir.Const) and key_node.value not in key.enum:
                    raise BuildError(
                        f"{key_node.value!r} is not a value of key {key.name} of table "
                        f"{node.table} ({', '.join(key.enum)})",
                        loc,
                    )
                if isinstance(key_node, ir.MpField):
                    f = self.graph.fields[key_node.name]
                    if f.enum is None or set(f.enum) != set(key.enum):
                        raise BuildError(
                            f"mp.{key_node.name} does not have the same enum values as key "
                            f"{key.name} of table {node.table}",
                            loc,
                        )
            elif unit not in units.INT_LIKE:
                raise UnitError(
                    f"key {key.name} of table {node.table} is an Int in [{key.lo}, {key.hi}]; "
                    f"the lookup passes a {unit}",
                    loc,
                )
        column = next(c for c in decl.columns if c.name == node.column)
        return column.unit

    # ------------------------------------------------------------------ dims per node
    def dims_of(self, node: ir.Node, dims: dict[str, Dims]) -> Dims:
        result = self._dims_of(node, dims)
        node.dims = result
        return result

    def _dims_of(self, node: ir.Node, dims: dict[str, Dims]) -> Dims:
        if isinstance(node, ir.Const):
            return frozenset()
        if isinstance(node, ir.TimeIndex):
            return frozenset({ir.TIME})
        if isinstance(node, ir.VarRead):
            return dims[node.name]
        if isinstance(node, ir.VarAt):
            return (dims[node.name] - {ir.TIME}) | self.dims_of(node.index, dims)
        if isinstance(node, ir.SumT):
            return dims[node.name] - {ir.TIME}
        if isinstance(node, ir.MpField):
            return frozenset({ir.POLICY})
        if isinstance(node, ir.TableLookup):
            out: Dims = frozenset()
            for k in node.keys:
                out = out | self.dims_of(k, dims)
            return out
        if isinstance(node, ir.ScenRead):
            return frozenset({ir.SCENARIO, ir.TIME})
        if isinstance(node, ir.ScenAt):
            return frozenset({ir.SCENARIO}) | self.dims_of(node.index, dims)
        if isinstance(node, ir.SeriesRead):
            # fixed by the declaration, never by J: the scenario axis is there after a mean too;
            # the units pass has made the offset agree with the declaration's time dimension
            group = frozenset({ir.POLICY, ir.SCENARIO})
            return group if node.offset is None else group | {ir.TIME}
        if isinstance(node, ir.SeriesAt):
            return frozenset({ir.POLICY, ir.SCENARIO}) | self.dims_of(node.index, dims)
        if isinstance(node, ir.RunConst):
            return frozenset()
        out = frozenset()
        for child in node.children():
            out = out | self.dims_of(child, dims)
        return out


# ----------------------------------------------------------------------------- components


@dataclass
class _Edge:
    reader: str
    target: str
    offset: int | None  # None: complete dependency (at, sum_t)
    node: ir.Node


@dataclass
class _Scc:
    members: list[str]
    edges: list[_Edge] = field(default_factory=lambda: [])  # internal edges
    direction: str = ir.NONE


def _tarjan(names: list[str], succ: dict[str, list[str]]) -> list[list[str]]:
    """Strongly connected components, in reverse topological order of the condensation."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    out: list[list[str]] = []
    counter = 0

    def visit(v: str) -> None:
        nonlocal counter
        index[v] = low[v] = counter
        counter += 1
        stack.append(v)
        on_stack.add(v)
        for w in succ.get(v, ()):
            if w not in index:
                visit(w)
                low[v] = min(low[v], low[w])
            elif w in on_stack:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp: list[str] = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                comp.append(w)
                if w == v:
                    break
            out.append(comp)

    for name in names:
        if name not in index:
            visit(name)
    return out


# ------------------------------------------------------------------------------------ check


def check(graph: ir.Graph) -> ir.Graph:
    """Check the graph in place and return it. Raises :class:`BuildError` on the first rule
    broken; warns :class:`BuildWarning` for unused variables and unread group series."""
    chk = _Checker(graph)
    variables = graph.variables
    order = list(variables)

    _check_units(chk, variables)
    _check_roles(graph)
    _check_dims(chk, variables)
    edges: list[_Edge] = _edges(graph)
    _check_domains(variables, edges)
    _check_series_domains(graph)

    # strongly connected components, direction and cycles
    succ: dict[str, list[str]] = {name: [] for name in order}
    for e in edges:
        succ[e.reader].append(e.target)
    sccs = [_Scc(sorted(c, key=order.index)) for c in _tarjan(order, succ)]
    by_name = {name: i for i, scc in enumerate(sccs) for name in scc.members}
    for e in edges:
        if by_name[e.reader] == by_name[e.target]:
            sccs[by_name[e.reader]].edges.append(e)
    for scc in sccs:
        _classify(scc, variables)

    # schedule order: condensation in topological order, members in same-period order
    graph.components = []
    done: set[int] = set()
    remaining = list(range(len(sccs)))
    while remaining:
        progressed = False
        for i in list(remaining):
            deps = {
                by_name[e.target]
                for e in edges
                if by_name[e.reader] == i and by_name[e.target] != i
            }
            if deps <= done:
                scc = sccs[i]
                members = _same_period_order(scc)
                for name in members:
                    variables[name].component = len(graph.components)
                    variables[name].direction = scc.direction
                graph.components.append(members)
                done.add(i)
                remaining.remove(i)
                progressed = True
        if not progressed:  # pragma: no cover - Tarjan guarantees a DAG
            raise BuildError("internal error: the component graph has a cycle")

    _warn_unused(graph)
    _warn_unread_series(graph)
    return graph


def _check_units(chk: _Checker, variables: dict[str, ir.Variable]) -> None:
    for v in variables.values():
        inferred = chk.unit_of(v.expr)
        if not units.declared_unit_ok(v.declared_unit, inferred):
            raise UnitError(
                f"{v.name} is declared {v.declared_unit} but its formula is {inferred}", v.loc
            )
        v.unit = v.declared_unit
        for what, node in (("init", v.init), ("terminal", v.terminal)):
            if node is None:
                continue
            bunit = chk.unit_of(node)
            if not units.declared_unit_ok(v.declared_unit, bunit):
                raise UnitError(
                    f"the {what} value of {v.name} is {bunit}, not {v.declared_unit}", v.loc
                )
            for n in ir.walk(node):
                if _relative_to_time(n, variables):
                    raise BuildError(
                        f"the {what} value of {v.name} reads {ir.render(n)}; a boundary value is "
                        "fixed before the months run, from the model point, tables, run "
                        "constants, variables without a time dimension and fixed months "
                        "(at(x, k), scen.x.at(k), series.x.at(k) with a constant k)",
                        v.loc,
                    )


def _relative_to_time(node: ir.Node, variables: dict[str, ir.Variable]) -> bool:
    """Whether a node may not form a boundary value because its value moves with ``t``. An
    allow-list: fields, tables, run constants, constants, ``sum_t``, arithmetic, ``where`` and
    calls are fixed; a variable read is fixed when the variable has no time dimension; ``at``,
    ``scen.x.at`` and ``series.x.at`` are fixed when their month is a constant. Everything
    else, ``t`` itself, a relative read of a scenario input or of a group series, and any node
    kind added later, is rejected until it is admitted here on purpose."""
    if isinstance(
        node,
        (
            ir.Const,
            ir.MpField,
            ir.TableLookup,
            ir.RunConst,
            ir.SumT,
            ir.Unary,
            ir.Binary,
            ir.Where,
            ir.Call,
        ),
    ):
        return False
    if isinstance(node, ir.VarRead):
        return variables[node.name].has_time
    if isinstance(node, (ir.VarAt, ir.ScenAt, ir.SeriesAt)):
        return not isinstance(node.index, ir.Const)
    return True


def _check_roles(graph: ir.Graph) -> None:
    by_role: dict[str, list[ir.Variable]] = {}
    for v in graph.variables.values():
        if v.role is not None:
            by_role.setdefault(v.role, []).append(v)
    for role in ("in_force", "reserve"):
        if len(by_role.get(role, [])) > 1:
            names = ", ".join(v.name for v in by_role[role])
            raise BuildError(f"only one variable carries role={role!r}; found {names}")
    wanted = {"in_force": units.COUNT, "decrement": units.COUNT, "reserve": units.MONEY}
    for role, vs in by_role.items():
        for v in vs:
            if v.declared_unit != wanted[role]:
                raise BuildError(
                    f"{v.name} has role={role!r} and must be {wanted[role]}, not {v.declared_unit}",
                    v.loc,
                )
            if not v.has_time:
                raise BuildError(f"{v.name} has role={role!r} and needs a time dimension", v.loc)


def _check_dims(chk: _Checker, variables: dict[str, ir.Variable]) -> None:
    dims: dict[str, Dims] = {
        name: frozenset({ir.TIME}) if v.has_time else frozenset() for name, v in variables.items()
    }
    changed = True
    while changed:  # dims only grow, so this converges
        changed = False
        for name, v in variables.items():
            d = chk.dims_of(v.expr, dims)
            for node in (v.init, v.terminal):
                if node is not None:
                    d = d | chk.dims_of(node, dims)
            if v.has_time:
                d = d | {ir.TIME}
            if d != dims[name]:
                dims[name] = d
                changed = True
    for name, v in variables.items():
        if not v.has_time and ir.TIME in dims[name]:
            culprit = next(
                ir.render(n)
                for n in ir.walk(v.expr)
                if isinstance(n, (ir.VarRead, ir.ScenRead, ir.SeriesRead, ir.TimeIndex))
                and ir.TIME in (n.dims or frozenset())
            )
            raise BuildError(
                f"{name} has no time dimension (def {name}(self)) but reads {culprit}; read a "
                f"fixed month with at() or a total with sum_t(), or give {name} a t parameter",
                v.loc,
            )
        v.dims = dims[name]


def _edges(graph: ir.Graph) -> list[_Edge]:
    variables = graph.variables
    edges: list[_Edge] = []
    for v in variables.values():
        for node, target, offset in graph.reads(v):
            tv = variables[target]
            if offset is not None and not tv.has_time and offset != 0:
                raise BuildError(f"{target} has no time dimension", node.loc)
            if offset is not None and tv.has_time and not v.has_time:
                raise BuildError(
                    f"{v.name} has no time dimension but reads {ir.render(node)}; use "
                    f"at(self.{target}, k) or sum_t(self.{target})",
                    node.loc,
                )
            edges.append(_Edge(v.name, target, offset, node))
    return edges


def _formula_first(v: ir.Variable) -> int:
    """The first month a variable's formula is evaluated for."""
    return 1 if v.init is not None else v.start


def _check_domains(variables: dict[str, ir.Variable], edges: list[_Edge]) -> None:
    for v in variables.values():
        is_state = v.init is not None or v.terminal is not None or v.state
        v.start = 0 if is_state or not v.has_time else 1
    demands: list[tuple[str, str, ir.Node, int]] = []  # reader, target, node, offset or at-k
    changed = True
    while changed:
        changed = False
        for e in edges:
            reader, target = variables[e.reader], variables[e.target]
            if not reader.has_time or not target.has_time:
                continue
            if e.offset is None:
                node = e.node
                if isinstance(node, ir.VarAt) and isinstance(node.index, ir.Const):
                    if isinstance(node.index.value, bool):
                        continue
                    need = int(node.index.value)
                else:
                    continue
            else:
                need = _formula_first(reader) + e.offset
            if need < target.start:
                if need < 0:
                    if e.reader == e.target:
                        raise BuildError(
                            f"{target.name} reads itself at {ir.render(e.node)} and so needs an "
                            "init value: @var(init=...)",
                            target.loc,
                        )
                    raise BuildError(
                        f"{reader.name} reads {ir.render(e.node)}, which is before t = 0 at "
                        f"t = {_formula_first(reader)}; a recursive variable needs an init "
                        "value, and the reference models read at most one month back",
                        e.node.loc,
                    )
                target.start = 0
                demands.append((reader.name, target.name, e.node, need))
                changed = True
        for v in variables.values():  # scalar readers of at(x, 0)
            if v.has_time:
                continue
            for node in ir.walk(v.expr):
                if (
                    isinstance(node, ir.VarAt)
                    and isinstance(node.index, ir.Const)
                    and not isinstance(node.index.value, bool)
                ):
                    target = variables[node.name]
                    need = int(node.index.value)
                    if need < 0:
                        raise BuildError(f"at({node.name}, {need}) reads before t = 0", node.loc)
                    if need < target.start:
                        target.start = 0
                        demands.append((v.name, node.name, node, need))
                        changed = True
    # a demanded state must be computable at t = 0: its reads at offset 0 must be states too
    # (handled by the loop above, which lowers their start) unless they are month quantities
    # by declaration
    for v in variables.values():
        if v.has_time and v.start == 0 and v.domain == "month":
            why = next((d for d in demands if d[1] == v.name), None)
            if why is not None:
                raise BuildError(
                    f"{v.name} declares domain='month' but {why[0]} needs it at t = 0 "
                    f"({ir.render(why[2])})",
                    why[2].loc,
                )
    # every relative read stays inside the target's domain
    for e in edges:
        if e.offset is None:
            continue
        reader, target = variables[e.reader], variables[e.target]
        if not reader.has_time or not target.has_time:
            continue
        first, last = reader.formula_range
        if first + e.offset < target.start:  # pragma: no cover - lowered above or errored
            raise BuildError(
                f"{reader.name} reads {ir.render(e.node)} outside the domain of {target.name}",
                e.node.loc,
            )
        max_offset = 0 if last == "H" else 1
        if e.offset > max_offset:
            raise BuildError(
                f"{reader.name} reads {ir.render(e.node)}, which is outside the domain of "
                f"{target.name} at t = {last}; a variable that reads a later month is backward "
                "recursive and declares a terminal value",
                e.node.loc,
            )


def _check_series_domains(graph: ir.Graph) -> None:
    """Group series reads stay inside the series' domain, as variable reads do: with ``(first,
    last)`` the reader's formula range, a relative read needs ``first + offset >= start`` and
    ``offset <= 0`` (``1`` from a formula that stops at ``H - 1``, a backward recursion's), and a
    fixed month ``at(k)`` needs ``k >= start``. A series is an input, so a read before its start
    is an error rather than a demand that lowers it: a sum of states declares ``start=0`` on its
    ``GroupSum``. Months beyond the horizon of a fixed read are checked when the run loads."""
    for v in graph.variables.values():
        for expr in (v.expr, v.init, v.terminal):
            if expr is None:
                continue
            for node in ir.walk(expr):
                if isinstance(node, ir.SeriesRead):
                    if node.offset is not None and v.has_time and expr is v.expr:
                        _check_series_offset(v, node, node.offset, graph.series[node.name])
                elif (
                    isinstance(node, ir.SeriesAt)
                    and isinstance(node.index, ir.Const)
                    and not isinstance(node.index.value, bool)
                ):
                    k = int(node.index.value)
                    decl = graph.series[node.name]
                    if k < 0:
                        raise BuildError(
                            f"{v.name} reads {ir.render(node)}, which is before t = 0", node.loc
                        )
                    if k < decl.start:
                        raise BuildError(
                            f"{v.name} reads {ir.render(node)}, which is before the start of "
                            f"series {node.name} at t = {decl.start}; a sum of states declares "
                            "GroupSum(..., start=0)",
                            node.loc,
                        )


def _check_series_offset(
    v: ir.Variable, node: ir.SeriesRead, offset: int, decl: ir.SeriesDecl
) -> None:
    first, last = v.formula_range
    need = first + offset
    if need < decl.start:
        if need < 0:
            raise BuildError(
                f"{v.name} reads {ir.render(node)}, which is before t = 0 at t = {first}; a "
                "series starts at t = 0 at the earliest",
                node.loc,
            )
        raise BuildError(
            f"{v.name} reads {ir.render(node)}, which is outside the domain of series "
            f"{node.name} at t = {first}: the series starts at t = {decl.start}; a sum of states "
            "declares GroupSum(..., start=0)",
            node.loc,
        )
    max_offset = 0 if last == "H" else 1
    if offset > max_offset:
        raise BuildError(
            f"{v.name} reads {ir.render(node)}, which is outside the domain of series "
            f"{node.name} at t = {last}; a variable that reads a later month is backward "
            "recursive and declares a terminal value",
            node.loc,
        )


def _classify(scc: _Scc, variables: dict[str, ir.Variable]) -> None:
    recursive = len(scc.members) > 1 or any(e.reader == e.target for e in scc.edges)
    if not recursive:
        return
    complete = [e for e in scc.edges if e.offset is None]
    if complete:
        e = complete[0]
        boundary = _boundary_of(variables[e.reader], e.node)
        if boundary is not None:
            # conservative: at(y, 0) of a member of the same component is rejected even when
            # y(0) would be computable, because the boundary is written before the loop runs
            raise BuildError(
                f"the {boundary} value of {e.reader} reads {ir.render(e.node)}, and {e.target} "
                f"depends on {e.reader}: compute the boundary from a variable outside the "
                "component",
                e.node.loc,
            )
        raise BuildError(
            f"{e.reader} reads all of {e.target} ({ir.render(e.node)}), and {e.target} depends "
            f"on {e.reader}: a complete read cannot be part of a recursion",
            e.node.loc,
        )
    offsets = [e.offset for e in scc.edges if e.offset is not None]
    lo, hi = min(offsets), max(offsets)
    members = ", ".join(scc.members)
    if lo < 0 < hi:
        back = next(e for e in scc.edges if e.offset is not None and e.offset < 0)
        fwd = next(e for e in scc.edges if e.offset is not None and e.offset > 0)
        raise BuildError(
            f"the component {{{members}}} mixes forward and backward references: "
            f"{back.reader} reads {ir.render(back.node)} and {fwd.reader} reads "
            f"{ir.render(fwd.node)} (D13: mixed loops are rejected)",
            back.node.loc,
        )
    if lo == 0 and hi == 0:
        e = scc.edges[0]
        raise BuildError(
            f"same-period cycle: {{{members}}} depend on each other within month t; one of "
            "them must read the others at t-1 or t+1",
            e.node.loc,
        )
    # a single-sign component may still hide a cycle of offset-0 edges
    zero_succ: dict[str, list[str]] = {m: [] for m in scc.members}
    for e in scc.edges:
        if e.offset == 0:
            zero_succ[e.reader].append(e.target)
    for comp in _tarjan(scc.members, zero_succ):
        if len(comp) > 1 or comp[0] in zero_succ[comp[0]]:
            e = next(
                x for x in scc.edges if x.offset == 0 and x.reader in comp and x.target in comp
            )
            raise BuildError(
                f"same-period cycle: {{{', '.join(sorted(comp, key=scc.members.index))}}} "
                "depend on each other within month t",
                e.node.loc,
            )
    scc.direction = ir.FORWARD if hi <= 0 else ir.BACKWARD
    # A member read at +k needs no terminal of its own: _check_domains has placed every +k read
    # at a month <= H inside the target's domain (its formula reaches H, or its terminal sits
    # there), and the offset-0 acyclicity check above orders the members without a terminal so
    # that the backend evaluates them at t = H before the loop (the annuity's {bel, inv_exp}).
    for e in scc.edges:
        target = variables[e.target]
        if e.offset is not None and e.offset < 0 and target.init is None:
            raise BuildError(
                f"{e.target} is forward recursive ({e.reader} reads {ir.render(e.node)}) and "
                f"needs an init value: @var(init=...)",
                target.loc,
            )
    for name in scc.members:
        v = variables[name]
        if scc.direction == ir.FORWARD and v.terminal is not None:
            raise BuildError(
                f"{name} has a terminal value but sits in the forward component {{{members}}}",
                v.loc,
            )
        if scc.direction == ir.BACKWARD and v.init is not None:
            raise BuildError(
                f"{name} has an init value but sits in the backward component {{{members}}}",
                v.loc,
            )


def _boundary_of(v: ir.Variable, node: ir.Node) -> str | None:
    """``"init"`` or ``"terminal"`` when ``node`` sits in that boundary value of ``v``, else
    ``None`` (it sits in the formula)."""
    for what, expr in (("init", v.init), ("terminal", v.terminal)):
        if expr is not None and any(n is node for n in ir.walk(expr)):
            return what
    return None


def _same_period_order(scc: _Scc) -> list[str]:
    """Members ordered so that every offset-0 read comes after the variable it reads."""
    members = scc.members
    zero_deps: dict[str, set[str]] = {m: set() for m in members}
    for e in scc.edges:
        if e.offset == 0 and e.reader != e.target:
            zero_deps[e.reader].add(e.target)
    out: list[str] = []
    while len(out) < len(members):
        for m in members:
            if m not in out and zero_deps[m] <= set(out):
                out.append(m)
                break
        else:  # pragma: no cover - the cycle check above rules this out
            raise BuildError(f"internal error: no same-period order for {members}")
    return out


def needed_variables(graph: ir.Graph) -> set[str]:
    """Outputs, role variables, the variables the product's invariants read, and everything they
    read, transitively."""
    wanted: set[str] = set(graph.outputs) if graph.outputs is not None else set(graph.variables)
    wanted |= {v.name for v in graph.variables.values() if v.role is not None}
    wanted |= graph.invariant_reads
    stack = list(wanted)
    while stack:
        name = stack.pop()
        for _node, target, _offset in graph.reads(graph.variables[name]):
            if target not in wanted:
                wanted.add(target)
                stack.append(target)
    return wanted


def _warn_unused(graph: ir.Graph) -> None:
    needed = needed_variables(graph)
    for name, v in graph.variables.items():
        if name not in needed:
            warnings.warn(
                BuildWarning(
                    f"{name} is read by no other variable and is not an output; it is left out "
                    f"of the generated code ({v.loc})"
                ),
                stacklevel=3,
            )


def _warn_unread_series(graph: ir.Graph) -> None:
    """A group series the stage declares and no formula reads. Its declaration is still the
    stage's input contract, so the run asks for it; the warning says so, as for an unused
    variable."""
    read = {node.name for _v, node in graph.series_reads()}
    for name, decl in graph.series.items():
        if name not in read:
            warnings.warn(
                BuildWarning(
                    f"series {name} is read by no formula; a run of the stage still needs it "
                    f"given ({decl.loc})"
                ),
                stacklevel=3,
            )
