"""Intermediate representation.

A typed graph: one :class:`Variable` per ``@var`` of the model, each with a formula made of the
nodes below, plus the data contracts the model declares. Nodes record the source location they
were traced from, so every later error points at the model, never at generated code.

The checker fills the fields marked *checked* (units, dimensions, domains, components); the
tracer leaves them empty.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass, field

from halley.errors import Loc
from halley.ir.units import Unit

Dims = frozenset[str]
"""A subset of ``{"time", "policy", "scenario"}``."""

TIME = "time"
POLICY = "policy"
SCENARIO = "scenario"

# ----------------------------------------------------------------------------------- nodes


@dataclass(eq=False, slots=True)
class Node:
    """Base class. Equality is identity; a node appears once in one formula."""

    loc: Loc | None
    unit: Unit | None = field(default=None, init=False)  # checked
    dims: Dims | None = field(default=None, init=False)  # checked

    def children(self) -> tuple[Node, ...]:
        return ()


@dataclass(eq=False, slots=True)
class Const(Node):
    """A literal: a number, a build-time value, a bool or a string (enum) constant."""

    value: float | int | bool | str


@dataclass(eq=False, slots=True)
class TimeIndex(Node):
    """The month index ``t``."""


@dataclass(eq=False, slots=True)
class VarRead(Node):
    """``x(t + offset)``, or ``x()`` for a variable without a time dimension (offset 0)."""

    name: str
    offset: int


@dataclass(eq=False, slots=True)
class VarAt(Node):
    """``at(x, index)``: the value of a time variable at an absolute month, constant or per
    policy. A complete dependency: the whole of ``x`` is computed first."""

    name: str
    index: Node

    def children(self) -> tuple[Node, ...]:
        return (self.index,)


@dataclass(eq=False, slots=True)
class SumT(Node):
    """``sum_t(x)``: the sum of ``x`` over its domain in ascending ``t``, sequentially (D19).
    A complete dependency."""

    name: str


@dataclass(eq=False, slots=True)
class MpField(Node):
    """A model point field or a load-time derived field."""

    name: str


@dataclass(eq=False, slots=True)
class TableLookup(Node):
    """``tables.<table>[keys]`` or ``tables.<table>.<column>`` for a keyless table."""

    table: str
    column: str
    keys: tuple[Node, ...]

    def children(self) -> tuple[Node, ...]:
        return self.keys


@dataclass(eq=False, slots=True)
class ScenRead(Node):
    """``scen.<name>(t + offset)``: a scenario input at a relative month. Negative months
    before the valuation date are history rows of the scenario file."""

    name: str
    offset: int


@dataclass(eq=False, slots=True)
class ScenAt(Node):
    """``scen.<name>.at(index)``: a scenario input at an absolute month."""

    name: str
    index: Node

    def children(self) -> tuple[Node, ...]:
        return (self.index,)


@dataclass(eq=False, slots=True)
class Unary(Node):
    op: str  # neg, not
    a: Node

    def children(self) -> tuple[Node, ...]:
        return (self.a,)


@dataclass(eq=False, slots=True)
class Binary(Node):
    op: str  # add sub mul truediv floordiv mod pow lt le gt ge eq ne and or
    a: Node
    b: Node

    def children(self) -> tuple[Node, ...]:
        return (self.a, self.b)


@dataclass(eq=False, slots=True)
class Where(Node):
    """``where(cond, a, b)``. Both branches are evaluated; the unused one is discarded."""

    cond: Node
    a: Node
    b: Node

    def children(self) -> tuple[Node, ...]:
        return (self.cond, self.a, self.b)


@dataclass(eq=False, slots=True)
class Call(Node):
    fn: str  # exp log expm1 log1p sqrt minimum maximum abs
    args: tuple[Node, ...]

    def children(self) -> tuple[Node, ...]:
        return self.args


def walk(node: Node) -> Iterator[Node]:
    """Every node of a formula, parents before children."""
    yield node
    for child in node.children():
        yield from walk(child)


# ------------------------------------------------------------------------------- variables


FORWARD = "forward"
BACKWARD = "backward"
NONE = "none"


@dataclass(eq=False, slots=True)
class Variable:
    name: str
    declared_unit: Unit
    has_time: bool
    timing: str | None
    init: Node | None
    terminal: Node | None
    role: str | None
    domain: str | None
    expr: Node
    loc: Loc
    source: str
    doc: str | None
    # checked
    unit: Unit | None = None
    dims: Dims | None = None
    start: int = 0
    """First month of the variable's domain: 0 for a state or boundary variable, 1 for a month
    quantity. The domain ends at the run horizon."""
    component: int = -1
    direction: str = NONE

    @property
    def state(self) -> bool:
        """Declared a state (``domain="state"``): defined from ``t = 0``."""
        return self.domain == "state"

    @property
    def formula_range(self) -> tuple[int, str]:
        """``(first, last)`` months the formula is evaluated for; ``last`` is ``"H"`` or
        ``"H-1"``."""
        if self.terminal is not None:
            return (self.start, "H-1")
        if self.init is not None:
            return (1, "H")
        return (self.start, "H")


# ------------------------------------------------------------------------------- contracts
#
# Declared by the model (``schema``, ``tables``, ``scenarios`` class attributes); validated by
# the loader. They live here so that the graph carries everything a run needs to check.


@dataclass(frozen=True, slots=True)
class FieldDecl:
    """A model point field. ``kind`` is ``number``, ``int``, ``enum``, ``date`` or ``string``."""

    name: str
    kind: str
    unit: Unit | None
    enum: tuple[str, ...] | None = None
    gt: float | None = None
    ge: float | None = None
    lt: float | None = None
    le: float | None = None
    unique: bool = False
    derived: bool = False
    doc: str = ""

    @property
    def readable(self) -> bool:
        """Whether a formula may read the field. Dates and strings are not numbers."""
        return self.kind in ("number", "int", "enum")


@dataclass(frozen=True, slots=True)
class KeyDecl:
    """A table key: an integer in ``[lo, hi]`` or an enum."""

    name: str
    enum: tuple[str, ...] | None = None
    lo: int | None = None
    hi: int | None = None


@dataclass(frozen=True, slots=True)
class ColumnDecl:
    name: str
    unit: Unit
    ge: float | None = None
    le: float | None = None


@dataclass(frozen=True, slots=True)
class TableDecl:
    name: str
    keys: tuple[KeyDecl, ...]
    columns: tuple[ColumnDecl, ...]
    doc: str = ""


@dataclass(frozen=True, slots=True)
class ScenarioDecl:
    name: str
    unit: Unit
    doc: str = ""


@dataclass(slots=True)
class Graph:
    model: str
    source_file: str
    statics: dict[str, object]
    variables: dict[str, Variable]
    fields: dict[str, FieldDecl]
    tables: dict[str, TableDecl]
    scenarios: dict[str, ScenarioDecl]
    horizon_field: str | None
    outputs: tuple[str, ...] | None
    """Variables kept as run outputs; ``None`` keeps them all."""
    scale_fields: dict[Unit, str]
    """Per-unit tolerance scales for invariants, as model point fields (open question 25)."""
    components: list[list[str]] = field(default_factory=lambda: [])
    """Checked: the variables of each component, in schedule order."""

    def reads(self, var: Variable) -> Iterator[tuple[Node, str, int | None]]:
        """Every variable read in ``var``'s formula and boundary values: ``(node, name,
        offset)`` with ``offset`` ``None`` for a complete dependency (``at``, ``sum_t``)."""
        exprs = [var.expr] + [e for e in (var.init, var.terminal) if e is not None]
        for expr in exprs:
            for node in walk(expr):
                if isinstance(node, VarRead):
                    yield node, node.name, node.offset
                elif isinstance(node, (VarAt, SumT)):
                    yield node, node.name, None


SYMBOLS = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "truediv": "/",
    "floordiv": "//",
    "mod": "%",
    "pow": "**",
    "lt": "<",
    "le": "<=",
    "gt": ">",
    "ge": ">=",
    "eq": "==",
    "ne": "!=",
    "and": "&",
    "or": "|",
}


def render(node: Node) -> str:
    """A canonical text form of a formula, for hashes and messages."""
    if isinstance(node, Const):
        return repr(node.value)
    if isinstance(node, TimeIndex):
        return "t"
    if isinstance(node, VarRead):
        if node.offset == 0:
            return f"{node.name}(t)"
        return f"{node.name}(t{node.offset:+d})"
    if isinstance(node, VarAt):
        return f"at({node.name}, {render(node.index)})"
    if isinstance(node, SumT):
        return f"sum_t({node.name})"
    if isinstance(node, MpField):
        return f"mp.{node.name}"
    if isinstance(node, TableLookup):
        keys = ", ".join(render(k) for k in node.keys)
        return f"tables.{node.table}.{node.column}[{keys}]"
    if isinstance(node, ScenRead):
        return f"scen.{node.name}(t{node.offset:+d})" if node.offset else f"scen.{node.name}(t)"
    if isinstance(node, ScenAt):
        return f"scen.{node.name}.at({render(node.index)})"
    if isinstance(node, Unary):
        return f"({'-' if node.op == 'neg' else '~'}{render(node.a)})"
    if isinstance(node, Binary):
        return f"({render(node.a)} {SYMBOLS[node.op]} {render(node.b)})"
    if isinstance(node, Where):
        return f"where({render(node.cond)}, {render(node.a)}, {render(node.b)})"
    if isinstance(node, Call):
        return f"{node.fn}({', '.join(render(a) for a in node.args)})"
    raise TypeError(f"unknown node {type(node).__name__}")


def graph_hash(graph: Graph) -> str:
    """A content hash of the graph: formulas, boundary values, units, statics and contracts.
    The same model source with the same build-time settings gives the same hash."""
    parts: list[str] = [f"model {graph.model}"]
    for name, value in sorted(graph.statics.items()):
        parts.append(f"static {name} = {value!r}")
    for name in sorted(graph.variables):
        v = graph.variables[name]
        parts.append(
            f"var {name}: {v.declared_unit} time={v.has_time} timing={v.timing} "
            f"role={v.role} domain={v.domain}"
        )
        parts.append(f"  = {render(v.expr)}")
        if v.init is not None:
            parts.append(f"  init {render(v.init)}")
        if v.terminal is not None:
            parts.append(f"  terminal {render(v.terminal)}")
    for name in sorted(graph.fields):
        parts.append(f"field {graph.fields[name]!r}")
    for name in sorted(graph.tables):
        parts.append(f"table {graph.tables[name]!r}")
    for name in sorted(graph.scenarios):
        parts.append(f"scenario {graph.scenarios[name]!r}")
    parts.append(f"horizon {graph.horizon_field}")
    parts.append(f"outputs {graph.outputs}")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
