"""Scheduler: the order in which components are evaluated and how.

The checker has grouped the variables into strongly connected components, given each a direction
and put them in dependency order. The schedule says, per component, whether it is evaluated for
all months at once (non-recursive) or one month at a time, forward from ``t = 1`` or backward from
the horizon, and leaves out variables that nothing needs.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from halley import ir
from halley.checker import needed_variables

WHOLE = "whole"
STEP = "step"


@dataclass(frozen=True, slots=True)
class Component:
    members: tuple[str, ...]
    direction: str  # ir.FORWARD, ir.BACKWARD or ir.NONE
    mode: str  # WHOLE or STEP

    @property
    def recursive(self) -> bool:
        return self.mode == STEP


@dataclass(frozen=True, slots=True)
class Schedule:
    components: tuple[Component, ...]
    needed: frozenset[str]

    def __iter__(self) -> Iterator[Component]:
        return iter(self.components)


def schedule(graph: ir.Graph) -> Schedule:
    needed = needed_variables(graph)
    comps: list[Component] = []
    for members in graph.components:
        kept = tuple(m for m in members if m in needed)
        if not kept:
            continue
        direction = graph.variables[kept[0]].direction
        recursive = direction != ir.NONE
        comps.append(Component(kept, direction, STEP if recursive else WHOLE))
    return Schedule(tuple(comps), frozenset(needed))
