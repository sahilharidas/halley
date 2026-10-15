# Halley

Actuarial projection engine in Python. Models are checked before they run and compiled when they do.

A model is a Python class. Building it traces the formulas into a typed dependency graph, checks
units, time references and calculation order, then generates vectorised code that runs over every
policy and scenario. Every run writes a manifest so any result can be reproduced exactly.

Status: prototype, milestone M0 (repository skeleton). Nothing runs a model yet.

Documentation: <https://halley.kndra.co.za>. Install with `pip install halley` (Python 3.12+).

## Development

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is pinned in `.python-version`.

```bash
uv sync
uv run pytest
uv run ruff check .
uv run pyright
```

## Layout

| Path | Role |
|---|---|
| `src/halley/authoring` | `Product`, `@var`, `where()`, `Static[...]`, unit types |
| `src/halley/tracing` | runs model methods once with symbolic `t` to capture the graph |
| `src/halley/ir` | typed graph: variables, time offsets, dimensions, source locations |
| `src/halley/checker` | units, time references, recursion direction, cycles, data contracts |
| `src/halley/scheduler` | orders strongly connected components, assigns time direction |
| `src/halley/codegen` | NumPy reference backend, then Numba |
| `src/halley/runtime` | data loading, batches, outputs, run manifest |
| `src/halley/aggregation` | group-level stage; IFRS 17 CSM roll-forward first |

## Licence

Apache 2.0. See `LICENSE`.
