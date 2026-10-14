# Halley

Actuarial projection engine in Python. Read `README.md` for the layout.

## Design documents

The spec, decisions, roadmap, prior art and open questions live in the Obsidian vault, not here:
`C:\Users\Sahil\Documents\Dhyana\05 🧪 Projects\halley\`. Read `Halley — Spec` and
`Halley — Decisions` before changing architecture. Decisions marked **firm** there are not reopened
without a strong reason.

## Commands

```bash
uv sync
uv run pytest
uv run ruff check .
uv run ruff format .
uv run pyright
```

## Rules

- No employer code, models or data in this repo, including as test fixtures (D29). Test data is
  synthetic or from a public source with a compatible licence (lifelib is MIT).
- Unit and time-reference mistakes in models are build errors, never warnings (D11).
- The NumPy backend is the executable spec; every other backend is tested against it (D16).
- Reruns are bit-identical: fixed-order reductions, fast-math off (D19).
- Errors point at the user's model source, never at generated code.
- Python 3.12 is the development interpreter so Numba stays available later; `requires-python` is `>=3.12`.
