# Halley

Actuarial projection engine in Python. Models are checked before they run and compiled when they do.

A model is a Python class. Building it traces the formulas into a typed dependency graph, checks
units, time references and calculation order, then generates vectorised code that runs over every
policy and scenario. Every run writes a manifest so any result can be reproduced exactly.

Status: prototype. The engine core (milestone M2) and both reference models are in.
`TermAssurance` (M3) reconciles to the term assurance spec's number script, column for column and
scalar for scalar, under the spec's tolerance, and every deliberate mistake the spec lists for the
checker fails the build the way the spec says. `BasicTerm` (M4) reconciles to lifelib's
`BasicTerm_ME` from lifelib's own inputs: every policy's present values, the sampled policies'
monthly flows and counts, and the portfolio totals agree under the same tolerance. The IFRS 17
measurement of an in-force annuity group over a reporting period is part of v1, not a later
version: its worked example (M10) is specified in `docs/spec/in-force-group.md`, and its
spreadsheet, reviewer's check and engine milestones (M9, M11) are pending. The spreadsheet and the
reviewer's check of reference case 1 (M1) are also still pending, so that reconciliation is
against the spec's own script, not yet against a reviewed spreadsheet.

Documentation: <https://halley.kndra.co.za>. Install with `pip install halley` (Python 3.12+).

## Example

```python
from halley import RunSettings, build, run
from halley.products.term_assurance import VALUATION_DATE, TermAssurance

inputs = "docs/spec/term-assurance"
compiled = build(TermAssurance)  # trace, check, schedule, generate the NumPy backend
result = run(
    compiled,
    model_points=f"{inputs}/model_points.parquet",
    tables={
        name: f"{inputs}/{name}.parquet"
        for name in ("mort_table", "lapse_table", "comm_table", "rates", "expenses")
    },
    settings=RunSettings(valuation_date=VALUATION_DATE),
    run_dir="runs/term-reference",
)
result.lane("reserve", "P05")[0]  # 537.9608658387098, the spec's worked example
```

The run directory holds `manifest.json` (model source hash and git commit, IR hash, the hash and
version of every input, backend versions, settings), a copy of every input, the generated backend
source, the outputs keyed on `policy_id` with a versioned schema, and the invariant results.
`halley.explain` shows any policy's value at any month with its formula and inputs; `halley.diff`
compares two runs; `halley.load_run` reads a run back without the model.

`build(TermAssurance, backend="numba")` generates the Numba backend instead (`pip install
halley[numba]`): compiled loops from the same IR, held to the NumPy reference within 1e-9 relative
and bit-identical between its own reruns. Its first build compiles every kernel, which takes a
while for a product; Numba caches them beside the generated module, so later builds load them.
The cache is `halley-numba-<user>` in the system's temporary directory, one per user and private
to them, since Numba loads the compiled kernels it finds there and on Linux and macOS that
directory is shared; `HALLEY_NUMBA_CACHE` names another.

`RunSettings(workers=4)` spreads the batches (`batch_size` policies each) over four processes on
either backend, with the same results bit for bit as one process at the same batch size: the
same run id, outputs, invariant results and errors. The workers are started with the spawn
method and import the model's class from its module, so define it at the top level of a module
and guard a script's entry point with `if __name__ == "__main__":`.

A model is written like this (an excerpt of `TermAssurance`):

```python
class TermAssurance(Product):
    model_point = SCHEMA                      # typed fields, ranges, derived fields, rules
    assumption_tables = (MORT_TABLE, LAPSE_TABLE, COMM_TABLE, RATES, EXPENSES)

    @var
    def q_m(self, t: Int) -> Prob[Monthly]:
        return where(t <= self.mp.rem_m, to_monthly_prob(self.q_ann(t)), 0)

    @var(role="decrement")
    def deaths(self, t: Int) -> Count:
        return self.pols_if(t - 1) * self.q_m(t)      # Count x Prob[Monthly] -> Count

    @var(init=lambda mp: mp.pols_if_init, role="in_force")
    def pols_if(self, t: Int) -> Count:
        return self.pols_if(t - 1) - self.deaths(t) - self.lapses(t) - self.maturities(t)

    @var(terminal=0, role="reserve")
    def reserve(self, t: Int) -> Money:               # backward recursion
        return (self.reserve(t + 1) + self.net_outgo(t + 1)) * self.v(t + 1)
```

Unit mismatches, data-dependent time offsets, missing boundary values, same-period cycles and
mixed forward/backward loops are build errors that point at the model's own source line. The
unit classes carry typed operator overloads, so pyright flags most unit mistakes in the editor.

## Development

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is pinned in `.python-version`.

```bash
uv sync
uv run pytest
uv run ruff check .
uv run pyright
```

The default `pytest` run includes the Numba backend's differentials (`tests/test_numba.py`,
marked `numba`), which hold it to the NumPy reference (D16). On a cold kernel cache they take
about two and a half minutes, nearly all of it compiling every product's kernels, and about a
minute once Numba's cache holds them, so keep the cache directory (`HALLEY_NUMBA_CACHE`)
between CI runs if you can. `uv run pytest -m "not numba"` leaves them out for a quicker loop
while editing; it is not the run a change passes before it is committed.

`docs/spec/term_assurance_inputs.py` regenerates the reference Parquet inputs from the spec's
formulas; `docs/spec/term_assurance_numbers.py` regenerates every number the spec prints.

## Layout

| Path | Role |
|---|---|
| `src/halley/authoring` | `Product`, `@var`, `where()`, `Static[...]`, unit types, data contracts |
| `src/halley/tracing` | runs model methods once with symbolic `t` to capture the graph |
| `src/halley/ir` | typed graph: variables, time offsets, dimensions, source locations; the unit rule table |
| `src/halley/checker` | units, time references and domains, recursion direction, cycles |
| `src/halley/scheduler` | orders strongly connected components, assigns time direction |
| `src/halley/codegen` | the backends, generated from the IR: the NumPy reference and Numba |
| `src/halley/runtime` | data contracts at load, batches, outputs, run manifest, invariants, explain and diff |
| `src/halley/aggregation` | aggregation stages over group series: portfolio totals and the IFRS 17 CSM roll-forward (NewGroup, InForceGroup) |
| `src/halley/products` | reference models: `TermAssurance` (reference case 1), `BasicTerm` (reference case 2, lifelib), the annuities `LevelAnnuity`, `FixedEscalationAnnuity` and `CPILinkedAnnuity` (the annuity spec) and their reference inputs |
| `docs/spec` | product specifications, their number scripts and the reference inputs |
| `benchmarks` | the Spec's workloads W1 and W2 on each backend (the harness and results come with M6's benchmark stage) |
| `tests/data/lifelib` | lifelib's `BasicTerm_ME` inputs and results (MIT), the oracle for reference case 2 |

## Licence

Apache 2.0. See `LICENSE`.
