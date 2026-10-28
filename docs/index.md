# Halley

Actuarial projection engine in Python. Models are checked before they run and compiled when they do.

A model is a Python class. Building it traces the formulas into a typed dependency graph, checks
units, time references and calculation order, then generates vectorised code that runs over every
policy and scenario. Every run writes a manifest so any result can be reproduced exactly.

**Status:** prototype, not yet reviewed. The engine core, the reference models and the IFRS 17
aggregation stage are in, each reconciled to its specification. `TermAssurance` reconciles to the
[term assurance specification](spec/term-assurance.md)'s number script under the specification's
tolerance, and the build errors the specification lists for the checker fire as it says.
`BasicTerm` reconciles to lifelib's `BasicTerm_ME` from lifelib's own inputs: every policy's
present values, the sampled policies' monthly flows and the portfolio totals agree under the same
tolerance. The annuities of the [annuity specification](spec/annuity.md) reconcile to its printed
worked example, and the aggregation stage measures the CSM of one group new at the valuation date,
with the revenue and IFIE lines, and the subsequent measurement of an in-force group over a
reporting period, reconciled to the [in-force worked example](spec/in-force-group.md)'s number
script and to its two reference workbooks. The workbooks of reference case 1 and of the in-force
example are built from the specifications by hand and recalculated in Excel. No reviewer has yet
checked a specification or a workbook, so every reconciliation is against the author's own
specification and script. The Numba backend and the worker processes are in; the benchmarks are
not.

## Install

```bash
pip install halley
```

Python 3.12 or later. Licensed under Apache 2.0.

## A first run

```python
from halley import RunSettings, build, run
from halley.products.term_assurance import VALUATION_DATE, TermAssurance
from halley.products.term_assurance_data import reference_model_points, reference_tables

compiled = build(TermAssurance)
result = run(
    compiled,
    model_points=reference_model_points(),
    tables=reference_tables(),
    settings=RunSettings(valuation_date=VALUATION_DATE),
    run_dir="runs/term-reference",
)
print(round(float(result.lane("reserve", "P05")[0]), 2))   # 537.96
```

The inputs are the specification's reference case, built from its formulas by the package, so
this works from `pip install halley` alone. The run directory holds the manifest, a copy of every
input, the generated backend source, the outputs keyed on `policy_id`, and the invariant results.
`halley.explain(result, "P05", "reserve", 13)` shows the value, the formula and every input it
read. The [tutorial](tutorial/index.md) continues from here: reading a run, writing a model,
loading your own data and running scenarios.

```{toctree}
:maxdepth: 2
:caption: Tutorial

tutorial/index
```

```{toctree}
:maxdepth: 2
:caption: Specifications

spec/README
spec/term-assurance
spec/annuity
spec/in-force-group
```

```{toctree}
:maxdepth: 1
:caption: Development

releasing
```
