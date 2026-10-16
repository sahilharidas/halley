# Halley

Actuarial projection engine in Python. Models are checked before they run and compiled when they do.

A model is a Python class. Building it traces the formulas into a typed dependency graph, checks
units, time references and calculation order, then generates vectorised code that runs over every
policy and scenario. Every run writes a manifest so any result can be reproduced exactly.

**Status:** prototype. The engine core and the term assurance reference model are in: the
`TermAssurance` model reconciles to the term assurance specification's number script under the
specification's tolerance, and the build errors the specification lists for the checker fire as it
says. The spreadsheet and the reviewer's check of reference case 1 are still pending.

## Install

```bash
pip install halley
```

Python 3.12 or later. Licensed under Apache 2.0.

## A first run

```python
from halley import RunSettings, build, run
from halley.products.term_assurance import VALUATION_DATE, TermAssurance

compiled = build(TermAssurance)
result = run(
    compiled,
    model_points="docs/spec/term-assurance/model_points.parquet",
    tables={
        name: f"docs/spec/term-assurance/{name}.parquet"
        for name in ("mort_table", "lapse_table", "comm_table", "rates", "expenses")
    },
    settings=RunSettings(valuation_date=VALUATION_DATE),
    run_dir="runs/term-reference",
)
print(result.lane("reserve", "P05")[0])   # 537.9608658387098
```

The run directory holds the manifest, a copy of every input, the generated backend source, the
outputs keyed on `policy_id`, and the invariant results. `halley.explain(result, "P05",
"reserve", 13)` shows the value, the formula and every input it read.

```{toctree}
:maxdepth: 2
:caption: Specifications

spec/README
spec/term-assurance
```
