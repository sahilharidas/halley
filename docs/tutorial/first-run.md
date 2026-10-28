# A first run

Halley ships with a complete model, a level term assurance, and with the inputs its
specification uses as a worked example. Running those is the quickest way to see what a run
is.

## Install

```bash
pip install halley
```

Python 3.12 or later. The package brings NumPy and Arrow with it; nothing else is needed for
this tutorial.

## Following along

Work in a file, not at the `>>>` prompt or in a notebook cell: Halley reads your model's source
to point its messages at your formulas, and a class typed at the prompt has no source file, so
`build` stops with `OSError: source code not available`. Make a folder for the tutorial, create
`tutorial.py` in it and paste each page's Python blocks into that file in order; run it with
`python tutorial.py` from that folder, or with `python -i tutorial.py` to be left at a prompt
with `result` and the other names in hand. Everything the pages write, the `runs/...`
directories, a CSV and two Parquet files, lands in that folder. A run refuses a directory that
already holds a run, so delete the `runs` folder before running the file again. A message that
points at a formula names the file and line it is on: where these pages show `term.py:4`, you
will see `tutorial.py` and your own line number.

## Build the model and run it

Building a model traces its formulas, checks them and generates the code that will run them.
Running it loads the inputs, projects every policy and writes a run directory.

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
    run_dir="runs/first-run",
)
print(result.n_policies, "policies,", result.H, "months")
```

```text
10 policies, 480 months
```

The ten model points and the five assumption tables are built in memory from the formulas in
the specification, so this works from the installed package alone. The valuation date is the
specification's, 1 January 2027.

A run writes to `run_dir` and refuses a directory that already holds a run, so give each run a
new name. The next pages keep using `runs/first-run`.

## Read a number out

Every variable of every policy is kept for every month. `lane` gives one policy's values over
time; index 0 is the valuation date.

```python
reserve = result.lane("reserve", "P05")
print(round(float(reserve[0]), 2))
```

```text
537.96
```

That is the reserve of policy P05 at the valuation date, and it is the number the specification's
worked example prints. Policy P05 is a 51-year-old man with a 200,000 sum assured and a little
over a year of cover left.

## What just happened

`build` ran the model's formulas once with a symbolic month `t` to capture every variable and
what it reads, checked the units and time references of every formula, worked out the order to
calculate the variables in and which run forwards and which backwards, and generated NumPy code
for the whole batch. `run` then checked the inputs against the model's declared schema, ran the
generated code over all ten policies at once, checked the built-in invariants on the results and
wrote everything to `runs/first-run`. The next page looks inside that directory.
