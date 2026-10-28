# Your own data

This page continues from [your own model](your-own-model.md), with `Term`, `compiled` and the
`mort` table still in hand. It runs the model on model points from a CSV file, shows what the
loader refuses, and writes the inputs as Parquet, the format a run directory keeps.

## Model points from a CSV file

Halley takes its inputs as Arrow tables (`pyarrow`, installed with Halley and imported as `pa`
on the previous page): a table is a sheet with typed columns, held in memory. Arrow reads CSV,
so a spreadsheet saved as CSV is a few lines away. The block below writes the CSV so that the
page can run itself; you would save yours from the spreadsheet, with the columns the schema
declares.

```python
from pathlib import Path

from pyarrow import csv

Path("model_points.csv").write_text(
    "policy_id,age,sum_assured,premium,pols_if_init,rem_m\n"
    "A,35,100000,20,1,120\n"
    "B,50,250000,90,1,60\n"
    "C,62,50000,60,1,36\n"
)
points = csv.read_csv("model_points.csv")
result = run(
    compiled,
    model_points=points,
    tables={"mort": mort},
    settings=RunSettings(valuation_date=dt.date(2027, 1, 1)),
    run_dir="runs/from-csv",
)
print([round(float(result.lane("reserve", pid)[0]), 2) for pid in ("A", "B", "C")])
```

```text
[648.35, 5634.01, 1367.45]
```

The CSV reader infers integer columns; the loader takes integers for number fields and checks
every value against the schema's bounds. A spreadsheet export works the same way: save it as
CSV, or read it with pandas and pass `pyarrow.Table.from_pandas(frame)`.

## What the loader refuses

The schema's bounds are checked before the run starts. A negative sum assured, written over the
column with `set_column`, which replaces one column of a table and keeps the rest:

```python
from halley import LoadError

bad = points.set_column(
    points.schema.get_field_index("sum_assured"),
    "sum_assured",
    pa.array([100_000, -250_000, 50_000]),
)
try:
    run(
        compiled,
        model_points=bad,
        tables={"mort": mort},
        settings=RunSettings(valuation_date=dt.date(2027, 1, 1)),
        run_dir="runs/negative",
    )
except LoadError as error:
    print(error)
```

```text
model points: field sum_assured must be > 0; policies B break this
```

A table must cover the whole key range it declares, so that no lookup can fall off its end
during the projection. A mortality table that stops at age 77 when the model declares ages 18 to
130:

```python
try:
    run(
        compiled,
        model_points=points,
        tables={"mort": mort.slice(0, 60)},
        settings=RunSettings(valuation_date=dt.date(2027, 1, 1)),
        run_dir="runs/short-table",
    )
except LoadError as error:
    print(error)
```

```text
table mort: the rows must cover every key combination exactly once; declared grid has 113 rows, the file 60 rows, 53 combination(s) missing
```

There is no silent extrapolation and no default: the run does not start.

## Parquet files and the manifest

Inputs are usually files. Parquet is the file format Arrow reads and writes for tables: typed,
so an integer column comes back as integers and a table written once is read back exactly, and
compact. pandas and most data tools read it, and the run directory keeps a copy of each input
in it, with its hash.

```python
import pyarrow.parquet as pq

pq.write_table(points, "model_points.parquet")
pq.write_table(mort, "mort.parquet")
from_files = run(
    compiled,
    model_points="model_points.parquet",
    tables={"mort": "mort.parquet"},
    settings=RunSettings(valuation_date=dt.date(2027, 1, 1)),
    run_dir="runs/from-parquet",
)
print(from_files.run_id == result.run_id)
print(from_files.manifest["inputs"]["tables"]["mort"]["copy"])
```

```text
True
inputs/tables/mort.parquet
```

The run id is the same whether the inputs came from files or from memory: it is computed from
the bytes of the Parquet files, not their paths, and the bytes are the same here because the
same pyarrow wrote both with the same options. A Parquet file of the same rows written with
other options, or by another writer, hashes differently and gets another id. The manifest also
records a `sha256` for every input and any `version` the file's metadata carries, so a run can
say exactly which basis it used.
