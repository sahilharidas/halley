# Reading a run

This page continues from the [first run](first-run.md), with `result` still in hand.

## The run directory

```python
for path in sorted(p for p in result.run_dir.rglob("*") if p.is_file()):
    print(path.relative_to(result.run_dir).as_posix())
```

```text
generated/numpy_backend.py
inputs/model_points.parquet
inputs/tables/comm_table.parquet
inputs/tables/expenses.parquet
inputs/tables/lapse_table.parquet
inputs/tables/mort_table.parquet
inputs/tables/rates.parquet
invariants.json
manifest.json
outputs/monthly.parquet
outputs/scalars.parquet
```

- `inputs/` holds a copy of every input exactly as it was read. A run can always be repeated
  from its own directory.
- `manifest.json` records the model, the hash of every input, the backend and its versions, the
  settings and a run id computed from the model's formulas, the input hashes, the settings that
  change the numbers and the backend with its versions. Two runs with the same inputs and
  settings get the same id, wherever the files were.
- `outputs/` holds the variables the model names in its `outputs` (every variable, when it
  names none), keyed on the policy id, as Parquet files you can open with any Arrow or pandas
  tool. `monthly.parquet` has one row per policy, scenario and month; `scalars.parquet` has the
  variables without a time dimension.
- `generated/numpy_backend.py` is the code that ran. You never need to read it, but it is there.
- `invariants.json` records the checks the run made on its own results.

## The outputs

`result` keeps every variable the run computed, whether or not the run directory stores it.

```python
print(len(result.variables), "variables, for example", result.variables[:3])
pols_if = result.lane("pols_if", "P05")
print([round(float(x), 4) for x in pols_if[:4]], "...", round(float(pols_if[-1]), 4))
```

```text
49 variables, for example ['age', 'claim_exp_pp', 'claim_expense']
[1.0, 0.9964, 0.9928, 0.9926] ... 0.0
```

Policy P05 starts as one policy in force, loses a little to death and lapse each month, and has
nothing in force after its cover ends. The lane has 481 entries: month 0 and the 480 months of
the run's horizon, which is the longest remaining term in the batch.

## The invariants

A run checks its own results. Some checks are built in, from the roles the model gives its
variables: in-force counts never go negative or increase, decrements never exceed what was in
force, probabilities stay between 0 and 1, and a reserve reaches its terminal value. Others are
declared by the model: here, that the reserve rolls forward month by month and equals the
present value of the future net outgo.

```python
for check in result.invariants:
    print(check.passed, check.name)
```

```text
True H1 in force never negative, never increasing
True H2 decrements never exceed the opening in-force count
True H2 probabilities in [0, 1]
True H3 reserve reaches its terminal value at t = 480
True H4 reserve roll-forward
True H5a reserve(0) = -pv_net_cf
True H5b reserve(t) disc(t) = PV of the net outgo of months t+1..H
```

A failing check is an error at the end of the run, naming the check, the policy and the month.

## Explaining a value

`explain` takes a run, a policy, a variable and a month, and shows the value, the formula that
produced it, where that formula is defined and the value of every input it read.

```python
from halley import explain

print(explain(result, "P05", "reserve", 13))
```

```text
reserve at t = 13 for policy P05 = 46.35078406367477 [Money]
  defined at TermAssurance.reserve (.../halley/products/term_assurance.py:461)
    @var(terminal=0, role="reserve")
    def reserve(self, t: Int) -> Money:
        return (self.reserve(t + 1) + self.net_outgo(t + 1)) * self.v(t + 1)
  inputs:
    reserve(t+1) = 0.0
    net_outgo(t+1) = 46.502524469400136
    v(t+1) = 0.9967369426185624
```

At month 13 policy P05 has one month of cover left, so the reserve is the next month's net outgo
discounted for one month, and the reserve after that is zero. Each input named here can be
explained in turn, down to the model point fields and the table lookups.

## Reading a run back

A run directory is complete in itself. `load_run` reads one back without the model, with the
variables the directory stores: `TermAssurance` names 25 of its 49 as `outputs`. A model that
names none, like the one on the next page, stores every variable.

```python
from halley import load_run

again = load_run("runs/first-run")
print(round(float(again.lane("reserve", "P05")[0]), 2), again.run_id == result.run_id)
print(len(again.variables), "stored variables, for example", again.variables[:3])
```

```text
537.96 True
25 stored variables, for example ['claim_expense', 'commission', 'commission_pp']
```

## Comparing two runs

`diff` compares two runs of the same policies variable by variable and reports the largest
difference in each. Here the same model runs again with interest at 5% instead of 4%. The
`rates` table has two rows, the interest rate (4%) and the expense inflation rate (3%);
`set_column` replaces its `rate` column with 5% and the same 3%, the Arrow way of overtyping
two cells.

```python
import pyarrow as pa

from halley import diff

tables = reference_tables()
rates = tables["rates"]
tables["rates"] = rates.set_column(
    rates.schema.get_field_index("rate"),
    "rate",
    pa.array([0.05, 0.03]),  # interest 5%, expense inflation still 3%
)
other = run(
    compiled,
    model_points=reference_model_points(),
    tables=tables,
    settings=RunSettings(valuation_date=VALUATION_DATE),
    run_dir="runs/five-percent",
)
change = diff(result, other)["reserve"]
print(change.policy_id, change.t, round(change.a, 2), round(change.b, 2))
```

```text
P09 148 3987.65 3194.08
```

The reserve that moves most is policy P09's, forty years of cover on a large sum assured, at
month 148: 3,987.65 at 4% against 3,194.08 at 5%.
