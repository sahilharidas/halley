# Scenarios

This page continues from [your own data](your-own-data.md). It gives `Term` an interest rate
read from a scenario file instead of the fixed 4%, runs it over two scenarios at once and reads
each scenario's reserve. Every run has a scenario dimension; the runs so far had one scenario. A
model reads a value that varies by scenario through a scenario input, and a scenario file
supplies it.

## A scenario input

The model below replaces `Term`'s fixed 4% with an interest rate read from a scenario input for
each month, and discounts the reserve with it.

```python
from halley import Factor, ScenarioInput, explain


class TermByRate(Term):
    scenario_inputs = (ScenarioInput("i", "Rate[Annual]", doc="annual effective interest"),)

    @var
    def v(self, t: Int) -> Factor:
        return (1 + self.scen.i(t)) ** (-1 / 12)

    @var(terminal=0, role="reserve")
    def reserve(self, t: Int) -> Money:
        return (self.reserve(t + 1) + self.death_outgo(t + 1) - self.premium_income(t + 1)) * self.v(t + 1)


by_rate = build(TermByRate)
```

`self.scen.i(t)` reads the input at month `t`. A scenario input is declared with a unit like
anything else, so the rate's unit is checked where it is used.

## A scenario file

A scenario file has three columns: the scenario number, the month and the value. The months must
cover every month the model reads, for every scenario. Here two scenarios hold a flat 4% and a
flat 5% for the 120 months the longer policy runs.

```python
rows = [(s, t, [0.04, 0.05][s]) for s in range(2) for t in range(0, 121)]
rates = pa.table(
    {
        "scenario": pa.array([s for s, _t, _i in rows], pa.int32()),
        "t": pa.array([t for _s, t, _i in rows], pa.int32()),
        "i": pa.array([i for _s, _t, i in rows], pa.float64()),
    }
)
two = run(
    by_rate,
    model_points=points,
    tables={"mort": mort},
    scenarios={"i": rates},
    settings=RunSettings(valuation_date=dt.date(2027, 1, 1)),
    run_dir="runs/two-rates",
)
print(two.n_scenarios, two.var("reserve").shape)
print(round(float(two.lane("reserve", "A", scenario=0)[0]), 2))
print(round(float(two.lane("reserve", "A", scenario=1)[0]), 2))
```

```text
2 (121, 3, 2)
648.35
603.28
```

`var` returns the whole array: 121 months (0 to 120) by 3 policies by 2 scenarios; `lane` picks
one policy and one scenario out of it. Scenario 0 reproduces the fixed-rate reserve of policy A
from the earlier pages, 648.35, and scenario 1 discounts at 5% and gets 603.28. A thousand
scenarios work the same way, with a longer file.

`explain` takes a scenario too:

```python
print(explain(two, "A", "reserve", 0, scenario=1))
```

```text
reserve at t = 0 for policy A = 603.2827235173675 [Money]
  defined at TermByRate.reserve (term.py:11)
    @var(terminal=0, role="reserve")
    def reserve(self, t: Int) -> Money:
        return (self.reserve(t + 1) + self.death_outgo(t + 1) - self.premium_income(t + 1)) * self.v(t + 1)
  inputs:
    reserve(t+1) = 607.6908583405032
    death_outgo(t+1) = 18.049713669010536
    premium_income(t+1) = 20.0
    v(t+1) = 0.9959424073510671
```

## A file that stops short

A scenario file that does not reach every month the model reads is refused before the run
starts, like any other input. Here the same two scenarios stop at month 99.

```python
import pyarrow.compute as pc

try:
    run(
        by_rate,
        model_points=points,
        tables={"mort": mort},
        scenarios={"i": rates.filter(pc.less(rates["t"], 100))},
        settings=RunSettings(valuation_date=dt.date(2027, 1, 1)),
        run_dir="runs/short-scenarios",
    )
except LoadError as error:
    print(error)
```

```text
scenario input i: the model reads months 1 to 120 but the file covers 0 to 99
```
