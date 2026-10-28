# Your own model

This page needs nothing from the earlier ones. It writes a term assurance from nothing: a model
point schema, one assumption table and seven variables. Then it makes the mistakes a first model
usually has, to show what the checker says about each.

## The inputs a model declares

A model says what its model points look like and what tables it reads, so that the loader can
check the data before anything runs.

```python
import datetime as dt

from halley import (
    BuildError,
    Column,
    Count,
    Field,
    Int,
    Key,
    Money,
    MoneyPerPolicy,
    Monthly,
    Prob,
    Product,
    Schema,
    Table,
    UnitError,
    build,
    to_monthly_prob,
    var,
    where,
)
from halley.authoring import int_field

SCHEMA = Schema(
    fields=(
        Field("policy_id", "string", unique=True),
        int_field("age", ge=18, le=80),
        Field("sum_assured", "number", "MoneyPerPolicy", gt=0),
        Field("premium", "number", "MoneyPerPolicy", gt=0),
        Field("pols_if_init", "number", "Count", gt=0),
        int_field("rem_m", ge=1, le=600),
    ),
    horizon="rem_m",
)

MORT = Table(
    "mort",
    keys=(Key("age", lo=18, hi=130),),
    columns=(Column("q_ann", "Prob[Annual]"),),
    doc="Annual mortality by age last birthday",
)
```

Each field has a kind and, for numbers, a unit. `sum_assured` and `premium` are money per
policy; `pols_if_init` is a count. The bounds are checked when the data is loaded. `horizon`
names the field whose largest value sets the run's horizon, H: every policy is projected over
all H months, and the `t <= self.mp.rem_m` in the formulas is what ends each policy's cover. The
table declares its key, with the range the file must cover, and its one value column with its
unit.

## The model

```python
V_M = (1 + 0.04) ** (-1 / 12)  # one month's discount at 4% a year


class Term(Product):
    model_point = SCHEMA
    assumption_tables = (MORT,)

    @var
    def age(self, t: Int) -> Int:
        return self.mp.age + (t - 1) // 12

    @var
    def q_m(self, t: Int) -> Prob[Monthly]:
        return where(t <= self.mp.rem_m, to_monthly_prob(self.tables.mort[self.age(t)]), 0)

    @var(role="decrement")
    def deaths(self, t: Int) -> Count:
        return self.pols_if(t - 1) * self.q_m(t)

    @var(init=lambda mp: mp.pols_if_init, role="in_force")
    def pols_if(self, t: Int) -> Count:
        return self.pols_if(t - 1) - self.deaths(t)

    @var(timing="start")
    def premium_income(self, t: Int) -> Money:
        return where(t <= self.mp.rem_m, self.pols_if(t - 1) * self.mp.premium, 0)

    @var(timing="end")
    def death_outgo(self, t: Int) -> Money:
        return self.deaths(t) * self.mp.sum_assured

    @var(terminal=0, role="reserve")
    def reserve(self, t: Int) -> Money:
        return (self.reserve(t + 1) + self.death_outgo(t + 1) - self.premium_income(t + 1)) * V_M


compiled = build(Term)
```

Reading it from the top:

- `age` is the age at the start of month `t`. `t` is an integer month, so the formula is
  ordinary integer arithmetic.
- `q_m` looks the annual rate up by age and converts it to a monthly probability with
  `to_monthly_prob`, which is `1 - (1 - q) ** (1/12)`. Converting is explicit because the two
  units are different: an annual probability used where a monthly one belongs does not build.
  `where(condition, a, b)` is how a formula chooses by data or time; a Python `if` on `t` or on
  a field is an error, since the formula runs for every policy and month at once.
- `deaths` reads the policies in force at the end of the previous month. The role `decrement`
  tells the built-in checks that this is a decrement.
- `pols_if` reads itself one month back, so it needs a starting value: `init` gives month 0 from
  the model point. The role `in_force` enables the checks that it never goes negative or
  increases.
- `premium_income` and `death_outgo` say when in the month the money moves: premiums at the
  start, claims at the end. Nothing on this page uses that, and it is optional; `pv_month`, the
  helper that discounts a cash flow over its month by that timing, needs it, so a model that
  will be valued declares it from the start.
- `reserve` reads itself one month forward, so it needs a terminal value, and runs backwards.
  Everything it reads at `t + 1` is already known by then. Halley finds the order; the methods
  can be written in any order.

The model built, so its units and time references are consistent. Run it on two policies and a
made-up mortality table.

```python
import pyarrow as pa

from halley import RunSettings, run

points = pa.table(
    {
        "policy_id": ["A", "B"],
        "age": [35, 50],
        "sum_assured": [100_000.0, 250_000.0],
        "premium": [20.0, 90.0],
        "pols_if_init": [1.0, 1.0],
        "rem_m": [120, 60],
    }
)
mort = pa.table(
    {
        "age": list(range(18, 131)),
        "q_ann": [min(1.0, 0.0005 * 1.09 ** (x - 18)) for x in range(18, 131)],
    }
)
result = run(
    compiled,
    model_points=points,
    tables={"mort": mort},
    settings=RunSettings(valuation_date=dt.date(2027, 1, 1)),
    run_dir="runs/term",
)
print(round(float(result.lane("reserve", "A")[0]), 2), round(float(result.lane("reserve", "B")[0]), 2))
for check in result.invariants:
    print(check.passed, check.name)
```

```text
648.35 5634.01
True H1 in force never negative, never increasing
True H2 decrements never exceed the opening in-force count
True H2 probabilities in [0, 1]
True H3 reserve reaches its terminal value at t = 120
```

Inputs can be Arrow tables built in Python, as here, or files; the [next page](your-own-data.md)
covers files. The invariants that apply are the built-in ones, chosen by the roles the model
declared. Policy B's cover ends at month 60, where its premiums and deaths stop; this model has
no maturity decrement, so its survivors stay in force to month 120, unlike the shipped model's.

## The mistakes

Each of the models below changes one formula of `Term` and tries to build. Each fails before
anything runs, with a message that names the variable and the line. The file and line are
wherever the formula is; here each snippet is shown as if saved on its own as `term.py`. A unit
error or a missing starting value points at the variable's declaration, its `@var` line; a cycle
or a bad time offset points at the read itself.

A premium per policy, forgotten to be multiplied by the policies in force, is money per policy
where the model declares money. `class Forgot(Term):` makes a copy of `Term` with one variable
redefined; everything else is inherited. The `try`/`except` is only so the page can print the
message; in your own work the build simply stops with it.

```python
class Forgot(Term):
    @var(timing="start")
    def premium_income(self, t: Int) -> Money:
        return where(t <= self.mp.rem_m, self.mp.premium, 0)


try:
    build(Forgot)
except UnitError as error:
    print(error)
```

```text
premium_income is declared Money but its formula is MoneyPerPolicy
  at Forgot.premium_income (term.py:2)
```

An annual probability used as a monthly one:

```python
class Annualised(Term):
    @var
    def q_m(self, t: Int) -> Prob[Monthly]:
        return where(t <= self.mp.rem_m, self.tables.mort[self.age(t)], 0)


try:
    build(Annualised)
except UnitError as error:
    print(error)
```

```text
q_m is declared Prob[Monthly] but its formula is Prob[Annual]
  at Annualised.q_m (term.py:2)
```

Deaths calculated from the policies in force at the end of the same month, which themselves
depend on the deaths:

```python
class Cycle(Term):
    @var(role="decrement")
    def deaths(self, t: Int) -> Count:
        return self.pols_if(t) * self.q_m(t)


try:
    build(Cycle)
except BuildError as error:
    print(error)
```

```text
same-period cycle: {deaths, pols_if} depend on each other within month t
  at Cycle.deaths (term.py:4)
```

A recursion with no starting value:

```python
class NoStart(Term):
    @var(role="in_force")
    def pols_if(self, t: Int) -> Count:
        return self.pols_if(t - 1) - self.deaths(t)


try:
    build(NoStart)
except BuildError as error:
    print(error)
```

```text
pols_if reads itself at pols_if(t-1) and so needs an init value: @var(init=...)
  at NoStart.pols_if (term.py:2)
```

A time offset that depends on data, which would make it impossible to say whether the variable
runs forwards or backwards:

```python
class Lagged(Term):
    @var(timing="start")
    def premium_income(self, t: Int) -> Money:
        return self.pols_if(t - self.mp.age) * self.mp.premium


try:
    build(Lagged)
except BuildError as error:
    print(error)
```

```text
the time offset of pols_if must be a constant: pols_if(t), pols_if(t-1), pols_if(t+1); data-dependent offsets go through lag() with declared bounds, and a fixed month through at(pols_if, k)
  at Lagged.premium_income (term.py:4)
```

`at(self.pols_if, k)` reads a variable at a fixed month `k`, with `at` imported from `halley`;
`lag()`, which the message names, is the planned way to write an offset that varies by policy
and is not available in this release (it fails the build with a message saying so). Neither is
needed in this tutorial.

In a spreadsheet each of these is a cell that calculates something, and the mistake shows up,
if it does, when a total looks wrong. Here none of them produces a number.
