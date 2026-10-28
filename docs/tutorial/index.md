# Tutorial

This is the place to start if you have never used Halley. It assumes you can read a formula and
have Python installed, and nothing else; the first page says how to run the code. By the end you
will have run the engine, read what it produced, written a small model of your own, fed it your
own data and run it over more than one scenario. Every code block on the pages that follow is
run by the test suite, so what they print is what you will see.

## Halley in five minutes

Halley projects insurance policies month by month, the way a cash-flow spreadsheet or a Prophet
model does: policies in force, deaths and lapses, premiums in and claims out, and a reserve that
is rolled back from the end of the term. You write the formulas as a Python class, one short
method per variable, and Halley does the rest.

What makes it different from a spreadsheet is what happens between writing and running.

- **It checks the model before it runs.** Every variable declares a unit, such as money, a
  count of policies or a monthly probability, and every formula is checked against those units.
  A premium per policy added to a total, an annual rate used as a monthly one, two formulas that
  each read the other within the same month, a recursion with no starting value: each of these
  refuses to build, with a message that names the line in your model. In a spreadsheet these are
  the mistakes that pass silently.
- **It runs every policy and scenario at once.** The formulas are turned into vectorised code,
  so a book of a hundred thousand policies runs in one pass rather than one policy at a time.
- **It records how it got each number.** Every run writes a folder with a copy of every input, a
  manifest of their hashes, the outputs and the results of the built-in checks, and any value of
  any policy at any month can be explained from its formula and inputs.

A model looks like this; these are four of the seven variables of the one the tutorial writes on
[its third page](your-own-model.md). Each `def` is one variable, a column of the projection
sheet; the names are the ones you would use there, and the type after the arrow is the
variable's unit (`t: Int` says the month is a whole number).

```python
class Term(Product):
    model_point = SCHEMA
    assumption_tables = (MORT,)

    @var
    def q_m(self, t: Int) -> Prob[Monthly]:
        return where(t <= self.mp.rem_m, to_monthly_prob(self.tables.mort[self.age(t)]), 0)

    @var(role="decrement")
    def deaths(self, t: Int) -> Count:
        return self.pols_if(t - 1) * self.q_m(t)

    @var(init=lambda mp: mp.pols_if_init, role="in_force")
    def pols_if(self, t: Int) -> Count:
        return self.pols_if(t - 1) - self.deaths(t)

    @var(terminal=0, role="reserve")
    def reserve(self, t: Int) -> Money:
        return (self.reserve(t + 1) + self.death_outgo(t + 1) - self.premium_income(t + 1)) * V_M
```

`@var` marks a `def` as a variable of the model. `t` is the month, counted from the valuation
date: the row number of the sheet. `self.mp` is the policy's own data, `self.tables` the
assumption tables, and `self.pols_if(t - 1)` reads a variable one month back, the cell one row
up. A variable that reads itself needs a boundary value: `init` is its value in row 0
(`lambda mp: mp.pols_if_init` is a one-line function that takes it from the model point) and
`terminal` its value in the last row. The reserve runs backwards from a terminal value of zero;
the policies in force run forwards from a starting count. Halley works out the order; you write
the formulas. `role` tells the built-in checks what a variable means, so an in-force count that
rises, or a decrement larger than the count in force, is caught.

## The pages

```{toctree}
:maxdepth: 1

first-run
reading-a-run
your-own-model
your-own-data
scenarios
where-next
```

Read them in order; each page continues from the one before.
