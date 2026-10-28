# Where next

You have run the shipped model, read a run directory, written a model, fed it data and run it over
two scenarios. What the rest of the site holds:

- **The specifications.** Each shipped model is written from a specification that fixes every
  formula, convention and reference number, so that a spreadsheet can be built from it by hand
  and the engine reconciled to the spreadsheet. The [term assurance
  specification](../spec/term-assurance.md) is the model you ran first; its section 15.1 lists
  every variable with its unit and formula, and the model as it runs is
  `halley/products/term_assurance.py` in the installed package, the file the explain output on
  [reading a run](reading-a-run.md) pointed into. Section 15.2 is an earlier authoring sketch
  whose syntax differs from the tutorial's. The [annuity specification](../spec/annuity.md) is a
  larger product, an annuity in payment under IFRS 17, with the risk adjustment and the coverage
  units a group's contractual service margin needs.
- **The aggregation stage.** Totals over a group of policies, and the IFRS 17 measurement of a
  group over a reporting period, are a second kind of model that reads the outputs of projection
  runs rather than model points. The [annuity specification](../spec/annuity.md) specifies the
  measurement of a group new at the valuation date and the [in-force worked
  example](../spec/in-force-group.md) the subsequent measurement of an in-force group over a
  reporting period; `halley.aggregation` holds both stages.
- **Faster runs.** `build(Term, backend="numba")` compiles the same model with Numba
  (`pip install halley[numba]`); it is held to the NumPy reference within 1e-9 relative and is
  bit-identical between its own reruns. `RunSettings(workers=4)` spreads a run's batches over
  four processes, on either backend, with the same results as one process at the same batch
  size, bit for bit. The workers import the model's class from its module, so a model run with
  workers must be defined at the top level of an importable module, and a script needs its run
  under `if __name__ == "__main__":`. The README states both guarantees and says how the Numba
  backend is held to the reference.
- **The source.** Halley is Apache 2.0 licensed, at [github.com/sahilharidas/halley](https://github.com/sahilharidas/halley).
  The README's layout table says which package does what.

If something on these pages did not work as printed, that is a defect: the pages are run by the
test suite, and an issue with the page and the output you saw is the right report.
