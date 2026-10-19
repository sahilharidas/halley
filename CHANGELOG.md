# Changelog

Changes that a user of Halley's runs, run directories or authoring API can see. The engine's
decisions and their reasons live in the design documents; this file records what changed and
when.

## Unreleased

### Run ids of existing runs change

The run id no longer depends on where the input files were. Its identity drops the `path`
strings of the model point, table and scenario files (their hashes stay) and gains every
`halley.*` metadata entry and bare `version` key of every input, which the manifest now records
under `inputs.<kind>.<name>.metadata`. A run is therefore addressed by one id wherever its
inputs live (M9). The cost: the `run_id` in the manifest of every run made before this change
differs from the id a recomputation of the same inputs and settings would give. No run
directory is rewritten; compare old runs by their manifests' hashes, not by id.

### Manifest additions

- `inputs.<kind>.<name>.metadata`: the input file's `halley.*` metadata and `version`, as
  strings, keys without the prefix.
- `outputs.kind` (`"projection"`), `outputs.id_field` (the schema's id field, which keys the
  output files), `outputs.monthly_sha256` and `outputs.scalars_sha256` (the written files).
- `outputs.variables.<name>` gains `unit`, `start` and `role` beside `time`.

`OUTPUT_SCHEMA_VERSION` stays 1: the row layout is unchanged, and `load_run` reads older
manifests (an output keyed on `policy_id` when `outputs.id_field` is absent).

### Input contracts and invariants

- `Table` takes `metadata` (required `halley.<key>` entries) and `rules`; `ScenarioInput` takes
  `gt`, `ge`, `lt`, `le`, `rules`, `column` and `covers`; a `Product` declares `input_rules`
  over every loaded input and `invariants` checked on every batch after the projection.
- An input rule's failure names the first failing table key, policy or scenario row when the
  rule's result has the shape of exactly one input.
- A table column may not take the name of a `TableView` attribute (`name`, `keys`, `columns`,
  `shape`, `metadata`, `at`, `index`, `key_values`, `key_range`); the declaration refuses it.
- An invariant's residual that is not a number, or infinite, fails the invariant; a `where` mask
  must be boolean; an empty window (`months=(5, 2)`) is refused at declaration when its order is
  fixed and at the run otherwise; the first failing invariant is reported before any later one
  is evaluated.
- The arrays an invariant's batch view takes from the loaded inputs (`a.mp.<field>`,
  `a.table(...)`, `a.scen(...)`, `a.scen_range(...)` and `a.scale(...)`) are read-only views:
  writing into one raises instead of changing the inputs of the later batches and of the run's
  result.

### The aggregation stage's language

- `halley.Aggregation` is the base class of an aggregation stage: a `Product` whose model points
  are groups, keyed on `group_id` (a group schema with another id field is a BuildError at the
  class), with `self.group` reading a group's field as `self.mp` does, and whose extra inputs
  are group series. It declares `bases`, a tuple of `Basis(name)`, and `series`, a tuple of
  `GroupSum(name, basis, variables, unit, scenarios="mean" | "each", *, time=True, start=None)`
  (`start` is 1 when left out; a series declared `time=False` takes none). A group sum adds
  `Money`, `Count` or `CoverageUnits` only: another unit, such as a probability, a rate, a
  factor, an index, a flag or `MoneyPerPolicy`, does not add across a group's members and is a
  UnitError at the declaration. A name, basis, unit, `time` or `start` of the wrong type
  (`start=True`, `start=1.0`, `time="yes"`) is a BuildError there too, as is a variable that is
  not a variable's name (`"two words"`) or is named twice (`("fcf", "fcf")`). A declaration's
  location is not an argument: one made with `dataclasses.replace()` from another, as a subclass
  replaces its parent's, is located at the `replace()` line, so a later error about it points at
  the subclass. A projection (a `Product` that is not an `Aggregation`) that sets `bases` or
  `series` is a BuildError at the class: nothing would read them. A formula reads a series
  as `self.series.<name>(t + k)`, `self.series.<name>.at(k)` or, for a series declared
  `time=False`, `self.series.<name>()`, which messages and `explain` quote as `series.<name>()`.
  The checker checks every read like a variable read: the unit is the declared one, the
  dimensions are fixed by the declaration (group, scenario and, with `time`, month; the
  scenario axis exists after a mean too), a relative read stays inside the series' domain (from
  `start`, which a sum of states declares as 0, up to `H`, or `H - 1` from a backward
  recursion), a boundary reads a series only at a constant month, and a series of an undeclared
  basis or a read of an undeclared series is a BuildError at the model line. A declared series
  no formula reads is a BuildWarning naming its declaration; a run still needs it given. Each
  series' basis and variables are part of the stage's graph hash, and so of `ir_sha256` and the
  run id: a stage edited to sum other outputs is another stage.
- `halley.aggregation.group_series(stage, bases, members)` sums each series from projection runs
  of the session: each named variable over a group's members in ascending policy id across the
  basis's runs, sequentially, several variables' totals added left to right per scenario, then
  the mean over scenarios as `cumsum_s((1.0 / J) * x)[-1]` for `scenarios="mean"`. Every
  variable must be stored by the runs with the declared unit and time dimension, a basis's runs
  must share the valuation date, horizon and J, read every scenario input they share from the
  same file (by the sha256 their manifests record, so that scenario s is one path in every run)
  and hold disjoint policies (`Basis.check`, which returns a `BasisInfo(valuation_date,
  horizon)`, exported from `halley.aggregation`), and every group must have members, and a
  member in the runs of each basis a series sums; a LoadError otherwise. `run(..., series=...)`
  runs a stage on them. Each series must carry float64 values and its provenance (`basis`,
  `variables`, `basis_scenarios`, `valuation_date`, `basis_horizon`, `source_runs`, `members`),
  and must sum what the stage declares: a series of other variables, or of another basis, is a
  LoadError (`series fcf_g: the stage sums fcf + cf of basis cur but was given fcf of basis
  cur`). Every basis must value at the stage's valuation date. The stage's horizon is its `cur`
  basis's, which every series of that basis records as `basis_horizon` (a `cur` basis that
  supplies only series without a time dimension sets it too), or `RunSettings.horizon`, which
  may not exceed it; a series read relatively is cut to that horizon and comes from a basis that
  reaches it, or the LoadError names the basis; a stage without a `cur` basis whose series
  disagree on the horizon asks for `RunSettings.horizon`. A constant `.at(k)` must be a month
  the series covers. A stage that reads a mean runs at one scenario: its own scenario
  inputs and the per-scenario series it reads pass `require_deterministic` when it loads (a
  declared series no formula reads does not count). A run-time error in a stage names the
  failing groups (`; groups G0`), as do a failing invariant, built-in or the stage's own
  (`invariant csm_small failed: group G0, t = 0: ...`), a failing input rule (`for group G1`)
  and a group field outside its bounds (`groups G1 break this`); `explain` says `for group G0`.
  The not-finite check now says `is not finite (NaN or infinity) at some month`, then the
  policies or groups.
- A stage's run directory keeps a copy of every series it read, `inputs/series/<name>.parquet`
  with the columns `<id field>`, `scenario`, `t`, `value`. The manifest records it under
  `inputs.series.<name>` (`basis`, `variables`, `basis_scenarios`, `valuation_date`,
  `basis_horizon`, `source_runs` with each run's `run_id`, `dir` and `manifest_sha256`,
  `members` as `[group id, member ids ascending]` per group, `copy`, `sha256` of the copy's
  bytes, `values_sha256`, `unit`, `time`, `start`, `scenarios`, `months`) and marks the outputs
  `outputs.kind = "aggregation"`, keyed on `group_id`. The run id covers each series by
  `values_sha256`, a hash of its name, unit, start, scenario treatment, group ids, shape and
  little-endian float64 values, not by the copy's bytes (which name the parquet writer, so a
  pyarrow upgrade would move the id) nor its source runs' directories. A projection's run id
  and graph hash are unchanged.
- `RunResult.variable_info(name)` returns a `VariableInfo(time, unit, start, role)` read from the
  manifest, for a run of the session and a loaded one alike.
- `require_deterministic(source, what, reason, error)` takes anything with `n_scenarios` (a run,
  a scenario input, a group series) and raises `error` (a RunError by default); its message now
  names the mean as the way a J > 1 basis reaches a stage.
- Removed: `halley.aggregation.AggregationInputs`, which took several runs of the same policies;
  a `Basis` takes the runs of disjoint policies instead.
- A variable may not be named `series_declarations` (a `Product` class method) and, in a stage,
  `group`, `series` or `bases`.

### Notes on determinism

- A power whose exponent is not a literal, such as `x(t) ** self.mp.k`, can differ in the last
  place between two batch compositions of the same run (`RunSettings.batch_size`) when the
  exponent's value is 0.5, 2 or -1, because NumPy chooses its sqrt, square or reciprocal fast
  path from the shapes of the batch. Reruns with the same batching stay bit-identical (D19) and
  another batching agrees within 1e-12 relative (D5), which is what the annuity specification's
  "Built-in invariants" promise. `recompute` and `explain` evaluate such a power as libm's pow,
  so they agree with the stored value within the same tolerance, and bit for bit where the batch
  took pow. A literal exponent (`x ** 0.5`, the mid-month `pv_month`) takes one path at every
  batch size; the shipped products raise only to literals and to the month index (`v_m ** t`,
  `(1 + i) ** (-t)`, the `-12 / t` of `ann_spot`), which took libm's pow at every batch shape
  measured.
- `recompute` and `explain` take NumPy's fast path where the generated code does for a 0-d
  exponent (a keyless table's value) against a scalar base, as in
  `self.tables.rates.f["a"] ** self.tables.power.e`, so they agree with the stored value bit for
  bit there.
