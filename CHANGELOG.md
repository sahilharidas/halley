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
- `invariants_sha256`: the sha256 of `invariants.json`, which `load_run(..., verify=True)` and a
  later stage that reads the run verify. A run made before has none, and such a stage asks for
  it to be rerun.

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
  runs a stage on them, unless its groups name a previous run or lock in a basis (below). Each
  series must carry float64 values and its provenance (`basis`,
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
  `members` as `[group id, member ids ascending]` per group (the members the runs of the
  series' own basis hold, which its sums ran over: a policy that left before the valuation date
  is in the `old` basis's series only), `copy`, `sha256` of the copy's
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

### The aggregation stage's loader and runs

- `halley.aggregation.stage.run_aggregation(compiled, *, groups, bases, scenarios=None,
  tables=None, previous=None, settings, run_dir)` runs a stage on its groups file and its bases,
  each a list of projection runs given as `RunResult`s or run directories. Every run is read
  through its directory once, with every file its manifest records a hash for verified first,
  including the outputs (`outputs.monthly_sha256`, `outputs.scalars_sha256`) and the invariant
  results (`invariants_sha256`): a changed or missing file is a LoadError naming it. The run's
  outputs, its manifest's sha256 (which the stage records) and its copied inputs are then taken
  from the bytes verified, so what was checked is what is used. What the loader asks of a basis
  run's inputs (their metadata, hashes and values) it reads from the verified copies, never
  from the manifest's description of them, which no hash covers. Then,
  in order: the bases are the ones declared, each a list of projection runs that `Basis.check`
  accepts (a policy in two runs is a LoadError naming it and both runs), at the stage's
  valuation date; the groups file holds the group key (`portfolio`, `cohort`,
  `profit_bucket`), `recognition_date` and the stage's fields, never `group_id` (derived as
  `portfolio-cohort-profit_bucket`) nor a field derived from the previous run, one row per
  group (a duplicate key names both rows), and passes the in-force 3.3 rules: month-end dates,
  `prev_valuation_date` at least a month before the valuation date and no earlier than the
  recognition date, at most one of `csm_prev` and `lc_prev` positive (the rules check the type
  of each column they read: a number column of strings names the file and column); one group
  per run (G1, `require_one_group`, the rule's one place); the previous aggregation run named by
  `prev_run_id`, found by scanning `RunSettings.runs_root/*/manifest.json` (no match or several
  is a LoadError; `previous=` overrides the search and must hold that run id), at one scenario
  (`require_deterministic`, D4) and with its membership copied (`inputs/membership.parquet`,
  below, whose `cur` rows must name exactly the groups it measured); G4 against its members, the
  `cur` rows of that verified copy (every policy at the valuation date was a member, in the same
  group; the `old` basis holds every member; `new` and `cur` hold the same policies), or one
  group key per policy across the bases without a previous run; every member, from its run's
  model point copy (which must hold the
  policies of the run's outputs), maps to exactly one group row and was issued on its
  `recognition_date`; the `old` and `new` bases read every financial input (`rf_df`, `cpi`,
  `ilp`) from one file, the `cur` and `new` bases every other table. The groups file's
  `discount_rec_sha256` and `ilp_sha256` must equal the `source_sha256` and `ilp_sha256`
  metadata of the stage's `p_l` and `p_rec` and of the `rf_df` (`p_l`) of every run of the
  `old` and `new` bases, and its `cpi_rec_sha256` the `rec_sha256` of those runs' `cpi`; an
  input there without the metadata is a LoadError, and so is a hash that no input the stage
  reads can verify. Each LoadError names the file and the row or key. The group fields
  `f_prev_0`, `f_prev_tau`, `lc_prev_tau` and `p_g_prev` are read from the previous run (its
  output of that name, or failing one its copy of the group series), with the unit and the time
  dimension checked and the month inside the domain the run records (`fcf_g(0)` of a run whose
  `fcf_g` starts at t = 1 is a LoadError, not the 0.0 stored there; D11), and `tau0`, `tau`,
  `i_acc`, `i_rec` derived from the dates; the previous run must value at
  `prev_valuation_date`, reach month `tau` and have closed at `csm_prev` and `lc_prev` (its
  `csm(0)`, `lc(0)`) within `1e-12 * P_g`, and the group's fixed attributes must be the ones it
  read from its groups file (its verified copy): the group key, `recognition_date` and every
  locked-in hash both files carry (B72(b): the locked-in basis is fixed for the life of the
  group), or a LoadError names this row and the previous run's. The stage's own scenario and
  table files are read once: the run uses, copies and hashes the bytes the checks read.
- `run()` refuses a stage whose group schema declares `prev_run_id`, a field derived from the
  previous run or the dates (`FROM_PREVIOUS`) or a locked-in hash (`LOCKED_IN_FIELDS`): such
  fields typed into the model points would bypass the previous run, G4 and the hash checks. It
  is a LoadError naming the fields and `run_aggregation`, before anything is written. A stage
  that sums its bases alone still runs through `run(..., series=group_series(...))`.
- `halley.authoring.contracts.group_schema(*fields, previous=False, locked_in=False, rules=(),
  valuation_date_rule=<month-end>)` builds a stage's group schema with the in-force 3.3 fields;
  `GROUP_KEY`, `PROFIT_BUCKETS`, `PREVIOUS_FIELDS`, `LOCKED_IN_FIELDS`, `FROM_PREVIOUS_RUN`,
  `FROM_DATES` and `FROM_PREVIOUS` name them, and `GROUP_ID` is defined there. At build, a
  stage's schema declares the fields the loader reads or fills as it takes them (`portfolio`
  and `profit_bucket` as `string` or `enum` fields, `cohort` as an `int` field of unit `Int`;
  `recognition_date` and `prev_valuation_date` as `date` fields; `prev_run_id` and the three
  locked-in hashes as `string` fields; `tau0`, `tau`, `i_acc`, `i_rec` as `int` fields of unit
  `Int`; `f_prev_0`, `f_prev_tau`, `lc_prev_tau`, `p_g_prev`, `csm_prev`, `lc_prev` as `number`
  fields of unit `Money`; none as a schema-derived field): a unit or a date declared otherwise
  is a UnitError (D11), another kind a BuildError, at the stage class. It declares
  `prev_run_id` beside any field derived from the previous run and beside `csm_prev` or
  `lc_prev`, which the loader compares with that run (a BuildError; a schema with the closing
  balances and no previous run would have had them compared with nothing).
- Provenance: an input that names its source in `halley.rebased_from` or `halley.derived_from`
  metadata (a basis run's copied input or the stage's own file) is recomputed by the helper
  registered for the key with `halley.aggregation.stage.register_provenance(key, helper, *,
  replace=False)` and must equal it bit for bit; a file whose key has no helper is a LoadError,
  since it cannot be verified. A key that has another helper already is a ValueError unless
  `replace=True`, and each `provenance` entry of the manifest records the `helper` that
  verified the file (its module and qualified name). A helper receives a `ProvenanceCheck` (the
  file, the key and value, the bases, the previous run, and `find_file(sha256)` /
  `find_run(run_id)` over the stage's own inputs and the directories of the previous run, the
  runs it read and the bases, each run read from the bytes verified). The previous run's bases
  are recorded relative to its `runs_root`: they are resolved under the stage's
  `RunSettings.runs_root` or, without one, under the absolute root the previous run recorded,
  and skipped otherwise, never resolved against the working directory. The design's section 10
  helpers register themselves in a later stage; until then no file of the shipped products
  carries either key.
- An aggregation run's directory also keeps `inputs/groups.parquet` (the file as given),
  `inputs/derived_groups.parquet` (`group_id` and the derived fields), `inputs/membership.parquet`
  (`policy_id`, `basis`, `group_id`: every member of every basis run, by basis in the declared
  order, then policy id; `inputs.membership` records its `copy`, `sha256`, `rows`, `columns`
  and the group `key`, and like every copy it is verified when a later run reads the run) and,
  beside every mean
  series, `inputs/series/<name>.pathwise.parquet`, the per-scenario group totals it averaged
  (`halley.scenarios = pathwise`; `inputs.series.<name>.pathwise` in the manifest, outside the
  run id), as does a stage run through `run(..., series=group_series(...))`. Its manifest adds
  `previous` (`run_id`, `dir`, `manifest_sha256`), `bases` (the same for every run of every
  basis, with `outputs_sha256`), `locked_in` (per group, the three hashes and, under
  `verified_by`, each hash's
  verifying files as `file`, `sha256` and `metadata`), `provenance` and `disclosures`
  (`confidence_level`, alpha of the `cur` basis's `ra_param` copies, a LoadError naming the
  copy when one holds no single numeric `alpha`; `para_120_curve_sha256`, the stage's `df_cur`
  file; and what the stage declares in the new class attribute `Aggregation.disclosures`, a
  dict of strings or tuples of strings that may not name either of those two, checked at
  build). Directories are recorded relative to `runs_root` when they lie under it.
  `inputs.series.<name>.source_runs` gains `outputs_sha256` beside `run_id`, `dir` and
  `manifest_sha256`, and `group_series(..., runs_root=None)` records `dir` the same way. Such a
  run's id covers the IR, the groups file's hash, every series' `values_sha256`, every scenario
  and table file's hash, the previous run's id and manifest hash, the settings but `label`,
  `runs_root` and `workers`, the backend and the horizon; not where any run lives.
- `halley.aggregation.stage.replay_aggregation(compiled, from_run, *, run_dir)` re-runs an
  aggregation run from its own directory alone, every file verified, the bases and the previous
  run unread: the same run id and the same bits when nothing the id covers changed. The replay's
  manifest carries the original's blocks and `replayed_from`, which records `matches` (the same
  run id, the same values bit for bit as the run's verified outputs hold them, and the same
  output hashes) and, when not, `differences` and the `environment` entries of the `backend`
  block that changed. It is a RunError (D21) when the run id differs with the same backend and
  NumPy, when a value differs with the same backend, NumPy and platform (another pyarrow or
  Python writes other bytes but cannot change a value; the platform's libm can), or when an
  output file differs with nothing in the backend block changed; a file that differs while the
  values agree, with another pyarrow or Python, is recorded only.
- `RunSettings` gains `runs_root` (a path; recorded, not in the run id) and `workers` (a
  positive int, default 1; recorded, not in the run id; every batch still runs in the calling
  process until the process pool of design section 12), `identity()` (the settings a run id
  covers) and `RunSettings.from_dict(manifest["settings"])`. A projection's run id is unchanged.
- `load_run(run_dir, verify=True)` checks every file the manifest records a hash for first and
  reads the run from the bytes checked; `halley.runtime.run.verified_run` (a `VerifiedRun`: the
  manifest, its sha256 and the verified bytes, with `read(entry, label)` for a copy and
  `result()` for the run), `verify_run` (the manifest only), `read_manifest` (the manifest and
  its bytes' sha256), `write_manifest` (which returns the sha256 of the bytes written) and
  `run_files` are the helpers. `RunResult.kind` is `outputs.kind`, and
  `RunResult.manifest_sha256` the sha256 of the manifest as `run()` wrote it or `load_run` read
  it.
- A run's copy of an input is the bytes it read and hashed, not what the file's path holds when
  the copy is made, and a `ReadFile` already read is accepted as an input source.
- A variable may not be named `disclosures` in a stage.

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
