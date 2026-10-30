# Changelog

Changes that a user of Halley's runs, run directories or authoring API can see. The engine's
decisions and their reasons live in the design documents; this file records what changed and
when.

## 0.2.0 (unreleased)

### Run ids of existing runs change

The run id no longer depends on where the input files were. Its identity drops the `path`
strings of the model point, table and scenario files (their hashes stay) and gains every
`halley.*` metadata entry and bare `version` key of every input, which the manifest now records
under `inputs.<kind>.<name>.metadata`. A run is therefore addressed by one id wherever its
inputs live (M9). The cost: the `run_id` in the manifest of every run made before this change
differs from the id a recomputation of the same inputs and settings would give. No run
directory is rewritten; compare old runs by their manifests' hashes, not by id.

### Formulas the NumPy backend computed wrongly

Three kinds of formula now compute what the checker types them as. A model that uses one stores
other values under the run id it had before (the identity holds the IR, which is unchanged), so
compare such a run with an older one by its values. No shipped product uses them; their values
and run ids do not move.

- `~` of a Bool is logical not wherever it appears. A comparison of plain Python numbers (`t`
  inside a recursive component, `self.run.*` and literals in a variable without a time
  dimension) was a Python bool in the generated code, and Python's `~True` is -2, which
  `where()` takes as true: `where(~(t > 9), 1.0, 0.0)` inside a recursion gave 1.0 after month
  9, where whole mode, `recompute` and `explain` gave 0.0.
- An integer literal raised to an Int power, a Factor, is a float power. Over the month column
  NumPy raised `2 ** t` in int64, which wraps past 2**63 without a word (`2 ** 63` was -9.2e18
  and `2 ** 64` was 0.0) and refuses a negative exponent (`2 ** -t` was a ValueError), while
  inside a recursive component the same formula was an exact Python integer. A base built from
  integer literals alone (`where(c, 2, 3) ** t`) is raised as a float too.
- A negative literal base keeps its sign: `(-2) ** t` was generated as `-2 ** t`, which Python
  reads as `-(2 ** t)`, on both backends (`recompute` and `explain` had it right).

### Formulas that are now build errors

Each of these built and then gave a wrong value or failed at run time: a model that uses one no
longer builds. No shipped product does.

- An enum model point field used as a table key lists the key's values in the key's order. A
  field declared `enum=("F", "M")` read a table keyed `enum=("M", "F")` at the F row for an M
  policy, on both backends, since the generated code reads the table at the field's position in
  its own enum (`recompute` and `explain` read the M row). It is a BuildError at the lookup's
  line naming both orders.
- An enum key that is not one of the key's values is a model point field or a `where()` between
  such fields. Anything else is a BuildError at the lookup's line: a `where()` with a field of
  other values (`where(t > 0, self.mp.band, self.mp.sex)`, band A, B or C against a key M, F,
  read outside the two-row table for band C, an IndexError on the NumPy backend and memory past
  the array on the Numba backend), a `minimum()` or `maximum()` of fields, or an enum column of
  another table.
- A float literal is not an Int. `where(c, self.mp.k, 2.5)`, and `minimum()` or `maximum()` of
  an Int and a float literal, are UnitErrors, as is an Int variable whose formula, `init` or
  `terminal` is a float literal. Stored in an Int variable such a value was truncated; as a
  table key it was an IndexError on the NumPy backend and truncated on the Numba backend. An
  integer literal still takes an Int's unit (`where(c, self.mp.k, 2)`).

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
- Every array an invariant's batch view returns is read-only (`a.var(...)`, `a.at(...)`,
  `a.mp.<field>`, `a.table(...)`, `a.scen(...)`, `a.scen_range(...)`, `a.scale(...)`, `a.ids`
  and `a.t`): writing into one raises instead of changing the inputs of the later invariants and
  batches, or the run's result and the ids its output files are keyed on.
- `build()` holds every build-time parameter declared `Static[T]` to its type, as given to
  `build()` and as the class's default: `Static[bool]` takes True or False only (the string
  `"False"` built the other branch and recorded a string in the manifest), `Static[int]` a
  Python int that is not a bool, `Static[float]` a Python number that is not a bool. Another
  value is a BuildError at the class naming the parameter, its type and the value
  (`build(CPILinkedAnnuity, cpi_lag_m=3.0)` failed with a message about time offsets).

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
  the subclass. A series name with a leading underscore, which `self.series.<name>` cannot
  read, is a BuildError at the declaration. A projection (a `Product` that is not an
  `Aggregation`) that sets `bases` or `series` to a stage's declarations (a `Basis`, a
  `GroupSum` or a tuple holding one) is a BuildError at the class: nothing would read them; a
  projection's own constant of either name (`bases = ("best", "prudent")`) is its business. A
  formula reads a series
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
  horizon)`, exported from `halley.aggregation`, and refuses a stage's run, `outputs.kind =
  "aggregation"`, whose group ids it would sum as policies), and every group must have
  members, and a member in the runs of each basis a series sums; a LoadError otherwise.
  `run(..., series=...)` runs a stage keyed on `group_id` on them (a stage on the group schema
  of in-force 3.3 runs through `run_aggregation`, below). Each series must carry float64 values
  and its provenance (`basis`,
  `variables`, `basis_scenarios`, `valuation_date`, `basis_horizon`, `source_runs`, `members`),
  and must sum what the stage declares: a series of other variables, or of another basis, is a
  LoadError (`series fcf_g: the stage sums fcf + cf of basis cur but was given fcf of basis
  cur`). The series of one basis must record the same source runs (by run id and manifest
  hash, not directory), valuation date, `basis_scenarios`, `basis_horizon` and members, as one
  `group_series` call sums them (`series fcf_g and series cu_g of basis cur record other
  source_runs; ...`), and a policy is a member of one group across the bases (G4); a series
  holds at most its basis's months, a series that keeps its scenarios exactly its basis's J and
  a mean's pathwise values the same J (`basis_scenarios` must be an int of at least 1). The run
  keeps its own read-only copy of every series' values and pathwise values: a caller that writes
  into the arrays it passed changes neither the run's inputs nor what `recompute` and `explain`
  read. Every basis must value at the stage's valuation date. The stage's horizon is its `cur`
  basis's, which every series of that basis records as `basis_horizon` (a `cur` basis that
  supplies only series without a time dimension sets it too), or `RunSettings.horizon`, which
  may not exceed it; a series read relatively is cut to that horizon and comes from a basis that
  reaches it, or the LoadError names the basis; a stage without a `cur` basis whose series
  disagree on the horizon asks for `RunSettings.horizon`. A constant `.at(k)` must be a month
  the series covers. A stage that reads a mean runs at one scenario: its own scenario
  inputs and the per-scenario series it reads pass `require_deterministic` when it loads (a
  declared series no formula reads does not count). A run-time error in a stage names the
  failing groups (`; groups G0`) at every batch size, including a read outside a series' or a
  variable's months that fails for every group of the batch (it named none in a batch of
  several, in a projection too), as do a failing invariant, built-in or the stage's own
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
- A variable may not be named `series_declarations` or `check_statics` (`Product` class
  methods) and, in a stage, `group`, `series` or `bases`.

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
  per run (G1, `require_one_group`, checked in one place; the loader also resolves one previous
  run per stage, and refuses groups that name several `prev_run_id` values, a second
  restriction of this version that lifting G1 must lift too); the previous aggregation run named by
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
  is a LoadError naming the fields and `run_aggregation`, before anything is written. So is
  every stage whose groups carry the group key or `recognition_date` of in-force 3.3 (a
  `group_schema` stage): `run_aggregation` derives each group's id from its key and checks G1,
  the groups file's rules and every member's group and issue date against the bases' runs,
  which `run()`, seeing only the group series, would take as typed. A stage keyed on
  `group_id` alone (a portfolio total) still runs through `run(..., series=group_series(...))`.
- `halley.aggregation.stage.load_aggregation_inputs(compiled, *, groups, bases,
  scenarios=None, tables=None, previous=None, settings)` returns a `StageInputs` (the model
  points with the derived fields, the series, the record and the scenario and table files as
  read), the inputs `run_aggregation` passes to `run(..., stage_record=StageRecord(...))`;
  `halley.aggregation.stage.find_run(run_id, runs_root)` returns the one directory under
  `runs_root` whose manifest carries that run id (none or several is a LoadError).
  `stage_record=` is the hook of `run_aggregation` and `replay_aggregation`: a `StageRecord`
  built by hand is taken as given, and none of the loader's checks run.
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
  replace=False)` and must equal it: bit for bit on the machine that built it, and within
  `halley.runtime.tolerance.LIBM_ULPS` units in the last place (eight) on another, whose C
  library's `exp` and `pow` may round a last bit the other way (a file built from another
  source, at another month or with another ILP is off by orders of magnitude more); a file
  whose key has no helper is a LoadError, since it cannot be verified. A key that has another
  helper already is a ValueError unless `replace=True`, and each `provenance` entry of the
  manifest records the `helper` that verified the file (its module and qualified name) and
  `ulps`, the largest distance found (0 when bit for bit). A helper receives a
  `ProvenanceCheck` (the
  file, the key and value, the bases, the previous run, and `find_file(sha256)` /
  `find_run(run_id)` over the stage's own inputs and the directories of the previous run, the
  runs it read and the bases, each run read from the bytes verified). The previous run's bases
  are recorded relative to its `runs_root`: they are resolved under the stage's
  `RunSettings.runs_root` or, without one, under the absolute root the previous run recorded,
  and skipped otherwise, never resolved against the working directory. A helper's LoadError
  that does not name the file being verified (a source the finder cannot find, say) is reported
  against that file and the metadata entry it recomputes from (`scenario input p_l (...):
  recomputing it from halley.rebased_from = ...: no file with sha256 ...`). The design's section
  10 helpers register their recomputations for both keys (below).
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
  dict of strings or tuples of strings (or functions of the stage's statics, below) that may
  not name either of those two or `para_120_curve_verified`, checked at build). Directories
  are recorded relative to `runs_root` when they lie under it.
  `inputs.series.<name>.source_runs` gains `outputs_sha256` beside `run_id`, `dir` and
  `manifest_sha256`, and `group_series(..., runs_root=None)` records `dir` the same way. Such a
  run's id covers the IR, the groups file's hash, every series' `values_sha256`, the
  membership's rows (by value: two runs whose groups hold other policies with the same totals
  have other ids), every scenario and table file's hash, the previous run's id and manifest
  hash, the disclosures (what the stage declares, the confidence level and the para 120 curve's
  hash), the settings but `label`, `runs_root` and `workers`, the backend and the horizon; not
  where any run lives.
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
  positive int, default 1; recorded, not in the run id; see "Process parallelism (M6)"
  below), `identity()` (the settings a run id covers) and
  `RunSettings.from_dict(manifest["settings"])`. A projection's run id is unchanged.
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
- A variable may not be named `disclosures` or `member_rules` in a stage.

### The CSM of one group (M7) and the cross-run helpers (M9)

- `halley.aggregation.CsmRollForward` is the monthly CSM and loss-component roll-forward of one
  group on its `cur` basis, as the annuity spec writes it ("Accretion and release", "Onerous
  groups and the loss component"), and `halley.aggregation.NewGroup` the stage of a group
  recognised at the valuation date. The roll-forward reads the group series `cu_g` (`cu`),
  `fcf_g` (`bel` + `ra`, formed first), `bel_g` and `ra_g`, and its own scenario input `p_l`,
  the locked-in curve (value column `rf_df`, months 0..H, rule `p_l(0) = 1`). It computes
  `j_l` and `d_l` (the locked-in one-month rate and discount factor), `d_cu` (d_L, or 1 with
  the build-time switch `discount_coverage_units = False`), `cu_next` (N(t), a backward
  recursion from 0 at H), `units`, `csm_open` (the CSM table's opening line, CSM(t - 1)),
  `acc`, `csm_pre`, `release`, `csm` (from the hook `csm_0`), `lc` (from the hook `lc_0`, by
  the ratio of Illustrative Example 8), `lc_alloc`, `lrc` and `lrc_0`, every division guarded
  inside its `where`, and stores them with the series by month. The table's changes for future
  service are 0 in every month of the roll and are not a stored line. `NewGroup` adds the
  series `premium_total` and the hooks `p_g`, `csm_0 = max(0, P_g - FCF_g(0))` and `lc_0 =
  max(0, FCF_g(0) - P_g)`, which it stores. A subclass that leaves a
  hook undefined is a BuildError located at the subclass, not at the hook's line in
  `halley.aggregation.csm`.
- The M12 lines of the months on the `cur` basis (design section 9), on `CsmRollForward`: the
  group series `ann_out_g`, `exp_out_g` and `inv_exp_g` and the scenario input `df_cur`, the
  composed current curve (value column `rf_df`, months 0..H, rule `df_cur(0) = 1`, one
  scenario through `require_deterministic`), from which it computes and stores `f` (the
  current one-month forward rate), `ifie_bel`, `ra_fin`, `ra_rel`, `sar`, `svc`, `lc_svc`,
  `revenue = (svc - lc_svc) + release` and `ifie = (ifie_bel + ra_fin) + acc`. Every run of a
  `CsmRollForward` stage therefore takes `df_cur` beside `p_l`; build it with
  `composed_curve(<the cur runs' rf_df>, <their ilp>, H)`.
- A stage class may set `discloses_current_curve` (a bool; `CsmRollForward` sets it, and must
  declare the `cur` basis and the scenario input `df_cur`, a BuildError otherwise): its
  `df_cur` is the composed curve of the cur runs, whose hash the manifest discloses as the para
  120 curve, so `run_aggregation` requires `df_cur`'s metadata to name the `rf_df` and `ilp`
  files every cur run read (`source_sha256`, `ilp_sha256`) and `rebased_from`, and the
  provenance check then rebuilds it from them bit for bit. A `df_cur` without that metadata,
  composed from another discount or ILP file, re-based at another month (`tau0` other than 0)
  or holding other values than its sources give is a LoadError naming scenario input `df_cur`,
  before anything is written, where it was an M1 failure after the run. The disclosures record
  the files under `para_120_curve_verified` (`source_sha256`, `ilp_sha256`, `composed_from`),
  which a stage may not declare. A variable may not be named `discloses_current_curve` in a
  stage.
- Both stages check the annuity spec's I11 (with the exact zero of the CSM from the last month
  with coverage units on, the last month included, and the recursion at a scale of the terms'
  absolute values, so that a CSM gone negative fails the sign check rather than the recursion
  against a negative tolerance), I12 (both forms), I13 (the ratio wherever F(t) > 0 and F(0) >
  0, so a group with LC(0) = 0 keeps LC(t) = 0; in-force 12.2's "and LC(0) > 0" left such a
  group unchecked), the design's M1 (`LRC(t) - LRC(t-1) = IFIE(t) - revenue(t) - LC_svc(t)`)
  and M2 (`LCA(t) = LC_svc(t) - SAR(t) [IFIE_BEL(t) + RA_fin(t)]`), both at 1e-12 x P_g, and, on
  `NewGroup`, I14 (CSM(0) = 0 in an `ONEROUS` group, LC(0) = 0 in any other, within 1e-12 x
  P_g: contracts that each tie G3's exact per-contract test can leave a group balance of a few
  ulps, which was a RunError) on every run. M1 and M2 hold only on the curve the cur runs' BEL
  and RA unwind on, so a `df_cur` at another level fails M1 in a stage that does not hold
  `df_cur` to the cur runs' files (one without `discloses_current_curve`, below). Their
  manifests disclose the approaches to
  discount rates and to the risk adjustment, three notes (the monthly CSM allocation step, the
  ignored timing convexity and the coverage-unit approach as built: `NOTE_UNITS_DISCOUNTED`,
  Method 1, or `NOTE_UNITS_UNDISCOUNTED` for `discount_coverage_units = False`), the confidence
  level and `para_120_curve_sha256`, the `df_cur` file's hash. The approach to discount rates
  no longer says the coverage units are discounted, which a stage built with undiscounted units
  disclosed. `csm.roll_notes(statics)` gives the notes; the in-force stage's are
  `csm.in_force_notes(statics)` (below).
- An entry of `Aggregation.disclosures` may be a function of the stage's statics (a read-only
  mapping of its build-time settings, as built) that returns a string or a tuple of strings:
  `build()` calls it with the statics it resolved, and one that raises or returns anything else
  is a BuildError located at the stage class, before any run; `run_aggregation` calls it again
  to record what it gives. `Product.check_statics(statics)` is the hook `build()` calls for this
  (nothing for a projection).
- A stage class may set `reads_to_run_off` (a bool; `CsmRollForward` sets it, since N(t) sums
  every later month of the units): a `RunSettings.horizon` below the cur basis's is then
  accepted only where every series the stage reads relatively is exactly 0 after it, and is a
  LoadError otherwise, from `run_aggregation` naming the basis, the run, the variable, the
  policy, the first month dropped and the last month with a value (`basis cur run ...: cu of
  policy 1 is 993.85... at t = 13, after the stage's horizon H = 12 ...: leave
  RunSettings.horizon out or set it to at least 655`), and from the series loader for a stage
  run through `run()`, which names the scenario of a series of several and that scenario's
  value. A stage cut before the run-off would release the whole CSM at H.
- A stage class may set `recognised_at_valuation` (a bool; `NewGroup` sets it, and must
  declare the `cur` basis and the scenario input `p_l`, a BuildError otherwise): P_L = P, so
  the loader requires `p_l`'s metadata to name the `rf_df` and `ilp` files every cur run read
  (`source_sha256`, `ilp_sha256`), `tau0 = 0` and `rebased_from`, and the provenance check then
  rebuilds `p_l` from them bit for bit. A `p_l` without that metadata (a bare discount file, a
  curve made by hand), re-based at another month or composed with another ILP or discount file
  is a LoadError naming scenario input `p_l`; so is a cur basis whose runs read different
  `rf_df` or `ilp` files. The manifest's `locked_in` block records the verified curve under
  `p_l` (`sha256`, `tau0`, `source_sha256`, `ilp_sha256`, `composed_from`). `run()` refuses
  such a stage, also one without member rules, with a LoadError before anything is written
  (`... locks in the composed curve of its cur basis's runs (P_L = P), which only
  run_aggregation can verify; run it with halley.aggregation.stage.run_aggregation`): it reads
  no basis run, and no invariant sees a wrong `p_l`. A variable may not be named
  `reads_to_run_off` or `recognised_at_valuation` in a stage.
- Member rules: `halley.aggregation.MemberRule(name, check, basis="cur", *, reads=(), doc="")`
  is a rule on every member of a basis, declared in the new stage attribute
  `Aggregation.member_rules`. `run_aggregation` evaluates each on every run of its basis with
  a `MemberView` (the run's verified model point copy as `mp`, the stored outputs the rule
  declares in `reads` through `output(name)`, the valuation date, J): a failure is a LoadError
  naming the rule, the run, the first failing row of its model point copy and the row's
  policy, and the file the run read its model points from (`member rule G3 ... fails for 1 of 1
  members of basis cur run ..., first at model points (<copy>): row 0, policy 1, which the run
  read from <file>`); a check that returns no booleans, or a shape other than one bool or one
  per row of the copy (a `(1,)` answer for a run of several rows is refused), a rule whose
  `reads` the run did not store, an output read through `output()` that the rule does not
  declare in `reads` (so the manifest's record of its reads is true, D21) and a basis the stage
  does not declare (a BuildError at the rule's line) are errors too. A rule whose own code
  raises, returns no booleans or another shape, or reads an undeclared output is a LoadError
  located at the rule's declaration: `LoadError` gains an optional `loc`, as `BuildError` and
  `RunError` have, and its message ends with the location when one is set. The manifest gains
  a `member_rules` block (each rule's `rule`, `basis`, `reads` and the `runs` it held for, with
  their number of `members`), which `replay_aggregation` carries over: a replay reads no basis,
  so it cannot run the rules again. `NewGroup` declares G5 (every contract issued on the
  valuation date) and G3 (a contract is onerous, by its run's `onerous` output, exactly in an
  `ONEROUS` group; at one scenario, `require_deterministic`), so `NewGroup` reads a
  deterministic basis (J = 1) until M12 (annuity spec, G6). `run()` refuses a stage that
  declares member rules, as it refuses one that reads a previous run or one recognised at the
  valuation date: it sees only the group series.
- `halley.aggregation.inforce` (exported by `halley.aggregation`) builds the inputs of an
  in-force group's runs at V as files that name their sources: `expected_in_force(prev, tau,
  cur_mps)` (the OLD run's model points: the previous projection run's `pols_if(tau)` at full
  precision as `pols_if_init`, the actual instalment at V, every previous model point kept;
  `halley.derived_from`, `halley.tau`), `check_membership(prev_mps, cur_mps)` (G4: no policy
  joins, no group key changes), `rebase_expense_table(table, cpi_act, tau, price_base)`
  (`maint_pm x I_act(0) / I_act(-tau)` priced at V; `halley.rebased_from`,
  `halley.cpi_act_sha256`, `halley.price_base`), `composed_curve(discount, ilp_table,
  horizon)`, `locked_in_curve(discount_rec, ilp_table, tau0, horizon)` (`p_rec` and `p_l`,
  with `halley.source_sha256`, `ilp_sha256`, `ilp`, `tau0`, `curve`) and
  `locked_in_cpi(cpi_act, cpi_rec, tau0, horizon)` (`halley.act_sha256`, `rec_sha256`,
  `estimated_rows`, `tau0`). Every such file also carries `halley.helper`, naming the helper
  that made it, and the curves and the CPI splice `halley.rebased_from`. A discount or CPI file
  of several scenarios given to a curve, CPI or rebase helper is refused through
  `require_deterministic`, the one J = 1 check of D4, with that check's message naming the
  file. Importing the package registers the recomputations `recompute_rebased` and
  `recompute_derived` for the two provenance keys, so the stage loader recomputes every one of
  these files a stage or its bases read, from the sources it names, and requires bit equality.
  `recompute_derived` also holds the `OLD` file's `halley.tau` to the period, `mi(V) -
  mi(V_prev)` with V_prev the valuation date of the run it names, so a file built at another
  month is a LoadError rather than a file that rebuilds bit for bit from its own metadata.
- `halley.runtime.loader.relatively_read(graph)` (the series some formula reads at a relative
  month) is public; `load_inputs` and `load_series` take `run_off`, which `run()` sets from the
  stage's `reads_to_run_off`.
- `halley.aggregation` also exports `run_aggregation` and `replay_aggregation`.
- `halley.products.annuity_data` builds the in-force group's reference inputs (in-force spec
  sections 3, 5 and 13): the model points of each variant and class at recognition and at V,
  the groups rows (at V with the three locked-in hashes; `with_previous_run` adds `prev_run_id`,
  `csm_prev` and `lc_prev` from the recognition run), the tables of versions 1 and 2, the
  curves, the CPI paths and the files the helpers derive from them; `model_points_table` takes
  the `valuation_date` its metadata records. `docs/spec/in_force_group_inputs.py` writes them to
  `docs/spec/in-force-group/*.parquet`.

### The in-force group (M11)

- `halley.aggregation.InForceGroup` (exported by `halley.aggregation`) is the stage of a group
  measured before: one reporting period `(V_prev, V]` closed in the in-force spec's order
  (section 9), then the months after V on the `cur` basis from the closing balances. It reads
  the bases `old`, `new` and `cur`, the series `fcf_old` and `fcf_new` (`bel` + `ra` of the old
  and new runs, from month 0) beside the roll-forward's, the scenario inputs `p_l`, `df_cur` and
  `p_rec` (the composed curve at recognition, value column `rf_df`, rule `p_rec(0) = 1`), and
  groups built with `group_schema(previous=True, locked_in=True)`, so the loader fills
  `f_prev_0`, `f_prev_tau`, `lc_prev_tau`, `p_g_prev`, `tau0`, `tau`, `i_acc` and `i_rec` from
  the previous run and the dates. It computes and stores `p_g` (P_g carried forward), the
  group inputs of the close as the lines it reads them through, `csm_open_per` and
  `lc_open_per` (the opening lines of in-force 11.1's tables, `csm_prev` and `lc_prev`) and
  `cu_act_per` (`cu_act`), `delta_fcf_l`, the close's `csm_acc` (`(csm_prev x P_rec(i_acc)) /
  P_rec(i_rec)`), `acc_per`, `lc_alloc_per`, `lca_per`, `rev`, `raw`, `csm_adj`, `lc_adj`,
  `csm_fs`, `lc_fs` with its two lines of the loss-component table, `lc_loss` (where positive)
  and `lc_reversal` (where negative), `n_0`, `u_per`, `r_per`, `csm_0`, `lc_0` and `lrc_0`, and
  the analysis of change `aoc_a` to `aoc_e` with `aoc_ba`, `aoc_cb`, `aoc_dc` (delta_FCF_L bit
  for bit) and `aoc_ed`, in the scalars file beside the roll's monthly lines. Its invariants
  are K1 to K7 (in-force 12.2: K1 and K2 sum the stored lines of the period's tables; K4's
  inequalities and "at most one positive" exactly; K5 where `cu_act > 0` and `CSM(0) > 0`; K7
  against the previous run's `lc(tau)`) beside I11 to I13, M1 and M2, every run. Its input rule
  `the locked-in curve covers [0, tau0 + H]` holds `p_rec` to the months of in-force 12.4. A
  later period's stage names the previous in-force run by `prev_run_id`, with `tau0` counted
  from the recognition date and `tau` from `prev_valuation_date` (in-force 3.4, 4).
- The stage also reads a table `cpi_act`, the actual CPI history at V (key `u` in [-26, 0],
  column `cpi`, metadata `estimated_rows`), which no formula reads: the loader rebuilds the old
  and new runs' rebased expense table and locked-in CPI path from it, and as an input of the
  stage it is copied into the run directory, enters the run id and carries its C9 estimate
  flag into the manifest. Every `run_aggregation` of the stage passes
  `tables={"cpi_act": ...}`. A limit of this version: the key is a dense grid, so the history
  holds exactly the months -26..0 (a longer one is a LoadError naming the range), and the
  expense rebase, which reads `I_act(-tau)`, serves periods of at most 26 months.
- Its manifest discloses the analysis of change's labels under `analysis_of_change` and five
  notes (`csm.in_force_notes`): the period's single release with the monthly step after V
  (`NOTE_PERIOD_RELEASE`, C12, in place of the monthly-step note, which says a period's release
  is the sum of its monthly releases), the ignored timing convexity, the coverage-unit approach
  as built, the C11 period allocation and the C17 locked-in runs.
- A stage class may set `rebuilds_locked_in` (a bool; `InForceGroup` sets it, and its groups
  must declare the locked-in hashes and, when it reads the basis `old`, `prev_run_id`, and the
  stage itself the scenario inputs `p_l` and `p_rec` and the table `cpi_act`, a BuildError at
  the class otherwise: without them the loader would check those files only when a run's
  metadata named them, or never). The loader then holds its locked-in runs and files to what
  they were built from, each a LoadError naming the file before anything is written, since
  each of these rebuilds bit for bit from its own metadata and passes every invariant:
  - every input checked against the group's locked-in hashes (the stage's `p_l` and `p_rec`,
    the old and new runs' `rf_df` and `cpi`) carries `halley.rebased_from`, so the provenance
    check rebuilds it, and `halley.tau0` equal to the group's months from its recognition date
    to V, never from `prev_valuation_date`; the manifest's `locked_in` block records the `tau0`
    held;
  - each curve is the one it is read as (`halley.curve`: `p_l` for the stage's `p_l` and the old
    and new runs' `rf_df`, `p_rec` for the stage's `p_rec`), the old and new runs discount on
    the stage's own `p_l` file, and they read an `ilp` table of exactly 0.0 when they read one
    (C17: P_L holds the premium already, so another would count it twice);
  - every file of a basis run that names the actual CPI history it was built from
    (`halley.act_sha256` of the locked-in CPI path, `halley.cpi_act_sha256` of a rebased expense
    table) names the stage's table `cpi_act`; the cur runs' CPI path, a plain file that names
    no history, holds the table's rows bit for bit from its first month to `u = 0`, in every
    scenario (in-force 5.3: `I_cur(u) = I_act(u)` for `u <= 0`), and a LoadError names the copy,
    the scenario and the first month that differs; its month column is `t` or `u`, as the
    scenario loader takes it (a path keyed on `t` was refused as `column u is missing`);
  - the runs of the bases `old`, `new` and `cur` are ordinary projections of one model (in-force
    5.5, C17): each policy is projected by one build in every basis that holds it, its
    `ir_sha256` (the class with its build-time settings, such as `include_inv_exp` or
    `cpi_lag_m`) on one backend, and each old run by its previous run's build (below), so a
    subclass under another name is caught too; a LoadError names the policy, both runs and their
    builds. Two builds of one class, each projecting its own policies in every basis, are
    accepted (they were refused). A basis run whose manifest records no model name,
    `ir_sha256` or backend name is a LoadError naming the run and the key;
  - each old run's model point copy carries `halley.derived_from` naming a run of the previous
    aggregation run's cur basis (the expected in-force at V, never retyped), its `ir_sha256` is
    that run's (the backend may differ between periods), and its non-financial tables are that
    run's by sha256, or a rebase of one (`halley.rebased_from`, the expense table in V prices);
    each new run reads the model point file of the cur run that holds its policies.

  A variable may not be named `rebuilds_locked_in` in a stage.
- The provenance check finds a source in the previous runs back to the group's recognition (each
  named by the `previous` block of the run after it) and the runs they read, beside the
  previous run's and the stage's bases: in a later period the files a group locks in at
  recognition are read only by the recognition run's bases.

### The Numba backend (M6)

- `build(model, *, backend="numpy", **statics)`: `backend="numba"` generates the Numba backend
  from the same IR and schedule (`pip install halley[numba]`; without numba the build is a
  BuildError saying so). A name outside the registry, `halley.codegen.BACKENDS`
  (`"numpy"`, `"numba"`), is a BuildError listing it. `CompiledModel.backend` names the
  backend, and `halley.codegen.Backend` is the protocol a backend implements (`name`,
  `artefact`, `generate`, `compile`, `run_batch`, `versions`).
- `backend` is now build()'s own argument, so a build-time parameter (`Static[...]`) named
  `backend` is a BuildError at the class that declares it.
- The generated module calls one compiled kernel per component in schedule order, with the
  NumPy backend's arrays in and out (the same shapes, month quantities exactly 0 at t = 0, Int
  and Bool arrays int64 and bool) and its run-time checks: a probability outside [0, 1], a
  lookup key outside its table, an `at()`, `scen.<name>.at()` or `series.<name>.at()` month
  outside its domain, and a non-finite result are RunErrors with the NumPy backend's text, its
  model line and the same policies (or groups) named.
- A run directory of the Numba backend holds its generated module as
  `generated/numba_backend.py` (a NumPy run's stays `generated/numpy_backend.py`).
- The manifest's `backend` block of a Numba run records `numba`, `llvmlite`, `numpy`,
  `using_svml` (whether Numba uses Intel's SVML), `processor` (the operating system's name for
  the machine), and `cpu_name` and `cpu_features`, the CPU and the instruction set extensions
  LLVM compiles the kernels for (`NUMBA_CPU_NAME` and `NUMBA_CPU_FEATURES` when set, else the
  host's), beside `pyarrow`, `python` and `platform`. A NumPy run's block is unchanged.
- The run id of a NumPy run is unchanged (its identity's backend entry is still the name and
  the NumPy version); a Numba run's identity adds the Numba and llvmlite versions, so the same
  inputs on the two backends are two runs. `replay_aggregation` treats a change of `numba` or
  `llvmlite` as one that moves the run id, and of `using_svml`, `processor`, `cpu_name` or
  `cpu_features` as one that can move a value. A stage run is replayed on the backend that
  made it: `replay_aggregation` with a stage built for the other backend is a LoadError saying
  which backend to build it with.
- The generated module is written to `<temp>/halley-numba-<user>/<ir_hash>-<text hash>.py`
  (the text hash is the first 16 hex digits of the sha256 of the module's text), or under the
  directory named by `HALLEY_NUMBA_CACHE`, and imported from there; Numba caches the compiled
  kernels beside it, so the first build of a product compiles every kernel (tens of seconds) and
  later builds, in any process, load them. A file holds one text only: two checkouts of a
  product (whose modules name different source files in their messages) have a file each, so
  neither overwrites the module the other is importing, and the module a run imported is the
  one its run directory holds as `generated/numba_backend.py`. The default directory is one
  per user, created private to them (mode 0o700 where modes apply), since on Linux and macOS
  the temporary directory is shared and Numba loads the compiled kernels it finds there: one
  that another user owns or can write to is a BuildError. A cache directory that cannot be
  created or written is a BuildError naming `HALLEY_NUMBA_CACHE`.

### Process parallelism (M6)

- `RunSettings(workers=n)` spreads a run's batches over `min(n, batches)` worker processes
  (`halley.runtime.workers`), on either backend; with `workers=1`, the default, or a single
  batch, every batch runs in the calling process as before. The workers are a
  `ProcessPoolExecutor` started with the spawn method (Windows has no other). Each is sent
  once, at its start, the generated source, the backend's name, the graph, the model's class
  and the inputs without their files' tables and bytes (`Inputs.for_workers()`), and compiles
  the batch function itself (on the Numba backend it loads the kernels the build cached); a
  task is one batch. The caller takes the results in ascending batch order, assembles the
  arrays and merges the invariant results as one process does, and raises the error of the
  first failing batch in that order.
- At the same `batch_size` a run with several workers is the run with one bit for bit, on both
  backends: the same run id, output files, arrays and invariant results, and the same error,
  message and location, when it fails. `workers` is recorded in the manifest
  (`settings.workers`) and is still not in the run id.
- A worker imports the model's class from its module, so with `workers` above 1 a model class
  defined inside a function, or in an interactive session, is a RunError at the class before
  any process starts (it runs with `workers=1`), and a script that starts such a run must
  guard its entry point with `if __name__ == "__main__":`. A worker that cannot start (its
  model's module fails to import, say) fails its batches with a RunError saying so; one that
  ends without a result, as when the system ends it for its memory, is a RunError suggesting a
  lower `batch_size` or fewer workers. A worker runs the checks under the caller's NumPy
  floating-point error handling (`np.geterr()`).
- An aggregation stage's batches are its groups, so `run(..., series=group_series(...))` over
  several groups spreads them over the workers too. A stage run made by `run_aggregation` (and
  its replay) measures one group (G1), a single batch, so it runs in the calling process
  whatever `workers` says.
- `BuildError`, `UnitError`, `LoadError` and `RunError` pickle with their message, location
  and notes (they share a base whose `__reduce__` passes the message and the `Loc`, where an
  exception's default pickling passes the formatted text alone), so one raised in a worker
  reaches the caller whole; its `__cause__` there is the worker's traceback.
- `halley.runtime.loader.ModelPoints` no longer holds the schema (its derived fields and rules
  are functions, which do not pickle, and nothing read it): `CompiledModel.schema` is the
  schema. `ReadFile.without_data()` is a file without its table and bytes.

### The GUI

`python -m halley.gui`, or the new `halley-gui` command, starts a web server on the loopback
interface and opens the GUI in the browser (`halley.gui`, `docs/gui.md`). It adds no dependency: the
server is the standard library's, and the pages are plain HTML, CSS and JavaScript inside the
package, which load nothing from elsewhere. Builds and runs are jobs that run one at a time, the one
running shown in a tray in the page's bottom corner; one that ends when its page no longer shows it,
or under an open dialog (beneath which the page is not drawn again), says so there, with a chip that
keeps what went wrong, opens it in a dialog, and stays until the tray is cleared or two minutes
after it was last opened, never going while the pointer or the focus is in the tray, and in its card
while the card is shown, with a link to what it would have shown. A chip that comes while the
pointer or the focus is in the tray leaves the chips there where they are, **Clear** lies outside
the chips' scroll and, pressed from the keyboard, gives the focus back to the control it came to the
tray from, and the page keeps room at its end for the tray. The engine is unchanged but for
`explain`'s reads of scenario inputs and the tracer's check of the inputs a model declares, below.

- The Models page lists the reference products and the IFRS 17 stages, and imports a model file of
  the user's as the module named after the file, with its folder on `sys.path` so that a run's
  worker processes import it too. Importing runs the file's top-level code. A file named like a
  module that is already importable (`json.py`) is refused, and a file that fails to import keeps
  its previous models. A file's helpers are the modules of the user's own it uses: those its import
  statements (relative ones too) and namespace name, those its import brought in, and those a build
  of its models imports, each noted on the thread that imported it, so a load during another model's
  build keeps its own; an edit to one, or to a loaded file, marks the files changed. A helper that
  puts an object, a module object or another module in its own place in `sys.modules`
  (`sys.modules[__name__] = _Basis()`) is followed by the file its import ran, the object's own
  attributes never read, and one that runs another file as itself (importlib's recipe) by both
  files; a module built into Python is never taken for a file of the user's named like it. **Reload
  the files** runs every loaded file again, in the order they were loaded, after taking out of
  `sys.modules` every module of the user's own the catalogue has seen them use or import, with its
  submodules (and the attribute a namespace package holds for one), so they import them afresh, one
  copy of each, as a new Python process would; every model is then a new class, and a file an
  earlier one's import ran that put something else in its own place is refused, as a load of it then
  is, never run twice. On an exception every module, file and model is as it was. A loaded file that
  is gone is let go at a reload, with its models. Builds, checks, runs and reruns hold the engine
  while they work, and a reload is refused while one does; a check's and a run's imports are noted
  as a build's are. A model file imported before the GUI loaded it, by an import no watch saw, shows
  as changed until a reload, since nothing records which version of it that import ran; so does a
  file saved while it ran (its modification time before and after its run differ, so a file dated
  ahead of the clock is not taken as saved). A load or a reload that ends once the user has gone to
  another page leaves that page as it is but for the side lists, and says so in the tray. A failed
  first load or reload takes out what its own import brought in of the user's (modules and folders
  imported as namespace packages alike); a failed first load that a build, check or run on another
  thread ran beside leaves what it imported, which that work may hold (`import h`, a package's `from
  . import h`, `from h import RATE`), and records it, so an edit to it shows and a reload takes it
  out. Its traceback is written out before the clean-up, since Python's suggestion of a near name
  imports the module again; an error whose writing trips (a `__getattr__` or a `__str__` of the
  user's that raises anything, `SystemExit` too, or a `name` that cannot be set) is still written,
  at the file's line, and the clean-up runs whatever happens. A build takes the backend and the
  build-time parameters the class declares (found as the tracer finds them), each in a field of its
  type: a check box, a number, a text, a choice of an enum's members, a date, or JSON for a tuple, a
  list or any other type, in which an enum's member is its name (or as Python writes it, never a
  label of its class's own `__str__` or `__repr__`) and a date its ISO text (a list's items read by
  its item type, as a tuple's are); a flag's members are typed joined by `|`; a dict's keys are read
  by its key type too (an enum's member by its name, a date in ISO form, a tuple as a list, a
  sequence as the tuple a key holds), and from a manifest as it records them (`str()` of each key,
  tuples of enum members and dates included, inside a sequence too); a `NewType` is read as its
  supertype, a `type` alias as its value (a generic one with its arguments in their places) and
  `Annotated[T, ...]` as `T`, while an alias whose value cannot be had (one that names what its
  module does not define) is taken as JSON gives it, as is a class `isinstance` cannot check (a
  `TypedDict`, a `Protocol` that is not runtime-checkable); a list a union's option that the GUI
  does not read (an `Iterable`) holds goes to it before one that reads only part of it. A setting a
  run's manifest cannot record exactly is warned of at the build, since the run cannot be rerun from
  its directory: a dict that is not the class's own default and whose keys are not in the order a
  manifest writes them (sorted as text by `str()`, which the warning gives), a NumPy number, an
  object recorded as its repr; and an enum's member its class's own `__repr__` writes as it writes
  another, unless the member makes no difference to the model. A rerun tries each dict recorded as
  its default is both ways, all at their defaults and all sorted first, up to 64 mixes, tracing each
  for its IR hash; a build that gives a mix beyond those warns of it, and its warnings name every
  setting that keeps its run from a rerun, beside one whose record does not read back at all. A
  manifest's date and time with a time zone of the `datetime` module's (an offset, as the form reads
  one) reads back, but for a date and time that is itself a dict's key, which a manifest records as
  its `str()`, without its zone's name or its `fold`, so the build warns of one that has either; a
  float that is not finite reads back at any depth of a tuple key. The server writes each value as
  Python does (`1` and `1.0` apart), and the form sends only the fields changed from what it holds,
  the class's defaults or the shown build's own settings, so an untouched form builds what
  `build(cls)` builds (a check box for a parameter without a default builds False) and an untouched
  **Build again** builds the shown build again, with the same IR hash. Build again from a build of
  the files as they were carries its settings to the class as it is now: an enum's member by its
  name, a setting at its old default as the default now. A build records its `BuildWarning`s and
  lists every variable with its unit, role, timing, recursion direction, formula, source, reads and
  readers.
- A run takes a reference product's example inputs (the term assurance's reference case 1 and
  the annuity spec's worked example, built in memory, and lifelib's inputs for `BasicTerm` in a
  checkout) or files, Parquet or CSV (read with `pyarrow.csv`), one per declared input, with the
  `RunSettings` fields; outputs a build does not compute are refused before the run, and an
  empty choice of outputs is sent as `None`, which stores the same variables under the same run
  id. A run of the reference example has the run id of `run()` on the same inputs. **Check the
  inputs** calls `load_inputs` as `run()` does, without projecting.
- A run's page shows its manifest, a variable's lanes and portfolio total over the months, the
  policies' values at a month beside their model point fields, the invariant results, every
  input copy (read back through `read_copy`, so its hash is checked) and the run's files.
  **Explain** gives `halley.explain`'s value, formula and inputs, each input with a target that
  opens it (a variable at a month, a table, a scenario input), notes a month that takes an
  `init` or `terminal` value, and recomputes the value with `recompute`, comparing the two bit
  for bit in the stored value's type (within 1e-9 relative for a Numba run).
- A run read back from its directory is explained after a rerun from its own directory: the model
  its manifest names is built with the recorded build-time settings, read back by their declared
  types as the manifest records them (an enum's member from its repr, or its value for an int or
  str enum, a date from its repr, through tuples and unions; a value that does not fit its type,
  which `build()` took, as recorded; a setting declared an enum, a date or another class whose
  record does not read back is refused before the build; a dict recorded in another order than it
  was given is tried as recorded too), and the recorded backend (a model whose IR hash does not
  come out as the run's is refused, saying the model or a setting the manifest cannot record
  exactly is why, and so is one a script defined in `__main__`) and run on the directory's copies
  of its inputs, a stage's group series included, with `RunSettings.from_dict` of its recorded
  settings, into `<runs>/reruns`; the rerun is compared with the run by run id and bit for bit,
  variable by variable. An aggregation run made by `run_aggregation` is replayed with
  `replay_aggregation` the same way.
- Comparing two runs lists every entry their run ids cover (for a projection the IR hash, each
  input's hash, the settings in the identity, the backend and its versions and the horizon; for
  a run of `run_aggregation` also its groups file, membership, previous run and disclosures)
  with whether it differs, and `halley.diff`'s largest difference in every variable both hold.
- Every API request must carry the session's token, a random string the server prints in the
  address it opens (`#token=...`, a fragment no browser sends to a server), and name the
  server's own address in its `Host` header (for a server listening on every interface, any IP
  address with its port); a cross-origin preflight is not answered. The
  requests a page's address alone can trigger read only the run directories under the runs
  folder, those the session made and those the user opened, so an address another site sends
  the GUI's tab to cannot make the server open a path of its choosing. The pages are served
  under a content security policy that loads nothing from another origin, and set every value
  as text.

### explain and scenario inputs

- `explain` (and `recompute`) read a scenario input at a month outside its file's months as an
  error of that input alone, `<scen.<name> read at month <u>, outside <first>..<last>>`, as they
  read a variable outside 0..H. Before, a month before the file's first was read from the file's
  last rows, so a boundary month's explanation could list another month's value without a word,
  and a month past the file's last failed the whole explanation with an `IndexError`. Only a
  boundary month (``t = 0`` or the horizon) can read such a month; a run's values are unchanged.

### The tracer and the inputs a model declares

- A model whose `assumption_tables` or `scenario_inputs` is one contract written without its
  trailing comma (`assumption_tables = (MORT)`) fails its build with a BuildError at the class,
  as `invariants` and `input_rules` already did, rather than with a TypeError that names no line.

### Known limitations

- `recompute` and `explain` evaluate a power with a non-literal array exponent (a model point
  field, a scalar variable or a scenario value) through NumPy's power loop, so where NumPy took
  its sqrt, square or reciprocal fast path for an exponent of 0.5, 2 or -1 (in a batch of one
  policy, for example) they agree with the stored value within 1e-12 relative, not bit for bit.
- Which non-literal power exponents take NumPy's fast path was measured on NumPy 2.5 (2.5.4 on
  Windows) only: another NumPy may choose differently, which can move the last bit of such a
  power between NumPy versions (within D5's 1e-12), and the documented shapes would need
  probing again.
- For literal and 0-d exponents, `recompute` and `explain` mirror NumPy's fast path by raising
  a one-element array, and for array exponents and for `exp`, `log`, `expm1`, `log1p` and
  `sqrt` they call the NumPy loop on a one-element array, which assumes NumPy dispatches one
  element as it does a batch: to the C library on Windows and on Linux without AVX-512, to its
  own kernels on an AVX-512 Linux machine. The tests check this on the machine they run on, and
  skip where its power loop cannot be told from the fast path.
- A per-group month `self.series.x.at(self.group.k)` outside a series' months is a RunError
  naming the groups when the batch runs, not a LoadError; a constant `.at(k)` is checked at
  load, as `at()` and `self.scen.x.at()` are.
- A group with no member in the runs of a basis is a LoadError, so an in-force group whose
  members have all left by the valuation date cannot run through `InForceGroup` (its new and
  cur bases hold none of them), and the release of its remaining CSM in that last period is not
  modelled.
- An `Invariant` copied with `dataclasses.replace()` keeps the location of the invariant it
  copies, so its errors point at the original declaration, not at the `replace()` line.
- A series copy in a stage's run directory holds the months the stage read (its `months`)
  beside its basis's `basis_horizon`, which may be larger; the loader holds a series' months to
  at most `basis_horizon` and to the stage's horizon, not to equality with `basis_horizon`.
- A stage without a `cur` basis takes its default horizon from its time series' months; one
  whose only series are scalars of other bases needs `RunSettings.horizon`.
- The manifest repeats a basis's membership in every series entry of that basis
  (`inputs.series.<name>.members`), beside `inputs/membership.parquet`, so it grows with the
  number of policies times the number of series.
- `group_schema` is imported from `halley.authoring.contracts`, and `register_provenance`,
  `find_run` and `load_aggregation_inputs` from `halley.aggregation.stage`; they are not
  re-exported from `halley` or `halley.aggregation`.
- A run named by `prev_run_id` or `derived_from` is found by reading every
  `runs_root/*/manifest.json` on each lookup: there is no index, so a lookup slows down with
  the number of runs filed there; a run filed in a nested directory is not found; and a replay
  or copy of a run filed under the same `runs_root` makes its run id ambiguous (a LoadError), so
  file replays and copies outside `runs_root`.
- Every run with `workers` above 1 starts its own worker processes, and each imports Halley,
  NumPy, pyarrow and the model (and, on the Numba backend, loads the cached kernels) before its
  first batch: a few seconds on the development machine, so workers pay off for runs whose
  batches take longer than that. Each batch's arrays of every kept variable are pickled back to
  the caller, which holds all of them as one process does; on Windows a run starts at most 61
  workers (the process pool's limit there). A worker applies Python's default warning filters,
  not the caller's.
- A run's `manifest.json` is not itself hashed: every file it lists is verified against the
  hash it records, and a later stage records the sha256 of the previous run's manifest, but an
  edit to a manifest that keeps its listed hashes consistent is not detected on load.
- `run(..., stage_record=)` takes a `StageRecord` as given: one built by hand runs none of the
  loader's checks.
- A stage's bases use one policy-id type, all integers or all strings; bases that mix them are
  a LoadError.
- The reference input files under `docs/spec`, and the hashes recorded in their metadata and in
  the in-force groups files, are pyarrow's writer bytes: a pyarrow upgrade can change them
  without changing a value. Rerun the `docs/spec/*_inputs.py` scripts, which the committed-files
  tests flag.
- A `MemberRule` is evaluated on one run of its basis at a time, so a rule across several runs
  of a basis (several product classes together) cannot be written as one; `replay_aggregation`
  does not evaluate member rules again, it carries over the original run's record of them.
- The CSM stages read deterministic bases only (J = 1), and a J > 1 projection run is a
  LoadError (`require_deterministic`): `NewGroup`'s G3 at J > 1, the para 47 test on each
  contract's probability-weighted mean FCF, is open with the specification's G6, and M1 and M2
  assume the cur runs discount on `df_cur`, which holds at J = 1 only (with pathwise
  deflators the mean BEL does not roll forward on one curve).
- The stage table `cpi_act` holds exactly the months u = -26..0 (its key is a dense grid), so
  trim the actual CPI history to those 27 months (a longer or shorter one is a LoadError naming
  the range); the expense rebase reads `I_act(-tau)` from it, so an in-force reporting period is
  at most 26 months long; and the cur runs' CPI path is held to `cpi_act` for u = -26..0 only
  (no accepted in-force run reads an earlier month: the locked-in path starts at u = -26, and a
  CPI lag above 15 months fails the old and new runs' coverage check).
- The in-force input rule "the locked-in curve covers [0, tau0 + H]" answers once per run (G1:
  one group per aggregation run), so its message names the file, not a group; it needs a
  per-group answer when G1 is lifted.
- A custom stage that locks in a basis (`group_schema(locked_in=True)`) has its locked-in files
  checked by hash only: the rebuild of each from the group's basis and `tau0`, and the
  actual-CPI, one-build and previous-basis checks, run only when the stage sets
  `rebuilds_locked_in = True`, as `InForceGroup` does.
- A failed run-time invariant on a stage's scalar outputs (K1 to K7 of the in-force close) names
  the invariant, the group and both sides, but no month, since the close is one value per group.
- An in-force run's manifest records its bases (run id and manifest sha256) and the provenance
  rebuilds, but not the structural checks of the old and new bases (the previous tables by
  sha256, the new runs on the cur model point files, one build per policy); `replay_aggregation`
  reads no basis run and does not run them again.
- A model point with no survivors at V keeps its previous `instalment_pm` in the old basis (the
  specification is silent; in-force open question 23 asks its author), so with an anniversary
  inside the period its expected escalation is reported in C - B rather than in delta_FCF_L.
- The Numba backend reproduces the NumPy backend's run-time checks, not the exceptions its
  Python arithmetic raises outside them. Wherever both operands of a division or modulo are
  plain Python numbers in the NumPy backend's code (literals, `self.run.*`, and `t` inside a
  recursive component: in a recursion, an `init`, a `terminal` or a variable without a time
  dimension), a zero divisor is a RunError reporting the ZeroDivisionError there, while the
  Numba backend gives inf, nan or 0. Stored, a float such value fails the finiteness check;
  in the discarded branch of a `where()`, as in the guarded `where(t > 3, 1.0 / (t - 3), 0.0)`
  inside a recursion, the Numba run succeeds where the NumPy run fails.
- An input declared with unit `Int` but held as floats, a model point field of kind `number`, a
  scenario input or a group series, is not held to whole numbers; these are the only Int
  values that can hold a fraction (a float literal beside an Int is a UnitError). Used as a
  month (`at()`, `scen.<name>.at()`, `series.<name>.at()`) such a value is tested against the
  domain before it is truncated, on both backends (12.5 is outside months 0..12, 3.7 reads
  month 3); used as a table key it is an IndexError on the NumPy backend, whatever its values
  (the array is a float array), and truncated to an integer after the range check on the Numba
  backend.
- The Numba kernels run on one core: the policy loop can run in parallel (`numba.prange` over
  blocks of 64 policies, `halley.codegen.numba_backend.PARALLEL`), but that is off, since it
  more than doubles the compile time for little gain on memory-bound kernels; spreading
  batches over processes (`RunSettings.workers`, design section 12) is the way to use several
  cores.
- The first build of a model on the Numba backend compiles every kernel, about half a second
  to a second per component on the development machine (the in-force stage, 60 components,
  about half a minute). The module's text names the model's source lines for the messages, so
  a change to it for the same IR hash (another Halley, the model's lines or file moved, two
  checkouts of one model) writes another module file and compiles its kernels again; the
  earlier files and their kernels stay in the cache directory, which is not pruned (delete it
  to reclaim the space).
- On the Numba backend the probability and finiteness checks still run in NumPy over every
  stored array after the kernels, as the NumPy backend's do, which is a large share of a
  batch's time.
- `recompute` and `explain` interpret the NumPy backend's semantics; for a Numba run they agree
  with the stored values within 1e-9 relative, not bit for bit.
- The GUI cannot stop a build or a run that has started (the engine has no way to stop a run
  part way); it runs them one at a time and cancels only one still waiting, and Ctrl+C waits for
  a started one to finish. It does not run an aggregation stage, which runs on projection run
  directories through `run_aggregation`; it reads, replays and explains a stage's runs. A run's
  page reads the whole run into memory, as `load_run` does, so a very large run is slow to open,
  and the session holds the runs it has used most recently, up to eight it made, six it read
  back and about 1.5 GB in all; a run it made and let go of is rerun to be explained again. A
  model file is imported as a top-level module, so a file inside a package that imports its
  siblings relatively does not load.

### Notes on determinism

- A power whose exponent is not a literal, such as `x(t) ** self.mp.k`, can differ in the last
  place between two batch compositions of the same run (`RunSettings.batch_size`) when the
  exponent's value is 0.5, 2 or -1, because NumPy chooses its sqrt, square or reciprocal fast
  path from the shapes of the batch. Reruns with the same batching stay bit-identical (D19) and
  another batching agrees within 1e-12 relative (D5), which is what the annuity specification's
  "Built-in invariants" promise. `recompute` and `explain` evaluate such a power through
  NumPy's power loop, so they agree with the stored value within the same tolerance, and bit
  for bit where the batch took the loop. A literal exponent (`x ** 0.5`, the mid-month
  `pv_month`) takes one path at every batch size; the shipped products raise only to literals
  and to the month index (`v_m ** t`, `(1 + i) ** (-t)`, the `-12 / t` of `ann_spot`), which
  took the loop at every batch shape measured over a window of two or more months. Over a
  one-month window (H = 1 for a formula
  from t = 1) the month column is a size-one array and takes the fast path at every batch size,
  so `recompute` and `explain` differ from the stored value in the last place there; the shipped
  products' month exponents start at t = 0 or reach -1 only at t = 12, so they are unaffected.
- `recompute` and `explain` take NumPy's fast path where the generated code does for a 0-d
  exponent (a keyless table's value) against a scalar base, as in
  `self.tables.rates.f["a"] ** self.tables.power.e`, so they agree with the stored value bit for
  bit there.
- On the Numba backend reruns with the same inputs, settings and batching are bit-identical
  (the same run id and output files): the kernels compile with fast-math off, so there is no
  contraction to FMA and no reassociation, and `sum_t` adds in ascending t. Across backends
  every value agrees within 1e-9 relative under the product's scales, with exact zeros in the
  same places and Int and Bool values equal, but not bit for bit: a power is LLVM's pow with a
  float exponent, where NumPy may take its fast path, and `exp`, `log`, `expm1`, `log1p` and
  `sqrt` are the C library's, where NumPy has its own vectorised versions. A kernel evaluates
  each lane with the same scalar code whatever the batch, so a non-literal exponent gives the
  same bits in a batch of one policy as in a batch of three (tested without SVML, whose
  vectorised maths the manifest's `using_svml` records).
- The number of workers does not move a bit (D19): a batch computes the same values in any
  process of the machine, and the caller assembles the batches and merges their invariant
  results in ascending batch order. The batch composition still can, as above, so compare runs
  bit for bit at the same `batch_size` (tested with one and two workers on both backends:
  TermAssurance, a two-scenario CPI-linked annuity run in three batches and a stage over two
  groups).
- Across machines the last bits move. NumPy's float64 `exp`, `log` and `pow` are the C
  library's where NumPy has no vectorised kernel (Windows, Linux without AVX-512) and NumPy's
  own on an AVX-512 Linux machine, and glibc and the Windows CRT round a last bit differently
  on some inputs. A run's values, the input files the section 10 helpers build and the
  committed reference files therefore agree with another machine's within those bits, not bit
  for bit: the provenance check allows `LIBM_ULPS` and records the distance, the
  reconciliations hold the specs' tolerances, and the tests that compare a committed file with
  its generator, or scenario 0 of a two-scenario run with the one-scenario run, allow 1e-12
  relative off the machine that wrote the files. D19's bit-identical reruns are a per-machine
  statement.
