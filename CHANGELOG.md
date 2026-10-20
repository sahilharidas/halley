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
  helpers register their recomputations for both keys (below).
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
  with coverage units on, the last month included), I12 (both forms), I13, the design's M1
  (`LRC(t) - LRC(t-1) = IFIE(t) - revenue(t) - LC_svc(t)`) and M2 (`LCA(t) = LC_svc(t) - SAR(t)
  [IFIE_BEL(t) + RA_fin(t)]`), both at 1e-12 x P_g, and, on `NewGroup`, I14 (CSM(0) = 0 in an
  `ONEROUS` group, LC(0) = 0 in any other) on every run. M1 and M2 hold only on the curve the
  cur runs' BEL and RA unwind on, so a `df_cur` at another level fails M1 in a stage run
  through `run()`, which does not check the file. Their manifests disclose the approaches to
  discount rates and to the risk adjustment, three notes (the monthly CSM allocation step, the
  ignored timing convexity and the coverage-unit approach as built: `NOTE_UNITS_DISCOUNTED`,
  Method 1, or `NOTE_UNITS_UNDISCOUNTED` for `discount_coverage_units = False`), the confidence
  level and `para_120_curve_sha256`, the `df_cur` file's hash. The approach to discount rates
  no longer says the coverage units are discounted, which a stage built with undiscounted units
  disclosed. `csm.roll_notes(statics)` gives the notes; the in-force stage's are
  `csm.in_force_notes(statics)` (below).
- An entry of `Aggregation.disclosures` may be a function of the stage's statics (a read-only
  mapping of its build-time settings, as built) that returns a string or a tuple of strings:
  `run_aggregation` calls it with the compiled stage's, and one that raises or returns
  anything else is a LoadError located at the stage class.
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
  must declare the locked-in hashes and, when it reads the basis `old`, `prev_run_id`, a
  BuildError at the class otherwise). The loader then holds its locked-in runs and files to
  what they were built from, each a LoadError naming the file before anything is written, since
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
    table) names the stage's table `cpi_act`;
  - each old run's model point copy carries `halley.derived_from` naming a run of the previous
    aggregation run's cur basis (the expected in-force at V, never retyped), and its
    non-financial tables are that run's by sha256, or a rebase of one (`halley.rebased_from`,
    the expense table in V prices); each new run reads the model point file of the cur run that
    holds its policies.

  A variable may not be named `rebuilds_locked_in` in a stage.
- The provenance check finds a source in the previous runs back to the group's recognition (each
  named by the `previous` block of the run after it) and the runs they read, beside the
  previous run's and the stage's bases: in a later period the files a group locks in at
  recognition are read only by the recognition run's bases.

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
