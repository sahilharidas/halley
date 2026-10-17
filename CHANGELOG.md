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
