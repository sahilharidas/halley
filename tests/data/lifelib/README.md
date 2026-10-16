# lifelib BasicTerm_ME fixtures (reference case 2)

The inputs and results of lifelib's `basiclife/BasicTerm_ME` model, taken from the `lifelib`
0.17.2 package on PyPI and used by `tests/test_lifelib_basic_term.py` to reconcile
`halley.products.lifelib_basic_term.BasicTerm` to lifelib (Decisions, D23). lifelib is MIT
licensed (copyright 2022 lifelib Developers); the licence text is in `LICENSE` beside this file.
Nothing here comes from any other source.

## Inputs, values unchanged

| File | lifelib source | Layout |
|---|---|---|
| `model_point_table.parquet` | `model_point_table.xlsx` | lifelib's columns and dtypes; `policy_id` is lifelib's index |
| `mort_table.parquet` | `mort_table.xlsx` | unpivoted: one row per `age` (18..120) and `duration` (0..5), column `mort_rate` |
| `disc_rate_ann.parquet` | `disc_rate_ann.xlsx` | `year` (0..150), `zero_spot` |
| `premium_table.parquet` | `premium_table.xlsx` | the two-level index as `age_at_entry`, `policy_term`; column `premium_rate` |

sha256 of the source workbooks in lifelib 0.17.2:

```
model_point_table.xlsx  1b8f46672b5a86ee8c295ef7ef8c21344f1cbc336c45912f9881c38b6167fb01
mort_table.xlsx         073abd43d9c4da171f02d9ae7fe02899860b5e9d221714fd50183bb544dae8a3
disc_rate_ann.xlsx      b8cd87028137f5e2150b8a0c8b51ce7dbe25c0375b352e83a16930c7627373a0
premium_table.xlsx      6b76b3cc7290ded7c609873f67f87711d15edd3fbd2b590f99ba7394b4ca338a
```

## lifelib's results

Produced by `make_reference.py`, which runs BasicTerm_ME with modelx 0.33.0 (GPL-3.0; used only
to produce these files, never by Halley or the test suite) under numpy 2.5.4, pandas 3.0.6 and
Python 3.12.14. lifelib indexes a month by its start (`t = 0 .. max_proj_len - 1`, here 0..276);
the test maps that to Halley's months.

| File | Contents |
|---|---|
| `results_pv.parquet` | every model point: `pv_premiums`, `pv_claims`, `pv_expenses`, `pv_commissions`, `pv_net_cf`, `pv_pols_if`, `net_premium_pp`, `premium_pp`, `proj_len` |
| `results_monthly.parquet` | 56 sampled model points (every 200th plus the edge cases listed in `sample_ids`), months `0 .. proj_len - 1`: the flows, the counts, `pols_if_bef_decr`, `mort_rate_mth`, `lapse_rate` |
| `results_totals.parquet` | lifelib's `result_cf()` and `result_pols()` by month, with `disc_factors` and `inflation_factor` |

## Regenerating

```bash
uv sync --group lifelib
uv run python tests/data/lifelib/make_reference.py
```

The script reads the workbooks from the installed `lifelib` package, so the lifelib version it
runs is the one it records. The test `test_live_lifelib_matches_committed_results` runs when
lifelib and modelx are importable and checks these files against a fresh lifelib run.
