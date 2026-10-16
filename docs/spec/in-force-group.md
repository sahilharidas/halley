# Halley in-force group: IFRS 17 subsequent measurement, worked example (v2, M10)

Version 0.1, draft, 2026-10-16. Not yet reviewed.

## 1. Purpose and status

**What this document is.** The complete definition of one reporting period's IFRS 17 measurement of an in-force group of annuities under the general measurement model: the inputs, the four projection runs the period needs, the change in locked-in fulfilment cash flows, the period close (accretion, loss-component allocation, remeasurement, release) and the monthly roll-forward after it. It is the worked example that roadmap milestone M10 calls for, in the D22 pattern:

1. this written spec;
2. a spreadsheet built from the spec by hand, with no other source;
3. a reviewer (a finance actuary) checks the spreadsheet against the spec, line by line;
4. the Halley engine (milestone M11, the aggregation stage with in-force groups), running the runs and the close defined here, reconciles to the spreadsheet.

The example is one group measured at a valuation date `V` one reporting period (12 months) after its recognition date, with a non-zero change in fulfilment cash flows at locked-in rates (a mortality and expense basis change, and actual deaths that differ from expected) over a period in which actual CPI differed from the inflation expected at recognition. It comes in two variants that share every input except the single premiums and the period's mortality experience: a profitable group (variant P) and an onerous group (variant O). Between them, and with the table of section 14.5, every branch of the remeasurement rules is exercised.

**What this document follows.** The design is the section "In-force groups (after v1)" of the vault note *Halley — Annuity Spec*, with its aggregation-stage sections (initial recognition, locked-in rates, coverage units, accretion and release, loss component). The policy projection is that note's annuity model, restated here in full so that the spreadsheet builder needs no other document; the script of section 14 reproduces the note's own example values to every printed digit (section 14.10). Where that note left a formula open (the loss-component allocation over a period longer than a month, open question 8 there), this document fixes one and records the choice in section 16 and the alternative in section 17. Nothing here changes the annuity model.

**Status.** Version 0.1, dated 2026-10-16. Draft for review. Nothing in sections 2 to 16 is open for this example; the only items open are in section 17. The basis is synthetic and formula-defined: it is nobody's experience and it reproduces no proprietary table. References such as "para 44" or "B119(b)" are to IFRS 17 *Insurance Contracts* as amended in June 2020.

**Who reads it.**

- The spreadsheet builder, who transcribes sections 3 to 10 into cells and sections 11 to 13 into output and check sheets. The builder cannot ask questions; where the document is silent the builder stops and records a question for section 17 rather than choosing.
- The reviewer, who reads the whole document, objects to specific numbered items in section 16, and then checks the spreadsheet against sections 5 to 13 cell by cell, without reading any engine code.
- The engine author, who builds the cross-run reads of milestone M9 and the in-force aggregation stage of M11 from sections 3, 5 and 7 to 10, and the reconciliation test from sections 11 and 13.

**"Built from the spec" means** that every cell formula in the spreadsheet is a transcription of a formula in this document, in the evaluation order of sections 6.10 and 9.1, using the reference inputs of section 13 and nothing else: no engine output, no other table, no rounding, no judgement. Where this document gives a computation form (a ratio of two curve values rather than a product of monthly factors, a sum in ascending `t`, a left-to-right subtraction), the spreadsheet uses that form. The numbers in section 14 may be used to spot mistakes but are not the target; the formulas are the target. If a cell cannot be built because the document is ambiguous, that is a defect in this document and goes to section 17, not into the spreadsheet as a private choice.

**Notation used throughout.**

```
mi(d)             month index of a date d: 12 * year(d) + month(d) - 1; the day is ignored
V                 the valuation date 2026-12-31; m_V = mi(V) = 24323
V_rec             the recognition date 2025-12-31; m_rec = 24311
V_prev            the previous valuation date; here V_prev = V_rec, so the previous run is the recognition run
tau0              mi(V) - mi(V_rec) = 12, months from recognition to V
tau               mi(V) - mi(V_prev) = 12, the length of the reporting period in months
t                 integer months from the run's own valuation date: t = 0, 1, 2, ...
                  "time t" is an instant (the end of calendar month m_V + t); "month t" is the interval (t-1, t]
u                 CPI month index relative to the run's valuation month: I(u) is the CPI of calendar month m_V + u
T                 the run's horizon in months; T_i a policy's own run-off month
x^y               x raised to the power y, computed as a float64 power (Excel ^ or POWER, Python **)
a // b            floor(a / b) for integers;  a mod b  the remainder, in [0, b)
min(a, b), exp(x), ln(x), expm1(x), log1p(x)     as usual, float64; ln is the natural logarithm
where(c, a, b)    a if condition c holds, otherwise b; the unused branch is discarded (the engine may still
                  evaluate it, so a discarded branch may divide by zero without affecting the result)
x(t)              value of variable x in month t (month quantities) or at time t (states)
x^rec, x^L, x^cur a quantity of the recognition run, of a run on the locked-in financial inputs, of the
                  current-basis run (sections 5.5, 7)
```

All amounts are in one unnamed currency. Everything is float64; nothing is rounded at any step. Where a formula's result depends on the order of float64 operations, the order is the one written, left to right, with the parentheses shown. Variable names are in `snake_case` where they are columns and in the annuity spec's symbols (`CSM`, `LC`, `N`, `U`, `R`, `F`) where they are group quantities; both are the names to use in the spreadsheet and in the Halley aggregation stage.

**Units used in this document.** These are the annuity spec's units, which extend the v1 unit table (D11 is firm; the extensions are that note's open question 2).

| Label | Meaning |
|---|---|
| `Money` | currency amount for the whole in-force count of a model point, or of the group |
| `PerPolicy[Money]` | currency amount per life (instalment, expense per policy, per-life values) |
| `Count` | number of lives (real-valued) |
| `Prob[Annual]`, `Prob[Monthly]` | probability in [0, 1] per year of age, per month |
| `Rate[Monthly]` | a one-month effective rate |
| `Rate[Annual, Effective]`, `Rate[Annual, Continuous]` | an annual rate, with its compounding |
| `Factor` | dimensionless multiplier (discount factors, escalation factors, improvement factors, stress multipliers) |
| `Index` | a CPI value |
| `CoverageUnits` | coverage units (money-valued, but never added to `Money`) |
| `Int`, `Bool`, `Date` | integer, boolean, calendar date |

`Count × PerPolicy[Money] → Money` is applied exactly once in every in-force amount; `Count × Prob[Monthly] → Count`; `Prob[Monthly] × PerPolicy[Money] → PerPolicy[Money]`; `Factor × X → X`; `Index / Index → Factor`; `Money / Money → Factor`; `CoverageUnits / CoverageUnits → Factor`; `Rate[Monthly] × Money → Money`. Every compounding step is a named conversion (section 6.1, 6.4, 6.6).

**Reconciliation tolerance.** For every cell and scalar in section 11, engine and spreadsheet agree if

```
|engine - spreadsheet| <= 1e-9 * max(|engine|, |spreadsheet|, scale)
scale = P_g                               for Money and CoverageUnits quantities of the group
      = pols_if_init_rec * single_premium  for Money and CoverageUnits quantities of one policy
      = single_premium                     for PerPolicy[Money] quantities
      = pols_if_init_rec                   for Count quantities
      = 1                                  for Prob, Rate, Factor, Index and integer quantities
```

`P_g` is the group's total premium at recognition (section 7.3) and `pols_if_init_rec` the policy's count at recognition (section 13). Cells that this document says are exactly 0 (sections 6.11, 9.4 and 10, and the statements marked "exactly" in section 12) must be exactly 0 in both. Both compute in float64; the only expected differences are last-bit differences in `exp`, `log` and power functions and in summation order. A difference larger than the tolerance is a defect in one of the two, never a convention difference, because the conventions are fixed here. The script of section 14 checks its own invariants at `1e-12` with the annuity spec's per-invariant scales (section 12), and its two implementations agree to about `2e-13` relative (section 14.9).

**Location.** This document lives in the Halley repository at `docs/spec/in-force-group.md` (Apache 2.0) with its number script `docs/spec/in_force_group_numbers.py` beside it. The design notes it depends on stay in the author's vault.

## 2. The group and its contracts

**Contracts.** Single-life immediate annuities in payment, bought with one single premium received on the issue date, paid monthly in advance on the 1st of each calendar month from `annuity_start_date` while the annuitant is alive on that date. No guarantee period, no death benefit, no surrender value, no further premiums; the contract boundary is the annuitant's remaining lifetime (para 34). Two classes:

| Class | Escalation on each anniversary of `annuity_start_date` | In the example |
|---|---|---|
| `LevelAnnuity` | none | one model point of 100 lives at recognition |
| `CPILinkedAnnuity` | payment × `E(t)`, `E(t) = I(t - L) / I(t - L - 12)` with lag `L = 3` months, no cap, no floor; the payment due on an anniversary is the first at the new amount | one model point of 50 lives at recognition |

The annuity spec's third class, fixed escalation, is not in this example: its escalation is contractual, it has no estimate that can change, and it would add sheets without exercising anything in sections 8 to 10 (C4).

**Classification.** Insurance contracts under the general measurement model: survival is the insured event (B26(d)); the CPI link is not separated (para 11(a), B10); there is no investment component and no investment-return service, so insurance coverage is the only service; the premium allocation approach is not available (paras 53–54); the variable fee approach does not apply (B101). Every future payment sits in the liability for remaining coverage; the liability for incurred claims is zero in the model because nothing is due and unpaid at `V`.

**The group.** One group per variant, keyed on (`portfolio`, `cohort`, `profit_bucket`) = (`IA`, 2025, `OTHER`) for variant P and (`IA`, 2025, `ONEROUS`) for variant O. Every contract in it was issued on the recognition date `V_rec` = 2025-12-31 (para 25; the group's single issue date is why no B73 weighted-average rate is needed, C6). The group was recognised at `V_rec` on the basis then current, closed its first balance sheet at `V_rec` with the contractual service margin `CSM(0)` or the loss component `LC(0)` of initial recognition, and is now measured at `V` = 2026-12-31 for the period `(V_rec, V]`.

**What happens in the period, as data.** Lives die (the actual in-force at `V` is a model point input); payments are made to the survivors (their total is the group input `cu_act`); CPI runs at 3.2% a year against the 2.5% expected at recognition, and the December 2025 index is revised on publication; at `V` the insurer adopts a lighter base mortality table (version 2) and a higher expense level (version 2); risk-free rates are 50 basis points higher and expected inflation is 3.0% a year. None of these events is modelled: each enters as an input of section 3, 5 or 13.

**Excluded, deliberately.** Insurance revenue, insurance finance income or expenses (IFIE) and the OCI option as outputs (milestone M12; section 11.2 gives the analysis of change their formulas would consume). Scenario statistics: every run has one scenario (`J = 1`, check G6). Contracts joining the group after recognition (para 28, B73), acquisition cash flows, reinsurance, tax, caps and floors on the CPI link, guarantee periods, interim-period reporting (B137), the actual-versus-expected result of the period's own payments and expenses (B97(c), insurance service result), rounding of payments, undiscounted coverage units (the annuity spec's switch; one formula only here, C13).

## 3. Inputs

### 3.1 Run settings

| Setting | Value | Rule |
|---|---|---|
| `valuation_date` `V` | 2026-12-31 | the last day of a calendar month; `m_V = 24323` |
| `recognition_date` `V_rec` | 2025-12-31 | a month-end; `m_rec = 24311`; `tau0 = 12` |
| `prev_valuation_date` `V_prev` | 2025-12-31 | a month-end; `tau = 12`; here equal to `V_rec` |
| `scenario_set` | one scenario, id 0, per run | `J = 1` (G6) |
| `cpi_lag_m` `L` | 3 | build-time constant of the CPI class |
| `include_inv_exp` | True | investment expenses are fulfilment cash flows (B65(ka)(i)) |
| `discount_coverage_units` | True | the only form in this document (C13) |
| `T` | 489 for every run at `V`; 501 for the recognition run | `T = 12 * 120 + 1 - min a0` over the group's policies at the run's date (section 4) |
| Arithmetic | IEEE float64, no fast-math, no rounding | |

All settings and every derived value go into the run manifest (D21).

### 3.2 Model points at `V`

One row per model point in the class file (`mp_level`, `mp_cpi`). Field names are Halley model point fields; the file is Parquet with Arrow types; a row outside a range is a load error, never silently adjusted.

| Field | Arrow type | Unit | Range or values | Meaning |
|---|---|---|---|---|
| `policy_id` | int64 | | ≥ 1; unique across the classes of a run | Canonical order (D19): group sums run in ascending `policy_id` |
| `portfolio` | dictionary<string> | | `IA` | Group key |
| `cohort` | int16 | | 2025; must equal `year(issue_date)` | Group key (para 22) |
| `profit_bucket` | enum | | `OTHER` (variant P), `ONEROUS` (variant O) | Group key (para 16); must match the stored group input (G3 is not re-run) |
| `sex` | enum `Sex` | | `F`, `M` | Mortality key |
| `date_of_birth` | date32 | Date | `a0` in [600, 1320] months | |
| `issue_date` | date32 | Date | equal to `recognition_date` for every row | Initial recognition date (para 25) |
| `annuity_start_date` | date32 | Date | day of month 1; `issue_date < annuity_start_date <= V + 1 day` | First payment date; anniversaries run from it |
| `pols_if_init` | float64 | Count | (0, 1e6] | **Actual** lives in force at `V` (the survivors of the period) |
| `instalment_pm` | float64 | PerPolicy[Money] | (0, 1e6] | **Actual** monthly payment in force at `V`, after every actual escalation up to `V` |
| `single_premium` | float64 | PerPolicy[Money] | ≥ 0 | Carried unchanged from recognition; read only by the recognition run |

A model point of the previous run whose actual in-force at `V` is zero (every life died) is absent from the file at `V`: it contributes nothing to the runs at `V` and keeps its expected in-force in the old-basis run (section 5.5). A `policy_id` present at `V` and absent from the previous run's membership is a load error: contracts do not join a group after recognition (G4, para 28). The rows of section 13 are the reference input; their hash enters the manifest of every reconciliation run.

Derived at load from the dates (D10):

| Field | Type | Definition |
|---|---|---|
| `a0` | int32 | `m_V - mi(date_of_birth)`: age at `V` in completed months, the day of birth ignored (section 4) |
| `m_c` | int32 | `mi(annuity_start_date) - m_V`: the month of the first payment relative to `V`; here −11 for every row |
| `t_end` | int32 | `T_i = 12 * 120 + 1 - a0`: the policy's first month at age 120 |

### 3.3 Group inputs

One row per group in the `groups` file. These are the inputs the annuity spec's in-force design names; the stage reads them with the model points.

| Field | Type | Unit | Rule | Source in this example |
|---|---|---|---|---|
| `portfolio`, `cohort`, `profit_bucket` | as 3.2 | | the group key | `IA`, 2025, `OTHER` or `ONEROUS` |
| `recognition_date` | date32 | Date | a month-end; every member's `issue_date` | 2025-12-31 |
| `prev_valuation_date` | date32 | Date | a month-end, `<= V - 1 month` | 2025-12-31 |
| `csm_prev` | float64 | Money | ≥ 0 | the closing CSM at `V_prev`: here `CSM(0)` of the recognition run (section 7.3) |
| `lc_prev` | float64 | Money | ≥ 0; at most one of `csm_prev`, `lc_prev` is positive | the closing LC at `V_prev`: here `LC(0)` of the recognition run |
| `cu_act` | float64 | CoverageUnits | ≥ 0 | the payments actually made to survivors in `(V_prev, V]`, summed over the group (section 13) |
| `prev_run_id` | string | | names a run directory whose manifest the loader reads | the recognition run of the variant |
| locked-in basis | three hashes | | the discount file at recognition, the ILP table and the locked-in CPI file of section 5.2, fixed for the life of the group (B72(b)) | assigned by the loader to the files of 3.6 |

### 3.4 What the previous run supplies

Read from the run named by `prev_run_id`, through the hashes in its manifest (D21), never retyped:

| Quantity | Symbol here | Where it comes from |
|---|---|---|
| each policy's expected in-force at `V` | `n_exp_i = pols_if_i^rec(tau)` | the previous run's `pols_if` output at its month `tau`, rolled forward from `V_prev` on the previous tables |
| the previous run's fulfilment cash flows at `V_prev` and at `V` | `F_prev(0) = FCF_g^rec(0)`, `F_prev(tau) = FCF_g^rec(tau)` | its group outputs `BEL_g + RA_g` at months 0 and `tau` (section 7.1) |
| the previous group membership | the set of `policy_id` | its model point files |
| the previous non-financial tables | `mort_base` v1, `mort_cif`, `expense` v1, `inv_expense`, `ra_param` | its manifest hashes; the loader re-reads the files |
| the previous run's monthly loss component at `tau` | `LC^rec(tau)` | its group output; read only by the check K7 of section 12 |

Because `V_prev = V_rec` here, the previous run is the recognition run, and `csm_prev` and `lc_prev` are that run's `CSM(0)` and `LC(0)`. In a later period `prev_run_id` would name the previous in-force run and `csm_prev`, `lc_prev` its closing balances; nothing else in sections 5 to 10 changes.

### 3.5 Assumption tables

Each table is a versioned Parquet file whose hash enters the manifest. Every value is synthetic and formula-defined; no published or proprietary table is reproduced. Version 1 is the basis at recognition (the annuity spec's example basis); version 2 is the basis adopted at `V`. The table and the checker rules follow the annuity spec.

| Table | Keys | Value, unit | Version 1 (at recognition) | Version 2 (at `V`) |
|---|---|---|---|---|
| `mort_base` | `sex`; `age` Int in [20, 120] | `q`, Prob[Annual] | section 6.1, `mort_base_v1` | `0.96 × v1` for ages 20 to 119, 1 at 120 (section 6.1) |
| `mort_cif` | `age` Int in [20, 120]; `year` Int in [2025, 2150] | `cif`, Factor | section 6.1 | unchanged |
| `expense` | none (one row) | `maint_pm`, PerPolicy[Money] per month | 5.00 in December 2025 prices | 5.30 in December 2026 prices |
| `inv_expense` | none | `ie_rate`, Rate[Annual, Effective] | 0.0015 | unchanged |
| `ilp` | none | `ilp`, Rate[Annual, Continuous] | 0.0050 | unchanged, and part of the locked-in basis |
| `ra_param` | none | `alpha`, `z_alpha`, `sigma_l`, `sigma_e` | 0.75, 0.6744897501960817, 0.10, 0.10 | unchanged |

The `expense` table's metadata records its price base (the valuation month of the run that priced it). Load checks: `q(·, 120) = 1`; `cif(·, 2025) = 1`; `cif(120, y) = 1`; `q_base × cif <= 1` on the key grid; `sigma_e <= z_alpha`; every lookup key of every run inside the ranges (the last calendar year used is `(24323 + 489) // 12 = 2067`, inside the table).

### 3.6 Scenario files

| File | Keys | Value, unit | Content | Used by |
|---|---|---|---|---|
| `discount_rec` | `scenario` = 0; `t` in [0, 501] | `rf_df`, Factor | the risk-free curve at recognition, section 5.1 | the recognition run; the locked-in basis |
| `discount_cur` | `scenario` = 0; `t` in [0, 489] | `rf_df`, Factor | the risk-free curve at `V`, section 5.3 | the current-basis run |
| `cpi_rec` | `scenario` = 0; `u` in [−24, 501] | `cpi`, Index | the recognition run's CPI path, section 5.2 | the recognition run; the locked-in splice |
| `cpi_act` | `u` in [−26, 0] | `cpi`, Index | actual history as known at `V`, section 5.2 | the splice and the current path; the expense rebase |
| `cpi_locked` | `scenario` = 0; `u` in [−26, 489] | `cpi`, Index | the locked-in path of section 5.2, built outside the engine and hashed | the two locked-in runs |
| `cpi_cur` | `scenario` = 0; `u` in [−26, 489] | `cpi`, Index | the current path of section 5.3 | the current-basis run |

The engine never interpolates, extrapolates or splices a curve: each file holds every row its run reads, built by the formulas of section 5 before the run, and the manifest records the file hashes and the flag that the December 2026 index (`u = 0` at `V`) is an estimate (section 5.2).

## 4. Time and ages

**Month index.** `t` counts months from the run's own valuation date. Month `t` coincides with calendar month `m_V + t`; time `t - 1` is its start, the 1st of that month; time `t` is its end. The one-day gap between the end of the valuation date and the 1st of the next month is ignored. There are no day counts; a month is 1/12 of a year.

```
Y(t)  = (m_V + t) // 12                       calendar year of month t
a0    = m_V - mi(date_of_birth)               age at V in completed months, counted as if every birth were at the end of its birth month
x(t)  = min(120, (a0 + t - 1) // 12)          age last birthday at the start of month t, the lookup age
T_i   = 12 * 120 + 1 - a0                     the policy's first month at age 120; pols_if is exactly 0 from T_i on
T     = max_i T_i over the group               the run's horizon
```

`a0` ignores the day of birth, so it can understate age by up to a month; every run does the same. `x(t)` steps up in the month after the birthday month. The clamp at 120 keeps every table key in range; beyond 120 the in-force is zero, so the value looked up does not matter.

**Anniversaries (CPI class).** `m_c = mi(annuity_start_date) - m_V`. Month `t`'s payment falls on an anniversary when `k = t - m_c` satisfies `k > 0` and `k mod 12 = 0`. For the recognition run `m_c = 1` and the anniversaries are `t = 13, 25, ...`; for the runs at `V`, `m_c = -11` and the anniversaries are `t = 1, 13, 25, ...`. So the payment of month 1 at `V` (1 January 2027) is the first escalated payment, and no escalation happened inside the period: `instalment_pm` at `V` equals the instalment at recognition for both classes.

**The previous run's index.** The recognition run counts months from `V_rec`; month `tau0 + t` of that run is month `t` of a run at `V`, and its CPI month `tau0 + u` is `u` at `V`. Section 12.3 checks the relation.

**Domains.** States (`pols_if`, `ann_pp`, `df`, `bel`, `v_be`, `v_st`, `ra`) are defined for `t = 0 .. T`; every other projection variable for `t = 1 .. T`; the group states `N`, `CSM`, `LC`, `F` for `t = 0 .. T`; `Acc`, `U`, `R`, `LCA` for `t = 1 .. T`. The CPI path is read at `u = t - 1` (expenses), `t - 3` and `t - 15` (escalation, both branches of the `where` evaluated), so a run needs `I(u)` for `u >= -14`; `-26` in the files leaves twelve months of margin for the splice checks of section 5.2. The spreadsheet leaves month quantities blank in row 0; the engine stores 0 for them there; every check excludes them at `t = 0`.

**Edge cases resolved.**

| Case | Resolution |
|---|---|
| A policy whose `T_i < T` (the CPI point: `T_i = 468` at `V`, `T = 489`) | `q_ann(T_i) = 1` by the clamp, `pols_if(T_i) = 0` exactly, every in-force amount and `bel`, `ra`, `cu` are exactly 0 after (section 6.11); per-life values continue and are excluded from every comparison beyond `T_i` |
| `tau0` not equal to `tau` | the previous run is an in-force run; sections 5.1 and 9.2 use `tau0` and `tau` as written and nothing else changes |
| Anniversary in month 1 at `V` | `E(1) = I(-2) / I(-14)`, both rows actual history (section 5.2) |
| The December 2026 CPI not yet published at `V` | its row is the estimate of section 5.2, flagged in the file and the manifest, used as actual |
| The December 2025 CPI revised since recognition | the published value replaces the estimate in `I_act`; the recognition path `I_rec` is not restated (section 5.2) |
| Valuation date not a month-end, `annuity_start_date` not on the 1st, `issue_date` not the recognition date | load errors |

## 5. Financial inputs of the runs at `V`

### 5.1 The locked-in curve, re-based to `V`

The composed curve at recognition is the locked-in basis of the group (B72(b)): one curve, no scenario dimension.

```
z_rec(s)   = 0.040 - 0.015 * exp(-s / 10)                 continuously compounded risk-free zero rate at term s years
rf_df_rec(t) = exp(-z_rec(t / 12) * (t / 12))              t = 0 .. 501;  rf_df_rec(0) = 1 exactly
P_rec(t)   = rf_df_rec(t) * exp(-ilp * t / 12)             composed curve at recognition, ilp = 0.0050
P_L(t)     = P_rec(tau0 + t) / P_rec(tau0)                 locked-in curve re-based to V, t = 0 .. 489;  P_L(0) = 1 exactly
d_L(t)     = P_L(t) / P_L(t - 1)                           one-month locked-in discount factor of month t
j_L(t)     = P_L(t - 1) / P_L(t) - 1                       one-month locked-in accretion rate of month t, Rate[Monthly]
```

Compute `t / 12` first, then `z_rec` of it, then the product, then `exp`. `P_L(t)` is a ratio of two `P_rec` cells, not a product of monthly factors; the locked-in runs read `df(t) = P_L(t)` directly (the ILP is inside `P_rec`). Checkpoints: `P_rec(12) = 0.9690612702196784`, `P_rec(13) = 0.9664087352366604`, `P_rec(501) = 0.15425915191945347`; `P_L(1) = 0.9972627788721588`, `P_L(12) = 0.9665612216490105`, `P_L(489) = 0.15918410595904237`; `j_L(1) = 0.002744734071933319`, `j_L(12) = 0.002927838156386864`. The locked-in discount file covers `t` in `[0, tau0 + T] = [0, 501]`, which is the recognition run's own horizon because the membership is unchanged.

### 5.2 CPI: history, the recognition path and the locked-in splice

Three paths, each `Index`-valued and each a function of a month index relative to its own valuation month.

```
I_rec(u) = 100 * 1.04^(u / 12)      u <= 0        the recognition run's path: history at 4% a year, December 2025 = 100 (an estimate then)
I_rec(u) = 100 * 1.025^(u / 12)     u >= 0        expected inflation 2.5% a year at recognition

I_act(u) = 100 * 1.04^((u + 12) / 12)        u <= -13    history before December 2025, unchanged (= I_rec(u + 12))
I_act(u) = 100.1 * 1.032^((u + 12) / 12)     -12 <= u <= 0   December 2025 published at 100.1 (revised from 100.0), then 3.2% a year to V

I_L(u)   = I_act(u)                                   u <= 0    the locked-in path: actual history to V
I_L(u)   = I_act(0) * I_rec(tau0 + u) / I_rec(tau0)   u > 0     then the inflation expected at recognition, from the actual level at V
```

The `u` of `I_act`, `I_L` and `I_cur` is relative to `V`; the `u` of `I_rec` is relative to `V_rec`. `I_L(u)` for `u > 0` is computed as the ratio of two `I_rec` cells times `I_act(0)`, as written. The December 2026 value `I_act(0)` is an estimate at `V` (published only in January 2027), flagged in `cpi_act` and `cpi_locked` and used as actual (C9). Checkpoints: `I_act(-14) = 99.34845327805992` (= `I_rec(-2)`), `I_act(-12) = 100.1`, `I_act(-2) = 102.76230218844762`, `I_act(0) = 103.3032`; `I_L(1) = 103.51598767815085`, `I_L(10) = 105.45090936406004`, `I_L(12) = 105.88578000000001`, `I_L(13) = 106.10388737010459`. The escalation factor of month 1 at `V`, `E(1) = I_act(-2) / I_act(-14) = 1.0343623760385368`, is the same on every path at `V` because both rows are history; at recognition the same anniversary was expected at `E^rec(13) = I_rec(10) / I_rec(-2) = 1.0274848910276622`. The next, `E_L(13) = I_L(10) / I_L(-2) = 1.026163360671718`, is where the locked-in and current paths part.

Reason for the splice: actual CPI to `V` is a fact for every run, and expected inflation after `V` stays at its locked-in level so that changes in CPI expectations stay out of the CSM (B97(a), B128(a)). The file is built outside the engine by these formulas and supplied with its hash (D4).

### 5.3 The current curve and the current CPI path at `V`

```
z_cur(s)   = 0.045 - 0.015 * exp(-s / 10)
rf_df_cur(t) = exp(-z_cur(t / 12) * (t / 12))             t = 0 .. 489
P_cur(t)   = rf_df_cur(t) * exp(-ilp * t / 12)            composed current curve; the ILP table is unchanged
I_cur(u)   = I_act(u)                                     u <= 0
I_cur(u)   = I_act(0) * 1.03^(u / 12)                     u > 0     current expected inflation 3.0% a year
```

Checkpoints: `P_cur(1) = 0.9970772395543179`, `P_cur(12) = 0.964228056970892`, `P_cur(489) = 0.13171516763686483`; `I_cur(1) = 103.55797355954225`, `I_cur(12) = 106.402296`, `E_cur(13) = 1.030333063965748`.

### 5.4 The expense rebase for the old-basis run

The previous expense table is priced at `V_prev`. Before the old-basis run the loader rebases it to `V` prices with actual CPI:

```
maint_pm_old = maint_pm_prev * I_act(0) / I_act(-tau) = 5.00 * 103.3032 / 100.1 = 5.160000000000001
```

The loader then checks that every run at `V` carries price base `V`: the version 2 table is priced at `V` already (5.30) and is not rebased. Reason: an expense level in `V` prices that merely follows actual inflation is not a change in estimate; the change for future service is the difference between 5.30 and 5.16, measured by the two locked-in runs (C8).

### 5.5 The four runs

Every run is an ordinary projection of section 6 of every policy of the group, with one scenario. They differ only in the inputs below.

| Run | Valuation date | Curve `df(t)` | CPI path `I(u)` | Non-financial tables | `pols_if_init` | `instalment_pm` | Purpose |
|---|---|---|---|---|---|---|---|
| `REC` | `V_rec` | `P_rec(t)` | `I_rec(u)` | version 1; `maint_pm = 5.00` (Dec 2025 prices) | lives at recognition | at recognition | initial recognition; supplies 3.4 |
| `OLD` | `V` | `P_L(t)` | `I_L(u)` | version 1; `maint_pm_old = 5.16` | `n_exp_i` (3.4) | actual at `V` | `FCF_g^L_old(0)` |
| `NEW` | `V` | `P_L(t)` | `I_L(u)` | version 2; `maint_pm = 5.30` | actual at `V` (3.2) | actual at `V` | `FCF_g^L_new(0)` |
| `CUR` | `V` | `P_cur(t)` | `I_cur(u)` | version 2; `maint_pm = 5.30` | actual at `V` | actual at `V` | the balance sheet `FCF_g(t)`, `CU_g(t)`, `N(0)`, months 1..T |

`OLD` and `NEW` share every financial input and the actual instalment; they differ only in the non-financial tables and the in-force, which is what makes their difference the para 44(c) change (section 8). `CUR` differs from `NEW` only in the curve and the CPI path, which is what makes their difference a financial effect (section 11.2). Each run's horizon is `T = 489` at `V` (`T_rec = 501`); a policy's run-off `T_i` is 12 months shorter at `V` than at recognition.

## 6. Projection of one policy

This section is the annuity spec's model, restated. Each run applies it to each policy with the run's inputs. Dimensions are `policy × scenario × time` with one scenario.

### 6.1 Mortality

Base table version 1, the Standard Ultimate Survival Model (Makeham), generated by formula; the female row is the male row four years younger:

```
A = 0.00022,  B = 2.7e-6,  c = 1.124
q_mk(x)        = 1 - exp(-A - B * c^x * (c - 1) / ln(c))         evaluate: c^x, then B * c^x, then * (c - 1), then / ln(c), then -A - that, exp, 1 -
mort_base_v1(M, x) = q_mk(x)             x = 20 .. 119
mort_base_v1(F, x) = q_mk(x - 4)         x = 20 .. 119
mort_base_v1(s, 120) = 1
mort_base_v2(s, x) = 0.96 * mort_base_v1(s, x)     x = 20 .. 119;     mort_base_v2(s, 120) = 1
```

Check values against the published SOA table (6 decimal places): `q_mk(65) = 0.005915`, `q_mk(100) = 0.289584`. Full precision for the ages this example reaches first: `mort_base_v1(M, 79) = 0.029132141` and `(F, 81) = 0.023174621`, both 8 significant figures; the table of section 14.1 lists more. Version 2 is a multiplicative change of the base table, the kind an experience study produces; the improvement scale is unchanged (C7).

Improvement factors, the convergence form of SOA MP-2014 formula B.1 with Halley's own parameters, one table for both versions and both sexes:

```
h(y)   = min(1, (y - 2026) / 20)
w(h)   = 1 - 3 * h^2 + 2 * h^3
g(x)   = 1                  x <= 85
       = (110 - x) / 25     85 < x < 110
       = 0                  x >= 110
r(x, y) = g(x) * (0.010 + 0.010 * w(h(y)))            x = 20 .. 120, y = 2026 .. 2150
cif(x, 2025) = 1;   cif(x, y) = cif(x, y - 1) * (1 - r(x, y))   in ascending y
```

Checkpoints: `r(65, 2026) = 0.02`, `r(79, 2027) = 0.0199275`, `cif(79, 2027) = 0.96047105`, `cif(65, 2036) = 0.8183053332960806`, `cif(110, y) = 1`, `cif(120, y) = 1`.

Rates in month `t`, best estimate and stressed (section 6.8), and the monthly conversion:

```
q_ann(t)    = where(x(t) >= 120, 1, mort_base[sex, x(t)] * cif[x(t), Y(t)])        Prob[Annual]
q_mth(t)    = to_monthly_cf(q_ann(t))                                                Prob[Monthly]
q_ann_st(t) = where(x(t) >= 120, 1, q_ann(t) * mort_mult)
q_mth_st(t) = to_monthly_cf(q_ann_st(t))
to_monthly_cf(q) = where(q >= 1, 1, -expm1(log1p(-where(q >= 1, 0, q)) / 12))      constant force within the year of age
```

A spreadsheet may compute `to_monthly_cf` as `1 - (1 - q)^(1/12)`, which also gives exactly 1 at `q = 1`; the guarded form exists because `log1p(-1)` is an error in Python and a warning in NumPy and the engine evaluates both branches of `where`. The limiting age closes the table: with `q = 1` at 120, every life still in force dies in its first month at age 120.

### 6.2 In-force: the only decrement

Death is the only decrement; there are no lapses, no surrenders and no maturities. Order of events in month `t`: at time `t - 1` the payment and the maintenance expense are paid for each of the `pols_if(t - 1)` lives; during the month the deaths occur; at time `t` the investment expense is paid on the closing BEL and `pols_if(t)` is the count after the deaths.

```
pols_if(0) = pols_if_init
deaths(t)  = pols_if(t - 1) * q_mth(t)                   Count × Prob[Monthly] → Count
pols_if(t) = pols_if(t - 1) - deaths(t)                  not a cumulative product of survival probabilities
```

### 6.3 Instalment and escalation

```
anniv(t)      = (t - m_c > 0) and ((t - m_c) mod 12 = 0)                Bool; the CPI class only
esc_factor(t) = 1                                                       level class
              = where(anniv(t), I(t - L) / I(t - L - 12), 1)            CPI class, L = 3, Index / Index → Factor; both branches evaluated
ann_pp(0)     = instalment_pm
ann_pp(t)     = ann_pp(t - 1) * esc_factor(t)                           PerPolicy[Money], paid at the start of month t
```

`ann_pp` is computed by its recursion, one month at a time; multiplying by 1 is exact in float64, so the level class's instalment is constant bit for bit. The telescoped form of invariant I9 is a check, not an evaluation order.

### 6.4 Expenses

```
exp_pp(t)  = maint_pm * (I(t - 1) / I(0))                   PerPolicy[Money], paid at the start of month t
ie_m       = (1 + ie_rate)^(1/12) - 1                       Rate[Monthly]; = 0.00012491414476678564
inv_exp(t) = ie_m * bel(t)                                  Money, paid at the end of month t on the closing BEL (section 6.7)
```

The expense paid at time 0 is at the run's price base; each later one carries the inflation of the whole months before it. `maint_pm` is the run's expense table value (section 5.5). The investment-expense rate is not stressed.

### 6.5 Cash flows and their timing

| Cash flow | Variable | Formula | Timing | Paid at time | Discounted to time 0 by |
|---|---|---|---|---|---|
| Annuity payment | `ann_out(t)` | `pols_if(t - 1) * ann_pp(t)` | start | `t - 1` | `df(t - 1)` |
| Maintenance expense | `exp_out(t)` | `pols_if(t - 1) * exp_pp(t)` | start | `t - 1` | `df(t - 1)` |
| Investment expense | `inv_exp(t)` | `ie_m * bel(t)` | end | `t` | `df(t)` |
| Single premium | | received on the issue date | before `t = 0` | | not a future cash flow (para 38(b)) |

Every in-force amount is `Count × PerPolicy[Money] → Money`, applied once. No cash flow is mid-month. `s_pp(t) = ann_pp(t) + exp_pp(t)` and `s_pp_st(t) = ann_pp(t) + exp_pp(t) * exp_mult` are the per-life start-of-month amounts on the two bases.

### 6.6 Discounting

```
df(t) = the run's composed curve at t: P_rec(t), P_L(t) or P_cur(t) (section 5.5);  df(0) = 1
v(t)  = df(t) / df(t - 1)                            one-month discount factor of month t, Factor
```

`v(t)` is a ratio of two `df` cells. Nominal cash flows, including the CPI-linked ones, are discounted at the nominal curve (B74). Rates carry no adjustment for non-financial risk (B90).

### 6.7 The best-estimate liability

`bel(t)` is the value at time `t`, on the run's curve, of all fulfilment cash flows of months `t + 1 .. T`: annuity payments, maintenance expenses and investment expenses (paras 32(a), 33, B65(b), (f), (h), (ka)(i)). It includes the payment at the start of month `t + 1`. Backward recursion, for `t = T - 1, ..., 0`, in the grouping written:

```
bel(T) = 0
bel(t) = ann_out(t+1) + exp_out(t+1) + inv_exp(t+1) * v(t+1) + v(t+1) * bel(t+1)       with inv_exp(t+1) = ie_m * bel(t+1)
```

Evaluate `ann_out + exp_out` first, add `inv_exp * v`, then add `v * bel`. Algebraically `bel(t) = ann_out(t+1) + exp_out(t+1) + v(t+1) * (1 + ie_m) * bel(t+1)`; that form explains the recursion and is the basis of implementation B (section 14.9), not the evaluation order.

### 6.8 Per-life values and the risk adjustment

Confidence-level technique on one latent factor at `alpha = 0.75`: annual mortality is scaled by `mort_mult` and maintenance expenses by `exp_mult`, both mean-one lognormal multipliers evaluated at `z_alpha`:

```
mort_mult = exp(-sigma_l * z_alpha - sigma_l * sigma_l / 2)      = 0.9301132044575212
exp_mult  = exp( sigma_e * z_alpha - sigma_e * sigma_e / 2)      = 1.0644401444946738
```

Per-life values at `t` for a life in force at `t`, best estimate and stressed, backward for `t = T - 1, ..., 0`:

```
v_be(T) = 0;   v_be(t) = s_pp(t+1)    + (1 - q_mth(t+1))    * (v(t+1) * (v_be(t+1) + ie_m * v_be(t+1)))
v_st(T) = 0;   v_st(t) = s_pp_st(t+1) + (1 - q_mth_st(t+1)) * (v(t+1) * (v_st(t+1) + ie_m * v_st(t+1)))
ra(t)   = pols_if(t) * (v_st(t) - v_be(t))                      Money; never a stressed total less the BEL
```

`v_st` conditions on stressed survival; `ra(t)` multiplies the difference by the best-estimate count, so `ra(t) >= 0` at every `t` and `ra(0)` equals the direct present value of the stressed cash flows less the BEL. The stress is applied to annual mortality after improvement, with age 120 forced back to `q = 1`. The `v_be` recursion is the per-life form of `bel` (invariant I6).

### 6.9 Coverage units

```
cu(t) = ann_out(t) = pols_if(t - 1) * ann_pp(t)              CoverageUnits, t = 1 .. T
```

The quantity of benefits of month `t` is the amount the policyholder can validly claim in it times the expected number of lives entitled to it (B119; Method 1 of the IFRS Interpretations Committee's 2022 agenda decision). Expenses are excluded. For the CPI class the units carry the run's expected indexation. Coverage units of the in-force group, for `N(0)` in section 9.5 and for months `1 .. T`, come from the `CUR` run: Method 1 counts the payments expected to be claimable, a current estimate (C14).

### 6.10 Evaluation order

A spreadsheet evaluates one (run, policy) sheet as follows; the engine does the same per policy.

1. Per-policy constants: `a0`, `m_c`, `T_i`; the run's constants: `m_V`, `T`, `ie_m`, `mort_mult`, `exp_mult`, `maint_pm`, `I(0)`.
2. Row `t = 0`: `pols_if(0) = pols_if_init`, `ann_pp(0) = instalment_pm`, `df(0) = 1`; every month quantity blank.
3. Rows `t = 1 .. T`, each from its own `t` and row `t - 1` only, columns left to right: `x`, `Y`, `q_ann`, `q_mth`, `deaths`, `pols_if`, `anniv`, `esc_factor`, `ann_pp`, `exp_pp`, `ann_out`, `exp_out`, `df`, `v`, `s_pp`, `q_ann_st`, `q_mth_st`, `s_pp_st`, `cu`.
4. The backward columns from row `T` upwards: `bel`, `inv_exp` (`inv_exp(t) = ie_m * bel(t)`, so row `t`'s `inv_exp` is available to row `t - 1`'s `bel`), `v_be`, `v_st`. Then `ra` for every row `0 .. T`.
5. The per-policy scalars of section 7.2 as sums down the columns in ascending `t`, then the check cells of section 12.1.

A chain `a + b + c` means `(a + b) + c`. IEEE float64 `+` and `×` are commutative, so only the grouping matters.

### 6.11 After a policy's run-off

For `t >= T_i`: `pols_if(t) = 0` exactly (because `q_mth(T_i) = 1` gives `deaths(T_i) = pols_if(T_i - 1)` and `pols_if(T_i - 1) - pols_if(T_i - 1) = 0`), and `bel(t)`, `ra(t)` are exactly 0; for `t > T_i`: `deaths`, `ann_out`, `exp_out`, `inv_exp`, `cu` are exactly 0 (each is a zero count times an amount, or `ie_m` times a zero BEL). `ann_pp`, `exp_pp`, `s_pp`, `s_pp_st`, `v_be` and `v_st` continue by their formulas and are not meaningful beyond `T_i`; they are excluded from every comparison there. In-force-weighted values therefore do not depend on `T`.

## 7. Group totals and initial recognition

### 7.1 Group totals

For each run, sums over the group's policies, sequential and left to right in ascending `policy_id` across both classes (D19):

```
BEL_g(t) = sum_i bel_i(t)       RA_g(t) = sum_i ra_i(t)       FCF_g(t) = BEL_g(t) + RA_g(t)       t = 0 .. T
CU_g(t)  = sum_i cu_i(t)                                                                           t = 1 .. T
F(t)     = FCF_g(t)             the present value of future cash outflows plus the RA; there are no future inflows
```

`FCF_g` is formed first as `BEL_g + RA_g`; `CSM(0)` and `LC(0)` use `P_g - FCF_g(0)` and `FCF_g(0) - P_g`, never `(P_g - BEL_g) - RA_g`.

### 7.2 Per-policy scalars

For each run and policy, sums in ascending `t` over `t = 1 .. T`:

```
pva_i = sum_t df(t - 1) * ann_out(t)
pve_i = sum_t [df(t - 1) * exp_out(t) + df(t) * inv_exp(t)]
```

with `bel_i(0)`, `ra_i(0)`, `ann_pp_i(1)`, `v_be_i(0)` and `v_st_i(0)` read from the columns. Check: `pva_i + pve_i = bel_i(0)` within tolerance (invariant I5).

### 7.3 Initial recognition and what the recognition run supplies

At `V_rec` the group is new: every contract's `issue_date` is `V_rec`, the premiums are received and coverage starts.

```
P_g    = sum_i pols_if_init_rec_i * single_premium_i           ascending policy_id
CSM(0) = max(0, P_g - FCF_g^rec(0))                             written where(gap > 0, gap, 0) with gap = P_g - FCF_g^rec(0)
LC(0)  = max(0, FCF_g^rec(0) - P_g)
```

If `LC(0) > 0` the loss goes to profit or loss, the CSM is zero and the loss component is set up (paras 47, 49). Load check G3 at recognition: contract `i` is onerous iff `pols_if_init_rec_i * single_premium_i - (bel_i^rec(0) + ra_i^rec(0)) < 0`; an onerous contract outside an `ONEROUS` group, or a non-onerous one inside, is an error. G3 is not re-run at `V`.

The recognition run then supplies, as the previous run of section 3.4: `csm_prev = CSM(0)`, `lc_prev = LC(0)`, `n_exp_i = pols_if_i^rec(12)`, `F_prev(0) = FCF_g^rec(0)` and `F_prev(12) = FCF_g^rec(12)`. Its own monthly roll-forward over months `1 .. 12` (section 10 applied at `V_rec` with `P_L = P_rec`) is a diagnostic: its `LC(12)` equals `LC_alloc` of section 9.3 (K7), and its `CSM(12)` is compared with the period-level close in section 14.8.

## 8. The change in locked-in fulfilment cash flows

```
delta_FCF_L = FCF_g^L_new(0) - FCF_g^L_old(0)
            = [BEL_g(0) + RA_g(0)]_NEW - [BEL_g(0) + RA_g(0)]_OLD
```

Both terms are group totals of section 7.1 from the `NEW` and `OLD` runs of section 5.5: the same locked-in curve `P_L`, the same locked-in CPI path `I_L`, the same actual instalments; different non-financial tables and in-force. `delta_FCF_L` is the increase in the fulfilment cash flows relating to future service, measured at locked-in rates (paras 44(c), B96(b), B96(d), B72(c)). Its two parts are not separated in the measurement; the script of section 14 splits it with an extra run (old tables, actual in-force) as a diagnostic only (section 14.4).

**Which changes are in it.** Mortality (base table and improvement), expense level, investment-expense rate and RA parameters: yes, at locked-in rates. The effect of actual deaths on future payments (the closing in-force differing from expected): yes, by inference from B96(b) (the annuity spec's open question 6). The period's own payments and expenses differing from expected: no, insurance service result (B97(c)). Discount curve, ILP, CPI path and CPI expectations: no, IFIE (B97(a), B128(a)); they are held fixed between `OLD` and `NEW`, which is why the two runs exist.

## 9. The period close

### 9.1 Order

The close of the period `(V_prev, V]` follows para 44(a)–(e) and B119(b): accrete, allocate the loss component, apply the change for future service, release, close. Every step below is one formula; the engine and the spreadsheet compute them in this order.

### 9.2 Step 1: accretion

```
CSM_acc = csm_prev * P_rec(tau0 - tau) / P_rec(tau0)         interest on the opening CSM at the locked-in forward rates of the period (para 44(b), B72(b))
acc_per = CSM_acc - csm_prev                                  the roll-forward line
```

Here `P_rec(0) / P_rec(12) = 1 / 0.9690612702196784 = 1.0319264949813836`, the locked-in one-year forward factor. `lc_prev` does not accrete: the loss component is part of the fulfilment cash flows and moves with them (step 2).

### 9.3 Step 2: the systematic allocation of the loss component

```
LC_alloc = where(F_prev(0) > 0, lc_prev * (F_prev(tau) / F_prev(0)), 0)        para 50(a), 51
lca_per  = lc_prev - LC_alloc                                                     the roll-forward line
```

`F_prev(0)` and `F_prev(tau)` are the previous run's fulfilment cash flows at `V_prev` and at `V`, on the previous run's own basis and curve (section 3.4). Reason: the para 51 items of the period (expected claims and expenses, the RA for risk expired, the finance effect on the loss component) are the expected amounts, and the previous run is the only expected-basis source of them; the ratio form is the annuity spec's monthly rule `LC(t) = LC(t - 1) * F(t) / F(t - 1)` telescoped over the period, so the period step gives exactly the loss component the previous run itself projected for `V` (K7), and the result does not depend on the step length (C11; the alternative is section 17, question 1). With `lc_prev = 0` the allocation is 0.

### 9.4 Step 3: the remeasurement rules

Applied to `(CSM_acc, LC_alloc, delta_FCF_L)`, giving `(CSM_adj, LC_adj)`. `delta_FCF_L` is the increase in locked-in fulfilment cash flows from changes for future service (section 8).

```
If LC_alloc > 0 and delta_FCF_L >= 0:    LC_adj  = LC_alloc + delta_FCF_L                                 para 48(a)
                                         CSM_adj = CSM_acc
If LC_alloc > 0 and delta_FCF_L < 0:     rev     = min(LC_alloc, -delta_FCF_L)                           paras 50(b), 44(c)(ii)
                                         LC_adj  = LC_alloc - rev
                                         CSM_adj = CSM_acc + (-delta_FCF_L - rev)
If LC_alloc = 0:                         CSM_adj = CSM_acc - delta_FCF_L                                 para 44(c)(i)
                                         if CSM_adj < 0:  LC_adj = -CSM_adj;  CSM_adj = 0                 para 48(a)
                                         else:            LC_adj = 0
csm_fs = CSM_adj - CSM_acc               the CSM roll-forward line "changes for future service"
lc_fs  = LC_adj - LC_alloc               the LC roll-forward line: a loss (> 0) or a reversal of losses (< 0)
```

Each increase in `LC` is a loss in profit or loss; each reduction under para 50(b) is a reversal of losses. In every branch `(CSM_adj - LC_adj) = (CSM_acc - LC_alloc) - delta_FCF_L` (K3), and at most one of `CSM_adj`, `LC_adj` is positive (K4). The para 48 onerous test is therefore run on the CSM after accretion and before the period's release. Section 14.5 shows every branch on the reference figures.

### 9.5 Step 4: release

```
N(0)  = sum over m = 1 .. T of CU_g(m) * P_L(m - 1) / P_L(0)         computed backward as in section 10, from the CUR run's units
R_per = where(cu_act + N(0) > 0, CSM_adj * (cu_act / (cu_act + N(0))), CSM_adj)      para 44(e), B119
```

This is Method 1 at the period level: the period's units are the payments actually claimable in it, undiscounted (they are the group input `cu_act`), and the future units are the expected payments of the `CUR` run valued at `V` on the locked-in curve. If no coverage remains (`cu_act + N(0) = 0`) the whole adjusted CSM is released. The expected units of the period, `sum over m = 1 .. 12 of CU_g^rec(m)`, are not used in the close; they appear in section 14.8 as a diagnostic of the period's experience.

### 9.6 Step 5: closing balances

```
CSM(0) = CSM_adj - R_per
LC(0)  = LC_adj
LRC(0) = FCF_g^cur(0) + CSM(0)             the liability for remaining coverage at V; the LC is a part of FCF_g^cur(0), not an addition to it
```

`CSM(0)` and `LC(0)` are the opening balances of months `1 .. T` (section 10). The roll-forward tables of section 11.1 are built from the lines named in 9.2 to 9.5.

### 9.7 Edge cases resolved

| Case | Resolution |
|---|---|
| `csm_prev = lc_prev = 0` (a group exactly at break-even at `V_prev`) | `CSM_acc = LC_alloc = 0`; the third rule applies: `CSM_adj = -delta_FCF_L` if that is positive, else `LC_adj = delta_FCF_L` |
| Both `csm_prev > 0` and `lc_prev > 0` | load error on the `groups` row |
| `F_prev(0) = 0` | `LC_alloc = 0` (no future service was expected at `V_prev`, so there is nothing to allocate to) |
| `delta_FCF_L = 0` with `LC_alloc > 0` | the first rule: `LC_adj = LC_alloc`, `CSM_adj = CSM_acc` |
| `cu_act + N(0) = 0` | `R_per = CSM_adj`; months `1 .. T` then start from `CSM(0) = 0` |
| `cu_act = 0` but `N(0) > 0` (no payment was due in the period) | `R_per = 0` |
| A model point with no survivors at `V` | absent from the files at `V`; present in `OLD` with `n_exp_i` (section 3.2); its expected units are in `F_prev` and `N` only through the runs that contain it |

## 10. Months after the close

Months `t = 1 .. T` at `V` run as for a new group, on the `CUR` run's `CU_g(t)` and `F(t) = FCF_g^cur(t)`, with the locked-in curve `P_L` for accretion and for the discounting of units. Written in evaluation order:

```
N(T) = 0;    N(t) = CU_g(t+1) + d_L(t+1) * N(t+1)          backward, t = T-1 .. 0;  d_L(t+1) = P_L(t+1) / P_L(t)
Acc(t)     = j_L(t) * CSM(t-1)                             j_L(t) = P_L(t-1) / P_L(t) - 1;  para 44(b)
CSM_pre(t) = CSM(t-1) + Acc(t)
U(t)       = CU_g(t) + N(t)                                the current month's units are undiscounted
R(t)       = where(U(t) > 0, CSM_pre(t) * (CU_g(t) / U(t)), CSM_pre(t))        para 44(e), B119(b)
CSM(t)     = CSM_pre(t) - R(t)
LC(t)      = where(F(t-1) > 0, LC(t-1) * (F(t) / F(t-1)), 0)                     para 50(a), Illustrative Example 8
LCA(t)     = LC(t-1) - LC(t)
```

There is no change for future service inside the months: the next remeasurement is at the next period close. `N(0)` of this recursion is the `N(0)` of section 9.5. The ratio `CU_g(t) / U(t)` is formed first, then multiplied into `CSM_pre(t)`; inside a `where` the division is by a guarded value because both branches are evaluated. With `CSM(0) = 0` every `Acc`, `R` and `CSM` is exactly 0; with `LC(0) = 0` every `LC` and `LCA` is exactly 0.

## 11. Outputs

### 11.1 The period's roll-forward tables

CSM, for the period `(V_prev, V]`:

| Line | Amount | Paragraph |
|---|---|---|
| Opening CSM | `csm_prev` | |
| Interest accretion | `acc_per = CSM_acc - csm_prev` | 44(b), B72(b) |
| Changes for future service | `csm_fs = CSM_adj - CSM_acc` | 44(c), B96 |
| Currency differences | 0 (one currency) | 44(d) |
| Release | `-R_per` | 44(e), B119 |
| Closing CSM | `CSM(0)` | |

Loss component:

| Line | Amount | Paragraph |
|---|---|---|
| Opening LC | `lc_prev` | |
| Systematic allocation | `-lca_per = -(lc_prev - LC_alloc)` | 50(a), 51, 52 |
| Losses on remeasurement | `lc_fs` where positive | 48(a) |
| Reversals of losses | `lc_fs` where negative | 50(b) |
| Closing LC | `LC(0)` | |

Per group and month after the close, `t = 0 .. T`: `CU_g(t)`, `N(t)`, `BEL_g(t)`, `RA_g(t)`, `F(t)`, `j_L(t)`, `Acc(t)`, `CSM_pre(t)`, `U(t)`, `R(t)`, `CSM(t)`, `LC(t)`, `LCA(t)` (section 10). Per group: `P_g`, `csm_prev`, `lc_prev`, `cu_act`, `N(0)`, `delta_FCF_L`, the five lines of each table above, `LRC(0)`, the locked-in basis hashes and `prev_run_id`.

Per run, policy and month: the columns of section 6.10 (engine outputs: `pols_if`, `deaths`, `ann_pp`, `ann_out`, `exp_out`, `inv_exp`, `bel`, `ra`, `cu`; the rest available through trace). Per run and policy at `t = 0`: `bel_i(0)`, `ra_i(0)`, `pva_i`, `pve_i`, `v_be_i(0)`, `v_st_i(0)`, `ann_pp_i(1)`. Per run: `BEL_g(0)`, `RA_g(0)`, `FCF_g(0)`.

### 11.2 Analysis of change of the fulfilment cash flows

Five values, four differences, all from quantities already defined; they are the pieces a later IFIE and revenue stage (M12) consumes and are reported, not interpreted, here:

```
A = FCF_g^rec(0)        fulfilment cash flows at recognition
B = F_prev(tau)         expected at V on the recognition basis (the previous run's own roll: unwind, and the period's cash flows paid out)
C = FCF_g^L_old(0)      at V on the locked-in financial inputs, old tables, expected in-force
D = FCF_g^L_new(0)      at V on the locked-in financial inputs, new tables, actual in-force
E = FCF_g^cur(0)        at V on the current basis: the balance sheet
B - A   the previous run's expected movement over the period
C - B   CPI experience over the period on the future cash flows (actual against expected index levels), on the locked-in curve: IFIE (B128(a))
D - C   delta_FCF_L: the CSM or the loss component (section 8)
E - D   the change in financial assumptions at V (curve and CPI expectations): IFIE (B97(a), para 87)
```

`C - B` is pure CPI experience because `B` and `C` share the curve (the recognition run values month 12 on the forward factors that `P_L` re-bases), the tables, the instalments and the expected in-force; section 12.3 checks that it vanishes when actual CPI equals expected.

### 11.3 Spreadsheet layout (fixed)

One workbook per variant. Sheets, in this order: `inputs` (the run settings of 3.1, the tables of 3.5 as rows, the three curves and four CPI paths of section 5 as columns keyed on `t` or `u`, the model points of 13 and the group inputs); one sheet per (run, policy), named `REC-11`, `OLD-11`, ... with the blocks of section 6.10 and rows `t = 0 .. T` of that run; a `group` sheet with the totals of 7.1 per run by `t`, the recognition block of 7.3, the inputs of 3.4 read from the `REC` sheets by reference, the close block of section 9 one step per row, the months block of section 10 by `t`, and the two roll-forward tables of 11.1; a `checks` sheet with one boolean cell per invariant of section 12 per run or group as that section says, and one `all_ok` cell. Helper columns beside a boolean (a residual `lhs - rhs`) are layout, not definition, and are not reconciled. The engine's reconciliation reads the `group` sheet and the (run, policy) sheets; it reads a policy's values beyond `T_i` as excluded where section 6.11 says so.

### 11.4 What milestone M12 needs

The analysis of change of 11.2 and the lines of 11.1 keep the pieces of insurance revenue (B121, B124: expected payments and expenses, RA release, CSM release, less the loss-component allocation) and of IFIE (para 87: unwind at current rates, accretion at locked-in rates, `C - B`, `E - D`) separate. Nothing more is defined here.

## 12. Invariants

A tolerance check passes when `|lhs - rhs| <= eps * S`, with `eps = 1e-12` within one implementation (the script and the engine) and `S` the scale named; between the spreadsheet and the engine the rule of section 1 applies, with the same scales where it names one. "Exactly" means bit for bit. Each check is a boolean cell on the `checks` sheet (statements marked **(sheet)**) or an engine check (the rest). I1 to I13 are the annuity spec's invariants with their numbers; K1 to K7 and Z1, Z2 are this document's.

### 12.1 Every run, every policy

- **I1 (sheet)** `pols_if(t) >= 0` and `pols_if(t) <= pols_if(t - 1)`, exactly.
- **I2 (sheet)** `q_ann`, `q_mth`, `q_ann_st`, `q_mth_st` in [0, 1]; `deaths(t) <= pols_if(t - 1)`, exactly.
- **I3 (sheet)** `bel(T) = v_be(T) = v_st(T) = 0`; `q_ann(T_i) = 1` and `pols_if(T_i) = 0`; `pols_if`, `bel`, `ra` exactly 0 for `t > T_i`.
- **I4 (sheet)** Roll-forward, `t = 1 .. T`: `bel(t) = (bel(t - 1) - ann_out(t) - exp_out(t)) / v(t) - inv_exp(t)`; `S = (|bel(t - 1)| + |ann_out(t)| + |exp_out(t)|) / v(t) + |inv_exp(t)|`.
- **I5 (sheet)** Direct PV: `bel(0) = pva + pve`, the sums of 7.2 in ascending `t`; `S = max(|lhs|, |rhs|)`.
- **I6** `bel(t) = pols_if(t) * v_be(t)`, `t = 0 .. T`; `S = max(|lhs|, |rhs|, bel(0))`.
- **I7 (sheet)** `ra(t) >= 0` exactly.
- **I8** Level class: `ann_pp(t) = instalment_pm` exactly.
- **I9 (sheet)** CPI class, telescoping: `ann_pp(t) = instalment_pm * I(n_K - L) / I(n_0 - L)`, with `n_K` the latest anniversary month in `1 .. t` (or `n_0` if none) and `n_0 = m_c` if `m_c = 1`, else the latest month `m_c + 12k <= 0`; at `V`, `n_0 = -11`; `S = |rhs|`.
- **I10** `cu(t) = ann_out(t) >= 0` exactly.

### 12.2 The close and the months after it

- **K1 (sheet)** `CSM(0) = csm_prev + acc_per + csm_fs - R_per`; `S = P_g`.
- **K2 (sheet)** `LC(0) = lc_prev - lca_per + lc_fs`; `S = P_g`.
- **K3 (sheet)** Conservation through the remeasurement rules: `CSM_adj - LC_adj = (CSM_acc - LC_alloc) - delta_FCF_L`; `S = P_g`.
- **K4 (sheet)** `CSM_acc >= 0`, `LC_alloc >= 0`, `CSM_adj >= 0`, `LC_adj >= 0`, `CSM(0) >= 0`, `LC(0) >= 0` exactly, and at most one of `CSM_adj`, `LC_adj` and at most one of `CSM(0)`, `LC(0)` is positive.
- **K5 (sheet)** Equal allocation at the period level: `R_per / cu_act = CSM(0) / N(0)` where `cu_act > 0` and `CSM(0) > 0`; `S = max(|lhs|, |rhs|)`.
- **K6 (sheet)** `CSM(t) >= 0` and `LC(t) >= 0` exactly, `t = 1 .. T`.
- **K7** `LC_alloc` equals the previous run's monthly `LC(tau)` (section 7.3); `S = P_g`.
- **I11 (sheet)** `CSM(t) = CSM(t - 1) + Acc(t) - R(t)`, `S = CSM(t - 1) + |Acc(t)| + R(t)`; `CSM(t) = 0` exactly after the last month with `CU_g > 0`.
- **I12 (sheet)** Equal allocation: `R(t) / CU_g(t) = CSM(t) / N(t)` where `CU_g(t) > 0`, `N(t) > 0` and `CSM(t) > 0`; `S = max(|lhs|, |rhs|)`.
- **I13 (sheet)** `LC(t) / F(t) = LC(0) / F(0)` where `F(t) > 0` and `LC(0) > 0`; `LC(t) = 0` exactly where `F(t) = 0`.

### 12.3 The shifted world

Two engine checks (and script checks) on constructed inputs, which test the re-basing of section 5 and the reads of section 3.4 rather than the reference figures:

- **Z1** With the actual CPI path equal to the recognition path (`I_act(u) = I_rec(tau + u)` for `u <= 0`), the previous tables, the expected in-force `n_exp` and the instalments at recognition, a run at `V` on `P_L` and the spliced `I_L` equals the recognition run shifted by `tau` months: `pols_if_i(t)`, `bel_i(t)`, `ra_i(t)` for `t = 0 .. T` and `cu_i(t)`, `ann_pp_i(t)` for `t = 1 .. T` equal the recognition run's values at `tau + t` within `1e-12 * bel_i^rec(tau)`; `FCF_g(0)` equals `F_prev(tau)` and `N(0)` equals `N^rec(tau)` within `1e-12` relative. So `C - B` of section 11.2 is zero when there is no CPI experience.
- **Z2** A `NEW` run with the `OLD` run's inputs gives `delta_FCF_L = 0` exactly: the two runs are the same sheet with different inputs and nothing else.

### 12.4 Load-time checks

G1 one group per aggregation run; G2 `cohort = year(issue_date)`; G3 the para 47 test at recognition only; G4 a `policy_id`'s group key unchanged between runs, and no `policy_id` at `V` that the previous run did not have; G6 `J = 1` in every run the stage reads (a run with two scenarios is rejected with the check named); the `groups` rules of 3.3; the price-base check of 5.4; the locked-in discount file covering `[0, tau0 + T]`; the CPI files covering `u >= -14`.

## 13. Reference inputs

Recognition date **2025-12-31**, valuation date **2026-12-31**. Two variants, four model points, two group rows. Dates are ISO `YYYY-MM-DD`. The `single_premium` values were chosen by the script of section 14 as `round(k * (bel_i^rec(0) + ra_i^rec(0)) / pols_if_init_rec, 2)` with `k = 1.04` for variant P and `k = 0.98` for variant O; that rule is how the inputs were picked and is not part of the model. The inputs are these numbers, exactly.

**Model points.** The same two lives, under each variant's own `policy_id`:

| `policy_id` | Variant | Class | `sex` | `date_of_birth` | `issue_date` | `annuity_start_date` | `pols_if_init` at recognition | `instalment_pm` | `single_premium` | `pols_if_init` at `V` (actual) |
|---|---|---|---|---|---|---|---|---|---|---|
| 11 | P | level | M | 1947-08-15 | 2025-12-31 | 2026-01-01 | 100 | 1000.00 | 126250.60 | 98 |
| 12 | P | CPI | F | 1945-11-30 | 2025-12-31 | 2026-01-01 | 50 | 800.00 | 132916.18 | 50 |
| 21 | O | level | M | 1947-08-15 | 2025-12-31 | 2026-01-01 | 100 | 1000.00 | 118966.91 | 95 |
| 22 | O | CPI | F | 1945-11-30 | 2025-12-31 | 2026-01-01 | 50 | 800.00 | 125247.94 | 47 |

`instalment_pm` at `V` equals the value at recognition for every row (section 4: the first anniversary falls at month 1 of the runs at `V`). `portfolio = IA`, `cohort = 2025` for every row; `profit_bucket = OTHER` for 11 and 12, `ONEROUS` for 21 and 22.

Derived values (hand checks, not results):

| `policy_id` | `a0` at recognition | age at recognition | `T_i` at recognition | `a0` at `V` | age at `V` | `T_i` at `V` | `x(1)` at `V` | `m_c` at recognition | `m_c` at `V` |
|---|---|---|---|---|---|---|---|---|---|
| 11, 21 | 940 | 78y 4m | 501 | 952 | 79y 4m | 489 | 79 | 1 | −11 |
| 12, 22 | 961 | 80y 1m | 480 | 973 | 81y 1m | 468 | 81 | 1 | −11 |

So `T_rec = 501` and `T = 489` at `V` for both groups. The recognition run's expected in-force at `V` is `n_exp = 97.35061873366202` for the level point and `48.97700540091314` for the CPI point (2.649 and 1.023 expected deaths from 100 and 50); the actual survivors are the last column of the model point table.

**Group inputs** (`groups` file), one row per variant:

| Variant | `profit_bucket` | `recognition_date` | `prev_valuation_date` | `csm_prev` | `lc_prev` | `cu_act` | `prev_run_id` |
|---|---|---|---|---|---|---|---|
| P | `OTHER` | 2025-12-31 | 2025-12-31 | `CSM(0)` of the P recognition run: 741187.7620383836 | 0 | 1667000.00 | the P recognition run |
| O | `ONEROUS` | 2025-12-31 | 2025-12-31 | 0 | `LC(0)` of the O recognition run: 370593.23796161637 | 1640400.00 | the O recognition run |

`csm_prev` and `lc_prev` are read from the recognition run's outputs at full precision, not retyped; the values above are what section 14.2 produces. `cu_act` is a datum: the payments actually made in the period, `1000.00 × 1187 + 800.00 × 600` life-months for variant P (the level point paid 100 lives for three months, 99 for five and 98 for four; the CPI point paid 50 lives every month) and `1000.00 × 1170 + 800.00 × 588` for variant O (100, 99, 98, 97, 96 and 95 lives for two months each; 50, 49 and 48 for four months each, the third death in December). The life-month stories are how the two numbers arose; the inputs are the two totals.

**Locked-in basis by hash:** `discount_rec`, `ilp` (0.0050) and `cpi_locked` of section 3.6, both variants the same files.

## 14. Worked example

The numbers in this section were produced by a Python script (plain float64 scalars, standard library only) implementing the formulas above exactly as written; the script also checks every invariant of section 12 for both variants, carries the independent implementation B of section 14.9, and reproduces the annuity spec's own example values (section 14.10). It is committed beside this document as `docs/spec/in_force_group_numbers.py`; running it with Python 3.12 or later regenerates every printed number and rechecks every invariant. The numbers are **illustrative**, not the reference implementation and not the engine: the spreadsheet is built from the formulas, and the engine reconciles to the spreadsheet. Values are shown to 10 significant figures (the annuity spec's convention) and, where a builder would locate a slip with them, in full float64 precision.

### 14.1 Constants and checkpoints

```
m_rec = 24311,  m_V = 24323,  tau0 = tau = 12,  mi(annuity_start_date) = 24312;  m_c = 1 at recognition, -11 at V
ie_m      = 0.00012491414476678564         1 + ie_m = 1.0001249141447668
z_alpha   = 0.6744897501960817             mort_mult = 0.9301132044575212         exp_mult = 1.0644401444946738
(c - 1) / ln(c) = 1.060792372894571        a check figure only; not held in a cell
P_rec(0) / P_rec(12) = 1.0319264949813836  the accretion factor of the period
maint_pm_old = 5.160000000000001           the rebase of section 5.4;  I_act(0) / I_act(-12) = 1.032 exactly
```

Mortality base tables (8 significant figures):

| `x` | `v1(M, x)` | `v1(F, x)` | `v2(M, x)` | `v2(F, x)` |
|---|---|---|---|---|
| 65 | 0.005914652 | 0.0037916077 | 0.0056780659 | 0.0036399434 |
| 70 | 0.010413327 | 0.0066185277 | 0.0099967939 | 0.0063537866 |
| 75 | 0.018433156 | 0.011670038 | 0.01769583 | 0.011203237 |
| 79 | 0.029132141 | 0.018433156 | 0.027966855 | 0.01769583 |
| 80 | 0.032658484 | 0.020668344 | 0.031352145 | 0.01984161 |
| 81 | 0.036606808 | 0.023174621 | 0.035142536 | 0.022247636 |
| 85 | 0.057664543 | 0.036606808 | 0.055357961 | 0.035142536 |
| 90 | 0.10091734 | 0.06455369 | 0.09688065 | 0.061971543 |
| 100 | 0.28958395 | 0.19288739 | 0.27800059 | 0.1851719 |
| 110 | 0.66711422 | 0.49803567 | 0.64042966 | 0.47811424 |
| 119 | 0.95711833 | 0.86098696 | 0.9188336 | 0.82654748 |
| 120 | 1 | 1 | 1 | 1 |

Curves and CPI (full precision):

| `t` or `u` | `P_rec(t)` | `P_cur(t)` | `P_L(t)` | `j_L(t)` | `I_act(u)`, `u <= 0` | `I_L(u)` | `I_cur(u)` |
|---|---|---|---|---|---|---|---|
| −14 | | | | | 99.34845327805992 | | |
| −12 | | | | | 100.1 | | |
| −2 | | | | | 102.76230218844762 | | |
| 0 | 1 | 1 | 1 | | 103.3032 | 103.3032 | 103.3032 |
| 1 | 0.9974927749679984 | 0.9970772395543179 | 0.9972627788721588 | 0.002744734071933319 | | 103.51598767815085 | 103.55797355954225 |
| 10 | | | | | | 105.45090936406004 | 105.87939767399733 |
| 12 | 0.9690612702196784 | 0.964228056970892 | 0.9665612216490105 | 0.002927838156386864 | | 105.88578000000001 | 106.402296 |
| 13 | 0.9664087352366604 | 0.9611881730411305 | 0.9637247535995906 | 0.002943234610115919 | | 106.10388737010459 | 106.6647127663285 |
| 24 | 0.9366570451962741 | 0.9273371518765582 | | | | 108.5329245 | 109.59436488 |
| 120 | 0.6738026036931537 | 0.6409408629381212 | 0.6645475588826909 | 0.003797230654428718 | | | |
| 489 | 0.16148169138569274 | 0.13171516763686483 | 0.15918410595904237 | 0.003818466563236367 | | 282.5603953080984 | 344.53288556544396 |
| 501 | 0.15425915191945347 | | | | | | |

The recognition path at the same calendar months: `I_rec(-2) = 99.34845327805992`, `I_rec(0) = 100`, `I_rec(10) = 102.07903469017418`, `I_rec(12) = 102.49999999999999`, `I_rec(24) = 105.06249999999999`. Escalation factors: `E(1) = 1.0343623760385368` on every path at `V` against `E^rec(13) = 1.0274848910276622` expected at recognition; `E_L(13) = 1.026163360671718`; `E_cur(13) = 1.030333063965748`. Improvement: `r(79, 2027) = r(81, 2027) = 0.0199275`, `cif(79, 2027) = 0.96047105`.

### 14.2 The recognition run (both variants share it except for `P_g`)

Per policy at `V_rec`, full precision:

| Policy (P / O) | `bel(0)` | `ra(0)` | `pva` | `pve` | per-life `(bel + ra) / n` | `pols_if(12)` expected |
|---|---|---|---|---|---|---|
| 11 / 21 (level, 100 lives) | 11816430.431717163 | 323049.8871179734 | 11624246.148216018 | 192184.28350114627 | 121394.80318835136 | 97.35061873366202 |
| 12 / 22 (CPI, 50 lives) | 6207483.320785807 | 182717.59834067416 | 6091687.957581365 | 115795.36320444106 | 127804.01838252964 | 48.97700540091314 |

Group, full precision:

| Quantity | Variant P | Variant O |
|---|---|---|
| `P_g` | 19270869.0 | 18159088.0 |
| `FCF_g^rec(0)` = `F_prev(0)` | 18529681.237961616 | 18529681.237961616 |
| `CSM(0)` = `csm_prev` | 741187.7620383836 | 0 |
| `LC(0)` = `lc_prev` | 0 | 370593.23796161637 |
| `F_prev(12)` | 17362216.290460568 | 17362216.290460568 |
| `F_prev(12) / F_prev(0)` | 0.9369948714978824 | 0.9369948714978824 |
| expected units of the period, `sum CU_g^rec(1 .. 12)` | 1661301.8187426073 | 1661301.8187426073 |
| diagnostic: monthly roll `CSM(12)`, `LC(12)`, `N^rec(12)` | 694305.3268248928, 0, 16591150.9813373 | 0, 347243.9633818289, 16591150.9813373 |

G3 at recognition: `126250.60 × 100 - 12139480.318835136 > 0` and `132916.18 × 50 - 6390200.919126482 > 0` for variant P; both differences negative for variant O's premiums. (The variant-P CSM is 3.85% of `P_g`; the variant-O loss is 2.04%.)

### 14.3 The runs at `V`

Per policy, full precision. `ann_pp(1) = 827.4899008308294 = 800 × E(1)` for the CPI point in every run at `V`.

| Run | Policy | `pols_if(0)` | `bel(0)` | `ra(0)` | `v_be(0)` | `v_st(0)` |
|---|---|---|---|---|---|---|
| OLD | 11 / 21 | 97.35061873366202 | 10964227.859034946 | 310058.5024982256 | 112626.17538191048 | 115811.14232440628 |
| OLD | 12 / 22 | 48.97700540091314 | 5955296.163348271 | 180262.50697587224 | 121593.71759460912 | 125274.271468826 |
| NEW | 11 | 98 | 11212612.357670635 | 312164.7497846746 | 114414.41181296569 | 117599.76640260522 |
| NEW | 12 | 50 | 6182874.449239052 | 184752.11047663252 | 123657.48898478116 | 127352.53119431381 |
| CUR | 11 | 98 | 10947558.446912333 | 296584.9987248728 | 111709.78007053399 | 114736.1576085429 |
| CUR | 12 | 50 | 6223351.849604842 | 186471.45971806458 | 124467.03699209694 | 128196.46618645823 |
| NEW | 21 | 95 | 10869369.122231744 | 302608.68601575604 | 114414.41181296569 | 117599.76640260522 |
| NEW | 22 | 47 | 5811901.982284707 | 173666.98384803458 | 123657.48898478116 | 127352.53119431381 |
| CUR | 21 | 95 | 10612429.10670072 | 287505.8661108461 | 111709.78007053399 | 114736.1576085429 |
| CUR | 22 | 47 | 5849950.738628555 | 175283.1721349807 | 124467.03699209694 | 128196.46618645823 |

The per-life values depend on the run's tables and financial inputs only, so `NEW` and `CUR` show the same `v_be(0)` under either variant and `bel(0)` is the per-life value times the count (I6): `NEW` 11 is `98 × 114414.41181296569`. `CUR` `pva` and `pve`: policy 11 10768430.185844645 and 179128.2610676719; policy 12 6108043.391720855 and 115308.4578839959; policy 21 10438784.363829017 and 173644.74287172238; policy 22 5741560.7882175995 and 108389.95041095614.

Group totals, full precision:

| Run | Variant P: `BEL_g(0)`, `RA_g(0)`, `FCF_g(0)` | Variant O: `BEL_g(0)`, `RA_g(0)`, `FCF_g(0)` |
|---|---|---|
| OLD | 16919524.022383217, 490321.00947409787, 17409845.031857315 | the same |
| NEW | 17395486.806909688, 496916.8602613071, 17892403.667170994 | 16681271.10451645, 476275.6698637906, 17157546.77438024 |
| CUR | 17170910.296517175, 483056.4584429374, 17653966.754960112 | 16462379.845329275, 462789.03824582684, 16925168.8835751 |

### 14.4 The change in locked-in fulfilment cash flows

```
Variant P:  delta_FCF_L = 17892403.667170994 - 17409845.031857315 =  482558.63531367853
Variant O:  delta_FCF_L = 17157546.77438024  - 17409845.031857315 = -252298.25747707486
```

Diagnostic split by an extra run (old tables, actual in-force; not an output): variant P, in-force experience +203360.48937581852 (two deaths against 2.649 expected, none against 1.023) and basis change +279198.14593786 (lighter mortality, higher expenses); variant O, experience −519895.7520038672 (five and three deaths) and basis change +267597.4945267923. The basis change differs between the variants only because it is measured on different in-force counts.

### 14.5 The period close, and every branch of the remeasurement rules

Variant P (the third rule, CSM stays positive), full precision:

```
csm_prev = 741187.7620383836      lc_prev = 0        cu_act = 1667000        N(0) = 17326847.31821964
1  CSM_acc  = 741187.7620383836 * 1.0319264949813836 = 764851.2894033649       acc_per = 23663.527364981244
2  LC_alloc = 0                                                                 lca_per = 0
3  CSM_adj  = 764851.2894033649 - 482558.63531367853 = 282292.65408968634      csm_fs = -482558.63531367853;  LC_adj = 0
4  R_per    = 282292.65408968634 * (1667000 / (1667000 + 17326847.31821964)) = 24775.48895089341     share 0.08776526272278437
5  CSM(0)   = 257517.16513879295       LC(0) = 0       LRC(0) = 17653966.754960112 + 257517.16513879295 = 17911483.920098905
```

Variant O (the second rule, a partial reversal), full precision:

```
csm_prev = 0      lc_prev = 370593.23796161637      cu_act = 1640400      N(0) = 16611371.824558262
1  CSM_acc  = 0                                                                              acc_per = 0
2  LC_alloc = 370593.23796161637 * (17362216.290460568 / 18529681.237961616) = 347243.96338182886     lca_per = 23349.27457978751
3  rev      = min(347243.96338182886, 252298.25747707486) = 252298.25747707486
   LC_adj   = 94945.705904754         CSM_adj = 0 + (252298.25747707486 - 252298.25747707486) = 0      lc_fs = -252298.25747707486
4  R_per    = 0                       share 0.0898762057606263
5  CSM(0)   = 0       LC(0) = 94945.705904754       LRC(0) = 16925168.8835751
```

K7: `LC_alloc` equals the recognition run's monthly `LC(12) = 347243.9633818289` (relative difference 1.7e-16).

Every branch of section 9.4, on the reference figures (`CSM_acc` of variant P, `LC_alloc` of variant O, `|delta_FCF_L|` of variant P, and multiples of them), 10 significant figures; the last column is the K3 residual:

| Case | Rule | `CSM_acc` | `LC_alloc` | `delta_FCF_L` | `CSM_adj` | `LC_adj` | residual |
|---|---|---|---|---|---|---|---|
| a | LC = 0, decrease in CSM, stays positive (variant P) | 764,851.2894 | 0 | 482,558.6353 | 282,292.6541 | 0 | 0 |
| b | LC = 0, change exceeds the CSM: CSM to 0, loss component set up | 764,851.2894 | 0 | 956,064.1118 | 0 | 191,212.8224 | 0 |
| c | LC = 0, favourable change: CSM increases | 764,851.2894 | 0 | −482,558.6353 | 1,247,409.925 | 0 | 0 |
| d | LC > 0, unfavourable change: loss added to the LC | 0 | 347,243.9634 | 482,558.6353 | 0 | 829,802.5987 | 0 |
| e | LC > 0, partial reversal (variant O's branch) | 0 | 347,243.9634 | −173,621.9817 | 0 | 173,621.9817 | 0 |
| f | LC > 0, full reversal, remainder to the CSM | 0 | 347,243.9634 | −520,865.9451 | 173,621.9817 | 0 | 0 |

### 14.6 Months 1 to 12 after the close

Variant P, 10 significant figures:

| `t` | `CU_g(t)` | `N(t)` | `j_L(t)` | `Acc(t)` | `R(t)` | `CSM(t)` | `F(t)` |
|---|---|---|---|---|---|---|---|
| 0 | | 17,326,847.32 | | | | 257,517.1651 | 17,653,966.75 |
| 1 | 139,374.4950 | 17,234,647.87 | 0.002744734072 | 706.8161373 | 2,071.474081 | 256,152.5072 | 17,560,295.75 |
| 2 | 139,077.9728 | 17,142,795.80 | 0.002762464704 | 707.6122599 | 2,067.112929 | 254,793.0065 | 17,467,032.55 |
| 3 | 138,782.0879 | 17,051,284.38 | 0.002779971042 | 708.3171798 | 2,062.761492 | 253,438.5622 | 17,374,169.31 |
| 4 | 138,486.8387 | 16,960,106.96 | 0.002797255581 | 708.9324327 | 2,058.419746 | 252,089.0749 | 17,281,698.32 |
| 5 | 138,192.2241 | 16,869,257.00 | 0.002814320792 | 709.4595249 | 2,054.087668 | 250,744.4468 | 17,189,612.01 |
| 6 | 137,898.2425 | 16,778,728.06 | 0.002831169117 | 709.8999338 | 2,049.765234 | 249,404.5815 | 17,097,902.89 |
| 7 | 137,604.8927 | 16,688,513.81 | 0.002847802974 | 710.2551088 | 2,045.452422 | 248,069.3841 | 17,006,563.64 |
| 8 | 137,312.1732 | 16,598,608.00 | 0.002864224758 | 710.5264717 | 2,041.149206 | 246,738.7614 | 16,915,587.01 |
| 9 | 137,020.0827 | 16,509,004.48 | 0.002880436836 | 710.7154173 | 2,036.855566 | 245,412.6213 | 16,824,753.02 |
| 10 | 136,701.8437 | 16,419,724.05 | 0.002896441553 | 710.8233138 | 2,032.173421 | 244,091.2712 | 16,734,295.27 |
| 11 | 136,384.3599 | 16,330,760.71 | 0.002912241228 | 710.8526632 | 2,027.502692 | 242,774.6211 | 16,644,206.81 |
| 12 | 136,067.6293 | 16,242,108.52 | 0.002927838156 | 710.8047991 | 2,022.843348 | 241,462.5826 | 16,554,384.24 |

`LC(t) = LCA(t) = 0` throughout. Full precision: `R(1) = 2071.474080741102`, `CSM(1) = 256152.50719531596`, `CSM(12) = 241462.58257576753`. `CU_g(1) = 98 × 1000 + 50 × 827.4899008308294`; the larger drop at `t = 10` follows the level point's lookup age stepping from 79 to 80 in month 9 (`x(t)` steps in the month after the birthday month, and the life was born in August). The last month with `CU_g > 0` is 489, and `CSM(489) = 0` exactly.

Variant O, 10 significant figures (`Acc`, `R`, `CSM` are exactly 0 throughout):

| `t` | `CU_g(t)` | `N(t)` | `LC(t)` | `LCA(t)` | `F(t)` |
|---|---|---|---|---|---|
| 0 | | 16,611,371.82 | 94,945.70590 | | 16,925,168.88 |
| 1 | 133,892.0253 | 16,522,706.10 | 94,440.37748 | 505.3284212 | 16,835,088.25 |
| 2 | 133,606.7670 | 16,434,373.64 | 93,937.24369 | 503.1337930 | 16,745,398.84 |
| 3 | 133,322.1224 | 16,346,367.97 | 93,436.26232 | 500.9813720 | 16,656,093.13 |
| 4 | 133,038.0903 | 16,258,682.71 | 92,937.39183 | 498.8704837 | 16,567,163.70 |
| 5 | 132,754.6693 | 16,171,311.57 | 92,440.59137 | 496.8004669 | 16,478,603.28 |
| 6 | 132,471.8582 | 16,084,248.38 | 91,945.82069 | 494.7706733 | 16,390,404.69 |
| 7 | 132,189.6555 | 15,997,487.05 | 91,453.04023 | 492.7804671 | 16,302,560.88 |
| 8 | 131,908.0599 | 15,911,021.57 | 90,962.21100 | 490.8292249 | 16,215,064.90 |
| 9 | 131,627.0703 | 15,824,846.05 | 90,472.13710 | 490.0739058 | 16,127,703.57 |
| 10 | 131,320.7287 | 15,738,980.70 | 89,984.08922 | 488.0478754 | 16,040,703.39 |
| 11 | 131,015.1152 | 15,653,419.75 | 89,498.02987 | 486.0593485 | 15,954,057.70 |
| 12 | 130,710.2280 | 15,568,157.50 | 89,013.41298 | 484.6168885 | 15,867,669.14 |

Full precision: `LC(1) = 94440.37748355305`, `LC(12) = 89013.41298440409`; `LC(t) / F(t)` is `94945.705904754 / 16925168.8835751` at every `t` (I13).

### 14.7 Analysis of change

Full precision, section 11.2:

| | Variant P | Variant O |
|---|---|---|
| A `FCF_g^rec(0)` | 18529681.237961616 | 18529681.237961616 |
| B `F_prev(12)` | 17362216.290460568 | 17362216.290460568 |
| C `FCF_g^L_old(0)` | 17409845.031857315 | 17409845.031857315 |
| D `FCF_g^L_new(0)` | 17892403.667170994 | 17157546.77438024 |
| E `FCF_g^cur(0)` | 17653966.754960112 | 16925168.8835751 |
| B − A, the previous run's expected movement | −1167464.9475010484 | −1167464.9475010484 |
| C − B, CPI experience (IFIE) | 47628.74139674753 | 47628.74139674753 |
| D − C, `delta_FCF_L` (CSM or LC) | 482558.63531367853 | −252298.25747707486 |
| E − D, financial assumptions at `V` (IFIE) | −238436.9122108817 | −232377.89080514014 |

`C - B` is positive because actual inflation (3.2%, with the revised December 2025 base) exceeded the 2.5% expected, raising every future expense and every future CPI-linked payment; `E - D` is negative because 50 basis points of higher rates outweigh the higher expected inflation (3.0% against 2.5%).

### 14.8 Named simplifications and diagnostics (from the same script)

- **One release per period against twelve monthly steps (C12, section 17 question 2).** In the shifted world of Z1 (no experience, no basis change, no financial change) the close of section 9 with `delta_FCF_L = 0` and the expected units `1661301.8187426073` as `cu_act` gives `CSM(0) = 695236.0518201796`, against the recognition run's monthly roll `CSM(12) = 694305.3268248928`: the period form is higher by 0.134%, because the whole opening CSM accretes for twelve months before any release and the period's units enter undiscounted. The loss component has no such difference: the ratio form telescopes exactly (`347243.96338182886` against `347243.9633818289`).
- **The period's unit experience.** Variant P paid `1667000` against `1661301.8187426073` expected (fewer deaths); variant O paid `1640400` (more deaths). The difference is not used in the close; it belongs to the insurance service result (B97(c)).
- **Z1 and Z2** held with a worst relative difference of 1.02e-15 between a run at `V` and the recognition run shifted by 12 months (`FCF_z(0) = 17362216.290460557` against `F_prev(12) = 17362216.290460568`; `N_z(0) = 16591150.981337298` against `N^rec(12) = 16591150.9813373`), and `delta_FCF_L = 0` exactly with unchanged inputs.
- **Invariants.** I1 to I10 held on every policy of every run of both variants with worst relative residuals I4 4.2e-16, I5 2.5e-15, I6 2.6e-15, I9 1.3e-15; K1 to K7 and I11 to I13 held on both variants with worst residuals I12 1.5e-15, I13 1.7e-15, K7 1.7e-16 and exact zeros for K1, K2, K3, K5 and I11. G6 fired on a constructed run with two scenarios.

### 14.9 Implementations A and B

Implementation A is the month-by-month recursion of sections 6, 9 and 10. Implementation B, in the same script, computes the headline figures without those recursions: survival as a running product of `1 - q_mth`, the CPI instalment by the telescoped form of I9, `bel(t)` and `ra(t)` as suffix sums of cash flows discounted to time 0 and accumulated to `t` (the closed form of section 6.7 and the per-life values, with `ra(t)` carrying `pols_if(t) / pols_if_st(t)` because `v_st` conditions on stressed survival), `N(t)` as a suffix sum of discounted units, `CSM(t)` as a product of monthly factors `a_L(t) × N(t) / U(t)` and `LC(t)` as the ratio `LC(0) × F(t) / F(0)`. The period close is the same five formulas in both. On 102 values per variant (`bel(0)`, `ra(0)`, `pva`, `pve`, `ann_pp(1)` per policy and run; `FCF_g(0)` per run; `delta_FCF_L`; the seven close quantities; `N(0)`; `CU_g`, `R`, `CSM` and `LC` for `t = 1 .. 12`; `F_prev(12)`) the two agree to a maximum relative difference of 1.06e-13 for variant P (`R(3)`) and 2.35e-13 for variant O (`LC(5)`).

### 14.10 The projection against the annuity spec

The script first runs the projection of section 6 on the annuity spec's own example inputs (its policy 1, level, male born 1960-06-15, 1000 a month, and its policy 3, CPI-linked, male born 1955-02-20, 1200 a month, both new at 2025-12-31, `T = 655`) and reproduces every value that note prints to the 10 significant figures shown there: `bel(0) = 179192.1480`, `ra(0) = 2648.249534`, `pva = 175222.1063`, `pve = 3970.041706` for policy 1; `bel(0) = 234466.2049`, `ra(0) = 5607.325647`, `ann_pp(13) = 1232.981869` for policy 3; and for its example 1 group `CSM(0) = 8159.602426`, `CSM(12) = 7854.527657`, `R(1) = 46.56786380`, `R(12) = 46.31473923`; for its example 2 `LC(0) = 6840.397574`, `LC(12) = 6585.155578`. The projection and the monthly aggregation restated here are therefore the vault's, bit for bit as far as those prints show.

## 15. Mapping to Halley

What milestones M9 and M11 need from this document; the engine's syntax for the aggregation stage is not fixed yet (the annuity spec's open questions 15 to 17), so this is the contract, not the code.

| Item | Where | What the engine does |
|---|---|---|
| Group input contract | 3.3 | a `groups` table with the fields and rules of 3.3; at most one of `csm_prev`, `lc_prev` positive; dates month-ends; the three locked-in hashes resolved to files |
| Cross-run reads (M9) | 3.4 | by `prev_run_id`: the manifest, the previous model point files (membership, G4), the outputs `pols_if` at `tau`, `FCF_g` at 0 and `tau`, `LC` at `tau`; the previous non-financial tables by their hashes |
| Re-basing in the loader (M9) | 5.1, 5.2, 5.4 | `P_L` from the locked-in discount file and `tau0`; the locked-in CPI file is read as supplied (built outside, hashed); the previous expense table rebased with `I_act(0) / I_act(-tau)` and the price-base check |
| Four projection runs | 5.5 | ordinary runs of the annuity model with the inputs of 5.5; the aggregation stage takes three of them (`OLD`, `NEW`, `CUR`) plus the group inputs; the IR's absolute-time and negative-time CPI reads (`I(0)`, `I(t - 15)`) are the door the roadmap names |
| The close and the months | 9, 10 | one backward component (`N`), then the five close formulas, then the forward components `CSM`, `LC` with inits from the close; every division inside a `where` guarded |
| Invariants | 12 | I1 to I13 as the annuity spec defines them; K1 to K7 and Z1, Z2 added for in-force groups, with the per-invariant scales (the roadmap's open question 25 door) |
| Outputs and manifest | 11 | the lines of 11.1 and the five values of 11.2 per group; the manifest records `prev_run_id`, the three locked-in hashes, the CPI estimate flag and the period-allocation note of C11 |
| Checker rules exercised | 12.4 | G1 to G6 and the `groups` rules; J = 1 is a check in one place, not a shape assumption (D4) |

## 16. Decision log

Each item is one convention chosen in this document, with the reason. Object by number. Items the annuity spec already fixes (the projection, the monthly aggregation, Method 1 coverage units, the Example 8 loss-component ratio, the remeasurement rules) are not repeated.

- **C1. First subsequent measurement: `V_prev = V_rec`, `tau0 = tau = 12`.** Reason: the roadmap asks for one period after recognition; the previous run is then the recognition run, which the annuity spec already defines, and sections 5 to 10 are written with `tau0` and `tau` so that a later period changes nothing.
- **C2. Two variants sharing every input except the premiums and the period's mortality experience.** Reason: the same two lives give one onerous and one profitable group with one recognition projection; different experience gives `delta_FCF_L` of both signs on the same basis change, so the two reference closes exercise two different remeasurement rules and the table of 14.5 covers the rest.
- **C3. Grouped model points (100 and 50 lives) with integer actual survivors.** Reason: an in-force count that differs from expected needs more than one life per model point; integer survivors are what administration data gives; `cu_act` then has a life-month story a reviewer can follow.
- **C4. Two classes, level and CPI-linked; fixed escalation omitted.** Reason: both classes the period mechanics touch (expense inflation on every class, index-linked payments on one) are present; fixed escalation has no estimate that can change and would add sheets.
- **C5. Ages 78 and 80 at recognition.** Reason: `T = 489` at `V` keeps every sheet near the term spec's 480 rows while the SULT-based table and the improvement scale still bite; the annuity spec's 65-year-old would need 655 rows per sheet across eight sheets per variant.
- **C6. Every contract issued on the recognition date.** Reason: the annuity spec's in-force design requires it; a group with several issue dates would need a B73 weighted-average curve that no input supplies (section 17, question 5).
- **C7. The basis change at `V` is a multiplicative base-table change (0.96) and an expense level change (5.16 rebased to 5.30); improvement scale, investment-expense rate and RA parameters unchanged.** Reason: an experience study typically moves the base table by a factor; the expense change exercises the rebase-versus-change distinction of 5.4; changing fewer tables keeps `delta_FCF_L` attributable, and the unchanged tables still pass through the rebase and hash machinery.
- **C8. Expense rebase with actual CPI, `maint_pm_prev × I_act(0) / I_act(-tau)`, and a price-base check on every run at `V`.** Reason: the annuity spec's rule; an expense level that merely follows inflation is not a change in estimate, and the check stops a table priced at `V_prev` entering a run at `V`.
- **C9. Actual CPI: 3.2% a year over the period, the December 2025 index revised from 100.0 to 100.1 on publication, the December 2026 index an estimate used as actual.** Reason: a different rate from the 2.5% expected is the point of the example; the revision exercises "published values replace earlier estimates" (I_act(-12) is not I_rec(0)); the valuation-month estimate is the normal state of a CPI file at a month-end and the annuity spec's rule for it.
- **C10. Current basis at `V`: rates 50 basis points higher, expected inflation 3.0%, ILP unchanged.** Reason: both financial inputs move so that `E - D` of 11.2 has two components of opposite sign and neither locked-in run can be confused with the current one; the ILP is part of the locked-in basis and stays.
- **C11. Loss-component allocation over the period by the ratio `F_prev(tau) / F_prev(0)` of the previous run.** Reason: it is the annuity spec's monthly ratio telescoped, so the period step reproduces the previous run's own projected `LC(tau)` (K7), it is independent of the step length, and the previous run is the only expected-basis source of the para 51 items; the annuity spec left this formula open (its question 8) and section 17 question 1 offers the alternative.
- **C12. Accretion of the whole opening CSM for the period, then one release at the period level with `cu_act` undiscounted and `N(0)` at `V`.** Reason: the annuity spec's steps 1 and 4; the difference from twelve monthly steps is 0.134% of the CSM here (14.8) and is the subject of section 17 question 2.
- **C13. Discounted coverage units only.** Reason: one formula per quantity; the undiscounted switch is the annuity spec's comparison device, not a measurement.
- **C14. Coverage units of the in-force group from the `CUR` run, discounted on `P_L`.** Reason: the annuity spec's rule; Method 1 counts the payments expected to be claimable, a current estimate; locked-in discounting keeps the units free of changes in current rates.
- **C15. The actual instalment at `V` is an input, not derived from the CPI path.** Reason: the annuity spec's schema; the actual escalation history is administration data, and here it is trivially the recognition instalment because the first anniversary falls at month 1 of the runs at `V`.
- **C16. The `OLD` run carries the previous run's expected in-force at full precision as `pols_if_init`, and a model point with no survivors stays in `OLD` only.** Reason: `n_exp` is a cross-run read, never retyped (D21); a dead model point still had expected future service at `V_prev`, which is what `delta_FCF_L` must remove.
- **C17. Four runs, each an ordinary projection; no run-specific formula.** Reason: the engine reuses the annuity model unchanged (D9: no variant needed); the spreadsheet builder copies one sheet layout eight times and changes inputs.
- **C18. The analysis of change of 11.2 is an output of five values and four differences, with no revenue or IFIE formula.** Reason: M12 defines those; the five values are already computed and give the reviewer the whole story of the period without a new definition.
- **C19. The diagnostic split of `delta_FCF_L` (an extra run with old tables and actual in-force) is script-only.** Reason: the measurement does not need it; the spreadsheet would need two more sheets; the script prints it because a reviewer will ask which part is experience.
- **C20. Z1 and Z2 as engine checks on constructed inputs.** Reason: they test the re-basing, the splice, the expense rebase, the age shift and the anniversary shift together and would catch an off-by-one in any of them; the reference figures alone cannot show that `C - B` is pure CPI experience.
- **C21. Tolerance `1e-9 × max(|a|, |b|, scale)` with `P_g` as the group scale.** Reason: the term spec's rule (its C33) with the natural group magnitude; the close's residuals are comparisons to zero of amounts of the order of `P_g`.
- **C22. The script carries implementation B and the annuity spec cross-check.** Reason: the annuity spec's authors reconciled two independent scripts before the spreadsheet existed, and the cross-check ties this document's restatement to the vault design without the vault being in the repository.
- **C23. Premiums at 1.04 and 0.98 times the per-life fulfilment cash flows at recognition, rounded to cents.** Reason: a CSM of about 4% and a loss of about 2% of premium are plausible magnitudes and keep variant P profitable after a 2.8% unfavourable change while variant O's loss component absorbs a 1.4% favourable one; the rule is outside the model.
- **C24. Names and symbols follow the annuity spec (`csm_prev`, `lc_prev`, `cu_act`, `CSM_acc`, `LC_alloc`, `CSM_adj`, `LC_adj`, `R_per`, `delta_FCF_L`, `P_L`, `I_L`).** Reason: the engine's trace and the spreadsheet then read like the design note.

## 17. Open questions

These are the only items open; everything else is fixed as written and can be changed only by editing the relevant item in section 16.

1. **Loss-component allocation over the period (C11).** The ratio uses the previous run's expected fulfilment cash flows at `V_prev` and `V`. The alternative is a ratio built from the period's actual para 51 items (actual payments and expenses, the RA actually released, the actual finance effect), which would need the insurance service result that section 2 excludes. Which does the reviewer want, and should the engine compute both and report the difference?
2. **One release per period (C12).** The period-level release differs from twelve monthly releases by 0.134% of the CSM here (14.8). Should the engine instead run the twelve monthly steps of section 10 inside the period on the previous run's expected units, then remeasure and true up the release to `cu_act`? That would change steps 1 and 4 and the roll-forward lines, and interim reporting (B137) would follow from it.
3. **Mortality experience adjusts the CSM (section 8).** The effect of actual deaths on future payments is taken to adjust the CSM by inference from B96(b). A reviewer who reads B97(c) more widely would route it through the insurance service result; `delta_FCF_L` would then be measured on the expected in-force in both runs and the experience part reported separately. Confirm.
4. **`cu_act` as an input.** The payments actually made are a group input. Should they instead be derived from the actual in-force history (a per-month count file), so that the engine can check them against the model points, or is an administration total the right contract?
5. **Groups with several issue dates (C6).** Not allowed here. Should the design admit them with a supplied B73 weighted-average curve as the locked-in basis, or keep one issue date per group and let the cohort be the group?
6. **The CPI estimate at `V` (C9).** The valuation-month index is an estimate flagged in the file. Should the next period's run restate the previous run's `C - B` when the published value differs, or is the restatement absorbed in the next period's CPI experience as this document assumes?
7. **RA parameter changes (C7).** None here. A change in `alpha` or `sigma` would adjust the CSM at locked-in rates (B96(d)) by the same two runs; confirm that no separate disclosure of its effect is wanted before M11.
8. **Tolerance (C21).** Is `1e-9 × max(|a|, |b|, P_g)` acceptable for the close's cells, given that `R_per` and `acc_per` are three orders of magnitude below `P_g`, or should the close's lines carry their own scale (`csm_prev + lc_prev`)?
9. **Spreadsheet scope.** Eight projection sheets of about 500 rows per variant. Does the reviewer want both variants checked cell by cell, or variant P in full and variant O's `group` sheet only?

## Sources

- IASB (2017, amended June 2020). IFRS 17 *Insurance Contracts*, paragraphs 25, 28, 32–34, 38, 40, 44, 47–52, 87, 88, B26, B65–B66, B72–B74, B90, B96–B97, B101, B119, B124, B128, B137, and Illustrative Example 8. https://www.ifrs.org/issued-standards/list-of-standards/ifrs-17-insurance-contracts/
- IFRS Interpretations Committee (2022). Agenda decision, "Transfer of Insurance Coverage under a Group of Annuity Contracts". Cited for Method 1 coverage units. https://www.ifrs.org/content/dam/ifrs/supporting-implementation/agenda-decisions/2022/transfer-of-insurance-coverage-under-a-group-of-annuity-contracts-jul-2022.pdf
- EY (April 2019). Insurance Accounting Alert on the TRG meeting of 4 April 2019 (submission S122). Cited for CPI changes as financial risk. https://www.ey.com/content/dam/ey-unified-site/ey-com/en-gl/technical/ifrs-technical-resources/documents/ey-iaa-ifrs-17-insurance-trg-april-2019.pdf
- IFoA IFRS 17 CSM Working Party (2019). Findings, session D3. Cited for the loss-component allocation. https://vle.actuaries.org.uk/pluginfile.php/144100/mod_resource/content/1/IFRS%2017%20CSM%20Working%20Party%20Findings%20D3.pdf
- Dickson, D.C.M., Hardy, M.R. and Waters, H.R. (2009). *Actuarial Mathematics for Life Contingent Risks*, 1st ed. Cambridge University Press. The Standard Ultimate Survival Model; policy value recursions.
- Society of Actuaries (October 2022). Exam FAM-L Tables, Standard Ultimate Life Table. Cited for the check values. https://www.soa.org/globalassets/assets/files/edu/2022/2022-10-exam-fam-l-tables.pdf
- Society of Actuaries (2014). *Mortality Improvement Scale MP-2014 Report*, Appendix B. Cited for the convergence form of the improvement rates. https://www.soa.org/globalassets/assets/files/research/exp-study/research-2014-mp-report.pdf
- UK Debt Management Office (2005). *Formulae for Calculating Gilt Prices from Yields*, Annex B. Cited for the three-month indexation lag. https://www.dmo.gov.uk/media/1sljygul/yldeqns.pdf
- Halley notes in the author's vault: *Halley — Annuity Spec* (the design this document follows, including its sources), *Halley — Spec*, *Halley — Decisions*, *Halley — Roadmap*.

**Provenance.** Written on 2026-10-16 from the vault notes, the public sources above and general actuarial knowledge. Every number is synthetic and chosen for this document; the mortality basis is generated from published Makeham parameters and no SOA, CMI or other restricted table is copied. No employer code, models, assumption tables, product terms, results or data were consulted (D29). The premiums, the actual survivors and the actual CPI path were set so that the two variants land on two different remeasurement rules; nothing else was tuned.
