# Halley annuity: product specification (reference case 1)

Version 0.1, draft, 2026-10-16. Carried into the repository from the author's vault note *Halley — Annuity Spec*; not yet reviewed.

The engine rules this specification refers to live in the author's vault: *Halley — Spec*, *Halley — Decisions*, *Halley — Roadmap*.

This is the product specification for Halley's annuity sample: a single-life immediate annuity in payment, measured under IFRS 17 with the general measurement model (GMM). It is the written spec for the annuity's reference case 1 (D22). A spreadsheet is built from it and checked against it by the reviewer, and the engine reconciles to the spreadsheet. The note serves roadmap milestones M5 (annuity and stochastic) and M7 (aggregation stage), and holds the design that M9 to M12 build on (section "In-force groups (M9 to M11)", section "Outputs").

Engine rules are in *Halley — Spec*. This note adds what the product needs and flags where it needs more than the Spec says today. References such as "para 38" or "B72(b)" are to IFRS 17 *Insurance Contracts* as amended in June 2020. Every choice below is a v1 default with a one-line reason. Real uncertainty is listed under section "Open questions".

## Purpose and scope

The annuity exercises what term assurance does not: survival as the insured event, horizons of 50 years or more and large reserves (D2), and the IFRS 17 aggregation stage (D3, D6).

**In scope**
- Single-life annuity in payment, paid monthly in advance. No guarantee period, no death benefit, no surrender value.
- Base product: level (D2).
- Variants as subclasses (D2, D9), in M5 alongside the level annuity: fixed-rate and CPI-linked escalation, which add index-linked cash flows.
- Mortality from a closed-form base table with generational improvements.
- Maintenance expenses inflated with the scenario CPI path. Investment expenses as an explicit cash flow.
- Discounting with a per-scenario risk-free curve plus an illiquidity premium (ILP).
- Per policy: best-estimate liability (BEL), risk adjustment (RA) and coverage units.
- Aggregation stage for one group recognised at the valuation date: contractual service margin (CSM), loss component (LC) and their monthly roll-forward.
- Deterministic runs (one scenario) and stochastic projection runs over scenario files (D4).

**Out of scope**
- GUI, GPU, joint life, deferred annuities, nested stochastic, regulatory capital.
- Guarantee periods, value protection, apportionable annuities, payment in arrears, other payment frequencies.
- Caps and floors on CPI escalation.
- Acquisition cash flows, reinsurance held, tax.
- Contracts joining a group after its initial recognition (para 28, B73).
- The OCI option as an output (para 88(b)).
- Rounding of payments.

**Later milestones, in v1 (M9 to M12)**
- In-force groups in the aggregation stage, and with them any remeasurement of the CSM or LC (M9 to M11). The design is in section "In-force groups (M9 to M11)". The projection stage still values in-force policies.
- Scenario statistics in the aggregation stage, i.e. a mean over more than one scenario (M12).
- Insurance revenue and insurance finance income or expenses (IFIE) as outputs (M12). Their formulas are in section "Outputs".

## Product definition

### Contract terms

| Feature | Definition |
|---|---|
| Lives | One annuitant |
| Premium | One single premium, received on the issue date. No further premiums |
| Benefit | A monthly annuity payment |
| Payment dates | The 1st of each calendar month, from `annuity_start_date`, while the annuitant is alive on that date |
| Payment timing | Monthly in advance: each payment covers the calendar month it opens |
| On death | Payments stop. No payment for the rest of the month of death, no refund, no death benefit |
| Guarantee period | None |
| Surrender | None. The contract boundary is the annuitant's remaining lifetime (para 34) |
| Escalation | Set by the class (below). An anniversary is an anniversary of `annuity_start_date`. The payment due on an anniversary is the first payment at the new amount |

### Base product and variants (D2, D9)

| Class | Escalation on each anniversary | Overrides |
|---|---|---|
| `LevelAnnuity` (base, D2) | None | — |
| `EscalatingAnnuity` | Abstract. Adds the anniversary mask `anniv`; not run on its own | — |
| `FixedEscalationAnnuity` | Payment × (1 + `esc_rate`) | `esc_factor` |
| `CPILinkedAnnuity` | Payment × E(t), with E(t) = I(t − L) / I(t − L − 12), lag L = 3 months, no cap and no floor | `esc_factor` |

- A CPI-linked payment falls if the index falls. Caps and floors are a later variant: they add an option that one deterministic path does not value (B39, B48). Some public index-linked instruments carry one, such as the floor on the capital value at redemption of South Africa's CPI bonds (RSA 1999, "Index Ratio" in section 2).
- Lag L = 3 months. Reason: it is the lag of UK index-linked gilts issued since 2005, where the reference index for the 1st of a month is the index for the month three months earlier (DMO 2005, Annex B); the DMO calls this design the global standard. Other markets differ (four months for South Africa's CPI bonds, RSA 1999, section 2), so L is a build-time constant.
- Each variant overrides exactly one hook variable, `esc_factor`. No two parents define the same variable differently, so the D9 conflict rule never fires.
- Build-time switches (D8): `include_inv_exp: Static[bool] = True` on `LevelAnnuity`; `cpi_lag_m: Static[int] = 3` on `CPILinkedAnnuity`; `discount_coverage_units: Static[bool] = True` on the aggregation stage.

### IFRS 17 classification

- Insurance contract: survival is the insured event (B26(d)).
- The CPI link is not separated from the host contract (para 11(a), B10).
- No investment component: nothing is repaid in all circumstances (Appendix A). So there is no investment-return service (B119B(a)), and insurance coverage is the only service.
- Not eligible for the variable fee approach: there is no clearly identified pool of underlying items (B101).
- Not eligible for the premium allocation approach: coverage runs for decades and variability is significant (paras 53–54).
- Measured under the GMM (paras 32–52). Every future payment sits in the liability for remaining coverage (LRC, para 40). The liability for incurred claims (LIC) is zero in the model, because no payment is due and unpaid at the valuation date.

## Conventions

### Notation

| Symbol | Meaning |
|---|---|
| V | Valuation date, a month-end |
| mi(d) | Month index of a date: 12 × year(d) + month(d) − 1 |
| m_V | mi(V) |
| t | Integer months from V (D10). Month t is the interval (t − 1, t] |
| T | Projection horizon of the run, in months |
| T_i | First month at age ω for policy i: 12ω + 1 − a0_i |
| ω | Limiting age, 120 |
| P(t) | Current discount factor from time t to time 0, for the scenario: P(0, t) |
| v(t) | One-month discount factor of month t: P(t) / P(t − 1) |
| f(t) | One-month effective forward rate of month t: 1 / v(t) − 1 |
| P_L(t) | Locked-in discount factor of the group, re-based to V |
| a_L(t), j_L(t), d_L(t) | Locked-in accumulation factor P_L(t − 1) / P_L(t), its rate a_L(t) − 1, and the discount factor P_L(t) / P_L(t − 1) |
| I(u) | CPI value for calendar month m_V + u |
| E(t) | CPI escalation ratio applied in month t |
| a0 | Age at V in months (see section "Ages") |
| m_c | mi(`annuity_start_date`) − m_V; the month of the first payment, relative to V |
| Acc(t), R(t) | CSM interest accretion and CSM release in month t |
| N(t), U(t) | Coverage units still to be provided after month t, and U(t) = CU_g(t) + N(t) |
| LCA(t) | Loss-component allocation in month t |
| s | Scenario index, 0..J − 1 |

### Time

- V is a run setting. It must be the last day of a calendar month; the loader rejects anything else. Valuation is at the end of that day.
- Month t coincides with calendar month m_V + t. Time t − 1 is its start, the 1st of that month. Time t is its end. The one-day gap between the end of V and the 1st of the next month is ignored.
- Calendar year of month t: Y(t) = (m_V + t) // 12.
- There are no day counts. A month is 1/12 of a year, and τ = t / 12 is time in years.
- Domains. A variable with an `init` or `terminal`, or a state ("at t"), is on 0..T: `pols_if`, `ann_pp`, `df`, `bel`, `v_be`, `v_st`, `ra`. Scalars (`ie_m`, `mort_mult`, `exp_mult`) have none. Every other variable is on 1..T. The checker resolves offsets over these domains only and derives each input's range from them: `rf_df` over t = 0..T, `cpi` over u = −(L + 11)..T − 1, table keys over t = 1..T. Any other out-of-domain read is a build error (D11). ann_pp(0) is an init value only (the instalment in force at V); its start timing applies to t = 1..T. The derived `cpi` range ends at T − 1 because nothing reads I(T). The scenario-file contract (U_f ≥ T) deliberately asks for one more row than the model reads, so that `cpi` and `discount` both end at T; the loader checks the contract, not the derived range.
- Horizon: every policy's in-force is exactly zero from its own T_i on. A projection run reads every class file of a model point set, and its horizon is T = max_i T_i = 12ω + 1 − min_i a0_i over all of them.
- Dependence on T:
  - In-force-weighted values do not depend on T. pols_if(t), bel(t), ra(t) and inv_exp(t) are exactly 0 for t ≥ T_i; deaths, ann_out, exp_out and cu are exactly 0 for t > T_i.
  - Per-life and per-policy values (v_be, v_st, s_pp, s_pp_st, ann_pp, exp_pp) continue past T_i. They do not depend on T for t < T_i (ann_pp, exp_pp, s_pp and s_pp_st for t ≤ T_i). Beyond that they are not meaningful, because no life is in force, and they are excluded from every comparison.

### Ages

- a0 = m_V − mi(`date_of_birth`). This is the age at V in completed months, counted as if every birth were at the end of its birth month. The day of birth is ignored, so a0 can understate age by up to one month (a birth on 1 January 1960 gives 65 years 11 months at 31 December 2025). Every implementation does the same.
- Lookup age in month t: x(t) = min(ω, (a0 + t − 1) // 12). This is age last birthday at the start of month t, on the same convention.
- x(t) steps up in the month after the birthday month (July, for a June birth).
- The clamp at ω is explicit in the model and keeps table keys in range without silent extrapolation. Beyond ω the in-force is zero, so the value looked up does not matter.
- Ages run from the date of birth, not from issue. lifelib differs (open question 23).

### Order of events in month t

1. Time t − 1: the annuity payment and the maintenance expense are paid for each of the pols_if(t − 1) lives in force.
2. During the month: deaths(t) = pols_if(t − 1) × q_mth(t). Death is the only decrement; there are no lapses.
3. Time t: the investment expense for the month is paid, on the closing BEL. pols_if(t) is the in-force after the month's deaths.

### Cash-flow timing

| Cash flow | Variable | Base | Timing | Paid at time | Discount to t = 0 |
|---|---|---|---|---|---|
| Annuity payment | `ann_out(t)` | pols_if(t − 1) | start | t − 1 | P(t − 1) |
| Maintenance expense | `exp_out(t)` | pols_if(t − 1) | start | t − 1 | P(t − 1) |
| Investment expense | `inv_exp(t)` | bel(t) | end | t | P(t) |
| Single premium | — | — | received on the issue date | before t = 0 | not in the BEL (para 38(b)) |

No cash flow in this product is mid-month (`pv_month` would use v(t)^(1/2)).

### Discounting

- Bottom-up approach (B80): a liquid risk-free curve plus an ILP. Reason: a non-surrenderable annuity is the illiquid liability that B79 describes. Keeping the ILP separate from the scenario file also makes its effect visible and versioned. The top-down approach (B81–B85) is not used.
- Composed current curve for each scenario: P(t) = `rf_df`(t) × exp(−ilp × t / 12), with the ILP continuously compounded. In a deterministic run (J = 1) the composed curve is the para 120 curve. When `rf_df` is a pathwise deflator, the per-scenario rates are diagnostics, not para 120 output. The disclosed curve is the deterministic market-implied composed curve, which before M12 is the J = 1 run that feeds the aggregation stage.
- One-month factor v(t) = P(t) / P(t − 1); one-month effective forward rate f(t) = 1 / v(t) − 1.
- Rates carry no adjustment for non-financial risk (B90).
- Nominal cash flows, including CPI-linked ones, are discounted at the nominal curve (B74).

### Compounding and unit conversions

Every conversion is a named helper, so the choice is visible in the model source (D11). Helper names are provisional. The formulas are standard compound-interest and fractional-age results (Kellison 2009, chapter 1; Dickson, Hardy and Waters 2009, section 3.3).

| Helper | From → to | Formula | Used by |
|---|---|---|---|
| `to_monthly(i)` | Rate[Annual, Effective] → Rate[Monthly] | (1 + i)^(1/12) − 1 | investment expense rate |
| `to_monthly_cf(q)` | Prob[Annual] → Prob[Monthly] | q_s = where(q ≥ 1, 0, q); result = where(q ≥ 1, 1, −expm1(log1p(−q_s) / 12)) | mortality |
| `annual_step(g)` | Rate[Annual, Effective] → Factor | 1 + g, applied once a year | fixed escalation |
| `cc_discount(s, months=m)` | Rate[Annual, Continuous] → Factor | exp(−s × m / 12) | ILP |
| `fwd_rate(p0, p1)` | Factor, Factor → Rate[Monthly] | p0 / p1 − 1 | locked-in accretion rate j_L |
| `pv_month(cf, t, v)` | cash flow → value at time t − 1 | start: cf(t); mid: cf(t) × v(t)^(1/2); end: cf(t) × v(t) | BEL and per-life values |
| `as_cu(x)` | Money → CoverageUnits | same value, new unit | coverage units |

For disclosure only: annualised one-month forward (P(t − 1) / P(t))^12 − 1; annualised spot P(t)^(−12/t) − 1.

- `to_monthly_cf` is mathematically 1 − (1 − q)^(1/12). The guarded form avoids evaluating log1p(−1), which is −inf in IEEE arithmetic, an exception in Python's `math.log1p` and a warning in NumPy. `where()` evaluates both branches, so the guard is on the argument, not only on the result.
- A spreadsheet (D22) may use 1 − (1 − q)^(1/12), which also gives exactly 1 at q = 1.
- The monthly mortality conversion assumes a constant force of mortality within each year of age. Reason: it depends only on the annual rate, so it is a pure unit conversion; it gives exactly 1 when q = 1; and it is lifelib's convention. Uniform distribution of deaths is closer to exact Makeham survival (an error of about 0.008% against 0.027% on a monthly annuity for a 65-year-old) but needs the month within the year of age. It is not used.

### Limiting age

- ω = 120. Reason: the closure truncates a negligible tail. On the example base table before improvement, survival from 65 to 120 is about 4e-13, and the value of a level annuity beyond 120 is about 3e-15 of the total.
- q = 1 at age 120 for both sexes, in every calendar year, on both the best-estimate and the stressed basis. The model forces it in `q_ann` and `q_ann_st`; it does not rely on the tables (see section "Assumption tables").
- Under constant force, q = 1 gives a monthly q of 1. So every life still in force dies in the first month at age 120, and pols_if is exactly 0 from then on.

### Rounding

- None. Amounts, index ratios, probabilities and discount factors are float64 and never rounded.
- Reason: rounding breaks linearity and the exact invariants (scaling, roll-forward, telescoping), and its effect on the BEL is immaterial.

### Evaluation order

- The model source fixes the evaluation order. Each formula is evaluated with the grouping shown there, and a chain a + b + c means (a + b) + c. IEEE float64 + and × are commutative, so only the grouping matters, not the order of operands.
- A second, algebraically equal form shown in this note, such as the folded BEL recursion or the SAR form of the LC, explains a formula. It is not an evaluation order.
- Choices that implementers could otherwise make differently:
  - pols_if(t) = pols_if(t − 1) − deaths(t), not a cumulative product of 1 − q_mth.
  - ann_pp by its recursion, one month at a time.
  - q_mth by the guarded expm1/log1p form; ie_m = (1 + ie_rate)^(1/12) − 1 with a power function.
  - Group sums: sequential, left to right, in ascending `policy_id` (D19). Not pairwise summation.
  - CSM and LC: as written in section "Accretion and release" and section "Onerous groups and the loss component". FCF = BEL + RA is formed first. CSM(0) and G3 use P − FCF, and LC(0) uses FCF − P; never (P − BEL) − RA.
  - RA: the per-life form pols_if(t) × (v_st(t) − v_be(t)), not a stressed direct PV less the BEL.
- Another grouping changes a result by about one unit in the last place per operation, far inside the worked example's 1e-9 tolerance, so implementations of it need not follow this order (section "Quantities"). Scripts that do follow it agree month by month to about 1e-14 relative; the RA, a difference of two values about 40 to 70 times its size, agrees correspondingly less closely.

### Units

| Unit | Used for | Status |
|---|---|---|
| `Money` | in-force cash flows, BEL, RA, CSM, LC | Spec |
| `PerPolicy[Money]` | instalment, expense per policy, per-life values | Spec ("per-policy") |
| `Count` | `pols_if`, `deaths` | Spec |
| `Rate[Monthly]` | monthly investment-expense rate, locked-in one-month rate | Spec |
| `Rate[Annual, Effective]`, `Rate[Annual, Continuous]` | escalation and investment-expense rates; ILP | Spec has `Rate[Annual]`; the compounding tag is proposed |
| `Prob[Annual]`, `Prob[Monthly]` | annual and monthly mortality | Proposed: a period tag on `Prob` |
| `Factor` | discount factors, escalation factors, improvement factors, stress multipliers | Proposed: dimensionless |
| `Index` | CPI | Proposed |
| `CoverageUnits` | coverage units | Proposed, so units cannot be added to money |
| `Int`, `Bool` | lookup ages, calendar years, anniversary mask | Proposed: exact integer and boolean types |

Rules the model relies on:
- `Count × PerPolicy[Money] → Money`; `Count × Prob[Monthly] → Count`.
- `Prob[Monthly] × PerPolicy[Money] → PerPolicy[Money]` (an expected amount per life); `1 − Prob[P] → Prob[P]`.
- `Count × Prob[Annual]` and `Prob[Annual] × PerPolicy[Money]` are errors, because the time step is monthly (D10).
- `Prob[P] × Factor → Prob[P]`: a scaled probability, such as q × cif or q × mort_mult. It is range-checked at run time, as the Spec requires for `Prob`.
- Every rule carries the tag P. An untagged `Prob` (only α, read by a load check) enters no rule.
- `Rate[Monthly] × Money → Money` and `Rate[Monthly] × PerPolicy[Money] → PerPolicy[Money]` (a charge for one month).
- `Factor × X → X` for X in `Money`, `PerPolicy[Money]`, `CoverageUnits`, `Factor`.
- `Factor / Factor → Factor`; `Index / Index → Factor`; `Money / Money → Factor`; `CoverageUnits / CoverageUnits → Factor`.
- Dimensionless and exact types: `Factor ± Factor → Factor`; `−Factor → Factor`; `Factor / literal → Factor`; `exp(Factor) → Factor`; `Int` with `+`, `−`, `//`, `%` or `minimum` → `Int`; `Int` comparisons → `Bool`; `Bool & Bool → Bool`.
- `CoverageUnits` adds only to `CoverageUnits`.
- There is no rule for 1 + Rate. Every compounding step goes through a named helper, and each helper accepts only its compounding tag: `to_monthly` and `annual_step` take `Rate[Annual, Effective]`, `cc_discount` takes `Rate[Annual, Continuous]`. So `to_monthly(ilp)` fails the build.
- The model never multiplies a discount factor by a survival probability directly. By the `Prob × Factor` rule the product would be typed `Prob`, and it can exceed 1 when rates are negative. The per-life recursions therefore multiply the factor into the money amount first (see `v_be`).

## Inputs

### Run settings

| Setting | Type | Rule |
|---|---|---|
| `valuation_date` | date | Last day of a calendar month |
| `m_val` | int32, derived | mi(`valuation_date`) |
| `scenario_set` | identifier | Names the discount and CPI files; J scenarios |
| `T` | int32, derived | 12ω + 1 − min a0 over every class file of the run; at most 841 (a0 ≥ 600) |
| `backend` | enum | NumPy reference in reference case 1 (D16) |

All settings, switches and derived values go into the run manifest (D21).

### Model point schema

Each class has its own model point file (`mp_level`, `mp_fixed`, `mp_cpi`). Subclasses extend the base schema. Fields are validated at load.

| Field | Arrow type | Unit | Range or values | Notes |
|---|---|---|---|---|
| `policy_id` | int64 | — | ≥ 1; unique across all classes in a run | Canonical order (D19): the loader sorts each file by it and forms batches in that order |
| `portfolio` | dictionary<string> | — | non-empty | Group key |
| `cohort` | int16 | — | [1900, 2200]; must equal year(`issue_date`) | Group key; para 22 |
| `profit_bucket` | enum `ProfitBucket` | — | `ONEROUS`, `NO_SIGNIFICANT_RISK`, `OTHER` | Group key; para 16 |
| `sex` | enum `Sex` | — | `F`, `M` | Mortality key |
| `date_of_birth` | date32 | — | a0 in [600, 1320] months (ages 50 to 110) | |
| `issue_date` | date32 | — | ≤ V; < `annuity_start_date` | Initial recognition date (para 25). Must equal V in the one-group aggregation stage of M7 (G5, lifted in M11) |
| `annuity_start_date` | date32 | — | day of month = 1; ≤ V + 1 day | First payment date; anniversaries run from it |
| `pols_if_init` | float64 | Count | (0, 1e6] | 1.0 for a single policy |
| `instalment_pm` | float64 | PerPolicy[Money] | (0, 1e6] | Monthly payment in force at V, after every escalation up to V. If m_c = 1, the first payment |
| `single_premium` | float64 | PerPolicy[Money] | ≥ 0; > 0 if `issue_date` = V | Used only at initial recognition |

`FixedEscalationAnnuity` adds:

| Field | Arrow type | Unit | Range | Notes |
|---|---|---|---|---|
| `esc_rate` | float64 | Rate[Annual, Effective] | [0, 0.10] | Compound, once a year on the anniversary |

`CPILinkedAnnuity` adds no field. Its lag is the class constant `cpi_lag_m`.

Derived at load from dates (D10):

| Field | Type | Definition |
|---|---|---|
| `a0` | int32 | m_V − mi(`date_of_birth`) |
| `m_c` | int32 | mi(`annuity_start_date`) − m_V. Always ≤ 1: the annuity is in payment by month 1 |
| `t_end` | int32 | T_i = 12ω + 1 − a0. Used by checks and output filters, not by the model formulas |

All money inputs are monthly amounts. No per-annum money amount enters the model, so no untracked division by 12 exists.

### Assumption tables

Each table is a Parquet file with typed keys, declared key ranges, a value unit, a version, a source and a licence in its metadata. Its hash goes into the manifest (D21). At load, every lookup key over t = 1..T must fall inside the table's range. Because x(t) and Y(t) are monotone in t, checking the first and last month is enough.

| Table | Keys | Value | Unit | Value range | Constraints |
|---|---|---|---|---|---|
| `mort_base` | `sex`: Sex; `age`: int16 [20, 120] | `q` | Prob[Annual] | [0, 1] | q(·, 120) = 1. Metadata: `base_year` |
| `mort_improvement` | `age`: int16 [20, 120]; `year`: int16 [base_year + 1, 2150] | `r` | Rate[Annual] | [−0.05, 0.15] | Source of `mort_cif`; not read by the model, so `r` sits outside the unit system and needs no compounding tag. r(120, ·) = 0 |
| `mort_cif` | `age`: int16 [20, 120]; `year`: int16 [base_year, 2150] | `cif` | Factor | (0, 3] | cif(·, base_year) = 1; cif(x, y) = cif(x, y − 1) × (1 − r(x, y)), multiplied in ascending year order; cif(120, y) = 1 for every y. Metadata: hash of `mort_improvement` |
| `expense` | none | `maint_pm` | PerPolicy[Money], per month | [0, 1e4] | Price base: valuation month (u = 0), recorded in the metadata |
| `inv_expense` | none | `ie_rate` | Rate[Annual, Effective] | [0, 0.01] | |
| `ilp` | none | `ilp` | Rate[Annual, Continuous] | [0, 0.02] | |
| `ra_param` | none | `alpha`; `z_alpha`; `sigma_l`; `sigma_e` | Prob; Factor; Factor; Factor | alpha in (0.5, 1); sigmas in [0, 0.5] | z_alpha = Φ⁻¹(alpha) within 1e-12; sigma_e ≤ z_alpha (see section "Risk adjustment") |

- Cross-table check at load: q_base(s, x) × cif(x, y) ≤ 1 for every (s, x, y) on the declared key grid. Together with cif(120, y) = 1, this keeps q_ann a probability and closes the table at ω. Either failure is a data-contract error naming the table row.
- `mort_cif` is generated outside the model from `mort_improvement`. Reason: the improvement product runs over calendar years at a fixed age, while age and year advance together along t, so it cannot be written as a recursion in t.
- The run does not load `mort_improvement`. The manifest records `mort_cif`'s hash, which covers the improvement table through the `mort_improvement` hash in `mort_cif`'s metadata.
- Improvement rates do not depend on sex in v1. A sex key can be added later without changing the model.
- Tables without keys hold one row (open question 17).

### Scenario files (D4)

| File | Keys | Value | Unit | Range | Constraints |
|---|---|---|---|---|---|
| `discount` | `scenario`: int32 [0, J); `t`: int32 [0, T_f], T_f ≥ T | `rf_df` | Factor | (0, 2] | rf_df(s, 0) = 1 exactly; every (s, t) present |
| `cpi` | `scenario`: int32 [0, J); `u`: int32 [u_min, U_f], U_f ≥ T | `cpi` | Index | (0, 1e6] | u_min ≤ −(L + 11) = −14; rows with u ≤ 0 identical across scenarios; every (s, u) present |

- A file must cover t = 0..T and u = u_min..T. For `cpi` this is one row beyond the range derived in section "Time", which ends at T − 1; the row is required by convention, so that `cpi` and `discount` both end at T. Rows beyond T are allowed and ignored, so one file set serves runs with different horizons.
- `rf_df` is the liquid risk-free discount factor from time t to time 0: a term structure in a deterministic run, possibly a pathwise deflator in a stochastic run. The engine treats both the same.
- The engine does not interpolate or extrapolate curves. Building the file, including any compounding conversion, interpolation and extrapolation, happens before the run, and the method is recorded in the file metadata.
- `cpi` for u ≤ 0 is published history or, for months not yet published at V (typically u = 0, the valuation month), an estimate fixed across scenarios. The file metadata flags estimated rows and the manifest records them. The CPI variant reads I(t − 3) and I(t − 15), and `where()` evaluates both branches for every month, hence u_min ≤ −14. Expenses read I(t − 1) for t ≥ 1. Reason: the expense paid at time 0 is at the price base, and each later one carries the inflation of the whole months before it; a lag of L would also be defensible.
- The CPI path and the discount curve of a scenario must be consistent (B59); see section "Scenarios" for the deterministic run.

### Group inputs

The one-group stage of M7 aggregates only groups that are new at V (G5), and those need no group input. The input contract for in-force groups (M9) is in section "In-force groups (M9 to M11)".

## Projection stage

### Variables

Dimensions (D14): `policy`, `scenario`, `time`. A variable with an `init` is forward-recursive. A variable with a `terminal` is backward-recursive. Others are non-recursive.

Timing: `start` and `end` are the Spec's labels for an amount that belongs to month t. "at t" marks a value at the instant t (a state); it is not a Spec label, and the source declares no timing for it. "—" marks keys, rates, factors and quantities.

| Variable | Dims | Unit | Timing | Definition | Kind |
|---|---|---|---|---|---|
| `age` | policy × time | Int | — | min(ω, (a0 + t − 1) // 12) | non-recursive |
| `cal_year` | time | Int | — | (m_V + t) // 12 | non-recursive |
| `q_ann` | policy × time | Prob[Annual] | — | 1 if age(t) ≥ ω, else mort_base[sex, age(t)] × mort_cif[age(t), cal_year(t)] | non-recursive |
| `q_mth` | policy × time | Prob[Monthly] | — | to_monthly_cf(q_ann(t)) | non-recursive |
| `deaths` | policy × time | Count | — | pols_if(t − 1) × q_mth(t) | forward component |
| `pols_if` | policy × time | Count | at t | pols_if(t − 1) − deaths(t); init pols_if_init | forward component |
| `anniv` | policy × time | Bool | — | k = t − m_c; k > 0 and k mod 12 = 0 (escalating classes only) | non-recursive |
| `esc_factor` | policy × time, or policy × scenario × time for CPI | Factor | — | Level: 1. Fixed: 1 + esc_rate if anniv(t), else 1. CPI: E(t) = I(t − L) / I(t − L − 12) if anniv(t), else 1 | non-recursive |
| `ann_pp` | policy × time, or policy × scenario × time for CPI | PerPolicy[Money] | start | ann_pp(t − 1) × esc_factor(t); init instalment_pm | forward |
| `exp_pp` | scenario × time | PerPolicy[Money] | start | maint_pm × (I(t − 1) / I(0)) | non-recursive |
| `ann_out` | policy × time, or policy × scenario × time | Money | start | pols_if(t − 1) × ann_pp(t) | non-recursive |
| `exp_out` | policy × scenario × time | Money | start | pols_if(t − 1) × exp_pp(t) | non-recursive |
| `ie_m` | scalar | Rate[Monthly] | — | to_monthly(ie_rate) if include_inv_exp, else 0 | non-recursive |
| `inv_exp` | policy × scenario × time | Money | end | ie_m × bel(t) if include_inv_exp, else 0 | backward component with `bel` |
| `df` | scenario × time | Factor | at t | rf_df(t) × cc_discount(ilp, months=t) | non-recursive |
| `v` | scenario × time | Factor | — | df(t) / df(t − 1) | non-recursive |
| `bel` | policy × scenario × time | Money | at t | see below; terminal 0 | backward |
| `s_pp` | policy × scenario × time | PerPolicy[Money] | start | ann_pp(t) + exp_pp(t) | non-recursive |
| `v_be` | policy × scenario × time | PerPolicy[Money] | at t | see below; terminal 0 | backward |
| `mort_mult`, `exp_mult` | scalar | Factor | — | exp(−σ_L z_α − σ_L² / 2), exp(σ_E z_α − σ_E² / 2) | non-recursive |
| `q_ann_st` | policy × time | Prob[Annual] | — | 1 if age(t) ≥ ω, else q_ann(t) × mort_mult | non-recursive |
| `q_mth_st` | policy × time | Prob[Monthly] | — | to_monthly_cf(q_ann_st(t)) | non-recursive |
| `s_pp_st` | policy × scenario × time | PerPolicy[Money] | start | ann_pp(t) + exp_pp(t) × exp_mult | non-recursive |
| `v_st` | policy × scenario × time | PerPolicy[Money] | at t | as v_be on the stressed basis; terminal 0 | backward |
| `ra` | policy × scenario × time | Money | at t | pols_if(t) × (v_st(t) − v_be(t)) | non-recursive |
| `cu` | policy × time, or policy × scenario × time | CoverageUnits | — | as_cu(ann_out(t)), i.e. pols_if(t − 1) × ann_pp(t) | non-recursive |

The backward recursions, for t = 0..T − 1, written in the evaluation order of the model source:

```text
bel(t)  = ann_out(t+1) + exp_out(t+1) + inv_exp(t+1) × v(t+1) + v(t+1) × bel(t+1)                    bel(T)  = 0
v_be(t) = s_pp(t+1)    + (1 − q_mth(t+1))    × (v(t+1) × (v_be(t+1) + ie_m × v_be(t+1)))              v_be(T) = 0
v_st(t) = s_pp_st(t+1) + (1 − q_mth_st(t+1)) × (v(t+1) × (v_st(t+1) + ie_m × v_st(t+1)))              v_st(T) = 0
```

Components and directions (Spec checker rules):

| Component | Internal offsets | Direction |
|---|---|---|
| {`pols_if`, `deaths`} | 0, −1 | forward |
| {`ann_pp`} | −1 | forward |
| {`bel`, `inv_exp`} | +1, +1, 0 | backward; the cycle sums to +1, so it is not a same-period cycle |
| {`v_be`} | +1 | backward |
| {`v_st`} | +1 | backward |

No component mixes directions (D13). `ra` combines a forward output with backward outputs, which is legal because it is not in a cycle. In each recursive component, the member whose formula, over its own domain (see section "Time"), would read outside a domain carries the `init` or `terminal`: `pols_if` at 0, `bel` at T. `deaths` and `inv_exp`, on 1..T, never do. The Spec's wording on this is open question 17.

### Model in Halley's authoring syntax

The Spec does not yet fix import paths, table-lookup syntax (`self.tables.<name>[keys]`), scenario inputs (`self.scen.<name>(t)`), absolute-time reads (`.at(0)`), run settings (`self.run`) or load-time derived fields (`self.mp.a0`). Those names are provisional.

```python
OMEGA = 120                                        # limiting age: q = 1 at OMEGA


class LevelAnnuity(Product):
    """Single-life annuity in payment, level, paid monthly in advance (D2)."""

    include_inv_exp: Static[bool] = True           # B65(ka)(i) judgement

    # ---- time-derived lookup keys (D10) ----------------------------------
    @var
    def age(self, t) -> Int:                       # age last birthday, start of month t
        return minimum((self.mp.a0 + t - 1) // 12, OMEGA)

    @var
    def cal_year(self, t) -> Int:                  # calendar year of month t
        return (self.run.m_val + t) // 12

    # ---- mortality --------------------------------------------------------
    @var
    def q_ann(self, t) -> Prob[Annual]:            # closed at OMEGA whatever the tables hold
        return where(self.age(t) >= OMEGA, 1.0,
                     self.tables.mort_base[self.mp.sex, self.age(t)]
                     * self.tables.mort_cif[self.age(t), self.cal_year(t)])

    @var
    def q_mth(self, t) -> Prob[Monthly]:
        return to_monthly_cf(self.q_ann(t))        # guarded 1 - (1 - q)^(1/12)

    @var
    def deaths(self, t) -> Count:
        return self.pols_if(t-1) * self.q_mth(t)   # Count × Prob[Monthly] → Count

    @var(init=lambda mp: mp.pols_if_init)
    def pols_if(self, t) -> Count:
        return self.pols_if(t-1) - self.deaths(t)

    # ---- annuity amount ---------------------------------------------------
    @var
    def esc_factor(self, t) -> Factor:             # hook: variants override this only
        return 1.0

    @var(timing="start", init=lambda mp: mp.instalment_pm)
    def ann_pp(self, t) -> PerPolicy[Money]:
        return self.ann_pp(t-1) * self.esc_factor(t)

    # ---- expenses ----------------------------------------------------------
    @var(timing="start")
    def exp_pp(self, t) -> PerPolicy[Money]:       # valuation-month prices, CPI-inflated
        return (self.tables.expense.maint_pm
                * (self.scen.cpi(t-1) / self.scen.cpi.at(0)))     # Index / Index → Factor

    @var
    def ie_m(self, t) -> Rate[Monthly]:
        if self.include_inv_exp:                   # build-time branch (D8)
            return to_monthly(self.tables.inv_expense.ie_rate)
        return 0.0

    # ---- cash flows --------------------------------------------------------
    @var(timing="start")
    def ann_out(self, t) -> Money:
        return self.pols_if(t-1) * self.ann_pp(t)  # Count × PerPolicy[Money] → Money

    @var(timing="start")
    def exp_out(self, t) -> Money:
        return self.pols_if(t-1) * self.exp_pp(t)

    @var(timing="end")
    def inv_exp(self, t) -> Money:
        if self.include_inv_exp:                   # unused branch never enters the graph
            return self.ie_m(t) * self.bel(t)      # Rate[Monthly] × Money: charge on the closing BEL
        return 0.0

    # ---- discounting ---------------------------------------------------------
    @var
    def df(self, t) -> Factor:                     # composed current curve P(t)
        return self.scen.rf_df(t) * cc_discount(self.tables.ilp.ilp, months=t)

    @var
    def v(self, t) -> Factor:                      # one-month factor of month t
        return self.df(t) / self.df(t-1)

    # ---- best-estimate liability ----------------------------------------------
    @var(terminal=0)
    def bel(self, t) -> Money:                     # value at t of months t+1..T
        return (pv_month(self.ann_out, t+1, self.v)
                + pv_month(self.exp_out, t+1, self.v)
                + pv_month(self.inv_exp, t+1, self.v)
                + self.v(t+1) * self.bel(t+1))

    # ---- per-life values, best estimate ------------------------------------------
    @var(timing="start")
    def s_pp(self, t) -> PerPolicy[Money]:
        return self.ann_pp(t) + self.exp_pp(t)

    @var(terminal=0)
    def v_be(self, t) -> PerPolicy[Money]:         # value at t per life in force at t
        p = 1 - self.q_mth(t+1)                    # Prob[Monthly]
        nxt = self.v_be(t+1) + self.ie_m(t+1) * self.v_be(t+1)   # Rate[Monthly] × PerPolicy[Money]
        return pv_month(self.s_pp, t+1, self.v) + p * (self.v(t+1) * nxt)

    # ---- risk adjustment: one mean-one latent factor at confidence level alpha ------
    @var
    def mort_mult(self, t) -> Factor:
        ra = self.tables.ra_param
        return exp(-ra.sigma_l * ra.z_alpha - ra.sigma_l * ra.sigma_l / 2)

    @var
    def exp_mult(self, t) -> Factor:
        ra = self.tables.ra_param
        return exp(ra.sigma_e * ra.z_alpha - ra.sigma_e * ra.sigma_e / 2)

    @var
    def q_ann_st(self, t) -> Prob[Annual]:
        return where(self.age(t) >= OMEGA, 1.0, self.q_ann(t) * self.mort_mult(t))

    @var
    def q_mth_st(self, t) -> Prob[Monthly]:
        return to_monthly_cf(self.q_ann_st(t))

    @var(timing="start")
    def s_pp_st(self, t) -> PerPolicy[Money]:
        return self.ann_pp(t) + self.exp_pp(t) * self.exp_mult(t)

    @var(terminal=0)
    def v_st(self, t) -> PerPolicy[Money]:
        p = 1 - self.q_mth_st(t+1)
        nxt = self.v_st(t+1) + self.ie_m(t+1) * self.v_st(t+1)
        return pv_month(self.s_pp_st, t+1, self.v) + p * (self.v(t+1) * nxt)

    @var
    def ra(self, t) -> Money:
        return self.pols_if(t) * (self.v_st(t) - self.v_be(t))

    # ---- coverage units (B119) ------------------------------------------------------
    @var
    def cu(self, t) -> CoverageUnits:              # amount payable × expected lives entitled
        return as_cu(self.ann_out(t))


class EscalatingAnnuity(LevelAnnuity):
    """Abstract: adds the anniversary mask. Not run on its own."""

    @var
    def anniv(self, t) -> Bool:                    # month t's payment falls on an anniversary
        k = t - self.mp.m_c                        # months since the first payment
        return (k > 0) & (k % 12 == 0)


class FixedEscalationAnnuity(EscalatingAnnuity):

    @var
    def esc_factor(self, t) -> Factor:
        return where(self.anniv(t), annual_step(self.mp.esc_rate), 1.0)


class CPILinkedAnnuity(EscalatingAnnuity):
    cpi_lag_m: Static[int] = 3                     # reference lag L, in months

    @var
    def esc_factor(self, t) -> Factor:
        L = self.cpi_lag_m                         # build-time constant, so offsets are fixed
        return where(self.anniv(t),
                     self.scen.cpi(t - L) / self.scen.cpi(t - L - 12), 1.0)
```

Notes on the model:
- The CPI lag is a build-time constant, so `t - L` and `t - L - 12` are fixed offsets and need no `lag()` (D12). A lag that varied by policy would need two reads, `lag(cpi, L, min=L_min, max=L_max)` and `lag(cpi, L + 12, min=L_min + 12, max=L_max + 12)`, and CPI history back to −(L_max + 11).
- In the level class `esc_factor` is 1, so `ann_pp` is constant. Multiplying by 1.0 is exact in float64.
- `exp_pp` has no policy dimension, so the expense cash flows of every class carry the scenario dimension.

## BEL and the reserve recursion

### Definition

BEL(t) is the value at time t, on the scenario's current composed curve, of all fulfilment cash flows of months t + 1..T: annuity payments, maintenance expenses and investment expenses (paras 32(a)(i)–(ii), 33). It includes the start-of-month payment of month t + 1. At t = 0 it includes the payment on the 1st of the month after V.

The fulfilment cash flows are those of B65: payments to the policyholder (B65(b)), payment administration and policy maintenance (B65(f), (h)), allocated directly attributable overheads (B65(l)) and investment-activity costs (B65(ka)(i)). Investment returns, reinsurance, costs not directly attributable and income tax are excluded (B66). There are no future premiums.

### In-force recursion

The `bel` recursion is in section "Variables", with inv_exp(t) = ie_m × bel(t). Algebraically, bel(t) = ann_out(t+1) + exp_out(t+1) + v(t+1) × (1 + ie_m) × bel(t+1).

### Per-life recursion

v_be(t) is the BEL per life in force at t. Under the best-estimate basis, bel(t) = pols_if(t) × v_be(t) at every t. This is checked on every run (I6). It holds because pols_if(t + 1) = pols_if(t) × (1 − q_mth(t + 1)). This is the policy-value recursion of the life-contingencies texts (Dickson, Hardy and Waters 2009, sections 7.3–7.4; Bowers et al. 1997, chapter 5 for the annuity values).

### Investment expenses

- **Inclusion.** Costs of managing the assets backing the BEL are fulfilment cash flows under B65(ka)(i). The investment activity enhances the insurance coverage. The annuity bought per unit of premium is priced on the expected return, and the annuitant receives it only by surviving to each payment date (IASB AP2A February 2020, paras 18–20; BC283I). Route B65(ka)(ii) is closed, because there is no investment-return service (B119B(a)). Inclusion is a judgement, so it sits behind `include_inv_exp`, default on.
- **Base.** The BEL only, not the RA or the CSM. Reason: the BEL is the cleanest proxy for the assets backing the contracts. Costs on surplus or shareholder assets are excluded: B65(ka)(i) admits costs only "to the extent" the activity enhances policyholder benefits, and other costs are not directly attributable (B66(d)).
- **Timing.** End of month t, on the closing BEL: inv_exp(t) = ie_m × bel(t). Reason: every edge in the {bel, inv_exp} component has offset 0 or +1, so it is a plain backward recursion. Charging on the opening BEL, ie_m × bel(t − 1), would create a same-period cycle with mixed offsets, which the checker rejects (D13).
- **Rate conversion.** ie_m = (1 + ie_rate)^(1/12) − 1. Reason: charging ie_m on each month's closing BEL divides the 12-month accumulation of the reserve by (1 + ie_rate). So ie_rate is read as an annual charge on the closing balance. A fee quoted as an annual percentage of average assets would instead be about ie_rate / 12 a month; the basis is a judgement (open question 3).
- **Not netted into the discount rate.** Including inv_exp gives the same BEL as discounting the other cash flows with v(t) × (1 + ie_m). It stays a cash flow because para 33(d) estimates cash flows separately from the time value, B74 requires rates consistent with the cash flows, and the para 120 curve must remain the liability curve. Netting would also remove expected investment expenses from insurance revenue (B124(a)).
- **Financial effects.** The investment-expense cash flows depend on the current curve. Their change from a change in rates is an effect of financial risk and goes to IFIE, not the CSM (B97(a)).

### Roll-forward and direct present value

Both are built-in invariants (I4 and I5 in section "Built-in invariants, checked on every run"). In the roll-forward the interest term f(t) × (bel(t − 1) − ann_out(t) − exp_out(t)) is the month's IFIE on the BEL at current rates in an expected run (para 87).

## Risk adjustment

### Method

Confidence-level technique on one latent factor.
- A standard normal factor Z drives systematic non-financial uncertainty: the level and trend of mortality and the level of expenses. Annual mortality is scaled by Λ_L(Z) = exp(−σ_L Z − σ_L² / 2) and maintenance expenses by Λ_E(Z) = exp(σ_E Z − σ_E² / 2).
- Both multipliers have mean 1. So the best-estimate tables are the mean of the model, and the BEL is the present value at multipliers 1. This approximates the para 33(a) mean up to second-order effects (gap in section "Known limitation"; open question 9).
- PV(Z), the expected present value of the fulfilment cash flows given Z, rises with Z, because lighter mortality and heavier expenses both raise it. For a monotone function, the α-quantile of PV(Z) is PV(z_α), with z_α = Φ⁻¹(α).
- So RA = PV(z_α) − BEL, and BEL + RA is exactly the α-quantile of this conditional expected PV. No conversion step is needed for the para 119 disclosure.
- The per-life recursions use expected decrements, so random variation in individual deaths and expenses is assumed diversified and is not in the RA (B88(a)). The para 47 test (G3) inherits this. Longevity and expense risk share one factor (comonotonic), so quantiles add exactly, and contract RAs are added with no diversification credit between contracts' systematic exposures or between the two risks.
- Load check σ_E ≤ z_α. Reason: the α-quantile of a mean-one lognormal, exp(σz − σ²/2), rises with σ only while σ < z. The check keeps the expense multiplier at or above 1, so the RA is never negative (I7). It also keeps a wider expense distribution from lowering the RA (B91(c)). The mortality multiplier needs no check: it falls with σ_L for any z_α > 0.

Reasons for this technique over the alternatives (IAA 2018 compares the main techniques; Boumezoued et al. 2020 show the conversion a cost-of-capital RA needs for para 119):
- Cost of capital needs a capital projection (regulatory capital is out of scope) and a separate conversion to a confidence level for para 119.
- A judgemental margin also needs an assumed distribution before a confidence level can be disclosed.
- This technique needs one extra backward recursion per policy and is additive per contract, which makes the para 47 contract-level test cheap.

### Formulas

```text
z_α = Φ⁻¹(α),   mort_mult = Λ_L(z_α),   exp_mult = Λ_E(z_α)
RA_i(t) = pols_if(t) × (v_st(t) − v_be(t))
RA_g(t) = Σ_i RA_i(t), summed in ascending policy_id order (D19)
```

The stressed basis (`q_ann_st`, `q_mth_st`, `s_pp_st`, `v_st`) is defined in section "Variables".

- The stress applies to annual mortality after improvement and before the monthly conversion. Age ω is forced back to q = 1, because the stress would otherwise let lives survive past the horizon.
- The investment-expense rate is not stressed. Reason: it is set by contract with the asset manager, so its uncertainty is small next to maintenance cost. The stressed recursion still charges ie_m on the stressed reserve.
- The per-life form conditions on best-estimate survival to t. So RA(t) is right at every t, not only at t = 0. At t = 0 it equals the direct present value of the stressed cash flows less the BEL, for any pols_if_init.
- The RA uses the same current curve as the BEL.

### Calibration

- The worked-example values are synthetic placeholders, not a calibration. They pass the load check (σ_E = 0.10 ≤ z_α ≈ 0.674).
- A real calibration would set σ_L from longevity level and trend uncertainty and σ_E from expense volatility, and document both (para 117(c)(ii)).
- The method behaves as B91 requires in two checkable ways. A wider distribution (larger σ_L, or larger σ_E within the load check) gives a larger RA (B91(c)). A shorter remaining duration gives a smaller RA, and the RA is zero once no life is in force (B91(b)).

### Disclosure

- Para 119: "The risk adjustment for non-financial risk corresponds to a confidence level of 75% for the expected present value of the future cash flows (annuity payments and expenses) under systematic longevity and expense risk, over the full run-off. Random variation in individual deaths and expenses is assumed diversified. No diversification benefit is taken between contracts' systematic exposures or between longevity and expense risk."

### Disaggregation (para 81)

- The RA change is disaggregated. Non-financial changes in the RA for future service, measured at locked-in rates, adjust the CSM (B96(d), B72(c)). The finance part (unwind at current rates and the effect of changes in the discount curve, the ILP and the CPI path; B97(a)(ii), B128(a)) goes to IFIE and never adjusts the CSM. The rest is released to the insurance service result.
- Reason: the RA here is a present value of cash-flow differences, so the split is mechanical, and it keeps financial effects out of the CSM as they are for the BEL.
- Before M12 the outputs carry RA(t) only. The release and finance split is presentation, and comes with revenue and IFIE (M12).

### Known limitation

The best-estimate tables are the mean of the factor model, but the BEL is not exactly the mean present value E[PV(Z)]. Two second-order effects remain. The present value is not linear in the mortality multiplier. And expense and survival move together under one factor, so the expected expense cash flows are not the product of expected expenses and expected survival. On example policy 1, E[PV(Z)] exceeds the BEL by about 0.08% of the BEL, or 5% of the RA (numerical integration over Z, q capped at 1); this is judged immaterial for v1. The confidence-level statement is unaffected. See open question 9.

## Aggregation stage: CSM for one group

### Level of aggregation (paras 14–24)

- One portfolio: immediate annuities in payment, covering all three classes. Reason: they share the same risks (longevity, expense) and are managed together (para 14).
- Group key: (`portfolio`, `cohort`, `profit_bucket`), declared on the model point.
- `cohort` is the calendar year of `issue_date`, so no group spans more than 12 months of issue (para 22).
- `profit_bucket` follows para 16(a)–(c). Whether a profitable contract has no significant possibility of becoming onerous is judgement (para 19), so the choice between `NO_SIGNIFICANT_RISK` and `OTHER` is not checked.
- One group may hold results from several product classes. The stage joins them on the group key.

Load-time and aggregation-time checks; each failure is an error naming the model point file and row:

| Check | Rule | Paragraph |
|---|---|---|
| G1 | Exactly one group per aggregation run in v1 | Spec |
| G2 | `cohort` = year(`issue_date`) for every contract | 22 |
| G3 | For a group new at V, contract i is onerous iff P_i − FCF_i(0) < 0, with P_i = pols_if_init_i × single_premium_i and FCF_i(0) = bel_i(0) + ra_i(0). An onerous contract outside an `ONEROUS` group, or a non-onerous contract inside one, is an error | 16(a), 47 |
| G5 | Every contract in the group is new at V: `issue_date` = V. An in-force group is rejected until M11 lifts the check | 25, 28 |
| G6 | The projection run has J = 1 | see section "Scenarios" |

G4 belongs to section "In-force groups (M9 to M11)", and M11 lifts G5.

### Inputs from the projection stage

For each policy i and t = 0..T: bel_i(t), ra_i(t), cu_i(t) (t ≥ 1), plus `single_premium`, `pols_if_init`, the group key and `issue_date`. For the run: df(t). Group totals sum over policies sequentially in ascending `policy_id`, across all classes (D19):

```text
BEL_g(t) = Σ_i bel_i(t)      RA_g(t) = Σ_i ra_i(t)      FCF_g(t) = BEL_g(t) + RA_g(t)      CU_g(t) = Σ_i cu_i(t)
```

### Initial recognition (paras 25, 38, 47)

The group is recognised at V, when the premiums are received and coverage starts.

```text
P_g    = Σ_i pols_if_init_i × single_premium_i
CSM(0) = max(0, P_g − FCF_g(0))
LC(0)  = max(0, FCF_g(0) − P_g)
```

- If LC(0) > 0, the loss goes to profit or loss, the CSM is zero and the loss component is set up (paras 47, 49).
- There are no acquisition cash flows in v1.

### Locked-in rates (B72(b))

- The locked-in curve P_L is the composed curve of the deterministic run at the group's initial recognition. It is one curve per group, with no scenario dimension. For a group new at V (G5), P_L = P of the same run; the re-basing for an in-force group is in section "In-force groups (M9 to M11)".
- Monthly accumulation: a_L(t) and j_L(t) = fwd_rate(P_L(t − 1), P_L(t)), as in section "Notation". Reason: the forward rates of the locked-in curve are the only choice under which accreted CSM and locked-in present values roll forward consistently. IFRS 17 does not prescribe this; it is an inference. EFRAG TEG (2017) discusses locked-in against current rates for the CSM.
- The CSM accretes at nominal locked-in rates for every class, including the CPI-linked one (B72(b)).

### Coverage units (B119)

```text
cu_i(t) = pols_if_i(t−1) × ann_pp_i(t)          (the amount payable in month t × the expected number entitled to it)
```

- Reason: the quantity of benefits is the amount the policyholder can validly claim, and the expected duration counts through the expected number of lives (TRG May 2018 summary paras 33–35; AP05 Example 12, paras B.44–B.46).
- The IFRS Interpretations Committee's final agenda decision on this fact pattern (2022) settles the choice between its Methods 1 and 2. A survival-only annuity whose policyholder can claim a periodic amount while alive fits it. Method 1 meets B119: the current period's quantity is the payment validly claimable in it, and future quantities are the present value, at the period end, of the payments expected to be validly claimable. Method 2, which also counts a present value in the current period, does not.
- **Discounted by default** (`discount_coverage_units = True`). Reason: it reproduces Method 1.
- Discount rates: the locked-in curve that accretes the CSM (B72(b)), with the payment timing of section "Accretion and release". Reason: the agenda decision does not name the rates, and locked-in rates keep the units free of changes in current rates.
- **Undiscounted, by switch.** `discount_coverage_units = False` weights every month equally. Reason it is kept: the decision says IFRS 17 does not prescribe a method. It did not consider an undiscounted one, but its two reasons for accepting Method 1 (units fall only where survival gives a valid claim; a period's quantity is the amount claimable in it) do not depend on discounting. The IFoA CSM Working Party (2019) notes that IFRS 17 neither requires nor forbids discounting. Used for comparison and for the allocation-frequency property test.
- Expenses are excluded. Coverage units measure benefits, not costs.
- CPI variant: the expected indexed amount on the run's CPI path.

### Accretion and release

Monthly steps, t = 1..T, written in evaluation order:

```text
Acc(t)      = j_L(t) × CSM(t−1)                                           interest accretion, para 44(b)
CSM_pre(t)  = CSM(t−1) + Acc(t)
N(t)        = Σ_{m > t} CU_g(m) × δ(t, m)                                 units still to be provided, valued at the end of month t
              δ(t, m) = P_L(m − 1) / P_L(t)     (discounted, default; the payment of month m is made at m − 1)
              δ(t, m) = 1                        (undiscounted)
              computed backward: N(T) = 0;  N(t) = CU_g(t+1) + d(t+1) × N(t+1),  d = d_L or 1
U(t)        = CU_g(t) + N(t)                                              current month's units are undiscounted
R(t)        = CSM_pre(t) × (CU_g(t) / U(t))                               release, para 44(e), B119
              R(t) = CSM_pre(t) if U(t) = 0
CSM(t)      = CSM_pre(t) − R(t)
```

- The release applies to the CSM after accretion and after any adjustment, in the order of para 44(a)–(e) and B119(b). For a group new at V (G5) there is no adjustment.
- B119(b) allocates the CSM equally to each unit. So R(t) / CU_g(t) = CSM(t) / N(t) whenever both denominators are positive (I12).
- v1 allocates monthly; the step is not a setting, and a reporting period's figures are the sums of its monthly rows. With undiscounted units the period-end CSM equals that of one step per period, because the ratio telescopes; with discounted units it does not exactly, because a monthly step discounts within the period (open question 4). Interim-period policy (B137) is out of scope.

### Roll-forward table

CSM, for month t:

| Line | Amount | Paragraph |
|---|---|---|
| Opening CSM | CSM(t − 1) | — |
| New contracts | CSM(0), at recognition only | 44(a), 38 |
| Interest accretion | Acc(t) | 44(b), B72(b) |
| Changes for future service | 0 for a group new at V; −ΔFCF^L at the remeasurement of an in-force group (M11) | 44(c), B96 |
| Currency differences | 0 (one currency) | 44(d) |
| Release | −R(t) | 44(e), B119 |
| Closing CSM | CSM(t) | — |

Loss component, for month t:

| Line | Amount | Paragraph |
|---|---|---|
| Opening LC | LC(t − 1) | — |
| Loss at recognition | LC(0), at recognition only | 47, 49 |
| Further losses | 0 for a group new at V; from unfavourable changes at a remeasurement (M11) | 48 |
| Reversals | 0 for a group new at V; from favourable changes at a remeasurement (M11) | 50(b) |
| Systematic allocation | −LCA(t) | 50(a), 51, 52 |
| Closing LC | LC(t) | — |

LRC(t) = BEL_g(t) + RA_g(t) + CSM(t). The LC is a part of BEL_g + RA_g, not an addition to it.

### Onerous groups and the loss component

The systematic allocation follows the ratio of IFRS 17 Illustrative Example 8. Reason: it is the IASB's own illustration, and the IFoA CSM Working Party (2019, session D3) finds it gives a near-linear LC run-off and a stable revenue profile, at some operational cost against the alternatives; that cost is low here because F is already computed.

```text
F(t)    = PV_out_g(t) + RA_g(t)                      present value of future cash outflows plus RA
                                                     (= FCF_g(t) for this product: no future inflows)
SAR(t)  = LC(t−1) / F(t−1)                           share of the opening outflows plus RA that is loss component
LCA(t)  = SAR(t) × [F(t−1) − F(t)]                   share of the para 51 items in month t
LC(t)   = LC(t−1) − LCA(t)
```

Evaluation order: LC(t) = LC(t − 1) × (F(t) / F(t − 1)), and LCA(t) = LC(t − 1) − LC(t); LC(t) = 0 if F(t − 1) = 0. This equals the SAR form in exact arithmetic and gives LC exactly 0 when F reaches 0.

- Example 8 computes the ratio from the opening loss component over the opening PV of outflows plus RA: on initial recognition in Year 1, from opening balances in Year 2. This note does the same each month, as its systematic basis under para 50(a). With no remeasurement, which is every run of a group new at V, the ratio is constant (I13).
- In an expected run the para 51 items of month t (claims and expenses released, RA released, less IFIE) sum to F(t − 1) − F(t). So LC(t) / F(t) stays constant, and the LC reaches zero when F does (para 52).
- F uses current rates, so the IFIE allocation follows para 51(c).
- The CSM of an onerous group stays zero in a run with no remeasurement. Coverage units are still computed.

### Scenarios

- Until M12, the aggregation stage reads a deterministic run (J = 1) and rejects J > 1 (G6).
- Reason: para 33(a) needs a probability-weighted mean, which is a scenario statistic, and the Roadmap places scenario statistics in M12.
- For v1 products the cash flows are linear in the index and have no options (B39, B59). So a deterministic run on the market-implied curve and the market-implied CPI path gives that mean, up to a timing convexity adjustment. The payments use a lagged index paid at the start of a later month, while breakeven forwards are defined for their own dates. v1 ignores this adjustment as immaterial, and the manifest records the assumption.
- M12: FCF_g = Σ_s w_s × FCF_g^(s) with w_s = 1/J, summed in ascending s. There is still one CSM and one locked-in curve per group. Per-scenario CSMs would be diagnostics, not IFRS 17 measures.

### Aggregation sketch (provisional syntax)

The Spec does not yet define how the aggregation stage is written. This sketch shows the schedule: one backward component (`cu_next`), then forward components (`csm`, `lc`), whose inits read `csm_0` and `lc_0`, computed from P_g and the BEL and RA at t = 0. Inside `where()`, every division is by a guarded value, because both branches are evaluated.

```python
class CsmGroup(Aggregation):
    discount_coverage_units: Static[bool] = True   # Method 1 of the 2022 agenda decision

    @gvar
    def cu_g(self, t) -> CoverageUnits:            # sequential, ascending policy_id (D19)
        return self.policies.sum("cu", t)

    @gvar
    def fcf_g(self, t) -> Money:                   # PV of future outflows + RA; no inflows
        return self.policies.sum("bel", t) + self.policies.sum("ra", t)

    @gvar
    def p_g(self, t) -> Money:                     # scalar P_g, sequential in ascending policy_id
        return self.policies.sum_mp(lambda mp: mp.pols_if_init * mp.single_premium)

    @gvar
    def csm_0(self, t) -> Money:                   # scalar max(0, P_g − FCF_g(0)), para 38
        gap = self.p_g(t) - self.fcf_g.at(0)
        return where(gap > 0, gap, 0.0)

    @gvar
    def lc_0(self, t) -> Money:                    # scalar max(0, FCF_g(0) − P_g), para 47
        gap = self.fcf_g.at(0) - self.p_g(t)
        return where(gap > 0, gap, 0.0)

    @gvar
    def j_l(self, t) -> Rate[Monthly]:             # locked-in one-month rate
        return fwd_rate(self.locked.df(t-1), self.locked.df(t))

    @gvar
    def d_l(self, t) -> Factor:                    # locked-in one-month discount factor
        return self.locked.df(t) / self.locked.df(t-1)

    @gvar(terminal=0)
    def cu_next(self, t) -> CoverageUnits:         # N(t): units of months after t, valued at t
        if self.discount_coverage_units:           # build-time branch (D8)
            return self.cu_g(t+1) + self.d_l(t+1) * self.cu_next(t+1)
        return self.cu_g(t+1) + self.cu_next(t+1)

    @gvar
    def acc(self, t) -> Money:                     # Acc(t), para 44(b)
        return self.j_l(t) * self.csm(t-1)

    @gvar
    def csm_pre(self, t) -> Money:
        return self.csm(t-1) + self.acc(t)

    @gvar
    def units(self, t) -> CoverageUnits:           # U(t)
        return self.cu_g(t) + self.cu_next(t)

    @gvar
    def release(self, t) -> Money:                 # R(t), para 44(e), B119
        u = self.units(t)
        share = self.cu_g(t) / where(u > 0, u, 1.0)
        return where(u > 0, self.csm_pre(t) * share, self.csm_pre(t))

    @gvar(init=lambda g: g.csm_0(0))
    def csm(self, t) -> Money:
        return self.csm_pre(t) - self.release(t)

    @gvar(init=lambda g: g.lc_0(0))
    def lc(self, t) -> Money:
        f0 = self.fcf_g(t-1)
        ratio = self.fcf_g(t) / where(f0 > 0, f0, 1.0)
        return where(f0 > 0, self.lc(t-1) * ratio, 0.0)

    @gvar
    def lc_alloc(self, t) -> Money:                # LCA(t), para 50(a)
        return self.lc(t-1) - self.lc(t)
```

### In-force groups (M9 to M11)

The one-group stage of M7 rejects in-force groups (G5); M11 lifts the check. This section fixes their design so that the code of M2 and M7 does not close it off. Before the check is lifted, a worked example of an in-force group with a non-zero ΔFCF^L, over a period in which actual CPI differs from expected, must reconcile (M10, in the D22 pattern). The cross-run reads and the group input contract are M9.

**Group inputs.**

| Input | Fields | Rule |
|---|---|---|
| `groups` | `portfolio`, `cohort`, `profit_bucket`, `recognition_date`, `prev_valuation_date`, `csm_prev` (Money ≥ 0), `lc_prev` (Money ≥ 0), `cu_act` (CoverageUnits ≥ 0), `prev_run_id` | Dates are month-ends. At most one of `csm_prev`, `lc_prev` is positive. `cu_act` is the payments actually made to survivors in (V_prev, V] |
| Locked-in basis | The discount file, ILP table and locked-in CPI file, identified by hash | Fixed for the life of the group (B72(b)). The discount file covers t ∈ [0, τ0 + T] |
| Previous run, by `prev_run_id` | Hashes of the previous non-financial tables; each policy's expected pols_if at V rolled forward from V_prev on those tables; the previous group membership | Read from the previous run's outputs |

- τ0 = mi(V) − mi(`recognition_date`), an integer because both dates are month-ends. The loader re-bases the locked-in curve to V: P_L(t) = P_rec(τ0 + t) / P_rec(τ0), where P_rec is the composed curve at recognition.
- Every contract's `issue_date` equals `recognition_date`. Reason: contracts issued on different dates would need B73 weighted-average rates, which no input supplies.
- G3 is not re-run: an in-force contract's `profit_bucket` must match the stored group input.
- G4: a `policy_id`'s group key does not change between runs (para 24).

**Locked-in CPI path.** For u ≤ 0, I_L(u) = I_act(u), the history as known at V: published values replace earlier estimates, and months not yet published keep their estimate. For u > 0, I_L(u) = I_act(0) × I_rec(τ0 + u) / I_rec(τ0), where I_rec is the CPI path of the recognition run, indexed from the recognition month. Reason: actual CPI to V is a fact for both runs, and expected inflation after V stays at its locked-in level, so changes in CPI expectations stay out of the CSM (B97(a)). Built outside the engine and supplied with its hash, the file keeps the engine reading only scenarios (D4; open question 7).

**Measurement of ΔFCF^L at V.**

```text
ΔFCF^L = [BEL^L_g(0) + RA^L_g(0)]_new − [BEL^L_g(0) + RA^L_g(0)]_old
```

- Both runs are ordinary projections on the same financial inputs: the locked-in curve P_L, the locked-in CPI path I_L and the actual `instalment_pm` at V, after every actual escalation to V.
- They differ only in the non-financial inputs: the tables `mort_base`, `mort_cif`, `expense`, `inv_expense` and `ra_param`, and pols_if at V. "New" uses current tables and the actual in-force. "Old" uses the previous tables and the expected in-force from the previous run.
- Both runs use expense levels in V prices. Before the old run the loader rebases the previous table, priced at V_prev: maint_pm_old = maint_pm_prev × I_act(0) / I_act(−τ), τ = mi(V) − mi(V_prev). It checks that both runs carry price base V.
- Reason: fixed financial inputs and rebased expenses keep rate changes and CPI experience out of the CSM; both are financial risk (B97(a), B128(a); TRG April 2019 submission S122, EY 2019). Actual against expected inflation over the period reaches IFIE through the current-basis run (current curve and CPI).
- The difference between measuring the same changes at current and at locked-in rates goes to IFIE (para 87). The OCI option stays off (para 88(a)).

**Closing the period (V_prev, V], in the order of para 44 and B119(b).**
1. Accrete: CSM_acc = csm_prev × P_rec(τ0 − τ) / P_rec(τ0).
2. Allocate the LC for the period's para 51 items (para 50(a)), giving LC_alloc.
3. Apply ΔFCF^L to (CSM_acc, LC_alloc) with the rules below, giving (CSM_adj, LC_adj). The para 48 onerous test is therefore run on the CSM before the period's release.
4. Release: R_per = CSM_adj × (cu_act / (cu_act + N(0))). This is Method 1 at the period level: the period's units are the payments actually claimable in it, and future units are valued at V.
5. CSM(0) = CSM_adj − R_per and LC(0) = LC_adj. Months 1..T then run as for a new group.

Coverage units of an in-force group, for N(0) in step 4 and in months 1..T, are the current-basis run's expected payments, discounted on P_L. Reason: Method 1 counts the payments expected to be claimable, a current estimate. Only the CPI-linked class is affected.

Remeasurement rules, with ΔFCF^L the increase in locked-in fulfilment cash flows from changes for future service:

```text
If LC > 0 and ΔFCF^L ≥ 0:   LC  = LC + ΔFCF^L                                        para 48(a)
If LC > 0 and ΔFCF^L < 0:   rev = min(LC, −ΔFCF^L);  LC = LC − rev;
                            CSM = CSM + (−ΔFCF^L − rev)                              paras 50(b), 44(c)(ii)
If LC = 0:                  CSM = CSM − ΔFCF^L;  if CSM < 0:
                            LC = −CSM;  CSM = 0                                      paras 44(c)(i), 48(a)
```

Each increase in LC is a loss in profit or loss; each reduction under para 50(b) is a reversal of losses.

**Which changes adjust the CSM.**

| Change | Treatment | Paragraphs |
|---|---|---|
| Mortality, improvement, expense level, investment-expense rate | Adjusts the CSM, measured at locked-in rates | 44(c), B96(b), B72(c) |
| RA parameters (α, σ_L, σ_E) | Non-financial part adjusts the CSM at locked-in rates | B96(d), B72(c) |
| Effect of actual deaths on future payments (closing in-force differs from expected) | Adjusts the CSM. This is an inference, see open question 6 | B96(b) |
| Payments and expenses of the current period differing from expected | Insurance service result, not the CSM | B97(c) |
| Discount curve or ILP | IFIE, not the CSM | B97(a), 87 |
| CPI path, for CPI-linked payments and expense inflation | IFIE, not the CSM | B97(a), B128(a); TRG April 2019, S122 |
| Fixed escalation | Contractual; no estimate changes | — |

## Outputs

**Per policy, scenario and month** (Parquet): `pols_if`, `deaths`, `ann_pp`, `ann_out`, `exp_out`, `inv_exp`, `bel`, `ra`, `cu`. On request for tracing: `q_ann`, `q_mth`, `q_mth_st`, `v_be`, `v_st`, `esc_factor`. Per-life values beyond the policy's run-off are written but flagged: `ann_pp` at t > T_i, `v_be` and `v_st` at t ≥ T_i.

**Per scenario and month:** `df`, `v`, and the annualised spot and forward rates of the composed curve: the para 120 curve in a deterministic run (J = 1), diagnostics only when `rf_df` is a pathwise deflator.

**Per policy at t = 0:** BEL, RA, the present value of annuity payments and of expenses, and the para 47 test result.

**Per group and month:** CU_g, N, BEL_g, RA_g, FCF_g, opening CSM, accretion, changes for future service (0 for a group new at V), release, closing CSM, LC and its allocation LCA.

**Per group:** P_g, CSM(0) or LC(0), the locked-in curve hash, and the disclosures: confidence level α (para 119), the composed curve (para 120), and the approaches to discount rates and the RA (para 117(c)).

**Manifest** (D21): as in the Spec, plus `include_inv_exp`, `cpi_lag_m`, `discount_coverage_units`, the RA parameters, the hashes of the scenario files and the locked-in basis, and two notes: the monthly CSM allocation step, and the ignored timing convexity between index and payment dates.

**Labelling.** In a stochastic run, bel_s(t) and ra_s(t) in scenario s are pathwise values at every t, including t = 0. Only the probability-weighted mean over scenarios at t = 0 is a para 33(a) estimate. Outputs are labelled accordingly.

**Outputs added in M12:**
- Insurance revenue in an expected run (B121, B124): expected annuity payments + expected maintenance and investment expenses + RA release + CSM release, less amounts allocated to the LC.
- IFIE (para 87): BEL unwind at current rates + RA finance part + CSM accretion at locked-in rates + effects of curve and CPI changes + the current-versus-locked-in difference on non-financial changes.

## Invariants and tests

### Built-in invariants, checked on every run

A tolerance check passes when |lhs − rhs| ≤ ε × S, with ε = 1e-12 within a backend and S the scale in the table. "Exact" means no tolerance. Between backends, outputs agree to 1e-9 relative (D19). Bit-for-bit checks need the same backend and the same batch composition (D19); the same policy in a different batch, or a run with a different T, agrees within 1e-12 relative (D5).

| # | Invariant | Scale S | Source |
|---|---|---|---|
| I1 | pols_if(t) ≥ 0 and pols_if(t) ≤ pols_if(t − 1) | exact | Spec |
| I2 | q_ann, q_mth, q_ann_st, q_mth_st in [0, 1]; deaths(t) ≤ pols_if(t − 1) | exact | Spec |
| I3 | bel(T) = v_be(T) = v_st(T) = 0; for every policy, q_ann(T_i) = 1 and pols_if(T_i) = 0. The second part stops the first from being true by construction only | exact | Spec, extended |
| I4 | Roll-forward, t = 1..T: bel(t) = [bel(t − 1) − ann_out(t) − exp_out(t)] / v(t) − inv_exp(t) | (\|bel(t − 1)\| + \|ann_out(t)\| + \|exp_out(t)\|) / v(t) + \|inv_exp(t)\| | Spec |
| I5 | Direct PV: bel(0) = Σ_t [P(t − 1)(ann_out(t) + exp_out(t)) + P(t) inv_exp(t)], summed in ascending t | max(\|lhs\|, \|rhs\|) | Spec |
| I6 | bel(t) = pols_if(t) × v_be(t), t = 0..T | max(\|lhs\|, \|rhs\|, bel_i(0)) | product |
| I7 | ra(t) ≥ 0; ra(t) = 0 when σ_L = σ_E = 0 | exact | product |
| I8 | Level: ann_pp(t) = instalment_pm. Fixed: ann_pp(t) = instalment_pm × (1 + g)^K(t), K(t) = number of anniversaries in months 1..t | Level exact; fixed \|rhs\| | product |
| I9 | CPI, telescoping: ann_pp(t) = instalment_pm × I(n_K − L) / I(n_0 − L), with n_K the latest anniversary month in 1..t (or n_0 if none) and n_0 = m_c if m_c = 1, else the latest month m_c + 12k ≤ 0 | \|rhs\| | product |
| I10 | cu(t) = ann_out(t) ≥ 0 | exact | product |
| I11 | CSM(t) = CSM(t − 1) + Acc(t) − R(t); CSM(t) ≥ 0; CSM(t) = 0 after the last month with CU_g > 0 | CSM(t − 1) + \|Acc(t)\| + R(t); the last two exact | aggregation |
| I12 | Equal allocation: R(t) / CU_g(t) = CSM(t) / N(t), where CU_g(t) > 0 and N(t) > 0. Step: R(t+1) / CU_g(t+1) = [R(t) / CU_g(t)] × a_L(t+1) × [CU_g(t+1) + d(t+1) × N(t+1)] / U(t+1), where CU_g(t) > 0 and CU_g(t+1) > 0 | max(\|lhs\|, \|rhs\|) | aggregation |
| I13 | LC(t) / F(t) = LC(0) / F(0) where F(t) > 0; LC(t) = 0 where F(t) = 0 | max(\|lhs\|, \|rhs\|); the second exact | aggregation |
| I14 | G3 holds for every contract of a group new at V | exact | aggregation |

- I4's scale is the size of the terms being combined. At t = T_i, bel(T_i) is exactly 0 while the right side keeps the rounding of fl(ann_out + exp_out), so a pure relative check would fail spuriously there.
- I6's floor bel_i(0) keeps the check meaningful where both sides are tiny near T_i.
- In I12, d = d_L for discounted units and 1 for undiscounted; with d = 1 the step factor reduces to a_L(t + 1).

### Property-based tests (Hypothesis, D24)

Random model points within the schema: sex, ages 50–110, `pols_if_init` in (0, 5e5], instalments 1 to 1e5, escalation rates 0–0.10, start dates up to 30 years before V or on V + 1 day, valuation month-ends from 2025 to 2040, and random flat or sloped curves and CPI paths within the file ranges. Each test also checks I1–I14.

| Property | Expected result |
|---|---|
| Doubling `pols_if_init` | `bel`, `ra`, `cu`, `ann_out`, `exp_out`, `inv_exp` double bit-for-bit (multiplying by 2 is exact) |
| Scaling `instalment_pm` by k in [0.5, 2] with `maint_pm` = 0 | `bel` and `ra` scale by k within 1e-12 |
| Older annuitant, otherwise identical level policy, no expenses | lower v_be(0) |
| Female against male, same age | higher v_be(0) |
| Larger σ_L, larger σ_E (within σ_E ≤ z_α) or larger α | RA not smaller at any t. At t = 0 strictly larger when the stressed driver is non-zero: always for σ_L; for σ_E when `maint_pm` > 0; for α when σ_L > 0, or σ_E > 0 and `maint_pm` > 0 |
| Fixed escalation with `esc_rate` = 0 | identical to level, bit-for-bit |
| CPI path I(u) = I0 × (1 + g)^(u/12) for all u | CPI-linked equals fixed escalation at g within 1e-12 |
| `include_inv_exp` = False on the curve P(t) × (1 + ie_m)^t, with curves and ie_rate drawn so that rf_df(t) × (1 + ie_m)^t ≤ 2 for t ≤ T (the file range) | same bel(0) as `include_inv_exp` = True on P(t), within 1e-12 |
| Rows of the model point files permuted | group results bit-identical: the loader sorts by `policy_id`, so the batches are the same (D19) |
| 12 monthly CSM steps against one 12-month step, `discount_coverage_units` = False | same closing CSM within 1e-12. The test harness computes the single step as CSM(0) × (P_L(0) / P_L(12)) × (N(12) / N(0)) |
| Same policy in two batches of different composition | same results within 1e-12 (D5) |
| Same policy run with T and with T' > T | in-force-weighted values at every t, and per-life values for t < T_i, agree within 1e-12 relative (D5) |

### Table-generator tests

These test the scripts that build the synthetic tables, not the engine.
- Female row x equals male row x − 4 exactly, for ages 24–119.
- cif follows its recursion in ascending year order; cif(120, y) = 1 for every y.
- The improvement hand checks in section "Mortality" hold.

### Differential tests

Every backend against the NumPy reference, 1e-9 relative (D16, D19).

## Checker rules exercised

Each rule in the Spec's "Checker rules", except dimensions, has a natural deliberate error in this product. Each build error must fail the build and point to the model source line, never to generated code. The unused-variable case must raise a warning and still build. Data-contract errors fail at load and name the file and row.

| Rule | How the annuity exercises it | Deliberate error that must fail |
|---|---|---|
| Units: Money adds only to Money | `bel` sums money cash flows | `self.bel(t+1) + self.pols_if(t)` |
| Units: per-policy against in-force | `ann_out`, `exp_out`, `ra` | `self.pols_if(t-1) * self.ann_out(t)` |
| Units: explicit rate conversion | `ie_m` uses `to_monthly` | `ie_m` returning `ie_rate` (Rate[Annual, Effective] declared as Rate[Monthly]) |
| Units: compounding basis (proposed) | `df` uses `cc_discount` on the continuous ILP | `to_monthly(self.tables.ilp.ilp)` |
| Units: probability period (proposed) | `q_mth` uses `to_monthly_cf` | `deaths = pols_if(t-1) * q_ann(t)`; `p = 1 - self.q_ann(t+1)` in `v_be` |
| Units: coverage units (proposed) | `cu` uses `as_cu` | `self.bel(t) + self.cu(t)` |
| Units: index ratios (proposed) | CPI `esc_factor` | `self.ann_pp(t-1) * self.scen.cpi(t-3)` |
| Dimensions | (scenario, time) inputs broadcast to policy × scenario × time | none: every broadcast the annuity uses is legal, so the test comes from the term model |
| Time offsets constant or `lag()` | t − 1, t + 1, t − L, t − L − 12 with L static | a per-policy lag written as `self.scen.cpi(t - self.mp.lag_m)`, with `lag_m: int32` added to the test's model point schema |
| Recursion needs `init` or `terminal` | `pols_if`, `ann_pp`; `bel`, `v_be`, `v_st` | `bel` without `terminal=0` |
| Same-period cycle | none in the model | `deaths` reading `self.pols_if(t)` instead of `self.pols_if(t-1)`: the cycle pols_if(t) → deaths(t) → pols_if(t) sums to 0, every read stays in its domain, and the offsets {0, 0, −1} are not mixed |
| Mixed forward/backward component | none in the model | `deaths` reading `self.pols_if(t+1)` instead of `t-1`: offsets −1 and +1 in {pols_if, deaths}. No elementary cycle sums to 0, but the closed walk pols_if(t) → pols_if(t − 1) → deaths(t − 1) → pols_if(t) does, and at t = T the read of pols_if(T + 1) is out of domain. The test asserts that the build fails at this line and accepts a mixed-direction, same-period-cycle or out-of-domain error |
| Reads across policies in projection | group sums only in aggregation | `self.policies.sum("cu", t)` inside `LevelAnnuity` |
| Enum coverage | `mort_base` keyed on `Sex`; `profit_bucket` in aggregation | `mort_base` without rows for `F`. The annuity has no in-source enum selection, so this row tests the load-time key-coverage contract on the `mort_base` lookup |
| Unused variable (warning) | none in the model | `anniv` defined on `LevelAnnuity` |
| Data contracts | ranges, key coverage, scenario completeness, CPI history, table closure | V not a month-end; `annuity_start_date` not on the 1st; CPI file starting at u = −13; `mort_cif` ending before Y(T); cif(120, y) ≠ 1; σ_E > z_α; an onerous contract in an `OTHER` group (G3); a contract with `issue_date` before V in an aggregation run (G5) |

## Worked example

Fully specified so that two people implementing from this note alone agree within the tolerance in section "Quantities".

All three examples are the reference case (D2). Examples 1 and 2 use the level class only, so they can reconcile before the escalating classes are built. Example 3 adds the fixed and CPI-linked variants.

### Run settings

| Setting | Value |
|---|---|
| Valuation date V | 2025-12-31 |
| m_V | 24311 (= 12 × 2025 + 11) |
| Scenarios | J = 1, scenario id 0 |
| `include_inv_exp` | True |
| `cpi_lag_m` | 3 |
| `discount_coverage_units` | True; example 1 also reports CSM(12) with False |
| T | 655 (= 12 × 120 + 1 − 786) for every example run, since each contains policy 1 or 4. The horizon test in example 3 is a separate run |
| Arithmetic | IEEE float64, no fast-math, no rounding |

### Model points

All are new business written on V, with the first payment on 1 January 2026.

| Field | Policy 1 | Policy 2 | Policy 3 | Policy 4 |
|---|---|---|---|---|
| Class | `LevelAnnuity` | `FixedEscalationAnnuity` | `CPILinkedAnnuity` | `LevelAnnuity` |
| `policy_id` | 1 | 2 | 3 | 4 |
| `portfolio` | IA | IA | IA | IA |
| `cohort` | 2025 | 2025 | 2025 | 2025 |
| `profit_bucket` | OTHER | OTHER | OTHER | ONEROUS |
| `sex` | M | F | M | M |
| `date_of_birth` | 1960-06-15 | 1957-10-10 | 1955-02-20 | 1960-06-15 |
| `issue_date` | 2025-12-31 | 2025-12-31 | 2025-12-31 | 2025-12-31 |
| `annuity_start_date` | 2026-01-01 | 2026-01-01 | 2026-01-01 | 2026-01-01 |
| `pols_if_init` | 1.0 | 1.0 | 1.0 | 1.0 |
| `instalment_pm` | 1,000.00 | 800.00 | 1,200.00 | 1,000.00 |
| `single_premium` | 190,000.00 | 220,000.00 | 250,000.00 | 175,000.00 |
| `esc_rate` | — | 0.03 | — | — |

Policy 4 is policy 1 with its own `policy_id`, a lower premium and the `ONEROUS` bucket.

Derived values (hand checks, not results):

| Value | Policy 1 | Policy 2 | Policy 3 | Policy 4 |
|---|---|---|---|---|
| a0 (months) | 786 | 818 | 850 | 786 |
| Age at V | 65 y 6 m | 68 y 2 m | 70 y 10 m | 65 y 6 m |
| x(1) | 65 | 68 | 70 | 65 |
| First month with x + 1 | t = 7 (July 2026) | t = 11 (November 2026) | t = 3 (March 2026) | t = 7 |
| m_c | 1 | 1 | 1 | 1 |
| Anniversary months | — | 13, 25, 37, … | 13, 25, 37, … | — |
| T_i | 655 | 623 | 591 | 655 |

### Mortality

Base table, the Standard Ultimate Survival Model of Dickson, Hardy and Waters (Makeham; SOA 2022; Lim, actuarialmath guide), generated in code:

```text
A_mk = 0.00022,  B_mk = 2.7e-6,  c_mk = 1.124,  ln = natural logarithm
q_M(x) = 1 − exp(−A_mk − B_mk × c_mk^x × (c_mk − 1) / ln(c_mk))           x = 20..119
q_F(x) = 1 − exp(−A_mk − B_mk × c_mk^(x−4) × (c_mk − 1) / ln(c_mk))       x = 20..119   (four-year setback)
q_M(120) = q_F(120) = 1
base_year = 2025
```

- Check values against the published SOA table (6 decimal places): q_M(65) = 0.005915, q_M(100) = 0.289584.
- The female setback is illustrative. It is close to a fit of US 2022 period life tables over ages 60–90 (about 4.1 years; Arias, Xu and Kochanek 2025), too light for women above about 85.

Improvement rates, the convergence form of SOA MP-2014 Formula B.1 with zero initial slope, Halley's own parameters:

```text
h(y)   = min(1, (y − 2026) / 20)
w(h)   = 1 − 3h² + 2h³
g(x)   = 1                     for x ≤ 85
       = (110 − x) / 25        for 85 < x < 110
       = 0                     for x ≥ 110
r(x,y) = g(x) × (0.010 + 0.010 × w(h(y)))                   x = 20..120, y = 2026..2150
cif(x, 2025) = 1;   cif(x, y) = cif(x, y − 1) × (1 − r(x, y))   y = 2026..2150
```

- Hand checks: r(65, 2026) = 0.02; r(65, 2036) = 0.015; r(65, 2046) = 0.01; r(95, 2026) = 0.012; r(110, y) = 0.
- r ≥ 0 and r(120, y) = 0, so cif ≤ 1 and cif(120, y) = 1: the table passes the load checks.
- The last calendar year used is Y(655) = (24311 + 655) // 12 = 2080, inside the table.

Monthly conversion: the guarded `to_monthly_cf` of section "Compounding and unit conversions".

### Expenses and investment expenses

| Assumption | Value |
|---|---|
| `maint_pm` | 5.00 per policy per month, in December 2025 prices (u = 0) |
| Expense inflation | exp_pp(t) = 5.00 × (I(t − 1) / I(0)) |
| `ie_rate` | 0.0015 annual effective; ie_m = 1.0015^(1/12) − 1 |
| `ilp` | 0.0050, continuously compounded |

### Discount curve

Risk-free curve, continuously compounded zero rates, τ = t / 12:

```text
z_rf(τ)  = 0.040 − 0.015 × exp(−τ / 10)
rf_df(t) = exp(−z_rf(t/12) × t/12)                           t = 0..655    (rf_df(0) = 1)
P(t)     = rf_df(t) × exp(−0.0050 × t/12)                    composed current curve
P_L(t)   = P(t)                                              locked-in curve (recognition at V)
```

The file holds rf_df(t) for every month t = 0..655, so there is no interpolation.

### CPI path

Base value I(0) = 100 for December 2025:

```text
I(u) = 100 × 1.04^(u/12)      for u ≤ 0       (history: 4% a year)
I(u) = 100 × 1.025^(u/12)     for u ≥ 0       (projection: 2.5% a year)
```

- The file holds u = −24..655. The model needs u ≥ −14.
- Policy 3's first escalation is in month 13 (1 January 2027): E(13) = I(10) / I(−2), which straddles history and projection. A lag error therefore changes the result.

### Risk adjustment parameters

| Parameter | Value |
|---|---|
| α | 0.75 |
| z_α | 0.6744897501960817 (Φ⁻¹(0.75); implementations use this literal) |
| σ_L | 0.10 |
| σ_E | 0.10 |

### Quantities

Results are reported per example: examples 1, 2 and 3 each form their own group, and a policy's values are reported in each example that contains it. Policies 1–4 never form one group: policy 4's bucket differs, and a run holds one group (G1). Two implementations agree when every reported value matches within 1e-9 relative, and zero values within 1e-9 × P_g absolute, so floating-point evaluation order, in the model or in generating the input files, does not matter here. D19's bit-identity is per backend and per batch composition, not across implementations; the bit-for-bit checks below apply within one backend.

Per policy i (all with pols_if_init = 1):

| Symbol | Definition |
|---|---|
| BEL_i(0) | bel(0) |
| RA_i(0) | ra(0) = pols_if(0) × (v_st(0) − v_be(0)) |
| A_i(1) | ann_pp(1), the payment on 1 January 2026 |
| PVA_i | Σ_{t=1..T} P(t − 1) × ann_out(t), summed in ascending t |
| PVE_i | Σ_{t=1..T} [P(t − 1) × exp_out(t) + P(t) × inv_exp(t)], summed in ascending t; maintenance and investment expenses |
| A_i(13) | ann_pp(13), the first anniversary payment (supplementary check) |

Check: PVA_i + PVE_i = BEL_i(0) within 1e-12 relative.

Group (sums sequential in ascending policy_id):

| Symbol | Definition |
|---|---|
| P_g | Σ_i pols_if_init_i × single_premium_i |
| BEL_g(0), RA_g(0), FCF_g(0) | Σ_i BEL_i(0), Σ_i RA_i(0), BEL_g(0) + RA_g(0) |
| CSM(0) | max(0, P_g − FCF_g(0)) |
| LC(0) | max(0, FCF_g(0) − P_g) |
| CU_g(t) | Σ_i cu_i(t), t = 1..12 |
| R(t) | Release in month t, t = 1..12, as in section "Accretion and release" |
| CSM(12) | Closing CSM at the end of month 12 |

### Results

The values below come from two scripts written independently from this note alone: implementation A (backward recursion) and implementation B (direct discounting). They agree on all 110 reported values, at most 9.3e-14 relative apart. A third reimplementation, by a reviewer, reproduces the headline figures. They compare only the quantities listed here, not every month.

They are not yet the D22 reference. That still needs the spreadsheet built from this note and the reviewer's check of it against the note. The spreadsheet and a script are then compared month by month on:
- pols_if, bel and ra for t = 0..T, and cu for t = 1..T;
- ann_pp for t = 1..T_i, and v_be and v_st for t = 0..T_i − 1.

If the spreadsheet disagrees with a value below, the note is ambiguous or wrong at that point. Fix the note, not the number.

### Example 1: level policy, profitable group

Policy 1 alone, as group (`IA`, 2025, `OTHER`). New at V, so τ0 = 0, no group inputs, and P_L = P. The premium was set about 4–5% above a rough estimate of BEL + RA, so the contract passes the para 47 test and the group is profitable.

Per policy:

| Quantity | Policy 1 |
|---|---|
| BEL_1(0) | 179,192.1480 |
| RA_1(0) | 2,648.249534 |
| A_1(1) | 1,000.000000 |
| PVA_1 | 175,222.1063 |
| PVE_1 | 3,970.041706 |
| A_1(13) (supplementary) | 1,000.000000 |

Values are from implementation A, which computes `bel`, `v_be` and `v_st` by backward recursion; implementation B, which discounts the projected cash flows directly, agrees on every value in this table to a maximum relative difference of 1.1e-14.

Group:

| Quantity | Value |
|---|---|
| P_g | 190,000.0000 |
| BEL_g(0) | 179,192.1480 |
| RA_g(0) | 2,648.249534 |
| CSM(0) | 8,159.602426 |
| LC(0) | 0 |
| CSM(12) | 7,854.527657 |
| CSM(12) with `discount_coverage_units` = False | 8,072.508520 |

Values are from implementation A, which computes `bel`, `v_be` and `v_st` by backward recursion; implementation B, which discounts the projected cash flows directly, agrees on every value in this table to a maximum relative difference of 1.1e-14.

Group, by month:

| Month t | CU_g(t) | R(t) |
|---|---|---|
| 1 | 1,000.000000 | 46.56786380 |
| 2 | 999.5156821 | 46.54598345 |
| 3 | 999.0315987 | 46.52412063 |
| 4 | 998.5477498 | 46.50227529 |
| 5 | 998.0641352 | 46.48044737 |
| 6 | 997.5807548 | 46.45863682 |
| 7 | 997.0976086 | 46.43684360 |
| 8 | 996.5570554 | 46.41238312 |
| 9 | 996.0167953 | 46.38794249 |
| 10 | 995.4768280 | 46.36352167 |
| 11 | 994.9371535 | 46.33912060 |
| 12 | 994.3977716 | 46.31473923 |

Values are from implementation A, which computes `bel`, `v_be` and `v_st` by backward recursion; implementation B, which discounts the projected cash flows directly, agrees on every value in this table to a maximum relative difference of 1.1e-14.

Hand checks, exact unless a tolerance is given; they do not replace the values above:
- A_1(1) = A_1(13) = 1,000.00.
- CU_g(1) = 1,000.00, because pols_if(0) = 1.
- P_g = 190,000.00.
- With either switch value, R(t) / CU_g(t) = CSM(t) / N(t) within I12's tolerance.

### Example 2: level policy, onerous group

Policy 4 alone, as group (`IA`, 2025, `ONEROUS`), with every other setting as in example 1.

| Quantity | Value |
|---|---|
| BEL_g(0) | 179,192.1480 |
| RA_g(0) | 2,648.249534 |
| CSM(0) | 0 |
| LC(0) | 6,840.397574 |
| LC(1) | 6,818.594999 |
| LC(12) | 6,585.155578 |

Values are from implementation A, which computes `bel`, `v_be` and `v_st` by backward recursion; implementation B, which discounts the projected cash flows directly, agrees on every value in this table to a maximum relative difference of 1.3e-14.

- BEL and RA must equal policy 1's in example 1 bit-for-bit on the same backend: both runs are one-policy batches with identical inputs (D19).
- CSM(0) must be 0, and LC(t) / F(t) must be the same at t = 0, 1 and 12 (I13).
- Data-contract test: example 1 with policy 1's premium set to 175,000.00 and its bucket left as `OTHER` must fail G3, naming policy 1's row in `mp_level`.

### Example 3: three classes in one group

Policies 1, 2 and 3 together, as group (`IA`, 2025, `OTHER`), with every setting as in example 1. Premiums were set as in example 1, so every contract passes the para 47 test.

Per policy:

| Quantity | Policy 1 | Policy 2 | Policy 3 |
|---|---|---|---|
| BEL_i(0) | 179,192.1480 | 206,382.8002 | 234,466.2049 |
| RA_i(0) | 2,648.249534 | 4,031.105418 | 5,607.325647 |
| A_i(1) | 1,000.000000 | 800.0000000 | 1,200.000000 |
| PVA_i | 175,222.1063 | 201,200.6846 | 229,894.9870 |
| PVE_i | 3,970.041706 | 5,182.115650 | 4,571.217872 |
| A_i(13) (supplementary) | 1,000.000000 | 824.0000000 | 1,232.981869 |

Values are from implementation A, which computes `bel`, `v_be` and `v_st` by backward recursion; implementation B, which discounts the projected cash flows directly, agrees on every value in this table to a maximum relative difference of 9.3e-14.

Group:

| Quantity | Value |
|---|---|
| P_g | 660,000.0000 |
| BEL_g(0) | 620,041.1532 |
| RA_g(0) | 12,286.68060 |
| CSM(0) | 27,672.16624 |
| LC(0) | 0 |
| CSM(12) | 26,894.10869 |

Values are from implementation A, which computes `bel`, `v_be` and `v_st` by backward recursion; implementation B, which discounts the projected cash flows directly, agrees on every value in this table to a maximum relative difference of 5.2e-14.

Group, by month:

| Month t | CU_g(t) | R(t) |
|---|---|---|
| 1 | 3,000.000000 | 136.9208221 |
| 2 | 2,998.144065 | 136.8378312 |
| 3 | 2,996.289390 | 136.7549127 |
| 4 | 2,994.311796 | 136.6663985 |
| 5 | 2,992.335685 | 136.5779662 |
| 6 | 2,990.361056 | 136.4896156 |
| 7 | 2,988.387909 | 136.4013465 |
| 8 | 2,986.358601 | 136.3105277 |
| 9 | 2,984.330831 | 136.2197926 |
| 10 | 2,982.304598 | 136.1291410 |
| 11 | 2,980.279900 | 136.0385725 |
| 12 | 2,978.215766 | 135.9462170 |

Values are from implementation A, which computes `bel`, `v_be` and `v_st` by backward recursion; implementation B, which discounts the projected cash flows directly, agrees on every value in this table to a maximum relative difference of 1.2e-14.

Hand checks, exact unless a tolerance is given; they do not replace the values above:
- A_i(1) = `instalment_pm` for each policy, and A_1(13) = 1,000.00.
- A_2(13) = 824.00 exactly: fl(1 + 0.03) = fl(1.03), and 800 × fl(1.03) rounds to 824.0.
- A_3(13) = 1,200 × 1.025^(10/12) × 1.04^(2/12) ≈ 1,232.98187, within 1e-12 relative. This is 1,200 × I(10) / I(−2) with L = 3; a lag of 2 or 4 gives a visibly different value (about 1,231.49 or 1,234.48).
- CU_g(1) = 3,000.00, because pols_if(0) = 1 and no escalation falls in month 1.
- P_g = 660,000.00.
- Policy 1's per-policy values equal example 1's within 1e-12 relative on the same backend (D5).
- Horizon test, a separate run: policy 3 alone has T = 591. Its BEL and RA, and every value compared under section "Results", must equal its values in this example within 1e-12 relative on the same backend (D5). Rows of the scenario files beyond 591 are ignored.

## Open questions

Questions marked (Spec-level) concern *Halley — Spec* rather than this product. They belong in *Halley — Open Questions* with a link back here.

1. **Scope against D2.** Resolved 2026-10-15: the fixed and CPI-linked variants are in M5 alongside the level annuity, and D2 now names them.
2. **Units (D11 is firm; Spec-level).** This note proposes `Prob[Annual]`/`Prob[Monthly]`, a compounding tag on annual rates (`Effective`, `Continuous`), `Factor`, `Index`, `CoverageUnits`, `Int` and `Bool`, and the operation rules in section "Units", including no rule for 1 + Rate. They extend D11's list rather than reverse it. Confirm, and update the Spec's Units table. Without the period tag on `Prob`, carried through `1 − Prob` and `Prob × Factor`, an annual q used as a monthly q passes the checker, in `deaths` or in `v_be`. Without the compounding tag, a continuous rate passed to `to_monthly` passes it. Also open: whether `Factor × Prob` should be its own unit rather than `Prob` (this note avoids forming it), and whether an untagged `Prob` stays a unit for α or α gets its own type, such as `Level`.
3. **Investment expenses.** Inclusion under B65(ka)(i) is a judgement; the counter-view is that asset management after inception serves the entity's spread. The rate basis, closing balance (default) or average assets, is a judgement too (see section "Investment expenses"). Confirm both with the reviewer.
4. **Coverage units.** Method 1, discounted, is the default (2022 agenda decision). Open: the discount rates (locked-in chosen; the decision is silent); whether to allocate once per reporting period, which with discounted units differs slightly from monthly (B137 is out of scope); and whether to keep the undiscounted switch, which the decision did not consider.
5. **CPI-linked coverage units** include expected future indexation (default). The alternative is the currently payable amount only.
6. **Mortality experience.** The effect of actual deaths on future payments is taken to adjust the CSM (B96(b)). This is an inference from B96 and B97, not an IASB or TRG statement.
7. **Locked-in CPI path for in-force groups.** Default: actual CPI history to V, then the locked-in expected inflation, as defined in section "In-force groups (M9 to M11)". The alternative is current CPI expectations discounted at locked-in nominal rates, which mixes bases. Confirm that a splice supplied as a hashed file respects D4 (firm).
8. **In-force groups.** Still open: the formula for the LC allocation over a period longer than a month (step 2); the worked example with a non-zero ΔFCF^L and CPI experience; step 4's single release per period, which with discounted units differs from the monthly allocation step (open question 4); and whether groups with differing issue dates should be allowed with a supplied B73 weighted-average curve.
9. **RA calibration.** The parameters are placeholders. Should σ vary by duration? Should level and trend risk be split? The factor is mean-centred, so the best-estimate tables are its mean, but the BEL is not exactly E[PV(Z)] (gap sized in section "Known limitation"). Should the BEL be computed as that mean instead, and above what gap? Should diversification be taken across groups? The RA leaves out random variation between individual lives; do small groups, and the per-contract para 47 test, need an idiosyncratic component?
10. **RA disaggregation** (para 81) is on by default. Confirm.
11. **ILP.** A flat continuously compounded spread is assumed; its size must not contradict market data (B78(a)). A term structure is the alternative. A zero ILP for a first reconciliation is optional.
12. **Acquisition cash flows** are excluded. Including them brings in para 38(b) at recognition and B125 amortisation.
13. **Guarantee periods** create an investment component (Appendix A) and possibly an investment-return service (B119B). Out of scope. Should the base class gain a `lives_paid` hook now to make that variant a one-hook override?
14. **Scenario mean (Spec-level).** Until M12 lifts the check (G6), aggregation requires J = 1. Is that acceptable against D4 for M7 to M11?
15. **Aggregation-stage language (Spec-level).** Is it written with the same `@var` language and checker? Does the scheduler support a backward component feeding a forward one (`cu_next` then `csm`), and inits computed from other variables at t = 0 (`csm_0`, `lc_0`, via `fcf_g.at(0)` from the backward BEL and RA)? The Spec's Data section has no contract yet for group inputs.
16. **Several product classes in one group (Spec-level).** The stage joins classes on the group key. The Spec does not yet say it can.
17. **Engine features assumed here but not in the Spec (Spec-level):**
    - table-lookup syntax; keyless tables; load-time derived model point fields;
    - absolute-time reads of inputs (`cpi.at(0)`), limited to inputs so the direction check stays sound; input rows at t ≤ 0; `lag()` on scenario inputs, where the Spec defines it on variables;
    - eager `where()` reading both branches, which sets the CPI history requirement and the rule that divisions inside `where()` are guarded; a unitless literal in `where()` taking the other branch's unit;
    - integer key types despite "float64 throughout"; variable domains (0..T against 1..T, section "Time"), on which the D11 build-time check of time references depends;
    - variables with fewer dimensions than policy × scenario × time (time-only, scalar), against the Spec's "policy and scenario always exist";
    - abstract product classes (`EscalatingAnnuity`);
    - the functions `minimum()`, `exp()`, and `expm1`/`log1p` inside helpers; the helpers `pv_month` and `fwd_rate`;
    - the "at t" (state) timing next to the Spec's start, mid and end;
    - which member of a recursive component carries the `init` or `terminal`: here the one whose formula would read outside a domain.
18. **Spec example timing (Spec-level).** The term example's reserve, (reserve(t+1) + net_outgo(t+1)) × v(t+1), discounts a start-timed premium by a full month. This note uses `pv_month` instead. The Spec example should change to match.
19. **Female mortality.** Is the four-year setback acceptable (see section "Mortality")? Or should the first reconciliation be unisex?
20. **Month-end valuation dates only (Spec-level).** Acceptable as a load-time rule?
21. **Horizon (Spec-level).** The annuity needs up to 841 months (age 50 at V); benchmarks W1 and W2 use 600.
22. **Citations.** The DHW section numbers are from the first edition (2009) and were not checked page by page. The parameters reproduce the SOA table's printed values at the ages checked (ten in the research; 65 and 100 for this note).
23. **lifelib reconciliation (D23) for annuities.** lifelib's PA_UK_S uses different conventions (age from entry age, ω = 115, base-table multipliers). A separate note should set the switches.
24. **Evaluation order (Spec-level).** This note makes the model source's grouping the canonical floating-point order, so that implementations that follow it agree to about 1e-14; the worked example's 1e-9 tolerance does not need it. Should the Spec say so for every model, and should generated code be required to keep it?
25. **Invariant tolerances (Spec-level).** The Spec gives one relative tolerance. This note needs a scale per invariant (section "Built-in invariants, checked on every run"). Should the Spec define scales, or let each product declare them?

## Sources

### Standards, IASB and TRG material

- IASB (2017, amended June 2020). IFRS 17 *Insurance Contracts*, including Illustrative Example 8. https://www.ifrs.org/issued-standards/list-of-standards/ifrs-17-insurance-contracts/. Paragraph text read from AASB 17 compiled, which has identical numbering: https://standards.aasb.gov.au/sites/default/files/2025-05/AASB17_07-17_ACOMPdec22_07-26.pdf and https://aasb.gov.au/admin/file/content105/c9/AASB17_07-17_COMPdec21_01-23.pdf
- IASB (2020). Basis for Conclusions on Amendments to IFRS 17, BC283A–BC283I, reproduced in AASB 2020-5 BC. Cited for investment expenses. https://aasb.gov.au/admin/file/content105/c9/AASB2020-5_BC_07-20.pdf
- IASB staff (February 2020). Agenda Paper 2A, "Contractual service margin attributable to investment services". Cited for investment expenses. https://www.ifrs.org/content/dam/ifrs/meetings/2020/february/iasb/ap2a-amendments-to-ifrs.pdf
- TRG for IFRS 17 (May 2018). Agenda Paper 05, "Determining the quantity of benefits for identifying coverage units", Example 12. Cited for coverage units. https://www.ifrs.org/-/media/feature/meetings/2018/may/trg-for-ifrs-17/ap05-quantity-of-benefits-for-identifying-coverage-units.pdf
- TRG for IFRS 17. Summary of the meeting of 2 May 2018. Cited for coverage units. https://www.ifrs.org/-/media/feature/meetings/2018/may/trg-for-ifrs-17/trg-for-ifrs17-meeting-summary.pdf
- IFRS Interpretations Committee (2022). Agenda decision, "Transfer of Insurance Coverage under a Group of Annuity Contracts (IFRS 17 Insurance Contracts)". Tentative decision in IFRIC Update March 2022; finalised at the Committee's June 2022 meeting; published July 2022 after the IASB did not object. Cited for coverage units (Methods 1 and 2; "IFRS 17 does not prescribe a method"). https://www.ifrs.org/content/dam/ifrs/supporting-implementation/agenda-decisions/2022/transfer-of-insurance-coverage-under-a-group-of-annuity-contracts-jul-2022.pdf
- EY (April 2019). Insurance Accounting Alert: TRG meeting of 4 April 2019 (submissions S93 and S122). Cited for CPI changes as financial risk. https://www.ey.com/content/dam/ey-unified-site/ey-com/en-gl/technical/ifrs-technical-resources/documents/ey-iaa-ifrs-17-insurance-trg-april-2019.pdf
- EFRAG TEG (December 2017). Paper 05-04A, "CSM: use of locked-in or current discount rate". Cited as background for locked-in rates. https://efrag.org/system/files/sites/webpublishing/Meeting%20Documents/1607180948242466/05-04A%20Issues%20paper%20on%20current%20vs%20locked-in%20rate%20for%20CSM%20-%20TEG%2017-12-18.pdf

### Actuarial practice

- IFoA IFRS 17 CSM Working Party (21 November 2019). Findings, session D3, Dublin. Cited for the loss-component allocation and for discounting of coverage units. https://vle.actuaries.org.uk/pluginfile.php/144100/mod_resource/content/1/IFRS%2017%20CSM%20Working%20Party%20Findings%20D3.pdf
- International Actuarial Association (2018). *Risk Adjustments for Insurance Contracts under IFRS 17*. Cited for the choice of RA technique. https://actuaries.org/product/risk-adjustments-for-insurance-contracts-under-ifrs-17/
- Boumezoued, A. et al. (November 2020). *IFRS 17: Deriving the confidence level for the Risk Adjustment: a case study for life (re)insurers*. Milliman. Cited for the confidence-level conversion that other techniques need. https://edge.sitecorecloud.io/millimaninc5660-milliman6442-prod27d5-0001/media/Milliman/PDFs/2020-Articles/London-Solvency-II/11-11-20-RA-percentile-conversion-v1.pdf

### Mortality and life contingencies

- Dickson, D.C.M., Hardy, M.R. and Waters, H.R. (2009). *Actuarial Mathematics for Life Contingent Risks*, 1st ed. Cambridge University Press. ISBN 978-0-521-11825-5. Sections 3.3 (fractional ages), 5.6 (annuities payable m times a year), 5.10 (increasing annuities), 7.3–7.4 (policy value recursions); the Standard Ultimate Survival Model. Front matter: https://assets.cambridge.org/97805211/18255/frontmatter/9780521118255_frontmatter.pdf
- Bowers, N.L., Gerber, H.U., Hickman, J.C., Jones, D.A. and Nesbitt, C.J. (1997). *Actuarial Mathematics*, 2nd ed. Society of Actuaries. Chapter 5 (life annuities). Cited for the per-life recursion.
- Society of Actuaries (October 2022). Exam FAM-L Tables, Standard Ultimate Life Table. Cited for the check values. https://www.soa.org/globalassets/assets/files/edu/2022/2022-10-exam-fam-l-tables.pdf
- Lim, T. actuarialmath (MIT licence), Standard Ultimate Life Table guide. Cited for the Makeham parameters. https://actuarialmath-guide.readthedocs.io/en/latest/sult.html
- Society of Actuaries (2014). *Mortality Improvement Scale MP-2014 Report*, sections 1.2, 3.3, 6.3 and Appendix B. Cited for the convergence form of the improvement rates. https://www.soa.org/globalassets/assets/files/research/exp-study/research-2014-mp-report.pdf
- Arias, E., Xu, J. and Kochanek, K. (2025). United States Life Tables, 2022. *National Vital Statistics Reports* 74(2). NCHS, public domain. Cited for the female setback. https://stacks.cdc.gov/view/cdc/174575

### Interest and index-linking

- Kellison, S.G. (2009). *The Theory of Interest*, 3rd ed. McGraw-Hill/Irwin. Chapter 1. Cited for the compounding conversions.
- UK Debt Management Office (2005). *Formulae for Calculating Gilt Prices from Yields*, 3rd ed., Introduction and Annex B (three-month indexation lag). Cited for L = 3. https://www.dmo.gov.uk/media/1sljygul/yldeqns.pdf
- Republic of South Africa, Department of Finance (1999). *Draft Technical Specifications for CPI Bonds*, section 2 ("Reference CPI", a four-month lag; "Index Ratio", a floor at redemption). Cited for lags differing by market and for an embedded floor. https://www.treasury.gov.za/divisions/alm/1999/cpi_bond_spec.pdf

### Open-source prior art

- lifelib (MIT licence), commit b706d8bb: basiclife (constant-force monthly conversion), uklib PA_UK_S (annuity in payment), ifrs17a and ifrs17sim (CSM mechanics). https://github.com/lifelib-dev/lifelib

### Halley notes

- *Halley — Spec*, *Halley — Decisions*, *Halley — Roadmap*, *Halley — Prior Art*.

## Provenance

- Written on 2026-10-14 from the public sources listed above, general actuarial knowledge and the Halley notes. Revised on 2026-10-15 after four review rounds; the worked-example values were filled in once the two implementations reconciled. For the revisions, the IFRS Interpretations Committee's final agenda decision, the DMO paper and the RSA draft specification were read again from their public URLs.
- The scope was given as a generic feature list: mortality with improvements, escalation, inflation, discounting with investment expenses, BEL, RA, and CSM with coverage units. No formula, structure, table, number or code of any existing model came with it.
- No employer code, models, assumption tables, product terms, results or data were consulted (D29).
- Every number in the worked example is synthetic and chosen for this note.
- The mortality basis is generated from published Makeham parameters. No SOA, CMI or other restricted table file is copied or redistributed.
- The premiums of policies 1–3 were set from a rough independent estimate of BEL + RA. Policy 4's premium was set below an independent estimate of policy 1's BEL, so the contract is onerous for any RA ≥ 0. Neither estimate is a reference result, and neither is reported here.
- The limiting-age figures (survival to 120, the tail of the annuity value) and the mean-PV gap in section "Known limitation" were computed for this note from the example basis.

## Implementation notes and deviations

Added by the engine author on 2026-10-17. This section is not part of the specification above and has not been reviewed. It records where the engine's implementation, as fixed in the design for milestones M5 to M12, differs from or goes beyond the specification's text, so that a reader of the code can trace each difference back to a choice. Nothing above was changed to match the engine.

- **1 + Rate (design section 1).** The specification says there is no rule for 1 + Rate. The engine keeps `literal ± rate → Factor` for effective rates (`Rate[Monthly]`, `Rate[Annual]`), because the compounding tag catches the real mistake: `1 + Rate[Annual, Continuous]` is a unit error whose hint says a continuously compounded rate accumulates through `cc_discount()`, and `to_monthly(self.tables.ilp.ilp)` fails the build as the specification requires. Removing the rule would rewrite two reconciled models.
- **Compounding tags (section 1).** `Rate[Annual, Effective]` is a second spelling of `Rate[Annual]`, one unit object printed `Rate[Annual]`, so existing hashes and run ids are unchanged; `Rate[Annual, Continuous]` is a new unit printed exactly so; `Rate[Monthly]` is effective by definition: `Rate[Monthly, Effective]` is the same annotation object (the PEP 696 default fills the second parameter in) and resolves to the same unit, and `Rate[Monthly, Continuous]` is a unit error. `Index` and `CoverageUnits` are implemented as proposed; `Money + CoverageUnits` is a unit error with the hint "coverage units are not money; convert with as_cu()". The specification's `PerPolicy[Money]` is the engine's `MoneyPerPolicy`.
- **Helpers (section 2).** `to_monthly_cf`, `annual_step`, `cc_discount`, `fwd_rate` and `as_cu` are Call nodes with a fixed emitted form. `ann_fwd(p0, p1)` and `ann_spot(p, m)` are added for the disclosed annualised forward and spot rates. `pv_month(cf, t, v)` is not a Call node: it expands at trace time, by `cf`'s declared timing, to `cf(t+k)` (start), `cf(t+k) × v(t+k)^0.5` (mid) or `cf(t+k) × v(t+k)` (end), and raises a build error at the call site when `cf` declares no timing, when `cf` or `v` has no time dimension, or when `v` is not a `Factor`. A time expression folds to one offset only when it is an add/sub tree of one `t` and integer literals, so `t - L - 12` and `t - (L + 12)` are the same read and `t - self.mp.lag_m` fails as the deliberate error requires.
- **Powers with a non-literal exponent (D19, D5).** NumPy chooses its sqrt, square or reciprocal fast path for a power whose exponent is not a literal (`x(t) ** self.mp.k`) from the shapes of the batch, so when that exponent is 0.5, 2 or -1 a policy's result can differ in the last place between a batch of one policy and a batch of three; the engine keeps the emitted power because "Built-in invariants" above allows exactly this ("Bit-for-bit checks need the same backend and the same batch composition (D19); the same policy in a different batch, or a run with a different T, agrees within 1e-12 relative (D5)"), and the annuity's own formulas raise only to literals and to the `-12 / t` of `ann_spot`, which took libm's pow at every batch shape measured over a window of two or more months (a month-column exponent over a one-month window, H = 1 for a formula from t = 1, takes the fast path at every batch size, so `recompute` and `explain` differ from the stored value in the last place there; `ann_spot`'s exponent is -1 only at t = 12).
- **Boundary values (section 5).** `init` and `terminal` callables may read variables as well as model point fields: `lambda g: g.csm_0()` for a scalar, `lambda g: at(g.fcf_g, 0)` for a fixed month. A name resolves as a model point field first and as a variable otherwise; a boundary may not read `at(member, 0)` of a member of its own recursive component.
- **Spellings (sections 5 and 8).** An absolute-time read of a variable is `at(x, 0)`, of a scenario input `self.scen.x.at(0)`; the sketch above writes `self.fcf_g.at(0)` for the former. A scalar variable is read as `self.x()`: `ie_m` is `def ie_m(self) -> Rate[Monthly]`, read as `self.ie_m()` where the sketch reads `self.ie_m(t+1)`.
- **Annuity product (section 8).** `ra` is declared `@var(domain="state")`; the horizon is the derived field `t_end`. `EscalatingAnnuity` is abstract through `abstract: ClassVar[bool] = True`, which `build()` refuses and subclasses do not inherit. Added variables, stored as outputs: `pva` and `pve` (sums over t of `pva_t = df(t-1) × ann_out(t)` and `pve_t = df(t-1) × exp_out(t) + df(t) × inv_exp(t)`), `premium_total = pols_if_init × single_premium`, `onerous = (premium_total − (bel(0) + ra(0))) < 0` (FCF formed first, as G3 requires), the disclosure series `fwd_ann` and `spot_ann`, and `run_off_month = t_end`, an output beside the derived field so that readers of the outputs can flag per-life values beyond it. `q_ann`, `q_mth`, `q_mth_st`, `v_be`, `v_st` and `esc_factor` are requested through the run settings for tracing. `CPILinkedAnnuity` also generates `cpi_lag`, a scalar `Int` variable holding the build-time constant `cpi_lag_m`, because invariant I9 reads the lag through the batch view, which exposes no statics; it is not an output. The derived field `m_c` carries no upper bound of its own: the rule `annuity_start_date <= V + 1 day` is the single source of the constraint, so the spec's named rule is the message that fires. A horizon given in the run settings must reach every policy's `t_end`: the input rule "the horizon covers every policy's run-off month" fails at load naming the first policy it would truncate, where the specification only derives T and says nothing of a shorter setting. The "reads across policies" deliberate error is spelled `self.series.cu(t)` inside a `Product`. A projection run holds one class file, so the horizon T = max over every class file of "Time" is not derived across runs: to form one group across classes, every class run of a basis takes the same `RunSettings.horizon` (the largest `t_end` over the group's files; example 3's three class runs all take T = 655), which the `Basis` of section 9 requires and the input rule above keeps safe.
- **Tolerance scales (section 8).** The product declares a Money scale of `max(pols_if_init × single_premium, pols_if_init × instalment_pm × 12)` per policy (the derived field `scale_money`; the floor of a year's payments keeps the scale positive for an in-force policy whose file carries no premium, which the schema allows before V), `instalment_pm` for `MoneyPerPolicy` and `pols_if_init` for `Count`; the specification's "Quantities" gives 1e-9 × P_g for zero values at the group level only.
- **Locked-in runs (section 8; in-force C17).** There is no `locked_in` switch. A locked-in projection run reads the re-based composed curve P_L through `rf_df` with an `ilp` table holding 0.0: `cc_discount(0, t) = exp(0.0) = 1.0` exactly, so `df = rf_df × 1.0` is bit-identical and one build serves all four runs of a period.
- **Aggregation stage (section 9; the answer to open questions 15 to 17).** It is written in the same `@var` language and checked by the same checker. An `Aggregation` is a `Product` whose model points are groups and whose extra inputs are group series: a `Basis` names projection runs with the same valuation date, horizon and J, pairwise disjoint policy ids (one run per product class, which is how several classes join one group) and every scenario input they share read from the same file (compared by the sha256 their manifests record, so that scenario s is one path in every run), and a `GroupSum` sums named output variables over the group's members in ascending `policy_id` across the basis's runs, then over variables left to right, then over scenarios as `cumsum_s((1.0 / J) * x)[-1]` (`scenarios="mean"`; exact when J = 1). Formulas read `self.series.<name>(t + k)`, `self.series.<name>.at(k)` or `self.series.<name>()`; the sketch's `@gvar`, `self.policies.sum(...)`, `self.policies.sum_mp(...)` and `self.locked.df(t)` become `@var`, `GroupSum` declarations and the scenario input `self.scen.p_l(t)`. A series' recorded unit must equal the declared one (D11 across runs). Each series records its basis's valuation date, horizon (`basis_horizon`, for a series without a time dimension too), J (`basis_scenarios`), source runs and members, which the loader holds to its values (at most the basis's months, an "each" series exactly its J) and to the other series of its basis (the same runs, by run id and manifest hash, and the same members: one `group_series` call); across bases a policy is a member of one group. A stage's outputs are keyed on `group_id`, and its manifest marks them `outputs.kind = "aggregation"`; a basis reads projection runs only. `CsmRollForward(Aggregation)` carries the monthly steps with the hooks `csm_0`, `lc_0` and `p_g` and adds `lrc = fcf_g(t) + csm(t)`; `NewGroup` (M7) takes `p_g` from the series `premium_total` and forms `lc_0` from `FCF_g(0) − P_g` as written above; `InForceGroup` (M11) adds the bases `old`, `new` and `cur`, the scenario input `p_rec` and the period close in the order of "In-force groups (M9 to M11)". G1 stays a check in one place.
- **Previous-run quantities (section 9).** For an in-force group, P_g is carried forward from recognition as the previous aggregation run's output, not summed again from the file at V; `f_prev_0`, `f_prev_tau`, `lc_prev_tau`, `tau0`, `tau`, `i_acc` and `i_rec` are derived group fields the loader fills from the previous run, never retyped by hand.
- **M12 revenue and IFIE (section 9).** On `CsmRollForward`, for the months after the close on the `cur` basis, with the scenario input `df_cur` (the one-scenario market-implied composed current curve; J = 1 and `df_cur(0) = 1`). Quoted from the design exactly:

  ```text
  f = fwd_rate(df_cur(t-1), df_cur(t))
  ifie_bel = f(t) * ((bel_g(t-1) - ann_out_g(t)) - exp_out_g(t))
  ra_fin = f(t) * ra_g(t-1)
  ra_rel = (ra_g(t-1) + ra_fin(t)) - ra_g(t)
  sar = lc(t-1) / where(f0 > 0, f0, 1.0)
  svc = ((ann_out_g(t) + exp_out_g(t)) + inv_exp_g(t)) + ra_rel(t)
  lc_svc = sar(t) * svc(t)
  revenue = (svc(t) - lc_svc(t)) + release(t)
  ifie = (ifie_bel(t) + ra_fin(t)) + acc(t)
  ```

  Invariants M1: `lrc(t) - lrc(t-1) = ifie(t) - revenue(t) - lc_svc(t)`, scale `p_g`, tolerance 1e-12; M2: `lc_alloc(t) = lc_svc(t) - sar(t) * (ifie_bel(t) + ra_fin(t))`, scale `p_g`. The reading behind them: B124, revenue excludes only the LC's share of the service items, and the LC's share of IFIE stays in IFIE; para 81, the RA is split by its finance unwind at the current forward rate. The close's IFIE pieces (the effects of curve and CPI changes and the current-against-locked-in difference, named under "Outputs added in M12" above) are the in-force group's analysis-of-change outputs, not the monthly `ifie`. Built in `halley.aggregation.csm` (stage B3a), with the group series `ann_out_g`, `exp_out_g` and `inv_exp_g` beside `bel_g` and `ra_g` and every line above stored by month. The design's rule "J = 1" on `df_cur` is not a rule of the input: the stage reads mean series, so the loader already holds every scenario input of the stage to `require_deterministic`, the one J = 1 check of D4, and a two-scenario `df_cur` is a LoadError there. The design makes `df_cur` the cur runs' composed curve and discloses its hash as the para 120 curve, so the loader checks the file as it checks `p_l` (below): `CsmRollForward` declares `discloses_current_curve`, and `df_cur`'s metadata must name the `rf_df` and `ilp` files every cur run read (`source_sha256`, `ilp_sha256`) and `rebased_from`, so that the provenance check rebuilds it from them bit for bit; a curve re-based at another month (`tau0` other than 0) is refused too. A LoadError names `df_cur` otherwise, and the disclosures record the files under `para_120_curve_verified`. M1 and M2 hold only when `df_cur` is the curve the cur runs' BEL and RA unwind on, so they remain the run-time check of its values, the only one for a stage run through `run()`, which reads no basis run; for a group new at V it is the same curve as `p_l`, and `f = j_L` bit for bit.
- **Roll-forward lines stored (section 9; "Roll-forward table", "Outputs").** The CSM table's opening line is stored as `csm_open(t) = csm(t − 1)` beside `acc`, `release` and `csm`. Its changes for future service are 0 in every month of the roll on the cur basis (in-force section 10: "There is no change for future service inside the months"), and are not a stored line: an in-force group's change for future service is its period close's `csm_fs` (in-force 11.1), a per-group figure of the in-force stage, which a monthly line of the same name would contradict. `d_cu(t)`, the units' one-month discount factor (`d_L(t)`, or 1 with `discount_coverage_units = False`), is a stored variable: the sketch's build-time branch inside `cu_next` moves into it, so `cu_next(t) = cu_g(t + 1) + d_cu(t + 1) × cu_next(t + 1)` in both builds (`x × 1.0` is exact, so the values are unchanged) and I12's step reads the same d.
- **The coverage-unit approach disclosed as built (section 9, "Outputs"; para 117(c)).** "Outputs" lists two notes. The manifest's notes add a third, the coverage-unit approach the stage was built with: discounted on the locked-in curve (Method 1), or undiscounted with every month weighted equally (`discount_coverage_units = False`). The approach to discount rates is a fixed string and does not mention the units, so a stage built with undiscounted units does not disclose discounted ones. A stage's `disclosures` entry may for this be a function of its statics, which the loader calls with the compiled stage's.
- **NewGroup reads a deterministic basis (G6, "Scenarios", "Labelling").** The design lets a basis with J > 1 feed a stage through mean series, so that the stage runs at S = 1. `NewGroup` does not take one: G3 reads each contract's `onerous` output, the para 47 test at one scenario, and refuses a run with J > 1 through `require_deterministic`; P_L = P names the composed curve of one run, which `composed_curve` and `locked_in_curve` build from a one-scenario discount file. At J > 1 only the probability-weighted mean of FCF_i(0) is a para 33(a) estimate ("Labelling"), so G3 would compare P_i with `cumsum_s((1.0 / J) * (bel_i(0) + ra_i(0)))[-1]`, and the locked-in and current curves would be the market-implied curve rather than one run's. The engine does not choose this ahead of the specification: it is open with G6 until M12 ("Until M12, the aggregation stage reads a deterministic run (J = 1) and rejects J > 1").
- **G3 and G5 as member rules (design section 9).** The design calls G3 and G5 input rules. An input rule of a stage sees the groups, not their contracts, and the table "Level of aggregation" wants a failure to name the contract's file and row, so the engine adds `MemberRule`: a rule on every member of a basis, which `run_aggregation` evaluates on each run of the basis from its verified model point copy and stored outputs. A failure names the rule, the run, the copy's row and its policy, and the file the run read its model points from; a rule whose own code fails is located at its declaration. A rule declares the outputs it reads, and its view serves no other, so that the record of what it read is true; it returns one bool for every row or one per row. `run()`, which sees only the group series, refuses a stage that declares member rules. The manifest's `member_rules` block records each rule, its basis and the runs it held for; a replay reads no basis, so it carries that record over and cannot run the rules again.
- **I14 split in two (design section 9).** I14 says G3 holds for every contract of a group new at V. The run-time invariant checks what the stage can see on its own figures, the group-level consequence: CSM(0) = 0 in an `ONEROUS` group and LC(0) = 0 in any other, within 1e-12 × P_g. The group sums P_g and FCF_g(0) in another order than each contract's exact G3 test (`P_i - (BEL_i(0) + RA_i(0))`, FCF formed first), so a contract-level tie can leave a balance of a few ulps of P_g on the wrong side: two contracts each priced at exactly their FCF, which G3 accepts in an `OTHER` group, leave an LC(0) of about 6e-11 on a P_g of about 4.1e5. The per-contract test is the member rule G3 at load, exactly, from each run's `onerous` output; it is not repeated on a replay.
- **I13 where F(0) > 0 (in-force 12.2).** I13's ratio `LC(t) / F(t) = LC(0) / F(0)` is checked wherever F(t) > 0 and F(0) > 0, a group with LC(0) = 0 included, where it requires LC(t) = 0. In-force 12.2 restates I13 "where F(t) > 0 and LC(0) > 0", which is weaker: a loss component that grows from 0 in a profitable group would pass it. The engine takes the form written above, which implies 12.2's.
- **I11's exact zero (design section 9; in-force open question 17).** The engine checks `CSM(t) = 0` exactly from the last month with `CU_g > 0` on, that month included: N is exactly 0 there, so the release is the whole CSM before it. The literal "after the last month with CU_g > 0" is vacuous when that month is H, as it is in every example. This takes the stricter reading that in-force open question 17 asks the specification's author to choose; it is the engine's reading until the author answers, not a change to the specification.
- **P_L = P checked by file (design section 9; "Locked-in rates").** For a group new at V the stage's `p_l` must be the composed curve of the runs it reads. `NewGroup` declares `recognised_at_valuation`, and the loader requires `p_l`'s metadata to name the `rf_df` and `ilp` files every cur run read (`source_sha256`, `ilp_sha256`), `tau0 = 0` and `rebased_from`, so that the provenance check rebuilds it from those files bit for bit. A curve that names no source, one re-based at another month or one composed with another ILP is a LoadError naming `p_l`. The files are compared, not `p_l(t)` against each run's `df(t)`: the helper composes in Python floats and the run in NumPy, whose `exp` need not round alike on every platform. The manifest's `locked_in` block records the verified curve under `p_l`. No invariant sees a wrong `p_l`: the accretion is on both sides of I11 and of M1, so the bare discount file passes every one with a CSM(12) of 7843.11 against 7854.53. `run()`, which reads no basis run and so cannot make the check, refuses a stage that declares `recognised_at_valuation`, as it refuses one with member rules.
- **A horizon below the run-off (design section 9, "Horizon"; D5).** The design lets `RunSettings.horizon` be at most the cur basis's horizon. N(t) needs every later month of the coverage units, so a stage cut before the group's run-off would release the whole CSM at H and its results would depend on T, against "a run with a different T agrees within 1e-12 relative" above. `CsmRollForward` declares `reads_to_run_off`: a horizon below the cur basis's is accepted only where every series the stage reads relatively is exactly 0 after it, and is a LoadError naming the basis, the run, the variable, the first month dropped and the last month with a value otherwise.
- **Cross-run helpers (section 10).** Where the specification says "the loader re-bases", a helper writes a hashed file with provenance metadata, which the ordinary loader reads and re-verifies by recomputing it and requiring bit equality: `expected_in_force` (the previous run's `pols_if` at τ as the old run's model point file, `instalment_pm` taken from the file at V; metadata `halley.derived_from`, `halley.tau`), `rebase_expense_table` (`maint_pm × I_act(0) / I_act(−τ)` in that order; `halley.price_base`, `halley.rebased_from`, `halley.cpi_act_sha256`), `locked_in_curve` (writes `p_rec`, the composed recognition curve over 0..τ0 + T, and `p_l = P_rec(τ0 + t) / P_rec(τ0)`, both with value column `rf_df` and metadata `source_sha256`, `ilp_sha256`, `ilp`, `tau0`) and `locked_in_cpi` (the splice above; `act_sha256`, `rec_sha256`, `estimated_rows`). Every projection run therefore stays an ordinary run (D4, C17). A model point with no survivors at V keeps its previous `instalment_pm`: the specification is silent, and the choice is recorded as in-force open question 23, not decided here.
- **Helper metadata beyond the design (section 10).** `composed_curve(discount, ilp_table, horizon)` is added for the composed current curve `df_cur` (the para 120 curve and the in-force script's `p_cur`). Every helper's file also carries `halley.helper`, naming the helper its recomputation dispatches on; the curves and the CPI splice carry `halley.rebased_from` (the source discount or recognition CPI file), so that the provenance check recomputes them; `locked_in_curve`'s two files carry `halley.curve` (`p_rec` or `p_l`). The recomputation of an `OLD` model point file holds its `halley.tau` to the period's length, `mi(V) - mi(V_prev)` with V_prev the named run's valuation date.
- **The third locked-in hash is `cpi_rec` (in-force 3.3, 13).** The groups file's `cpi_rec_sha256` is the hash of the recognition CPI path `cpi_rec`, which is fixed for the group's life, where in-force 3.3 and 13 name the locked-in CPI file `cpi_locked`, which splices the actual history to V and changes every period; `cpi_locked` is verified through its `halley.rec_sha256` and its rebuild. See the engine note under in-force 3.3.
- **In-force stage (`InForceGroup`, M11).** Beside the bases `old`, `new` and `cur` and the scenario inputs `p_l`, `df_cur` and `p_rec`, the stage reads the table `cpi_act`, the actual CPI history at V (key `u` in -26..0, column `cpi`, metadata `estimated_rows`), which no formula reads: it is the source the loader rebuilds the old and new runs' rebased expense table and locked-in CPI path from, and as an input of the stage it is copied into the run directory, enters the run id and carries its C9 estimate flag. Its key is a dense grid, so the history holds exactly the months -26..0, and the expense rebase, which reads `I_act(-tau)`, serves periods of at most 26 months. The stage declares `rebuilds_locked_in`, so the loader holds its locked-in runs and files to what they were built from, before anything is written: every input checked against the group's locked-in hashes names its source and carries the group's `tau0`, so the provenance check rebuilds it at that month, and each curve is the one it is read as; every file built from the actual CPI history names the stage's `cpi_act`, and the cur runs' CPI path holds its rows for u <= 0; each policy is projected by one build in the old, new and cur bases; and the old basis is the previous run's (each old run's model points derived from a cur run of the previous aggregation run, on that run's build and non-financial tables or their rebase), the new runs on the cur runs' model point files. Its manifest's notes put `NOTE_PERIOD_RELEASE` in place of `NOTE_MONTHLY_STEP`: "The period close releases the adjusted CSM once for the reporting period, in proportion to the coverage units actually provided in it (cu_act, undiscounted) against those still to be provided at V (N(0)); one release per period differs from twelve monthly steps", where the monthly-step note says a period's release is the sum of its monthly releases. The manifest discloses the analysis of change under `analysis_of_change`, the labels of the stored scalars `aoc_a` to `aoc_e` (the five FCF values A to E of in-force 11.2) and `aoc_ba`, `aoc_cb`, `aoc_dc` (delta_FCF_L, bit for bit) and `aoc_ed` (the four differences).
