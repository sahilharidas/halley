# Halley term assurance: product specification (reference case 1)

Version 0.1, draft, 2026-10-14. Not yet reviewed.

## 1. Purpose and status

**What this document is.** The complete definition of a level term assurance model: inputs, basis, conventions, formulas, outputs and checks. It is reference case 1 in Halley's validation chain:

1. this written spec;
2. a spreadsheet built from the spec by hand, with no other source;
3. a reviewer (a finance actuary) checks the spreadsheet against the spec, line by line;
4. the Halley engine, running a model written from the same spec, reconciles to the spreadsheet.

The spreadsheet is the thing the engine is measured against, so the spreadsheet must be right, and it can only be right if this document is unambiguous. Every quantity below has exactly one formula. Every edge case is resolved in the text, not left to the builder. The three conventions the roadmap names as mandatory content (cash flow timing, decrement order, discounting) are in sections 8, 7 and 9.

**Status.** Version 0.1, dated 2026-10-14. Draft for review. Nothing in sections 2 to 16 is open for this product; the only items open for the reviewer are in section 17. Proposals this document makes for the Halley notes change no formula here and are listed in section 15.6. The basis is synthetic and formula-defined: it is nobody's experience, it reproduces no published table, and it is sized to be built into a spreadsheet inside the prototype time budget.

**Who reads it.**

- The spreadsheet builder, who transcribes sections 3 to 10 into cells and sections 11 to 13 into output and check sheets. The builder cannot ask questions; where the document is silent the builder stops and records a question for section 17 rather than choosing.
- The reviewer, who reads the whole document, objects to specific numbered items in section 16, and then checks the spreadsheet against sections 4.3, 5 and 6 to 13 cell by cell, without reading any engine code.
- The engine author, who writes the Halley `TermAssurance` class from section 15 and the reconciliation test from sections 11 and 13.

**"Built from the spec" means** that every cell formula in the spreadsheet is a transcription of a formula in this document, in the evaluation order given in section 7.1, using the reference model points of section 13 and the basis of section 6 and nothing else: no engine output, no other table, no rounding, no judgement. Where this document gives a computation form (a power rather than a running product, a sum in ascending `t`, a left-to-right subtraction), the spreadsheet uses that form. The illustrative numbers in section 14 may be used to spot mistakes but are not the target; the formulas are the target. If a cell cannot be built because the document is ambiguous, that is a defect in this document and goes to section 17, not into the spreadsheet as a private choice.

**Notation used throughout.**

```
t                 integer month index from the valuation date: t = 0, 1, 2, ...
                  "time t" is an instant; "month t" is the interval from time t-1 to time t
x^y               x raised to the power y, computed as a float64 power (Excel ^ or POWER, Python **)
floor(x)          largest integer <= x;  a // b is floor(a / b) for integers
min(a, b), max(a, b), exp(x), ln(x)      as usual, float64; ln is the natural logarithm
where(c, a, b)    a if condition c holds, otherwise b; the value of the unused branch is discarded
                  (the engine may still evaluate it, so a discarded branch may divide by zero
                  without affecting the result)
a = b, a <> b, a <= b    inside a condition: tests on exact values
x(t)              value of variable x in month t (month quantities) or at time t (state variables)
1/12              the float64 result of dividing 1 by 12 (0.08333333333333333)
```

All amounts are in one unnamed currency. Everything is float64; nothing is rounded at any step. Where a formula's result depends on the order of float64 operations, the order is the one written, left to right, with the parentheses shown. Variable and field names are in `snake_case` and are the names to use in the spreadsheet column headers and in the Halley model.

**Units used in this document.**

| Label | Meaning |
|---|---|
| `Money` | currency amount for the whole in-force count of the model point (in-force total) |
| `Money per policy` | currency amount per policy |
| `Count` | number of policies (real-valued) |
| `Prob` | probability in [0, 1]; annual or monthly as stated in the variable name (`_ann`, `_m`) |
| `Rate[Annual]`, `Rate[Monthly]` | interest or inflation rate per year or per month |
| `Fraction` | dimensionless proportion (commission rate) |
| `factor` | dimensionless multiplier (`infl`, `v`, `disc`) |
| `Int`, `Bool`, `Date` | integer, boolean, calendar date |

`Money`, `Count`, `Prob`, `Rate[Annual]`, `Rate[Monthly]` and the per-policy versus in-force distinction (written `MoneyPerPolicy` in section 15) are Halley v1 units (Spec unit table, D11). `Bool` and `Date` are types the Spec already has, as `Static[bool]`, `where` conditions and typed model point fields. `Fraction`, `factor` and `Int` are labels this document uses; whether they become Halley types is proposed in sections 15.4 and 15.6.

Every `Money` quantity below is an in-force total unless its name ends in `_pp` or its unit says per policy. In-force amounts are always `Count` times a per-policy amount, applied exactly once.

**Reconciliation tolerance (proposed; see section 17).** For every cell and scalar in section 11, engine and spreadsheet agree if

```
|engine - spreadsheet| <= 1e-9 * max(|engine|, |spreadsheet|, scale)
scale = sum_assured * pols_if_init   for Money (in-force) quantities
      = sum_assured                  for Money per policy quantities
      = pols_if_init                 for Count quantities
      = 1                            for Prob, Fraction, factor and integer quantities
```

For a portfolio total of section 11.3, `scale` is the sum over the model points in the file of the per-policy scale: the sum of `sum_assured * pols_if_init` for Money totals and the sum of `pols_if_init` for Count totals (3,970,000 and 11.5 for the reference points of section 13). Money per policy quantities are not totalled.

Cells that this document says are exactly 0 (sections 7.4 and 10.3, the statements marked "exactly" in section 12, and section 4.5 when the sheet carries rows beyond `rem_m`) must be exactly 0 in both. Both compute in float64; the only expected differences are last-bit differences in `exp` and power functions and in summation order. A difference larger than the tolerance is a defect in one of the two, never a convention difference, because the conventions are fixed here. The same tolerance applies to every invariant in section 12 that compares two computed values. Halley's cross-backend tolerance (1e-9 relative, D19) is a separate test.

**Location.** This document lives in the Halley repository at `docs/spec/term-assurance.md` (Apache 2.0) because the reconciliation test cites it; the design notes it depends on stay in the author's vault.

## 2. Product

**Level term assurance**, one policy (or a group of identical policies, see `pols_if_init`) per model point, on a single life, in force at the valuation date.

- **Benefit.** A level `sum_assured` per policy is paid once, on death during the term. The claim is paid at the end of the policy month in which death occurs, together with a claim expense borne by the insurer (section 6.5). Nothing else is ever paid to the policyholder.
- **Premium.** A level `premium` per policy per month, paid in advance at the start of each policy month for the premium term (`prem_term_y` whole years from the entry date). The premium term may be shorter than the policy term; after it the policy is paid-up and cover continues to the end of the term with no premium due. The premium is a model point input, as charged, and is not derived from the basis. The premium due at time 0 (the valuation date) is unpaid at the valuation instant and is projected as income in month 1. A policy that dies or lapses during a month has paid that month's premium; there is no refund.
- **On death.** The policy leaves the in-force count during the month; `sum_assured` and the claim expense are paid at the end of the month.
- **On lapse.** The policy leaves the in-force count at the end of the month. Nothing is paid. Lapses occur only while premiums are payable; after the premium term the lapse rate is zero (a paid-up policy has nothing to stop paying).
- **At the end of the term.** Cover ends at time `rem_m` (section 4.3), the end of the last policy month. Every policy still in force at that instant leaves as a "maturity" with no payment. The in-force count is zero from then on, every cash flow is zero, and the reserve at that instant is zero.
- **Insurer cash flows that are not policyholder benefits.** A per-policy maintenance expense each month and a per-claim expense, both inflating, and commission as a proportion of premium. These are basis items (section 6), not contract terms; they are in the reference case so that the reserve is a gross premium reserve.

**Excluded, deliberately.** No surrender value and no paid-up value: lapse pays nothing. No maturity or survival benefit. No riders and no waiver formulas. No reinsurance. No tax. No new business during the projection: every model point is in force at the valuation date (a model point whose entry date is after the valuation date is rejected at load) and nothing enters after `t = 0`. No acquisition expenses or initial commission beyond the policy-year-0 commission rate (a policy entered on the valuation date is treated as already issued with its acquisition costs already paid). No premium frequency other than monthly. No decreasing or indexed cover, no premium reviews. No interest on cash balances. No profit signature or profit testing; outputs are cash flows, present values and reserves. No portfolio-level quantity inside the projection of a policy (totals are formed after projection, section 11.3). No product variants or build-time switches in version 0.1.

## 3. Model point

One row per model point. Field names are Halley model point fields. The file is Parquet; column types are Arrow types. "Range" is validated at load; a row outside the range is a load error, never silently adjusted. Dates are inputs; ages and durations are derived (sections 4 and 5) and never appear in the file.

### 3.1 Input fields

| Field | Arrow type | Unit | Range or values | Meaning |
|---|---|---|---|---|
| `policy_id` | string | string | non-empty, unique within the file | Identifier, used only to label outputs. |
| `sex` | string (enum) | string | exactly `M` or `F` | Mortality table selector. The enum has exactly these two values; every lookup keyed on it must cover both. |
| `birth_date` | date32 | Date | such that `18 <= age_at_entry <= 65` | Date of birth of the life insured. |
| `entry_date` | date32 | Date | `entry_date <= valuation_date` | Date the policy came into force. The first premium was due on this date; policy anniversaries fall on its month and day. |
| `maturity_date` | date32 | Date | `maturity_date.day = entry_date.day`; `12 <= term_m <= 480`; `maturity_date > valuation_date` | Date on which cover ends. The term in months is derived from it. |
| `prem_term_y` | int16 | Int (years) | `1 <= prem_term_y` and `12 * prem_term_y <= term_m` | Premium-paying term in whole years from `entry_date`. |
| `sum_assured` | float64 | Money per policy | `0 < sum_assured <= 10 000 000` | Level death benefit per policy. |
| `premium` | float64 | Money per policy | `0 < premium <= 100 000` | Level monthly premium per policy, payable in advance. An input, not derived from the basis. |
| `pols_if_init` | float64 | Count | `0 < pols_if_init <= 1 000 000` | Policies in force at the valuation date, before any projection. Usually 1. A non-integer value represents a group of identical policies and is allowed. |

**Not model point fields.** There is no age field, no duration field and no term-in-months field: all are derived from the dates. There is no premium frequency field: premiums are monthly.

### 3.2 Derived fields

Computed once per model point at load, from the formulas of sections 4.3 and 5. All are integers.

| Derived field | Unit | Meaning | Range implied by the schema |
|---|---|---|---|
| `age_at_entry` | Int (years) | age last birthday on `entry_date` | [18, 65] |
| `dur_m` | Int (months) | completed policy months from `entry_date` to the valuation date | [0, `term_m` - 1] |
| `term_m` | Int (months) | policy term in months | [12, 480] |
| `rem_m` | Int (months) | policy months remaining after the valuation date; the policy's own projection horizon | [1, 480] |
| `prem_term_m` | Int (months) | premium-paying term in months from entry | [12, `term_m`] |
| `prem_rem_m` | Int (months) | premium-paying months remaining after the valuation date | [0, `rem_m`] |
| `dur_y_last` | Int (years) | completed policy years at the start of the final policy month | [0, 39] |

### 3.3 Cross-field rules, validated at load

```
18 <= age_at_entry <= 65
12 <= term_m <= 480
prem_term_m <= term_m
dur_m >= 0                 (entry on or before the valuation date)
rem_m >= 1                 (at least one month of cover left at the valuation date)
```

A model point that fails any rule is rejected at load with the rule named. In particular a policy already past its maturity date, or maturing exactly on the valuation date (`rem_m = 0`), is not projected: it is a load error for this reference case. Within these ranges the mortality age `age(t)` of section 5 lies in [18, 104] for every month of every policy, inside the table range [18, 120] of section 6.1, and the policy-year keys are clamped, so every lookup key of every valid model point is inside every table's declared range at any batch horizon. Every model point that passes these rules is a valid policy on which every invariant of section 12 holds; the property-based tests draw random points from exactly these ranges, which are rectangular in `sex`, `sum_assured`, `premium`, `pols_if_init`, and in `entry_date` and `maturity_date` taken against the run-level valuation date; the joint constraints are `age_at_entry` (`birth_date` with `entry_date`), `term_m` together with the same-day rule of 3.1 (`entry_date` with `maturity_date`) and `prem_term_m <= term_m` (`prem_term_y` with both dates), so the generator must draw the three dates and `prem_term_y` together.

**Run-level input.** `valuation_date` (a date) is one value per run, not a model point field, and is recorded in the run manifest. For version 0.1 it must be the first day of a calendar month (validated at load; section 4.1 says why). For the reference case it is 2027-01-01.

## 4. Time

### 4.1 The month index

`t` is an integer count of months from the valuation date. Time 0 is the valuation date. Time `t` is the calendar date `t` calendar months after the valuation date:

```
date(t) = the first day of the calendar month t months after valuation_date
```

Because the valuation date is the first of a month, `date(t)` is always the first of a month and no day-of-month clipping arises. For the reference valuation date 2027-01-01, time 1 is 2027-02-01, time 14 is 2028-03-01, time 480 is 2067-01-01.

Policy month `t` (for `t >= 1`) is the interval from time `t-1` to time `t`, open at the start and closed at the end: an event "at time t" belongs to month `t`, not month `t+1`. The start of month `t` is time `t-1`; the end of month `t` is time `t`. There is no month 0.

**State variables and month quantities.** Variables fall into two groups:

- **State at time `t`**, defined for `t = 0, 1, ..., H`: `pols_if(t)`, `infl(t)`, `disc(t)`, `reserve(t)`, `reserve_pp(t)`, `reserve_floored(t)`. At `t = 0` they take their initial or valuation-date value.
- **Quantities of month `t`**, defined for `t = 1, ..., H`: every other variable in sections 5 to 9 (lookups, rates, decrements, per-policy amounts, cash flows, `v`, `pv_net_cf_t`). At `t = 0` they have no meaning. The spreadsheet leaves them blank in row 0; the engine stores 0 for them at `t = 0`; every invariant and every reconciliation check excludes them at `t = 0`.

Each cash flow declares the instant within the month at which it is paid: `start` (time `t-1`) or `end` (time `t`). The `mid` timing permitted by Halley is not used in this product.

### 4.2 Whole months between two dates

One helper is used for every date difference. For dates `d1 <= d2`:

```
months_between(d1, d2) = 12 * (d2.year - d1.year) + (d2.month - d1.month) - where(d2.day < d1.day, 1, 0)
```

That is: count calendar months, then subtract one if the day of month of `d2` has not yet reached the day of month of `d1`. Any fraction of a month beyond the whole count is dropped. There is no month-end clause: the two uses below never need one, because the valuation date is the first of a month and `maturity_date` has the same day of month as `entry_date`. Consequence for `dur_m`: with `valuation_date.day = 1`, an entry on the 1st of any month counts that month as complete and an entry on any later day does not. A 29 February entry date (P07) follows the same rule with no special case.

### 4.3 Derived durations

All computed once per model point:

```
dur_m       = months_between(entry_date, valuation_date)        whole months in force at the valuation date
term_m      = months_between(entry_date, maturity_date)         policy term in months (the day term is 0 by the schema)
rem_m       = term_m - dur_m                                    months of cover remaining from the valuation date
prem_term_m = 12 * prem_term_y                                  premium-paying term in months from entry
prem_rem_m  = max(prem_term_m - dur_m, 0)                       premium-paying months remaining (0 if the premium term is over)
dur_y_last  = (term_m - 1) // 12                                completed policy years at the start of the final policy month
```

**Alignment rule for entry dates that do not fall on the first of a month.** The policy's own month boundaries fall on the entry day of each month, so the valuation date generally falls part way through a policy month. The rule is: the valuation date is treated as a policy month boundary. The policy is treated as if it had been in force for exactly `dur_m` whole months, and the days beyond that (up to one month less a day) are ignored. Projection month `t` is therefore policy month `dur_m + t` of the policy (counting policy months from 1), policy anniversaries fall at the end of projection month `t` whenever `(dur_m + t)` is a multiple of 12, and the policy matures at time `rem_m`. Consequences, stated so they are not mistaken for errors:

- the projected maturity instant `date(rem_m)` can differ from `maturity_date` by up to a month less a day (P06 in section 13 matures on 2036-11-10 but is projected to time 119 = 2036-12-01);
- the premium for policy month `dur_m + 1`, which was in fact due on the entry day before the valuation date, is counted as receivable at time 0; the error is at most one month's premium per policy;
- the cover for the elapsed fraction of the current policy month is counted again;
- a policy with `entry_date` equal to the valuation date has `dur_m = 0` and is in force from time 0 with its full term ahead of it.

In a real valuation these effects would be handled by exact-day durations or by an unearned premium adjustment. Neither is in version 0.1.

### 4.4 Projection horizon

Each policy is projected for `H = rem_m` months: rows `t = 0, 1, ..., rem_m`. The spreadsheet for a policy has exactly these rows. A batch run in the engine uses the largest `rem_m` in the batch as its horizon `H` and masks every policy beyond its own `rem_m` as described next. The reference points span `rem_m` from 1 to 480, so the reference batch horizon is 480.

### 4.5 After the end of the term

For `t > rem_m` every count, every Money variable and every reserve variable is exactly 0.0 (not blank, not NaN, not merely small), and the remaining variables take the values below. These rows are never needed by the spreadsheet; the rule exists so that the engine's common-horizon batch and a spreadsheet with extra rows give identical values, and so that the batch horizon changes none of a policy's results (P11).

| Variable | Value for `t > rem_m` | Why |
|---|---|---|
| `pols_if(t)`, `deaths(t)`, `lapses(t)`, `maturities(t)` | 0 | `pols_if(rem_m) = 0` exactly (section 7.4), so every later count is `0 * rate = 0` |
| `q_m(t)`, `w_ann(t)`, `w_m(t)` | 0 | masked by `where(t <= rem_m, ...)` and `where(t <= prem_rem_m, ...)`, and `prem_rem_m <= rem_m` |
| `premium_pp(t)`, `commission_pp(t)` | 0 | masked by `t <= prem_rem_m` |
| every in-force cash flow in section 8, `net_cf(t)`, `net_outgo(t)`, `pv_net_cf_t(t)` and the five `pv_*_t(t)` columns of section 9.3 | 0 | each is `pols_if(t-1)` or `deaths(t)` times an amount, or a sum of such |
| `reserve(t)`, `reserve_pp(t)`, `reserve_floored(t)` | 0 | terminal condition (section 10.3) |
| `dur_y(t)`, `age(t)`, `q_ann(t)` | held at their `t = rem_m` values | the clamp in section 5 stops the keys advancing, so every lookup stays inside its table |
| `comm_rate(t)` | continues by its formula (clamped key) | harmless: it multiplies a zero `premium_income` |
| `maint_pp(t)`, `claim_exp_pp(t)`, `infl(t)`, `v(t)`, `disc(t)` | continue by their formulas | pure functions of `t`; they multiply zero counts |

### 4.6 Edge cases resolved

| Case | Resolution |
|---|---|
| Brand new policy, `entry_date = valuation_date` | `dur_m = 0`, `rem_m = term_m`, `prem_rem_m = prem_term_m`. Policy year 0 applies in months 1 to 12 (20% commission). The first premium is projected at time 0. No acquisition expense. P02 in section 13. |
| Entry date after the valuation date | Load error (`dur_m < 0`): no new business in this reference case. |
| Policy past its maturity date, or maturing on the valuation date | Load error (`rem_m <= 0`). |
| One month of cover left (`rem_m = 1`) | One row of month quantities (t = 1); `maturities(1)` takes every survivor; `reserve(1) = 0`; `reserve(0)` is one month of net outgo discounted. P03. |
| First month of the projection (`t = 1`) | Driven by `pols_if(0) = pols_if_init`. Premium at time 0 is included if `prem_rem_m >= 1`. |
| Last month of the premium term (`t = prem_rem_m`) | Premium is paid at time `prem_rem_m - 1`; lapses apply in that month. P10 (`prem_rem_m = 1`): one premium at time 0, lapses in month 1 only. |
| Month after the premium term (`t = prem_rem_m + 1`) | `premium_income = 0`, `commission = 0`, `w_ann = w_m = 0`, `lapses = 0`. Deaths and maintenance expenses continue. P05 from `t = 3`. |
| Premium term already over at the valuation date (`prem_rem_m = 0`) | No premiums, no commission, no lapses in the projection. P08. |
| Policy anniversary inside the projection | `dur_y(t)` steps by 1 at the first month of the new policy year; `age`, the lapse key and the commission key step with it. P05 at `t = 3`. |
| Final month of the term (`t = rem_m`) | Deaths and their claims are paid; lapses apply only if `rem_m <= prem_rem_m`; then every survivor leaves as a maturity and `pols_if(rem_m) = 0` exactly. |
| Month after the end of the term (`t = rem_m + 1`) | All zeros (section 4.5). |

## 5. Age

**Convention: age last birthday at entry, advancing on each policy anniversary.** Age is computed once at entry from the dates, and thereafter increases by one at each policy anniversary. It does not advance on the calendar birthday and it is not recomputed from `birth_date` month by month. This keeps the mortality age constant across each policy year, which is what a table keyed on integer age assumes, and it is the convention under which an integer entry age plus completed policy years indexes an attained-age table (the `x(t-12)` idiom; also what lifelib's BasicTerm does).

```
age_at_entry = entry_date.year - birth_date.year
               - where((entry_date.month, entry_date.day) < (birth_date.month, birth_date.day), 1, 0)

dur_y(t)     = min((dur_m + t - 1) // 12, dur_y_last)        completed policy years at the start of month t, 0-based
age(t)       = age_at_entry + dur_y(t)                        mortality age for month t
```

The tuple comparison `(month, day) < (month, day)` is lexicographic: first month, then day; it is 0 when the entry date is the birthday itself (P08 entered on the life's 52nd birthday). A 29 February birth date compares as `(2, 29)`, so on 28 February of a non-leap year the birthday has not yet been reached. `dur_y(t)` is 0 throughout the first policy year, 1 throughout the second, and so on. The `min` with `dur_y_last` has no effect for `t <= rem_m`; it exists only to hold the keys at their final in-term values after the end of the term (section 4.5), so that an engine projecting past a policy's maturity never produces a lookup key outside a table's range. The lookup key for mortality is the pair `(sex, age(t))`; `dur_y(t)` is also the key for the lapse and commission tables (clamped further in sections 6.2 and 6.6).

Worked check: P05 has `birth_date` 1975-04-18, `entry_date` 2018-02-10. `age_at_entry = 2018 - 1975 - 1 = 42` because `(2, 10) < (4, 18)`. With `dur_m = 106`, `dur_y(1) = 106 // 12 = 8`, `dur_y(3) = 108 // 12 = 9`, so the mortality age is 50 in months 1 and 2 and 51 from month 3.

**Named simplification.** Age last birthday labels a life whose exact age is in `[x, x + 1)` as `x`, so on average it understates the attained age by half a year. On this basis that understates `q` by about 1.8% at age 20, 3.0% at 30, 3.9% at 40 and 4.4% at 50, and by 4.7% to 4.8% from 60 to 89, peaking near age 75. Above 89 it falls, to 4.3% (male) and 4.5% (female) at 100 and to 4.0% and 4.3% at 104, the highest mortality age of section 3.3, because `1 - exp(-mu)` flattens as `mu` grows. Up to age 99 the female figure is within 0.2 points of the male. The Makeham constant `A` dilutes the effect at young ages, and the Gompertz limit `c^(1/2) - 1 = 4.88%` is approached only while `q` is small. The table in section 6.1 is synthetic and is declared an age-last-birthday table; nothing is calibrated, so the label has no bias to correct. Section 17 asks whether the reviewer wants age nearest birthday instead.

## 6. Basis

Every value below is synthetic, chosen for this reference case, and defined by formula or by the short tables printed here. No published or proprietary table is reproduced. The basis is a single set of assumptions used for the projection and for the reserve (there is no separate valuation basis and no margin). Basis version identifier: `term-ref-basis-0.1`. In Halley each table below is a versioned assumption table with the declared key range (section 6.8); the formula-defined mortality table is materialised to rows before loading.

Constants used in several places (each written out once here and used by name everywhere else):

```
i    = 0.04                                   Rate[Annual], annual effective interest
e    = 0.03                                   Rate[Annual], annual effective expense inflation
i_m  = (1 + i)^(1/12) - 1                     Rate[Monthly]           = 0.0032737397821989145
v_m  = (1 + i)^(-1/12)                        one-month discount factor = 0.9967369426185624
e_m  = (1 + e)^(1/12) - 1                     Rate[Monthly]           = 0.0024662697723036864
```

`v_m` is computed directly as the power `(1 + i)^(-1/12)`, not as `1 / (1 + i_m)`; the two can differ in the last bit. `i / 12` and `e / 12` are not used anywhere. The product `v_m * (1 + i_m)` equals 1 algebraically and to within one unit in the last place in float64.

### 6.1 Mortality

Annual probability of death at integer age `x`, by sex, from a Gompertz-Makeham force of mortality `mu(x) = A + B * c^x` integrated exactly over the year of age from `x` to `x + 1`:

```
q_ann(sex, x) = 1 - exp(-(A[sex] + B[sex] * c^x * (c - 1) / ln(c)))
```

Evaluate in exactly the order written: `c^x` first, then `B * c^x`, then `* (c - 1)`, then `/ ln(c)`, then `A +`, then negate, `exp`, and `1 -`. The value 1.049205868725707 of `(c - 1) / ln(c)` is a check figure only; do not hold it in a cell. Each row computes `B * c^x`, multiplies by `(c - 1)`, then divides by `ln(c)`, so the quotient is never formed on its own. `(c - 1)` and `ln(c)` are computed from the `c` cell, as for every expression in this document: `(c - 1)` is the float64 difference 0.10000000000000009, never a typed 0.1. A typed 0.1 leaves the `(M, 50)` chain below unchanged at the 15 significant digits a spreadsheet displays, so check `q_ann(M, 70)` instead: 0.0141847561621227 with `(c - 1)` from the cell, 0.0141847561621226 with a typed 0.1. `q_ann(F, 53)` is a second check: 0.00195138413143703 against 0.00195138413143692.

| Parameter | `M` | `F` |
|---|---|---|
| `A` | 0.0002 | 0.00015 |
| `B` | 0.000017 | 0.000011 |
| `c` | 1.1 | 1.1 |

- Key: `(sex, x)` with `sex` in `{M, F}` and integer `x` from 18 to 120 inclusive. The schema rules in section 3 and the clamp in section 5 guarantee every `age(t)` is in this range for every `t`. A key outside it is a load error.
- Value unit: `Prob`, annual. The largest value in the table is `q_ann(M, 120) = 0.80867951`, below 1, so no cap is needed and none is applied; the run-time check that every `Prob` lies in [0, 1] stays active.
- Ultimate table: no select period, no smoker distinction, no improvement.
- Rate applied in month `t`, and its monthly conversion (constant force within the year of age):

```
q_ann(t) = mort_table[sex, age(t)]                                   Prob, annual
q_m(t)   = where(t <= rem_m, 1 - (1 - q_ann(t))^(1/12), 0)           Prob, monthly
```

Compute `q_ann` first, then `q_m` from `q_ann` as written. Do not substitute the algebraically equal `1 - exp(-mu / 12)`; the two differ in the last bits. Twelve successive monthly survivals `(1 - q_m)^12` reproduce the annual survival `1 - q_ann` up to float64. `q_ann / 12` is not used anywhere; on P01 it alone would lower the present value of claims by 0.10% (section 14.5).

Checkpoint values (8 significant figures, from the formulas above):

| `x` | `q_ann(M, x)` | `q_m(M, x)` | `q_ann(F, x)` | `q_m(F, x)` |
|---|---|---|---|---|
| 18 | 0.00029912472 | 0.000024930478 | 0.00021414554 | 0.000017847214 |
| 25 | 0.00039317592 | 0.000032770566 | 0.00027500839 | 0.000022920255 |
| 30 | 0.0005111056 | 0.000042602114 | 0.00035132644 | 0.000029281919 |
| 35 | 0.00070100329 | 0.000058435718 | 0.00047422518 | 0.000039527357 |
| 40 | 0.0010067596 | 0.000083935369 | 0.00067212307 | 0.000056027517 |
| 45 | 0.0014989865 | 0.00012500144 | 0.00099075723 | 0.000082600618 |
| 50 | 0.0022912131 | 0.00019113522 | 0.0015037072 | 0.00012539538 |
| 55 | 0.0035657808 | 0.00029763514 | 0.0023292646 | 0.00019431292 |
| 60 | 0.005615063 | 0.00046913048 | 0.0036573985 | 0.00030529532 |
| 65 | 0.0089065965 | 0.0007452636 | 0.0057926558 | 0.00048400769 |
| 70 | 0.014184756 | 0.0011898184 | 0.0092218937 | 0.00077175859 |
| 80 | 0.0360697 | 0.0030566763 | 0.023510351 | 0.00198063 |
| 90 | 0.090595898 | 0.0078825788 | 0.059617902 | 0.0051093194 |
| 100 | 0.21807612 | 0.020291131 | 0.14717055 | 0.013178698 |
| 104 | 0.30237439 | 0.029560345 | 0.2078534 | 0.019230096 |
| 110 | 0.47151266 | 0.05175721 | 0.33812169 | 0.033804867 |
| 120 | 0.80867951 | 0.12874198 | 0.65703603 | 0.085316789 |

Full-precision intermediate chain for `(M, 50)`, so the builder can locate a slip: `1.1^50 = 117.39085287969579`; `B * c^x = 0.0019956444989548284`; `* (c - 1) = 0.000199564449895483`; `/ ln(c) = 0.0020938419201935793`; `A + ... = 0.0022938419201935793`; `q_ann = 0.00229121307525193`; `q_m = 0.00019113522468450217`.

### 6.2 Lapse

Annual lapse rate by completed policy years, 0-based, applied only while premiums are payable:

| Key `k = min(dur_y(t), 5)` | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| `lapse_table[k]` (annual, `Prob`) | 0.15 | 0.10 | 0.08 | 0.06 | 0.05 | 0.04 |
| `1 - (1 - lapse_table[k])^(1/12)` (monthly, 10 s.f.) | 0.01345194701 | 0.008741610955 | 0.006924382628 | 0.005143012832 | 0.004265318778 | 0.003396053199 |

- Key range 0 to 5. The key is clamped with `min(dur_y(t), 5)` before the lookup, so row 5 means "policy year 5 and later" and the lookup can never leave the range. Value unit `Prob`, annual.
- Rate in month `t`:

```
w_ann(t) = where(t <= prem_rem_m, lapse_table[min(dur_y(t), 5)], 0)     Prob, annual
w_m(t)   = 1 - (1 - w_ann(t))^(1/12)                                     Prob, monthly
```

`w_m(t)` is exactly 0 when `w_ann(t)` is 0. Since `prem_rem_m <= rem_m`, both are zero beyond the term. The rate is independent of the mortality rate; the two are combined in section 7.

### 6.3 Interest

Flat annual effective rate `i = 0.04` (`Rate[Annual]`), the same for every policy and every `t`, and the same for discounting and for accumulating within the reserve recursion. `i_m` and `v_m` are as at the top of this section. In version 0.1 the rate is a basis constant, not a scenario file; the run has one scenario (D4) and the scenario dimension has length 1. Every formula uses `v(t)`, `v_m` and `i_m` only, so a one-row scenario file can replace the constant later without changing a formula.

### 6.4 Expense inflation

Annual rate `e = 0.03` (`Rate[Annual]`), converted to `e_m` as at the top of this section and compounded monthly from the valuation date. The inflation index is a function of the instant at which an amount is paid:

```
infl(s) = (1 + e_m)^s          inflation factor for a payment at time s; infl(0) = 1
```

Compute `e_m` first, then `1 + e_m`, then the power; not a running product. A start-of-month-`t` amount would be indexed by `infl(t-1)`; an end-of-month-`t` amount is indexed by `infl(t)`. Every expense in this product is end-timed (section 8), so every expense of month `t` uses `infl(t)`. The base date of the price level is the valuation date (time 0), not the entry date: both expense amounts in 6.5 are stated at time-0 prices, and month 1's expenses already carry one month of inflation. Checkpoints: `infl(1) = 1.0024662697723037`, `infl(12) = 1.0300000000000011`, `infl(14) = 1.0350867806921342`, `infl(480) = 3.262037791999209`.

### 6.5 Expenses

| Item | Constant | Amount at time-0 prices | Unit | Paid |
|---|---|---|---|---|
| Maintenance expense | `maint_base` | 5.00 per policy per month | Money per policy | at the end of each month of the term, for every policy in force at the start of that month, including months after the premium term |
| Claim expense | `claim_exp_base` | 250.00 per death | Money per policy (per death) | at the end of the month of death, with the claim |

Both inflate by `infl(t)` for a payment at time `t` (section 8 gives the per-policy amounts `maint_pp(t)` and `claim_exp_pp(t)`). No initial or acquisition expense, even for a policy entered on the valuation date.

### 6.6 Commission

Proportion of premium income, by completed policy years, paid at the same instant as the premium:

| Key `k = min(dur_y(t), 1)` | 0 | 1 |
|---|---|---|
| `comm_table[k]` (`Fraction`) | 0.20 | 0.02 |

```
comm_rate(t) = comm_table[min(dur_y(t), 1)]          Fraction
```

Row 1 means "policy year 1 and later". Commission is zero whenever premium income is zero because it is defined as a multiple of it (section 8). No clawback on lapse.

### 6.7 Not in the basis

Each absence has the stated consequence for a real valuation.

- **Acquisition expenses and initial commission beyond the year-0 rate.** A model point with `dur_m = 0` (P02) is treated as issued an instant before the valuation date with its acquisition costs already spent. Its reserve at `t = 0` therefore excludes new business strain; a pricing or profit test would need it.
- **A separate valuation basis with margins.** The reserve in section 10 is a best-estimate gross premium reserve on this single basis. A statutory reserve would use prudent assumptions and would be larger.
- **A yield curve.** Interest is flat. Under a curve `v(t+1)` becomes a per-month factor and `disc(t)` a product of them; every formula is written with `v(t)`, `v_m` and `disc(t)` so that the substitution is local.
- **Select mortality, mortality improvement, smoker status, rating loadings, occupation class.** None. Section 17 asks about a select period.
- **Lapse dependence on anything but policy year and premium status.** No dependence on premium size, age or economic conditions; no lapse after the premium term (by construction, section 6.2); no mass-lapse scenario.
- **Expense per premium other than commission, expense dependence on sum assured, expense on lapse or at maturity, investment expenses, expense overruns.** None.
- **Tax, reinsurance, cost of capital, interest on cash.** None.
- **Stochastic scenarios.** One deterministic scenario; scenario files are a later milestone.

### 6.8 Assumption table files

Each table is a versioned Parquet file whose hash enters the run manifest. At load, for every policy, every key the projection can generate (`age(t)` and the clamped `dur_y(t)` keys for `t = 1 .. H`) must lie in the declared range; a key outside the range is an error, never an extrapolation.

| File | Keys and declared ranges | Value column and unit | Version |
|---|---|---|---|
| `mort_table` | `sex` in {`M`, `F`}; `age` Int in [18, 120] | `q_ann`, `Prob` annual | `term-mort-0.1` |
| `lapse_table` | `dur_y` Int in [0, 5] (clamped key) | `w_ann`, `Prob` annual | `term-lapse-0.1` |
| `comm_table` | `dur_y` Int in [0, 1] (clamped key) | `comm_rate`, `Fraction` | `term-comm-0.1` |
| `rates` | `name` in {`i`, `e`} (two rows) | `rate`, `Rate[Annual]` | `term-rates-0.1` |
| `expenses` | `name` in {`maint_base`, `claim_exp_base`} (two rows) | `amount`, Money per policy | `term-expenses-0.1` |

## 7. Decrements

### 7.1 Evaluation order

A spreadsheet evaluates one policy as follows; the engine does the same per policy and the order of sections 7 to 10 follows it. Two builders following this order lay out identical sheets, and the left-to-right order inside each row is part of the definition where float64 results depend on it.

1. Per-policy constants from sections 4 and 5: `age_at_entry`, `dur_m`, `term_m`, `rem_m`, `prem_term_m`, `prem_rem_m`, `dur_y_last`. The basis block, in the order the section 11 layout gives, each constant in one cell.
2. Row `t = 0`: `pols_if(0) = pols_if_init`, `infl(0) = 1`, `disc(0) = 1`; every month quantity blank.
3. Rows `t = 1 .. rem_m`, each row from its own `t` and from row `t-1` only, columns left to right: `dur_y`, `age`, `q_ann`, `q_m`, `w_ann`, `w_m`, `comm_rate`, `infl`, `deaths`, `lapses`, `maturities`, `pols_if`, `premium_pp`, `commission_pp`, `maint_pp`, `claim_exp_pp`, `premium_income`, `commission`, `maint_expense`, `death_outgo`, `claim_expense`, `net_cf`, `net_outgo`, `v`, `disc`, `pv_net_cf_t`, `pv_premiums_t`, `pv_commission_t`, `pv_maint_expense_t`, `pv_death_outgo_t`, `pv_claim_expense_t`.
4. The reserve column, from row `rem_m` upwards: `reserve(rem_m) = 0`, then `reserve(t)` from row `t+1`. Then `reserve_pp`, `reserve_floored` for every row `0 .. rem_m`.
5. Per-policy scalars (section 11.2) as sums down the columns in ascending `t`, then the check cells of section 12 (row booleans, scalar booleans and `<id>_ok`).

### 7.2 Order of decrements within a month

Within month `t`, deaths are taken first from the policies in force at the start of the month, then lapses from the survivors of death, then (in the final month only) every remaining policy leaves as a maturity. This is a calculation convention, not a statement about when lapses happen in reality. Because the lapse rate is applied to survivors of death, `deaths(t) + lapses(t) <= pols_if(t-1)` always holds for rates in [0, 1].

**Independent and dependent rates.** `q_ann` and `w_ann` are independent (single-decrement) rates. The order above converts them to dependent monthly rates without any further formula: the dependent monthly death rate is `q_m(t)` and the dependent monthly lapse rate is `(1 - q_m(t)) * w_m(t)`. The total monthly exit probability is `1 - (1 - q_m(t)) * (1 - w_m(t))`, below 1 for any rates in [0, 1). The alternative, constant forces of both decrements acting together within the month, gives a dependent death probability of `(mu_d / (mu_d + mu_w)) * (1 - (1 - q_m) * (1 - w_m))`, where `mu_d = -ln(1 - q_m(t))` and `mu_w = -ln(1 - w_m(t))` are the constant forces that reproduce `q_m` and `w_m` on their own (annual forces give the same ratio). That probability is lower than `q_m` by about `q_m * w_m / 2`. Total exits are the same under both models, so deaths-first moves that amount from lapse to death and slightly overstates deaths, claims and reserves. The difference is at most 6.1e-7 per month on P01, at most 1.2e-6 per month on any reference point (P09, final policy year), at most 5.1e-6 per month for a new entrant in policy year 0, and at most 5.1e-5 per month for any model point the schema admits (male, mortality age 104, premiums still payable, lapse key 5). Relative to `q_m` it is about `w_m / 2`: 0.17% where the lapse key is 5, and at most 0.67% in policy year 0, where the lapse rate is highest. Section 14.5 gives the script-produced figures. It is not modelled. Drawing both decrements from the opening count (so that deaths and lapses overlap) is not used either, because at high rates it can exit more than the count.

### 7.3 Formulas

For `t = 1 .. H`:

```
deaths(t)     = pols_if(t-1) * q_m(t)                                        Count x Prob -> Count
lapses(t)     = (pols_if(t-1) - deaths(t)) * w_m(t)                          Count x Prob -> Count
maturities(t) = where(t = rem_m, pols_if(t-1) - deaths(t) - lapses(t), 0)   Count
```

with `q_m(t)` from section 6.1 and `w_m(t)` from section 6.2. All three are Count quantities of month `t`. Deaths are spread through the month and paid at its end; lapses and maturities happen at the end of the month. Deaths and lapses in the maturity month are real decrements with their usual cash flows; only the survivors mature.

### 7.4 In-force recursion

```
pols_if(0) = pols_if_init
pols_if(t) = pols_if(t-1) - deaths(t) - lapses(t) - maturities(t)           for t >= 1, evaluated left to right
```

`pols_if(t)` is the number of policies in force at time `t`, after month `t`'s decrements and before anything that happens in month `t+1`. It is the count that pays the premium due at time `t` and the count to which `reserve(t)` applies. `pols_if(t-1)` is the opening count of month `t`; the spreadsheet may carry it as a helper column.

At `t = rem_m` the subtraction is `((a - b) - c) - ((a - b) - c)` with both sides computed by the same left-to-right expression, so `pols_if(rem_m)` is exactly `0.0`, not merely small. The spreadsheet must compute `maturities` and `pols_if` as written (not, for example, `pols_if(t) = where(t = rem_m, 0, ...)`) so that the engine and the spreadsheet agree bit for bit on this row. For `t > rem_m`, `pols_if(t-1) = 0` so every count is 0 with no further rule. The number of policies surviving to the end of the term is reported as `maturities(rem_m)`, not as `pols_if(rem_m)`.

### 7.5 Conversion of annual to monthly probabilities

Stated once more because it is the single most common silent difference between models:

```
q_m = 1 - (1 - q_ann)^(1/12)
w_m = 1 - (1 - w_ann)^(1/12)
```

Twelve successive monthly survivals reproduce the annual survival exactly (up to float64), whichever month of the policy year is being projected. `q_ann / 12` and `w_ann / 12` are not used. The exponent is the float64 value of `1/12`.

## 8. Cash flows

Every cash flow has a per-policy form and an in-force form. Per-policy amounts are what one policy pays or receives in month `t`; in-force amounts are per-policy amounts multiplied by the relevant count, exactly once. Only in-force amounts are discounted, summed and fed to the reserve. The spreadsheet shows both. Sign convention: each variable is a positive amount; `net_cf` and `net_outgo` carry the signs.

### 8.1 Per-policy amounts (Money per policy, month `t`)

```
premium_pp(t)    = where(t <= prem_rem_m, premium, 0)          premium due at the start of month t
commission_pp(t) = premium_pp(t) * comm_rate(t)                 commission on that premium
maint_pp(t)      = maint_base * infl(t)                         maintenance expense, inflated to the end of month t
claim_exp_pp(t)  = claim_exp_base * infl(t)                     claim expense per death, inflated to the end of month t
```

`prem_rem_m <= rem_m` always, so `premium_pp(t) = 0` for `t > rem_m` without a separate rule. `maint_pp` and `claim_exp_pp` are pure functions of `t` and are not masked; they multiply counts that are zero after the term.

### 8.2 In-force cash flows (Money, total for the model point, month `t`)

| Variable | Unit | Timing | Formula | Zero when |
|---|---|---|---|---|
| `premium_income(t)` | Money | start (time `t-1`) | `pols_if(t-1) * premium_pp(t)` | `t > prem_rem_m` |
| `commission(t)` | Money | start (time `t-1`) | `premium_income(t) * comm_rate(t)` | whenever `premium_income(t) = 0` |
| `maint_expense(t)` | Money | end (time `t`) | `pols_if(t-1) * maint_pp(t)` | `t > rem_m` |
| `death_outgo(t)` | Money | end (time `t`) | `deaths(t) * sum_assured` | whenever `deaths(t) = 0`, in particular `t > rem_m` |
| `claim_expense(t)` | Money | end (time `t`) | `deaths(t) * claim_exp_pp(t)` | whenever `deaths(t) = 0` |

Every in-force formula is `Count x Money per policy -> Money`, and `commission` is `Money x Fraction -> Money`. No in-force amount is multiplied by a count a second time. In each product the per-policy amount is formed first (for example `maint_pp(t) = maint_base * infl(t)`) and then multiplied by the count; this order is part of the definition.

### 8.3 Derived cash flow aggregates (Money, month `t`)

```
net_cf(t)    = premium_income(t) - commission(t) - maint_expense(t) - death_outgo(t) - claim_expense(t)
net_outgo(t) = death_outgo(t) + claim_expense(t) + maint_expense(t) - (premium_income(t) - commission(t)) * (1 + i_m)
```

- `net_cf(t)` is the undiscounted insurer-view net cash flow of month `t` (positive = net income). It mixes start and end timings, so it is a reporting quantity only and is never discounted as a single amount; its present value is defined from its components in section 9.3.
- `net_outgo(t)` is month `t`'s net outgo **accumulated to the end of month `t`**: the start-of-month income net of commission is rolled forward one month at `i_m`, and the end-of-month outgo is added. It is the quantity the reserve recursion in section 10 consumes, and it is the `net_outgo` of the Halley authoring example.

**Timing rules restated.** Premium and commission of month `t` occur at time `t-1` (the start of month `t`) and are computed on the opening count `pols_if(t-1)`. Maintenance expense, death benefit and claim expense of month `t` occur at time `t` (the end of month `t`); maintenance is charged for every policy in force at the start of the month, including those that die or lapse during it. Month 1's premium is received at time 0, the valuation date, by `pols_if_init` policies; it is a future cash flow of the valuation. The last premium is paid at time `prem_rem_m - 1`; the last possible claim at time `rem_m`. Nothing is paid at or after maturity other than the flows of the maturity month itself. No cash flow depends on the reserve, and nothing here reads any variable at `t+1`.

## 9. Discounting and present values

### 9.1 Discount factors

```
v(t)    = v_m                     one-month discount factor for month t (constant: flat interest)
disc(t) = v_m^t                   discount factor from time t to time 0; disc(0) = 1
```

`disc(t)` is a direct power with `v_m` held in one cell (`POWER(v_m, t)` or `v_m ^ t`), not a running product `disc(t-1) * v_m`; the two differ in the last bits. Checkpoints: `disc(1) = 0.9967369426185624`, `disc(12) = 0.961538461538462`, `disc(14) = 0.9552735892121151`, `disc(480) = 0.20828904466294507`.

### 9.2 Discount factor for each timing

| Timing of a month-`t` flow | Instant | Factor to time 0 |
|---|---|---|
| start | time `t-1` | `disc(t-1)` |
| end | time `t` | `disc(t)` |

A `mid` flow would use `v_m^(t - 1/2)`; none exists in this product. Halley's discounting helper reads the declared timing and applies exactly this rule.

### 9.3 Present values at the valuation date

Per-month discounted net cash flow (`Money`, month `t`):

```
pv_net_cf_t(t) = (premium_income(t) - commission(t)) * disc(t-1)
                 - (maint_expense(t) + death_outgo(t) + claim_expense(t)) * disc(t)
```

Per-month discounted components (`Money`, month `t`), one column each:

```
pv_premiums_t(t)      = premium_income(t) * disc(t-1)
pv_commission_t(t)    = commission(t)     * disc(t-1)
pv_maint_expense_t(t) = maint_expense(t)  * disc(t)
pv_death_outgo_t(t)   = death_outgo(t)    * disc(t)
pv_claim_expense_t(t) = claim_expense(t)  * disc(t)
```

Per-policy present values: each is the sum of its column over `t = 1, 2, ..., rem_m` in ascending `t`, accumulating left to right in one float64 running total per quantity (fixed order, so reruns are bit-identical). Extending any sum beyond `rem_m` adds exact zeros (section 4.5), so a common batch horizon gives the same values.

```
pv_premiums      = sum_t pv_premiums_t(t)
pv_commission    = sum_t pv_commission_t(t)
pv_maint_expense = sum_t pv_maint_expense_t(t)
pv_death_outgo   = sum_t pv_death_outgo_t(t)
pv_claim_expense = sum_t pv_claim_expense_t(t)
pv_net_cf        = pv_premiums - pv_commission - pv_maint_expense - pv_death_outgo - pv_claim_expense
```

All are `Money` in-force totals at the valuation date. `pv_net_cf` is the present value of future profit on this basis (positive = profit). It is defined from the component PVs, not as a sum of discounted `net_cf(t)`; it also equals `sum_t pv_net_cf_t(t)` within tolerance (invariant P9). The engine's reductions sum in ascending `t` in a fixed order (D19); a spreadsheet SUM may differ in the last bits, which the tolerance in section 1 covers.

**Named simplification.** Claims are discounted from the end of the month of death. Deaths occur on average mid-month and claims are settled some weeks later, so end of month is a reasonable proxy; relative to mid-month discounting it understates `pv_death_outgo` by a factor `v_m^(1/2)`, about 0.16% at 4% (section 14.5). A mid-month convention is available in Halley (`timing="mid"`) but is not used here, to keep the reserve recursion and the roll-forward on two instants only.

## 10. Reserve

### 10.1 Definition

`reserve(t)` is the **prospective gross premium reserve** at time `t`, on the basis of section 6, for the `pols_if(t)` policies in force at time `t`, taken immediately **before** the premium due at time `t`: the present value at time `t` of all outgo (claims, claim expenses, maintenance expenses, commission) in months `t+1, t+2, ..., rem_m` minus the present value of all premium income in those months, with each item discounted from its own instant, allowing for the probabilities of death and lapse of section 7. Unit: Money, in-force total. It may be negative (section 10.5).

### 10.2 Backward recursion

The reserve is computed by backward recursion from the end of the term, one month at a time. In month `t+1`, premium less commission comes in at the start (time `t`) and is accumulated to time `t+1`; claims, claim expenses and maintenance expenses go out at the end (time `t+1`); the reserve needed at time `t+1` is `reserve(t+1)`. Discounting the total from time `t+1` to time `t` gives:

```
reserve(rem_m) = 0
reserve(t)     = (reserve(t+1) + net_outgo(t+1)) * v(t+1)         for t = rem_m - 1, rem_m - 2, ..., 0
```

with `net_outgo` from section 8.3 and `v(t+1) = v_m`. Evaluate in exactly the order written: the sum first, then the multiplication. Written out, using `v_m * (1 + i_m) = 1`:

```
reserve(t) = v_m * ( reserve(t+1) + death_outgo(t+1) + claim_expense(t+1) + maint_expense(t+1) )
             - ( premium_income(t+1) - commission(t+1) )
```

The two forms are algebraically identical; the spreadsheet and the engine compute the first (two cell references and one multiplication per row) so that their rounding matches the Halley authoring example's shape exactly. In Halley terms `reserve` is a backward-recursive variable with `terminal = 0` whose right-hand side reads only itself at `t+1` and two forward variables at `t+1` (`net_outgo` and `v`, the latter a constant-valued `@var` in version 0.1).

### 10.3 Terminal condition and after the term

`reserve(rem_m) = 0` because cover ends at time `rem_m` with nothing payable. In a batch projected to a common horizon `H >= rem_m` the declared terminal value is `reserve(H) = 0`; because every flow of months `t > rem_m` is zero, the recursion gives `reserve(t) = 0` exactly for every `t >= rem_m`, so each policy also reaches zero at its own maturity instant. In the spreadsheet, where `H = rem_m`, the two statements coincide.

### 10.4 Total and per-policy

```
reserve_pp(t) = where(pols_if(t) > 0, reserve(t) / pols_if(t), 0)       Money per policy in force at time t
```

`reserve(t)` is the headline in-force total and `reserve(0)` is the valuation result for the model point. `reserve_pp(t)` is reported for the reviewer's intuition and equals the per-policy prospective reserve of textbook form (section 12, invariant P5); it is zero at and after maturity because `pols_if` is zero there. It is paired with the end-of-month count `pols_if(t)`; with that pairing `pols_if(t) * reserve_pp(t) = reserve(t)`, so P5 is the textbook per-policy recursion and the roll-forward of section 10.6 has the per-policy reading given there.

**Why the in-force total is the primary variable.** The total recursion reads only forward variables already computed (`net_outgo` and the constant-valued `v`); it has no decrement logic of its own, so the forward and backward components stay cleanly separated and the roll-forward invariant compares two routes through the same numbers. A per-policy recursion would need the survival probability `(1 - q_m)(1 - w_m)` inside the backward component, duplicating section 7 in a second place where a formula change could silently diverge. The per-policy recursion is kept as a check (P5), not as the definition.

### 10.5 Negative reserves: no floor

The reserve is **not floored at zero** anywhere: not inside the recursion, not on output. A negative `reserve(t)` means that, on this basis, the future premiums net of commission exceed the future outgo for the policies in force at `t`. This happens for young and recently issued policies whose level premium front-loads the cost of cover and whose premium carries a loading over the basis cost (P01, P02, P06 and P09 in section 14.4 all have negative `reserve(0)`; P01 stays negative for 32 months, P02 for 59, P09 for 50, P06 for its whole term).

For presentation only, a floored figure is also output:

```
reserve_floored(t) = where(reserve(t) > 0, reserve(t), 0)                Money
```

Nothing in the model reads it.

Consequences, stated explicitly:

- A negative reserve treats the policy as an asset: it relies on the policyholder continuing to pay premiums that exceed the cost of cover, and the policyholder can lapse at any time at no cost. A valuation that recognises negative reserves on term assurance is taking credit for future profit that only persistency delivers.
- The roll-forward and PV invariants of section 12 hold as written for the unfloored `reserve(t)`. Flooring per policy inside the recursion (`reserve(t) = max(0, ...)`) is a different reserve: it changes every earlier `reserve(t)`, breaks the roll-forward (a floored reserve releases more than the recursion rolls forward) and breaks the PV identity. It is not used in version 0.1; if it is wanted later it is a build-time variant with its own invariants.
- Flooring at output only (`reserve_floored`) preserves both identities for the unfloored variable and changes no other number.
- The portfolio total in section 11.3 is the plain sum, so negative and positive policy reserves offset. An aggregate floor (total reserve across a portfolio at or above zero, as some regimes required when no surrender value exists) is a portfolio quantity and belongs in the aggregation stage, never in a policy's projection (D5, D6). Nothing in this document computes it; `reserve_floored` is not summed across policies as a reserve.
- Under IFRS 17 the fulfilment cash flows of a profitable group are negative and the contractual service margin absorbs them; under Solvency II a negative best estimate liability is permitted within the contract boundary. Both start from the unfloored value, which is another reason the primary variable is not floored.

### 10.6 Roll-forward identity

For every `t = 1 .. rem_m`:

```
(reserve(t-1) + premium_income(t) - commission(t)) * (1 + i_m)
    - maint_expense(t) - death_outgo(t) - claim_expense(t) - reserve(t)  = 0
```

Derivation: the recursion at `t-1` is `reserve(t-1) = (reserve(t) + net_outgo(t)) * v_m`. Multiply by `(1 + i_m)` and use `v_m * (1 + i_m) = 1`: `reserve(t-1) * (1 + i_m) = reserve(t) + death_outgo(t) + claim_expense(t) + maint_expense(t) - (premium_income(t) - commission(t)) * (1 + i_m)`. Rearranging gives the identity. The per-policy reading is the one a finance actuary expects: each policy in force at time `t-1` pays its net premium, the fund earns a month's interest, every such policy incurs the month's maintenance expense, the deaths of the month take `sum_assured` plus claim expense, and the survivors need `reserve_pp(t)` each. Lapses do not appear because a lapsing policy releases exactly the reserve it carried and pays nothing; the release is already inside `reserve(t-1)` through the survival probability. The identity holds algebraically; in float64, `v_m * (1 + i_m)` is exactly 1.0 on this basis (the script of section 14 prints the check), so the residual comes only from rounding in the roll-forward's own additions and multiplications. It is of the order of the unit roundoff times the largest term, and below 4e-12 on every reference policy (section 14). The identity is one between in-force totals and holds for any definition of `reserve_pp`; the pairing of `reserve(t)`, taken before the premium, with the end-of-month count `pols_if(t)` is what gives the identity as written its per-policy reading and makes `reserve(0) = -pv_net_cf` with the time-0 premium counted as future income. A reserve taken after the premium satisfies a shifted identity (the net premium of month `t+1` added after the interest), not this one.

### 10.7 What this reserve is not

- Not a statutory or prudent reserve: single best-estimate basis, no margins, no floor, no acquisition expense.
- Not a net premium reserve: the actual office premium and actual expenses are used.
- Not an IFRS 17 or Solvency II liability: no risk adjustment, no risk margin, no contract boundary logic, no discount curve.
- Not a reserve that feeds any cash flow: no interest on reserves, no reserve-based surrender value, no capital charge (D13: a cash flow that reads `reserve` at `t-1`, such as interest on the opening reserve, would form a mixed forward/backward component and fail the build; one that reads `reserve(t)` or `reserve_pp(t)` would join the backward component and build, which is a different design, not a checker error).
- Not exact for the partial month at the valuation date (section 4.3).

## 11. Outputs

All outputs carry the dimensions `policy`, `scenario` (length 1 in version 0.1) and, for monthly outputs, `time`.

**Spreadsheet layout (fixed).** One sheet per policy, named by `policy_id`, with these blocks in this order: the inputs of 3.1; the derived constants of 3.2 and 4.3; the basis constants `i`, `e`, `i_m`, `v_m`, `e_m`, the mortality parameters `A` and `B` by sex and `c` (6.1), `maint_base` and `claim_exp_base` (6.5), and the lapse and commission tables of 6.2 and 6.6; the monthly block of 7.1 and 11.1 with rows `t = 0 .. rem_m`; the scalars of 11.2; the check cells of section 12. `q_ann` is either computed inline per row from the formula of 6.1 or looked up from a materialised `(sex, age, q_ann)` block for ages 18 to 120, both built from the formula of 6.1 in the stated order; the builder states which. A summary sheet holds the 11.2 scalars per policy, the totals of 11.3 and the check cells named there.

### 11.1 Per policy, per month (rows `t = 0 .. rem_m`; engine: `t = 0 .. H`)

In this column order (the evaluation order of section 7.1), each labelled with its unit. "Engine" marks the columns the engine stores as outputs; the others are spreadsheet columns the reviewer follows formula by formula and remain available in the engine through trace and explain but need not be stored.

| Column | Unit | State or month | Defined in | Engine |
|---|---|---|---|---|
| `t` | Int | | 4.1 | key |
| `dur_y`, `age` | Int | month | 5 | |
| `q_ann`, `q_m` | Prob annual, Prob monthly | month | 6.1 | |
| `w_ann`, `w_m` | Prob annual, Prob monthly | month | 6.2 | |
| `comm_rate` | Fraction | month | 6.6 | |
| `infl` | factor | state (`infl(0) = 1`) | 6.4 | |
| `deaths`, `lapses`, `maturities` | Count (in month `t`) | month | 7.3 | yes |
| `pols_if` | Count (at time `t`) | state | 7.4 | yes |
| `premium_pp` | Money per policy | month | 8.1 | |
| `commission_pp` | Money per policy | month | 8.1 | yes (stored so that it stays in the generated code: no other variable reads it, and the checker removes unused variables (15.3), which would leave P5, P7 and trace nothing to read) |
| `maint_pp`, `claim_exp_pp` | Money per policy | month | 8.1 | |
| `premium_income`, `commission` | Money, start | month | 8.2 | yes |
| `maint_expense`, `death_outgo`, `claim_expense` | Money, end | month | 8.2 | yes |
| `net_cf` | Money, reporting (no timing) | month | 8.3 | yes |
| `net_outgo` | Money, end | month | 8.3 | |
| `v` | factor | month | 9.1 | |
| `disc` | factor | state (`disc(0) = 1`) | 9.1 | |
| `pv_net_cf_t` | Money | month | 9.3 | yes |
| `pv_premiums_t`, `pv_commission_t`, `pv_maint_expense_t`, `pv_death_outgo_t`, `pv_claim_expense_t` | Money | month | 9.3 | |
| `reserve` | Money, at time `t` | state | 10.2 | yes |
| `reserve_pp` | Money per policy, at time `t` | state | 10.4 | yes |
| `reserve_floored` | Money, at time `t` | state | 10.5 | yes |

Row `t = 0`: state columns take `pols_if(0) = pols_if_init`, `infl(0) = 1`, `disc(0) = 1`, `reserve(0)` from the recursion, `reserve_pp(0)` and `reserve_floored(0)` from it; every month column is blank (spreadsheet) or 0 (engine) and is excluded from every check.

### 11.2 Per policy, scalars

| Scalar | Unit | Definition |
|---|---|---|
| `age_at_entry`, `dur_m`, `term_m`, `rem_m`, `prem_term_m`, `prem_rem_m`, `dur_y_last` | Int | sections 4.3, 5 |
| `pv_premiums`, `pv_commission`, `pv_maint_expense`, `pv_death_outgo`, `pv_claim_expense`, `pv_net_cf` | Money | 9.3 |
| `reserve_0` | Money | `reserve(0)` |
| `reserve_pp_0` | Money per policy | `reserve_pp(0)` |
| `reserve_floored_0` | Money | `reserve_floored(0)` |
| `maturities_n` | Count | `maturities(rem_m)`, the survivors to the end of the term |

### 11.3 Portfolio totals

Formed after projection, outside the policy loop (D5, D6): the sum over model points of each Money scalar in 11.2, summed in ascending `policy_id` order, and the sum over model points of each monthly Money and Count column in 11.1 by `t`, reading a policy's values beyond its own horizon as zero:

```
total_<x>      = sum over policies p in ascending policy_id of <x>[p]
                 for x in pv_premiums, pv_commission, pv_maint_expense, pv_death_outgo,
                          pv_claim_expense, pv_net_cf, reserve_0, reserve_floored_0
total_<col>(t) = sum over policies p in ascending policy_id of where(t <= rem_m[p], col[p](t), 0)
                 for t = 0 .. H and for each Money and Count column of 11.1
```

`H` is the batch horizon of section 4.4, the largest `rem_m` in the file (480 for the reference points). State columns (`pols_if`, `reserve`, `reserve_floored`) are totalled at `t = 0` as well; month columns are not (blank there, section 11.1). `reserve_pp` and `commission_pp` are Money per policy and are not summed. Each total is compared under the tolerance of section 1 with the portfolio scale defined there. In the spreadsheet these live on a summary sheet as plain SUMs across the policy sheets; the summary sheet also holds `<id>_ok` for every (sheet) invariant of section 12 and every policy, and one `all_ok` cell that is TRUE when all of them are TRUE. In Halley the totals are not part of the policy projection (a policy's results never depend on another policy); they are a post-processing sum and are listed here so the reconciliation test has a one-line headline to compare. The summed `reserve_floored_0` is a presentation total, not an aggregate-floored reserve (section 10.5).

### 11.4 What a later IFRS 17 stage needs

The monthly columns keep premiums, commission, each expense type and claims separate, with their timings, which is what a fulfilment cash flow roll-forward consumes. Nothing more is defined here.

## 12. Invariants

Each statement is checked for every policy and every `t` in the stated range. "Equals" and "within tolerance" for float64 quantities mean the rule of section 1: `|a - b| <= 1e-9 * max(|a|, |b|, scale)` with the scale of the quantity's unit. An inequality `a <= b` between float64 quantities means `a <= b + 1e-9 * scale`; this matters only at `t = rem_m`, where `deaths + lapses + maturities` can exceed `pols_if(t-1)` by one unit in the last place (`b + c + ((a - b) - c)` is not exactly `a` in float64 even though `((a - b) - c) - ((a - b) - c)` is exactly 0). "Exactly" means bit for bit, no tolerance. Month quantities are checked for `t = 1 .. H`, state quantities for `t = 0 .. H`. H1 to H5 specialise Halley's five built-in invariants to this product. The Halley notes do not yet say how the engine identifies `pols_if`, the decrements, `reserve`, `i_m` and the timed flows in a user model (section 15.6). Until they do, the reconciliation test computes H1 to H5 in the forms given here. Every statement that reads a column 11.1 does not store is computed through trace: H2 and H5b (the rates and `disc`), and P2, P3, P4, P5, P7 and P10 (the rates, `age`, `dur_y`, `net_outgo`, the per-policy amounts and the `pv_*_t` columns). P1 to P11 are checked by the reconciliation test of sections 11 and 13 and by the property-based tests (the Halley notes define no product-level invariant hook yet; if one is added they move there). Statements marked **(sheet)** are also check cells the spreadsheet must contain; the unmarked statements may be checked in the spreadsheet. A check cell is a boolean. For a statement over a range of `t` there is one cell per row in that range, TRUE when the statement holds for that row under the rule above (exactly, or within tolerance, as the statement says; inequalities use `a <= b + 1e-9 * scale`). For a scalar statement (H3, H5a, P8, P9) there is one cell. Each (sheet) invariant has one per-policy cell `<id>_ok`, TRUE when every row cell and scalar cell of that invariant is TRUE; these cells sit after the scalars of 11.2 on the policy sheet (7.1 step 5) and are repeated on the summary sheet (11.3). The scale of section 1 is taken from the policy's own `sum_assured` and `pols_if_init` input cells. Helper columns such as a residual `lhs - rhs` beside each boolean are layout, not definition: they are the builder's choice, are not outputs and are not reconciled.

**Halley built-in invariants, specialised.**

- **H1 (sheet)** In force never negative, never increasing: `pols_if(t) >= 0` and, for `t >= 1`, `pols_if(t) <= pols_if(t-1)`, exactly. There is no new business, so the count never rises.
- **H2 (sheet)** Decrement probabilities in range and decrements bounded: `0 <= q_ann(t) <= 1`, `0 <= q_m(t) <= 1`, `0 <= w_ann(t) <= 1`, `0 <= w_m(t) <= 1` exactly, and `deaths(t) + lapses(t) + maturities(t) <= pols_if(t-1)` within tolerance for `t >= 1` (equality in month `rem_m`).
- **H3 (sheet)** Terminal value reached: `reserve(rem_m) = 0` exactly, `reserve(H) = 0` exactly, and `pols_if(rem_m) = 0` exactly.
- **H4 (sheet)** Reserve roll-forward reconciles every month, for `t = 1 .. H`:

```
(reserve(t-1) + premium_income(t) - commission(t)) * (1 + i_m)
    - maint_expense(t) - death_outgo(t) - claim_expense(t)   equals   reserve(t)
```

- **H5a (sheet)** PV by recursion equals PV of directly discounted cash flows: `reserve(0)` equals `-pv_net_cf`.
- **H5b** (engine; optional in the sheet) The same at every instant: for every `t = 0 .. H - 1`, with the inner sum accumulated in ascending `s`,

```
reserve(t) * disc(t)  equals  sum over s = t+1 .. H of
    [ (death_outgo(s) + claim_expense(s) + maint_expense(s)) * disc(s) - (premium_income(s) - commission(s)) * disc(s-1) ]
```

**Product-specific.**

- **P1 (sheet)** `maturities(t) = 0` exactly for `t <> rem_m`, and `maturities(rem_m) = pols_if(rem_m - 1) - deaths(rem_m) - lapses(rem_m)` exactly.
- **P2 (sheet)** For every `t > prem_rem_m`: `premium_income(t) = 0`, `commission(t) = 0`, `w_ann(t) = 0`, `w_m(t) = 0` and `lapses(t) = 0` exactly; for `1 <= t <= prem_rem_m`: `premium_income(t) = pols_if(t-1) * premium` exactly.
- **P3** The number of months with `premium_pp(t) <> 0` equals `prem_rem_m`.
- **P4** `age(t)` lies in 18 to 120 for every `t`; `dur_y(t)` is non-decreasing in `t` and steps by 0 or 1 between consecutive months.
- **P5** Per-policy recursion identity, for `t = 0 .. rem_m - 1` (textbook prospective reserve per policy in force, scale `sum_assured`):

```
reserve_pp(t)  equals  -(premium_pp(t+1) - commission_pp(t+1))
    + v_m * ( q_m(t+1) * (sum_assured + claim_exp_pp(t+1)) + maint_pp(t+1)
              + (1 - q_m(t+1)) * (1 - w_m(t+1)) * reserve_pp(t+1) )
```

  Proof that it follows from section 10: multiply both sides by `pols_if(t)`; `pols_if(t) * reserve_pp(t) = reserve(t)`, `pols_if(t) * q_m(t+1) = deaths(t+1)`, `pols_if(t) * (1 - q_m(t+1)) * (1 - w_m(t+1)) = pols_if(t+1)` for `t+1 < rem_m` (and `reserve_pp(rem_m) = 0` covers `t+1 = rem_m`), which turns the right-hand side into `(reserve(t+1) + net_outgo(t+1)) * v_m`.

- **P6** `reserve_floored(t) >= 0` and `reserve_floored(t) >= reserve(t)` exactly.
- **P7** For `t > rem_m`, every Count, every per-policy premium and commission amount, every in-force cash flow, `net_cf`, `net_outgo`, `pv_net_cf_t`, the five `pv_*_t` columns, `q_m`, `w_ann`, `w_m` and every reserve variable is exactly 0, and `age(t) = age(rem_m)`, `q_ann(t) = q_ann(rem_m)` exactly (engine only; the spreadsheet has no such rows).
- **P8 (sheet)** All policies exit: `sum over t = 1 .. H of (deaths(t) + lapses(t) + maturities(t))` equals `pols_if_init` (scale `pols_if_init`).
- **P9 (sheet)** PV consistency: `pv_net_cf` equals `sum over t = 1 .. H of pv_net_cf_t(t)`.
- **P10** Unit identities: `death_outgo(t) = deaths(t) * sum_assured`, `premium_income(t) = pols_if(t-1) * premium_pp(t)` and `maint_expense(t) = pols_if(t-1) * maint_pp(t)` exactly; these guard against multiplying by the in-force count twice.
- **P11** (engine only) Batch independence: a policy's outputs for `t = 0 .. rem_m` and all its scalars, projected alone to `H = rem_m` and in a batch to a larger `H`, agree within tolerance, `Int` quantities agree exactly, and every cell that is exactly 0 in one run is exactly 0 in the other (D5). The horizon itself changes no bits: the extra months add exact zeros to every running sum and the reserve recursion passes through them as exact zeros (sections 4.5, 9.3, 10.3), so on a single code path the two runs are bit-identical, which the script of section 14 confirmed for the reference points at `H = 480`. Bit-identity across batch compositions is not required of the engine, because a batch of one and a lane of a larger batch can take different SIMD and scalar paths for `exp` and powers (Halley Spec, Debugging, which names `exp`/`log`; powers are computed through the same kernels). D19 requires bit-identical reruns on the same backend, and a lone run and a batch lane are not reruns of each other.

## 13. Reference model points

Valuation date **2027-01-01**. Ten synthetic policies. Dates are ISO `YYYY-MM-DD`. `premium` values were chosen by the script of section 14 as `round(1.25 * break-even monthly premium at entry on this basis, 2)`, where the break-even premium makes the present value of premiums net of commission at entry equal the present value of all outgo at entry; that pricing rule is how the inputs were picked and is not part of the model. The inputs are these numbers, exactly.

| `policy_id` | `sex` | `birth_date` | `entry_date` | `maturity_date` | `prem_term_y` | `sum_assured` | `premium` | `pols_if_init` | Why it is here |
|---|---|---|---|---|---|---|---|---|---|
| P01 | M | 1986-03-15 | 2024-07-01 | 2044-07-01 | 20 | 250000 | 52.96 | 1 | typical in-force policy, 30 months in; entry on the 1st; negative reserve at outset |
| P02 | F | 1991-11-02 | 2027-01-01 | 2052-01-01 | 25 | 400000 | 51.64 | 1 | brand new: entered on the valuation date; policy year 0 (20% commission) in months 1 to 12 |
| P03 | M | 1966-05-20 | 2007-01-15 | 2027-01-15 | 20 | 150000 | 39.99 | 1 | one month left: matures at t = 1; single-row hand check |
| P04 | F | 1976-08-09 | 2015-03-20 | 2045-03-20 | 20 | 300000 | 68.98 | 1 | premium term (20y) shorter than policy term (30y), ending inside the projection |
| P05 | M | 1975-04-18 | 2018-02-10 | 2028-02-10 | 9 | 200000 | 46.40 | 1 | worked example: premiums end at t = 2, anniversary at t = 3 |
| P06 | F | 2003-02-28 | 2026-11-10 | 2036-11-10 | 10 | 100000 | 10.76 | 1 | partial month in force (1 month 22 days); youngest life; negative reserve throughout |
| P07 | M | 1980-06-01 | 2020-02-29 | 2040-02-29 | 20 | 500000 | 105.55 | 2.5 | grouped model point (count 2.5); leap-day entry and maturity |
| P08 | M | 1958-12-31 | 2010-12-31 | 2030-12-31 | 10 | 120000 | 128.12 | 1 | paid up: premium term over 6 years ago; oldest life; entered on his birthday (tuple equality) |
| P09 | F | 1996-10-31 | 2026-12-15 | 2066-12-15 | 40 | 1000000 | 116.62 | 1 | longest horizon (480 months, sets the batch horizon); entered 17 days before valuation so `dur_m = 0` |
| P10 | M | 1990-05-05 | 2022-02-01 | 2032-02-01 | 5 | 200000 | 39.27 | 1 | one premium left (`prem_rem_m = 1`), then 60 paid-up months; lapses stop after month 1 |

Derived per-policy constants (sections 4.3 and 5), which the spreadsheet must reproduce before projecting anything:

| `policy_id` | `age_at_entry` | `dur_m` | `term_m` | `rem_m` | `prem_term_m` | `prem_rem_m` | `dur_y_last` | `dur_y(1)` | `age(1)` | `age(rem_m)` |
|---|---|---|---|---|---|---|---|---|---|---|
| P01 | 38 | 30 | 240 | 210 | 240 | 210 | 19 | 2 | 40 | 57 |
| P02 | 35 | 0 | 300 | 300 | 300 | 300 | 24 | 0 | 35 | 59 |
| P03 | 40 | 239 | 240 | 1 | 240 | 1 | 19 | 19 | 59 | 59 |
| P04 | 38 | 141 | 360 | 219 | 240 | 99 | 29 | 11 | 49 | 67 |
| P05 | 42 | 106 | 120 | 14 | 108 | 2 | 9 | 8 | 50 | 51 |
| P06 | 23 | 1 | 120 | 119 | 120 | 119 | 9 | 0 | 23 | 32 |
| P07 | 39 | 82 | 240 | 158 | 240 | 158 | 19 | 6 | 45 | 58 |
| P08 | 52 | 192 | 240 | 48 | 120 | 0 | 19 | 16 | 68 | 71 |
| P09 | 30 | 0 | 480 | 480 | 480 | 480 | 39 | 0 | 30 | 69 |
| P10 | 31 | 59 | 120 | 61 | 60 | 1 | 9 | 4 | 35 | 40 |

Points to notice: `dur_m` loses a month for every entry day other than the 1st (P03: `12 * 20 + 0 - 1 = 239`; P09: `12 * 1 + (1 - 12) - 1 = 0`); P07's `term_m` is 240 with no special leap-day rule because `maturity_date` carries the same day of month as `entry_date`; P08's `prem_rem_m` is clamped to 0 (the unclamped value is -72), so it projects no premiums, commission or lapses; P03 has a single projection row; P05's `age` changes from 50 to 51 at `t = 3`; both `sex` values, entry days before, on and after the valuation day of month, and a maturity inside the first year are covered. The ten rows, as a Parquet file with the Arrow types of section 3.1, are the reference input; its hash appears in the manifest of every reconciliation run.

## 14. Worked example

The numbers in this section were produced by a Python script (plain float64 scalars, standard library only) implementing the formulas above exactly as written; the script also checks every invariant of section 12 for all ten reference points at their own horizon and in a batch to `H = 480`. The script is committed beside this document as `docs/spec/term_assurance_numbers.py` (C37); running it with Python 3.12 or later regenerates every printed number and rechecks every invariant. The numbers are **illustrative**, not the reference implementation and not the engine: the spreadsheet is built from the formulas, and the engine reconciles to the spreadsheet. Values are shown to 6 significant figures; selected values and scalars also in full float64 precision.

### 14.1 Policy P05: inputs and constants

`sex = M`, `birth_date = 1975-04-18`, `entry_date = 2018-02-10`, `maturity_date = 2028-02-10`, `prem_term_y = 9`, `sum_assured = 200000`, `premium = 46.40`, `pols_if_init = 1`. Derived: `age_at_entry = 42`, `dur_m = 106`, `term_m = 120`, `rem_m = 14`, `prem_term_m = 108`, `prem_rem_m = 2`, `dur_y_last = 9`. Basis constants: `i_m = 0.00327374`, `v_m = 0.996737`, `e_m = 0.00246627`.

### 14.2 Policy P05: months 0 to 3, month 13 and the final month 14

"n/a" marks a month quantity at `t = 0`, which is undefined (blank in the spreadsheet, stored as 0 by the engine).

| variable | t = 0 | t = 1 | t = 2 | t = 3 | t = 13 | t = 14 |
|---|---|---|---|---|---|---|
| `dur_y` | n/a | 8 | 8 | 9 | 9 | 9 |
| `age` | n/a | 50 | 50 | 51 | 51 | 51 |
| `q_ann` | n/a | 0.00229121 | 0.00229121 | 0.0025001 | 0.0025001 | 0.0025001 |
| `q_m` | n/a | 0.000191135 | 0.000191135 | 0.00020858 | 0.00020858 | 0.00020858 |
| `w_ann` | n/a | 0.04 | 0.04 | 0 | 0 | 0 |
| `w_m` | n/a | 0.00339605 | 0.00339605 | 0 | 0 | 0 |
| `comm_rate` | n/a | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 |
| `infl` | 1 | 1.00247 | 1.00494 | 1.00742 | 1.03254 | 1.03509 |
| `deaths` | n/a | 0.000191135 | 0.00019045 | 0.000207087 | 0.000206655 | 0.000206612 |
| `lapses` | n/a | 0.0033954 | 0.00338323 | 0 | 0 | 0 |
| `maturities` | n/a | 0 | 0 | 0 | 0 | 0.990358 |
| `pols_if` | 1 | 0.996413 | 0.99284 | 0.992633 | 0.990564 | 0 |
| `premium_pp` | n/a | 46.4 | 46.4 | 0 | 0 | 0 |
| `commission_pp` | n/a | 0.928 | 0.928 | 0 | 0 | 0 |
| `maint_pp` | n/a | 5.01233 | 5.02469 | 5.03709 | 5.1627 | 5.17543 |
| `claim_exp_pp` | n/a | 250.617 | 251.235 | 251.854 | 258.135 | 258.772 |
| `premium_income` | n/a | 46.4 | 46.2336 | 0 | 0 | 0 |
| `commission` | n/a | 0.928 | 0.924672 | 0 | 0 | 0 |
| `maint_expense` | n/a | 5.01233 | 5.00667 | 5.00102 | 5.11505 | 5.1266 |
| `death_outgo` | n/a | 38.227 | 38.0899 | 41.4174 | 41.3311 | 41.3225 |
| `claim_expense` | n/a | 0.0479017 | 0.0478476 | 0.0521557 | 0.053345 | 0.0534654 |
| `net_cf` | n/a | 2.18472 | 2.16445 | -46.4706 | -46.4995 | -46.5025 |
| `net_outgo` | n/a | -2.33359 | -2.31278 | 46.4706 | 46.4995 | 46.5025 |
| `v` | n/a | 0.996737 | 0.996737 | 0.996737 | 0.996737 | 0.996737 |
| `disc` | 1 | 0.996737 | 0.993485 | 0.990243 | 0.958401 | 0.955274 |
| `pv_net_cf_t` | n/a | 2.32597 | 2.29771 | -46.0171 | -44.5651 | -44.4226 |
| `pv_premiums_t` | n/a | 46.4 | 46.0827 | 0 | 0 | 0 |
| `pv_commission_t` | n/a | 0.928 | 0.921654 | 0 | 0 | 0 |
| `pv_maint_expense_t` | n/a | 4.99598 | 4.97405 | 4.95222 | 4.90227 | 4.89731 |
| `pv_death_outgo_t` | n/a | 38.1023 | 37.8418 | 41.0133 | 39.6117 | 39.4743 |
| `pv_claim_expense_t` | n/a | 0.0477453 | 0.0475358 | 0.0516468 | 0.0511259 | 0.0510741 |
| `reserve` | 537.961 | 542.056 | 546.143 | 501.46 | 46.3508 | 0 |
| `reserve_pp` | 537.961 | 544.007 | 550.082 | 505.182 | 46.7923 | 0 |
| `reserve_floored` | 537.961 | 542.056 | 546.143 | 501.46 | 46.3508 | 0 |

Inputs on the left of each equation below are shown to 6 significant figures; the results on the right are full-precision float64 computed from full-precision inputs, so retyping a printed left-hand side reproduces the right-hand side only to about 6 digits. Reading the table against the formulas, month 1: `age(1) = 42 + 8 = 50`; `q_ann = 0.00229121307525193` (chain in section 6.1); `q_m = 1 - (1 - 0.00229121)^(1/12) = 0.00019113522468450217`; `dur_y = 8` so the lapse key is `min(8, 5) = 5`, `w_ann = 0.04` (because `1 <= prem_rem_m = 2`), `w_m = 0.003396053198917559`; `deaths = 1 * q_m`; `lapses = (1 - 0.000191135) * 0.00339605 = 0.0033954040935263437`; `pols_if(1) = 1 - 0.000191135 - 0.0033954 - 0 = 0.9964134606817892`; `premium_income = 1 * 46.40`; `commission = 46.40 * 0.02 = 0.9279999999999999`; `maint_pp = 5 * 1.0024662697723037 = 5.012331348861519` and `maint_expense = 1 * maint_pp`; `death_outgo = 0.000191135 * 200000 = 38.227044936900434`; `claim_expense = 0.000191135 * 250.617 = 0.04790165392789101`; `net_outgo = 38.227 + 0.0479017 + 5.01233 - (46.40 - 0.928) * 1.003273739782199 = -2.333585555686305`. Month 3: `dur_y = 9`, `age = 51`, `t = 3 > prem_rem_m = 2` so premium, commission, `w_ann`, `w_m` and lapses are 0. Month 14: `maturities(14) = 0.990564 - 0.000206612 - 0 = 0.990357590196841` and `pols_if(14) = 0` exactly. Full precision for the reserve column: `reserve(13) = 46.35078406367477`, `reserve(3) = 501.4602926431264`, `reserve(2) = 546.1429251678909`, `reserve(1) = 542.0555952821584`, `reserve(0) = 537.9608658387098`.

### 14.3 Three hand-checkable steps (P05)

Step A, the last recursion step, `t = 13`:

```
reserve(13) = (reserve(14) + net_outgo(14)) * v_m = (0 + 46.502524469400136) * 0.9967369426185624 = 46.35078406367477
```

Step B, the first recursion step, `t = 0`:

```
reserve(0) = (reserve(1) + net_outgo(1)) * v_m = (542.0555952821584 + (-2.333585555686305)) * v_m = 537.9608658387098
```

Step C, the roll-forward (H4) at `t = 1`:

```
(reserve(0) + premium_income(1) - commission(1)) * (1 + i_m) = (537.9608658387098 + 46.4 - 0.928) * 1.003273739782199 = 585.3428732218482
- maint_expense(1) - death_outgo(1) - claim_expense(1)         = 585.3428732218482 - 5.012331348861519 - 38.227044936900434 - 0.04790165392789101
                                                               = 542.0555952821584 = reserve(1)      residual 0.0
```

The per-policy recursion P5 at `t = 0` gives `537.9608658387098` against `reserve_pp(0) = 537.9608658387098`, residual 0.0.

### 14.4 Scalars

Policy P05:

```
pv_premiums      =   92.48272174
pv_commission    =    1.849654435
pv_maint_expense =   69.06663598
pv_death_outgo   =  558.8156962
pv_claim_expense =    0.7116009441
pv_net_cf        = -537.9608658
reserve_0        =  537.9608658      (full float64: 537.9608658387098)
reserve_pp_0     =  537.9608658
reserve_floored_0=  537.9608658
maturities_n     =    0.9903575902
H5a check: reserve_0 + pv_net_cf = -2.3e-13     P9 check: pv_net_cf - sum of pv_net_cf_t = -1.1e-13
```

Illustrative scalars for all ten reference points from the same script (6 significant figures), with the portfolio totals of section 11.3. These are for spotting transcription errors; the spreadsheet is the reference, not this table.

| `policy_id` | `rem_m` | `pv_premiums` | `pv_commission` | `pv_maint_expense` | `pv_death_outgo` | `pv_claim_expense` | `pv_net_cf` | `reserve_0` | `reserve_pp_0` | `reserve_floored_0` | `maturities_n` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P01 | 210 | 5662.06 | 113.241 | 656.93 | 4232.38 | 5.5603 | 653.949 | -653.949 | -653.949 | 0 | 0.446152 |
| P02 | 300 | 5327.89 | 208.399 | 670.589 | 3421.64 | 3.15621 | 1024.11 | -1024.11 | -1024.11 | 0 | 0.268067 |
| P03 | 1 | 39.99 | 0.7998 | 4.99598 | 63.9914 | 0.106915 | -29.9041 | 29.9041 | 29.9041 | 29.9041 | 0.996177 |
| P04 | 219 | 4982.58 | 99.6517 | 765.998 | 9278.97 | 10.5313 | -5172.57 | 5172.57 | 5172.57 | 5172.57 | 0.669244 |
| P05 | 14 | 92.4827 | 1.84965 | 69.0666 | 558.816 | 0.711601 | -537.961 | 537.961 | 537.961 | 537.961 | 0.990358 |
| P06 | 119 | 726.667 | 34.1462 | 382.244 | 169.998 | 0.488444 | 139.79 | -139.79 | -139.79 | 0 | 0.517823 |
| P07 | 158 | 25545.8 | 510.915 | 1425.69 | 26085.5 | 15.998 | -2492.37 | 2492.37 | 996.947 | 2492.37 | 1.40619 |
| P08 | 48 | 0 | 0 | 229.531 | 5892.76 | 13.0659 | -6135.36 | 6135.36 | 6135.36 | 6135.36 | 0.946647 |
| P09 | 480 | 13280 | 495.604 | 803.96 | 9419.33 | 4.51075 | 2556.64 | -2556.64 | -2556.64 | 0 | 0.136924 |
| P10 | 61 | 39.27 | 0.7854 | 295.665 | 791.159 | 1.07052 | -1049.41 | 1049.41 | 1049.41 | 1049.41 | 0.991339 |
| total | | 55696.7 | 1465.39 | 5304.67 | 59914.6 | 55.2 | -11043.1 | 11043.1 | | 15417.6 | |

In the script every policy satisfied H1 to H5 and P1 to P11 at its own horizon and at `H = 480`, with worst absolute residuals: H4 3.6e-12, H5a 2.5e-11, H5b 5.8e-12, P5 1.1e-11, P8 1.8e-15, P9 2.2e-11, and a one-ulp H2 overshoot of 2.8e-17 at `t = rem_m`. Observations the reviewer can use as sanity checks: P01, P02, P06 and P09 have negative reserves at `t = 0` (for 32, 59, 119 and 50 months respectively); P04's reserve peaks at 8403.11 at `t = 99`, the last premium month; P08 has no premiums and no lapses, so its reserve is the pure present value of claims and expenses; P07's `reserve_pp_0` is `reserve_0 / 2.5`; P10's single premium of 39.27 at time 0 is its whole `pv_premiums`, undiscounted.

Policy P03 (one row) is a hand check in full precision: `age(1) = 59`, `q_ann(1) = 0.005123997031102556`, `q_m(1) = deaths(1) = 0.0004280058561031419`, `lapses(1) = 0.0033945996682607847`, `maturities(1) = 0.996177394475636`, `pols_if(1) = 0`; `premium_income(1) = 39.99`, `commission(1) = 0.7998000000000001`, `maint_expense(1) = 5.012331348861519`, `death_outgo(1) = 64.20087841547128`, `claim_expense(1) = 0.10726535850210452`; `net_outgo(1) = 30.001976606022573`; `reserve(0) = (0 + 30.001976606022573) * v_m = 29.904078434800574`.

### 14.5 Magnitudes of named simplifications (P01, from the same script)

- Replacing the constant-force conversion by the simple division changes P01 as follows. `q_ann / 12` alone (lapse conversion unchanged) lowers `pv_death_outgo` by 0.10% and moves `reserve_0` from -653.949 to -658.388, because `q_ann / 12` is below `1 - (1 - q_ann)^(1/12)` at every age. `w_ann / 12` alone (mortality conversion unchanged) raises `pv_death_outgo` by 0.91% and moves `reserve_0` to -651.130, because fewer lapses leave more lives exposed to death. Both together raise `pv_death_outgo` by 0.81% and move `reserve_0` to -655.617.
- Discounting claims and claim expenses from mid-month instead of end of month raises `pv_death_outgo` by 0.16% and moves `reserve_0` to -647.017.
- Deaths-before-lapses against a simultaneous constant-force model: the exact dependent death rate minus `q_m` is negative everywhere, so deaths-first overstates the monthly death probability by about `q_m * w_m / 2` and moves that probability from lapse to death, with total exits unchanged. On P01 the difference is -6.06e-7 at worst (`q_m * w_m / 2 = 6.1e-7`). Over all ten reference points the most negative difference is -1.19e-6 (P09, `t = 469` to 480, age 69, lapse key 5); over new entrants in policy year 0 (lapse key 0, entry age 18 to 65, either sex) it is -5.02e-6 (male, age 65), 0.67% of `q_m`; over the whole admissible model point space it is -5.00e-5 (male, mortality age 104, premiums payable, lapse key 5), where `q_m * w_m / 2` is 5.02e-5 and the difference is 0.17% of `q_m`.

## 15. Mapping to Halley

The Halley model is one class, `TermAssurance(Product)`, with no build-time switches; the authoring example's `has_waiver` is omitted because nothing in this product would read it. Model point fields are those of section 3.1, read as `self.mp.<field>`; the derived fields of section 3.2 are computed by the loader and exposed on the model point; `valuation_date` is a run setting. Basis constants and tables are typed inputs. There is no `lag()`: every time offset is the constant `t-1` or `t+1`. Every month quantity and state variable of sections 5 to 10 is exactly one `@var`; the derived fields, the basis constants and the section 9.3 sums are loader fields, constants and reductions, as the table marks them.

### 15.1 Variable table

| Spec variable | Halley declaration | Unit (return type) | Timing | `init` / `terminal` | Direction | Reads at offsets | Checker rules exercised |
|---|---|---|---|---|---|---|---|
| `age_at_entry`, `dur_m`, `term_m`, `rem_m`, `prem_term_m`, `prem_rem_m`, `dur_y_last` | derived model point fields (loader) | Int | | | | | schema ranges and cross-field rules at load |
| `dur_y(t)` | `@var -> Int` = `where((mp.dur_m + t - 1) // 12 < mp.dur_y_last, (mp.dur_m + t - 1) // 12, mp.dur_y_last)` | Int | | | forward, non-recursive | 0 | time-dependent pure function; `where` clamp on a symbolic key (the `min` of section 5) keeps every duration key in range |
| `age(t)` | `@var -> Int` = `mp.age_at_entry + dur_y(t)` | Int | | | forward, non-recursive | 0 | key range check against `mort_table` [18, 120] at load |
| `q_ann(t)` | `@var -> Prob` = `mort_table[mp.sex, age(t)]` | Prob (annual) | | | forward, non-recursive | 0 | **enum coverage** on `sex` (both `M` and `F` rows must exist); table key range at load; Prob in [0, 1] at run time |
| `q_m(t)` | `@var -> Prob` = `where(t <= mp.rem_m, 1 - (1 - q_ann(t)) ** (1/12), 0)` | Prob (monthly) | | | forward, non-recursive | 0 | explicit annual-to-monthly conversion (`1 - Prob` and `Prob ** float`, 15.4 (e)); `where` on `t` and model point data; Prob range |
| `w_ann(t)` | `@var -> Prob` = `where(t <= mp.prem_rem_m, lapse_table[where(dur_y(t) < 5, dur_y(t), 5)], 0)` | Prob (annual) | | | forward, non-recursive | 0 | `where`; `where` clamp on a symbolic key; key range [0, 5] |
| `w_m(t)` | `@var -> Prob` = `1 - (1 - w_ann(t)) ** (1/12)` | Prob (monthly) | | | forward, non-recursive | 0 | explicit conversion (15.4 (e)); Prob range |
| `comm_rate(t)` | `@var -> Fraction` = `comm_table[where(dur_y(t) < 1, dur_y(t), 1)]` | Fraction | | | forward, non-recursive | 0 | `where` clamp on a symbolic key; key range [0, 1] |
| `i`, `e` | basis constants | Rate[Annual] | | | | | unit: cannot be used as monthly without conversion |
| `maint_base`, `claim_exp_base` | basis constants (`expenses` table, 6.8) | Money per policy | | | | | MoneyPerPolicy x factor -> MoneyPerPolicy in `maint_pp`, `claim_exp_pp` (15.4 (a)) |
| `i_m`, `e_m` | `to_monthly(i)`, `to_monthly(e)` = `(1 + r) ** (1/12) - 1` | Rate[Monthly] | | | | | the explicit conversion is the only path from annual to monthly |
| `v_m` | `(1 + i) ** (-1/12)` | factor | | | | | |
| `infl(t)` | `@var -> factor` = `(1 + e_m) ** t` | factor | | | forward, non-recursive | 0 | pure function of `t`; defined at `t = 0` |
| `deaths(t)` | `@var -> Count` = `pols_if(t-1) * q_m(t)` | Count | | | forward | -1 (`pols_if`), 0 | Count x Prob -> Count (in the Spec only as a comment in its example; 15.4 (e)); constant lag |
| `lapses(t)` | `@var -> Count` = `(pols_if(t-1) - deaths(t)) * w_m(t)` | Count | | | forward | -1, 0 | Count x Prob -> Count and Count - Count -> Count (15.4 (e)); same-period read of `deaths(t)` is acyclic |
| `maturities(t)` | `@var -> Count` = `where(t == mp.rem_m, pols_if(t-1) - deaths(t) - lapses(t), 0)` | Count | | | forward | -1, 0 | `where` on `t` and model point data |
| `pols_if(t)` | `@var(init=lambda mp: mp.pols_if_init) -> Count` = `pols_if(t-1) - deaths(t) - lapses(t) - maturities(t)` | Count | | `init` | **forward recursive** | -1 (self), 0 | init presence; SCC with all offsets <= 0; fixed lag `t-1`; no same-period cycle (`deaths` and `lapses` read `pols_if(t-1)`, not `pols_if(t)`) |
| `premium_pp(t)` | `@var -> MoneyPerPolicy` = `where(t <= mp.prem_rem_m, mp.premium, 0)` | Money per policy | | | forward, non-recursive | 0 | `where` |
| `commission_pp(t)` | `@var -> MoneyPerPolicy` = `premium_pp(t) * comm_rate(t)` | Money per policy | | | forward | 0 | Money-per-policy x Fraction (rule not in the v1 unit table; see 15.4); read by no other variable, so it is stored as an output to keep it in the generated code (11.1) |
| `maint_pp(t)` | `@var -> MoneyPerPolicy` = `maint_base * infl(t)` | Money per policy | | | forward | 0 | Money-per-policy x factor (rule not in the v1 unit table; see 15.4) |
| `claim_exp_pp(t)` | `@var -> MoneyPerPolicy` = `claim_exp_base * infl(t)` | Money per policy | | | forward | 0 | same (rule not in the v1 unit table; see 15.4) |
| `premium_income(t)` | `@var(timing="start") -> Money` = `pols_if(t-1) * premium_pp(t)` | Money | start | | forward | -1, 0 | Count x Money-per-policy -> Money |
| `commission(t)` | `@var(timing="start") -> Money` = `premium_income(t) * comm_rate(t)` | Money | start | | forward | 0 | Money x Fraction -> Money (rule not in the v1 unit table; see 15.4) |
| `maint_expense(t)` | `@var(timing="end") -> Money` = `pols_if(t-1) * maint_pp(t)` | Money | end | | forward | -1, 0 | Count x Money-per-policy -> Money |
| `death_outgo(t)` | `@var(timing="end") -> Money` = `deaths(t) * mp.sum_assured` | Money | end | | forward | 0 | Count x Money-per-policy -> Money (multiplying by `pols_if` again is a unit error) |
| `claim_expense(t)` | `@var(timing="end") -> Money` = `deaths(t) * claim_exp_pp(t)` | Money | end | | forward | 0 | same |
| `net_cf(t)` | `@var -> Money` (reporting; no timing, never discounted directly) | Money | none | | forward | 0 | Money adds only to Money |
| `net_outgo(t)` | `@var(timing="end") -> Money` = `death_outgo(t) + claim_expense(t) + maint_expense(t) - (premium_income(t) - commission(t)) * (1 + i_m)` | Money | end | | forward | 0 | explicit `(1 + i_m)` on the start-timed part, written out so the operation order matches the spreadsheet (15.4); Money x (1 + Rate[Monthly]) (rule not in the v1 unit table; see 15.4) |
| `v(t)` | `@var -> factor` = `v_m` | factor | | | forward, non-recursive | 0 | constant |
| `disc(t)` | `@var -> factor` = `v_m ** t` | factor | | | forward, non-recursive | 0 | pure function of `t`; defined at `t = 0` |
| `pv_net_cf_t(t)` | `@var -> Money` = `(premium_income(t) - commission(t)) * disc(t-1) - (maint_expense(t) + death_outgo(t) + claim_expense(t)) * disc(t)` | Money | | | forward | -1 (`disc`), 0 | explicit `disc(t-1)` on start flows and `disc(t)` on end flows, written out so the operation order matches the spreadsheet (15.4) |
| `pv_premiums_t(t)`, `pv_commission_t(t)`, `pv_maint_expense_t(t)`, `pv_death_outgo_t(t)`, `pv_claim_expense_t(t)` | `@var -> Money`, each one product of section 9.3 | Money | | | forward | -1 (`disc`), 0 | not stored (11.1); the `pv_*` reductions sum them in ascending `t` |
| `reserve(t)` | `@var(terminal=0) -> Money` = `(reserve(t+1) + net_outgo(t+1)) * v(t+1)` | Money | | `terminal = 0` | **backward recursive** | +1 (self, `net_outgo`, `v`) | terminal presence; SCC with all offsets >= 0; fixed lag `t+1`; no forward variable reads it, so the graph has no mixed component |
| `reserve_pp(t)` | `@var -> MoneyPerPolicy` = `where(pols_if(t) > 0, reserve(t) / pols_if(t), 0)` | Money per policy | | | after backward | 0 | Money / Count -> Money-per-policy (a division rule the unit system must provide); `where` guards the division |
| `reserve_floored(t)` | `@var -> Money` = `where(reserve(t) > 0, reserve(t), 0)` | Money | | | after backward | 0 | `where` instead of `max` on a symbolic value |
| `pv_*`, `reserve_0`, `reserve_pp_0`, `reserve_floored_0`, `maturities_n` | per-policy reductions over `t` | Money, Count | | | after projection | all `t` | fixed-order reduction (D19) |
| portfolio totals | post-processing sum over policies | Money, Count | | | outside the projection | | reads across policies are forbidden inside the projection; this lives outside it |

Direction is per strongly connected component: `pols_if` with `deaths`, `lapses` and `maturities` forms the only forward recursive component (internal offsets -1 and 0); `reserve` is the only backward component (internal offset +1); everything else is acyclic. No component mixes signs. No variable reads another policy.

### 15.2 Authoring sketch

The sketch below is the shape the model takes; names match this document so that trace and explain output reads like the spreadsheet. It is not executable until the authoring package exists.

```python
class TermAssurance(Product):
    # no build-time switches in version 0.1 (section 2)

    # model point fields: sex, birth_date, entry_date, maturity_date, prem_term_y,
    #                     sum_assured, premium, pols_if_init
    # derived at load:    age_at_entry, dur_m, term_m, rem_m, prem_term_m, prem_rem_m, dur_y_last
    # basis:              mort_table[sex, age], lapse_table[dur_y], comm_table[dur_y],
    #                     i, e (Rate[Annual]); maint_base, claim_exp_base (Money per policy)

    i_m = to_monthly(basis.i)                 # Rate[Annual] -> Rate[Monthly], (1 + i) ** (1/12) - 1
    v_m = (1 + basis.i) ** (-1 / 12)
    e_m = to_monthly(basis.e)

    @var
    def dur_y(self, t) -> Int:
        k = (self.mp.dur_m + t - 1) // 12          # plain Python local at trace time (D8)
        return where(k < self.mp.dur_y_last, k, self.mp.dur_y_last)
    @var
    def age(self, t) -> Int:            return self.mp.age_at_entry + self.dur_y(t)

    @var
    def q_ann(self, t) -> Prob:         return mort_table[self.mp.sex, self.age(t)]   # enum-covered lookup
    @var
    def q_m(self, t) -> Prob:
        return where(t <= self.mp.rem_m, 1 - (1 - self.q_ann(t)) ** (1 / 12), 0)
    @var
    def w_ann(self, t) -> Prob:
        return where(t <= self.mp.prem_rem_m, lapse_table[where(self.dur_y(t) < 5, self.dur_y(t), 5)], 0)
    @var
    def w_m(self, t) -> Prob:           return 1 - (1 - self.w_ann(t)) ** (1 / 12)
    @var
    def comm_rate(self, t) -> Fraction: return comm_table[where(self.dur_y(t) < 1, self.dur_y(t), 1)]
    @var
    def infl(self, t) -> factor:        return (1 + self.e_m) ** t

    @var
    def deaths(self, t) -> Count:       return self.pols_if(t-1) * self.q_m(t)
    @var
    def lapses(self, t) -> Count:       return (self.pols_if(t-1) - self.deaths(t)) * self.w_m(t)
    @var
    def maturities(self, t) -> Count:
        return where(t == self.mp.rem_m, self.pols_if(t-1) - self.deaths(t) - self.lapses(t), 0)
    @var(init=lambda mp: mp.pols_if_init)
    def pols_if(self, t) -> Count:
        return self.pols_if(t-1) - self.deaths(t) - self.lapses(t) - self.maturities(t)

    @var
    def premium_pp(self, t) -> MoneyPerPolicy:
        return where(t <= self.mp.prem_rem_m, self.mp.premium, 0)
    @var
    def commission_pp(self, t) -> MoneyPerPolicy: return self.premium_pp(t) * self.comm_rate(t)
    @var
    def maint_pp(self, t) -> MoneyPerPolicy:      return basis.maint_base * self.infl(t)
    @var
    def claim_exp_pp(self, t) -> MoneyPerPolicy:  return basis.claim_exp_base * self.infl(t)

    @var(timing="start")
    def premium_income(self, t) -> Money: return self.pols_if(t-1) * self.premium_pp(t)
    @var(timing="start")
    def commission(self, t) -> Money:     return self.premium_income(t) * self.comm_rate(t)
    @var(timing="end")
    def maint_expense(self, t) -> Money:  return self.pols_if(t-1) * self.maint_pp(t)
    @var(timing="end")
    def death_outgo(self, t) -> Money:    return self.deaths(t) * self.mp.sum_assured
    @var(timing="end")
    def claim_expense(self, t) -> Money:  return self.deaths(t) * self.claim_exp_pp(t)

    @var
    def net_cf(self, t) -> Money:
        return (self.premium_income(t) - self.commission(t) - self.maint_expense(t)
                - self.death_outgo(t) - self.claim_expense(t))
    @var(timing="end")
    def net_outgo(self, t) -> Money:
        return (self.death_outgo(t) + self.claim_expense(t) + self.maint_expense(t)
                - (self.premium_income(t) - self.commission(t)) * (1 + self.i_m))

    @var
    def v(self, t) -> factor:             return self.v_m
    @var
    def disc(self, t) -> factor:          return self.v_m ** t
    @var
    def pv_net_cf_t(self, t) -> Money:
        return ((self.premium_income(t) - self.commission(t)) * self.disc(t-1)
                - (self.maint_expense(t) + self.death_outgo(t) + self.claim_expense(t)) * self.disc(t))
    @var
    def pv_premiums_t(self, t) -> Money:      return self.premium_income(t) * self.disc(t-1)
    @var
    def pv_commission_t(self, t) -> Money:    return self.commission(t) * self.disc(t-1)
    @var
    def pv_maint_expense_t(self, t) -> Money: return self.maint_expense(t) * self.disc(t)
    @var
    def pv_death_outgo_t(self, t) -> Money:   return self.death_outgo(t) * self.disc(t)
    @var
    def pv_claim_expense_t(self, t) -> Money: return self.claim_expense(t) * self.disc(t)

    @var(terminal=0)
    def reserve(self, t) -> Money:        # backward recursion
        return (self.reserve(t+1) + self.net_outgo(t+1)) * self.v(t+1)
    @var
    def reserve_pp(self, t) -> MoneyPerPolicy:
        return where(self.pols_if(t) > 0, self.reserve(t) / self.pols_if(t), 0)
    @var
    def reserve_floored(self, t) -> Money:
        return where(self.reserve(t) > 0, self.reserve(t), 0)
```

### 15.3 Checker rules this model exercises

| Checker rule | Where in this model, and what change would make the build fail |
|---|---|
| Unit and dimension mismatches are errors | Count x Prob -> Count (`deaths`, `lapses`), Count x Money-per-policy -> Money (`premium_income`, `maint_expense`, `death_outgo`, `claim_expense`), Money adds only to Money (`net_cf`, `net_outgo`, the `pv` sums), Prob range on `q_ann`, `q_m`, `w_ann`, `w_m`; writing `death_outgo = pols_if(t-1) * deaths(t) * sum_assured` or adding `premium_pp` to `premium_income` must fail; Count x Money-per-policy -> Money is the one product rule in the Spec unit table; the other rules used here, including Count x Prob -> Count, are rules (a) to (e) of 15.4 |
| Time offsets must be constants, or `lag()` with bounds | all offsets are `t-1` or `t+1`; `lag()` is unused; a data-dependent offset such as `pols_if(t - mp.dur_m)` must fail |
| Recursive variables need `init` or `terminal` | `pols_if` (`init`), `reserve` (`terminal`); removing either must fail the build |
| SCC direction | forward component {`pols_if`, `deaths`, `lapses`, `maturities`}; backward component {`reserve`} |
| Same-period cycles | none; `lapses` reads `deaths` at offset 0 but `deaths` does not read `lapses`; making `deaths` read `pols_if(t)` must fail |
| Mixed forward/backward components | none; a cash flow that reads `reserve` at `t-1` (interest on the opening reserve) would form a mixed component and fail; one that reads `reserve(t)` or `reserve_pp(t)` (a reserve-based surrender value) joins the backward component and builds, which is a different design, not a checker error |
| Reads across policies | none; the portfolio totals of 11.3 are outside the projection |
| Enum coverage | `mort_table[self.mp.sex, ...]` must cover `M` and `F`; a table missing `F` must fail the build |
| Unused variables | `net_cf`, `pv_net_cf_t`, `commission_pp` and the reserve derivatives are read by no other variable but are outputs, so none is unused; `commission_pp` is an output only so that the engine-side P5 and P7 checks and trace can read it; dropping it or `reserve_floored` from the outputs makes it a warning and removes it from generated code |
| Python `if` on a symbolic value | every conditional is `where()`: `dur_y`, `q_m`, `w_ann`, `comm_rate`, `maturities`, `premium_pp`, `reserve_pp`, `reserve_floored` |
| Prob in [0, 1] at run time | `q_ann`, `q_m`, `w_ann`, `w_m` |
| Lookup keys in range at load | `age(t)` in [18, 120], clamped `dur_y` keys in [0, 5] and [0, 1], including the values beyond the term |
| Errors point to the user's code | a unit error in `commission` must name the `commission` method and line, not generated code |

### 15.4 Notes for the engine

- **`t = 0` for month quantities.** The engine evaluates month quantities from `t = 1`; whatever it stores at `t = 0` for them (this spec says 0) is excluded from invariants and reconciliation. Forward variables that read `pols_if(t-1)` are not evaluated at `t = 0`. State variables (`pols_if`, `infl`, `disc`, `reserve`, `reserve_pp`, `reserve_floored`) are defined at `t = 0`.
- **Batch horizon.** Projecting every policy to `H = max rem_m` requires the key clamping in section 5 so that no lookup leaves its range, and the `where` masks so that all outputs beyond a policy's term are exact zeros (P7). The engine's reconciliation reads spreadsheet rows beyond a policy's horizon as zeros. Tables are sized to the product, not to the horizon; if the engine later masks inactive rows out of lookups, the clamp becomes redundant but stays harmless. The engine's P11 test compares a lone run with a batch lane under the tolerance of section 1, with `Int` values and exact zeros compared exactly. Changing only the horizon changes no bits, which is a property of the formulas; bit-identity across batch compositions is not required of the backend.
- **Two conversions, one helper each.** `to_monthly` for `Rate[Annual]` is `(1 + r) ** (1/12) - 1`; the probability conversion `1 - (1 - q) ** (1/12)` is a different formula for a different unit. Exposing both as named helpers keeps `q / 12` and `i / 12` out of models. The v1 unit table has a single `Prob`; `q_ann / 12` is the same class of error as `i / 12` and a `Prob[Annual]` / `Prob[Monthly]` pair would catch it at build time. Until then the names `q_ann` and `q_m` carry the distinction. Listed in section 15.6 as a proposal for the Halley notes.
- **Units this model needs that the v1 unit table does not define.** None of (a) to (e) is in the Spec unit table. Three of the rules in (e), `Count x Prob -> Count`, `Count - Count -> Count` and a literal 0 in a `where` branch, are already used by the Spec's authoring example (`deaths`, `pols_if`, `premium_income`), so they need writing down rather than deciding. D11 (firm) makes unit mismatches build errors, so every variable that 15.1 types `Money`, `MoneyPerPolicy`, `Count` or `Prob` keeps that type and is checked; section 15.3 depends on this. Interim rule until the notes decide (15.6): the engine implements the rules of (c), (d) and (e), and the Money x Fraction, MoneyPerPolicy x Fraction, MoneyPerPolicy x factor and Money x factor products of (a), as checked rules. Only the dimensionless quantities (`comm_rate`, `infl`, `v`, `v_m`, `disc`) and the integer quantities of (b) (`dur_y`, `age`, table keys) may be plain float64 or int; the Spec's own example already multiplies Money by an unannotated `v(t+1)`. (a) A dimensionless type, two labels here: `Fraction` (`comm_rate`, a proportion of Money) and `factor` (`infl`, `v`, `disc`, `v_m`, a multiplier), with the rules Money x Fraction -> Money (`commission`), MoneyPerPolicy x Fraction -> MoneyPerPolicy (`commission_pp`), MoneyPerPolicy x factor -> MoneyPerPolicy (`maint_pp`, `claim_exp_pp`) and Money x factor -> Money (`pv_net_cf_t`, the `pv_*_t` columns, the reserve recursion); the Spec's own authoring example already multiplies Money by an unannotated `v(t+1)`. (b) `Int` as the return type of `dur_y` and `age` and as the key type of `mort_table`, `lapse_table` and `comm_table` (the Spec's typed keys). (c) `1 + Rate[Monthly] -> factor` and `factor ** t` (`infl`, `disc`), `(1 + Rate[Annual]) ** (-1/12)` (`v_m`) and Money x `(1 + i_m)` (`net_outgo`); these would be internal to Halley's accumulation and discounting helpers if the model called them, but the sketch writes `net_outgo`, `disc` and `pv_net_cf_t` inline (next bullet), so the rules are needed. (d) Money / Count -> MoneyPerPolicy for `reserve_pp`; the v1 unit table lists only the product rule. (e) `1 - Prob -> Prob` and `Prob ** float -> Prob` for the inline conversions in `q_m` and `w_m` (internal to Halley if the probability helper of the previous bullet exists and the sketch calls it); `Count - Count -> Count` for the decrement subtractions; a literal 0 in the unused branch of `where` taking the unit of the other branch (`q_m`, `w_ann`, `premium_pp`, `maturities`, `reserve_pp`, `reserve_floored`); and `Count x Prob -> Count`, which the Spec states only as a comment in its authoring example. The `Prob[Annual]` / `Prob[Monthly]` pair is in the previous bullet.
- **Explicit factors instead of the timing helpers.** Assumed: a Money variable that is not a cash flow (`net_cf`, `pv_net_cf_t`, the `pv_*_t` columns, `reserve`, `reserve_pp`, `reserve_floored`) may carry no timing, and the explicit `(1 + i_m)` in `net_outgo` and `disc(t-1)` / `disc(t)` in `pv_net_cf_t` and the `pv_*_t` columns stand in for the accumulation and discounting helpers, so that the operation order matches the spreadsheet cell for cell. `net_outgo` is declared `timing="end"` because its value is an amount at time `t`, although it is built from start and end flows.
- **`where` instead of `min` and `max`.** Under capture by tracing (D8) Python's builtin `min` and `max` call `bool()` on a comparison of a symbolic value, the same failure as a Python `if`. The `min(a, b)` notation of section 1 (used in sections 5, 6.2 and 6.6 for the three key clamps) is MIN in the spreadsheet and `where(a < b, a, b)` in Halley; `max(reserve(t), 0)` is likewise written `where(reserve(t) > 0, reserve(t), 0)`. The `max` in `prem_rem_m` (section 4.3) is computed by the loader on plain integers, outside tracing, and is unaffected. `reserve_pp` relies on the `where` semantics of section 1: its discarded branch divides by `pols_if(t) = 0` at and after maturity, and a backend that evaluates both branches must mask that division rather than raise.
- **Power versus running product.** `disc(t)` and `infl(t)` are powers; a backend that rewrites them as running products changes the last bits and must be checked against the tolerance, not assumed exact. `v_m` is one constant; `v(t+1)` in the recursion is that constant.
- **Bit-exact zeros.** `maturities` and `pols_if` are computed as written so that `pols_if(rem_m)` is exactly 0; the NumPy reference must not reorder the left-to-right subtraction.

### 15.5 Alignment with lifelib BasicTerm (reference case 2)

Reference case 2 reproduces lifelib's model from lifelib's own inputs, so this spec need not match it; where it can match at no cost it does, and the differences below are deliberate.

**Same as lifelib.**

- `pols_if(t)` is the count at time `t`; lifelib's `pols_if(t)` is the same number, with `init` at `t = 0`.
- Deaths first from the opening count, lapses from the survivors of death.
- Constant-force conversion `1 - (1 - q) ** (1/12)` for both mortality and lapse.
- 0-based completed policy years as the duration key; age = integer age last birthday at entry plus completed policy years.
- Premiums in advance on the opening count, including the first projected month; exactly `prem_term_m` premiums over a full premium term; nothing paid at the maturity instant; count zero at maturity.
- Inflation index compounded from the valuation date (lifelib writes `(1 + e) ** (months / 12)`, this spec `(1 + e_m) ** t`; the same value up to last-bit rounding, indexed at the payment instant here).
- First-year commission as a rate on premium by policy year; net cash flow in the insurer's view, premiums less outgo; premium as a data input for the reproduction.

**Deliberately different.**

- **Month index.** lifelib's month `k` runs from time `k` to `k+1`; this spec's month `t` runs from `t-1` to `t`. lifelib's flow at index `k` is this spec's flow at index `k+1`; counts are unshifted. The spec's convention follows Halley's authoring example (`deaths(t) = pols_if(t-1) * q_monthly(t)`, `init` at `t = 0`).
- **Claim timing.** Claims here are at the end of the month and discounted with `disc(t)`; lifelib values month-`k` claims at the start of month `k`. For reference case 2 the lifelib product declares `death_outgo` with `timing="start"`; that one declaration is the whole difference.
- **Maintenance expense timing.** End of month here (this spec's choice, C18); start of month in lifelib. One declaration again.
- **Acquisition expense.** None here (no new business); lifelib charges one at issue.
- **Discounting.** Flat annual effective rate here; lifelib reads an annual spot curve by projection year. For reference case 2 the discount factor is loaded as lifelib defines it.
- **Mortality.** Ultimate Gompertz-Makeham by sex here; lifelib has a select-by-duration table that ignores sex. For reference case 2 lifelib's table is loaded with its own keys.
- **Lapse and commission tables, expense amounts, claim expense, premium term, grouped counts, dates as inputs**: data or schema differences, not structural ones.
- **Reserve.** lifelib has none; this spec's reserve is additional output and does not affect the cash flows.

### 15.6 Proposals for the Halley notes

These are gaps between this model and the Halley notes as they stand. None changes a formula or a number in this document, and none is a question for the reviewer. They are candidates for the vault's Open Questions note.

1. **Probability units and conversion helpers.** A `Prob[Annual]` / `Prob[Monthly]` pair, and one named helper each for converting rates and probabilities, so that `q_ann / 12` fails the build the way `i / 12` would (15.4). This extends the v1 units fixed by D11 (firm); `q_ann / 12` is the class of mistake D11 exists to catch.
2. **Unit rules.** The rules (a) to (e) of 15.4, which extend the v1 rules fixed by D11 (firm): dimensionless `Fraction` and `factor`, `Int` keys and return types, the accumulation and discounting products, Money / Count, and the `Prob` and `Count` arithmetic the sketch writes inline. Three of the (e) rules are already used by the Spec's own authoring example and need only writing into the unit table. Until then the interim rule of 15.4 applies.
3. **Binding the built-in invariants.** A way for the engine to identify the in-force count, the decrements, the reserve, the accumulation rate and the timed flows in a user model, so that H1 to H5 run on every run as the Spec intends (section 12).
4. **Product-level invariants.** A hook through which a product declares its own invariants, such as P1 to P11, so that they run with the built-in ones rather than only in the reconciliation and property tests (section 12).
5. **Reconciliation tolerance.** The rule of C33: `1e-9 * max(|a|, |b|, scale)` with a scale per unit, and summed scales for portfolio totals (section 1).
6. **Timing on Money variables that are not cash flows.** The Spec says each cash flow declares a timing. 15.4 assumes that other Money variables (`net_cf`, the PV columns, the reserves) carry none, and declares `net_outgo` `end` although it is built from start and end flows.
7. **Discarded `where` branches.** A backend that evaluates both branches of `where` must mask errors in the discarded one, such as the division by `pols_if(t) = 0` in `reserve_pp` (section 1, 15.4).

## 16. Decision log

Each item is one convention chosen in this document, with the reason. Object by number.

- **C1. Period convention.** `pols_if(t)` is the count at time `t` after month `t`'s decrements; month `t` flows are driven by `pols_if(t-1)`. Reason: this is the convention of the Halley authoring example, so the spec, the model and the spreadsheet share one indexing.
- **C2. Month quantities are undefined at `t = 0`; state variables are defined there.** Blank in the spreadsheet, 0 in the engine, excluded from every check. Reason: nothing reads a month quantity at `t = 0`, and inventing values that look computed is worse than declaring none.
- **C3. Valuation date on the first of a month; 2027-01-01 for the reference case.** Reason: `date(t)` then needs no day-of-month clipping and `months_between` needs no month-end clause; the date itself is synthetic and free.
- **C4. Valuation date is a policy month boundary; partial months are dropped.** Reason: one integer `dur_m`, no fractional first period, every `t`-indexed lookup stays integer-keyed; the consequences (one premium counted that was already due, cover counted twice for the elapsed fraction) are stated in section 4.3.
- **C5. Closed-form completed-months formula `12 * dy + dm - [day2 < day1]`, no month-end clause.** Reason: exact, three cells, no EOMONTH, handles 29 February and 31st dates by the same rule; the two uses never need a month-end clause under C3 and C6.
- **C6. `maturity_date` is an input with the same day of month as `entry_date`; `term_m` is derived; the premium term is an input in whole years.** Reason: the Halley Spec lists entry, maturity and birth dates as model point inputs with durations derived; the same-day rule makes `term_m` an exact month difference; a premium term in years is a product parameter, not a duration.
- **C7. Policies past or at maturity, and policies entered after the valuation date, are load errors, not zero projections.** Reason: a silent zero row hides data errors; the schema should admit only policies the invariants can be checked on.
- **C8. `prem_rem_m` is clamped at 0.** Reason: the derived field then has a declared range [0, `rem_m`] for the data contract and property tests; a negative "months remaining" invites a transcription error; `where(t <= prem_rem_m)` behaves identically.
- **C9. Age last birthday at entry by the lexicographic (month, day) comparison, advancing on policy anniversaries.** Reason: a one-line closed form, what in-force data carries, what lifelib and the `x(t-12)` idiom assume, and no birthday-date arithmetic in the sheet; on a synthetic basis the half-year understatement of age last birthday, relative to age nearest birthday, has nothing to bias against (section 5).
- **C10. `dur_y(t)` is 0-based completed policy years, clamped at `dur_y_last`; lapse and commission tables are short, with the key clamped in the formula.** Reason: 0-based is the natural `// 12` and the lifelib convention; the clamp keeps every lookup key in range when the engine projects past a policy's term, so the load-time range check means what it says and tables stay the size of the product; in Halley the clamp is `where(a < b, a, b)` (D8), MIN in the spreadsheet.
- **C11. Synthetic Gompertz-Makeham mortality by sex, exact one-year integral, ultimate, ages 18 to 120, no cap.** Reason: formula-defined so the spreadsheet needs no external file; the exact integral is the standard meaning of `q_x` for a GM force and costs one constant; by sex so the enum-coverage rule is exercised; 18 to 120 covers the schema (maximum age 104) with rectangular ranges; the maximum table value is below 1 so no cap is needed.
- **C12. Constant-force monthly conversion `1 - (1 - q)^(1/12)` for mortality and lapse, computed from the annual value.** Reason: twelve monthly steps reproduce the annual rate exactly; `q / 12` does not (on P01's PV of claims: -0.10% for `q_ann / 12` alone, +0.91% for `w_ann / 12` alone, +0.81% for both; section 14.5).
- **C13. Deaths first, then lapses from survivors, then maturity.** Reason: guarantees decrements never exceed the in-force count and gives one unambiguous formula for each; the difference from a simultaneous constant-force model is second order (deaths-first gives the higher death rate, by about `q_m * w_m / 2`: below 1.2e-6 per month on every reference point and below 5.1e-5 per month anywhere in the model point space, and at most about 0.67% of `q_m`, in policy year 0; section 14.5).
- **C14. Maturity as an explicit third decrement with `pols_if(rem_m) = 0` exactly, computed as written.** Reason: makes "the policy is gone" a computed fact, so every later value is a true zero and the terminal reserve is zero without a special case; the left-to-right subtraction makes the zero bit-exact in engine and sheet alike.
- **C15. `q_m` is masked to 0 after the term and `w_ann`, `w_m` after the premium term, by `where`.** Reason: the "everything zero after the term" invariant then covers the rates, and the clamped keys keep the unused lookups in range.
- **C16. Lapse rate zero after the premium term.** Reason: a paid-up policy has no premium to stop paying; exercises `prem_rem_m = 0` and `prem_rem_m = 1`.
- **C17. Premium is a model point input, monthly, level, in advance; the premium due at time 0 is projected.** Reason: pricing is outside the projection (no circularity); treating time 0 as a due date follows from C4.
- **C18. Timing: premium and commission `start`; maintenance expense, claims and claim expense `end`; no `mid`.** Reason: chosen here, following the Halley authoring example (Halley Spec, authoring language), which declares `premium_income` with `timing="start"` and `death_outgo` with `timing="end"`; the roadmap's M1 entry requires the timing convention to be stated, not which one is used; premiums are paid in advance and claims and expenses are settled after the event; maintenance and claim expense are placed with claims at the end so that all outgo is at one instant and `net_outgo` has a single accumulation term; `mid` would add a half-month exponent with no product meaning here. The start-of-month alternative for maintenance is reviewer question 1; mid-month claims are question 11.
- **C19. Maintenance expense on `pols_if(t-1)` (every policy in force at the start of the month), for every month of the term including paid-up months.** Reason: one driver count for all month-`t` flows; the alternative (survivors only) would need a fourth count.
- **C20. Commission 20% in policy year 0, 2% after, as a multiple of premium income, `Fraction` table with a clamped key.** Reason: a duration-keyed table exercises a second range-checked lookup on the same clamped key and front-loads cost so that early reserves are visibly negative, which is what the no-floor decision and the roll-forward invariant need to be tested on; the levels are a reviewer question.
- **C21. No acquisition expense, even for a policy entered on the valuation date.** Reason: no new business in the projection; a brand-new policy is treated as issued and paid for; avoids a model-point-conditional flow that only one reference point would exercise.
- **C22. Flat 4% annual effective interest; `i_m = (1 + i)^(1/12) - 1`; `v_m = (1 + i)^(-1/12)`; `v(t) = v_m`; `disc(t) = v_m^t` as a direct power.** Reason: one constant from the annual rate, one explicit conversion chain, and one form fixed so two spreadsheets agree bit for bit; a term structure would add a scenario file the M1 reference case does not need.
- **C23. Expense inflation 3% annual effective, converted to `e_m`, index `(1 + e_m)^s` at the payment instant `s`, base date the valuation date.** Reason: indexing by instant is the timing-consistent rule (start flows would use `infl(t-1)`, end flows use `infl(t)`); written as a power of the declared monthly rate so the Rate[Annual] to Rate[Monthly] conversion is the one used; time-0 prices are the stated amounts.
- **C24. Per-policy amount columns (`premium_pp`, `commission_pp`, `maint_pp`, `claim_exp_pp`) beside the in-force totals.** Reason: makes the per-policy versus in-force unit rule visible in every row, makes "multiplied by the count twice" impossible to miss, and gives the per-policy recursion check (P5) its inputs directly.
- **C25. In-force (total) `Money` reserve by backward recursion in the Halley example's shape, with `net_outgo` accumulated to month end; the per-policy reserve is a derived ratio.** Reason: matches `reserve(t) = (reserve(t+1) + net_outgo(t+1)) * v(t+1)` line for line; the backward component reads only two forward variables at `t+1` (`net_outgo` and `v`, the latter a constant-valued `@var` in version 0.1) and carries no decrement logic.
- **C26. Reserve defined before the premium due at `t`, paired with the end-of-month count `pols_if(t)`.** Reason: the standard prospective definition, the pairing under which the roll-forward as written has its per-policy reading and `reserve(0) = -pv_net_cf` with the time-0 premium counted as future income (section 10.6).
- **C27. No floor on negative reserves; `reserve_floored = where(reserve > 0, reserve, 0)` reported separately.** Reason: the PV and roll-forward invariants hold only for the unfloored reserve; aggregate flooring belongs to the aggregation stage; IFRS 17 and Solvency II start unfloored; `where` rather than `max` because of capture by tracing.
- **C28. Single basis for projection and reserve, no margins.** Reason: keeps the reference case to one set of tables; a valuation basis is a second table set the spreadsheet would have to carry.
- **C29. Present values from component sums in ascending `t`; `pv_net_cf` from the components; a monthly `pv_net_cf_t` column as a row-level check.** Reason: fixed summation order (D19); mixed-timing `net_cf` cannot be discounted as one number; the monthly column lets the reviewer check the PV identity row by row.
- **C30. Fractional `pols_if_init` allowed.** Reason: grouped model points are normal; Count is float64.
- **C31. Ten reference points, chosen to hit every row of the edge-case table in section 4.6 that a valid model point can exercise; the two load-error rows are exercised by the loader's tests, not by a reference point.** Reason: the spreadsheet must show each resolved case at least once, including `rem_m = 1`, `prem_rem_m = 0` and `prem_rem_m = 1`, a brand-new policy, a partial month, a leap-day entry, a birthday entry and a 480-month horizon.
- **C32. Premium inputs set at 1.25 times break-even on this basis, rounded to cents.** Reason: plausible magnitudes and a mix of negative and positive reserves; the rule is outside the model.
- **C33. Reconciliation and invariant tolerance `1e-9 * max(|a|, |b|, scale)` with the scale of the unit.** Reason: residuals of comparisons to zero (roll-forward) scale with the policy's Money magnitude; a fixed currency floor is too loose on single-policy Count cells and can fail correct schema-maximum policies; a tiny absolute floor fails spuriously where large components cancel; scaling by the natural magnitude of each unit is both tight and robust. A portfolio total takes the sum of the per-policy scales, so that a headline total passes whenever every per-policy cell passes. Proposed for the Halley notes (section 15.6); the reviewer's side of it is question 12.
- **C34. No rounding anywhere; premiums are inputs already in cents.** Reason: rounding is a convention that would have to be specified per cell.
- **C35. Variable names follow the Halley authoring example (`pols_if`, `deaths`, `lapses`, `premium_income`, `death_outgo`, `net_outgo`, `reserve`, `v`) and the model point fields `pols_if_init`, `premium`, `sum_assured`; `prem_rem_m` is the remaining-months quantity the example calls `prem_term_m`; annual and monthly probabilities carry `_ann` and `_m` suffixes.** Reason: trace and explain output then reads like the spreadsheet; the suffixes carry a unit distinction the v1 unit table cannot (section 15.4 lists the others).
- **C36. Portfolio totals are post-processing sums in `policy_id` order and never feed a policy's projection.** Reason: D5 and D6; the reconciliation test still gets a one-line headline.
- **C37. The spec lives in the repository at `docs/spec/term-assurance.md`, with the reference model point file and basis tables beside it once built, and the generating script of section 14 (standard library only) committed beside it as a separate file; emitting the Parquet files needs pyarrow and is a separate step.** Reason: the reconciliation test must cite a versioned file under the project licence; nothing in it is proprietary; without the script the printed numbers of sections 6, 13 and 14 cannot be regenerated from the repository.
- **C38. Lapse table levels 0.15, 0.10, 0.08, 0.06, 0.05, 0.04 by 0-based completed policy year (key clamped per C10, monthly conversion per C12, zero after the premium term per C16).** Reason: round synthetic values falling from a high first-year rate to a low ultimate rate; the levels are a reviewer question.
- **C39. Expense amounts `maint_base = 5.00` per policy per month and `claim_exp_base = 250.00` per death, both at time-0 prices (inflation per C23, timing per C18 and C19).** Reason: round synthetic values of plausible magnitude; the levels are a reviewer question.

## 17. Questions for the reviewer

These are the only items open for the reviewer; proposals for the Halley notes are in section 15.6. Everything else is fixed as written and can be changed only by editing the relevant item in section 16.

The items from 14 onward were raised by the spreadsheet builder (2026-10-17) while transcribing sections 3 to 13 into the workbook; the builder chose the most literal reading in each case, as section 1 instructs.

1. **Maintenance expense timing (C18, C19).** Maintenance expense is charged on the start-of-month count and paid at the end of the month, inflated to the end of the month, as fixed in C18. Would you rather have it paid at the start with the premium (lifelib and the textbook profit-test layout)? If so it moves into the start-timed group with `infl(t-1)`, and `net_outgo` and H4 change accordingly.
2. **Lapses after the premium term (C16).** The lapse rate is zero once premiums stop. Is that your expectation for paid-up term business, or would you keep a small residual lapse rate running to the end of the term?
3. **Decrement order (C13).** Deaths then lapses from survivors. Is the dependent-rate approximation acceptable, or do you want independent rates combined under constant forces (which differs at second order, deaths-first giving the higher death rate and so slightly higher claims and reserves: below 1.2e-6 per month on the reference points and below 5.1e-5 per month anywhere in the model point space, the absolute worst being a male aged 104 with premiums still payable, and at most about 0.67% of `q_m` in relative terms, in policy year 0; section 14.5)?
4. **Age basis (C9).** Age last birthday at entry, advancing on anniversaries. Would you prefer age nearest birthday (unbiased for a table defined at exact integer ages, at the cost of birthday-date arithmetic and a tie rule), or age recomputed at each policy anniversary from the birth date?
5. **Partial-month alignment (C4).** The days between the last policy month boundary and the valuation date are dropped, so P06 is projected to mature 21 days late and one premium already due is counted as future income. Is dropping (rather than rounding to the nearest boundary, or carrying a shortened first month) the convention you want the engine to carry forward?
6. **Policies at or past maturity, or entered after the valuation date (C7).** Rejected at load. Would you rather they load and project as zeros so that a mixed portfolio file never fails?
7. **Negative reserves (C27).** Unfloored at policy level, floored figure reported alongside and summed only as a presentation total. Is that the presentation you would sign off for reference case 1, or should version 0.1 also specify a build-time variant with the floor inside the recursion, with its own invariants?
8. **Acquisition expense on the brand-new policy (C21).** P02 is valued as in force at its issue instant with no strain beyond the 20% first-year commission. Acceptable for the reference case, or should a policy with `dur_m = 0` carry an initial expense in month 1?
9. **Basis plausibility (C11, C20, C22, C23, C38, C39).** The synthetic parameters give male `q_ann` of 0.0023 at 50 and 0.014 at 70, female about two thirds of male, lapses from 15% down to 4%, 20% then 2% commission, 4% interest, 3% inflation, 5.00 per month maintenance and 250 per claim. Are any of these far enough from what you would expect that they weaken the reference case?
10. **Select mortality (C11).** The table is ultimate only. Do you want a select period (say two durations) added so the spreadsheet exercises a two-key mortality lookup and a select-to-ultimate transition, at the cost of a second table?
11. **Claim timing (C18).** Death claims and claim expenses are paid at the end of the month of death. Would a mid-month payment (Halley's `mid` timing, `disc(t - 1/2)`, 0.16% on the PV of claims) be a better reference convention, given that the engine supports it and the spreadsheet can do it with one extra factor, at the cost of a third instant in the roll-forward?
12. **Tolerance (C33).** Is `1e-9 * max(|a|, |b|, scale)` with the stated scales acceptable to you as "reconciled", given that Excel and NumPy may differ in the last bit of `exp` and `^` over 480 months, or do you prefer a looser absolute tolerance in currency units (for example 0.01) on Money cells?
13. **Scope of the spreadsheet check.** The layout is fixed in section 11 (one sheet per policy with rows to the policy's own `rem_m`, one column per variable in the order of section 7.1, one check cell per (sheet) invariant per policy, plus a summary sheet). Do you want a common horizon (480 rows for every policy) so the zero-after-term rule is visible, and do you want to check all ten policies cell by cell or the worked example P05 in full plus the scalars of the other nine?
14. **Row `t = 0` of `infl` and `disc` (7.1, 6.4, 9.1).** Step 2 of 7.1 says "Row `t = 0`: `pols_if(0) = pols_if_init`, `infl(0) = 1`, `disc(0) = 1`; every month quantity blank", while 6.4 defines `infl(s) = (1 + e_m)^s` with "`infl(0) = 1`" and 9.1 defines `disc(t) = v_m^t` as "a direct power" with "`disc(0) = 1`". Is the typed 1 the intended cell in row 0, or the power formula of every other row (`(1 + e_m)^0` and `v_m^0` are exactly 1.0 in float64, so the values agree either way)? The spreadsheet builder typed the literal 1 for `infl(0)` and `disc(0)`, as step 2 states, and `pols_if(0)` as a reference to the `pols_if_init` input cell.
15. **Range of `t` for P3 (12).** P3, "The number of months with `premium_pp(t) <> 0` equals `prem_rem_m`", names no range of `t`; section 12 checks month quantities for `t = 1 .. H`, and the policy sheet has rows to `rem_m` only (4.4). Confirm the range. The spreadsheet builder counted over the sheet's rows `t = 1 .. rem_m`, as one cell per policy, `SUMPRODUCT(--(premium_pp range <> 0)) = prem_rem_m`.
16. **The step condition of P4 at `t = 1` (12, 4.1).** P4 says "`dur_y(t)` is non-decreasing in `t` and steps by 0 or 1 between consecutive months". `dur_y` is a month quantity, blank at `t = 0` (4.1, C2), so at `t = 1` the row above holds no `dur_y` to step from. Is a step check at `t = 1`, against a duration at time 0 that the sheet does not hold, part of P4, or does the step condition start at `t = 2`? The spreadsheet builder checked the age range at every `t = 1 .. rem_m` and the step condition from `t = 2` only.
17. **The by-`t` totals as "plain SUMs" (11.3).** Section 11.3 defines `total_<col>(t)` with `where(t <= rem_m[p], col[p](t), 0)` and says that in the spreadsheet the totals "live on a summary sheet as plain SUMs across the policy sheets". The policy sheets have rows to their own `rem_m` (1 to 480 on the reference points, section 4.4), so a plain `SUM` of one cell address across the sheets cannot form a by-`t` total: the same row holds a different `t`, or a cell of the scalars or check blocks, on each sheet. Do INDEX-guarded terms count as the intended "plain SUMs", or should the sentence be amended? The spreadsheet builder put the totals by `t = 0 .. 480` on the summary sheet as the 11.3 formula literally, one term per policy, `IF(t <= rem_m[p], INDEX(<col>[p], row of t), 0)`, added in ascending `policy_id` order, month columns blank at `t = 0`; the scalar totals are plain `SUM`s down the summary sheet's scalar block.
18. **The `q_ann(M, 70)` check figure in Excel (6.1).** Section 6.1 says "check `q_ann(M, 70)` instead: 0.0141847561621227 with `(c - 1)` from the cell, 0.0141847561621226 with a typed 0.1", with `q_ann(F, 53)` as a second check, "0.00195138413143703 against 0.00195138413143692". In Excel 365 `POWER(1.1, 70)` returns 789.7469567994431, five ulps below the correctly rounded float64 value 789.7469567994436, and the cancellation in `1 - exp(-mu)` amplifies that to about 64 ulps of `q`, so the cell-computed `q_ann(M, 70)` is 0.014184756162122558, which displays as ...1226, the figure the section attributes to a typed 0.1, although the cell's `(c - 1)` is 0.10000000000000009 (verified in the workbook); `q_ann(F, 53)` is 0.0019513841314370284, displays as ...703 and matches the from-cell figure. The `(M, 70)` figure therefore cannot tell a typed 0.1 from the cell difference in Excel. Should the section say so, or rely on `(F, 53)` alone? The differences are about 1e-14 relative, far inside the tolerance of section 1. The spreadsheet builder computed `(c - 1)` and `ln(c)` from the `c` cell in every cell of the materialised table, as 6.1 instructs, and typed 0.1 nowhere.
19. **`H` on the policy sheet: H3 and P8 (12, 4.4; question 13).** H3 asserts "`reserve(rem_m) = 0` exactly, `reserve(H) = 0` exactly, and `pols_if(rem_m) = 0` exactly", and P8 sums "over `t = 1 .. H`", while 4.4 gives the policy sheet "exactly these rows" `t = 0 .. rem_m`, so on the sheet `H = rem_m` and the two reserve statements of H3 are one cell. Confirm that no common-horizon rows are wanted on the policy sheets (question 13 leaves this open; the engine checks `reserve(H)` at the batch horizon). The spreadsheet builder built H3 as one cell testing `reserve(rem_m) = 0` and `pols_if(rem_m) = 0` exactly, and P8 as a `SUMPRODUCT` of `deaths + lapses + maturities` over `t = 1 .. rem_m` compared with `pols_if_init` under the tolerance with scale `pols_if_init`.
20. **Where the materialised `q_ann` block lives (11).** Section 11 lets `q_ann` be "looked up from a materialised `(sex, age, q_ann)` block for ages 18 to 120" but names no sheet for it, while the fixed layout puts the mortality parameters `A`, `B` and `c` in every policy sheet's basis block. The spreadsheet builder put the one materialised table on a shared `mortality` sheet with its own typed `A`, `B` and `c` cells, so the policy sheets' copies of those parameters feed no formula. Should section 11 say where the block lives and accept a shared sheet, or should `q_ann` be computed inline per row from the policy sheet's own parameters, which turns the `q_ann` cell from a lookup into the formula text of 6.1?
21. **The run-level input on the policy sheet (3.3, 11).** The fixed layout lists "the inputs of 3.1" as the first block of a policy sheet, and 3.1 holds the policy's fields only; `valuation_date` is a run setting of 3.3, yet `dur_m` and the other derived constants of 3.2 read it. The spreadsheet builder typed `valuation_date` on every policy sheet, above the 3.1 fields, so that every formula on a sheet reads that sheet alone. Confirm, or say where the run-level input belongs in the fixed layout.
