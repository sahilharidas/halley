"""Build the reference workbook for the term assurance specification (D22, reference case 1).

This generator reads nothing but ``docs/spec/term-assurance.md``. The only numbers typed into
the workbook are that document's reference inputs: the valuation date of section 3.3, the basis
constants and short tables of section 6 and the ten model points of section 13. Every other cell
is an Excel formula transcribing one formula of the specification, in its stated evaluation order
and computation form (powers where it says power, ratios of two curve cells, left-to-right
subtraction, sums in ascending ``t``). No engine output, number script or other document is read
and nothing is rounded. Python computes a policy's ``rem_m`` only to know how many rows to lay
out; the ``inputs`` sheet recomputes it by formula and ``checks!rows_ok`` compares the two.

Sheets written to ``docs/spec/term-assurance/term-assurance.xlsx``:

``readme``        what each sheet and column is, with units and the spec sections.
``inputs``        valuation date; basis constants, with ``i_m``, ``v_m`` and ``e_m`` by formula;
                  the lapse and commission tables; the ten model points with their derived
                  fields computed by formula from the dates (sections 4.3 and 5); and the
                  mortality table of section 6.1 materialised for ages 18 to 120 and both sexes
                  from the Gompertz-Makeham formula (the materialised-block option of section 11,
                  looked up by INDEX/MATCH on ``(sex, age)``).
``P01`` .. ``P10``  one sheet per policy: headers in row 1, one row per ``t = 0 .. rem_m``, one
                  column per section 11.1 quantity in the section 7.1 order, then the row-level
                  check cells of section 12, then the policy's constants linked from ``inputs``
                  (sheet-scoped defined names point at that block, so formulas read as the spec).
``scalars``       the section 11.2 scalars per policy and the section 11.3 scalar totals.
``totals_t``      the section 11.3 portfolio totals by ``t`` over the batch horizon.
``checks``        one boolean per section 12 invariant per policy, a per-policy ``ok`` and the
                  ``all_ok`` cell (defined name ``all_ok``) with its label in the cell to its left.

Run from the repository root, then recalculate in Excel so the cached values exist:

    uv run --frozen python docs/spec/term_assurance_workbook.py
    uv run --frozen python docs/spec/recalc_excel.py docs/spec/term-assurance/term-assurance.xlsx
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import openpyxl
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter
from openpyxl.workbook.defined_name import DefinedName
from openpyxl.worksheet.worksheet import Worksheet

OUT = Path("docs/spec/term-assurance/term-assurance.xlsx")

# ----------------------------------------------------------------------------------------------
# Reference inputs: the only typed numbers in the workbook.
# ----------------------------------------------------------------------------------------------

# Section 3.3: the run-level input.
VALUATION_DATE = dt.date(2027, 1, 1)

# Section 6 constants in the order section 11 lists them: (symbol, value or formula, unit, section).
# ``i_m``, ``v_m`` and ``e_m`` are formulas of ``i`` and ``e``; ``v_m`` is the direct power.
BASIS: list[tuple[str, float | str, str, str]] = [
    ("i", 0.04, "Rate[Annual]", "6"),
    ("e", 0.03, "Rate[Annual]", "6"),
    ("i_m", "=(1+i)^(1/12)-1", "Rate[Monthly]", "6"),
    ("v_m", "=(1+i)^(-1/12)", "factor", "6"),
    ("e_m", "=(1+e)^(1/12)-1", "Rate[Monthly]", "6"),
    ("A_M", 0.0002, "Gompertz-Makeham A, sex M", "6.1"),
    ("A_F", 0.00015, "Gompertz-Makeham A, sex F", "6.1"),
    ("B_M", 0.000017, "Gompertz-Makeham B, sex M", "6.1"),
    ("B_F", 0.000011, "Gompertz-Makeham B, sex F", "6.1"),
    ("c", 1.1, "Gompertz-Makeham c (both sexes)", "6.1"),
    ("maint_base", 5.00, "Money per policy, per month, time-0 prices", "6.5"),
    ("claim_exp_base", 250.00, "Money per policy (per death), time-0 prices", "6.5"),
]
# Excel reserves the names R and C, so the spec's ``c`` is the defined name ``gm_c``.
EXCEL_NAME = {"c": "gm_c"}

# Section 6.2: annual lapse rate by clamped key k = min(dur_y, 5).
LAPSE_TABLE = [0.15, 0.10, 0.08, 0.06, 0.05, 0.04]
# Section 6.6: commission rate by clamped key k = min(dur_y, 1).
COMM_TABLE = [0.20, 0.02]
# Section 6.1: the table's declared age range.
MORT_AGES = range(18, 121)


@dataclass(frozen=True)
class ModelPoint:
    """One row of the section 13 table (the section 3.1 input fields)."""

    policy_id: str
    sex: str
    birth_date: dt.date
    entry_date: dt.date
    maturity_date: dt.date
    prem_term_y: int
    sum_assured: float
    premium: float
    pols_if_init: float

    @property
    def rem_m(self) -> int:
        """Section 4.3 ``rem_m``, used only to size the policy sheet (see ``checks!rows_ok``)."""
        term_m = months_between(self.entry_date, self.maturity_date)
        dur_m = months_between(self.entry_date, VALUATION_DATE)
        return term_m - dur_m


def months_between(d1: dt.date, d2: dt.date) -> int:
    """Section 4.2: whole calendar months from ``d1`` to ``d2``."""
    return 12 * (d2.year - d1.year) + (d2.month - d1.month) - (1 if d2.day < d1.day else 0)


def d(iso: str) -> dt.date:
    return dt.date.fromisoformat(iso)


# Section 13, exactly as printed.
MODEL_POINTS = [
    ModelPoint("P01", "M", d("1986-03-15"), d("2024-07-01"), d("2044-07-01"), 20, 250000, 52.96, 1),
    ModelPoint("P02", "F", d("1991-11-02"), d("2027-01-01"), d("2052-01-01"), 25, 400000, 51.64, 1),
    ModelPoint("P03", "M", d("1966-05-20"), d("2007-01-15"), d("2027-01-15"), 20, 150000, 39.99, 1),
    ModelPoint("P04", "F", d("1976-08-09"), d("2015-03-20"), d("2045-03-20"), 20, 300000, 68.98, 1),
    ModelPoint("P05", "M", d("1975-04-18"), d("2018-02-10"), d("2028-02-10"), 9, 200000, 46.40, 1),
    ModelPoint("P06", "F", d("2003-02-28"), d("2026-11-10"), d("2036-11-10"), 10, 100000, 10.76, 1),
    ModelPoint(
        "P07", "M", d("1980-06-01"), d("2020-02-29"), d("2040-02-29"), 20, 500000, 105.55, 2.5
    ),
    ModelPoint(
        "P08", "M", d("1958-12-31"), d("2010-12-31"), d("2030-12-31"), 10, 120000, 128.12, 1
    ),
    ModelPoint(
        "P09", "F", d("1996-10-31"), d("2026-12-15"), d("2066-12-15"), 40, 1000000, 116.62, 1
    ),
    ModelPoint("P10", "M", d("1990-05-05"), d("2022-02-01"), d("2032-02-01"), 5, 200000, 39.27, 1),
]

# ----------------------------------------------------------------------------------------------
# Layout constants.
# ----------------------------------------------------------------------------------------------

# inputs sheet: fixed rows of each block.
VALUATION_ROW = 2
BASIS_FIRST_ROW = 5
LAPSE_HEADER_ROW = 19
COMM_HEADER_ROW = 28
MP_HEADER_ROW = 33
MP_FIRST_ROW = 34
MORT_HEADER_ROW = 46
MORT_FIRST_ROW = 47

# inputs sheet: model point columns, inputs (3.1) then derived fields (3.2) in that order.
MP_INPUT_FIELDS = [
    "policy_id", "sex", "birth_date", "entry_date", "maturity_date", "prem_term_y",
    "sum_assured", "premium", "pols_if_init",
]  # fmt: skip
MP_DERIVED_FIELDS = [
    "age_at_entry", "dur_m", "term_m", "rem_m", "prem_term_m", "prem_rem_m", "dur_y_last",
]  # fmt: skip
MP_COL = {
    name: get_column_letter(k + 1) for k, name in enumerate(MP_INPUT_FIELDS + MP_DERIVED_FIELDS)
}

# Policy sheet: the section 11.1 columns in the section 7.1 evaluation order, with units.
QUANTITIES: list[tuple[str, str, str]] = [
    ("t", "Int", "4.1"),
    ("dur_y", "Int", "5"),
    ("age", "Int", "5"),
    ("q_ann", "Prob annual", "6.1"),
    ("q_m", "Prob monthly", "6.1"),
    ("w_ann", "Prob annual", "6.2"),
    ("w_m", "Prob monthly", "6.2"),
    ("comm_rate", "Fraction", "6.6"),
    ("infl", "factor (state; infl(0) = 1)", "6.4"),
    ("deaths", "Count", "7.3"),
    ("lapses", "Count", "7.3"),
    ("maturities", "Count", "7.3"),
    ("pols_if", "Count (state)", "7.4"),
    ("premium_pp", "Money per policy", "8.1"),
    ("commission_pp", "Money per policy", "8.1"),
    ("maint_pp", "Money per policy", "8.1"),
    ("claim_exp_pp", "Money per policy", "8.1"),
    ("premium_income", "Money, start", "8.2"),
    ("commission", "Money, start", "8.2"),
    ("maint_expense", "Money, end", "8.2"),
    ("death_outgo", "Money, end", "8.2"),
    ("claim_expense", "Money, end", "8.2"),
    ("net_cf", "Money, reporting (no timing)", "8.3"),
    ("net_outgo", "Money, end", "8.3"),
    ("v", "factor", "9.1"),
    ("disc", "factor (state; disc(0) = 1)", "9.1"),
    ("pv_net_cf_t", "Money", "9.3"),
    ("pv_premiums_t", "Money", "9.3"),
    ("pv_commission_t", "Money", "9.3"),
    ("pv_maint_expense_t", "Money", "9.3"),
    ("pv_death_outgo_t", "Money", "9.3"),
    ("pv_claim_expense_t", "Money", "9.3"),
    ("reserve", "Money (state)", "10.2"),
    ("reserve_pp", "Money per policy (state)", "10.4"),
    ("reserve_floored", "Money (state)", "10.5"),
]

# Policy sheet: row-level check cells of section 12, after one blank column.
ROW_CHECKS: list[tuple[str, str]] = [
    ("H1", "pols_if >= 0 and non-increasing, exactly (t = 0 .. rem_m)"),
    (
        "H2",
        "rates in [0, 1] exactly; deaths + lapses + maturities <= pols_if(t-1) within tolerance",
    ),
    ("H4_lhs", "helper: roll-forward left-hand side"),
    ("H4", "roll-forward equals reserve(t) within tolerance"),
    ("H5b_term", "helper: month-s term of the H5b sum"),
    ("H5b_sum", "helper: sum of H5b_term over s = t+1 .. rem_m, ascending"),
    ("H5b", "reserve(t) * disc(t) equals the H5b sum within tolerance (t = 0 .. rem_m - 1)"),
    ("P1", "maturities exactly 0 except at rem_m, where it equals the survivors exactly"),
    ("P2", "premium, commission and lapse quantities exactly 0 after prem_rem_m; premium_income"),
    ("P4", "age in 18 .. 120; dur_y steps by 0 or 1"),
    ("P5_rhs", "helper: per-policy recursion right-hand side"),
    ("P5", "reserve_pp(t) equals the per-policy recursion within tolerance (scale sum_assured)"),
    ("P6", "reserve_floored >= 0 and >= reserve, exactly"),
    ("P10", "unit identities, exactly"),
]
FIRST_CHECK_COL = len(QUANTITIES) + 2  # one blank column after the quantities
COL = {name: get_column_letter(k + 1) for k, (name, _, _) in enumerate(QUANTITIES)}
COL.update({name: get_column_letter(FIRST_CHECK_COL + k) for k, (name, _) in enumerate(ROW_CHECKS)})

# Policy sheet: the constants block (labels and values), after one more blank column.
CONST_LABEL_COL = get_column_letter(FIRST_CHECK_COL + len(ROW_CHECKS) + 1)
CONST_VALUE_COL = get_column_letter(FIRST_CHECK_COL + len(ROW_CHECKS) + 2)
CONST_FIELDS = MP_INPUT_FIELDS + MP_DERIVED_FIELDS
# Section 1 scales, from the policy's own sum_assured and pols_if_init cells.
SCALES = [
    ("scale_money", "=sum_assured*pols_if_init", "Money (in-force) scale"),
    ("scale_pp", "=sum_assured", "Money per policy scale"),
    ("scale_count", "=pols_if_init", "Count scale"),
]

# scalars sheet columns (section 11.2).
SCALAR_COLUMNS = [
    "policy_id", "age_at_entry", "dur_m", "term_m", "rem_m", "prem_term_m", "prem_rem_m",
    "dur_y_last", "pv_premiums", "pv_commission", "pv_maint_expense", "pv_death_outgo",
    "pv_claim_expense", "pv_net_cf", "reserve_0", "reserve_pp_0", "reserve_floored_0",
    "maturities_n",
]  # fmt: skip
SCOL = {name: get_column_letter(k + 1) for k, name in enumerate(SCALAR_COLUMNS)}
PV_SUMS = {
    "pv_premiums": "pv_premiums_t",
    "pv_commission": "pv_commission_t",
    "pv_maint_expense": "pv_maint_expense_t",
    "pv_death_outgo": "pv_death_outgo_t",
    "pv_claim_expense": "pv_claim_expense_t",
}
SCALAR_TOTALS = [
    "pv_premiums", "pv_commission", "pv_maint_expense", "pv_death_outgo", "pv_claim_expense",
    "pv_net_cf", "reserve_0", "reserve_floored_0",
]  # fmt: skip

# totals_t sheet: the Money and Count monthly columns of 11.1 (per-policy Money is not summed).
STATE_TOTALS = ["pols_if", "reserve", "reserve_floored"]
MONTH_TOTALS = [
    "deaths", "lapses", "maturities", "premium_income", "commission", "maint_expense",
    "death_outgo", "claim_expense", "net_cf", "net_outgo", "pv_net_cf_t", "pv_premiums_t",
    "pv_commission_t", "pv_maint_expense_t", "pv_death_outgo_t", "pv_claim_expense_t",
]  # fmt: skip
TOTAL_COLUMNS = [
    "deaths", "lapses", "maturities", "pols_if", "premium_income", "commission",
    "maint_expense", "death_outgo", "claim_expense", "net_cf", "net_outgo", "pv_net_cf_t",
    "pv_premiums_t", "pv_commission_t", "pv_maint_expense_t", "pv_death_outgo_t",
    "pv_claim_expense_t", "reserve", "reserve_floored",
]  # fmt: skip

# checks sheet columns: the section 12 invariants the sheet can hold, a layout check, the
# per-policy ``ok``, then helper scalars.
CHECK_COLUMNS = [
    "policy_id", "H1", "H2", "H3", "H4", "H5a", "H5b", "P1", "P2", "P3", "P4", "P5", "P6",
    "P8", "P9", "P10", "rows_ok", "ok", "", "exits_sum", "pv_net_cf_t_sum", "scale_money",
    "scale_count",
]  # fmt: skip
CCOL = {name: get_column_letter(k + 1) for k, name in enumerate(CHECK_COLUMNS) if name}

BOLD = Font(bold=True)


class Row:
    """Cell addresses of one row of a policy sheet by spec symbol: ``Row(3).q_m == "E3"``."""

    def __init__(self, r: int) -> None:
        self.r = r

    def __getattr__(self, name: str) -> str:
        return f"{COL[name]}{self.r}"


def define(wb: openpyxl.Workbook, name: str, ref: str) -> None:
    """Add a workbook-level defined name for ``ref``."""
    wb.defined_names[name] = DefinedName(name, attr_text=ref)


def define_local(ws: Worksheet, name: str, ref: str) -> None:
    """Add a defined name scoped to ``ws``: formulas on that sheet resolve it, others do not."""
    ws.defined_names[name] = DefinedName(name, attr_text=ref)


def rng(sheet: str, name: str, first: int, last: int) -> str:
    """Absolute range of column ``name`` on ``sheet`` from row ``first`` to row ``last``."""
    return f"{sheet}!${COL[name]}${first}:${COL[name]}${last}"


def set_widths(ws: Worksheet, width: float, upto: int) -> None:
    for k in range(1, upto + 1):
        ws.column_dimensions[get_column_letter(k)].width = width


# ----------------------------------------------------------------------------------------------
# readme
# ----------------------------------------------------------------------------------------------


def write_readme(wb: openpyxl.Workbook) -> None:
    ws = wb.active
    ws.title = "readme"
    lines = [
        "Halley term assurance: reference workbook for docs/spec/term-assurance.md (version 0.1).",
        "Built from the specification alone: typed cells are its reference inputs (sections 3.3,"
        " 6, 13); every other cell is a formula transcribing one formula of the spec.",
        "",
        "Sheets",
        "inputs: valuation date, basis constants, lapse and commission tables, model points with"
        " derived fields by formula, mortality table materialised from the 6.1 formula.",
        "P01 .. P10: one sheet per policy, headers in row 1, one row per t = 0 .. rem_m, one"
        " column per 11.1 quantity in the 7.1 order; row checks of section 12 to the right;"
        " then the policy's constants linked from inputs (sheet-scoped names point at them).",
        "scalars: section 11.2 scalars per policy (pv sums over t = 1 .. rem_m) and the"
        " section 11.3 scalar totals in the row labelled total.",
        "totals_t: section 11.3 totals by t = 0 .. H over the ten policies, policies beyond"
        " their own rem_m read as zero.",
        "checks: one boolean per section 12 invariant per policy, rows_ok (the sheet's row count"
        " equals the formula-derived rem_m), ok per policy, and all_ok.",
        "",
        "Conventions",
        "q_ann is looked up from the materialised (sex, age, q_ann) block on inputs, ages 18 to"
        " 120, each cell the section 6.1 formula in its stated order (section 11 option 2).",
        "Row t = 0 holds the state variables only (pols_if, infl, disc, reserve, reserve_pp,"
        " reserve_floored); month quantities are blank there (section 4.1).",
        "Tolerance (section 1): |a - b| <= 1e-9 * max(|a|, |b|, scale); inequalities use"
        " a <= b + 1e-9 * scale; 'exactly' is a plain = comparison.",
        "P7 and P11 are engine-only (rows beyond rem_m, batch independence) and have no cell.",
        "The spec's c is the defined name gm_c because Excel reserves the names R and C.",
        "",
        "Policy sheet columns (symbol, unit, spec section)",
    ]
    for k, line in enumerate(lines, start=1):
        ws.cell(row=k, column=1, value=line)
    row = len(lines) + 1
    for name, unit, section in QUANTITIES:
        ws.cell(row=row, column=1, value=name)
        ws.cell(row=row, column=2, value=unit)
        ws.cell(row=row, column=3, value=section)
        row += 1
    row += 1
    ws.cell(row=row, column=1, value="Policy sheet check columns (section 12)")
    row += 1
    for name, meaning in ROW_CHECKS:
        ws.cell(row=row, column=1, value=name)
        ws.cell(row=row, column=2, value=meaning)
        row += 1
    ws.column_dimensions["A"].width = 24
    ws.column_dimensions["B"].width = 60


# ----------------------------------------------------------------------------------------------
# inputs
# ----------------------------------------------------------------------------------------------


def write_inputs(wb: openpyxl.Workbook) -> None:
    ws = wb.create_sheet("inputs")

    ws["A1"] = "Run settings (section 3.3)"
    ws["A1"].font = BOLD
    ws[f"A{VALUATION_ROW}"] = "valuation_date"
    ws[f"B{VALUATION_ROW}"] = VALUATION_DATE
    ws[f"B{VALUATION_ROW}"].number_format = "yyyy-mm-dd"
    ws[f"C{VALUATION_ROW}"] = "Date, first of a month"
    define(wb, "valuation_date", f"inputs!$B${VALUATION_ROW}")

    ws["A4"] = "Basis constants (section 6), each in one cell"
    ws["A4"].font = BOLD
    for k, (symbol, value, unit, section) in enumerate(BASIS):
        row = BASIS_FIRST_ROW + k
        ws[f"A{row}"] = symbol
        ws[f"B{row}"] = value
        ws[f"C{row}"] = unit
        ws[f"D{row}"] = section
        define(wb, EXCEL_NAME.get(symbol, symbol), f"inputs!$B${row}")

    ws[f"A{LAPSE_HEADER_ROW - 1}"] = "Lapse table (section 6.2): key k = min(dur_y, 5)"
    ws[f"A{LAPSE_HEADER_ROW - 1}"].font = BOLD
    ws[f"A{LAPSE_HEADER_ROW}"] = "k"
    ws[f"B{LAPSE_HEADER_ROW}"] = "lapse_table"
    for k, w_ann in enumerate(LAPSE_TABLE):
        ws[f"A{LAPSE_HEADER_ROW + 1 + k}"] = k
        ws[f"B{LAPSE_HEADER_ROW + 1 + k}"] = w_ann
    first, last = LAPSE_HEADER_ROW + 1, LAPSE_HEADER_ROW + len(LAPSE_TABLE)
    define(wb, "lapse_key", f"inputs!$A${first}:$A${last}")
    define(wb, "lapse_table", f"inputs!$B${first}:$B${last}")

    ws[f"A{COMM_HEADER_ROW - 1}"] = "Commission table (section 6.6): key k = min(dur_y, 1)"
    ws[f"A{COMM_HEADER_ROW - 1}"].font = BOLD
    ws[f"A{COMM_HEADER_ROW}"] = "k"
    ws[f"B{COMM_HEADER_ROW}"] = "comm_table"
    for k, rate in enumerate(COMM_TABLE):
        ws[f"A{COMM_HEADER_ROW + 1 + k}"] = k
        ws[f"B{COMM_HEADER_ROW + 1 + k}"] = rate
    first, last = COMM_HEADER_ROW + 1, COMM_HEADER_ROW + len(COMM_TABLE)
    define(wb, "comm_key", f"inputs!$A${first}:$A${last}")
    define(wb, "comm_table", f"inputs!$B${first}:$B${last}")

    ws[f"A{MP_HEADER_ROW - 1}"] = (
        "Model points (section 13) and derived fields by formula (sections 4.3 and 5)"
    )
    ws[f"A{MP_HEADER_ROW - 1}"].font = BOLD
    for name in MP_INPUT_FIELDS + MP_DERIVED_FIELDS:
        ws[f"{MP_COL[name]}{MP_HEADER_ROW}"] = name
        ws[f"{MP_COL[name]}{MP_HEADER_ROW}"].font = BOLD
    for k, mp in enumerate(MODEL_POINTS):
        write_model_point(ws, mp, MP_FIRST_ROW + k)

    ws[f"A{MORT_HEADER_ROW - 1}"] = (
        "Mortality table q_ann(sex, x) (section 6.1): 1 - exp(-(A + B * c^x * (c - 1) / ln(c)))"
    )
    ws[f"A{MORT_HEADER_ROW - 1}"].font = BOLD
    ws[f"A{MORT_HEADER_ROW}"] = "x"
    ws[f"B{MORT_HEADER_ROW}"] = "M"
    ws[f"C{MORT_HEADER_ROW}"] = "F"
    for k, age in enumerate(MORT_AGES):
        row = MORT_FIRST_ROW + k
        ws[f"A{row}"] = age
        ws[f"B{row}"] = f"=1-EXP(-(A_M+B_M*gm_c^$A{row}*(gm_c-1)/LN(gm_c)))"
        ws[f"C{row}"] = f"=1-EXP(-(A_F+B_F*gm_c^$A{row}*(gm_c-1)/LN(gm_c)))"
    last = MORT_FIRST_ROW + len(MORT_AGES) - 1
    define(wb, "mort_age", f"inputs!$A${MORT_FIRST_ROW}:$A${last}")
    define(wb, "mort_sex", f"inputs!$B${MORT_HEADER_ROW}:$C${MORT_HEADER_ROW}")
    define(wb, "mort_table", f"inputs!$B${MORT_FIRST_ROW}:$C${last}")

    set_widths(ws, 14, len(MP_COL))
    ws.column_dimensions["A"].width = 16


def write_model_point(ws: Worksheet, mp: ModelPoint, row: int) -> None:
    """One model point row: the 3.1 inputs typed, the 3.2 derived fields by formula."""
    for name in MP_INPUT_FIELDS:
        cell = ws[f"{MP_COL[name]}{row}"]
        cell.value = getattr(mp, name)
        if isinstance(cell.value, dt.date):
            cell.number_format = "yyyy-mm-dd"
    birth, entry, mat = (f"{MP_COL[f]}{row}" for f in ("birth_date", "entry_date", "maturity_date"))
    prem_term_y = f"{MP_COL['prem_term_y']}{row}"
    dur_m, term_m = f"{MP_COL['dur_m']}{row}", f"{MP_COL['term_m']}{row}"
    prem_term_m = f"{MP_COL['prem_term_m']}{row}"
    val = "valuation_date"
    # Section 5: age last birthday at entry, lexicographic (month, day) comparison.
    ws[f"{MP_COL['age_at_entry']}{row}"] = (
        f"=YEAR({entry})-YEAR({birth})"
        f"-IF(OR(MONTH({entry})<MONTH({birth}),"
        f"AND(MONTH({entry})=MONTH({birth}),DAY({entry})<DAY({birth}))),1,0)"
    )
    # Section 4.2 months_between, applied as section 4.3 says.
    ws[dur_m] = (
        f"=12*(YEAR({val})-YEAR({entry}))+(MONTH({val})-MONTH({entry}))"
        f"-IF(DAY({val})<DAY({entry}),1,0)"
    )
    ws[term_m] = (
        f"=12*(YEAR({mat})-YEAR({entry}))+(MONTH({mat})-MONTH({entry}))"
        f"-IF(DAY({mat})<DAY({entry}),1,0)"
    )
    ws[f"{MP_COL['rem_m']}{row}"] = f"={term_m}-{dur_m}"
    ws[prem_term_m] = f"=12*{prem_term_y}"
    ws[f"{MP_COL['prem_rem_m']}{row}"] = f"=MAX({prem_term_m}-{dur_m},0)"
    ws[f"{MP_COL['dur_y_last']}{row}"] = f"=INT(({term_m}-1)/12)"


# ----------------------------------------------------------------------------------------------
# One sheet per policy
# ----------------------------------------------------------------------------------------------


def write_policy(wb: openpyxl.Workbook, mp: ModelPoint, mp_row: int) -> None:
    ws = wb.create_sheet(mp.policy_id)
    for name, _, _ in QUANTITIES:
        ws[f"{COL[name]}1"] = name
        ws[f"{COL[name]}1"].font = BOLD
    for name, _ in ROW_CHECKS:
        ws[f"{COL[name]}1"] = name
        ws[f"{COL[name]}1"].font = BOLD
    write_constants_block(ws, mp_row)
    last = mp.rem_m + 2
    for t in range(mp.rem_m + 1):
        write_policy_row(ws, t, mp.rem_m)
    for t in range(mp.rem_m + 1):
        write_check_row(ws, t, mp.rem_m, last)
    ws.freeze_panes = "B2"
    set_widths(ws, 15, FIRST_CHECK_COL + len(ROW_CHECKS) + 2)
    ws.column_dimensions[COL["t"]].width = 6


def write_constants_block(ws: Worksheet, mp_row: int) -> None:
    """The policy's inputs, derived fields and scales, linked from ``inputs`` and named."""
    ws[f"{CONST_LABEL_COL}1"] = "constant"
    ws[f"{CONST_VALUE_COL}1"] = "value (linked from inputs)"
    ws[f"{CONST_LABEL_COL}1"].font = BOLD
    ws[f"{CONST_VALUE_COL}1"].font = BOLD
    row = 2
    for name in CONST_FIELDS:
        ws[f"{CONST_LABEL_COL}{row}"] = name
        ws[f"{CONST_VALUE_COL}{row}"] = f"=inputs!${MP_COL[name]}${mp_row}"
        if name.endswith("_date"):
            ws[f"{CONST_VALUE_COL}{row}"].number_format = "yyyy-mm-dd"
        if name != "policy_id":
            define_local(ws, name, f"{ws.title}!${CONST_VALUE_COL}${row}")
        row += 1
    row += 1
    ws[f"{CONST_LABEL_COL}{row}"] = "section 1 scales"
    row += 1
    for name, formula, _ in SCALES:
        ws[f"{CONST_LABEL_COL}{row}"] = name
        ws[f"{CONST_VALUE_COL}{row}"] = formula
        define_local(ws, name, f"{ws.title}!${CONST_VALUE_COL}${row}")
        row += 1


def write_policy_row(ws: Worksheet, t: int, rem_m: int) -> None:
    """Row ``t`` of the monthly block (section 7.1 steps 2 to 4)."""
    cur, prv, nxt = Row(t + 2), Row(t + 1), Row(t + 3)
    ws[cur.t] = t
    if t == 0:
        # Step 2: state variables take their initial values; month quantities stay blank.
        ws[cur.infl] = 1
        ws[cur.pols_if] = "=pols_if_init"
        ws[cur.disc] = 1
    else:
        # Step 3: each row from its own t and row t-1 only, columns left to right.
        ws[cur.dur_y] = f"=MIN(INT((dur_m+{cur.t}-1)/12),dur_y_last)"
        ws[cur.age] = f"=age_at_entry+{cur.dur_y}"
        ws[cur.q_ann] = f"=INDEX(mort_table,MATCH({cur.age},mort_age,0),MATCH(sex,mort_sex,0))"
        ws[cur.q_m] = f"=IF({cur.t}<=rem_m,1-(1-{cur.q_ann})^(1/12),0)"
        ws[cur.w_ann] = (
            f"=IF({cur.t}<=prem_rem_m,INDEX(lapse_table,MATCH(MIN({cur.dur_y},5),lapse_key,0)),0)"
        )
        ws[cur.w_m] = f"=1-(1-{cur.w_ann})^(1/12)"
        ws[cur.comm_rate] = f"=INDEX(comm_table,MATCH(MIN({cur.dur_y},1),comm_key,0))"
        ws[cur.infl] = f"=(1+e_m)^{cur.t}"
        ws[cur.deaths] = f"={prv.pols_if}*{cur.q_m}"
        ws[cur.lapses] = f"=({prv.pols_if}-{cur.deaths})*{cur.w_m}"
        ws[cur.maturities] = f"=IF({cur.t}=rem_m,{prv.pols_if}-{cur.deaths}-{cur.lapses},0)"
        ws[cur.pols_if] = f"={prv.pols_if}-{cur.deaths}-{cur.lapses}-{cur.maturities}"
        ws[cur.premium_pp] = f"=IF({cur.t}<=prem_rem_m,premium,0)"
        ws[cur.commission_pp] = f"={cur.premium_pp}*{cur.comm_rate}"
        ws[cur.maint_pp] = f"=maint_base*{cur.infl}"
        ws[cur.claim_exp_pp] = f"=claim_exp_base*{cur.infl}"
        ws[cur.premium_income] = f"={prv.pols_if}*{cur.premium_pp}"
        ws[cur.commission] = f"={cur.premium_income}*{cur.comm_rate}"
        ws[cur.maint_expense] = f"={prv.pols_if}*{cur.maint_pp}"
        ws[cur.death_outgo] = f"={cur.deaths}*sum_assured"
        ws[cur.claim_expense] = f"={cur.deaths}*{cur.claim_exp_pp}"
        ws[cur.net_cf] = (
            f"={cur.premium_income}-{cur.commission}-{cur.maint_expense}"
            f"-{cur.death_outgo}-{cur.claim_expense}"
        )
        ws[cur.net_outgo] = (
            f"={cur.death_outgo}+{cur.claim_expense}+{cur.maint_expense}"
            f"-({cur.premium_income}-{cur.commission})*(1+i_m)"
        )
        ws[cur.v] = "=v_m"
        ws[cur.disc] = f"=v_m^{cur.t}"
        ws[cur.pv_net_cf_t] = (
            f"=({cur.premium_income}-{cur.commission})*{prv.disc}"
            f"-({cur.maint_expense}+{cur.death_outgo}+{cur.claim_expense})*{cur.disc}"
        )
        ws[cur.pv_premiums_t] = f"={cur.premium_income}*{prv.disc}"
        ws[cur.pv_commission_t] = f"={cur.commission}*{prv.disc}"
        ws[cur.pv_maint_expense_t] = f"={cur.maint_expense}*{cur.disc}"
        ws[cur.pv_death_outgo_t] = f"={cur.death_outgo}*{cur.disc}"
        ws[cur.pv_claim_expense_t] = f"={cur.claim_expense}*{cur.disc}"
    # Step 4: the reserve by backward recursion from reserve(rem_m) = 0, then the two derived.
    if t == rem_m:
        ws[cur.reserve] = 0
    else:
        ws[cur.reserve] = f"=({nxt.reserve}+{nxt.net_outgo})*{nxt.v}"
    ws[cur.reserve_pp] = f"=IF({cur.pols_if}>0,{cur.reserve}/{cur.pols_if},0)"
    ws[cur.reserve_floored] = f"=IF({cur.reserve}>0,{cur.reserve},0)"


def write_check_row(ws: Worksheet, t: int, rem_m: int, last: int) -> None:
    """Row ``t`` of the section 12 row-level check cells."""
    cur, prv, nxt = Row(t + 2), Row(t + 1), Row(t + 3)
    tol = "1E-9*"
    if t == 0:
        ws[cur.H1] = f"={cur.pols_if}>=0"
    else:
        ws[cur.H1] = f"=AND({cur.pols_if}>=0,{cur.pols_if}<={prv.pols_if})"
        ws[cur.H2] = (
            f"=AND({cur.q_ann}>=0,{cur.q_ann}<=1,{cur.q_m}>=0,{cur.q_m}<=1,"
            f"{cur.w_ann}>=0,{cur.w_ann}<=1,{cur.w_m}>=0,{cur.w_m}<=1,"
            f"{cur.deaths}+{cur.lapses}+{cur.maturities}<={prv.pols_if}+{tol}scale_count)"
        )
        ws[cur.H4_lhs] = (
            f"=({prv.reserve}+{cur.premium_income}-{cur.commission})*(1+i_m)"
            f"-{cur.maint_expense}-{cur.death_outgo}-{cur.claim_expense}"
        )
        ws[cur.H4] = (
            f"=ABS({cur.H4_lhs}-{cur.reserve})"
            f"<={tol}MAX(ABS({cur.H4_lhs}),ABS({cur.reserve}),scale_money)"
        )
        ws[cur.H5b_term] = (
            f"=({cur.death_outgo}+{cur.claim_expense}+{cur.maint_expense})*{cur.disc}"
            f"-({cur.premium_income}-{cur.commission})*{prv.disc}"
        )
        ws[cur.P1] = (
            f"=IF({cur.t}<>rem_m,{cur.maturities}=0,"
            f"{cur.maturities}={prv.pols_if}-{cur.deaths}-{cur.lapses})"
        )
        ws[cur.P2] = (
            f"=IF({cur.t}>prem_rem_m,"
            f"AND({cur.premium_income}=0,{cur.commission}=0,{cur.w_ann}=0,{cur.w_m}=0,"
            f"{cur.lapses}=0),{cur.premium_income}={prv.pols_if}*premium)"
        )
        age_ok = f"{cur.age}>=18,{cur.age}<=120"
        if t == 1:
            ws[cur.P4] = f"=AND({age_ok})"
        else:
            step = f"{cur.dur_y}-{prv.dur_y}"
            ws[cur.P4] = f"=AND({age_ok},OR({step}=0,{step}=1))"
        ws[cur.P10] = (
            f"=AND({cur.death_outgo}={cur.deaths}*sum_assured,"
            f"{cur.premium_income}={prv.pols_if}*{cur.premium_pp},"
            f"{cur.maint_expense}={prv.pols_if}*{cur.maint_pp})"
        )
    if t < rem_m:
        ws[cur.H5b_sum] = f"=SUM({nxt.H5b_term}:{COL['H5b_term']}${last})"
        lhs = f"{cur.reserve}*{cur.disc}"
        ws[cur.H5b] = (
            f"=ABS({lhs}-{cur.H5b_sum})<={tol}MAX(ABS({lhs}),ABS({cur.H5b_sum}),scale_money)"
        )
        ws[cur.P5_rhs] = (
            f"=-({nxt.premium_pp}-{nxt.commission_pp})"
            f"+v_m*({nxt.q_m}*(sum_assured+{nxt.claim_exp_pp})+{nxt.maint_pp}"
            f"+(1-{nxt.q_m})*(1-{nxt.w_m})*{nxt.reserve_pp})"
        )
        ws[cur.P5] = (
            f"=ABS({cur.reserve_pp}-{cur.P5_rhs})"
            f"<={tol}MAX(ABS({cur.reserve_pp}),ABS({cur.P5_rhs}),scale_pp)"
        )
    ws[cur.P6] = f"=AND({cur.reserve_floored}>=0,{cur.reserve_floored}>={cur.reserve})"


# ----------------------------------------------------------------------------------------------
# scalars, totals_t, checks
# ----------------------------------------------------------------------------------------------


def write_scalars(wb: openpyxl.Workbook) -> None:
    ws = wb.create_sheet("scalars")
    for name in SCALAR_COLUMNS:
        ws[f"{SCOL[name]}1"] = name
        ws[f"{SCOL[name]}1"].font = BOLD
    for k, mp in enumerate(MODEL_POINTS):
        row, mp_row, last = 2 + k, MP_FIRST_ROW + k, mp.rem_m + 2
        pid = mp.policy_id
        for name in ["policy_id", *MP_DERIVED_FIELDS]:
            ws[f"{SCOL[name]}{row}"] = f"=inputs!${MP_COL[name]}${mp_row}"
        # Section 9.3: each PV is the sum of its column over t = 1 .. rem_m.
        for name, column in PV_SUMS.items():
            ws[f"{SCOL[name]}{row}"] = f"=SUM({rng(pid, column, 3, last)})"
        ws[f"{SCOL['pv_net_cf']}{row}"] = "=" + "-".join(f"{SCOL[name]}{row}" for name in PV_SUMS)
        ws[f"{SCOL['reserve_0']}{row}"] = f"={pid}!${COL['reserve']}$2"
        ws[f"{SCOL['reserve_pp_0']}{row}"] = f"={pid}!${COL['reserve_pp']}$2"
        ws[f"{SCOL['reserve_floored_0']}{row}"] = f"={pid}!${COL['reserve_floored']}$2"
        ws[f"{SCOL['maturities_n']}{row}"] = f"={pid}!${COL['maturities']}${last}"
    # Section 11.3: Money scalars summed over policies in ascending policy_id order.
    total_row = 2 + len(MODEL_POINTS)
    ws[f"A{total_row}"] = "total"
    ws[f"A{total_row}"].font = BOLD
    for name in SCALAR_TOTALS:
        col = SCOL[name]
        ws[f"{col}{total_row}"] = f"=SUM({col}2:{col}{total_row - 1})"
    set_widths(ws, 16, len(SCALAR_COLUMNS))


def write_totals_t(wb: openpyxl.Workbook) -> None:
    ws = wb.create_sheet("totals_t")
    horizon = max(mp.rem_m for mp in MODEL_POINTS)
    ws["A1"] = "t"
    ws["A1"].font = BOLD
    letters = {name: get_column_letter(k + 2) for k, name in enumerate(TOTAL_COLUMNS)}
    for name, letter in letters.items():
        ws[f"{letter}1"] = f"total_{name}"
        ws[f"{letter}1"].font = BOLD
    for t in range(horizon + 1):
        row = t + 2
        ws[f"A{row}"] = t
        for name, letter in letters.items():
            if t == 0 and name not in STATE_TOTALS:
                continue  # month quantities are blank at t = 0 and are not totalled there
            terms = []
            for k, mp in enumerate(MODEL_POINTS):
                rem = f"inputs!${MP_COL['rem_m']}${MP_FIRST_ROW + k}"
                src = f"{mp.policy_id}!${COL[name]}:${COL[name]}"
                terms.append(f"IF($A{row}<={rem},INDEX({src},$A{row}+2),0)")
            ws[f"{letter}{row}"] = "=" + "+".join(terms)
    ws.freeze_panes = "B2"
    set_widths(ws, 18, len(TOTAL_COLUMNS) + 1)


def write_checks(wb: openpyxl.Workbook) -> None:
    ws = wb.create_sheet("checks")
    for name, letter in CCOL.items():
        ws[f"{letter}1"] = name
        ws[f"{letter}1"].font = BOLD
    tol = "1E-9*"
    for k, mp in enumerate(MODEL_POINTS):
        row, mp_row, last = 2 + k, MP_FIRST_ROW + k, mp.rem_m + 2
        pid, srow = mp.policy_id, 2 + k  # the policy's row on scalars is the same row
        c = {name: f"{letter}{row}" for name, letter in CCOL.items()}
        reserve_0 = f"scalars!${SCOL['reserve_0']}${srow}"
        pv_net_cf = f"scalars!${SCOL['pv_net_cf']}${srow}"
        ws[c["policy_id"]] = f"=inputs!$A${mp_row}"
        ws[c["H1"]] = f"=AND({rng(pid, 'H1', 2, last)})"
        ws[c["H2"]] = f"=AND({rng(pid, 'H2', 3, last)})"
        ws[c["H3"]] = f"=AND({pid}!${COL['reserve']}${last}=0,{pid}!${COL['pols_if']}${last}=0)"
        ws[c["H4"]] = f"=AND({rng(pid, 'H4', 3, last)})"
        ws[c["H5a"]] = (
            f"=ABS({reserve_0}-(-{pv_net_cf}))"
            f"<={tol}MAX(ABS({reserve_0}),ABS({pv_net_cf}),{c['scale_money']})"
        )
        ws[c["H5b"]] = f"=AND({rng(pid, 'H5b', 2, last - 1)})"
        ws[c["P1"]] = f"=AND({rng(pid, 'P1', 3, last)})"
        ws[c["P2"]] = f"=AND({rng(pid, 'P2', 3, last)})"
        ws[c["P3"]] = (
            f"=SUMPRODUCT(--({rng(pid, 'premium_pp', 3, last)}<>0))"
            f"=inputs!${MP_COL['prem_rem_m']}${mp_row}"
        )
        ws[c["P4"]] = f"=AND({rng(pid, 'P4', 3, last)})"
        ws[c["P5"]] = f"=AND({rng(pid, 'P5', 2, last - 1)})"
        ws[c["P6"]] = f"=AND({rng(pid, 'P6', 2, last)})"
        ws[c["P8"]] = (
            f"=ABS({c['exits_sum']}-{c['scale_count']})"
            f"<={tol}MAX(ABS({c['exits_sum']}),ABS({c['scale_count']}),{c['scale_count']})"
        )
        ws[c["P9"]] = (
            f"=ABS({pv_net_cf}-{c['pv_net_cf_t_sum']})"
            f"<={tol}MAX(ABS({pv_net_cf}),ABS({c['pv_net_cf_t_sum']}),{c['scale_money']})"
        )
        ws[c["P10"]] = f"=AND({rng(pid, 'P10', 3, last)})"
        ws[c["rows_ok"]] = f"=MAX({pid}!$A:$A)=inputs!${MP_COL['rem_m']}${mp_row}"
        ws[c["ok"]] = f"=AND({c['H1']}:{c['rows_ok']})"
        ws[c["exits_sum"]] = (
            f"=SUMPRODUCT({rng(pid, 'deaths', 3, last)}+{rng(pid, 'lapses', 3, last)}"
            f"+{rng(pid, 'maturities', 3, last)})"
        )
        ws[c["pv_net_cf_t_sum"]] = f"=SUM({rng(pid, 'pv_net_cf_t', 3, last)})"
        ws[c["scale_money"]] = (
            f"=inputs!${MP_COL['sum_assured']}${mp_row}*inputs!${MP_COL['pols_if_init']}${mp_row}"
        )
        ws[c["scale_count"]] = f"=inputs!${MP_COL['pols_if_init']}${mp_row}"
    all_row = 3 + len(MODEL_POINTS)
    ok = CCOL["ok"]
    ws[f"A{all_row}"] = "all_ok"
    ws[f"A{all_row}"].font = BOLD
    ws[f"B{all_row}"] = f"=AND({ok}2:{ok}{all_row - 2})"
    define(wb, "all_ok", f"checks!$B${all_row}")
    set_widths(ws, 12, len(CHECK_COLUMNS))


def build() -> openpyxl.Workbook:
    wb = openpyxl.Workbook()
    write_readme(wb)
    write_inputs(wb)
    for k, mp in enumerate(MODEL_POINTS):
        write_policy(wb, mp, MP_FIRST_ROW + k)
    write_scalars(wb)
    write_totals_t(wb)
    write_checks(wb)
    return wb


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    wb = build()
    wb.save(OUT)
    rows = ", ".join(f"{mp.policy_id}: {mp.rem_m}" for mp in MODEL_POINTS)
    print(f"wrote {OUT} ({len(wb.sheetnames)} sheets; rem_m by policy: {rows})")
    print("formulas only: recalculate with docs/spec/recalc_excel.py before reading values")


if __name__ == "__main__":
    main()
