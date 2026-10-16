# pyright: basic
"""Reference case 1: the engine against the term assurance spec (docs/spec/term-assurance.md).

Until the spreadsheet exists, the spec's own number script is the oracle: it implements the
formulas as written in plain float64 scalars. Every monthly column, every scalar and every
portfolio total must agree under the spec's tolerance (section 1), and the invariants of section
12 must hold on the engine's outputs in the forms the spec gives.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pytest

from halley import CompiledModel, RunResult, RunSettings, run
from halley.aggregation import portfolio_totals
from halley.products.term_assurance import VALUATION_DATE
from halley.runtime import within

REFERENCE_INPUTS = Path(__file__).resolve().parent.parent / "docs" / "spec" / "term-assurance"

STATE_COLS = {"pols_if", "infl", "disc", "reserve", "reserve_pp", "reserve_floored"}
"""Defined at t = 0; every other monthly column is a month quantity, undefined there."""

MONEY_PP_COLS = {"premium_pp", "commission_pp", "maint_pp", "claim_exp_pp", "reserve_pp"}
COUNT_COLS = {"deaths", "lapses", "maturities", "pols_if"}
MONEY_COLS = {
    "premium_income",
    "commission",
    "maint_expense",
    "death_outgo",
    "claim_expense",
    "net_cf",
    "net_outgo",
    "pv_net_cf_t",
    "pv_premiums_t",
    "pv_commission_t",
    "pv_maint_expense_t",
    "pv_death_outgo_t",
    "pv_claim_expense_t",
    "reserve",
    "reserve_floored",
}
EXACT_ZERO_AFTER_TERM = [  # P7
    "pols_if",
    "deaths",
    "lapses",
    "maturities",
    "q_m",
    "w_ann",
    "w_m",
    "premium_pp",
    "commission_pp",
    "premium_income",
    "commission",
    "maint_expense",
    "death_outgo",
    "claim_expense",
    "net_cf",
    "net_outgo",
    "pv_net_cf_t",
    "pv_premiums_t",
    "pv_commission_t",
    "pv_maint_expense_t",
    "pv_death_outgo_t",
    "pv_claim_expense_t",
    "reserve",
    "reserve_pp",
    "reserve_floored",
]
SCALARS = (
    "pv_premiums",
    "pv_commission",
    "pv_maint_expense",
    "pv_death_outgo",
    "pv_claim_expense",
    "pv_net_cf",
    "reserve_0",
    "reserve_pp_0",
    "reserve_floored_0",
    "maturities_n",
)


def _scale(name: str, mp: dict[str, Any]) -> float:
    """The tolerance scale of a column or scalar, by its unit (spec section 1)."""
    base = name[:-2] if name.endswith("_0") else name
    if base in MONEY_COLS or base in (
        "pv_premiums",
        "pv_commission",
        "pv_maint_expense",
        "pv_death_outgo",
        "pv_claim_expense",
        "pv_net_cf",
    ):
        return float(mp["sum_assured"] * mp["pols_if_init"])
    if base in MONEY_PP_COLS:
        return float(mp["sum_assured"])
    if base in COUNT_COLS or base == "maturities_n":
        return float(mp["pols_if_init"])
    return 1.0


@pytest.fixture(scope="module")
def projections(oracle: ModuleType, term_run: RunResult) -> dict[str, Any]:
    return {mp["policy_id"]: oracle.project_mp(mp, H=term_run.H) for mp in oracle.REF}


def test_horizon_and_dimensions(term_run: RunResult) -> None:
    assert term_run.H == 480  # the largest rem_m (P09)
    assert term_run.n_policies == 10
    assert term_run.n_scenarios == 1
    assert list(term_run.ids) == [f"P{i:02d}" for i in range(1, 11)]
    assert all(r.passed for r in term_run.invariants)


def test_every_monthly_column_reconciles(
    oracle: ModuleType, term_run: RunResult, projections: dict[str, Any]
) -> None:
    worst: dict[str, float] = {}
    for mp in oracle.REF:
        pid, n = mp["policy_id"], mp["rem_m"]
        r = projections[pid]
        for col in oracle.MONTH_COLS:
            first = 0 if col in STATE_COLS else 1
            engine = term_run.lane(col, pid)[first : n + 1]
            expected = np.array(r[col][first : n + 1], dtype=np.float64)
            ok = within(engine, expected, _scale(col, mp))
            assert ok.all(), f"{pid} {col}: first difference at t = {first + int(np.argmin(ok))}"
            worst[col] = max(worst.get(col, 0.0), float(np.abs(engine - expected).max()))
    assert worst["reserve"] < 1e-9
    assert worst["pols_if"] < 1e-15


def test_month_quantities_are_zero_at_t_zero_and_after_the_term(
    oracle: ModuleType, term_run: RunResult
) -> None:
    for mp in oracle.REF:
        pid, n = mp["policy_id"], mp["rem_m"]
        for col in oracle.MONTH_COLS:
            if col not in STATE_COLS:
                assert term_run.lane(col, pid)[0] == 0.0, (pid, col)
        for col in EXACT_ZERO_AFTER_TERM:  # P7
            tail = term_run.lane(col, pid)[n + 1 :]
            assert not tail.any(), (pid, col)
        assert term_run.lane("pols_if", pid)[n] == 0.0  # H3, exactly
        age = term_run.lane("age", pid)
        assert (age[n:] == age[n]).all()
        assert (term_run.lane("q_ann", pid)[n:] == term_run.lane("q_ann", pid)[n]).all()


def test_scalars_reconcile(
    oracle: ModuleType, term_run: RunResult, projections: dict[str, Any]
) -> None:
    for mp in oracle.REF:
        pid = mp["policy_id"]
        s = projections[pid]["scalars"]
        for name in SCALARS:
            engine = term_run.lane(name, pid)[0]
            assert within(engine, s[name], _scale(name, mp)), (pid, name, engine, s[name])


def test_spec_headline_values(term_run: RunResult) -> None:
    # Section 14: P05 in full precision, and P03's one-row hand check.
    assert within(term_run.lane("reserve", "P05")[0], 537.9608658387098, 200_000)
    assert within(term_run.lane("reserve", "P05")[13], 46.35078406367477, 200_000)
    assert within(term_run.lane("reserve", "P03")[0], 29.904078434800574, 150_000)
    assert within(term_run.lane("maturities", "P03")[1], 0.996177394475636, 1)
    # Section 6 checkpoints: infl(480) and disc(480) as direct powers.
    assert within(term_run.lane("infl", "P09")[480], 3.262037791999209, 1)
    assert within(term_run.lane("disc", "P09")[480], 0.20828904466294507, 1)
    assert term_run.lane("infl", "P09")[0] == 1.0
    assert term_run.lane("disc", "P09")[0] == 1.0


def test_portfolio_totals_reconcile(
    oracle: ModuleType, term_run: RunResult, projections: dict[str, Any]
) -> None:
    money_scale = sum(mp["sum_assured"] * mp["pols_if_init"] for mp in oracle.REF)  # 3,970,000
    count_scale = sum(mp["pols_if_init"] for mp in oracle.REF)  # 11.5
    assert money_scale == 3_970_000
    assert count_scale == 11.5
    totals = portfolio_totals(
        term_run,
        [
            "pv_premiums",
            "pv_net_cf",
            "reserve_0",
            "reserve_floored_0",
            "reserve",
            "pols_if",
            "deaths",
        ],
    )
    for name in ("pv_premiums", "pv_net_cf", "reserve_0", "reserve_floored_0"):
        expected = sum(projections[mp["policy_id"]]["scalars"][name] for mp in oracle.REF)
        assert within(totals[name][0, 0, 0], expected, money_scale), name
    for col, scale in (("reserve", money_scale), ("pols_if", count_scale), ("deaths", count_scale)):
        for t in (0, 1, 14, 100, 480):
            if col == "deaths" and t == 0:
                continue
            expected = sum(projections[mp["policy_id"]][col][t] or 0.0 for mp in oracle.REF)
            assert within(totals[col][t, 0, 0], expected, scale), (col, t)


def test_section_12_invariants_on_engine_outputs(oracle: ModuleType, term_run: RunResult) -> None:
    i_m = oracle.I_M
    v_m = oracle.V_M
    for mp in oracle.REF:
        pid, n = mp["policy_id"], mp["rem_m"]
        sm = mp["sum_assured"] * mp["pols_if_init"]
        sp = mp["sum_assured"]
        sc = mp["pols_if_init"]
        col = {name: term_run.lane(name, pid) for name in oracle.MONTH_COLS}
        horizon = term_run.H
        t = np.arange(1, horizon + 1)
        # H1
        assert (col["pols_if"] >= 0).all()
        assert (col["pols_if"][1:] <= col["pols_if"][:-1]).all()
        # H2
        for name in ("q_ann", "q_m", "w_ann", "w_m"):
            assert ((col[name][1:] >= 0) & (col[name][1:] <= 1)).all()
        dec = col["deaths"][1:] + col["lapses"][1:] + col["maturities"][1:]
        assert (dec <= col["pols_if"][:-1] + 1e-9 * sc).all()
        # H3
        assert col["reserve"][n] == 0.0
        assert col["reserve"][horizon] == 0.0
        assert col["pols_if"][n] == 0.0
        # H4 roll-forward
        lhs = (col["reserve"][:-1] + col["premium_income"][1:] - col["commission"][1:]) * (1 + i_m)
        lhs = lhs - col["maint_expense"][1:] - col["death_outgo"][1:] - col["claim_expense"][1:]
        assert within(lhs, col["reserve"][1:], sm).all()
        # H5a and H5b
        pv_net_cf = term_run.lane("pv_net_cf", pid)[0]
        assert within(col["reserve"][0], -pv_net_cf, sm)
        outgo = (col["death_outgo"] + col["claim_expense"] + col["maint_expense"]) * col["disc"]
        income = np.zeros_like(outgo)
        income[1:] = (col["premium_income"][1:] - col["commission"][1:]) * col["disc"][:-1]
        term = outgo - income  # month s contributes outgo(s) disc(s) - income(s) disc(s-1)
        tail = np.cumsum(term[::-1])[::-1]  # sum over s = t+1 .. H, ascending inside each run
        for u in (0, 1, n // 2, n - 1):
            acc = 0.0
            for s in range(u + 1, horizon + 1):
                acc += term[s]
            assert within(col["reserve"][u] * col["disc"][u], acc, sm), (pid, u)
        del tail
        # P1
        assert (col["maturities"][1:][t != n] == 0.0).all()
        assert col["maturities"][n] == col["pols_if"][n - 1] - col["deaths"][n] - col["lapses"][n]
        # P2
        prem_rem = mp["prem_rem_m"]
        after = t > prem_rem
        for name in ("premium_income", "commission", "w_ann", "w_m", "lapses"):
            assert (col[name][1:][after] == 0.0).all(), (pid, name)
        during = ~after
        assert (
            col["premium_income"][1:][during] == col["pols_if"][:-1][during] * mp["premium"]
        ).all()
        # P3
        assert int((col["premium_pp"][1:] != 0.0).sum()) == prem_rem
        # P4
        assert ((col["age"][1:] >= 18) & (col["age"][1:] <= 120)).all()
        steps = np.diff(col["dur_y"][1:])
        assert np.isin(steps, [0, 1]).all()
        # P5 per-policy recursion, t = 0 .. n-1
        u = np.arange(0, n)
        rhs = -(col["premium_pp"][u + 1] - col["commission_pp"][u + 1]) + v_m * (
            col["q_m"][u + 1] * (sp + col["claim_exp_pp"][u + 1])
            + col["maint_pp"][u + 1]
            + (1 - col["q_m"][u + 1]) * (1 - col["w_m"][u + 1]) * col["reserve_pp"][u + 1]
        )
        assert within(col["reserve_pp"][u], rhs, sp).all()
        # P6
        assert (col["reserve_floored"] >= 0).all()
        assert (col["reserve_floored"] >= col["reserve"]).all()
        # P8
        exits = float(np.cumsum(col["deaths"][1:] + col["lapses"][1:] + col["maturities"][1:])[-1])
        assert within(exits, sc, sc)
        # P9
        assert within(pv_net_cf, float(np.cumsum(col["pv_net_cf_t"][1:])[-1]), sm)
        # P10 unit identities, exactly
        assert (col["death_outgo"][1:] == col["deaths"][1:] * sp).all()
        assert (col["premium_income"][1:] == col["pols_if"][:-1] * col["premium_pp"][1:]).all()
        assert (col["maint_expense"][1:] == col["pols_if"][:-1] * col["maint_pp"][1:]).all()


def test_p11_batch_independence(
    compiled_term: CompiledModel, term_run: RunResult, tmp_path: Path
) -> None:
    tables = {
        name: REFERENCE_INPUTS / f"{name}.parquet" for name in term_run.manifest["inputs"]["tables"]
    }
    alone = run(
        compiled_term,
        model_points=REFERENCE_INPUTS / "model_points.parquet",
        tables=tables,
        settings=RunSettings(valuation_date=VALUATION_DATE, batch_size=1),
        run_dir=tmp_path / "batch-of-one",
    )
    for name, a in alone.arrays.items():
        b = term_run.arrays[name]
        if a.dtype.kind in "ib":
            assert np.array_equal(a, b), name
            continue
        assert within(a, b, 1.0).all(), name
        assert np.array_equal(a == 0.0, b == 0.0), f"{name}: exact zeros differ"


def test_rerun_is_bit_identical_and_the_manifest_cites_the_inputs(
    compiled_term: CompiledModel, term_run: RunResult, tmp_path: Path
) -> None:
    tables = {
        name: REFERENCE_INPUTS / f"{name}.parquet" for name in term_run.manifest["inputs"]["tables"]
    }
    again = run(
        compiled_term,
        model_points=REFERENCE_INPUTS / "model_points.parquet",
        tables=tables,
        settings=RunSettings(valuation_date=VALUATION_DATE),
        run_dir=tmp_path / "again",
    )
    assert again.run_id == term_run.run_id
    for name, a in again.arrays.items():
        assert np.array_equal(a, term_run.arrays[name]), name
    manifest = term_run.manifest
    mp_file = REFERENCE_INPUTS / "model_points.parquet"
    assert (
        manifest["inputs"]["model_points"]["sha256"]
        == hashlib.sha256(mp_file.read_bytes()).hexdigest()
    )
    versions = {name: info["version"] for name, info in manifest["inputs"]["tables"].items()}
    assert versions == {
        "mort_table": "term-mort-0.1",
        "lapse_table": "term-lapse-0.1",
        "comm_table": "term-comm-0.1",
        "rates": "term-rates-0.1",
        "expenses": "term-expenses-0.1",
    }
    assert manifest["ir_sha256"] == compiled_term.ir_hash
    assert manifest["model"]["name"].endswith("TermAssurance")
    assert manifest["settings"]["valuation_date"] == "2027-01-01"
    for name, info in manifest["inputs"]["tables"].items():
        copy = term_run.run_dir / info["copy"]
        assert hashlib.sha256(copy.read_bytes()).hexdigest() == info["sha256"], name


def test_outputs_are_keyed_on_policy_id_with_a_schema_version(term_run: RunResult) -> None:
    monthly = pq.read_table(term_run.run_dir / "outputs" / "monthly.parquet")
    assert monthly.schema.metadata[b"halley.output_schema"] == b"1"
    assert monthly.num_rows == 10 * 481
    assert monthly.column_names[:3] == ["policy_id", "scenario", "t"]
    stored = set(monthly.column_names[3:])
    assert {"pols_if", "reserve", "deaths", "net_cf"} <= stored
    assert "dur_y" not in stored  # not an output: available through explain, not stored
    scalars = pq.read_table(term_run.run_dir / "outputs" / "scalars.parquet")
    assert set(scalars.column_names) == {"policy_id", "scenario", *SCALARS}
