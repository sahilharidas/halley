# pyright: basic
"""The annuity reference case: the engine against docs/spec/annuity.md (design section 8).

Until the spreadsheet exists, the in-force group spec's number script is the oracle: its
``project`` restates the annuity model in plain float64 scalars and reproduces the worked
example. Examples 1, 2 and 3 reconcile per policy (BEL, RA, A(1), PVA, PVE, A(13)) and as group
sums against the printed ten-significant-figure values and against the script, every column and
every month, under the spec's tolerance and scales; then the horizon test, the bit-identity of
policy 1 between examples 1 and 2, the outputs and the manifest, and the tests of the scripts
that generate the reference basis.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import NormalDist
from types import ModuleType
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from halley import CompiledModel, RunResult, RunSettings, build, load_run, run
from halley.aggregation import portfolio_totals
from halley.products import annuity_data as data
from halley.products.annuity import (
    BASIS_VERSION,
    OMEGA,
    VALUATION_DATE,
    LevelAnnuity,
)
from halley.runtime import recompute, within

SPEC = Path(__file__).resolve().parent.parent / "docs" / "spec" / "annuity"
TABLE_NAMES = ("mort_base", "mort_cif", "expense", "inv_expense", "ilp", "ra_param")
TABLES = {name: SPEC / f"{name}.parquet" for name in TABLE_NAMES}
SCENARIOS = {"rf_df": SPEC / "discount.parquet", "cpi": SPEC / "cpi.parquet"}
T_EXAMPLE = 655
"""Every example run has policy 1 or 4, so T = 655; the horizon test runs policy 3 alone."""

OUTPUTS: tuple[str, ...] = LevelAnnuity.outputs or ()
TRACE_OUTPUTS = (*OUTPUTS, "q_ann", "q_mth", "q_mth_st", "v_be", "v_st", "esc_factor")
"""The outputs plus the variables the spec lists "on request for tracing"."""

STATE_COLS = {"pols_if", "ann_pp", "df", "bel", "v_be", "v_st", "ra"}
"""Defined at t = 0; every other column is a month quantity, undefined there."""
COUNT_COLS = {"deaths", "pols_if"}
PER_POLICY_COLS = {"ann_pp", "exp_pp", "s_pp", "s_pp_st", "v_be", "v_st"}
MONEY_COLS = {"ann_out", "exp_out", "inv_exp", "bel", "ra", "cu"}
EXACT_COLS = {"age", "cal_year", "anniv"}
"""Integer and boolean columns compare exactly."""
PER_LIFE_LAST = {"ann_pp": 0, "exp_pp": 0, "s_pp": 0, "s_pp_st": 0, "v_be": -1, "v_st": -1}
"""Per-life values are compared up to t_end (per-policy amounts) or t_end - 1 (values at t);
beyond that no life is in force and the spec excludes them from every comparison."""

PRINTED = {
    1: {
        "bel_0": "179192.148",
        "ra_0": "2648.249534",
        "ann_pp_1": "1000",
        "pva": "175222.1063",
        "pve": "3970.041706",
        "ann_pp_13": "1000",
    },
    2: {
        "bel_0": "206382.8002",
        "ra_0": "4031.105418",
        "ann_pp_1": "800",
        "pva": "201200.6846",
        "pve": "5182.11565",
        "ann_pp_13": "824",
    },
    3: {
        "bel_0": "234466.2049",
        "ra_0": "5607.325647",
        "ann_pp_1": "1200",
        "pva": "229894.987",
        "pve": "4571.217872",
        "ann_pp_13": "1232.981869",
    },
}
PRINTED[4] = PRINTED[1]
"""The spec's per-policy results to ten significant figures, as ``f"{x:.10g}"`` prints them."""

EXAMPLE_1_CU = (
    "1000",
    "999.5156821",
    "999.0315987",
    "998.5477498",
    "998.0641352",
    "997.5807548",
    "997.0976086",
    "996.5570554",
    "996.0167953",
    "995.476828",
    "994.9371535",
    "994.3977716",
)
EXAMPLE_3_CU = (
    "3000",
    "2998.144065",
    "2996.28939",
    "2994.311796",
    "2992.335685",
    "2990.361056",
    "2988.387909",
    "2986.358601",
    "2984.330831",
    "2982.304598",
    "2980.2799",
    "2978.215766",
)
EXAMPLE_GROUPS = {
    1: {"p_g": "190000", "bel_g": "179192.148", "ra_g": "2648.249534"},
    2: {"p_g": "175000", "bel_g": "179192.148", "ra_g": "2648.249534"},
    3: {"p_g": "660000", "bel_g": "620041.1532", "ra_g": "12286.6806"},
}


def _ten(x: float) -> str:
    return f"{float(x):.10g}"


def _scale(col: str, mp: dict[str, Any]) -> float:
    """The spec's tolerance scale by unit: pols_if_init x single_premium for in-force money,
    instalment_pm per policy, pols_if_init for counts, 1 otherwise."""
    if col in MONEY_COLS:
        return float(mp["pols_if_init"] * mp["single_premium"])
    if col in PER_POLICY_COLS:
        return float(mp["instalment_pm"])
    if col in COUNT_COLS:
        return float(mp["pols_if_init"])
    return 1.0


# ------------------------------------------------------------------------------ oracle


def _oracle_mp(o: ModuleType, row: dict[str, Any]) -> dict[str, Any]:
    dob = row["date_of_birth"]
    cls = "cpi" if row["cls"] == "cpi" else "level"
    return o.make_mp(
        row["policy_id"], cls, row["sex"], (dob.year, dob.month, dob.day), 1.0, row["instalment_pm"]
    )


def _oracle_run(o: ModuleType, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The script's AS run: the recognition basis at 2025-12-31 on the composed curve p_rec and
    the CPI path i_rec, every policy in force with one life."""
    mps = [_oracle_mp(o, r) for r in rows]
    return o.make_run(
        "AS",
        o.M_REC,
        o.MORT_V1,
        o.MAINT_V1,
        o.p_rec,
        o.i_rec,
        {r["policy_id"]: 1.0 for r in rows},
        {r["policy_id"]: float(r["instalment_pm"]) for r in rows},
        mps,
    )


def _project_fixed(o: ModuleType, row: dict[str, Any], horizon: int) -> dict[str, Any]:
    """Section 6 of the in-force spec for the fixed-escalation class, which its script does not
    carry: ``project`` with ``esc_factor = 1 + esc_rate`` on an anniversary, line for line."""
    dob = row["date_of_birth"]
    sex, inst, g = row["sex"], float(row["instalment_pm"]), float(row["esc_rate"])
    m_v = o.M_REC
    a0 = m_v - o.mi(dob.year, dob.month)
    m_c = o.M_START - m_v
    mort, cif, maint, ie_m = o.MORT_V1, o.CIF, o.MAINT_V1, o.IE_M
    dfn, cpi = o.p_rec, o.i_rec
    r: dict[str, Any] = {c: [None] * (horizon + 1) for c in o.COLS}
    r["pols_if"][0] = 1.0
    r["ann_pp"][0] = inst
    r["df"][0] = dfn(0)
    i0 = cpi(0)
    for t in range(1, horizon + 1):
        x = min(OMEGA, (a0 + t - 1) // 12)
        y = (m_v + t) // 12
        q_ann = 1.0 if x >= OMEGA else mort[(sex, x)] * cif[(x, y)]
        q_mth = o.to_monthly_cf(q_ann)
        open_ = r["pols_if"][t - 1]
        deaths = open_ * q_mth
        k = t - m_c
        anniv = k > 0 and k % 12 == 0
        esc = (1 + g) if anniv else 1.0
        ann_pp = r["ann_pp"][t - 1] * esc
        exp_pp = maint * (cpi(t - 1) / i0)
        df = dfn(t)
        row_t = {
            "age": x,
            "cal_year": y,
            "q_ann": q_ann,
            "q_mth": q_mth,
            "deaths": deaths,
            "pols_if": open_ - deaths,
            "anniv": anniv,
            "esc_factor": esc,
            "ann_pp": ann_pp,
            "exp_pp": exp_pp,
            "ann_out": open_ * ann_pp,
            "exp_out": open_ * exp_pp,
            "df": df,
            "v": df / r["df"][t - 1],
            "s_pp": ann_pp + exp_pp,
            "q_ann_st": 1.0 if x >= OMEGA else q_ann * o.MORT_MULT,
            "s_pp_st": ann_pp + exp_pp * o.EXP_MULT,
            "cu": open_ * ann_pp,
        }
        row_t["q_mth_st"] = o.to_monthly_cf(row_t["q_ann_st"])
        for name, val in row_t.items():
            r[name][t] = val
    r["bel"][horizon] = 0.0
    r["v_be"][horizon] = 0.0
    r["v_st"][horizon] = 0.0
    r["inv_exp"][horizon] = ie_m * r["bel"][horizon]
    for t in range(horizon - 1, -1, -1):
        u = t + 1
        r["inv_exp"][u] = ie_m * r["bel"][u]
        r["bel"][t] = (
            r["ann_out"][u]
            + r["exp_out"][u]
            + r["inv_exp"][u] * r["v"][u]
            + r["v"][u] * r["bel"][u]
        )
        r["v_be"][t] = r["s_pp"][u] + (1 - r["q_mth"][u]) * (
            r["v"][u] * (r["v_be"][u] + ie_m * r["v_be"][u])
        )
        r["v_st"][t] = r["s_pp_st"][u] + (1 - r["q_mth_st"][u]) * (
            r["v"][u] * (r["v_st"][u] + ie_m * r["v_st"][u])
        )
    for t in range(horizon + 1):
        r["ra"][t] = r["pols_if"][t] * (r["v_st"][t] - r["v_be"][t])
    pva = pve = 0.0
    for t in range(1, horizon + 1):
        pva += r["df"][t - 1] * r["ann_out"][t]
        pve += r["df"][t - 1] * r["exp_out"][t] + r["df"][t] * r["inv_exp"][t]
    r["scalars"] = {
        "bel_0": r["bel"][0],
        "ra_0": r["ra"][0],
        "ann_pp_1": r["ann_pp"][1],
        "pva": pva,
        "pve": pve,
        "t_end": o.T_OMEGA - a0,
    }
    return r


# ---------------------------------------------------------------------------- fixtures


def _run(
    compiled: CompiledModel,
    model_points: Any,
    run_dir: Path,
    *,
    horizon: int | None = None,
    outputs: tuple[str, ...] | None = None,
    **kw: Any,
) -> RunResult:
    return run(
        compiled,
        model_points=model_points,
        tables=TABLES,
        scenarios=SCENARIOS,
        settings=RunSettings(valuation_date=VALUATION_DATE, horizon=horizon, outputs=outputs, **kw),
        run_dir=run_dir,
    )


@pytest.fixture(scope="module")
def runs_dir(tmp_path_factory: Any) -> Path:
    return Path(tmp_path_factory.mktemp("annuity"))


@pytest.fixture(scope="module")
def run_level(compiled_annuity: dict[str, CompiledModel], runs_dir: Path) -> RunResult:
    """Policies 1 and 4 (the level class file), the product's own outputs, H = 655."""
    return _run(compiled_annuity["level"], SPEC / "mp_level.parquet", runs_dir / "level")


@pytest.fixture(scope="module")
def run_fixed(compiled_annuity: dict[str, CompiledModel], runs_dir: Path) -> RunResult:
    """Policy 2 at the example horizon, with the tracing outputs stored."""
    return _run(
        compiled_annuity["fixed"],
        SPEC / "mp_fixed.parquet",
        runs_dir / "fixed",
        horizon=T_EXAMPLE,
        outputs=TRACE_OUTPUTS,
    )


@pytest.fixture(scope="module")
def run_cpi(compiled_annuity: dict[str, CompiledModel], runs_dir: Path) -> RunResult:
    """Policy 3 at the example horizon, with the tracing outputs stored."""
    return _run(
        compiled_annuity["cpi"],
        SPEC / "mp_cpi.parquet",
        runs_dir / "cpi",
        horizon=T_EXAMPLE,
        outputs=TRACE_OUTPUTS,
    )


@pytest.fixture(scope="module")
def run_cpi_alone(compiled_annuity: dict[str, CompiledModel], runs_dir: Path) -> RunResult:
    """The horizon test: policy 3 alone, so T = 591."""
    return _run(compiled_annuity["cpi"], SPEC / "mp_cpi.parquet", runs_dir / "cpi-alone")


@pytest.fixture(scope="module")
def one_policy_runs(
    compiled_annuity: dict[str, CompiledModel], runs_dir: Path
) -> dict[int, RunResult]:
    """Examples 1 and 2: policy 1 alone and policy 4 alone, one-policy batches."""
    out: dict[int, RunResult] = {}
    for pid in (1, 4):
        table = data.model_points_table([data.example_policy(pid)])
        out[pid] = _run(compiled_annuity["level"], table, runs_dir / f"policy-{pid}")
    return out


@pytest.fixture(scope="module")
def expected(inforce_oracle: ModuleType) -> dict[int, dict[str, Any]]:
    """The script's projection of every example policy at T = 655."""
    o = inforce_oracle
    rows = {pid: data.example_policy(pid) for pid in (1, 2, 3, 4)}
    as_run = _oracle_run(o, [rows[1], rows[3], rows[4]])
    assert as_run["T"] == T_EXAMPLE
    out = {pid: o.project(mp, as_run) for pid, mp in zip((1, 3, 4), as_run["mps"], strict=True)}
    out[2] = _project_fixed(o, rows[2], T_EXAMPLE)
    return out


def _engine(runs: dict[str, RunResult], pid: int) -> RunResult:
    return runs["level"] if pid in (1, 4) else runs["fixed"] if pid == 2 else runs["cpi"]


@pytest.fixture(scope="module")
def runs(run_level: RunResult, run_fixed: RunResult, run_cpi: RunResult) -> dict[str, RunResult]:
    return {"level": run_level, "fixed": run_fixed, "cpi": run_cpi}


# ----------------------------------------------------------------- dimensions and columns


def test_horizon_and_dimensions(runs: dict[str, RunResult], run_cpi_alone: RunResult) -> None:
    assert runs["level"].H == T_EXAMPLE
    assert list(runs["level"].ids) == [1, 4]
    assert runs["fixed"].H == runs["cpi"].H == T_EXAMPLE
    assert run_cpi_alone.H == 591  # 12 x 120 + 1 - 850
    for res in (*runs.values(), run_cpi_alone):
        assert res.n_scenarios == 1
        assert all(r.passed for r in res.invariants), [r for r in res.invariants if not r.passed]
    assert runs["level"].lane("run_off_month", 1)[0] == 655
    assert runs["fixed"].lane("run_off_month", 2)[0] == 623
    assert runs["cpi"].lane("run_off_month", 3)[0] == 591


@pytest.mark.parametrize("pid", [1, 2, 3, 4])
def test_every_column_reconciles_with_the_script(
    pid: int, runs: dict[str, RunResult], expected: dict[int, dict[str, Any]]
) -> None:
    res = _engine(runs, pid)
    want = expected[pid]
    mp = data.example_policy(pid)
    t_end = want["scalars"]["t_end"]
    worst: dict[str, float] = {}
    for col in want:
        if col in ("scalars", "T", "a0", "m_c"):
            continue
        if col == "anniv" and col not in res.arrays:
            assert mp["cls"] == "level"
            continue
        first = 0 if col in STATE_COLS else 1
        last = res.H if col not in PER_LIFE_LAST else t_end + PER_LIFE_LAST[col]
        engine = res.lane(col, pid)[first : last + 1]
        if col in EXACT_COLS:
            assert np.array_equal(engine, np.array(want[col][first : last + 1])), col
            continue
        target = np.array(want[col][first : last + 1], dtype=np.float64)
        ok = within(engine, target, _scale(col, mp))
        assert ok.all(), (col, int(np.flatnonzero(~ok)[0]) + first)
        worst[col] = float(np.abs(engine - target).max())
    assert worst["pols_if"] < 1e-15
    assert worst["bel"] < 1e-8


@pytest.mark.parametrize("pid", [1, 2, 3, 4])
def test_scalars_reconcile_with_the_printed_values_and_the_script(
    pid: int, runs: dict[str, RunResult], expected: dict[int, dict[str, Any]]
) -> None:
    res = _engine(runs, pid)
    mp = data.example_policy(pid)
    got = {
        "bel_0": res.lane("bel", pid)[0],
        "ra_0": res.lane("ra", pid)[0],
        "ann_pp_1": res.lane("ann_pp", pid)[1],
        "pva": res.lane("pva", pid)[0],
        "pve": res.lane("pve", pid)[0],
        "ann_pp_13": res.lane("ann_pp", pid)[13],
    }
    assert {k: _ten(v) for k, v in got.items()} == PRINTED[pid]
    want = expected[pid]["scalars"]
    money = _scale("bel", mp)
    for name in ("bel_0", "ra_0", "pva", "pve"):
        assert within(got[name], want[name], money), name
    assert within(got["ann_pp_1"], want["ann_pp_1"], _scale("ann_pp", mp))
    assert within(got["ann_pp_13"], expected[pid]["ann_pp"][13], _scale("ann_pp", mp))
    # the direct present value equals the recursion (I5) within 1e-12 relative
    assert within(got["pva"] + got["pve"], got["bel_0"], 0.0, rel=1e-12)
    assert got["ann_pp_1"] == mp["instalment_pm"]  # A(1) is the instalment, exactly


def test_the_anniversary_payments_are_the_hand_checks(runs: dict[str, RunResult]) -> None:
    assert runs["fixed"].lane("ann_pp", 2)[13] == 824.0  # 800 x fl(1.03) rounds to 824.0
    a13 = runs["cpi"].lane("ann_pp", 3)[13]
    assert within(a13, 1200 * 1.025 ** (10 / 12) * 1.04 ** (2 / 12), 0.0, rel=1e-12)
    cpi = pq.read_table(SCENARIOS["cpi"]).to_pydict()
    index = dict(zip(cpi["u"], cpi["cpi"], strict=True))
    assert a13 == 1200.0 * (index[10] / index[-2])  # E(13) = I(10) / I(-2) with L = 3
    assert list(runs["cpi"].lane("anniv", 3)[:40].nonzero()[0]) == [13, 25, 37]
    assert runs["level"].lane("ann_pp", 1)[1:].min() == 1000.0


def test_month_quantities_are_zero_at_t_zero_and_after_the_run_off(
    runs: dict[str, RunResult],
) -> None:
    for pid in (2, 3):
        res = _engine(runs, pid)
        t_end = int(res.lane("run_off_month", pid)[0])
        for col in ("deaths", "ann_out", "exp_out", "inv_exp", "cu", "q_mth", "v"):
            assert res.lane(col, pid)[0] == 0.0, col
        for col in ("deaths", "ann_out", "exp_out", "inv_exp", "cu"):
            assert not res.lane(col, pid)[t_end + 1 :].any(), col
        for col in ("pols_if", "bel", "ra"):
            assert not res.lane(col, pid)[t_end:].any(), col
        assert res.lane("q_ann", pid)[t_end] == 1.0
        assert res.lane("age", pid)[t_end] == OMEGA
        assert res.lane("deaths", pid)[t_end] == res.lane("pols_if", pid)[t_end - 1]


# ------------------------------------------------------------------------- group sums


def test_group_sums_of_the_three_examples(
    runs: dict[str, RunResult], one_policy_runs: dict[int, RunResult]
) -> None:
    for example, pid in ((1, 1), (2, 4)):
        totals = portfolio_totals(one_policy_runs[pid], ["bel", "ra", "cu", "premium_total"])
        got = {
            "p_g": _ten(totals["premium_total"][0, 0, 0]),
            "bel_g": _ten(totals["bel"][0, 0, 0]),
            "ra_g": _ten(totals["ra"][0, 0, 0]),
        }
        assert got == EXAMPLE_GROUPS[example], example
        assert tuple(_ten(x) for x in totals["cu"][1:13, 0, 0]) == EXAMPLE_1_CU
    # example 3: policies 1, 2 and 3 from the three class runs, summed sequentially in
    # ascending policy_id (D19); the aggregation stage's Basis formalises this join (design 9)
    lanes = [(pid, _engine(runs, pid)) for pid in (1, 2, 3)]
    bel_g = ra_g = p_g = 0.0
    cu_g = np.zeros(12)
    for pid, res in lanes:
        bel_g += res.lane("bel", pid)[0]
        ra_g += res.lane("ra", pid)[0]
        cu_g = cu_g + res.lane("cu", pid)[1:13]
        p_g += res.lane("premium_total", pid)[0]
    assert {"p_g": _ten(p_g), "bel_g": _ten(bel_g), "ra_g": _ten(ra_g)} == EXAMPLE_GROUPS[3]
    assert tuple(_ten(x) for x in cu_g) == EXAMPLE_3_CU
    fcf_g = bel_g + ra_g
    assert p_g - fcf_g > 0  # example 3 is profitable: CSM(0) = 27,672.16624
    assert _ten(p_g - fcf_g) == "27672.16624"


def test_the_para_47_test_per_policy(
    runs: dict[str, RunResult], one_policy_runs: dict[int, RunResult]
) -> None:
    assert not runs["level"].lane("onerous", 1)[0]
    assert runs["level"].lane("onerous", 4)[0]
    assert not runs["fixed"].lane("onerous", 2)[0]
    assert not runs["cpi"].lane("onerous", 3)[0]
    assert runs["level"].var("onerous").dtype == np.bool_
    res = one_policy_runs[4]
    gap = res.lane("premium_total", 4)[0] - (res.lane("bel", 4)[0] + res.lane("ra", 4)[0])
    assert _ten(-gap) == "6840.397574"  # LC(0) of example 2


# -------------------------------------------------------------- horizon and bit-identity


def test_the_horizon_test_policy_3_alone(
    runs: dict[str, RunResult], run_cpi_alone: RunResult
) -> None:
    """Policy 3 at T = 591 against its values in the T = 655 run: in-force-weighted values at
    every t and per-life values before the run-off agree within 1e-12 relative (D5)."""
    short, long = run_cpi_alone, runs["cpi"]
    t_end = 591
    for col in ("pols_if", "bel", "ra", "cu", "deaths", "ann_out", "exp_out", "inv_exp"):
        a, b = short.lane(col, 3), long.lane(col, 3)[: t_end + 1]
        assert within(a, b, 0.0, rel=1e-12).all(), col
    for col, last in (("ann_pp", t_end), ("v_be", t_end - 1), ("v_st", t_end - 1)):
        a, b = short.lane(col, 3)[: last + 1], long.lane(col, 3)[: last + 1]
        assert within(a, b, 0.0, rel=1e-12).all(), col
    for name in ("pva", "pve"):
        assert within(short.lane(name, 3)[0], long.lane(name, 3)[0], 0.0, rel=1e-12), name
    assert np.array_equal(short.lane("bel", 3), long.lane("bel", 3)[: t_end + 1])
    assert not long.lane("bel", 3)[t_end:].any()


def test_policy_1_is_bit_identical_between_examples_1_and_2(
    runs: dict[str, RunResult], one_policy_runs: dict[int, RunResult]
) -> None:
    """Both runs are one-policy batches with identical inputs but the id, the premium and the
    bucket (D19); and the two-policy batch of the class file gives the same lanes (D5)."""
    one, four = one_policy_runs[1], one_policy_runs[4]
    for col in ("bel", "ra", "pols_if", "cu", "v_be", "v_st", "pva", "pve"):
        assert np.array_equal(one.lane(col, 1), four.lane(col, 4)), col
        assert np.array_equal(one.lane(col, 1), runs["level"].lane(col, 1)), col
        assert np.array_equal(four.lane(col, 4), runs["level"].lane(col, 4)), col
    assert one.run_id != four.run_id  # the model point file differs


def test_rerun_is_bit_identical_d19(
    compiled_annuity: dict[str, CompiledModel], run_level: RunResult, runs_dir: Path
) -> None:
    again = _run(compiled_annuity["level"], SPEC / "mp_level.parquet", runs_dir / "level-again")
    assert again.run_id == run_level.run_id
    for name, arr in run_level.arrays.items():
        assert np.array_equal(arr, again.arrays[name]), name


def test_two_scenarios_share_the_history_and_split_the_path(
    compiled_annuity: dict[str, CompiledModel], run_cpi_alone: RunResult, runs_dir: Path
) -> None:
    """A stochastic run (D4): two CPI scenarios with identical history rows, 2.5% a year in the
    first and 3.5% in the second, the discount curve the same in both. Every invariant passes
    (I9 per scenario through ``a.scen``), scenario 0 is the deterministic run bit for bit,
    ``onerous`` is per policy and scenario, and the monthly file carries both scenario ids."""

    def cpi(s: int, u: int) -> float:
        if u <= 0 or s == 0:
            return data.cpi_path(u)
        return data.CPI_BASE * 1.035 ** (u / 12)

    res = run(
        compiled_annuity["cpi"],
        model_points=SPEC / "mp_cpi.parquet",
        tables=TABLES,
        scenarios={
            "rf_df": data.discount_table(n_scenarios=2),
            "cpi": data.cpi_table(cpi, n_scenarios=2),
        },
        settings=RunSettings(valuation_date=VALUATION_DATE),
        run_dir=runs_dir / "two-scenarios",
    )
    assert res.n_scenarios == 2
    assert res.H == run_cpi_alone.H
    assert res.manifest["dimensions"]["scenarios"] == 2
    assert all(r.passed for r in res.invariants)
    assert [r.name for r in res.invariants] == [r.name for r in run_cpi_alone.invariants]
    for name in OUTPUTS:
        assert np.array_equal(res.var(name)[:, :, :1], run_cpi_alone.var(name)), name
    bel = res.var("bel")
    assert bel.shape == (res.H + 1, 1, 2)
    assert bel[0, 0, 0] != bel[0, 0, 1]  # the faster CPI path raises the liability
    assert bel[0, 0, 1] > bel[0, 0, 0]
    assert res.arrays["onerous"].shape == (1, 1, 2)
    monthly = pq.read_table(res.run_dir / "outputs" / "monthly.parquet")
    assert sorted(set(monthly.column("scenario").to_pylist())) == [0, 1]
    assert monthly.num_rows == 2 * (res.H + 1)


def test_the_interpreter_agrees_with_the_generated_code(runs: dict[str, RunResult]) -> None:
    level, cpi = runs["level"], runs["cpi"]
    for name, t in (("bel", 0), ("q_mth", 7), ("df", 120), ("v_be", 100), ("ra", 0)):
        assert recompute(level, 1, name, t) == level.lane(name, 1)[t], name
    assert recompute(cpi, 3, "esc_factor", 13) == cpi.lane("esc_factor", 3)[13]
    assert recompute(cpi, 3, "exp_pp", 13) == cpi.lane("exp_pp", 3)[13]
    assert recompute(level, 1, "ie_m", 0) == level.lane("ie_m", 1)[0]


# ------------------------------------------------------------------- outputs and manifest


def test_the_disclosure_series_and_the_curve(runs: dict[str, RunResult]) -> None:
    res = runs["level"]
    df = res.lane("df", 1)
    assert df[0] == 1.0
    rf = pq.read_table(SCENARIOS["rf_df"]).column("rf_df").to_numpy()
    t = np.arange(T_EXAMPLE + 1)
    np.testing.assert_array_equal(df, rf * np.exp(((-data.ILP) * t) / 12))
    v = res.lane("v", 1)
    assert v[0] == 0.0
    np.testing.assert_array_equal(v[1:], df[1:] / df[:-1])
    fwd, spot = res.lane("fwd_ann", 1), res.lane("spot_ann", 1)
    assert fwd[0] == 0.0
    assert spot[0] == 0.0
    np.testing.assert_array_equal(fwd[1:], (df[:-1] / df[1:]) ** 12 - 1)
    np.testing.assert_array_equal(spot[1:], df[1:] ** (-12 / t[1:]) - 1)
    # the composed curve's spot rate tends to the long rate plus the illiquidity premium
    assert abs(spot[T_EXAMPLE] - (np.exp(0.040 + 0.0050) - 1)) < 1e-3


def test_outputs_and_manifest(run_level: RunResult, run_cpi: RunResult) -> None:
    manifest = run_level.manifest
    assert manifest["model"]["statics"] == {"include_inv_exp": True}
    assert run_cpi.manifest["model"]["statics"] == {"include_inv_exp": True, "cpi_lag_m": 3}
    tables = manifest["inputs"]["tables"]
    assert tables["mort_base"]["metadata"] == {
        "version": BASIS_VERSION,
        "basis": BASIS_VERSION,
        "base_year": "2025",
        "source": "Standard Ultimate Survival Model (Makeham)",
    }
    cif_meta = tables["mort_cif"]["metadata"]
    assert cif_meta["base_year"] == "2025"
    improvement = hashlib.sha256((SPEC / "mort_improvement.parquet").read_bytes()).hexdigest()
    assert cif_meta["mort_improvement_sha256"] == improvement
    assert tables["expense"]["metadata"]["price_base"] == "2025-12"
    for name in TABLE_NAMES:
        assert tables[name]["sha256"] == hashlib.sha256(TABLES[name].read_bytes()).hexdigest()
        assert tables[name]["version"] == BASIS_VERSION
    scen = manifest["inputs"]["scenarios"]
    assert scen["rf_df"]["months"] == [0, T_EXAMPLE]
    assert scen["cpi"]["months"] == [-24, T_EXAMPLE]
    assert scen["cpi"]["metadata"]["estimated_rows"] == "none"
    assert run_level.compiled is not None
    assert manifest["ir_sha256"] == run_level.compiled.ir_hash
    assert manifest["dimensions"] == {
        "policies": 2,
        "scenarios": 1,
        "months": T_EXAMPLE + 1,
        "horizon": T_EXAMPLE,
    }
    variables = manifest["outputs"]["variables"]
    assert set(variables) == set(OUTPUTS)
    assert variables["bel"] == {"time": True, "unit": "Money", "start": 0, "role": "reserve"}
    assert variables["pols_if"] == {"time": True, "unit": "Count", "start": 0, "role": "in_force"}
    assert variables["deaths"] == {"time": True, "unit": "Count", "start": 1, "role": "decrement"}
    assert variables["cu"] == {"time": True, "unit": "CoverageUnits", "start": 1, "role": None}
    assert variables["ra"] == {"time": True, "unit": "Money", "start": 0, "role": None}
    assert variables["onerous"] == {"time": False, "unit": "Bool", "start": 0, "role": None}
    assert variables["run_off_month"] == {"time": False, "unit": "Int", "start": 0, "role": None}
    assert variables["fwd_ann"] == {"time": True, "unit": "Rate[Annual]", "start": 1, "role": None}
    assert variables["pva"]["unit"] == "Money"
    assert manifest["outputs"]["id_field"] == "policy_id"
    assert manifest["outputs"]["kind"] == "projection"
    names = [r.name for r in run_level.invariants]
    assert names[:3] == [
        "H1 in force never negative, never increasing",
        "H2 decrements never exceed the opening in-force count",
        "H2 probabilities in [0, 1]",
    ]
    assert names[-1] == "I8 ann_pp = instalment_pm"
    assert [r.name for r in run_cpi.invariants][-1] == "I9 ann_pp telescopes along the CPI path"
    assert "I4 roll-forward of bel" in names
    on_disk = json.loads((run_level.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk["outputs"] == manifest["outputs"]


def test_output_files_are_keyed_on_the_integer_policy_id(run_level: RunResult) -> None:
    monthly = pq.read_table(run_level.run_dir / "outputs" / "monthly.parquet")
    assert monthly.column_names[:3] == ["policy_id", "scenario", "t"]
    assert pa.types.is_integer(monthly.schema.field("policy_id").type)
    assert monthly.num_rows == 2 * (T_EXAMPLE + 1)
    assert "v_be" not in monthly.column_names  # not an output unless requested
    scalars = pq.read_table(run_level.run_dir / "outputs" / "scalars.parquet")
    assert set(scalars.column_names) == {
        "policy_id",
        "scenario",
        "pva",
        "pve",
        "onerous",
        "premium_total",
        "run_off_month",
    }
    assert pa.types.is_boolean(scalars.schema.field("onerous").type)
    assert pa.types.is_integer(scalars.schema.field("run_off_month").type)
    back = load_run(run_level.run_dir)
    assert list(back.ids) == [1, 4]
    for name in OUTPUTS:
        assert np.array_equal(back.var(name), run_level.var(name)), name


def test_requested_tracing_outputs_are_stored(run_cpi: RunResult) -> None:
    monthly = pq.read_table(run_cpi.run_dir / "outputs" / "monthly.parquet")
    for name in ("q_ann", "q_mth", "q_mth_st", "v_be", "v_st", "esc_factor"):
        assert name in monthly.column_names, name
    assert set(run_cpi.manifest["outputs"]["variables"]) == set(TRACE_OUTPUTS)
    assert run_cpi.manifest["settings"]["outputs"] == list(TRACE_OUTPUTS)


def test_include_inv_exp_off_drops_the_charge(runs_dir: Path, run_level: RunResult) -> None:
    compiled = build(LevelAnnuity, include_inv_exp=False)
    assert "V_ie_m[...] = 0.0" in compiled.source
    res = _run(compiled, SPEC / "mp_level.parquet", runs_dir / "no-inv-exp")
    assert res.manifest["model"]["statics"] == {"include_inv_exp": False}
    assert not res.var("inv_exp").any()
    assert res.lane("bel", 1)[0] < run_level.lane("bel", 1)[0]
    assert res.lane("pve", 1)[0] < run_level.lane("pve", 1)[0]
    assert np.array_equal(res.lane("ann_out", 1), run_level.lane("ann_out", 1))
    assert all(r.passed for r in res.invariants)


# ----------------------------------------------------------------- the reference basis


def test_the_reference_basis_equals_the_script_bit_for_bit(inforce_oracle: ModuleType) -> None:
    o = inforce_oracle
    tables = data.reference_tables()
    base = tables["mort_base"].to_pydict()
    for sex, age, q in zip(base["sex"], base["age"], base["q"], strict=True):
        assert q == o.MORT_V1[(sex, age)], (sex, age)
    assert len(base["q"]) == len(o.MORT_V1) == 2 * (OMEGA - 20 + 1)
    cif = tables["mort_cif"].to_pydict()
    for age, y, c in zip(cif["age"], cif["year"], cif["cif"], strict=True):
        assert c == o.CIF[(age, y)], (age, y)
    assert len(cif["cif"]) == len(o.CIF)
    scen = data.reference_scenarios()
    disc = scen["rf_df"].to_pydict()
    assert disc["t"] == list(range(656))
    assert all(x == o.rf_df(t, 0.040) for t, x in zip(disc["t"], disc["rf_df"], strict=True))
    cpi = scen["cpi"].to_pydict()
    assert cpi["u"] == list(range(-24, 656))
    assert all(x == o.i_rec(u) for u, x in zip(cpi["u"], cpi["cpi"], strict=True))
    params = tables["ra_param"].to_pydict()
    assert params == {
        "alpha": [o.ALPHA],
        "z_alpha": [o.Z_ALPHA],
        "sigma_l": [o.SIGMA_L],
        "sigma_e": [o.SIGMA_E],
    }
    assert tables["expense"].to_pydict() == {"maint_pm": [o.MAINT_V1]}
    assert tables["inv_expense"].to_pydict() == {"ie_rate": [o.IE_RATE]}
    assert tables["ilp"].to_pydict() == {"ilp": [o.ILP]}


def test_the_derived_constants_equal_the_script_bit_for_bit(
    inforce_oracle: ModuleType, run_level: RunResult
) -> None:
    o = inforce_oracle
    assert run_level.lane("ie_m", 1)[0] == o.IE_M
    assert run_level.lane("mort_mult", 1)[0] == o.MORT_MULT
    assert run_level.lane("exp_mult", 1)[0] == o.EXP_MULT
    assert run_level.lane("age", 1)[1] == 65
    assert run_level.lane("age", 1)[7] == 66
    assert run_level.lane("cal_year", 1)[1] == 2026
    assert run_level.lane("cal_year", 1)[655] == 2080
    assert run_level.inputs is not None
    mp = run_level.inputs.model_points.columns
    assert list(mp["a0"]) == [786, 786]
    assert list(mp["m_c"]) == [1, 1]
    assert list(mp["t_end"]) == [655, 655]
    assert list(mp["scale_money"]) == [190_000.0, 175_000.0]


def test_the_committed_files_are_the_generated_tables() -> None:
    generated = {
        **data.example_model_points(),
        **data.reference_tables(),
        "mort_improvement": data.mort_improvement_table(),
        "discount": data.reference_scenarios()["rf_df"],
        "cpi": data.reference_scenarios()["cpi"],
    }
    for name, table in generated.items():
        on_disk = pq.read_table(SPEC / f"{name}.parquet")
        assert on_disk.equals(table), name
        assert hashlib.sha256((SPEC / f"{name}.parquet").read_bytes()).hexdigest() == data.sha256(
            table
        ), name


def test_female_rows_are_the_male_rows_four_years_younger() -> None:
    base = data.mort_base_table().to_pydict()
    q = {(s, x): v for s, x, v in zip(base["sex"], base["age"], base["q"], strict=True)}
    for x in range(24, OMEGA):
        assert q[("F", x)] == q[("M", x - 4)], x
    assert q[("F", OMEGA)] == q[("M", OMEGA)] == 1.0
    assert round(data.q_makeham(65), 6) == 0.005915  # the published SOA table, 6 d.p.
    assert round(data.q_makeham(100), 6) == 0.289584
    assert f"{q[('M', 79)]:.8g}" == "0.029132141"  # in-force spec 6.1, 8 s.f.
    assert f"{q[('F', 81)]:.8g}" == "0.023174621"


def test_cif_follows_its_recursion_in_ascending_year() -> None:
    cif = data.cumulative_improvement()
    for x in range(20, OMEGA + 1):
        assert cif[(x, 2025)] == 1.0
        c = 1.0
        for y in range(2026, 2151):
            c = c * (1 - data.improvement_rate(x, y))
            assert cif[(x, y)] == c, (x, y)
    assert all(cif[(OMEGA, y)] == 1.0 for y in range(2025, 2151))
    assert all(cif[(110, y)] == 1.0 for y in range(2025, 2151))
    assert f"{cif[(79, 2027)]:.8g}" == "0.96047105"
    assert cif[(65, 2036)] == 0.8183053332960806
    table = data.mort_cif_table().to_pydict()
    assert min(table["cif"]) > 0
    assert max(table["cif"]) <= 1.0


def test_the_improvement_hand_checks() -> None:
    r = data.improvement_rate
    assert r(65, 2026) == 0.02
    assert r(65, 2036) == pytest.approx(0.015, abs=1e-15)
    assert r(65, 2046) == pytest.approx(0.01, abs=1e-15)
    assert r(95, 2026) == pytest.approx(0.012, abs=1e-15)
    assert all(r(110, y) == 0.0 for y in range(2026, 2151))
    assert all(r(OMEGA, y) == 0.0 for y in range(2026, 2151))
    assert all(r(x, y) >= 0 for x in range(20, 121) for y in range(2026, 2151, 25))
    assert data.Z_ALPHA == 0.6744897501960817
    assert abs(data.Z_ALPHA - NormalDist().inv_cdf(data.ALPHA)) <= 1e-12
    assert data.PRICE_BASE == "2025-12"
    assert data.VALUATION_DATE.isoformat() == "2025-12-31"
