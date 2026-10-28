# pyright: basic
"""The CSM of one group (design sections 9 and 11; annuity spec, "Aggregation stage: CSM for
one group" and "Worked example").

``NewGroup`` runs through ``run_aggregation`` on the annuity runs of each example: example 1
(policy 1 alone, a profitable group), example 2 (policy 4 alone, onerous) and example 3
(policies 1, 2 and 3 from three class runs in one basis), every class run at H = 655 so that the
runs of a basis agree. Each reconciles CSM(0), LC(0), CU_g(t) and R(t) for t = 1..12, CSM(12)
with either value of ``discount_coverage_units``, LC(1) and LC(12) against the spec's printed
ten-significant-figure values and against ``in_force_group_numbers``: the figures of its
``annuity_spec_examples`` (policy 1 alone with the premiums of examples 1 and 2, recognised and
rolled monthly on ``p_rec``) at full precision, and the same functions on the group totals of
example 3. Then the M12 lines and M1 and M2, the member rules G3 and G5 at load, I11 to I14, M1
and M2 as run-time invariants (and failing on broken subclasses), the locked-in curve of a group
new at V and the current curve ``df_cur``, each held to the files the cur runs read (and
``run()`` refusing a stage it cannot hold to them), the horizon against the run-off (D5), twelve
monthly steps against one step with undiscounted units, and the aggregation run's directory:
keyed on ``group_id``, with its disclosures, notes (the coverage-unit approach as built) and
member rules.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import inspect
import json
import re
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from halley import (
    BuildError,
    CompiledModel,
    CoverageUnits,
    Factor,
    Field,
    Int,
    LoadError,
    Money,
    RunError,
    RunResult,
    RunSettings,
    Schema,
    build,
    load_run,
    run,
    var,
    where,
)
from halley.aggregation import (
    Aggregation,
    Basis,
    CsmRollForward,
    GroupSum,
    MemberRule,
    NewGroup,
    composed_curve,
    group_series,
    locked_in_curve,
    replay_aggregation,
    run_aggregation,
)
from halley.aggregation.csm import (
    DF_CUR,
    DISCOUNT_RATES,
    NOTE_MONTHLY_STEP,
    NOTE_TIMING_CONVEXITY,
    NOTE_UNITS_DISCOUNTED,
    NOTE_UNITS_UNDISCOUNTED,
    P_L,
    RISK_ADJUSTMENT,
)
from halley.aggregation.csm import (
    I14 as I14_INVARIANT,
)
from halley.products import annuity_data as data
from halley.products.annuity import VALUATION_DATE
from halley.runtime import within
from test_annuity import _project_fixed  # the fixed class's line-for-line oracle (stage A4)

SPEC = Path(__file__).resolve().parent.parent / "docs" / "spec" / "annuity"
TABLE_NAMES = ("mort_base", "mort_cif", "expense", "inv_expense", "ilp", "ra_param")
TABLES = {name: SPEC / f"{name}.parquet" for name in TABLE_NAMES}
SCENARIOS = {"rf_df": SPEC / "discount.parquet", "cpi": SPEC / "cpi.parquet"}
T_EXAMPLE = 655
"""Every class run of an example takes H = 655, policy 1's run-off, so the runs of a basis agree
(a projection run holds one class file)."""

EXAMPLES = {1: (1,), 2: (4,), 3: (1, 2, 3)}
"""The policies of each example's group."""
BUCKET = {1: "OTHER", 2: "ONEROUS", 3: "OTHER"}
GID = {1: "IA-2025-OTHER", 2: "IA-2025-ONEROUS", 3: "IA-2025-OTHER"}

PRINTED = {
    1: {
        "p_g": "190000.0000",
        "bel_g": "179192.1480",
        "ra_g": "2648.249534",
        "csm_0": "8159.602426",
        "lc_0": "0",
        "csm_12": "7854.527657",
    },
    2: {
        "bel_g": "179192.1480",
        "ra_g": "2648.249534",
        "csm_0": "0",
        "lc_0": "6840.397574",
        "lc_1": "6818.594999",
        "lc_12": "6585.155578",
    },
    3: {
        "p_g": "660000.0000",
        "bel_g": "620041.1532",
        "ra_g": "12286.68060",
        "csm_0": "27672.16624",
        "lc_0": "0",
        "csm_12": "26894.10869",
    },
}
"""The spec's "Group" tables (thousands separators dropped), with ``discount_coverage_units``
at its default."""

CSM_12_UNDISCOUNTED = "8072.508520"
"""Example 1's CSM(12) with ``discount_coverage_units = False``."""

PRINTED_MONTHS = {
    1: (
        (
            "1000.000000",
            "999.5156821",
            "999.0315987",
            "998.5477498",
            "998.0641352",
            "997.5807548",
            "997.0976086",
            "996.5570554",
            "996.0167953",
            "995.4768280",
            "994.9371535",
            "994.3977716",
        ),
        (
            "46.56786380",
            "46.54598345",
            "46.52412063",
            "46.50227529",
            "46.48044737",
            "46.45863682",
            "46.43684360",
            "46.41238312",
            "46.38794249",
            "46.36352167",
            "46.33912060",
            "46.31473923",
        ),
    ),
    3: (
        (
            "3000.000000",
            "2998.144065",
            "2996.289390",
            "2994.311796",
            "2992.335685",
            "2990.361056",
            "2988.387909",
            "2986.358601",
            "2984.330831",
            "2982.304598",
            "2980.279900",
            "2978.215766",
        ),
        (
            "136.9208221",
            "136.8378312",
            "136.7549127",
            "136.6663985",
            "136.5779662",
            "136.4896156",
            "136.4013465",
            "136.3105277",
            "136.2197926",
            "136.1291410",
            "136.0385725",
            "135.9462170",
        ),
    ),
}
"""The spec's "Group, by month" tables: CU_g(t) and R(t) for t = 1..12."""

REL = 1e-12
"""The engine against the script: the same formulas in the same order, so they agree far inside
the spec's 1e-9 between implementations; zeros within REL x P_g."""

M1 = "M1 LRC(t) - LRC(t-1) = IFIE(t) - revenue(t) - LC_svc(t)"
M2 = "M2 LCA(t) = LC_svc(t) - SAR(t) [IFIE_BEL(t) + RA_fin(t)]"
I11_ZERO = "I11 CSM(t) = 0 from the last month with CU_g > 0"
I14 = "I14 G3 for the group: CSM(0) = 0 if ONEROUS, else LC(0) = 0"
INVARIANTS = (
    "I11 CSM(t) = CSM(t-1) + Acc(t) - R(t)",
    "I11 CSM(t) >= 0",
    I11_ZERO,
    "I12 R(t) / CU_g(t) = CSM(t) / N(t)",
    "I12 R(t+1) / CU_g(t+1) = [R(t) / CU_g(t)] a_L(t+1) [CU_g(t+1) + d(t+1) N(t+1)] / U(t+1)",
    "I13 LC(t) / F(t) = LC(0) / F(0)",
    "I13 LC(t) = 0 where F(t) = 0",
    M1,
    M2,
    I14,
)
EXACT = ("I11 CSM(t) >= 0", I11_ZERO, "I13 LC(t) = 0 where F(t) = 0")
"""The invariants the spec states exactly, without a tolerance. I14 is checked within
1e-12 x P_g (a contract-level tie, ``test_i14_accepts_contracts_that_each_tie``) and is exact on
the examples too."""

G3_NAME = "G3 a contract is onerous exactly in an ONEROUS group"
G5_NAME = "G5 every contract is new at V"

M12_LINES = (
    "ann_out_g",
    "exp_out_g",
    "inv_exp_g",
    "f",
    "ifie_bel",
    "ra_fin",
    "ra_rel",
    "sar",
    "svc",
    "lc_svc",
    "revenue",
    "ifie",
)


def _ten(x: float) -> str:
    return f"{float(x):.10g}"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def groups_file(bucket: str, recognition: dt.date = VALUATION_DATE, cohort: int = 2025) -> Any:
    """A groups file of one row: the group key and the recognition date (a group new at V)."""
    return pa.table(
        {
            "portfolio": pa.array(["IA"], pa.string()),
            "cohort": pa.array([cohort], pa.int16()),
            "profit_bucket": pa.array([bucket], pa.string()),
            "recognition_date": pa.array([recognition], pa.date32()),
        }
    )


def _project(compiled: CompiledModel, model_points: Any, run_dir: Path, **kw: Any) -> RunResult:
    return run(
        compiled,
        model_points=model_points,
        tables=TABLES,
        scenarios=kw.pop("scenarios", SCENARIOS),
        settings=RunSettings(
            valuation_date=VALUATION_DATE, horizon=kw.pop("horizon", T_EXAMPLE), **kw
        ),
        run_dir=run_dir,
    )


def _class_of(pid: int) -> str:
    return str(data.example_policy(pid)["cls"])


def _one_policy(pid: int, **changes: Any) -> Any:
    row = {**data.example_policy(pid), **changes}
    return data.model_points_table([row], fixed=row["cls"] == "fixed")


def _stage(
    compiled: CompiledModel,
    runs: list[RunResult],
    curves: dict[str, Any],
    run_dir: Path,
    groups: Any,
    **settings: Any,
) -> RunResult:
    """The stage on the runs of its cur basis, with its two curves ``p_l`` and ``df_cur``."""
    return run_aggregation(
        compiled,
        groups=groups,
        bases={"cur": runs},
        scenarios=curves,
        settings=RunSettings(valuation_date=VALUATION_DATE, **settings),
        run_dir=run_dir,
    )


# ---------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def runs_dir(tmp_path_factory: Any) -> Path:
    return Path(tmp_path_factory.mktemp("csm-group"))


@pytest.fixture(scope="module")
def class_runs(compiled_annuity: dict[str, CompiledModel], runs_dir: Path) -> dict[int, RunResult]:
    """Each example policy alone in its class file, at H = 655."""
    return {
        pid: _project(
            compiled_annuity[_class_of(pid)], _one_policy(pid), runs_dir / f"policy-{pid}"
        )
        for pid in (1, 2, 3, 4)
    }


@pytest.fixture(scope="module")
def curves(runs_dir: Path) -> dict[str, Path]:
    """The stage's two curves, written by the section 10 helpers from the discount file and the
    ILP table the example runs read: ``p_l``, the locked-in curve of a group new at V, re-based
    with tau0 = 0 (P_L = P), and ``df_cur``, the composed current curve (the para 120 curve)."""
    _p_rec, p_l = locked_in_curve(SCENARIOS["rf_df"], TABLES["ilp"], 0, T_EXAMPLE)
    df_cur = composed_curve(SCENARIOS["rf_df"], TABLES["ilp"], T_EXAMPLE)
    out = {"p_l": runs_dir / "p_l.parquet", "df_cur": runs_dir / "df_cur.parquet"}
    pq.write_table(p_l, out["p_l"])
    pq.write_table(df_cur, out["df_cur"])
    return out


@pytest.fixture(scope="module")
def p_l(curves: dict[str, Path]) -> Path:
    return curves["p_l"]


@pytest.fixture(scope="module")
def stages(
    compiled_new_group: dict[bool, CompiledModel],
    class_runs: dict[int, RunResult],
    curves: dict[str, Path],
    runs_dir: Path,
) -> dict[tuple[int, bool], RunResult]:
    """Each example's recognition stage with discounted (True) and undiscounted coverage units."""
    out: dict[tuple[int, bool], RunResult] = {}
    for example, pids in EXAMPLES.items():
        for switch in (True, False):
            out[(example, switch)] = _stage(
                compiled_new_group[switch],
                [class_runs[pid] for pid in pids],
                curves,
                runs_dir / f"example-{example}-{'discounted' if switch else 'undiscounted'}",
                groups_file(BUCKET[example]),
                runs_root=runs_dir,
            )
    return out


def _roll_undiscounted(
    csm0: float, lc0: float, cu: list[Any], f: list[float], p_l: Any, horizon: int
) -> dict[str, list[Any]]:
    """The script's ``monthly_roll`` with undiscounted units, d = 1 in N(t), line for line."""
    n = [0.0] * (horizon + 1)
    for t in range(horizon - 1, -1, -1):
        n[t] = cu[t + 1] + n[t + 1]
    csm, lc, rel = [csm0] + [0.0] * horizon, [lc0] + [0.0] * horizon, [0.0] * (horizon + 1)
    for t in range(1, horizon + 1):
        j = p_l(t - 1) / p_l(t) - 1
        pre = csm[t - 1] + j * csm[t - 1]
        u = cu[t] + n[t]
        rel[t] = pre * (cu[t] / u) if u > 0 else pre
        csm[t] = pre - rel[t]
        lc[t] = lc[t - 1] * (f[t] / f[t - 1]) if f[t - 1] > 0 else 0.0
    return {"n": n, "csm": csm, "lc": lc, "r": rel}


@pytest.fixture(scope="module")
def expected(inforce_oracle: ModuleType) -> dict[int, dict[str, Any]]:
    """The script's figures for each example at full precision: ``annuity_spec_examples``'s
    run of policy 1 alone, recognised with the premiums of examples 1 (190,000) and 2 (175,000,
    policy 4's) and rolled monthly on ``p_rec``; for example 3 the same functions on the group
    totals of policies 1 and 3 (``project``) and 2 (the fixed class's restatement), summed in
    ascending policy_id. ``undiscounted`` is the roll with d = 1."""
    o = inforce_oracle
    one = [o.make_mp(1, "level", "M", (1960, 6, 15), 1, 1000.0)]
    run1 = o.run_group(
        o.make_run(
            "AS1", o.M_REC, o.MORT_V1, o.MAINT_V1, o.p_rec, o.i_rec, {1: 1.0}, {1: 1000.0}, one
        )
    )
    out: dict[int, dict[str, Any]] = {}
    for example, premium in ((1, 190000.0), (2, 175000.0)):
        one[0]["single_premium"] = premium
        rg = o.recognise(run1)
        out[example] = {"g": run1["g"], **rg}
    rows = {pid: data.example_policy(pid) for pid in (1, 2, 3)}
    mps = []
    for pid in (1, 3):
        row, dob = rows[pid], rows[pid]["date_of_birth"]
        birth = (dob.year, dob.month, dob.day)
        mps.append(o.make_mp(pid, row["cls"], row["sex"], birth, 1, row["instalment_pm"]))
    lives, instalments = {1: 1.0, 3: 1.0}, {1: 1000.0, 3: 1200.0}
    run3 = o.make_run(
        "AS3", o.M_REC, o.MORT_V1, o.MAINT_V1, o.p_rec, o.i_rec, lives, instalments, mps
    )
    assert run3["T"] == T_EXAMPLE
    pols = {mp["policy_id"]: o.project(mp, run3) for mp in mps}
    pols[2] = _project_fixed(o, rows[2], T_EXAMPLE)
    months = T_EXAMPLE + 1
    g: dict[str, list[Any]] = {"bel": [0.0] * months, "ra": [0.0] * months}
    g["cu"] = [None] + [0.0] * T_EXAMPLE
    p_g = 0.0
    for pid in (1, 2, 3):
        for t in range(T_EXAMPLE + 1):
            g["bel"][t] += pols[pid]["bel"][t]
            g["ra"][t] += pols[pid]["ra"][t]
        for t in range(1, T_EXAMPLE + 1):
            g["cu"][t] += pols[pid]["cu"][t]
        p_g += rows[pid]["pols_if_init"] * rows[pid]["single_premium"]
    g["fcf"] = [g["bel"][t] + g["ra"][t] for t in range(T_EXAMPLE + 1)]
    gap = p_g - g["fcf"][0]
    out[3] = {
        "g": g,
        "p_g": p_g,
        "fcf_0": g["fcf"][0],
        "csm_0": gap if gap > 0 else 0.0,
        "lc_0": -gap if -gap > 0 else 0.0,
    }
    for e in out.values():
        cu, f = e["g"]["cu"], e["g"]["fcf"]
        e["roll"] = o.monthly_roll(e["csm_0"], e["lc_0"], cu, f, o.p_rec, T_EXAMPLE)
        e["undiscounted"] = _roll_undiscounted(e["csm_0"], e["lc_0"], cu, f, o.p_rec, T_EXAMPLE)
    return out


def _agree(got: Any, want: Any, p_g: float) -> None:
    got_ = np.asarray(got, dtype=np.float64)
    want_ = np.asarray([0.0 if w is None else w for w in np.atleast_1d(want)], dtype=np.float64)
    ok = within(got_, want_.reshape(got_.shape), p_g, rel=REL)
    assert ok.all(), int(np.flatnonzero(~np.atleast_1d(ok))[0])


# ----------------------------------------------------------------- the worked example


@pytest.mark.parametrize("example", [1, 2, 3])
def test_recognition_reconciles_with_the_printed_values(
    example: int, stages: dict[tuple[int, bool], RunResult]
) -> None:
    res = stages[(example, True)]
    gid = GID[example]
    assert res.ids.tolist() == [gid]
    got = {
        "p_g": res.lane("p_g", gid)[0],
        "bel_g": res.lane("bel_g", gid)[0],
        "ra_g": res.lane("ra_g", gid)[0],
        "csm_0": res.lane("csm_0", gid)[0],
        "lc_0": res.lane("lc_0", gid)[0],
        "csm_12": res.lane("csm", gid)[12],
        "lc_1": res.lane("lc", gid)[1],
        "lc_12": res.lane("lc", gid)[12],
    }
    assert {k: _ten(got[k]) for k in PRINTED[example]} == {
        k: _ten(float(v)) for k, v in PRINTED[example].items()
    }
    # CSM(0) and LC(0) are the max(0, .) of one gap: one of them is exactly zero (paras 38, 47)
    assert min(got["csm_0"], got["lc_0"]) == 0.0
    assert res.lane("csm", gid)[0] == got["csm_0"]
    assert res.lane("lc", gid)[0] == got["lc_0"]
    assert res.lane("lrc_0", gid)[0] == res.lane("lrc", gid)[0]
    if example in PRINTED_MONTHS:
        cu, release = PRINTED_MONTHS[example]
        assert tuple(_ten(x) for x in res.lane("cu_g", gid)[1:13]) == tuple(
            _ten(float(x)) for x in cu
        )
        assert tuple(_ten(x) for x in res.lane("release", gid)[1:13]) == tuple(
            _ten(float(x)) for x in release
        )


@pytest.mark.parametrize("example", [1, 2, 3])
@pytest.mark.parametrize("discounted", [True, False])
def test_every_month_reconciles_with_the_script(
    example: int,
    discounted: bool,
    stages: dict[tuple[int, bool], RunResult],
    expected: dict[int, dict[str, Any]],
) -> None:
    """The group series and every line of the roll-forward, at every month, against the script's
    group totals and its monthly roll (``undiscounted``: d = 1)."""
    res = stages[(example, discounted)]
    gid = GID[example]
    want = expected[example]
    roll = want["roll"] if discounted else want["undiscounted"]
    p_g = float(res.lane("p_g", gid)[0])
    assert p_g == want["p_g"]
    _agree(res.lane("fcf_g", gid), want["g"]["fcf"], p_g)
    _agree(res.lane("bel_g", gid), want["g"]["bel"], p_g)
    _agree(res.lane("ra_g", gid), want["g"]["ra"], p_g)
    _agree(res.lane("cu_g", gid)[1:], want["g"]["cu"][1:], p_g)
    assert res.lane("cu_g", gid)[0] == 0.0  # a month quantity, stored as 0 at t = 0
    _agree(res.lane("csm_0", gid)[0], want["csm_0"], p_g)
    _agree(res.lane("lc_0", gid)[0], want["lc_0"], p_g)
    _agree(res.lane("cu_next", gid), roll["n"], p_g)
    _agree(res.lane("csm", gid), roll["csm"], p_g)
    _agree(res.lane("release", gid)[1:], roll["r"][1:], p_g)
    _agree(res.lane("lc", gid), roll["lc"], p_g)
    lc = res.lane("lc", gid)
    np.testing.assert_array_equal(res.lane("lc_alloc", gid)[1:], lc[:-1] - lc[1:])
    np.testing.assert_array_equal(
        res.lane("lrc", gid), res.lane("fcf_g", gid) + res.lane("csm", gid)
    )
    # the CSM table's opening line is the previous month's closing CSM (annuity spec, "Outputs")
    csm = res.lane("csm", gid)
    np.testing.assert_array_equal(res.lane("csm_open", gid)[1:], csm[:-1])
    np.testing.assert_array_equal(
        res.lane("csm_pre", gid)[1:], res.lane("csm_open", gid)[1:] + res.lane("acc", gid)[1:]
    )


def test_the_undiscounted_switch(stages: dict[tuple[int, bool], RunResult]) -> None:
    """Example 1's CSM(12) with discount_coverage_units = False, and what the switch changes:
    N(t) is the plain sum of the later units, while the accretion stays on P_L."""
    on, off = stages[(1, True)], stages[(1, False)]
    gid = GID[1]
    assert _ten(off.lane("csm", gid)[12]) == _ten(float(CSM_12_UNDISCOUNTED))
    assert off.manifest["model"]["statics"] == {"discount_coverage_units": False}
    assert on.manifest["model"]["statics"] == {"discount_coverage_units": True}
    assert off.lane("csm_0", gid)[0] == on.lane("csm_0", gid)[0]
    assert (off.lane("d_cu", gid)[1:] == 1.0).all()
    np.testing.assert_array_equal(on.lane("d_cu", gid), on.lane("d_l", gid))
    np.testing.assert_array_equal(off.lane("j_l", gid), on.lane("j_l", gid))
    cu = off.lane("cu_g", gid)
    tail = np.cumsum(cu[::-1])[::-1]  # N(t) = units of months after t, undiscounted
    assert within(off.lane("cu_next", gid)[:-1], tail[1:], tail[1], rel=REL).all()
    assert on.lane("cu_next", gid)[0] < off.lane("cu_next", gid)[0]


def test_example_2_is_onerous_and_its_loss_component_runs_off_with_f(
    stages: dict[tuple[int, bool], RunResult],
) -> None:
    """CSM(0) = 0 exactly and LC(t) / F(t) is the same at t = 0, 1 and 12 (I13); BEL and RA
    equal policy 1's in example 1 bit for bit (both one-policy runs on identical inputs, D19)."""
    one, two = stages[(1, True)], stages[(2, True)]
    gid = GID[2]
    assert two.lane("csm", gid).max() == 0.0
    assert two.lane("release", gid).max() == 0.0
    lc, f = two.lane("lc", gid), two.lane("fcf_g", gid)
    ratios = [lc[t] / f[t] for t in (0, 1, 12)]
    assert within(ratios, ratios[0], 0.0, rel=REL).all()
    for name in ("bel_g", "ra_g", "fcf_g", "cu_g"):
        np.testing.assert_array_equal(two.lane(name, gid), one.lane(name, GID[1]))
    # the units still run: coverage units are computed for an onerous group too
    np.testing.assert_array_equal(two.lane("cu_next", gid), one.lane("cu_next", GID[1]))


def test_twelve_monthly_steps_equal_one_step_with_undiscounted_units(
    stages: dict[tuple[int, bool], RunResult], p_l: Path
) -> None:
    """Annuity spec, "Accretion and release": with undiscounted units the period-end CSM of
    twelve monthly steps is that of one step per period, because the ratio telescopes; with
    discounted units it is not, because a monthly step discounts within the period."""
    p = pq.read_table(p_l).column("rf_df").to_numpy()

    def one_step(res: RunResult, gid: str) -> float:
        csm_acc = (float(res.lane("csm", gid)[0]) * p[0]) / p[12]
        units = 0.0
        for x in res.lane("cu_g", gid)[1:13]:
            units += float(x)
        n_12 = float(res.lane("cu_next", gid)[12])
        return csm_acc - csm_acc * (units / (units + n_12))

    for example in (1, 3):
        gid = GID[example]
        off, on = stages[(example, False)], stages[(example, True)]
        monthly = float(off.lane("csm", gid)[12])
        assert within(one_step(off, gid), monthly, 0.0, rel=REL), example
        assert not within(one_step(on, gid), float(on.lane("csm", gid)[12]), 0.0, rel=1e-6)


# ----------------------------------------------------------------- M12: revenue and IFIE


@pytest.mark.parametrize("example", [1, 2, 3])
def test_m1_and_m2_hold_and_the_m12_lines_are_the_designs(
    example: int, stages: dict[tuple[int, bool], RunResult]
) -> None:
    """Design sections 9 and 11: M1 and M2 hold as run-time invariants on examples 1 to 3 with
    either switch value, and again recomputed from the stored lines; revenue and IFIE are the
    sums the design writes, in its grouping; the month's service items less the finance unwind
    are F(t - 1) - F(t), the para 51 items of an expected run ("Onerous groups and the loss
    component"); and for a group new at V the current and the locked-in curves are one curve,
    P_L = P, so f = j_L bit for bit."""
    gid = GID[example]
    for switch in (True, False):
        res = stages[(example, switch)]
        results = {r.name: r for r in res.invariants}
        assert results[M1].passed, switch
        assert results[M2].passed, switch
        x = {name: res.lane(name, gid) for name in (*M12_LINES, "lrc", "lc_alloc", "release")}
        x.update({name: res.lane(name, gid) for name in ("acc", "fcf_g", "lc", "j_l")})
        p_g = float(res.lane("p_g", gid)[0])
        lrc = x["lrc"]
        m1 = (x["ifie"][1:] - x["revenue"][1:]) - x["lc_svc"][1:]
        assert within(lrc[1:] - lrc[:-1], m1, p_g, rel=REL).all()
        m2 = x["lc_svc"][1:] - x["sar"][1:] * (x["ifie_bel"][1:] + x["ra_fin"][1:])
        assert within(x["lc_alloc"][1:], m2, p_g, rel=REL).all()
        np.testing.assert_array_equal(
            x["revenue"][1:], (x["svc"][1:] - x["lc_svc"][1:]) + x["release"][1:]
        )
        np.testing.assert_array_equal(
            x["ifie"][1:], (x["ifie_bel"][1:] + x["ra_fin"][1:]) + x["acc"][1:]
        )
        para_51 = x["svc"][1:] - (x["ifie_bel"][1:] + x["ra_fin"][1:])
        f = x["fcf_g"]
        assert within(para_51, f[:-1] - f[1:], p_g, rel=REL).all()
        np.testing.assert_array_equal(x["f"], x["j_l"])
        if example == 2:  # onerous: the loss component takes its share, the CSM is nil
            assert x["lc_svc"][1] > 0.0
            assert (x["release"] == 0.0).all()
            ratio = x["lc"][0] / f[0]
            assert within(x["sar"][1:13], ratio, 0.0, rel=REL).all()
        else:  # profitable: no loss component, so no share of anything
            assert (x["sar"] == 0.0).all()
            assert (x["lc_svc"] == 0.0).all()
        assert (x["revenue"][1:] > 0.0).all() or example == 2


def test_df_cur_is_the_composed_curve_of_the_cur_runs(
    compiled_new_group: dict[bool, CompiledModel],
    class_runs: dict[int, RunResult],
    curves: dict[str, Path],
    tmp_path: Path,
) -> None:
    """Design section 9: df_cur is the cur runs' composed curve, whose file the manifest
    discloses as the para 120 curve (``discloses_current_curve``). The loader requires its
    metadata to name the rf_df and ilp files every cur run read, and the provenance check then
    rebuilds it from them, so a df_cur that names no source, one composed from another discount
    file, one re-based 24 months on, or one whose values are not what its sources give is a
    LoadError naming scenario input df_cur, before anything is written."""
    stage = compiled_new_group[True]
    level = class_runs[1]
    why = "the stage's current curve is the composed curve of its cur basis's runs"
    other_file = data.discount_table(lambda s, t: data.rf_df(t, 0.05))
    other = composed_curve(other_file, TABLES["ilp"], T_EXAMPLE)
    _p_rec, later = locked_in_curve(SCENARIOS["rf_df"], TABLES["ilp"], 24, T_EXAMPLE - 24)
    good = pq.read_table(curves["df_cur"])
    forged = data.discount_table(lambda s, t: data.rf_df(t, 0.05))
    forged = forged.cast(good.schema.remove_metadata()).replace_schema_metadata(
        good.schema.metadata
    )
    rf_sha = _sha(SCENARIOS["rf_df"])
    cases = {
        "bare": (other_file, r"carries no metadata halley\.source_sha256", True),
        "other": (
            other,
            rf"halley\.source_sha256 is \w+, but basis cur run {level.run_id}: scenario input "
            rf"rf_df is {rf_sha}",
            True,
        ),
        "later": (later, r"halley\.tau0 is 24, a curve re-based 24 months after", True),
        "forged": (
            forged,
            r"it is not what its source \(halley\.rebased_from = \w+\) gives",
            False,
        ),
    }
    for name, (curve, message, explained) in cases.items():
        with pytest.raises(LoadError, match=rf"^scenario input df_cur: {message}") as info:
            _stage(
                stage, [level], {**curves, "df_cur": curve}, tmp_path / name, groups_file("OTHER")
            )
        assert (why in str(info.value)) is explained, name
        assert not (tmp_path / name).exists()


def test_m1_checks_the_current_curve_and_df_cur_is_deterministic(
    class_runs: dict[int, RunResult], curves: dict[str, Path], tmp_path: Path
) -> None:
    """M1 is the run-time check of the curve: in a stage that does not hold df_cur to the cur
    runs' files (no ``discloses_current_curve``), a df_cur at another level fails M1 at month 1,
    since M1 holds only on the curve the runs' BEL and RA unwind on. A df_cur of two scenarios
    beside the stage's mean series is refused by the one J = 1 check (D4)."""
    level = data.discount_table(lambda s, t: data.rf_df(t, 0.05))
    with pytest.raises(RunError, match=rf"invariant {re.escape(M1)} failed: group {GID[1]}, t = 1"):
        _run_unlocked(class_runs, {**curves, "df_cur": level}, tmp_path / "level")
    two = data.discount_table(n_scenarios=2)
    with pytest.raises(
        LoadError,
        match=r"the stage reads the mean series \w+, so it runs at S = 1; scenario input df_cur "
        r"has J = 2 scenarios.*require_deterministic",
    ):
        _run_unlocked(class_runs, {**curves, "df_cur": two}, tmp_path / "two")


# ---------------------------------------------------------------------------- invariants


def test_i11_to_i14_m1_and_m2_hold_as_run_time_invariants(
    stages: dict[tuple[int, bool], RunResult],
) -> None:
    for key, res in stages.items():
        results = {r.name: r for r in res.invariants}
        assert [r.name for r in res.invariants][-len(INVARIANTS) :] == list(INVARIANTS), key
        assert all(r.passed for r in res.invariants), key
        on_disk = json.loads((res.run_dir / "invariants.json").read_text(encoding="utf-8"))
        assert [r["name"] for r in on_disk][-len(INVARIANTS) :] == list(INVARIANTS)
        for name in (*EXACT, I14):  # the exact ones are exact, and so is I14 here
            assert results[name].worst == 0.0, (key, name)
    # the last month with units releases the whole CSM: CSM(H) = 0 although CSM(H - 1) > 0
    csm = stages[(1, True)].lane("csm", GID[1])
    assert csm[-1] == 0.0
    assert csm[-2] > 0.0


class Unchecked(NewGroup):
    """``NewGroup`` without its member rules: I14 alone stands for G3."""

    member_rules: ClassVar[tuple[MemberRule, ...]] = ()


class Unequal(NewGroup):
    """A release share that overstates the units: I12 fails, I11 still holds."""

    @var
    def units(self, t: Int) -> CoverageUnits:
        return self.series.cu_g(t) + self.cu_next(t) * 1.001


class Fading(NewGroup):
    """A loss component that runs off faster than F: I13 fails."""

    @var(init=lambda g: g.lc_0())
    def lc(self, t: Int) -> Money:
        f0 = self.series.fcf_g(t - 1)
        ratio = self.series.fcf_g(t) / where(f0 > 0, f0, 1.0)
        return where(f0 > 0, (self.lc(t - 1) * ratio) * 0.999, 0.0)


class FastAccretion(NewGroup):
    """An accretion a ten-thousandth above the locked-in rate: the release per unit no longer
    steps by a_L, so I12's step form fails; I11 and the equal allocation still hold."""

    @var
    def acc(self, t: Int) -> Money:
        return (self.j_l(t) * self.csm(t - 1)) * 1.0001


class Leaky(NewGroup):
    """A closing CSM that keeps a millionth of the release: I11's recursion fails."""

    @var(init=lambda g: g.csm_0())
    def csm(self, t: Int) -> Money:
        return self.csm_pre(t) - self.release(t) * 0.999999


class KeepsTheTail(NewGroup):
    """A release a little short of the share of the units: the last month with units leaves a
    CSM behind, so I11's exact zero fails there, at H; the recursion still holds."""

    @var
    def release(self, t: Int) -> Money:
        u = self.units(t)
        share = self.series.cu_g(t) / where(u > 0, u, 1.0)
        return where(u > 0, self.csm_pre(t) * share, self.csm_pre(t)) * 0.9999999


class LcBlind(NewGroup):
    """Revenue that keeps the loss component's share of the service items: M1 fails where the
    loss component takes a share, in an onerous group."""

    @var
    def revenue(self, t: Int) -> Money:
        return self.svc(t) + self.release(t)


class LateShare(NewGroup):
    """The loss component's share taken of the closing F: M2 fails; M1, in which the share
    cancels, does not see it."""

    @var
    def sar(self, t: Int) -> Factor:
        f1 = self.series.fcf_g(t)
        return self.lc(t - 1) / where(f1 > 0, f1, 1.0)


class DriftingLc(NewGroup):
    """A loss component that grows from 0 in a profitable group, by 5e-13 of F a month: I13's
    ratio fails, since LC(0) = 0 requires LC(t) = 0 wherever F(t) > 0; every other invariant,
    M2 included, holds within its tolerance."""

    @var(init=lambda g: g.lc_0())
    def lc(self, t: Int) -> Money:
        f0 = self.series.fcf_g(t - 1)
        ratio = self.series.fcf_g(t) / where(f0 > 0, f0, 1.0)
        return where(f0 > 0, self.lc(t - 1) * ratio + self.series.fcf_g(t) * 5e-13, 0.0)


class OverRelease(NewGroup):
    """A release of one and a half times the CSM before it at month 2: the CSM goes negative,
    and I11's sign check fails there; the recursion still holds, at a tolerance taken from the
    terms' absolute values."""

    @var
    def release(self, t: Int) -> Money:
        u = self.units(t)
        share = self.series.cu_g(t) / where(u > 0, u, 1.0)
        spec = where(u > 0, self.csm_pre(t) * share, self.csm_pre(t))
        return where(t == 2, self.csm_pre(t) * 1.5, spec)


class LcKeptAtZeroF(NewGroup):
    """A loss component carried over where F(t) = 0, at the run-off, instead of released."""

    @var(init=lambda g: g.lc_0())
    def lc(self, t: Int) -> Money:
        f0 = self.series.fcf_g(t - 1)
        ratio = self.series.fcf_g(t) / where(f0 > 0, f0, 1.0)
        spec = where(f0 > 0, self.lc(t - 1) * ratio, 0.0)
        return where(self.series.fcf_g(t) > 0, spec, self.lc(t - 1))


BROKEN: list[tuple[type[NewGroup], int, str, str, str]] = [
    (Unequal, 1, "OTHER", "I12 R(t) / CU_g(t) = CSM(t) / N(t)", "t = 1"),
    (FastAccretion, 1, "OTHER", INVARIANTS[4], "t = 1"),
    (Leaky, 1, "OTHER", INVARIANTS[0], "t = 1"),
    (OverRelease, 1, "OTHER", "I11 CSM(t) >= 0", "t = 2"),
    (KeepsTheTail, 1, "OTHER", I11_ZERO, f"t = {T_EXAMPLE}"),
    (Fading, 4, "ONEROUS", "I13 LC(t) / F(t) = LC(0) / F(0)", "t = 1"),
    (DriftingLc, 1, "OTHER", "I13 LC(t) / F(t) = LC(0) / F(0)", "t = 1"),
    (LcKeptAtZeroF, 4, "ONEROUS", "I13 LC(t) = 0 where F(t) = 0", f"t = {T_EXAMPLE}"),
    (LcBlind, 4, "ONEROUS", M1, "t = 1"),
    (LateShare, 4, "ONEROUS", M2, "t = 1"),
]
"""Each broken stage, the policy of its group, the bucket, the first invariant that fails and
where."""


@pytest.mark.parametrize(("broken", "pid", "bucket", "name", "month"), BROKEN)
def test_the_invariants_fail_on_broken_stages(
    broken: type[NewGroup],
    pid: int,
    bucket: str,
    name: str,
    month: str,
    class_runs: dict[int, RunResult],
    curves: dict[str, Path],
    tmp_path: Path,
) -> None:
    """Each invariant fails on a stage broken where it looks, at its declaration, naming the
    group and the month; every invariant declared before it passed on the same run."""
    gid = f"IA-2025-{bucket}"
    with pytest.raises(
        RunError, match=rf"^invariant {re.escape(name)} failed: group {gid}, {month}: lhs = "
    ) as info:
        _stage(build(broken), [class_runs[pid]], curves, tmp_path / "stage", groups_file(bucket))
    declared = {i.name: i for i in NewGroup.invariants}[name]
    loc, at = info.value.loc, declared.loc
    assert loc is not None
    assert at is not None
    assert loc.file.endswith("csm.py")
    assert (loc.file, loc.line) == (at.file, at.line)


def test_i14_fails_when_g3_is_not_checked(
    compiled_annuity: dict[str, CompiledModel],
    curves: dict[str, Path],
    tmp_path: Path,
) -> None:
    """G3 broken at load: policy 1 at a premium of 175,000 in an OTHER group, checked only by
    I14 when the member rules are dropped."""
    onerous = _project(
        compiled_annuity["level"], _one_policy(1, single_premium=175_000.0), tmp_path / "level"
    )
    with pytest.raises(
        RunError, match=rf"invariant {re.escape(I14)} failed: group {GID[1]}: lhs = 6840\.39757"
    ) as info:
        _stage(build(Unchecked), [onerous], curves, tmp_path / "unchecked", groups_file("OTHER"))
    loc, at = info.value.loc, I14_INVARIANT.loc
    assert loc is not None
    assert at is not None
    assert (loc.file, loc.line) == (at.file, at.line)


def test_i14_accepts_contracts_that_each_tie(
    compiled_annuity: dict[str, CompiledModel],
    curves: dict[str, Path],
    tmp_path: Path,
) -> None:
    """G3 decides each contract on fl(P_i - fl(BEL_i(0) + RA_i(0))); the group sums P_g and
    FCF_g(0) = sum BEL + sum RA in another order. Two contracts each priced exactly at their
    FCF are not onerous, so G3 accepts them in an OTHER group, yet the group's LC(0) is a few
    ulps of P_g: I14 accepts it within 1e-12 x P_g. Policy 1 and a copy at another instalment;
    the first instalment whose sums do not tie is taken, so the test does not hang on one
    platform's last bits."""
    level = compiled_annuity["level"]
    base = data.example_policy(1)
    stage = build(NewGroup)
    for k, instalment in enumerate((1259.0, *(1000.0 + 37.0 * i for i in range(40)))):
        rows = [dict(base), {**base, "policy_id": 5, "instalment_pm": instalment}]
        first = _project(level, data.model_points_table(rows), tmp_path / f"first-{k}")
        bel = [float(first.lane("bel", p)[0]) for p in (1, 5)]
        ra = [float(first.lane("ra", p)[0]) for p in (1, 5)]
        fcf = [bel[0] + ra[0], bel[1] + ra[1]]
        gap = ((bel[0] + bel[1]) + (ra[0] + ra[1])) - (fcf[0] + fcf[1])
        if gap > 0:
            break
    else:
        pytest.skip("no instalment leaves a group balance on this platform")
    assert gap <= 1e-12 * (fcf[0] + fcf[1])
    tied = [{**row, "single_premium": f} for row, f in zip(rows, fcf, strict=True)]
    runs = _project(level, data.model_points_table(tied), tmp_path / f"tied-{k}")
    assert [bool(runs.lane("onerous", p)[0]) for p in (1, 5)] == [False, False]
    res = _stage(stage, [runs], curves, tmp_path / "stage", groups_file("OTHER"))
    assert res.lane("lc_0", GID[1])[0] == gap
    assert res.lane("csm_0", GID[1])[0] == 0.0
    i14 = {r.name: r for r in res.invariants}[I14]
    assert i14.passed
    assert i14.worst > 0.0


class NoPremium(CsmRollForward):
    """A roll-forward that defines its opening balances but not P_g."""

    @var
    def csm_0(self) -> Money:
        return self.series.fcf_g.at(0) * 0.0

    @var
    def lc_0(self) -> Money:
        return self.series.fcf_g.at(0) * 0.0


def test_a_hook_left_undefined_is_a_build_error_at_the_subclass() -> None:
    """Errors point at the model source: the class being built, not the hook's line in csm.py."""
    with pytest.raises(
        BuildError, match="NoPremium does not define p_g, a hook of CsmRollForward"
    ) as info:
        build(NoPremium)
    loc = info.value.loc
    assert loc is not None
    assert (loc.file, loc.line) == (__file__, inspect.getsourcelines(NoPremium)[1])
    with pytest.raises(BuildError, match="CsmRollForward is abstract"):
        build(CsmRollForward)


# -------------------------------------------------------------------------- member rules


def test_g3_names_the_contract_on_the_wrong_side_of_its_group(
    compiled_annuity: dict[str, CompiledModel],
    compiled_new_group: dict[bool, CompiledModel],
    curves: dict[str, Path],
    tmp_path: Path,
) -> None:
    """Annuity spec, example 2's data-contract test: example 1 with policy 1's premium at
    175,000 and its bucket left OTHER fails G3, naming policy 1's row in mp_level; a profitable
    contract in an ONEROUS group fails it too."""
    path = tmp_path / "mp_level.parquet"
    pq.write_table(_one_policy(1, single_premium=175_000.0), path)
    level = _project(compiled_annuity["level"], path, tmp_path / "level")
    assert level.lane("onerous", 1)[0]
    stage = compiled_new_group[True]
    with pytest.raises(LoadError) as info:
        _stage(stage, [level], curves, tmp_path / "stage", groups_file("OTHER"))
    message = str(info.value)
    copy = level.run_dir / "inputs" / "model_points.parquet"
    assert message.startswith(
        f"member rule {G3_NAME} fails for 1 of 1 members of basis cur run {level.run_id}, first "
        f"at model points ({copy}): row 0, policy 1, which the run read from {path}"
    )
    assert message.count(level.run_id) == 1
    assert "an onerous contract outside an ONEROUS group" in message
    assert info.value.loc is None  # the data's failure, not the rule's
    assert not (tmp_path / "stage").exists()

    path = tmp_path / "mp_level_onerous.parquet"
    pq.write_table(_one_policy(1, profit_bucket="ONEROUS"), path)
    profitable = _project(compiled_annuity["level"], path, tmp_path / "level-2")
    assert not profitable.lane("onerous", 1)[0]
    with pytest.raises(LoadError, match=f"member rule {G3_NAME} fails for 1 of 1 members"):
        _stage(stage, [profitable], curves, tmp_path / "stage-2", groups_file("ONEROUS"))


def test_g5_rejects_a_contract_in_force_before_v(
    compiled_annuity: dict[str, CompiledModel],
    compiled_new_group: dict[bool, CompiledModel],
    curves: dict[str, Path],
    tmp_path: Path,
) -> None:
    """A contract issued a year before V, in payment since, in a group recognised then: G5
    rejects the group, naming the contract's row (an in-force group is measured by the
    in-force stage)."""
    issued = dt.date(2024, 12, 31)
    points = _one_policy(1, issue_date=issued, cohort=2024, annuity_start_date=dt.date(2025, 1, 1))
    in_force = _project(compiled_annuity["level"], points, tmp_path / "in-force")
    with pytest.raises(LoadError) as info:
        _stage(
            compiled_new_group[True],
            [in_force],
            curves,
            tmp_path / "stage",
            groups_file("OTHER", recognition=issued, cohort=2024),
        )
    message = str(info.value)
    assert message.startswith(
        f"member rule {G5_NAME} fails for 1 of 1 members of basis cur run {in_force.run_id}, "
        "first at model points ("
    )
    assert ": row 0, policy 1 (issue_date = V" in message


def test_g3_reads_a_deterministic_run(
    compiled_annuity: dict[str, CompiledModel],
    compiled_new_group: dict[bool, CompiledModel],
    curves: dict[str, Path],
    tmp_path: Path,
) -> None:
    """D4: a basis of two scenarios reaches the stage through a mean, but G3 reads each
    contract's onerous output, a test at one scenario: require_deterministic says so."""
    two = {
        "rf_df": data.discount_table(n_scenarios=2),
        "cpi": data.cpi_table(n_scenarios=2),
    }
    level = _project(compiled_annuity["level"], _one_policy(1), tmp_path / "two", scenarios=two)
    assert level.n_scenarios == 2
    with pytest.raises(
        LoadError,
        match=rf"G3 reads each contract's para 47 test, onerous, at one scenario; basis cur run "
        rf"{level.run_id} has J = 2 scenarios.*require_deterministic",
    ):
        _stage(compiled_new_group[True], [level], curves, tmp_path / "stage", groups_file("OTHER"))


def _always(view: Any) -> Any:
    return np.ones(1, dtype=bool)


def test_member_rules_are_checked_at_build_and_at_load(
    compiled_annuity: dict[str, CompiledModel],
    class_runs: dict[int, RunResult],
    curves: dict[str, Path],
    tmp_path: Path,
) -> None:
    line = 0

    def declare(**kw: Any) -> MemberRule:
        nonlocal line
        rule = MemberRule("mine", **kw)
        line = rule.loc.line if rule.loc is not None else 0
        return rule

    class OtherBasis(NewGroup):
        member_rules: ClassVar[tuple[MemberRule, ...]] = (declare(check=_always, basis="old"),)

    with pytest.raises(BuildError, match="member rule mine checks basis 'old', which") as info:
        build(OtherBasis)
    assert info.value.loc is not None
    assert info.value.loc.line == line
    assert info.value.loc.file == __file__
    one_name: Any = "onerous"  # a string where a tuple of names belongs
    with pytest.raises(BuildError, match=r"reads is a tuple of output names, as \('onerous',\)"):
        MemberRule("mine", _always, reads=one_name)
    with pytest.raises(BuildError, match="its basis is a Basis name"):
        MemberRule("mine", _always, "two words")
    no_comma: Any = MemberRule("mine", _always)  # (rule) where (rule,) belongs
    without_comma = type("NoComma", (NewGroup,), {"member_rules": no_comma})
    with pytest.raises(BuildError, match=r"NoComma\.member_rules must be a tuple of MemberRule"):
        build(without_comma)

    class Named(NewGroup):
        @var
        def member_rules(self) -> Money:
            return self.p_g()

    with pytest.raises(BuildError, match="member_rules is a stage class attribute in a stage"):
        build(Named)

    def stage_with(rule: MemberRule) -> CompiledModel:
        class WithRule(NewGroup):
            member_rules: ClassVar[tuple[MemberRule, ...]] = (rule,)

        return build(WithRule)

    # the rule's own faults are located at its declaration; the data's are not
    cases = {
        "floats": (MemberRule("floats", lambda v: np.ones(1)), "must return booleans, not float64"),
        "unstored": (
            MemberRule("unstored", _always, reads=("q_mth",)),
            f"basis cur run {class_runs[1].run_id} did not store q_mth; add it to outputs",
        ),
        "raises": (MemberRule("raises", lambda v: 1 / 0), "raised ZeroDivisionError"),
        "shape": (
            MemberRule("shape", lambda v: np.ones(3, dtype=bool)),
            "returned shape \\(3,\\); it gives one value per row",
        ),
        # an output the rule reads without declaring it: the manifest records the declared reads
        # as what the rule read (D21), so the view serves no other
        "undeclared": (
            MemberRule("undeclared", lambda v: v.output("bel")[0, :, 0] > 0),
            r"output\('bel'\) reads bel, which the rule does not declare in reads \(it declares: "
            r"none\)",
        ),
        "other": (
            MemberRule("other", lambda v: v.output("bel")[0, :, 0] > 0, reads=("onerous",)),
            r"reads bel, which the rule does not declare in reads \(it declares: onerous\)",
        ),
    }
    for name, (rule, message) in cases.items():
        with pytest.raises(LoadError, match=f"^member rule {name}.*{message}") as info:
            _stage(stage_with(rule), [class_runs[1]], curves, tmp_path / name, groups_file("OTHER"))
        if name in ("floats", "raises", "shape", "undeclared", "other"):
            assert info.value.loc == rule.loc, name
            assert info.value.loc is not None
            assert info.value.loc.file == __file__
        else:
            assert info.value.loc is None, name
        assert not (tmp_path / name).exists()

    # a run of two members: one bool for every row, or one per row; a (1,) answer would be one
    # row's answer applied to both
    rows = [data.example_policy(1), {**data.example_policy(1), "policy_id": 5}]
    two = _project(compiled_annuity["level"], data.model_points_table(rows), tmp_path / "two")
    one_row = MemberRule("one_row", lambda v: np.ones(1, dtype=bool))
    with pytest.raises(
        LoadError,
        match=r"^member rule one_row returned shape \(1,\); it gives one value per row of the "
        r"model point copy, 2 here, or one bool for every row",
    ) as info:
        _stage(stage_with(one_row), [two], curves, tmp_path / "one-row", groups_file("OTHER"))
    assert info.value.loc == one_row.loc
    declared = MemberRule(
        "declared", lambda v: np.asarray(v.output("bel")[0, :, 0] > 0), reads=("bel",)
    )
    for rule in (MemberRule("every", lambda v: True), declared):
        res = _stage(stage_with(rule), [two], curves, tmp_path / rule.name, groups_file("OTHER"))
        assert res.manifest["member_rules"] == [
            {
                "rule": rule.name,
                "basis": "cur",
                "reads": list(rule.reads),
                "runs": [{"run_id": two.run_id, "members": 2}],
            }
        ]


def test_run_refuses_a_stage_with_member_rules(
    compiled_new_group: dict[bool, CompiledModel], class_runs: dict[int, RunResult], tmp_path: Path
) -> None:
    """run() sees only the group series, so it would skip G3 and G5: it refuses the stage."""
    series = group_series(NewGroup, {"cur": [class_runs[1]]}, {GID[1]: [1]})
    groups = groups_file("OTHER").append_column("group_id", pa.array([GID[1]]))
    with pytest.raises(
        LoadError,
        match=rf"NewGroup checks every member of its bases \({G5_NAME}, {G3_NAME}\); run it "
        "with halley.aggregation.stage.run_aggregation",
    ):
        run(
            compiled_new_group[True],
            model_points=groups,
            series=series,
            settings=RunSettings(valuation_date=VALUATION_DATE),
            run_dir=tmp_path / "stage",
        )
    assert not (tmp_path / "stage").exists()


def test_run_refuses_a_stage_that_locks_in_the_curve_of_its_runs(
    class_runs: dict[int, RunResult], curves: dict[str, Path], tmp_path: Path
) -> None:
    """P_L = P for a group new at V is verified by run_aggregation from the files the cur runs
    read. run() reads no basis run, and no invariant can see a p_l that is not the runs' curve
    (the bare discount file, without its illiquidity premium, passes every one), so it refuses
    a recognised_at_valuation stage even without member rules, before anything is written."""
    series = group_series(Unchecked, {"cur": [class_runs[1]]}, {GID[1]: [1]})
    groups = groups_file("OTHER").append_column("group_id", pa.array([GID[1]]))
    bare = pq.read_table(SCENARIOS["rf_df"])
    stage = build(Unchecked)
    for name, p_l in (("bare", bare), ("good", curves["p_l"])):
        with pytest.raises(
            LoadError,
            match=r"^Unchecked locks in the composed curve of its cur basis's runs \(P_L = P\), "
            r"which only run_aggregation can verify; run it with "
            r"halley\.aggregation\.stage\.run_aggregation",
        ):
            run(
                stage,
                model_points=groups,
                scenarios={**curves, "p_l": p_l},
                series=series,
                settings=RunSettings(valuation_date=VALUATION_DATE),
                run_dir=tmp_path / name,
            )
        assert not (tmp_path / name).exists()


# --------------------------------------------------------------- the locked-in curve


def test_a_new_group_locks_in_the_composed_curve_of_its_runs(
    compiled_new_group: dict[bool, CompiledModel],
    class_runs: dict[int, RunResult],
    curves: dict[str, Path],
    tmp_path: Path,
) -> None:
    """Annuity spec, "Locked-in rates": for a group new at V, P_L = P of the same run. The
    stage's p_l must name, in its metadata, the discount file and the ILP table its cur runs
    read, with tau0 = 0, so the provenance check rebuilds it from them: a curve naming no source
    (the bare discount file, without its illiquidity premium, or a curve scaled by hand), one
    re-based 24 months on, or one composed with another ILP or discount file is a LoadError
    naming scenario input p_l."""
    stage = compiled_new_group[True]
    level = class_runs[1]
    why = "a group recognised at the valuation date locks in the composed curve of the runs"
    bare = pq.read_table(SCENARIOS["rf_df"])
    table = pq.read_table(curves["p_l"])
    scaled = pa.table(
        {
            "scenario": table.column("scenario"),
            "t": table.column("t"),
            "rf_df": pa.array([0.999 * x for x in table.column("rf_df").to_pylist()], pa.float64()),
        }
    )
    _p_rec, later = locked_in_curve(SCENARIOS["rf_df"], TABLES["ilp"], 24, T_EXAMPLE - 24)
    _p_rec, no_ilp = locked_in_curve(SCENARIOS["rf_df"], data.ilp_table(0.0), 0, T_EXAMPLE)
    other_file = data.discount_table(lambda s, t: data.rf_df(t, 0.05))
    _p_rec, other = locked_in_curve(other_file, TABLES["ilp"], 0, T_EXAMPLE)
    rf_sha, ilp_sha = _sha(SCENARIOS["rf_df"]), _sha(TABLES["ilp"])
    cases = {
        "bare": (bare, r"carries no metadata halley\.source_sha256"),
        "scaled": (scaled, r"carries no metadata halley\.source_sha256"),
        "later": (later, r"halley\.tau0 is 24, a curve re-based 24 months after"),
        "no-ilp": (
            no_ilp,
            rf"halley\.ilp_sha256 is \w+, but basis cur run {level.run_id}: table ilp is "
            rf"{ilp_sha}",
        ),
        "other": (
            other,
            rf"halley\.source_sha256 is \w+, but basis cur run {level.run_id}: scenario input "
            rf"rf_df is {rf_sha}",
        ),
    }
    for name, (curve, message) in cases.items():
        with pytest.raises(LoadError, match=rf"^scenario input p_l: {message}") as info:
            _stage(stage, [level], {**curves, "p_l": curve}, tmp_path / name, groups_file("OTHER"))
        assert why in str(info.value), name
        assert not (tmp_path / name).exists()


def test_the_runs_of_a_cur_basis_compose_one_curve(
    compiled_annuity: dict[str, CompiledModel],
    compiled_new_group: dict[bool, CompiledModel],
    class_runs: dict[int, RunResult],
    curves: dict[str, Path],
    tmp_path: Path,
) -> None:
    """The group's locked-in curve is the composed curve of every run of its cur basis, so the
    runs read one discount file and one ILP table: a class run on an ILP table of the same
    values in other bytes (written uncompressed) composes its curve from another file, a
    LoadError naming both runs' copies before anything is written. Basis.check compares the
    scenario inputs only, so this is where the tables are held together."""
    ilp = tmp_path / "ilp-uncompressed.parquet"
    pq.write_table(pq.read_table(TABLES["ilp"]), ilp, compression="none")
    assert _sha(ilp) != _sha(TABLES["ilp"])
    fixed = run(
        compiled_annuity[_class_of(2)],
        model_points=_one_policy(2),
        tables={**TABLES, "ilp": ilp},
        scenarios=SCENARIOS,
        settings=RunSettings(valuation_date=VALUATION_DATE, horizon=T_EXAMPLE),
        run_dir=tmp_path / "policy-2",
    )
    where = tmp_path / "stage"
    with pytest.raises(
        LoadError,
        match=rf"^basis cur run {fixed.run_id}: table ilp .* is not the file basis cur run "
        rf"{class_runs[1].run_id}: table ilp is .*; the runs of basis cur compose one curve, the "
        r"group's locked-in curve p_l$",
    ):
        _stage(
            compiled_new_group[True], [class_runs[1], fixed], curves, where, groups_file("OTHER")
        )
    assert not where.exists()


def test_the_stage_flags_are_checked_at_build() -> None:
    """``reads_to_run_off``, ``recognised_at_valuation`` and ``discloses_current_curve`` are
    bools, and a stage recognised at the valuation date, or one that discloses the current
    curve, declares the curve the loader verifies against its cur runs."""

    class Stringly(NewGroup):
        reads_to_run_off: Any = "yes"

    with pytest.raises(BuildError, match=r"Stringly\.reads_to_run_off is True or False, not 'yes'"):
        build(Stringly)

    class Numeric(NewGroup):
        discloses_current_curve: Any = 1

    with pytest.raises(
        BuildError, match=r"Numeric\.discloses_current_curve is True or False, not 1"
    ):
        build(Numeric)

    class NoCurve(NewGroup):
        scenario_inputs = (DF_CUR,)

    with pytest.raises(
        BuildError,
        match="NoCurve is recognised_at_valuation, so its locked-in curve p_l is the composed "
        "curve of its cur basis's runs, but it declares no scenario input p_l",
    ):
        build(NoCurve)

    class NoCurrent(NewGroup):
        scenario_inputs = (P_L,)

    with pytest.raises(
        BuildError,
        match="NoCurrent is discloses_current_curve, so the current curve it discloses df_cur "
        "is the composed curve of its cur basis's runs, but it declares no scenario input "
        "df_cur",
    ):
        build(NoCurrent)

    class Named(NewGroup):
        @var
        def discloses_current_curve(self) -> Money:
            return self.p_g()

    with pytest.raises(BuildError, match="discloses_current_curve is a stage class attribute"):
        build(Named)


class Unlocked(NewGroup):
    """``NewGroup`` without its member rules and without the checks of its curves against the
    cur runs' files, so that its p_l is held by the scenario rule alone and its df_cur by M1, the
    run-time check."""

    member_rules: ClassVar[tuple[MemberRule, ...]] = ()
    recognised_at_valuation: ClassVar[bool] = False
    discloses_current_curve: ClassVar[bool] = False


def _run_unlocked(
    class_runs: dict[int, RunResult], curves: dict[str, Any], run_dir: Path, **settings: Any
) -> RunResult:
    return _stage(
        build(Unlocked), [class_runs[1]], curves, run_dir, groups_file("OTHER"), **settings
    )


def test_the_locked_in_curve_starts_at_one(
    class_runs: dict[int, RunResult], curves: dict[str, Path], tmp_path: Path
) -> None:
    """The stage's p_l is re-based to V: p_l(0) = 1 is a rule of the input, which a stage whose
    curve is not the cur runs' (a stage of no ``recognised_at_valuation``) is held to alone."""
    table = pq.read_table(curves["p_l"])
    shifted = pa.table(
        {
            "scenario": table.column("scenario"),
            "t": table.column("t"),
            "rf_df": pa.array([0.999 * x for x in table.column("rf_df").to_pylist()], pa.float64()),
        }
    )
    with pytest.raises(LoadError, match=r"scenario input p_l.*: rule p_l\(0\) = 1 fails"):
        _run_unlocked(class_runs, {**curves, "p_l": shifted}, tmp_path / "s")
    assert _run_unlocked(class_runs, dict(curves), tmp_path / "ok").H == T_EXAMPLE


# ---------------------------------------------------------------- the horizon and the run-off


def test_a_horizon_that_cuts_the_run_off_is_refused(
    compiled_new_group: dict[bool, CompiledModel],
    class_runs: dict[int, RunResult],
    curves: dict[str, Path],
    tmp_path: Path,
) -> None:
    """N(t) needs every later month of the units, so a stage horizon below the run-off would
    release the whole CSM at H (D5: a run with a different T agrees). Example 1 at H = 12 is a
    LoadError naming the basis, the run, the variable and the first month dropped, from
    run_aggregation (the series loader's own check, for a stage run through run(), is
    test_the_run_off_check_names_the_scenario_of_a_per_scenario_series)."""
    level = class_runs[1]
    with pytest.raises(LoadError) as info:
        _stage(
            compiled_new_group[True],
            [level],
            curves,
            tmp_path / "h12",
            groups_file("OTHER"),
            horizon=12,
        )
    message = str(info.value)
    cu_13 = float(level.lane("cu", 1)[13])
    assert message.startswith(
        f"basis cur run {level.run_id}: cu of policy 1 is {cu_13!r} at t = 13, after the stage's "
        "horizon H = 12 (RunSettings.horizon); the loader cuts series cu_g at H"
    )
    assert message.endswith(
        "NewGroup reads its series to the run-off, so its results would change with H (D5): "
        "leave RunSettings.horizon out or set it to at least 655"
    )
    assert not (tmp_path / "h12").exists()


def test_stage_results_do_not_depend_on_t(
    compiled_annuity: dict[str, CompiledModel],
    compiled_new_group: dict[bool, CompiledModel],
    class_runs: dict[int, RunResult],
    curves: dict[str, Path],
    tmp_path: Path,
) -> None:
    """D5 and design section 11: policy 2 (fixed escalation) runs off at month 623. Its stage on
    the runs at T = 655, the same stage cut at H = 623 and the stage on a run at T = 623 agree
    on every line within 1e-12 x P_g over months 0..623, and after them the run at T = 655
    holds exact zeros; H = 622 would cut the run-off and is refused."""
    long = class_runs[2]
    assert long.lane("run_off_month", 2)[0] == 623
    short = _project(compiled_annuity["fixed"], _one_policy(2), tmp_path / "short", horizon=623)
    stage = compiled_new_group[True]
    gid = GID[3]
    full = _stage(stage, [long], curves, tmp_path / "full", groups_file("OTHER"))
    cut = _stage(stage, [long], curves, tmp_path / "cut", groups_file("OTHER"), horizon=623)
    own = _stage(stage, [short], curves, tmp_path / "own", groups_file("OTHER"))
    assert (full.H, cut.H, own.H) == (T_EXAMPLE, 623, 623)
    p_g = float(full.lane("p_g", gid)[0])
    lines = ("csm", "lc", "release", "cu_next", "acc", "lrc", "revenue", "ifie", "svc")
    for name in lines:
        whole = full.lane(name, gid)
        for other in (cut, own):
            _agree(other.lane(name, gid), whole[:624], p_g)
        assert (whole[624:] == 0.0).all(), name
    for name in ("csm_0", "lc_0", "p_g"):
        for other in (cut, own):
            _agree(other.lane(name, gid), full.lane(name, gid), p_g)
    assert all(r.passed for r in (*cut.invariants, *own.invariants))
    with pytest.raises(
        LoadError, match=r"leave RunSettings\.horizon out or set it to at least 623"
    ):
        _stage(stage, [long], curves, tmp_path / "h622", groups_file("OTHER"), horizon=622)


class EachToTheRunOff(Aggregation):
    """A stage that reads a per-scenario series to the run-off, keyed on ``group_id`` alone so
    that run() runs it on ``group_series``."""

    model_point = Schema(fields=(Field("group_id", "string", unique=True),), id_field="group_id")
    bases = (Basis("cur"),)
    series = (GroupSum("cu_each", "cur", "cu", "CoverageUnits", scenarios="each"),)
    reads_to_run_off: ClassVar[bool] = True
    outputs: ClassVar[tuple[str, ...] | None] = ("units",)

    @var
    def units(self, t: Int) -> CoverageUnits:
        return self.series.cu_each(t)


def test_the_run_off_check_names_the_scenario_of_a_per_scenario_series(
    compiled_annuity: dict[str, CompiledModel], tmp_path: Path
) -> None:
    """The series loader's refusal of a horizon that cuts the run-off names the value that would
    be dropped: for a series of several scenarios, the scenario that holds it and its value
    there, not scenario 0's."""
    two = {"rf_df": data.discount_table(n_scenarios=2), "cpi": data.cpi_table(n_scenarios=2)}
    level = _project(compiled_annuity["level"], _one_policy(1), tmp_path / "two", scenarios=two)
    s = group_series(EachToTheRunOff, {"cur": [level]}, {GID[1]: [1]})["cu_each"]
    values = np.array(s.values)
    assert values.shape == (T_EXAMPLE + 1, 1, 2)
    values[13:, :, 0] = 0.0  # scenario 0 holds no units after month 12, scenario 1 does
    late = float(values[13, 0, 1])
    assert late > 0.0
    groups = pa.table({"group_id": pa.array([GID[1]])})
    with pytest.raises(
        LoadError,
        match=rf"^series cu_each: basis cur \(run {level.run_id}\) holds {re.escape(repr(late))} "
        rf"for group {GID[1]}, scenario 1 at t = 13, after the stage's horizon H = 12, where the "
        r"loader cuts it; .* at least 655$",
    ):
        run(
            build(EachToTheRunOff),
            model_points=groups,
            series={"cu_each": dataclasses.replace(s, values=values)},
            settings=RunSettings(valuation_date=VALUATION_DATE, horizon=12),
            run_dir=tmp_path / "stage",
        )
    assert not (tmp_path / "stage").exists()


# ------------------------------------------------------------------ the run directory


def test_the_aggregation_run_is_keyed_on_the_group_and_discloses(
    stages: dict[tuple[int, bool], RunResult],
    class_runs: dict[int, RunResult],
    curves: dict[str, Path],
) -> None:
    res = stages[(3, True)]
    gid, d, m = GID[3], res.run_dir, res.manifest
    assert res.kind == "aggregation"
    assert m["outputs"]["kind"] == "aggregation"
    assert m["outputs"]["id_field"] == "group_id"
    monthly = pq.read_table(d / "outputs" / "monthly.parquet")
    scalars = pq.read_table(d / "outputs" / "scalars.parquet")
    assert monthly.column_names[:3] == ["group_id", "scenario", "t"]
    assert scalars.column_names[:2] == ["group_id", "scenario"]
    assert set(monthly.column("group_id").to_pylist()) == {gid}
    assert monthly.num_rows == T_EXAMPLE + 1
    for name in ("csm_open", "csm", "lc", "release", "cu_next", "cu_g", "fcf_g", "lrc"):
        assert name in monthly.column_names, name
    for name in M12_LINES:
        assert name in monthly.column_names, name
    for name in ("p_g", "csm_0", "lc_0", "lrc_0"):
        assert name in scalars.column_names, name
    variables = m["outputs"]["variables"]
    assert variables["csm"] == {"time": True, "unit": "Money", "start": 0, "role": None}
    assert variables["release"] == {"time": True, "unit": "Money", "start": 1, "role": None}
    assert variables["cu_g"]["unit"] == "CoverageUnits"
    assert variables["j_l"]["unit"] == "Rate[Monthly]"
    assert variables["f"]["unit"] == "Rate[Monthly]"
    assert variables["sar"]["unit"] == "Factor"
    assert variables["revenue"] == {"time": True, "unit": "Money", "start": 1, "role": None}
    assert variables["p_g"] == {"time": False, "unit": "Money", "start": 0, "role": None}
    # the disclosures: alpha of the cur basis, the para 120 curve df_cur and the files it was
    # verified to be composed from, the stage's approach strings and its notes, the coverage-unit
    # approach the stage was built with among them
    run_ids = [class_runs[p].run_id for p in (1, 2, 3)]
    rf_sha = _sha(SCENARIOS["rf_df"])
    composed_from = [
        f"basis cur run {run_ids[0]}: scenario input rf_df",
        f"basis cur run {run_ids[0]}: table ilp",
    ]
    assert m["disclosures"] == {
        "discount_rates": DISCOUNT_RATES,
        "risk_adjustment": RISK_ADJUSTMENT,
        "notes": [NOTE_MONTHLY_STEP, NOTE_TIMING_CONVEXITY, NOTE_UNITS_DISCOUNTED],
        "confidence_level": 0.75,
        "para_120_curve_sha256": _sha(curves["df_cur"]),
        "para_120_curve_verified": {
            "source_sha256": rf_sha,
            "ilp_sha256": _sha(TABLES["ilp"]),
            "composed_from": composed_from,
        },
    }
    assert "coverage units" not in DISCOUNT_RATES
    # a stage built with undiscounted units discloses them as such (para 117(c))
    off = stages[(1, False)].manifest
    assert off["model"]["statics"] == {"discount_coverage_units": False}
    assert off["disclosures"]["notes"] == [
        NOTE_MONTHLY_STEP,
        NOTE_TIMING_CONVEXITY,
        NOTE_UNITS_UNDISCOUNTED,
    ]
    assert off["disclosures"]["discount_rates"] == DISCOUNT_RATES
    assert stages[(1, True)].manifest["disclosures"]["notes"][-1] == NOTE_UNITS_DISCOUNTED
    # both curves name their source, so the loader recomputed them with the registered helper
    assert m["provenance"] == [
        {
            "file": f"scenario input {name}",
            "sha256": _sha(curves[name]),
            "key": "rebased_from",
            "value": rf_sha,
            "helper": "halley.aggregation.inforce.recompute_rebased",
            "ulps": 0,  # written and recomputed on this machine: bit for bit
        }
        for name in ("p_l", "df_cur")
    ]
    # P_L = P: the locked-in curve was verified against the files the cur runs read
    assert m["locked_in"] == {
        gid: {
            "p_l": {
                "sha256": _sha(curves["p_l"]),
                "tau0": 0,
                "source_sha256": rf_sha,
                "ilp_sha256": _sha(TABLES["ilp"]),
                "composed_from": composed_from,
            }
        }
    }
    # the member rules that ran, on which runs (D21: a replay cannot run them again)
    held = [{"run_id": rid, "members": 1} for rid in run_ids]
    assert m["member_rules"] == [
        {"rule": G5_NAME, "basis": "cur", "reads": [], "runs": held},
        {"rule": G3_NAME, "basis": "cur", "reads": ["onerous"], "runs": held},
    ]
    assert [e["run_id"] for e in m["bases"]["cur"]] == run_ids
    assert m["previous"] is None
    series = m["inputs"]["series"]
    assert set(series) == {
        "cu_g",
        "fcf_g",
        "bel_g",
        "ra_g",
        "ann_out_g",
        "exp_out_g",
        "inv_exp_g",
        "premium_total",
    }
    assert series["fcf_g"]["variables"] == ["bel", "ra"]
    assert series["premium_total"]["members"] == [[gid, [1, 2, 3]]]
    # D21: the run reads back bit for bit from its verified directory
    loaded = load_run(d, verify=True)
    assert loaded.ids.tolist() == [gid]
    for name in ("csm", "lc", "cu_next", "p_g", "revenue", "ifie"):
        np.testing.assert_array_equal(loaded.var(name), res.var(name))
    assert loaded.run_id == res.run_id


class BadNotes(NewGroup):
    """Notes that depend on the build but give no strings."""

    disclosures: ClassVar[Any] = {**NewGroup.disclosures, "notes": lambda statics: 1}


class FailingNotes(NewGroup):
    """Notes that read a static the stage does not have."""

    disclosures: ClassVar[Any] = {
        **NewGroup.disclosures,
        "notes": lambda statics: (statics["no_such_static"],),
    }


def test_a_disclosure_of_the_build_is_checked_at_build() -> None:
    """A disclosure may be a function of the stage's statics (the coverage-unit note is); build()
    calls it with the statics it resolved, and one that raises or gives no string or tuple of
    strings is a BuildError at the stage class, before any run (the loader calls it again to
    record what it gives). An entry that is neither text nor a function is a BuildError too."""
    cases = (
        (BadNotes, r"notes gives 1, not a string or a tuple of strings"),
        (FailingNotes, r"notes raised KeyError: 'no_such_static'"),
    )
    for broken, message in cases:
        with pytest.raises(
            BuildError, match=rf"^{broken.__name__}\.disclosures: {message}"
        ) as info:
            build(broken)
        loc = info.value.loc
        assert loc is not None
        assert (loc.file, loc.line) == (__file__, inspect.getsourcelines(broken)[1])

    class Mistyped(NewGroup):
        disclosures: ClassVar[Any] = {"notes": 1}

    with pytest.raises(
        BuildError,
        match=r"Mistyped\.disclosures: notes is a string or a tuple of strings, or a function of "
        r"the stage's statics that returns one, not 1",
    ):
        build(Mistyped)


def test_the_stage_reads_factor_and_rate_units_of_the_curve(
    stages: dict[tuple[int, bool], RunResult], p_l: Path
) -> None:
    """j_L and d_L are the locked-in curve's forward rate and discount factor of each month."""
    res = stages[(1, True)]
    gid = GID[1]
    p = pq.read_table(p_l).column("rf_df").to_numpy()
    np.testing.assert_array_equal(res.lane("j_l", gid)[1:], (p[:-1] / p[1:]) - 1)
    np.testing.assert_array_equal(res.lane("d_l", gid)[1:], p[1:] / p[:-1])
    acc = res.lane("acc", gid)
    csm = res.lane("csm", gid)
    np.testing.assert_array_equal(acc[1:], res.lane("j_l", gid)[1:] * csm[:-1])
    assert res.manifest["outputs"]["variables"]["d_l"]["unit"] == "Factor"


def test_the_stage_reruns_and_replays_bit_for_bit_d19_d21(
    compiled_new_group: dict[bool, CompiledModel],
    stages: dict[tuple[int, bool], RunResult],
    class_runs: dict[int, RunResult],
    curves: dict[str, Path],
    runs_dir: Path,
    tmp_path: Path,
) -> None:
    """D19: the same inputs give the same run id and the same bits; D21: the run replays from
    its own directory alone (its copies of the groups, the series and both curves), carrying
    over the record of the member rules and of the locked-in curve it cannot check again."""
    res = stages[(3, True)]
    again = _stage(
        compiled_new_group[True],
        [class_runs[p] for p in EXAMPLES[3]],
        curves,
        tmp_path / "again",
        groups_file("OTHER"),
        runs_root=runs_dir,
    )
    assert again.run_id == res.run_id
    replay = replay_aggregation(compiled_new_group[True], res.run_dir, run_dir=tmp_path / "replay")
    assert replay.run_id == res.run_id
    assert replay.manifest["replayed_from"]["matches"] is True
    for block in ("member_rules", "locked_in", "disclosures"):
        assert replay.manifest[block] == res.manifest[block], block
    for other in (again, replay):
        for name in res.arrays:
            assert _bits_equal(other.var(name), res.var(name)), name


def _bits_equal(a: Any, b: Any) -> bool:
    x, y = np.asarray(a), np.asarray(b)
    if x.dtype.kind == "f":
        x, y = (np.ascontiguousarray(v, dtype=np.float64).view(np.uint64) for v in (x, y))
    return bool(np.array_equal(x, y))
