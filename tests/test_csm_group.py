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
example 3. Then the member rules G3 and G5 at load, I11 to I14 as run-time invariants (and
failing on broken subclasses), twelve monthly steps against one step with undiscounted units,
and the aggregation run's directory: keyed on ``group_id``, with its disclosures and notes.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
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
    Int,
    LoadError,
    Money,
    RunError,
    RunResult,
    RunSettings,
    build,
    load_run,
    run,
    var,
    where,
)
from halley.aggregation import (
    MemberRule,
    NewGroup,
    group_series,
    locked_in_curve,
    replay_aggregation,
    run_aggregation,
)
from halley.aggregation.csm import (
    DISCOUNT_RATES,
    NOTE_MONTHLY_STEP,
    NOTE_TIMING_CONVEXITY,
    RISK_ADJUSTMENT,
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

INVARIANTS = (
    "I11 CSM(t) = CSM(t-1) + Acc(t) - R(t)",
    "I11 CSM(t) >= 0",
    "I11 CSM(t) = 0 from the last month with CU_g > 0",
    "I12 R(t) / CU_g(t) = CSM(t) / N(t)",
    "I12 R(t+1) / CU_g(t+1) = [R(t) / CU_g(t)] a_L(t+1) [CU_g(t+1) + d(t+1) N(t+1)] / U(t+1)",
    "I13 LC(t) / F(t) = LC(0) / F(0)",
    "I13 LC(t) = 0 where F(t) = 0",
    "I14 G3 for the group: CSM(0) = 0 if ONEROUS, else LC(0) = 0",
)

G3_NAME = "G3 a contract is onerous exactly in an ONEROUS group"
G5_NAME = "G5 every contract is new at V"


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
        settings=RunSettings(valuation_date=VALUATION_DATE, horizon=T_EXAMPLE, **kw),
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
    p_l: Any,
    run_dir: Path,
    groups: Any,
    **settings: Any,
) -> RunResult:
    return run_aggregation(
        compiled,
        groups=groups,
        bases={"cur": runs},
        scenarios={"p_l": p_l},
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
def p_l(runs_dir: Path) -> Path:
    """The locked-in curve of a group new at V: the composed curve of the runs it reads,
    re-based to V with tau0 = 0 (P_L = P), written by the section 10 helper."""
    _p_rec, curve = locked_in_curve(SCENARIOS["rf_df"], TABLES["ilp"], 0, T_EXAMPLE)
    path = runs_dir / "p_l.parquet"
    pq.write_table(curve, path)
    return path


@pytest.fixture(scope="module")
def stages(
    compiled_new_group: dict[bool, CompiledModel],
    class_runs: dict[int, RunResult],
    p_l: Path,
    runs_dir: Path,
) -> dict[tuple[int, bool], RunResult]:
    """Each example's recognition stage with discounted (True) and undiscounted coverage units."""
    out: dict[tuple[int, bool], RunResult] = {}
    for example, pids in EXAMPLES.items():
        for switch in (True, False):
            out[(example, switch)] = _stage(
                compiled_new_group[switch],
                [class_runs[pid] for pid in pids],
                p_l,
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


# ---------------------------------------------------------------------------- invariants


def test_i11_to_i14_hold_as_run_time_invariants(stages: dict[tuple[int, bool], RunResult]) -> None:
    for key, res in stages.items():
        results = {r.name: r for r in res.invariants}
        assert [r.name for r in res.invariants][-len(INVARIANTS) :] == list(INVARIANTS), key
        assert all(r.passed for r in res.invariants), key
        on_disk = json.loads((res.run_dir / "invariants.json").read_text(encoding="utf-8"))
        assert [r["name"] for r in on_disk][-len(INVARIANTS) :] == list(INVARIANTS)
        # the exact ones are exact
        for name in INVARIANTS[1:3] + INVARIANTS[6:]:
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


def test_the_invariants_fail_on_broken_stages(
    compiled_annuity: dict[str, CompiledModel],
    class_runs: dict[int, RunResult],
    p_l: Path,
    tmp_path: Path,
) -> None:
    gid = GID[1]
    i12 = r"invariant I12 R\(t\) / CU_g\(t\) = CSM\(t\) / N\(t\) failed"
    with pytest.raises(RunError, match=rf"{i12}: group {gid}, t = 1: lhs = ") as info:
        _stage(build(Unequal), [class_runs[1]], p_l, tmp_path / "unequal", groups_file("OTHER"))
    assert info.value.loc is not None
    assert info.value.loc.file.endswith("csm.py")
    i13 = r"invariant I13 LC\(t\) / F\(t\) = LC\(0\) / F\(0\) failed"
    with pytest.raises(RunError, match=rf"{i13}: group {GID[2]}, t = 1: lhs = "):
        _stage(build(Fading), [class_runs[4]], p_l, tmp_path / "fading", groups_file("ONEROUS"))
    # G3 broken at load: policy 1 at a premium of 175,000 in an OTHER group, checked only by I14
    onerous = _project(
        compiled_annuity["level"], _one_policy(1, single_premium=175_000.0), tmp_path / "level"
    )
    i14 = r"invariant I14 G3 for the group: CSM\(0\) = 0 if ONEROUS, else LC\(0\) = 0 failed"
    with pytest.raises(RunError, match=rf"{i14}: group {gid}: lhs = 6840\.39757"):
        _stage(build(Unchecked), [onerous], p_l, tmp_path / "unchecked", groups_file("OTHER"))


# -------------------------------------------------------------------------- member rules


def test_g3_names_the_contract_on_the_wrong_side_of_its_group(
    compiled_annuity: dict[str, CompiledModel],
    compiled_new_group: dict[bool, CompiledModel],
    p_l: Path,
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
        _stage(stage, [level], p_l, tmp_path / "stage", groups_file("OTHER"))
    message = str(info.value)
    assert message.startswith(
        f"member rule {G3_NAME} fails for 1 of 1 members of basis cur run {level.run_id}, first "
        f"at basis cur run {level.run_id}: model points ("
    )
    assert f": row 0, policy 1, which the run read from {path}" in message
    assert "an onerous contract outside an ONEROUS group" in message
    assert not (tmp_path / "stage").exists()

    path = tmp_path / "mp_level_onerous.parquet"
    pq.write_table(_one_policy(1, profit_bucket="ONEROUS"), path)
    profitable = _project(compiled_annuity["level"], path, tmp_path / "level-2")
    assert not profitable.lane("onerous", 1)[0]
    with pytest.raises(LoadError, match=f"member rule {G3_NAME} fails for 1 of 1 members"):
        _stage(stage, [profitable], p_l, tmp_path / "stage-2", groups_file("ONEROUS"))


def test_g5_rejects_a_contract_in_force_before_v(
    compiled_annuity: dict[str, CompiledModel],
    compiled_new_group: dict[bool, CompiledModel],
    p_l: Path,
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
            p_l,
            tmp_path / "stage",
            groups_file("OTHER", recognition=issued, cohort=2024),
        )
    message = str(info.value)
    assert message.startswith(
        f"member rule {G5_NAME} fails for 1 of 1 members of basis cur run {in_force.run_id}"
    )
    assert ": row 0, policy 1 (issue_date = V" in message


def test_g3_reads_a_deterministic_run(
    compiled_annuity: dict[str, CompiledModel],
    compiled_new_group: dict[bool, CompiledModel],
    p_l: Path,
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
        _stage(compiled_new_group[True], [level], p_l, tmp_path / "stage", groups_file("OTHER"))


def _always(view: Any) -> Any:
    return np.ones(1, dtype=bool)


def test_member_rules_are_checked_at_build_and_at_load(
    class_runs: dict[int, RunResult], p_l: Path, tmp_path: Path
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
        "output": (
            MemberRule("output", lambda v: v.output("q_mth")[0, :, 0] > 0),
            "did not store q_mth, which a member rule reads",
        ),
    }
    for name, (rule, message) in cases.items():
        with pytest.raises(LoadError, match=f"^member rule {name}.*{message}"):
            _stage(stage_with(rule), [class_runs[1]], p_l, tmp_path / name, groups_file("OTHER"))


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


def test_the_locked_in_curve_starts_at_one(
    compiled_new_group: dict[bool, CompiledModel],
    class_runs: dict[int, RunResult],
    p_l: Path,
    tmp_path: Path,
) -> None:
    """The stage's p_l is re-based to V: p_l(0) = 1 is a rule of the input. A curve that names
    no source is not recomputed, and is checked by the rule alone."""
    table = pq.read_table(p_l)
    values = table.column("rf_df").to_pylist()
    shifted = pa.table(
        {
            "scenario": table.column("scenario"),
            "t": table.column("t"),
            "rf_df": pa.array([0.999 * x for x in values], pa.float64()),
        }
    )
    with pytest.raises(LoadError, match=r"scenario input p_l.*: rule p_l\(0\) = 1 fails"):
        _stage(
            compiled_new_group[True], [class_runs[1]], shifted, tmp_path / "s", groups_file("OTHER")
        )


# ------------------------------------------------------------------ the run directory


def test_the_aggregation_run_is_keyed_on_the_group_and_discloses(
    stages: dict[tuple[int, bool], RunResult], class_runs: dict[int, RunResult], p_l: Path
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
    for name in ("csm", "lc", "release", "cu_next", "cu_g", "fcf_g", "lrc"):
        assert name in monthly.column_names, name
    for name in ("p_g", "csm_0", "lc_0", "lrc_0"):
        assert name in scalars.column_names, name
    variables = m["outputs"]["variables"]
    assert variables["csm"] == {"time": True, "unit": "Money", "start": 0, "role": None}
    assert variables["release"] == {"time": True, "unit": "Money", "start": 1, "role": None}
    assert variables["cu_g"]["unit"] == "CoverageUnits"
    assert variables["j_l"]["unit"] == "Rate[Monthly]"
    assert variables["p_g"] == {"time": False, "unit": "Money", "start": 0, "role": None}
    # the disclosures: alpha of the cur basis, the stage's approach strings and its notes
    assert m["disclosures"] == {
        "discount_rates": DISCOUNT_RATES,
        "risk_adjustment": RISK_ADJUSTMENT,
        "notes": [NOTE_MONTHLY_STEP, NOTE_TIMING_CONVEXITY],
        "confidence_level": 0.75,
        "para_120_curve_sha256": None,
    }
    # p_l names its source, so the loader recomputed it with the registered helper (design 10)
    assert m["provenance"] == [
        {
            "file": "scenario input p_l",
            "sha256": _sha(p_l),
            "key": "rebased_from",
            "value": _sha(SCENARIOS["rf_df"]),
            "helper": "halley.aggregation.inforce.recompute_rebased",
        }
    ]
    assert [e["run_id"] for e in m["bases"]["cur"]] == [class_runs[p].run_id for p in (1, 2, 3)]
    assert m["previous"] is None
    series = m["inputs"]["series"]
    assert set(series) == {"cu_g", "fcf_g", "bel_g", "ra_g", "premium_total"}
    assert series["fcf_g"]["variables"] == ["bel", "ra"]
    assert series["premium_total"]["members"] == [[gid, [1, 2, 3]]]
    # D21: the run reads back bit for bit from its verified directory
    loaded = load_run(d, verify=True)
    assert loaded.ids.tolist() == [gid]
    for name in ("csm", "lc", "cu_next", "p_g"):
        np.testing.assert_array_equal(loaded.var(name), res.var(name))
    assert loaded.run_id == res.run_id


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
    p_l: Path,
    runs_dir: Path,
    tmp_path: Path,
) -> None:
    """D19: the same inputs give the same run id and the same bits; D21: the run replays from
    its own directory alone (its copies of the groups, the series and p_l)."""
    res = stages[(3, True)]
    again = _stage(
        compiled_new_group[True],
        [class_runs[p] for p in EXAMPLES[3]],
        p_l,
        tmp_path / "again",
        groups_file("OTHER"),
        runs_root=runs_dir,
    )
    assert again.run_id == res.run_id
    replay = replay_aggregation(compiled_new_group[True], res.run_dir, run_dir=tmp_path / "replay")
    assert replay.run_id == res.run_id
    assert replay.manifest["replayed_from"]["matches"] is True
    for other in (again, replay):
        for name in res.arrays:
            assert _bits_equal(other.var(name), res.var(name)), name


def _bits_equal(a: Any, b: Any) -> bool:
    x, y = np.asarray(a), np.asarray(b)
    if x.dtype.kind == "f":
        x, y = (np.ascontiguousarray(v, dtype=np.float64).view(np.uint64) for v in (x, y))
    return bool(np.array_equal(x, y))
