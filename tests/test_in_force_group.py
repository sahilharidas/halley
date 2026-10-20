# pyright: basic
"""The in-force group: one reporting period's measurement of a group of annuities in force
(design sections 9 and 11; in-force spec, sections 3 to 12 and 14).

For each variant (P, a profitable group; O, an onerous one) the eight projection runs of in-force
5.5 (REC, OLD, NEW and CUR, each for the level and the CPI-linked class, the two classes of a run
at the same horizon), the recognition stage ``NewGroup`` on REC and the in-force stage
``InForceGroup`` on OLD, NEW and CUR with the recognition stage as its previous run, every file
read from ``docs/spec/in-force-group`` or built from it by the cross-run helpers. Reconciled
against ``docs/spec/in_force_group_numbers.py`` (``build_variant``): every per-run, per-policy
column, the group totals, the recognition, the inputs the previous run supplies, the change in
locked-in fulfilment cash flows, N, the period close and the months after it, the analysis of
change, and the headline figures section 14 prints; K1 to K7, I11 to I13, M1 and M2 as run-time
invariants; the six remeasurement cases of 14.5; the shifted world Z1 and Z2 on constructed
inputs (12.3, 14.8); G6 and the load-time checks of 12.4; the locked-in files held to the group's
tau0; batch and horizon independence (D5); reruns, replays and the run directory (D19, D21); and
the committed reference workbooks, read through their cached values.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import shutil
import warnings
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
    Field,
    LoadError,
    Money,
    RunResult,
    RunSettings,
    build,
    load_run,
    run,
    var,
)
from halley.aggregation import (
    InForceGroup,
    composed_curve,
    expected_in_force,
    locked_in_cpi,
    locked_in_curve,
    rebase_expense_table,
    replay_aggregation,
    run_aggregation,
)
from halley.aggregation.csm import (
    AOC_LABELS,
    CLOSE_INVARIANTS,
    CLOSE_OUTPUTS,
    CPI_ACT_FIRST_MONTH,
    ROLL_INVARIANTS,
    ROLL_OUTPUTS,
    in_force_notes,
)
from halley.authoring.contracts import group_schema
from halley.products import annuity_data as data
from halley.runtime import within

IF_DIR = Path(__file__).resolve().parent.parent / "docs" / "spec" / "in-force-group"
V_REC, V = data.IF_RECOGNITION_DATE, data.IF_VALUATION_DATE
TAU = data.IF_TAU0
"""tau0 = tau = 12 (in-force C1)."""
T, T_REC = data.IF_T, data.IF_T_REC
VARIANTS = ("P", "O")
CLASSES = ("level", "cpi")
RUN_NAMES = ("REC", "OLD", "NEW", "CUR")
GID = {"P": "IA-2025-OTHER", "O": "IA-2025-ONEROUS"}

REC_TABLES = {
    "mort_base": "mort_base_v1",
    "mort_cif": "mort_cif",
    "expense": "expense_v1",
    "inv_expense": "inv_expense",
    "ilp": "ilp",
    "ra_param": "ra_param",
}
"""The recognition run's tables: version 1, priced at V_rec (in-force 5.5)."""
OLD_TABLES = {**REC_TABLES, "expense": "expense_old", "ilp": "ilp_zero"}
"""OLD: version 1 rebased to V prices (5.4); the ILP is inside P_L, so its table is 0 (C17)."""
NEW_TABLES = {**REC_TABLES, "mort_base": "mort_base_v2", "expense": "expense_v2", "ilp": "ilp_zero"}
"""NEW: version 2 on the locked-in financial inputs."""
CUR_TABLES = {**REC_TABLES, "mort_base": "mort_base_v2", "expense": "expense_v2"}
"""CUR: version 2 on the current curve, composed with the ILP."""
RUN_INPUTS = {
    "OLD": (OLD_TABLES, "p_l", "cpi_locked"),
    "NEW": (NEW_TABLES, "p_l", "cpi_locked"),
    "CUR": (CUR_TABLES, "discount_cur", "cpi_cur"),
}
"""Each run at V's tables, discount file and CPI file (in-force 5.5)."""

REL = 1e-12
"""The engine against the script: the same formulas in the same order, so they agree far inside
the spec's 1e-9 between implementations, with the spec's scales (in-force 1)."""
SPEC_REL = 1e-9
"""The spec's tolerance between the engine and the spreadsheet (in-force 1, C21)."""

STATE_COLS = {"pols_if", "ann_pp", "df", "bel", "v_be", "v_st", "ra"}
"""Defined at t = 0; every other column is a month quantity, undefined there (in-force 4)."""
COUNT_COLS = {"deaths", "pols_if"}
PER_POLICY_COLS = {"ann_pp", "exp_pp", "s_pp", "s_pp_st", "v_be", "v_st"}
MONEY_COLS = {"ann_out", "exp_out", "inv_exp", "bel", "ra", "cu"}
EXACT_COLS = {"age", "cal_year", "anniv"}
PER_LIFE_LAST = {"ann_pp": 0, "exp_pp": 0, "s_pp": 0, "s_pp_st": 0, "v_be": -1, "v_st": -1}
"""Per-life values are compared to t_end (amounts of month t) or t_end - 1 (values at t); beyond
that no life is in force and the spec excludes them from every comparison (in-force 6.11)."""

K_NAMES = tuple(inv.name for inv in CLOSE_INVARIANTS)
ROLL_NAMES = tuple(inv.name for inv in ROLL_INVARIANTS)


def _file(name: str) -> Path:
    return IF_DIR / f"{name}.parquet"


def _ten(x: float) -> str:
    return f"{float(x):.10g}"


def _printed(x: float, shown: str) -> bool:
    """``x`` printed as the spec prints ``shown``: with thousands separators and the same number
    of decimals (in-force 14, ten significant figures)."""
    decimals = len(shown.split(".")[1]) if "." in shown else 0
    return f"{float(x):,.{decimals}f}" == shown


def _agree(got: Any, want: Any, scale: Any, rel: float = REL) -> None:
    ok = within(got, want, scale, rel=rel)
    if not np.all(ok):
        i = int(np.flatnonzero(~np.asarray(ok).reshape(-1))[0])
        g = np.asarray(got, dtype=np.float64).reshape(-1)
        w = np.broadcast_to(np.asarray(want, dtype=np.float64), np.shape(got)).reshape(-1)
        raise AssertionError(f"element {i}: {g[i]!r} against {w[i]!r} (scale {scale!r})")


def _project(
    compiled: CompiledModel,
    points: Any,
    run_dir: Path,
    *,
    at: dt.date,
    tables: dict[str, Any],
    rf_df: Any,
    cpi: Any,
    horizon: int,
    **settings: Any,
) -> RunResult:
    """One projection run of the annuity: ``tables`` by the model's names, each a committed
    file's name or a table; ``rf_df`` and ``cpi`` likewise."""

    def source(x: Any) -> Any:
        return _file(x) if isinstance(x, str) else x

    return run(
        compiled,
        model_points=points,
        tables={name: source(x) for name, x in tables.items()},
        scenarios={"rf_df": source(rf_df), "cpi": source(cpi)},
        settings=RunSettings(valuation_date=at, horizon=horizon, **settings),
        run_dir=run_dir,
    )


def _sum_lanes(runs: list[RunResult], name: str) -> np.ndarray:
    """A variable summed over the policies of the runs in ascending policy id, sequentially,
    with plain floats (in-force 7.1)."""
    lanes = sorted((int(pid), r) for r in runs for pid in r.ids.tolist())
    total = np.zeros(runs[0].H + 1)
    for pid, r in lanes:
        total = total + np.asarray(r.lane(name, pid), dtype=np.float64)
    return total


class Chain:
    """One variant's measurement (in-force 5.5, 7.3, 9, 10), every run under one runs root: the
    recognition runs REC, the recognition stage (the previous run), the runs at V of the bases
    old (on ``expected_in_force``'s model points), new and cur, and the in-force stage."""

    def __init__(
        self,
        k: str,
        compiled_annuity: dict[str, CompiledModel],
        compiled_new_group: CompiledModel,
        compiled_stage: CompiledModel,
        root: Path,
    ) -> None:
        self.k = k
        self.gid = GID[k]
        self.root = root
        self.compiled = compiled_annuity
        self.compiled_stage = compiled_stage
        self.rec = {
            c: _project(
                compiled_annuity[c],
                _file(f"mp_{c}_{k}_rec"),
                root / f"rec-{k}-{c}",
                at=V_REC,
                tables=REC_TABLES,
                rf_df="discount_rec",
                cpi="cpi_rec",
                horizon=T_REC,
            )
            for c in CLASSES
        }
        _p_rec, p_l = locked_in_curve(_file("discount_rec"), _file("ilp"), 0, T_REC)
        self.recognised = run_aggregation(
            compiled_new_group,
            groups=_file(f"groups_{k}_rec"),
            bases={"cur": [self.rec["level"], self.rec["cpi"]]},
            scenarios={
                "p_l": p_l,
                "df_cur": composed_curve(_file("discount_rec"), _file("ilp"), T_REC),
            },
            settings=RunSettings(valuation_date=V_REC, runs_root=root),
            run_dir=root / f"recognised-{k}",
        )
        self.old = {c: self.at_v("OLD", c, root / f"old-{k}-{c}") for c in CLASSES}
        self.new = {c: self.at_v("NEW", c, root / f"new-{k}-{c}") for c in CLASSES}
        self.cur = {c: self.at_v("CUR", c, root / f"cur-{k}-{c}") for c in CLASSES}
        self.groups = data.with_previous_run(pq.read_table(_file(f"groups_{k}")), self.recognised)
        self.stage = self.in_force(root / f"in-force-{k}")

    def points(self, name: str, c: str) -> Any:
        """A run at V's model points: the expected in-force for OLD (in-force 3.4, C16), the
        actual survivors otherwise."""
        if name == "OLD":
            return expected_in_force(self.rec[c], TAU, _file(f"mp_{c}_{self.k}_v"))
        return _file(f"mp_{c}_{self.k}_v")

    def at_v(self, name: str, c: str, run_dir: Path, **kw: Any) -> RunResult:
        tables, rf_df, cpi = RUN_INPUTS[name]
        args: dict[str, Any] = {"tables": tables, "rf_df": rf_df, "cpi": cpi, "horizon": T}
        args.update(kw)
        points = args.pop("points", None)
        return _project(
            self.compiled[c],
            self.points(name, c) if points is None else points,
            run_dir,
            at=V,
            **args,
        )

    def runs(self, name: str) -> dict[str, RunResult]:
        return {"REC": self.rec, "OLD": self.old, "NEW": self.new, "CUR": self.cur}[name]

    def bases(self, **override: Any) -> dict[str, Any]:
        return {
            "old": [self.old["level"], self.old["cpi"]],
            "new": [self.new["level"], self.new["cpi"]],
            "cur": [self.cur["level"], self.cur["cpi"]],
            **override,
        }

    def in_force(
        self,
        run_dir: Path,
        *,
        compiled: CompiledModel | None = None,
        groups: Any = None,
        bases: Any = None,
        scenarios: dict[str, Any] | None = None,
        tables: dict[str, Any] | None = None,
        **settings: Any,
    ) -> RunResult:
        settings.setdefault("runs_root", self.root)
        return run_aggregation(
            self.compiled_stage if compiled is None else compiled,
            groups=self.groups if groups is None else groups,
            bases=self.bases() if bases is None else bases,
            scenarios={
                "p_l": _file("p_l"),
                "df_cur": _file("df_cur"),
                "p_rec": _file("p_rec"),
                **(scenarios or {}),
            },
            tables={"cpi_act": _file("cpi_act"), **(tables or {})},
            settings=RunSettings(valuation_date=V, **settings),
            run_dir=run_dir,
        )

    def lane(self, name: str, t: int | None = None) -> Any:
        """The in-force stage's ``name`` for the group, over time or at month ``t``."""
        values = self.stage.lane(name, self.gid)
        return values if t is None else float(values[t])


@pytest.fixture(scope="module")
def chains(
    compiled_annuity: dict[str, CompiledModel],
    compiled_new_group: dict[bool, CompiledModel],
    compiled_in_force_group: CompiledModel,
    tmp_path_factory: Any,
) -> dict[str, Chain]:
    root = Path(tmp_path_factory.mktemp("in-force")) / "runs"
    return {
        k: Chain(k, compiled_annuity, compiled_new_group[True], compiled_in_force_group, root)
        for k in VARIANTS
    }


def _p_g(v: dict[str, Any]) -> float:
    return float(v["recog"]["p_g"])


def _mp(v: dict[str, Any], pid: int) -> dict[str, Any]:
    return next(mp for mp in v["mps"] if mp["policy_id"] == pid)


def _cls(v: dict[str, Any], pid: int) -> str:
    return "cpi" if _mp(v, pid)["cls"] == "cpi" else "level"


def _scale(col: str, mp: dict[str, Any]) -> float:
    """The spec's scale of a policy's column (in-force 1): ``pols_if_init_rec x single_premium``
    for Money and CoverageUnits, ``single_premium`` per policy, ``pols_if_init_rec`` for counts,
    1 otherwise."""
    if col in MONEY_COLS:
        return float(mp["n_rec"] * mp["single_premium"])
    if col in PER_POLICY_COLS:
        return float(mp["single_premium"])
    if col in COUNT_COLS:
        return float(mp["n_rec"])
    return 1.0


# ---------------------------------------------------------------- per run, per policy (6, 7)


@pytest.mark.parametrize("k", VARIANTS)
def test_every_run_reconciles_per_policy_to_the_script(
    k: str, chains: dict[str, Chain], inforce_variants: dict[str, dict[str, Any]]
) -> None:
    """Every column of section 6.10 of every (run, policy), from t = 0 (states) or 1 (month
    quantities) to T, per-life columns to the policy's run-off only, against the script's
    ``build_variant(k)[run]["pols"][pid]`` with the spec's scales; integer and boolean columns
    exactly. The runs at V hold T = 489 and REC T = 501 (in-force 3.1, 13)."""
    v, chain = inforce_variants[k], chains[k]
    checked = 0
    for name in RUN_NAMES:
        script_run = v[name.lower()]
        assert script_run["T"] == (T_REC if name == "REC" else T)
        for pid in v["ids"]:
            res = chain.runs(name)[_cls(v, pid)]
            assert script_run["T"] == res.H, name
            want = script_run["pols"][pid]
            t_end = want["scalars"]["t_end"]
            for col in want:
                if col in ("scalars", "T", "a0", "m_c"):
                    continue
                if col == "anniv" and col not in res.arrays:
                    assert _cls(v, pid) == "level"
                    continue
                first = 0 if col in STATE_COLS else 1
                last = res.H if col not in PER_LIFE_LAST else t_end + PER_LIFE_LAST[col]
                got = res.lane(col, pid)[first : last + 1]
                expect = want[col][first : last + 1]
                if col in EXACT_COLS:
                    assert np.array_equal(got, np.array(expect)), (name, pid, col)
                    continue
                try:
                    _agree(got, np.array(expect, dtype=np.float64), _scale(col, _mp(v, pid)))
                except AssertionError as exc:
                    raise AssertionError(f"{name} policy {pid} {col}: {exc}") from None
                checked += 1
            for scalar, column in (("pva", "pva"), ("pve", "pve")):
                _agree(
                    res.lane(column, pid)[0],
                    want["scalars"][scalar],
                    _scale("bel", _mp(v, pid)),
                )
    assert checked > 150


@pytest.mark.parametrize("k", VARIANTS)
def test_the_group_totals_reconcile(
    k: str, chains: dict[str, Chain], inforce_variants: dict[str, dict[str, Any]]
) -> None:
    """Section 7.1 for every run: BEL_g, RA_g, FCF_g = BEL_g + RA_g (t = 0..T) and CU_g (t =
    1..T), summed in ascending policy id across both classes; REC as the recognition stage
    stored them, CUR as the in-force stage did, and OLD and NEW as the in-force stage read them
    (its copies of the series fcf_old and fcf_new), beside the runs' own sums."""
    v, chain = inforce_variants[k], chains[k]
    p_g = _p_g(v)
    rec = chain.recognised
    for name, values in (("bel_g", "bel"), ("ra_g", "ra"), ("fcf_g", "fcf")):
        _agree(rec.lane(name, chain.gid), v["rec"]["g"][values], p_g)
        _agree(chain.lane(name), v["cur"]["g"][values], p_g)
    _agree(rec.lane("cu_g", chain.gid)[1:], v["rec"]["g"]["cu"][1:], p_g)
    _agree(chain.lane("cu_g")[1:], v["cur"]["g"]["cu"][1:], p_g)
    series = chain.stage.manifest["inputs"]["series"]
    for name, run_name in (("fcf_old", "old"), ("fcf_new", "new")):
        table = pq.read_table(chain.stage.run_dir / series[name]["copy"])
        assert table.column("group_id").to_pylist() == [chain.gid] * (T + 1)
        assert table.column("t").to_pylist() == list(range(T + 1))
        _agree(np.asarray(table.column("value")), v[run_name]["g"]["fcf"], p_g)
        runs = list(chain.runs(run_name.upper()).values())
        for column in ("bel", "ra"):
            _agree(_sum_lanes(runs, column), v[run_name]["g"][column], p_g)
        _agree(_sum_lanes(runs, "cu")[1:], v[run_name]["g"]["cu"][1:], p_g)


@pytest.mark.parametrize("k", VARIANTS)
def test_the_recognition_and_the_previous_runs_supply(
    k: str, chains: dict[str, Chain], inforce_variants: dict[str, dict[str, Any]]
) -> None:
    """Section 7.3 (``recog``) and 3.4 (``inputs``): P_g, FCF_g^rec(0), CSM(0) and LC(0) of the
    recognition stage; the groups row at V takes csm_prev, lc_prev and prev_run_id from it, and
    the loader derives F_prev(0), F_prev(tau), LC^rec(tau), P_g and the month counts from it and
    the dates, never retyped; the OLD runs start from n_exp; cu_act is the datum of section 13."""
    v, chain = inforce_variants[k], chains[k]
    p_g = _p_g(v)
    rec = chain.recognised
    recog, inputs = v["recog"], v["inputs"]
    assert rec.lane("p_g", chain.gid)[0] == recog["p_g"]
    for name, want in (
        ("csm_0", recog["csm_0"]),
        ("lc_0", recog["lc_0"]),
        ("fcf_g", recog["fcf_0"]),
    ):
        _agree(rec.lane(name, chain.gid)[0], want, p_g)
    groups = chain.groups.to_pylist()[0]
    assert groups["prev_run_id"] == rec.run_id
    assert groups["csm_prev"] == rec.lane("csm", chain.gid)[0]
    assert groups["lc_prev"] == rec.lane("lc", chain.gid)[0]
    _agree(groups["csm_prev"], inputs["csm_prev"], p_g)
    _agree(groups["lc_prev"], inputs["lc_prev"], p_g)
    assert groups["cu_act"] == inputs["cu_act"] == data.IN_FORCE_CU_ACT[k]
    derived = pq.read_table(chain.stage.run_dir / "inputs" / "derived_groups.parquet")
    row = derived.to_pylist()[0]
    assert row["group_id"] == chain.gid
    assert (row["tau0"], row["tau"], row["i_acc"], row["i_rec"]) == (TAU, TAU, 0, TAU)
    assert row["p_g_prev"] == rec.lane("p_g", chain.gid)[0]
    assert row["f_prev_0"] == rec.lane("fcf_g", chain.gid)[0]
    assert row["f_prev_tau"] == rec.lane("fcf_g", chain.gid)[TAU]
    assert row["lc_prev_tau"] == rec.lane("lc", chain.gid)[TAU]
    _agree(row["f_prev_0"], inputs["f_prev_0"], p_g)
    _agree(row["f_prev_tau"], inputs["f_prev_tau"], p_g)
    _agree(row["lc_prev_tau"], v["rec_roll"]["lc"][TAU], p_g)
    cu_exp = sum(float(x) for x in rec.lane("cu_g", chain.gid)[1 : TAU + 1])
    _agree(cu_exp, inputs["cu_exp"], p_g)
    for c in CLASSES:
        pid = chain.old[c].ids.tolist()[0]
        n_exp = chain.old[c].lane("pols_if", pid)[0]
        assert n_exp == chain.rec[c].lane("pols_if", pid)[TAU]  # read at full precision (C16)
        _agree(n_exp, inputs["n_exp"][pid], _mp(v, pid)["n_rec"])


# ---------------------------------------------------------------- the close (8, 9, 10, 11)


CLOSE_STEPS = {
    "csm_acc": "csm_acc",
    "acc_per": "acc_per",
    "lc_alloc_per": "lc_alloc",
    "lca_per": "lca_per",
    "csm_adj": "csm_adj",
    "lc_adj": "lc_adj",
    "csm_fs": "csm_fs",
    "lc_fs": "lc_fs",
    "r_per": "r_per",
    "csm_0": "csm_0",
    "lc_0": "lc_0",
}
"""The stage's output for each step the script's ``close_period`` returns."""

ROLL_LINES = {
    "j_l": "j_l",
    "acc": "acc",
    "csm_pre": "csm_pre",
    "units": "u",
    "release": "r",
    "csm": "csm",
    "lc": "lc",
    "lc_alloc": "lca",
}
"""The stage's output for each line of the script's ``monthly_roll`` (months 1..T)."""


@pytest.mark.parametrize("k", VARIANTS)
def test_the_close_and_the_months_reconcile(
    k: str, chains: dict[str, Chain], inforce_variants: dict[str, dict[str, Any]]
) -> None:
    """Section 8 (delta_FCF_L), 9.5 (N(t) for t = 0..T, from the CUR units on P_L), the five
    steps of the close (9.2 to 9.6) and the months 1..T after it (10), against the script's
    ``delta``, ``n``, ``close`` and ``roll``, with P_g the scale of every amount; the closing
    balances open the months, and LRC(0) is FCF_g^cur(0) + CSM(0)."""
    v, chain = inforce_variants[k], chains[k]
    p_g = _p_g(v)
    _agree(chain.lane("delta_fcf_l", 0), v["delta"], p_g)
    _agree(chain.lane("cu_next"), v["n"], p_g)
    _agree(chain.lane("n_0", 0), v["n"][0], p_g)
    for name, key in CLOSE_STEPS.items():
        try:
            _agree(chain.lane(name, 0), v["close"][key], p_g)
        except AssertionError as exc:
            raise AssertionError(f"{name}: {exc}") from None
    for name, key in ROLL_LINES.items():
        scale = 1.0 if name == "j_l" else p_g
        _agree(chain.lane(name)[1:], v["roll"][key][1:], scale)
    assert chain.lane("csm", 0) == chain.lane("csm_0", 0)
    assert chain.lane("lc", 0) == chain.lane("lc_0", 0)
    assert chain.lane("lrc_0", 0) == chain.lane("fcf_g", 0) + chain.lane("csm_0", 0)
    u = chain.lane("u_per", 0)
    assert u == v["inputs"]["cu_act"] + chain.lane("n_0", 0)
    assert chain.lane("rev", 0) == min(chain.lane("lc_alloc_per", 0), -chain.lane("delta_fcf_l", 0))
    assert chain.lane("raw", 0) == chain.lane("csm_acc", 0) - chain.lane("delta_fcf_l", 0)
    assert chain.lane("p_g", 0) == v["recog"]["p_g"]


@pytest.mark.parametrize("k", VARIANTS)
def test_the_analysis_of_change(
    k: str, chains: dict[str, Chain], inforce_variants: dict[str, dict[str, Any]]
) -> None:
    """Section 11.2: A to E and the four differences; D - C is delta_FCF_L bit for bit, and the
    manifest labels each (C18)."""
    v, chain = inforce_variants[k], chains[k]
    p_g = _p_g(v)
    want = {
        "aoc_a": v["inputs"]["f_prev_0"],
        "aoc_b": v["inputs"]["f_prev_tau"],
        "aoc_c": v["old"]["g"]["fcf"][0],
        "aoc_d": v["new"]["g"]["fcf"][0],
        "aoc_e": v["cur"]["g"]["fcf"][0],
    }
    for name, value in want.items():
        _agree(chain.lane(name, 0), value, p_g)
    for name, (b, a) in {
        "aoc_ba": ("aoc_b", "aoc_a"),
        "aoc_cb": ("aoc_c", "aoc_b"),
        "aoc_dc": ("aoc_d", "aoc_c"),
        "aoc_ed": ("aoc_e", "aoc_d"),
    }.items():
        assert chain.lane(name, 0) == chain.lane(b, 0) - chain.lane(a, 0)
        _agree(chain.lane(name, 0), want[b] - want[a], p_g)
    assert chain.lane("aoc_dc", 0) == chain.lane("delta_fcf_l", 0)
    assert chain.lane("aoc_a", 0) == chain.recognised.lane("fcf_g", chain.gid)[0]
    assert chain.lane("aoc_b", 0) == chain.recognised.lane("fcf_g", chain.gid)[TAU]
    assert chain.stage.manifest["disclosures"]["analysis_of_change"] == list(AOC_LABELS)


# ------------------------------------------------------------- the headline figures (14)


HEADLINES: dict[str, dict[str, float]] = {
    "P": {
        "p_g": 19270869.0,
        "aoc_a": 18529681.237961616,
        "aoc_b": 17362216.290460568,
        "aoc_c": 17409845.031857315,
        "aoc_d": 17892403.667170994,
        "aoc_e": 17653966.754960112,
        "aoc_ba": -1167464.9475010484,
        "aoc_cb": 47628.74139674753,
        "aoc_dc": 482558.63531367853,
        "aoc_ed": -238436.9122108817,
        "delta_fcf_l": 482558.63531367853,
        "n_0": 17326847.31821964,
        "csm_acc": 764851.2894033649,
        "acc_per": 23663.527364981244,
        "lc_alloc_per": 0.0,
        "lca_per": 0.0,
        "csm_adj": 282292.65408968634,
        "csm_fs": -482558.63531367853,
        "lc_adj": 0.0,
        "lc_fs": 0.0,
        "r_per": 24775.48895089341,
        "csm_0": 257517.16513879295,
        "lc_0": 0.0,
        "lrc_0": 17911483.920098905,
    },
    "O": {
        "p_g": 18159088.0,
        "aoc_a": 18529681.237961616,
        "aoc_b": 17362216.290460568,
        "aoc_c": 17409845.031857315,
        "aoc_d": 17157546.77438024,
        "aoc_e": 16925168.8835751,
        "aoc_ba": -1167464.9475010484,
        "aoc_cb": 47628.74139674753,
        "aoc_dc": -252298.25747707486,
        "aoc_ed": -232377.89080514014,
        "delta_fcf_l": -252298.25747707486,
        "n_0": 16611371.824558262,
        "csm_acc": 0.0,
        "acc_per": 0.0,
        "lc_alloc_per": 347243.96338182886,
        "lca_per": 23349.27457978751,
        "rev": 252298.25747707486,
        "csm_adj": 0.0,
        "lc_adj": 94945.705904754,
        "lc_fs": -252298.25747707486,
        "r_per": 0.0,
        "csm_0": 0.0,
        "lc_0": 94945.705904754,
        "lrc_0": 16925168.8835751,
    },
}
"""Sections 14.5 and 14.7 at full precision: the period close and the analysis of change."""

RECOGNITION_HEADLINES: dict[str, dict[str, float]] = {
    "P": {"csm_0": 741187.7620383836, "lc_0": 0.0, "csm_12": 694305.3268248928, "lc_12": 0.0},
    "O": {"csm_0": 0.0, "lc_0": 370593.23796161637, "csm_12": 0.0, "lc_12": 347243.9633818289},
}
"""Section 14.2: the recognition stage's CSM(0), LC(0) and its own roll to month 12."""

POLICY_HEADLINES = {
    ("REC", "level"): (11816430.431717163, 323049.8871179734, 100.0),
    ("REC", "cpi"): (6207483.320785807, 182717.59834067416, 50.0),
    ("OLD", "level"): (10964227.859034946, 310058.5024982256, 97.35061873366202),
    ("OLD", "cpi"): (5955296.163348271, 180262.50697587224, 48.97700540091314),
}
"""Sections 14.2 and 14.3: bel(0), ra(0) and pols_if(0) of REC and OLD by class, full precision,
the same for both variants under their own policy ids."""

VARIANT_POLICY_HEADLINES = {
    "P": {
        ("NEW", "level"): (11212612.357670635, 312164.7497846746, 98.0),
        ("NEW", "cpi"): (6182874.449239052, 184752.11047663252, 50.0),
        ("CUR", "level"): (10947558.446912333, 296584.9987248728, 98.0),
        ("CUR", "cpi"): (6223351.849604842, 186471.45971806458, 50.0),
    },
    "O": {
        ("NEW", "level"): (10869369.122231744, 302608.68601575604, 95.0),
        ("NEW", "cpi"): (5811901.982284707, 173666.98384803458, 47.0),
        ("CUR", "level"): (10612429.10670072, 287505.8661108461, 95.0),
        ("CUR", "cpi"): (5849950.738628555, 175283.1721349807, 47.0),
    },
}
"""Section 14.3: NEW and CUR, by variant."""

PER_LIFE_HEADLINES = {
    "OLD": {
        "level": (112626.17538191048, 115811.14232440628),
        "cpi": (121593.71759460912, 125274.271468826),
    },
    "NEW": {
        "level": (114414.41181296569, 117599.76640260522),
        "cpi": (123657.48898478116, 127352.53119431381),
    },
    "CUR": {
        "level": (111709.78007053399, 114736.1576085429),
        "cpi": (124467.03699209694, 128196.46618645823),
    },
}
"""Section 14.3: v_be(0) and v_st(0) by run and class, the same under either variant."""

MONTHS_PRINTED = {
    "P": (
        ("cu_g", "cu_next", "j_l", "acc", "release", "csm", "fcf_g"),
        {
            0: (None, "17,326,847.32", None, None, None, "257,517.1651", "17,653,966.75"),
            1: (
                "139,374.4950",
                "17,234,647.87",
                "0.002744734072",
                "706.8161373",
                "2,071.474081",
                "256,152.5072",
                "17,560,295.75",
            ),
            2: (
                "139,077.9728",
                "17,142,795.80",
                "0.002762464704",
                "707.6122599",
                "2,067.112929",
                "254,793.0065",
                "17,467,032.55",
            ),
            3: (
                "138,782.0879",
                "17,051,284.38",
                "0.002779971042",
                "708.3171798",
                "2,062.761492",
                "253,438.5622",
                "17,374,169.31",
            ),
            4: (
                "138,486.8387",
                "16,960,106.96",
                "0.002797255581",
                "708.9324327",
                "2,058.419746",
                "252,089.0749",
                "17,281,698.32",
            ),
            5: (
                "138,192.2241",
                "16,869,257.00",
                "0.002814320792",
                "709.4595249",
                "2,054.087668",
                "250,744.4468",
                "17,189,612.01",
            ),
            6: (
                "137,898.2425",
                "16,778,728.06",
                "0.002831169117",
                "709.8999338",
                "2,049.765234",
                "249,404.5815",
                "17,097,902.89",
            ),
            7: (
                "137,604.8927",
                "16,688,513.81",
                "0.002847802974",
                "710.2551088",
                "2,045.452422",
                "248,069.3841",
                "17,006,563.64",
            ),
            8: (
                "137,312.1732",
                "16,598,608.00",
                "0.002864224758",
                "710.5264717",
                "2,041.149206",
                "246,738.7614",
                "16,915,587.01",
            ),
            9: (
                "137,020.0827",
                "16,509,004.48",
                "0.002880436836",
                "710.7154173",
                "2,036.855566",
                "245,412.6213",
                "16,824,753.02",
            ),
            10: (
                "136,701.8437",
                "16,419,724.05",
                "0.002896441553",
                "710.8233138",
                "2,032.173421",
                "244,091.2712",
                "16,734,295.27",
            ),
            11: (
                "136,384.3599",
                "16,330,760.71",
                "0.002912241228",
                "710.8526632",
                "2,027.502692",
                "242,774.6211",
                "16,644,206.81",
            ),
            12: (
                "136,067.6293",
                "16,242,108.52",
                "0.002927838156",
                "710.8047991",
                "2,022.843348",
                "241,462.5826",
                "16,554,384.24",
            ),
        },
    ),
    "O": (
        ("cu_g", "cu_next", "lc", "lc_alloc", "fcf_g"),
        {
            0: (None, "16,611,371.82", "94,945.70590", None, "16,925,168.88"),
            1: ("133,892.0253", "16,522,706.10", "94,440.37748", "505.3284212", "16,835,088.25"),
            2: ("133,606.7670", "16,434,373.64", "93,937.24369", "503.1337930", "16,745,398.84"),
            3: ("133,322.1224", "16,346,367.97", "93,436.26232", "500.9813720", "16,656,093.13"),
            4: ("133,038.0903", "16,258,682.71", "92,937.39183", "498.8704837", "16,567,163.70"),
            5: ("132,754.6693", "16,171,311.57", "92,440.59137", "496.8004669", "16,478,603.28"),
            6: ("132,471.8582", "16,084,248.38", "91,945.82069", "494.7706733", "16,390,404.69"),
            7: ("132,189.6555", "15,997,487.05", "91,453.04023", "492.7804671", "16,302,560.88"),
            8: ("131,908.0599", "15,911,021.57", "90,962.21100", "490.8292249", "16,215,064.90"),
            9: ("131,627.0703", "15,824,846.05", "90,472.13710", "490.0739058", "16,127,703.57"),
            10: ("131,320.7287", "15,738,980.70", "89,984.08922", "488.0478754", "16,040,703.39"),
            11: ("131,015.1152", "15,653,419.75", "89,498.02987", "486.0593485", "15,954,057.70"),
            12: ("130,710.2280", "15,568,157.50", "89,013.41298", "484.6168885", "15,867,669.14"),
        },
    ),
}
"""Section 14.6, months 0 to 12 after the close, as the spec prints them (ten significant
figures with thousands separators; blank where a month quantity is undefined at t = 0)."""

MONTHS_FULL = {
    "P": {
        ("release", 1): 2071.474080741102,
        ("csm", 1): 256152.50719531596,
        ("csm", 12): 241462.58257576753,
    },
    "O": {("lc", 1): 94440.37748355305, ("lc", 12): 89013.41298440409},
}
"""Section 14.6 at full precision."""


@pytest.mark.parametrize("k", VARIANTS)
def test_the_headline_figures_of_section_14(k: str, chains: dict[str, Chain]) -> None:
    """The figures section 14 prints, through the engine: the recognition (14.2), every (run,
    policy)'s bel(0), ra(0) and pols_if(0) and the per-life values (14.3), the CPI point's
    ann_pp(1) = 800 x E(1) (14.3), the close and the analysis of change at full precision (14.5,
    14.7), the months after the close at ten significant figures and in full (14.6), K7's
    relative difference (14.5), CSM(489) = 0 exactly; each within REL of the spec's scale."""
    chain = chains[k]
    p_g = chain.lane("p_g", 0)
    for name, value in HEADLINES[k].items():
        try:
            _agree(chain.lane(name, 0), value, p_g)
        except AssertionError as exc:
            raise AssertionError(f"{name}: {exc}") from None
    rec = chain.recognised
    want = RECOGNITION_HEADLINES[k]
    _agree(rec.lane("csm_0", chain.gid)[0], want["csm_0"], p_g)
    _agree(rec.lane("lc_0", chain.gid)[0], want["lc_0"], p_g)
    _agree(rec.lane("csm", chain.gid)[TAU], want["csm_12"], p_g)
    _agree(rec.lane("lc", chain.gid)[TAU], want["lc_12"], p_g)
    _agree(rec.lane("cu_next", chain.gid)[TAU], 16591150.9813373, p_g)
    expected_units = sum(float(x) for x in rec.lane("cu_g", chain.gid)[1 : TAU + 1])
    _agree(expected_units, 1661301.8187426073, p_g)
    for (name, c), (bel, ra, n) in {**POLICY_HEADLINES, **VARIANT_POLICY_HEADLINES[k]}.items():
        res = chain.runs(name)[c]
        pid = res.ids.tolist()[0]
        premium = _premium(chain, pid)
        n_rec = 100.0 if c == "level" else 50.0
        _agree(res.lane("bel", pid)[0], bel, n_rec * premium)
        _agree(res.lane("ra", pid)[0], ra, n_rec * premium)
        _agree(res.lane("pols_if", pid)[0], n, n_rec)
        if name in PER_LIFE_HEADLINES:
            v_be, v_st = PER_LIFE_HEADLINES[name][c]
            _agree(res.lane("v_be", pid)[0], v_be, premium)
            _agree(res.lane("v_st", pid)[0], v_st, premium)
        if c == "cpi" and name != "REC":
            assert res.lane("ann_pp", pid)[1] == 827.4899008308294  # 800 x E(1), I_act history
    columns, rows = MONTHS_PRINTED[k]
    for t, shown in rows.items():
        for name, text in zip(columns, shown, strict=True):
            if text is not None:
                assert _printed(chain.lane(name, t), text), (name, t, chain.lane(name, t), text)
    for (name, t), value in MONTHS_FULL[k].items():
        _agree(chain.lane(name, t), value, p_g)
    assert chain.lane("csm", T) == 0.0
    assert chain.lane("lc", T) == 0.0
    if k == "O":  # K7: LC_alloc against the recognition run's monthly LC(12), 1.7e-16 relative
        lc_alloc, lc_12 = chain.lane("lc_alloc_per", 0), rec.lane("lc", chain.gid)[TAU]
        assert abs(lc_alloc - lc_12) <= 1e-15 * lc_12
        assert _ten(lc_alloc) == "347243.9634"


def _premium(chain: Chain, pid: int) -> float:
    """A policy's single premium as the committed model point file holds it (the script's
    pricing rule, in-force 13), never retyped."""
    c = "level" if pid % 10 == 1 else "cpi"
    table = pq.read_table(_file(f"mp_{c}_{chain.k}_rec")).to_pylist()
    return float(next(r["single_premium"] for r in table if r["policy_id"] == pid))


@pytest.mark.parametrize("k", VARIANTS)
def test_every_invariant_holds_at_run_time(k: str, chains: dict[str, Chain]) -> None:
    """K1 to K7 (in-force 12.2) on the close and I11 to I13, M1 and M2 on the months after it,
    every one checked by the run and passed, in declaration order after the built-in H1 to H3;
    the ones the spec states exactly hold with a worst residual of exactly 0."""
    results = {r.name: r for r in chains[k].stage.invariants}
    names = [r.name for r in chains[k].stage.invariants]
    assert names[-len(K_NAMES) - len(ROLL_NAMES) :] == [*ROLL_NAMES, *K_NAMES]
    for name in (*ROLL_NAMES, *K_NAMES):
        assert results[name].passed, (name, results[name].detail)
    for name in K_NAMES:
        if name.startswith(("K4", "K6")):
            assert results[name].worst == 0.0, name
    stored = load_run(chains[k].stage.run_dir)
    assert [r["name"] for r in _invariants_json(stored)] == names


def _invariants_json(r: RunResult) -> list[dict[str, Any]]:
    return json.loads((r.run_dir / "invariants.json").read_text(encoding="utf-8"))


# ------------------------------------------------------ the remeasurement rules (9.4, 14.5)


class Remeasured(InForceGroup):
    """The in-force stage with its change in locked-in fulfilment cash flows taken from a group
    field, so that the remeasurement rules run on the cases of in-force 14.5 with the close's own
    CSM_acc and LC_alloc."""

    model_point = group_schema(
        Field("case_delta", "number", "Money"), previous=True, locked_in=True
    )

    @var
    def delta_fcf_l(self) -> Money:
        return self.group.case_delta


REMEASUREMENT_CASES = {
    "a": ("P", lambda csm, lc, dp: dp, ("282,292.6541", "0")),
    "b": ("P", lambda csm, lc, dp: 1.25 * csm, ("0", "191,212.8224")),
    "c": ("P", lambda csm, lc, dp: -dp, ("1,247,409.925", "0")),
    "d": ("O", lambda csm, lc, dp: dp, ("0", "829,802.5987")),
    "e": ("O", lambda csm, lc, dp: -0.5 * lc, ("0", "173,621.9817")),
    "f": ("O", lambda csm, lc, dp: -1.5 * lc, ("173,621.9817", "0")),
}
"""In-force 14.5's table: the case's variant (whose CSM_acc and LC_alloc it takes), its
delta_FCF_L as the script's ``main`` builds it from CSM_acc of P, LC_alloc of O and |delta_FCF_L|
of P, and the printed CSM_adj and LC_adj."""


@pytest.fixture(scope="module")
def compiled_remeasured() -> CompiledModel:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return build(Remeasured)


@pytest.mark.parametrize("case", sorted(REMEASUREMENT_CASES))
def test_every_branch_of_the_remeasurement_rules(
    case: str,
    chains: dict[str, Chain],
    compiled_remeasured: CompiledModel,
    inforce_oracle: ModuleType,
    tmp_path: Path,
) -> None:
    """Each case of in-force 14.5 through the stage's own where-forms: CSM_adj and LC_adj to ten
    significant figures as printed and as the script's ``remeasure`` gives them, K3's
    conservation exactly 0 and K4 (both non-negative, at most one positive), and every
    invariant of the run passed."""
    k, delta_of, (csm_shown, lc_shown) = REMEASUREMENT_CASES[case]
    chain = chains[k]
    csm_a, lc_a = chains["P"].lane("csm_acc", 0), chains["O"].lane("lc_alloc_per", 0)
    dp = abs(chains["P"].lane("delta_fcf_l", 0))
    delta = delta_of(csm_a, lc_a, dp)
    groups = chain.groups.append_column("case_delta", pa.array([delta], pa.float64()))
    res = chain.in_force(tmp_path / f"case-{case}", compiled=compiled_remeasured, groups=groups)
    gid = chain.gid

    def one(name: str) -> float:
        return float(res.lane(name, gid)[0])

    assert one("delta_fcf_l") == delta
    assert one("csm_acc") == chain.lane("csm_acc", 0)
    assert one("lc_alloc_per") == chain.lane("lc_alloc_per", 0)
    csm_adj, lc_adj = one("csm_adj"), one("lc_adj")
    assert _printed(csm_adj, csm_shown), (csm_adj, csm_shown)
    assert _printed(lc_adj, lc_shown), (lc_adj, lc_shown)
    want_csm, want_lc = inforce_oracle.remeasure(one("csm_acc"), one("lc_alloc_per"), delta)
    p_g = one("p_g")
    _agree(csm_adj, want_csm, p_g)
    _agree(lc_adj, want_lc, p_g)
    k3 = (csm_adj - lc_adj) - ((one("csm_acc") - one("lc_alloc_per")) - delta)
    assert k3 == 0.0
    assert csm_adj >= 0.0
    assert lc_adj >= 0.0
    assert min(csm_adj, lc_adj) == 0.0
    assert min(one("csm_0"), one("lc_0")) == 0.0
    failed = [r.name for r in res.invariants if not r.passed]
    assert not failed, failed


# ---------------------------------------------------------- the shifted world (12.3, 14.8)


def _shifted_cpi_act() -> Any:
    """``I_act(u) = I_rec(tau + u)``: actual CPI to V equal to the recognition path (Z1), as the
    committed ``cpi_act`` file is laid out."""
    months = list(range(CPI_ACT_FIRST_MONTH, 1))
    return pa.table(
        {
            "u": pa.array(months, pa.int32()),
            "cpi": pa.array([data.cpi_path(TAU + u) for u in months], pa.float64()),
        }
    ).replace_schema_metadata(
        {
            b"halley.version": b"in-force-z1",
            b"halley.basis": b"in-force-z1",
            b"halley.estimated_rows": b"0",
        }
    )


@pytest.mark.parametrize("k", VARIANTS)
def test_the_shifted_world_z1_and_z2(
    k: str,
    chains: dict[str, Chain],
    inforce_oracle: ModuleType,
    inforce_variants: dict[str, dict[str, Any]],
    tmp_path: Path,
) -> None:
    """Z1 (in-force 12.3): with actual CPI equal to the recognition path, the previous tables,
    the expected in-force and the instalments at recognition, a run at V on P_L and the spliced
    path is the recognition run shifted by tau: pols_if, bel, ra (t = 0..T), cu and ann_pp (t =
    1..T) equal REC's at tau + t within 1e-12 x bel^rec(tau); a stage on such runs has FCF_g(0)
    = F_prev(tau) and N(0) = N^rec(tau) within 1e-12 relative. Z2: a NEW run with the OLD run's
    inputs gives delta_FCF_L = 0 exactly. And the named simplification of 14.8: the close with
    the expected units as cu_act gives the script's period-form CSM(0) and LC(0)."""
    chain, v = chains[k], inforce_variants[k]
    cpi_act = _shifted_cpi_act()
    expense = rebase_expense_table(_file("expense_v1"), cpi_act, TAU, data.price_base(V))
    cpi_locked = locked_in_cpi(cpi_act, _file("cpi_rec"), TAU, T)
    tables = {**OLD_TABLES, "expense": expense}
    runs: dict[str, dict[str, RunResult]] = {}
    for name in ("OLD", "NEW", "CUR"):  # one set of inputs, three runs (Z2: NEW = OLD's inputs)
        runs[name] = {
            c: chain.at_v(
                "OLD",
                c,
                tmp_path / f"{name}-z-{c}",
                tables=tables,
                cpi=cpi_locked,
                label=f"{name}-Z",
            )
            for c in CLASSES
        }
    for c in CLASSES:
        z, rec = runs["OLD"][c], chain.rec[c]
        pid = z.ids.tolist()[0]
        scale = float(rec.lane("bel", pid)[TAU])
        for col in ("pols_if", "bel", "ra"):
            _agree(z.lane(col, pid), rec.lane(col, pid)[TAU : TAU + T + 1], scale)
        for col in ("cu", "ann_pp"):
            _agree(z.lane(col, pid)[1:], rec.lane(col, pid)[TAU + 1 : TAU + T + 1], scale)
        shifted = rec.lane("ann_pp", pid)[TAU + 1 : TAU + T + 1]  # and per life, relative
        assert (np.abs(z.lane("ann_pp", pid)[1:] - shifted) <= 1e-12 * np.abs(shifted)).all()
    groups = chain.groups.set_column(
        chain.groups.column_names.index("cu_act"),
        "cu_act",
        pa.array([v["inputs"]["cu_exp"]], pa.float64()),
    )
    stage = chain.in_force(
        tmp_path / "in-force-z",
        groups=groups,
        bases={name.lower(): [runs[name]["level"], runs[name]["cpi"]] for name in runs},
        scenarios={"df_cur": composed_curve(_file("p_l"), _file("ilp_zero"), T)},
        tables={"cpi_act": cpi_act},
    )
    gid = chain.gid
    assert stage.lane("delta_fcf_l", gid)[0] == 0.0  # Z2, exactly
    rec = chain.recognised
    fcf_z, f_prev_tau = stage.lane("fcf_g", gid)[0], rec.lane("fcf_g", gid)[TAU]
    assert abs(fcf_z - f_prev_tau) <= 1e-12 * f_prev_tau
    n_z, n_rec = stage.lane("cu_next", gid)[0], rec.lane("cu_next", gid)[TAU]
    assert abs(n_z - n_rec) <= 1e-12 * n_rec
    z = inforce_oracle.shifted_world(v)
    p_g = _p_g(v)
    _agree(stage.lane("csm_0", gid)[0], z["close_z"]["csm_0"], p_g)
    _agree(stage.lane("lc_0", gid)[0], z["close_z"]["lc_0"], p_g)
    if k == "P":
        assert _ten(stage.lane("csm_0", gid)[0]) == "695236.0518"  # 14.8, against 694305.3268
    else:
        _agree(stage.lane("lc_0", gid)[0], 347243.96338182886, p_g)
    assert all(r.passed for r in stage.invariants)


# ------------------------------------------------------------------- G6 and 12.4 at load


class EachFcf(InForceGroup):
    """An in-force stage that reads FCF_g per scenario, so a J > 1 cur basis reaches it other
    than through a mean. Its df_cur is a fixed file, not the composed curve of its cur runs: no
    composed curve is one scenario of two (the helpers refuse a J = 2 discount file)."""

    discloses_current_curve: ClassVar[bool] = False
    series = tuple(
        dataclasses.replace(s, scenarios="each") if s.name == "fcf_g" else s
        for s in InForceGroup.series
    )


def test_g6_a_two_scenario_basis_reaches_the_stage_only_through_a_mean(
    chains: dict[str, Chain], tmp_path: Path
) -> None:
    """G6 (in-force 12.4; design section 9): the stage reads mean series, so it runs at S = 1,
    and a two-scenario cur basis read per scenario is a LoadError naming the check,
    require_deterministic, before anything is written."""
    chain = chains["P"]
    two_rf = data.discount_table(
        lambda s, t: data.rf_df(t, data.RF_LEVEL_CUR + 0.005 * s), t_max=T, n_scenarios=2
    )
    two_cpi = data.cpi_table(
        lambda s, u: (
            data.cpi_current(u) if u <= 0 else data.cpi_actual(0) * (1.03 + 0.01 * s) ** (u / 12)
        ),
        u_lo=data.IF_CPI_U_LO,
        u_hi=T,
        n_scenarios=2,
    )
    cur = [
        chain.at_v("CUR", c, tmp_path / f"cur-two-{c}", rf_df=two_rf, cpi=two_cpi) for c in CLASSES
    ]
    assert cur[0].n_scenarios == 2
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compiled = build(EachFcf)
    with pytest.raises(
        LoadError,
        match=r"the stage reads the mean series \w+, so it runs at S = 1; series fcf_g has J = 2 "
        r"scenarios.*\(require_deterministic\)",
    ):
        chain.in_force(
            tmp_path / "each",
            compiled=compiled,
            bases=chain.bases(cur=cur),
            # its source, discount_cur, is read by no run here: a fixed file names none
            scenarios={"df_cur": _without(pq.read_table(_file("df_cur")), "rebased_from")},
        )
    assert not (tmp_path / "each" / "manifest.json").exists()


def test_the_load_time_checks_of_section_12_4(chains: dict[str, Chain], tmp_path: Path) -> None:
    """In-force 12.4: a locked-in discount file (p_rec) not covering [0, tau0 + T] is a
    LoadError naming the rule; a CPI file starting at u = -13 is a LoadError of the run that
    reads it, since month 1's escalation reads I(-14)."""
    chain = chains["P"]
    p_rec, _p_l = locked_in_curve(_file("discount_rec"), _file("ilp"), TAU, T - 9)
    assert pq.read_table(_file("p_rec")).num_rows == TAU + T + 1
    assert p_rec.num_rows == TAU + T - 8  # months 0..492, not 0..501
    with pytest.raises(
        LoadError,
        match=r"input rule the locked-in curve covers \[0, tau0 \+ H\] fails \(p_rec, the "
        r"composed curve at recognition, reaches the run-off",
    ):
        chain.in_force(tmp_path / "short", scenarios={"p_rec": p_rec})
    assert not (tmp_path / "short" / "manifest.json").exists()
    act = pq.read_table(_file("cpi_act"))
    late = act.slice(-13 - CPI_ACT_FIRST_MONTH)  # u = -13..0
    assert late.column("u").to_pylist()[0] == -13
    cpi = locked_in_cpi(late, _file("cpi_rec"), TAU, T)
    assert cpi.column("u").to_pylist()[0] == -13
    for c in CLASSES:
        with pytest.raises(
            LoadError,
            match=r"scenario input cpi.*reads months -14 to 489 but the file covers -13 to 489",
        ):
            chain.at_v("OLD", c, tmp_path / f"old-13-{c}", cpi=cpi)


# --------------------------------------------- the locked-in basis, held to the group's tau0


def _without(table: Any, key: str) -> Any:
    meta = dict(table.schema.metadata)
    del meta[f"halley.{key}".encode()]
    return table.replace_schema_metadata(meta)


def test_the_locked_in_files_are_rebuilt_at_the_groups_tau0(
    chains: dict[str, Chain], tmp_path: Path
) -> None:
    """The stage rebuilds its locked-in files (``rebuilds_locked_in``): a curve re-based at
    another month than the group's tau0 = 12, which rebuilds bit for bit from its own metadata,
    is a LoadError naming the file; so is one that names no source, and a locked-in CPI path of
    the old and new runs spliced at another month. The manifest records the tau0 held."""
    chain = chains["P"]
    discount, ilp = _file("discount_rec"), _file("ilp")
    _p_rec, p_l_0 = locked_in_curve(discount, ilp, 0, T)
    with pytest.raises(
        LoadError, match=r"scenario input p_l.*halley\.tau0 is 0; the stage rebuilds .* tau0 = 12"
    ):
        chain.in_force(tmp_path / "p_l-0", scenarios={"p_l": p_l_0})
    p_rec_6, _p_l = locked_in_curve(discount, ilp, 6, T)
    with pytest.raises(LoadError, match=r"scenario input p_rec.*halley\.tau0 is 6"):
        chain.in_force(tmp_path / "p_rec-6", scenarios={"p_rec": p_rec_6})
    bare = _without(pq.read_table(_file("p_l")), "rebased_from")
    with pytest.raises(
        LoadError,
        match=r"scenario input p_l.*carries no metadata halley\.rebased_from; the stage rebuilds",
    ):
        chain.in_force(tmp_path / "bare", scenarios={"p_l": bare})
    spliced = locked_in_cpi(_file("cpi_act"), _file("cpi_rec"), 6, T)
    old = [chain.at_v("OLD", c, tmp_path / f"old-6-{c}", cpi=spliced) for c in CLASSES]
    new = [chain.at_v("NEW", c, tmp_path / f"new-6-{c}", cpi=spliced) for c in CLASSES]
    with pytest.raises(
        LoadError,
        match=rf"basis old run {old[0].run_id}: scenario input cpi.*halley\.tau0 is 6; .*"
        r"locked_in_cpi",
    ):
        chain.in_force(tmp_path / "cpi-6", bases=chain.bases(old=old, new=new))
    for name in ("p_l-0", "p_rec-6", "bare", "cpi-6"):
        assert not (tmp_path / name / "manifest.json").exists()
    locked = chain.stage.manifest["locked_in"][chain.gid]
    assert locked["tau0"] == TAU
    files = {e["file"].split(": ")[-1] for es in locked["verified_by"].values() for e in es}
    assert {
        "scenario input p_l",
        "scenario input p_rec",
        "scenario input rf_df",
        "scenario input cpi",
    } <= files


class Unlocked(InForceGroup):
    """An in-force stage whose groups lock in no basis."""

    model_point = group_schema(previous=True)


class Stringly(InForceGroup):
    rebuilds_locked_in: ClassVar[Any] = "yes"


def test_the_stage_is_checked_at_build_and_refused_by_run(
    chains: dict[str, Chain], compiled_in_force_group: CompiledModel, tmp_path: Path
) -> None:
    """``rebuilds_locked_in`` is a bool, and a stage that sets it declares the locked-in basis
    (a BuildError at the class otherwise); ``run()``, which reads no previous run and checks no
    hash, refuses the in-force stage before anything is written."""
    with pytest.raises(BuildError, match=r"Unlocked is rebuilds_locked_in.*discount_rec_sha256"):
        build(Unlocked)
    with pytest.raises(
        BuildError, match=r"Stringly\.rebuilds_locked_in is True or False, not 'yes'"
    ):
        build(Stringly)
    with pytest.raises(LoadError, match=r"reads a previous run .*run_aggregation"):
        run(
            compiled_in_force_group,
            model_points=chains["P"].groups,
            settings=RunSettings(valuation_date=V),
            run_dir=tmp_path / "run",
        )
    assert not (tmp_path / "run").exists()


# ------------------------------------------------------------------- D5: batches and T


def _agree_runs(
    a: RunResult, pa_id: Any, b: RunResult, pb_id: Any, scales: dict[str, float]
) -> None:
    for name in sorted(a.arrays):
        if name not in b.arrays:
            continue
        unit = a.manifest["outputs"]["variables"].get(name, {}).get("unit")
        x = np.asarray(a.lane(name, pa_id), dtype=np.float64)
        y = np.asarray(b.lane(name, pb_id), dtype=np.float64)
        try:
            _agree(x, y, scales.get(str(unit), 1.0))
        except AssertionError as exc:
            raise AssertionError(f"{name}: {exc}") from None


def test_the_in_force_runs_do_not_depend_on_their_batch(
    chains: dict[str, Chain], tmp_path: Path
) -> None:
    """D5: a run at V holding both variants' policies of a class (11 and 21, or 12 and 22), in
    one batch and in two, gives each policy what the variant's one-policy run gives, within
    1e-12 of the spec's scales."""
    for name in ("OLD", "NEW", "CUR"):
        for c in CLASSES:
            points = [chains[k].points(name, c) for k in VARIANTS]
            tables = [pq.read_table(p) if isinstance(p, Path) else p for p in points]
            both = pa.concat_tables([t.replace_schema_metadata(None) for t in tables])
            for size in (1, 2):
                res = chains["P"].at_v(
                    name, c, tmp_path / f"{name}-{c}-{size}", points=both, batch_size=size
                )
                assert res.ids.tolist() == [
                    pid for t in tables for pid in t.column("policy_id").to_pylist()
                ]
                for k in VARIANTS:
                    one = chains[k].runs(name)[c]
                    pid = one.ids.tolist()[0]
                    n = 100.0 if c == "level" else 50.0
                    premium = _premium(chains[k], pid)
                    scales = {"Money": n * premium, "MoneyPerPolicy": premium, "Count": n}
                    _agree_runs(res, pid, one, pid, scales)


def _stage_agrees(
    a: RunResult, b: RunResult, gid: str, p_g: float, last: int | None = None
) -> None:
    """Every stored variable of two stage runs agrees within 1e-12 of the spec's scale, P_g for
    amounts and coverage units, 1 otherwise, over months ``0..last``."""
    variables = a.manifest["outputs"]["variables"]
    assert set(variables) == set(b.manifest["outputs"]["variables"])
    for name, info in variables.items():
        scale = p_g if info["unit"] in ("Money", "CoverageUnits") else 1.0
        x, y = a.lane(name, gid), b.lane(name, gid)
        if info["time"] and last is not None:
            x, y = x[: last + 1], y[: last + 1]
        try:
            _agree(x, y, scale)
        except AssertionError as exc:
            raise AssertionError(f"{name}: {exc}") from None


def test_aggregation_is_independent_of_base_batching_d5(
    compiled_annuity: dict[str, CompiledModel],
    compiled_new_group: dict[bool, CompiledModel],
    compiled_in_force_group: CompiledModel,
    tmp_path: Path,
) -> None:
    """D5: variant P with a second level life (policy 13) in its group, so that a class run
    holds two policies; its runs at V made in batches of one (with two workers) and of two give
    in-force stages that agree within 1e-12 of the spec's scales, every stored variable."""
    root = tmp_path / "runs"
    rows = [dict(r) for r in data.IN_FORCE_MODEL_POINTS["P"]]
    extra = {
        **rows[0],
        "policy_id": 13,
        "date_of_birth": dt.date(1946, 5, 20),
        "pols_if_at_v": 97.0,
    }
    level = [rows[0], extra]
    at_rec = data.model_points_table(level, valuation_date=V_REC)
    at_v = data.model_points_table(
        [{**r, "pols_if_init": r["pols_if_at_v"]} for r in level], valuation_date=V
    )
    rec = {
        "level": _project(
            compiled_annuity["level"],
            at_rec,
            root / "rec-level",
            at=V_REC,
            tables=REC_TABLES,
            rf_df="discount_rec",
            cpi="cpi_rec",
            horizon=T_REC,
        ),
        "cpi": _project(
            compiled_annuity["cpi"],
            _file("mp_cpi_P_rec"),
            root / "rec-cpi",
            at=V_REC,
            tables=REC_TABLES,
            rf_df="discount_rec",
            cpi="cpi_rec",
            horizon=T_REC,
        ),
    }
    _p_rec, p_l = locked_in_curve(_file("discount_rec"), _file("ilp"), 0, T_REC)
    recognised = run_aggregation(
        compiled_new_group[True],
        groups=_file("groups_P_rec"),
        bases={"cur": [rec["level"], rec["cpi"]]},
        scenarios={
            "p_l": p_l,
            "df_cur": composed_curve(_file("discount_rec"), _file("ilp"), T_REC),
        },
        settings=RunSettings(valuation_date=V_REC, runs_root=root),
        run_dir=root / "recognised",
    )
    groups = data.with_previous_run(pq.read_table(_file("groups_P")), recognised)
    points = {
        "OLD": {
            "level": expected_in_force(rec["level"], TAU, at_v),
            "cpi": expected_in_force(rec["cpi"], TAU, _file("mp_cpi_P_v")),
        },
        "NEW": {"level": at_v, "cpi": _file("mp_cpi_P_v")},
        "CUR": {"level": at_v, "cpi": _file("mp_cpi_P_v")},
    }
    stages = []
    for size, settings in ((1, {"batch_size": 1, "workers": 2}), (2, {"batch_size": 2})):
        bases: dict[str, list[RunResult]] = {}
        for name, (tables, rf_df, cpi) in RUN_INPUTS.items():
            bases[name.lower()] = [
                _project(
                    compiled_annuity[c],
                    points[name][c],
                    tmp_path / f"{name}-{c}-{size}",
                    at=V,
                    tables=tables,
                    rf_df=rf_df,
                    cpi=cpi,
                    horizon=T,
                    **settings,
                )
                for c in CLASSES
            ]
        assert bases["cur"][0].ids.tolist() == [11, 13]
        assert bases["cur"][0].manifest["settings"]["batch_size"] == size
        stages.append(
            run_aggregation(
                compiled_in_force_group,
                groups=groups,
                bases=bases,
                scenarios={"p_l": _file("p_l"), "df_cur": _file("df_cur"), "p_rec": _file("p_rec")},
                tables={"cpi_act": _file("cpi_act")},
                settings=RunSettings(valuation_date=V, runs_root=root),
                run_dir=tmp_path / f"in-force-{size}",
            )
        )
    gid = GID["P"]
    assert all(r.passed for s in stages for r in s.invariants)
    _stage_agrees(stages[0], stages[1], gid, float(stages[1].lane("p_g", gid)[0]))


def test_stage_results_do_not_depend_on_t(
    chains: dict[str, Chain],
    compiled_annuity: dict[str, CompiledModel],
    compiled_new_group: dict[bool, CompiledModel],
    tmp_path: Path,
) -> None:
    """D5: the chain on files that reach eleven months further, its runs at V to T' = 500, gives
    an in-force stage (H = 500) that agrees with the reference chain's (H = 489) within 1e-12 of
    the spec's scales on months 0..489, with every amount and unit exactly 0 after the run-off;
    the same bases with the stage's horizon at 489 agree too."""
    long_t = T + 11
    root = tmp_path / "runs"
    tables = data.in_force_tables()
    scenarios = {
        **data.in_force_scenarios(),
        "discount_rec": data.discount_table(t_max=TAU + long_t + 8),
        "cpi_rec": data.cpi_table(u_hi=TAU + long_t + 8),
        "discount_cur": data.discount_table(
            lambda s, t: data.rf_df(t, data.RF_LEVEL_CUR), t_max=long_t
        ),
        "cpi_cur": data.cpi_table(
            lambda s, u: data.cpi_current(u), u_lo=data.IF_CPI_U_LO, u_hi=long_t
        ),
    }
    p_rec, p_l = locked_in_curve(scenarios["discount_rec"], tables["ilp"], TAU, long_t)
    cpi_locked = locked_in_cpi(_file("cpi_act"), scenarios["cpi_rec"], TAU, long_t)
    df_cur = composed_curve(scenarios["discount_cur"], tables["ilp"], long_t)
    p = chains["P"]
    rec = {
        c: _project(
            compiled_annuity[c],
            _file(f"mp_{c}_P_rec"),
            root / f"rec-{c}",
            at=V_REC,
            tables=REC_TABLES,
            rf_df=scenarios["discount_rec"],
            cpi=scenarios["cpi_rec"],
            horizon=T_REC,
        )
        for c in CLASSES
    }
    _p, p_l_rec = locked_in_curve(scenarios["discount_rec"], tables["ilp"], 0, T_REC)
    recognised = run_aggregation(
        compiled_new_group[True],
        groups=_file("groups_P_rec"),
        bases={"cur": [rec["level"], rec["cpi"]]},
        scenarios={
            "p_l": p_l_rec,
            "df_cur": composed_curve(scenarios["discount_rec"], tables["ilp"], T_REC),
        },
        settings=RunSettings(valuation_date=V_REC, runs_root=root),
        run_dir=root / "recognised",
    )
    assert recognised.lane("csm_0", p.gid)[0] == p.recognised.lane("csm_0", p.gid)[0]
    groups = data.with_previous_run(data.in_force_groups("P", tables, scenarios), recognised)
    files = {
        "OLD": (OLD_TABLES, p_l, cpi_locked),
        "NEW": (NEW_TABLES, p_l, cpi_locked),
        "CUR": (CUR_TABLES, scenarios["discount_cur"], scenarios["cpi_cur"]),
    }
    bases: dict[str, list[RunResult]] = {}
    for name, (run_tables, rf_df, cpi) in files.items():
        bases[name.lower()] = [
            _project(
                compiled_annuity[c],
                expected_in_force(rec[c], TAU, _file(f"mp_{c}_P_v"))
                if name == "OLD"
                else _file(f"mp_{c}_P_v"),
                root / f"{name}-{c}",
                at=V,
                tables=run_tables,
                rf_df=rf_df,
                cpi=cpi,
                horizon=long_t,
            )
            for c in CLASSES
        ]
    stages = {}
    for horizon in (None, T):
        stages[horizon] = run_aggregation(
            p.compiled_stage,
            groups=groups,
            bases=bases,
            scenarios={"p_l": p_l, "df_cur": df_cur, "p_rec": p_rec},
            tables={"cpi_act": _file("cpi_act")},
            settings=RunSettings(valuation_date=V, runs_root=root, horizon=horizon),
            run_dir=tmp_path / f"in-force-{horizon}",
        )
    assert long_t == stages[None].H
    assert stages[T].H == T
    p_g = p.lane("p_g", 0)
    for res in stages.values():
        assert all(r.passed for r in res.invariants)
        _stage_agrees(res, p.stage, p.gid, p_g, last=T)
    variables = stages[None].manifest["outputs"]["variables"]
    for name, info in variables.items():
        if info["time"] and info["unit"] in ("Money", "CoverageUnits"):
            assert (stages[None].lane(name, p.gid)[T + 1 :] == 0.0).all(), name


# -------------------------------------------------------------- D19, D21: the run directory


def test_the_in_force_run_reruns_and_replays_bit_identically(
    chains: dict[str, Chain], compiled_in_force_group: CompiledModel, tmp_path: Path
) -> None:
    """D19: the stage rerun on the same inputs has the same run id and every stored variable bit
    for bit. D21: replayed from its own directory alone (its copies of the groups file, the
    derived fields, the series, the curves and cpi_act) it matches, and ``load_run`` reads back
    what it stored."""
    chain = chains["O"]
    again = chain.in_force(tmp_path / "again")
    assert again.run_id == chain.stage.run_id
    for name in chain.stage.arrays:
        a = np.ascontiguousarray(chain.stage.arrays[name], dtype=np.float64).view(np.uint64)
        b = np.ascontiguousarray(again.arrays[name], dtype=np.float64).view(np.uint64)
        assert np.array_equal(a, b), name
    moved = tmp_path / "moved"
    shutil.copytree(chain.stage.run_dir, moved)
    replay = replay_aggregation(compiled_in_force_group, moved, run_dir=tmp_path / "replay")
    assert replay.run_id == chain.stage.run_id
    assert replay.manifest["replayed_from"]["matches"] is True
    loaded = load_run(chain.stage.run_dir)
    for name in (*ROLL_OUTPUTS, *CLOSE_OUTPUTS):
        assert np.array_equal(loaded.lane(name, chain.gid), chain.lane(name)), name


def test_the_run_directory_records_the_period(chains: dict[str, Chain]) -> None:
    """The in-force run's directory (design section 9, "Outputs"; in-force 11): outputs keyed on
    group_id with every line of the roll and every step of the close; its previous run, its
    bases, the locked-in hashes with the files that verified them, the provenance of every file
    built by a helper (the OLD model points, the rebased expense table, the locked-in curve and
    CPI path, p_l, p_rec, df_cur), cpi_act copied with its estimate flag (C9), and the
    disclosures with the in-force notes (C11, C12, C17)."""
    chain = chains["P"]
    m, d = chain.stage.manifest, chain.stage.run_dir
    assert m["outputs"]["kind"] == "aggregation"
    assert m["outputs"]["id_field"] == "group_id"
    scalars = pq.read_table(d / "outputs" / "scalars.parquet")
    monthly = pq.read_table(d / "outputs" / "monthly.parquet")
    for name in CLOSE_OUTPUTS:
        assert name in scalars.column_names, name
    for name in ROLL_OUTPUTS:
        assert name in monthly.column_names, name
    assert m["previous"]["run_id"] == chain.recognised.run_id
    assert [e["run_id"] for e in m["bases"]["old"]] == [chain.old[c].run_id for c in CLASSES]
    locked = m["locked_in"][chain.gid]
    groups = chain.groups.to_pylist()[0]
    for name in ("discount_rec_sha256", "ilp_sha256", "cpi_rec_sha256"):
        assert locked[name] == groups[name]
    proven = {(e["file"].split(": ")[-1], e["key"]) for e in m["provenance"]}
    assert {
        ("model points", "derived_from"),
        ("table expense", "rebased_from"),
        ("scenario input cpi", "rebased_from"),
        ("scenario input rf_df", "rebased_from"),
        ("scenario input p_l", "rebased_from"),
        ("scenario input p_rec", "rebased_from"),
        ("scenario input df_cur", "rebased_from"),
    } <= proven
    cpi_act = m["inputs"]["tables"]["cpi_act"]
    assert cpi_act["metadata"]["estimated_rows"] == "0"
    assert (d / cpi_act["copy"]).exists()
    statics = dict(chain.compiled_stage.graph.statics)
    assert m["disclosures"]["notes"] == list(in_force_notes(statics))
    assert m["disclosures"]["confidence_level"] == 0.75
    assert m["disclosures"]["para_120_curve_sha256"] == m["inputs"]["scenarios"]["df_cur"]["sha256"]


# -------------------------------------------------------- the reference workbooks (11.3)


GROUP_CLOSE_ROWS = {
    "csm_prev": lambda c: c.groups.to_pylist()[0]["csm_prev"],
    "lc_prev": lambda c: c.groups.to_pylist()[0]["lc_prev"],
    "cu_act": lambda c: c.groups.to_pylist()[0]["cu_act"],
    "step 1: CSM_acc": lambda c: c.lane("csm_acc", 0),
    "acc_per": lambda c: c.lane("acc_per", 0),
    "F_prev(0)": lambda c: c.recognised.lane("fcf_g", c.gid)[0],
    "F_prev(tau)": lambda c: c.recognised.lane("fcf_g", c.gid)[TAU],
    "step 2: LC_alloc": lambda c: c.lane("lc_alloc_per", 0),
    "lca_per": lambda c: c.lane("lca_per", 0),
    "FCF_g^L_new(0)": lambda c: c.lane("aoc_d", 0),
    "FCF_g^L_old(0)": lambda c: c.lane("aoc_c", 0),
    "section 8: delta_FCF_L": lambda c: c.lane("delta_fcf_l", 0),
    "rev = min(LC_alloc, -delta_FCF_L)": lambda c: c.lane("rev", 0),
    "CSM_acc - delta_FCF_L": lambda c: c.lane("raw", 0),
    "LC_adj": lambda c: c.lane("lc_adj", 0),
    "CSM_adj": lambda c: c.lane("csm_adj", 0),
    "csm_fs": lambda c: c.lane("csm_fs", 0),
    "lc_fs": lambda c: c.lane("lc_fs", 0),
    "step 4: N(0)": lambda c: c.lane("n_0", 0),
    "R_per": lambda c: c.lane("r_per", 0),
    "step 5: CSM(0)": lambda c: c.lane("csm_0", 0),
    "LC(0) = LC_adj": lambda c: c.lane("lc_0", 0),
    "FCF_g^cur(0)": lambda c: c.lane("aoc_e", 0),
    "LRC(0)": lambda c: c.lane("lrc_0", 0),
    "Opening CSM": lambda c: c.groups.to_pylist()[0]["csm_prev"],
    "Interest accretion": lambda c: c.lane("acc_per", 0),
    "Changes for future service": lambda c: c.lane("csm_fs", 0),
    "Currency differences": lambda c: 0.0,
    "Release": lambda c: -c.lane("r_per", 0),
    "Closing CSM": lambda c: c.lane("csm_0", 0),
    "Opening LC": lambda c: c.groups.to_pylist()[0]["lc_prev"],
    "Systematic allocation": lambda c: -c.lane("lca_per", 0),
    "Losses on remeasurement": lambda c: max(c.lane("lc_fs", 0), 0.0),
    "Reversals of losses": lambda c: min(c.lane("lc_fs", 0), 0.0),
    "Closing LC": lambda c: c.lane("lc_0", 0),
    "A = ": lambda c: c.lane("aoc_a", 0),
    "B = ": lambda c: c.lane("aoc_b", 0),
    "C = ": lambda c: c.lane("aoc_c", 0),
    "D = ": lambda c: c.lane("aoc_d", 0),
    "E = ": lambda c: c.lane("aoc_e", 0),
    "B - A": lambda c: c.lane("aoc_ba", 0),
    "C - B": lambda c: c.lane("aoc_cb", 0),
    "D - C": lambda c: c.lane("aoc_dc", 0),
    "E - D": lambda c: c.lane("aoc_ed", 0),
    "P_g = ": lambda c: c.recognised.lane("p_g", c.gid)[0],
    "FCF_g^rec(0)": lambda c: c.recognised.lane("fcf_g", c.gid)[0],
    "gap = ": lambda c: c.recognised.lane("gap", c.gid)[0],
    "CSM(0) = where(gap": lambda c: c.recognised.lane("csm_0", c.gid)[0],
    "LC(0) = max(0": lambda c: c.recognised.lane("lc_0", c.gid)[0],
    "diagnostic: expected units": lambda c: sum(
        float(x) for x in c.recognised.lane("cu_g", c.gid)[1 : TAU + 1]
    ),
}
"""The rows of the group sheet's recognition block (7.3), close block (9) and roll-forward and
analysis-of-change tables (11.1, 11.2), by the start of their label, and the engine's value."""

EXACT_ROWS = {
    "tau0": lambda c: TAU,
    "tau": lambda c: TAU,
    "P_rec(tau0 - tau)": lambda c: pq.read_table(_file("p_rec")).column("rf_df").to_pylist()[0],
    "P_rec(tau0)": lambda c: pq.read_table(_file("p_rec")).column("rf_df").to_pylist()[TAU],
}
"""Rows that are inputs the engine reads unchanged: equal exactly."""

LAYOUT_ROWS = ("step 3: 9.4 rule applied", "G3 diagnostic", "last month with CU_g > 0")
"""Rows the spreadsheet holds as layout or diagnostics, not reconciled (in-force 11.3)."""

HEADINGS = ("7.3 ", "9 ", "11.1 ", "11.2 ", "Helpers")
"""The block headings of the group sheet's quantity column."""

MONTHS_COLUMNS = {
    "CU_g(t) (CUR run, section 10)": ("cu_g", 1),
    "F(t) = FCF_g^cur(t)": ("fcf_g", 0),
    "d_L(t)": ("d_l", 1),
    "j_L(t)": ("j_l", 1),
    "N(t)": ("cu_next", 0),
    "Acc(t)": ("acc", 1),
    "CSM_pre(t)": ("csm_pre", 1),
    "U(t)": ("units", 1),
    "R(t)": ("release", 1),
    "CSM(t)": ("csm", 0),
    "LC(t)": ("lc", 0),
    "LCA(t)": ("lc_alloc", 1),
}
"""The group sheet's months block (in-force 10): by header, the stage's variable and the first
month defined (month quantities are blank in row 0, in-force 4)."""

POLICY_COLUMNS = {"bel": 0, "ra": 0, "pols_if": 0, "cu": 1}
"""The (run, policy) sheets' columns reconciled, and the first month each is defined."""


def _workbook(k: str, folder: Path = IF_DIR) -> Any:
    """The committed workbook of variant ``k``, read through its cached values; a missing
    workbook fails the reconciliation rather than skipping it."""
    path = folder / f"in-force-group-{k}.xlsx"
    if not path.exists():
        pytest.fail(f"the reference workbook {path} is missing (in-force 11.3)")
    openpyxl = pytest.importorskip("openpyxl")
    return openpyxl.load_workbook(path, data_only=True, read_only=True)


def _cached(value: Any, where: str) -> float:
    if value is None:
        pytest.skip(
            f"{where} holds no cached value; recalculate the workbook (docs/spec/recalc_excel.py)"
        )
    return float(value)


@pytest.mark.parametrize("k", VARIANTS)
def test_the_engine_reconciles_to_the_reference_workbook(k: str, chains: dict[str, Chain]) -> None:
    """In-force 1 and 11.3: the group sheet's recognition, close and roll-forward blocks and its
    months block (t = 0..T), and every (run, policy) sheet's bel, ra, pols_if and cu columns,
    against the engine within 1e-9 x max(|engine|, |workbook|, scale), with the spec's scales;
    the cached values are the spreadsheet's (read with data_only=True)."""
    chain = chains[k]
    wb = _workbook(k)
    p_g = chain.lane("p_g", 0)
    rows = list(wb["group"].iter_rows(values_only=True))
    header = list(rows[0])
    labels = header.index("quantity")
    seen: set[str] = set()
    for i, row in enumerate(rows[1:], start=2):
        label = row[labels]
        if not isinstance(label, str) or label.startswith(HEADINGS + LAYOUT_ROWS):
            continue
        where = f"group!AJ{i} {label!r}"
        exact = EXACT_ROWS.get(label)
        if exact is not None:
            assert _cached(row[labels + 1], where) == exact(chain), where
            seen.add(label)
            continue
        getter = next(
            (f for prefix, f in GROUP_CLOSE_ROWS.items() if label.startswith(prefix)), None
        )
        assert getter is not None, f"{where} has no engine counterpart"
        try:
            _agree(getter(chain), _cached(row[labels + 1], where), p_g, rel=SPEC_REL)
        except AssertionError as exc:
            raise AssertionError(f"{where}: {exc}") from None
        seen.add(label)
    assert len(seen) >= 50
    for title, (name, first) in MONTHS_COLUMNS.items():
        j = header.index(title)
        engine = chain.lane(name)
        scale = 1.0 if name in ("d_l", "j_l") else p_g
        for t in range(first, T + 1):
            cell = _cached(rows[t + 1][j], f"group {title} t = {t}")
            try:
                _agree(engine[t], cell, scale, rel=SPEC_REL)
            except AssertionError as exc:
                raise AssertionError(f"group {title} t = {t}: {exc}") from None
    for name in RUN_NAMES:
        for c in CLASSES:
            res = chain.runs(name)[c]
            pid = res.ids.tolist()[0]
            sheet = f"{name}-{pid}"
            prows = list(wb[sheet].iter_rows(values_only=True))
            pheader = list(prows[0])
            n_rec = 100.0 if c == "level" else 50.0
            premium = _premium(chain, pid)
            for col, first in POLICY_COLUMNS.items():
                j = pheader.index(col)
                scale = n_rec if col == "pols_if" else n_rec * premium
                lane = res.lane(col, pid)
                assert len(prows) == res.H + 2, sheet
                for t in range(first, res.H + 1):
                    cell = _cached(prows[t + 1][j], f"{sheet} {col} t = {t}")
                    try:
                        _agree(lane[t], cell, scale, rel=SPEC_REL)
                    except AssertionError as exc:
                        raise AssertionError(f"{sheet} {col} t = {t}: {exc}") from None


def test_a_missing_workbook_fails_and_an_uncached_cell_skips(tmp_path: Path) -> None:
    """The reconciliation's two outcomes besides a pass: a workbook that is not there fails it,
    and a compared cell without a cached value (a workbook saved without recalculation) skips
    it with the cell named."""
    with pytest.raises(pytest.fail.Exception, match=r"in-force-group-P\.xlsx is missing"):
        _workbook("P", tmp_path)
    with pytest.raises(pytest.skip.Exception, match=r"group!AJ19 holds no cached value"):
        _cached(None, "group!AJ19")
    assert _cached(1.5, "group!AJ19") == 1.5
