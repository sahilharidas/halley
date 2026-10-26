# pyright: basic
"""The cross-run helpers of milestone M9 and the in-force group's reference inputs (design
section 10; in-force spec sections 3, 5, 7.3, 13 and 14).

Every file ``docs/spec/in_force_group_inputs.py`` writes is the table
``halley.products.annuity_data`` builds, and every value in it equals
``docs/spec/in_force_group_numbers.py`` bit for bit: the curves (``p_rec``, ``p_lock``,
``p_cur``), the CPI paths (``i_act``, ``i_lock``, ``i_cur``), the rebased expense
(``maint_old = 5.160000000000001``), the tables of versions 1 and 2 and the model points. The
recognition runs on these inputs give the script's ``n_exp`` through ``expected_in_force``
(with the actual instalment at V, and a model point with no survivors kept), and the
recognition stage gives its CSM(0) and LC(0). Then G4 (``check_membership``), the helpers'
refusals, and the recomputations the stage loader runs for every file that names its source.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from halley import CompiledModel, LoadError, RunResult, RunSettings, run
from halley.aggregation import (
    check_membership,
    composed_curve,
    expected_in_force,
    locked_in_cpi,
    locked_in_curve,
    rebase_expense_table,
    run_aggregation,
)
from halley.aggregation import stage as stage_module
from halley.aggregation.inforce import recompute_derived, recompute_rebased
from halley.aggregation.stage import ProvenanceCheck
from halley.products import annuity_data as data
from halley.runtime import within
from halley.runtime.loader import ReadFile, TableMatch, file_difference, file_match, read_source
from halley.runtime.tolerance import LIBM_ULPS
from libm import assert_generated, same_values

IF_DIR = Path(__file__).resolve().parent.parent / "docs" / "spec" / "in-force-group"
V_REC, V = data.IF_RECOGNITION_DATE, data.IF_VALUATION_DATE
TAU = data.IF_TAU0

REC_TABLES = {
    "mort_base": "mort_base_v1",
    "mort_cif": "mort_cif",
    "expense": "expense_v1",
    "inv_expense": "inv_expense",
    "ilp": "ilp",
    "ra_param": "ra_param",
}
"""The recognition run's tables (version 1), by the model's table names."""
CUR_TABLES = {**REC_TABLES, "mort_base": "mort_base_v2", "expense": "expense_v2"}
"""The current-basis run's tables at V (version 2)."""


def _same(got: Any, want: Any) -> bool:
    """Equal bit for bit on the laptop that wrote the files, where the script and the files
    share one C library; within 1e-12 relative on another machine (:mod:`libm`)."""
    return same_values(got, want)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file(name: str) -> Path:
    return IF_DIR / f"{name}.parquet"


def _column(name: str, column: str) -> list[Any]:
    return pq.read_table(_file(name)).column(column).to_pylist()


@pytest.fixture(scope="module")
def files() -> dict[str, Any]:
    return data.in_force_files()


@pytest.fixture(scope="module")
def variants(inforce_oracle: ModuleType) -> dict[str, dict[str, Any]]:
    """The script's two variants (sections 5 to 10 and 14)."""
    return {k: inforce_oracle.build_variant(k) for k in ("P", "O")}


def _project(
    compiled: CompiledModel,
    points: Any,
    run_dir: Path,
    *,
    at: dt.date,
    tables: dict[str, str],
    rf_df: Any,
    cpi: Any,
    horizon: int,
) -> RunResult:
    return run(
        compiled,
        model_points=points,
        tables={name: _file(file) for name, file in tables.items()},
        scenarios={"rf_df": rf_df, "cpi": cpi},
        settings=RunSettings(valuation_date=at, horizon=horizon),
        run_dir=run_dir,
    )


def _rec_run(compiled: dict[str, CompiledModel], cls: str, points: Any, run_dir: Path) -> RunResult:
    """A recognition run (REC, in-force 5.5): version 1 at V_rec on the recognition curve and
    CPI path, H = 501."""
    return _project(
        compiled[cls],
        points,
        run_dir,
        at=V_REC,
        tables=REC_TABLES,
        rf_df=_file("discount_rec"),
        cpi=_file("cpi_rec"),
        horizon=data.IF_T_REC,
    )


@pytest.fixture(scope="module")
def rec_runs(
    compiled_annuity: dict[str, CompiledModel], tmp_path_factory: Any
) -> dict[tuple[str, str], RunResult]:
    """The recognition runs of both variants, one per class, from the committed files."""
    root = Path(tmp_path_factory.mktemp("in-force-rec"))
    return {
        (k, cls): _rec_run(
            compiled_annuity, cls, _file(f"mp_{cls}_{k}_rec"), root / f"rec-{k}-{cls}"
        )
        for k in ("P", "O")
        for cls in ("level", "cpi")
    }


# ------------------------------------------------------------------ the committed files


def test_the_committed_files_are_the_generated_tables(files: dict[str, Any]) -> None:
    """``docs/spec/in_force_group_inputs.py`` wrote exactly these tables, from the bytes the
    loader hashes for an in-memory table. A pyarrow upgrade may change the bytes without a
    value: rerun the script then. Off the laptop that wrote them, another C library's ``exp``
    and ``pow`` move last bits, and the values are held to 1e-12 relative (:mod:`libm`)."""
    on_disk = {p.stem for p in IF_DIR.glob("*.parquet")}
    assert on_disk == set(files)
    for name, table in files.items():
        assert_generated(_file(name), table, name)


def test_the_curves_equal_the_script_bit_for_bit(inforce_oracle: ModuleType) -> None:
    """Section 5.1 and 5.3: the risk-free curves, the composed curve at recognition P_rec, the
    locked-in curve P_L = P_rec(tau0 + t) / P_rec(tau0) and the composed current curve
    P_cur, with the spec's checkpoints."""
    o = inforce_oracle
    rec, cur = range(data.IF_T_REC + 1), range(data.IF_T + 1)
    assert _column("discount_rec", "t") == list(rec)
    assert _same(_column("discount_rec", "rf_df"), [o.rf_df(t, 0.040) for t in rec])
    assert _same(_column("discount_cur", "rf_df"), [o.rf_df(t, 0.045) for t in cur])
    p_rec = _column("p_rec", "rf_df")
    p_l = _column("p_l", "rf_df")
    p_cur = _column("df_cur", "rf_df")
    assert _column("p_rec", "t") == list(rec)
    assert _column("p_l", "t") == _column("df_cur", "t") == list(cur)
    assert _same(p_rec, [o.p_rec(t) for t in rec])
    assert _same(p_l, [o.p_lock(t) for t in cur])
    assert _same(p_cur, [o.p_cur(t) for t in cur])
    assert p_l[0] == 1.0
    assert (p_rec[12], p_rec[13], p_rec[501]) == (
        0.9690612702196784,
        0.9664087352366604,
        0.15425915191945347,
    )
    assert (p_l[1], p_l[12], p_l[489]) == (
        0.9972627788721588,
        0.9665612216490105,
        0.15918410595904237,
    )
    assert p_l[0] / p_l[1] - 1 == 0.002744734071933319  # j_L(1)
    assert p_l[11] / p_l[12] - 1 == 0.002927838156386864  # j_L(12)
    assert (p_cur[1], p_cur[12], p_cur[489]) == (
        0.9970772395543179,
        0.964228056970892,
        0.13171516763686483,
    )
    assert p_rec[0] / p_rec[12] == 1.0319264949813836  # the period's accretion factor


def test_the_cpi_paths_equal_the_script_bit_for_bit(inforce_oracle: ModuleType) -> None:
    """Section 5.2 and 5.3: the recognition path, the actual history as known at V, the
    locked-in splice and the current path, with the spec's checkpoints."""
    o = inforce_oracle
    assert _column("cpi_rec", "u") == list(range(-24, data.IF_T_REC + 1))
    assert _same(_column("cpi_rec", "cpi"), [o.i_rec(u) for u in range(-24, 502)])
    act = dict(zip(_column("cpi_act", "u"), _column("cpi_act", "cpi"), strict=True))
    assert list(act) == list(range(-26, 1))
    assert _same(list(act.values()), [o.i_act(u) for u in act])
    for name, path in (("cpi_locked", o.i_lock), ("cpi_cur", o.i_cur)):
        assert _column(name, "u") == list(range(-26, data.IF_T + 1)), name
        assert _same(_column(name, "cpi"), [path(u) for u in range(-26, 490)]), name
    locked = dict(zip(_column("cpi_locked", "u"), _column("cpi_locked", "cpi"), strict=True))
    current = dict(zip(_column("cpi_cur", "u"), _column("cpi_cur", "cpi"), strict=True))
    assert (act[-14], act[-12], act[-2], act[0]) == (
        99.34845327805992,
        100.1,
        102.76230218844762,
        103.3032,
    )
    assert (locked[1], locked[10], locked[12], locked[13]) == (
        103.51598767815085,
        105.45090936406004,
        105.88578000000001,
        106.10388737010459,
    )
    assert (current[1], current[12]) == (103.55797355954225, 106.402296)
    assert act[-2] / act[-14] == 1.0343623760385368  # E(1), the same on every path at V
    assert locked[10] / locked[-2] == 1.026163360671718  # E_L(13)
    assert current[10] / current[-2] == 1.030333063965748  # E_cur(13)
    for name in ("cpi_act", "cpi_locked", "cpi_cur", "cpi_rec"):
        assert pq.read_table(_file(name)).schema.metadata[b"halley.estimated_rows"] == b"0"
    meta = pq.read_table(_file("cpi_locked")).schema.metadata
    assert meta[b"halley.act_sha256"].decode() == _sha(_file("cpi_act"))
    assert meta[b"halley.rec_sha256"].decode() == _sha(_file("cpi_rec"))
    assert meta[b"halley.rebased_from"].decode() == _sha(_file("cpi_rec"))
    assert meta[b"halley.tau0"] == b"12"


def test_the_expense_rebase(inforce_oracle: ModuleType, tmp_path: Path) -> None:
    """Section 5.4: maint_pm_old = maint_pm_prev x I_act(0) / I_act(-tau), in that order, priced
    at V; version 2 is priced at V already and not rebased."""
    o = inforce_oracle
    maint_old = _column("expense_old", "maint_pm")
    assert maint_old == [5.160000000000001]
    assert _same(maint_old, [o.MAINT_V1 * o.i_act(0) / o.i_act(-o.TAU)])
    meta = pq.read_table(_file("expense_old")).schema.metadata
    assert meta[b"halley.price_base"] == b"2026-12"
    assert meta[b"halley.rebased_from"].decode() == _sha(_file("expense_v1"))
    assert meta[b"halley.cpi_act_sha256"].decode() == _sha(_file("cpi_act"))
    assert meta[b"halley.helper"] == b"rebase_expense_table"
    assert meta[b"halley.version"] == b"annuity-ref-basis-0.1"  # the source's, carried over
    for name, value, base in (("expense_v1", 5.00, b"2025-12"), ("expense_v2", 5.30, b"2026-12")):
        assert _column(name, "maint_pm") == [value]
        assert pq.read_table(_file(name)).schema.metadata[b"halley.price_base"] == base
    # a rebase spans the months between the two price bases, priced on the actual history
    with pytest.raises(LoadError, match="priced at 2025-12, which is not tau = 6 months before"):
        rebase_expense_table(_file("expense_v1"), _file("cpi_act"), 6, "2026-12")
    unpriced = data.expense_table().replace_schema_metadata({})
    with pytest.raises(LoadError, match=r"carries no metadata halley\.price_base"):
        rebase_expense_table(unpriced, _file("cpi_act"), 12, "2026-12")
    short = pa.table({"u": pa.array([-5, -4, -3, -2, -1, 0], pa.int32()), "cpi": [1.0] * 6})
    with pytest.raises(LoadError, match="cpi_act: covers no month -12; the rebase needs"):
        rebase_expense_table(_file("expense_v1"), short, 12, "2026-12")
    with pytest.raises(LoadError, match="price_base '2026-13' is not YYYY-MM"):
        rebase_expense_table(_file("expense_v1"), _file("cpi_act"), 12, "2026-13")


def test_the_tables_of_versions_1_and_2(inforce_oracle: ModuleType) -> None:
    """Section 3.5: version 2 of the base table is 0.96 x version 1 below 120 and 1 at 120; the
    other tables are unchanged at V; ilp_zero is the locked-in runs' ILP of 0.0 (C17)."""
    o = inforce_oracle
    for name, table in (("mort_base_v1", o.MORT_V1), ("mort_base_v2", o.MORT_V2)):
        keys = zip(_column(name, "sex"), _column(name, "age"), strict=True)
        assert _same(_column(name, "q"), [table[k] for k in keys]), name
    keys = list(zip(_column("mort_cif", "age"), _column("mort_cif", "year"), strict=True))
    assert _same(_column("mort_cif", "cif"), [o.CIF[k] for k in keys])
    v2 = dict(zip(_column("mort_base_v2", "age"), _column("mort_base_v2", "q"), strict=False))
    assert v2[120] == 1.0
    assert f"{o.MORT_V2[('M', 79)]:.8g}" == "0.027966855"  # 14.1, 8 s.f.
    assert _column("ilp", "ilp") == [0.005]
    assert _column("ilp_zero", "ilp") == [0.0]
    assert _column("inv_expense", "ie_rate") == [0.0015]
    assert _sha(_file("mort_base_v1")) == _sha(IF_DIR.parent / "annuity" / "mort_base.parquet")


def test_the_model_points_and_the_groups_rows(
    inforce_oracle: ModuleType, variants: dict[str, dict[str, Any]]
) -> None:
    """Section 13: the model points at recognition and at V, the premiums (which the script
    sets by its pricing rule, and the files hold as the printed numbers), the actual survivors,
    cu_act, and the groups rows with the three locked-in hashes."""
    o = inforce_oracle
    for k, v in variants.items():
        for cls in ("level", "cpi"):
            mps = [mp for mp in v["mps"] if mp["cls"] == cls]
            at_rec = {mp["policy_id"]: mp["n_rec"] for mp in mps}
            for at, counts in (("rec", at_rec), ("v", v["spec"]["n_act"])):
                table = pq.read_table(_file(f"mp_{cls}_{k}_{at}")).to_pylist()
                assert [r["policy_id"] for r in table] == [mp["policy_id"] for mp in mps]
                for r, mp in zip(table, mps, strict=True):
                    assert r["sex"] == mp["sex"]
                    assert (r["date_of_birth"].year, r["date_of_birth"].month) == mp["dob"][:2]
                    assert r["pols_if_init"] == counts[mp["policy_id"]]
                    assert r["instalment_pm"] == mp["inst"]
                    assert r["single_premium"] == mp["single_premium"]
                    assert r["profit_bucket"] == v["spec"]["bucket"]
                    assert r["issue_date"] == V_REC
                meta = pq.read_table(_file(f"mp_{cls}_{k}_{at}")).schema.metadata
                when = V_REC if at == "rec" else V
                assert meta[b"halley.valuation_date"].decode() == when.isoformat()
        groups = pq.read_table(_file(f"groups_{k}")).to_pylist()[0]
        assert groups["cu_act"] == o.GROUPS[k]["cu_act"]
        assert groups["profit_bucket"] == o.GROUPS[k]["bucket"]
        assert groups["recognition_date"] == groups["prev_valuation_date"] == V_REC
        assert groups["discount_rec_sha256"] == _sha(_file("discount_rec"))
        assert groups["ilp_sha256"] == _sha(_file("ilp"))
        assert groups["cpi_rec_sha256"] == _sha(_file("cpi_rec"))
        assert not {"prev_run_id", "csm_prev", "lc_prev"} & set(groups)  # read, never retyped
        rec = pq.read_table(_file(f"groups_{k}_rec")).to_pylist()[0]
        assert rec == {
            "portfolio": "IA",
            "cohort": 2025,
            "profit_bucket": o.GROUPS[k]["bucket"],
            "recognition_date": V_REC,
        }


# ---------------------------------------------------------------- the recognition runs


def test_expected_in_force_is_the_recognition_run_at_tau_bit_for_bit(
    rec_runs: dict[tuple[str, str], RunResult], variants: dict[str, dict[str, Any]]
) -> None:
    """Section 3.4: n_exp_i = pols_if_i^rec(tau) from the recognition run's outputs at full
    precision, the OLD run's pols_if_init; the instalment is the file at V's; every other column
    is the recognition copy's, with its Arrow types."""
    for (k, cls), rec in rec_runs.items():
        old = expected_in_force(rec, TAU, _file(f"mp_{cls}_{k}_v"))
        copy = pq.read_table(rec.run_dir / "inputs" / "model_points.parquet")
        assert old.schema.remove_metadata() == copy.schema.remove_metadata()
        pids = old.column("policy_id").to_pylist()
        n_exp = [variants[k]["inputs"]["n_exp"][pid] for pid in pids]
        assert _same(old.column("pols_if_init").to_pylist(), n_exp), (k, cls)
        at_v = pq.read_table(_file(f"mp_{cls}_{k}_v"))
        assert old.column("instalment_pm").equals(at_v.column("instalment_pm"))
        for name in copy.column_names:
            if name not in ("pols_if_init", "instalment_pm"):
                assert old.column(name).equals(copy.column(name)), name
        assert old.schema.metadata == {
            b"halley.derived_from": rec.run_id.encode(),
            b"halley.tau": b"12",
            b"halley.helper": b"expected_in_force",
        }
    assert _same(
        [
            expected_in_force(rec_runs[("P", c)], TAU, _file(f"mp_{c}_P_v"))
            .column("pols_if_init")
            .to_pylist()[0]
            for c in ("level", "cpi")
        ],
        [97.35061873366202, 48.97700540091314],
    )


def test_expected_in_force_takes_the_actual_instalment_and_keeps_every_model_point(
    compiled_annuity: dict[str, CompiledModel], tmp_path: Path
) -> None:
    """An anniversary inside the period escalates the instalment before V: the OLD run carries
    the actual instalment of the file at V (in-force 5.5). A model point with no survivors is
    absent at V and stays in OLD with its expected in-force and its previous instalment."""
    start = dt.date(2025, 7, 1)  # m_c = -5 at V_rec: an anniversary in month 7 of the period
    rows = [dict(r) for r in data.IN_FORCE_MODEL_POINTS["P"]]
    cpi_row = {**rows[1], "issue_date": dt.date(2025, 6, 30), "annuity_start_date": start}
    rec = _rec_run(
        compiled_annuity, "cpi", data.model_points_table([cpi_row]), tmp_path / "rec-cpi"
    )
    escalated = rec.lane("ann_pp", 12)
    assert escalated[7] > escalated[6] == 800.0  # the period's escalation, on the REC path
    actual = 826.4  # what the administration paid from month 7, an actual CPI ratio
    at_v = data.model_points_table(
        [{**cpi_row, "pols_if_init": 50.0, "instalment_pm": actual}], valuation_date=V
    )
    old = expected_in_force(rec, TAU, at_v)
    assert old.column("instalment_pm").to_pylist() == [actual]
    assert old.column("pols_if_init").to_pylist() == [float(rec.lane("pols_if", 12)[TAU])]

    second = {**rows[0], "policy_id": 13, "date_of_birth": dt.date(1946, 5, 20)}
    rec = _rec_run(
        compiled_annuity,
        "level",
        data.model_points_table([rows[0], second]),
        tmp_path / "rec-level",
    )
    survivors = data.model_points_table([{**rows[0], "pols_if_init": 98.0}], valuation_date=V)
    old = expected_in_force(rec, TAU, survivors)
    assert old.column("policy_id").to_pylist() == [11, 13]
    assert _same(
        old.column("pols_if_init").to_pylist(),
        [rec.lane("pols_if", 11)[TAU], rec.lane("pols_if", 13)[TAU]],
    )
    assert old.column("instalment_pm").to_pylist() == [1000.0, 1000.0]


def test_expected_in_force_refuses_what_it_cannot_derive(
    compiled_annuity: dict[str, CompiledModel],
    rec_runs: dict[tuple[str, str], RunResult],
    tmp_path: Path,
) -> None:
    rec = rec_runs[("P", "level")]
    at_v = _file("mp_level_P_v")
    with pytest.raises(LoadError, match=r"covers months 0\.\.501, not month tau = 600"):
        expected_in_force(rec, 600, at_v)
    with pytest.raises(LoadError, match="tau is a whole number of months, at least 0, not -1"):
        expected_in_force(rec, -1, at_v)
    joined = data.model_points_table(
        [{**data.IN_FORCE_MODEL_POINTS["P"][0], "policy_id": 19}], valuation_date=V
    )
    with pytest.raises(LoadError, match=r"row 0, policy 19 is not in the previous projection run"):
        expected_in_force(rec, TAU, joined)
    # the expected in-force is a mean: a run of two scenarios is refused (D4, one check)
    two = data.discount_table(n_scenarios=2, t_max=data.IF_T_REC)
    two_cpi = data.cpi_table(n_scenarios=2, u_hi=data.IF_T_REC)
    stochastic = _project(
        compiled_annuity["level"],
        _file("mp_level_P_rec"),
        tmp_path / "two",
        at=V_REC,
        tables=REC_TABLES,
        rf_df=two,
        cpi=two_cpi,
        horizon=data.IF_T_REC,
    )
    with pytest.raises(LoadError, match=r"has J = 2 scenarios.*require_deterministic"):
        expected_in_force(stochastic, TAU, at_v)
    # given as its directory, the run is read from its verified files
    from_dir = expected_in_force(rec.run_dir, TAU, at_v)
    assert from_dir.equals(expected_in_force(rec, TAU, at_v))


def test_the_recognition_stage_of_the_in_force_example(
    compiled_new_group: dict[bool, CompiledModel],
    rec_runs: dict[tuple[str, str], RunResult],
    variants: dict[str, dict[str, Any]],
    tmp_path: Path,
) -> None:
    """Section 7.3 and 14.2: NewGroup on the two class runs of each variant gives the script's
    P_g, CSM(0) and LC(0), F_prev(0) and F_prev(12), and its monthly roll to month 12; the
    groups file at V takes prev_run_id, csm_prev and lc_prev from that run. At recognition the
    locked-in and the current curves are both the composed recognition curve (P_L = P)."""
    _p_rec, p_l = locked_in_curve(_file("discount_rec"), _file("ilp"), 0, data.IF_T_REC)
    df_cur = composed_curve(_file("discount_rec"), _file("ilp"), data.IF_T_REC)
    for k, v in variants.items():
        stage = run_aggregation(
            compiled_new_group[True],
            groups=_file(f"groups_{k}_rec"),
            bases={"cur": [rec_runs[(k, "level")], rec_runs[(k, "cpi")]]},
            scenarios={"p_l": p_l, "df_cur": df_cur},
            settings=RunSettings(valuation_date=V_REC),
            run_dir=tmp_path / f"recognised-{k}",
        )
        gid = f"IA-2025-{v['spec']['bucket']}"
        p_g = v["recog"]["p_g"]
        assert stage.lane("p_g", gid)[0] == p_g
        got = {
            "csm_0": stage.lane("csm_0", gid)[0],
            "lc_0": stage.lane("lc_0", gid)[0],
            "f_prev_0": stage.lane("fcf_g", gid)[0],
            "f_prev_tau": stage.lane("fcf_g", gid)[TAU],
            "csm_12": stage.lane("csm", gid)[TAU],
            "lc_12": stage.lane("lc", gid)[TAU],
            "n_12": stage.lane("cu_next", gid)[TAU],
        }
        want = {
            "csm_0": v["recog"]["csm_0"],
            "lc_0": v["recog"]["lc_0"],
            "f_prev_0": v["inputs"]["f_prev_0"],
            "f_prev_tau": v["inputs"]["f_prev_tau"],
            "csm_12": v["rec_roll"]["csm"][TAU],
            "lc_12": v["rec_roll"]["lc"][TAU],
            "n_12": v["rec_roll"]["n"][TAU],
        }
        for name in got:
            assert within(got[name], want[name], p_g, rel=1e-12), (k, name)
        assert all(r.passed for r in stage.invariants)
        groups = data.with_previous_run(pq.read_table(_file(f"groups_{k}")), stage)
        row = groups.to_pylist()[0]
        assert row["prev_run_id"] == stage.run_id
        assert row["csm_prev"] == got["csm_0"]
        assert row["lc_prev"] == got["lc_0"]
    assert _ten(variants["P"]["recog"]["csm_0"]) == "741187.762"
    assert _ten(variants["O"]["recog"]["lc_0"]) == "370593.238"


def _ten(x: float) -> str:
    return f"{float(x):.10g}"


# --------------------------------------------------------------------------------- G4


def test_check_membership_is_g4(tmp_path: Path) -> None:
    """Every policy at V was a member, with its group key unchanged; a model point with no
    survivors is simply absent at V. A LoadError names the file at V, the row and the
    policy."""
    rows = [dict(r) for r in data.IN_FORCE_MODEL_POINTS["P"]]
    prev = tmp_path / "mp_rec.parquet"
    pq.write_table(data.model_points_table(rows), prev)
    check_membership(prev, data.model_points_table(rows[:1], valuation_date=V))
    check_membership(prev, _file("mp_level_P_v"))
    joined = tmp_path / "mp_v.parquet"
    pq.write_table(data.model_points_table([rows[0], {**rows[1], "policy_id": 99}]), joined)
    with pytest.raises(LoadError) as info:
        check_membership(prev, joined)
    assert str(info.value) == (
        f"model points at V ({joined}): row 1, policy 99 is not in previous model points "
        f"({prev}); contracts do not join a group after recognition (G4, para 28)"
    )
    for name, value in (("cohort", 2026), ("profit_bucket", "ONEROUS"), ("portfolio", "IB")):
        moved = data.model_points_table([rows[0], {**rows[1], name: value}])
        with pytest.raises(
            LoadError,
            match=f"row 1, policy 12 has {name} {value!r}, but row 1 of previous model points "
            ".* a policy's group key does not change",
        ):
            check_membership(prev, moved)
    with pytest.raises(LoadError, match="column cohort is missing"):
        check_membership(prev, data.model_points_table(rows).drop_columns(["cohort"]))


def test_the_curve_and_cpi_helpers_need_every_month(tmp_path: Path) -> None:
    disc, ilp = _file("discount_rec"), _file("ilp")
    with pytest.raises(LoadError, match="covers no month 502; the composed curve needs every "):
        locked_in_curve(disc, ilp, 13, data.IF_T)
    p_rec, p_l = locked_in_curve(disc, ilp, 0, 24)
    assert p_l.column("rf_df").to_pylist() == p_rec.column("rf_df").to_pylist()  # P_L = P
    assert p_l.column("rf_df").to_pylist()[0] == 1.0
    two = data.discount_table(n_scenarios=2)
    # the one J = 1 check of D4, with the file as what it checks
    with pytest.raises(
        LoadError,
        match=r"a composed or locked-in curve is one deterministic path; discount file has J = 2 "
        r"scenarios.*\(require_deterministic\)",
    ):
        composed_curve(two, ilp, 12)
    act, rec = _file("cpi_act"), _file("cpi_rec")
    with pytest.raises(LoadError, match="covers no month 502; the splice needs every month"):
        locked_in_cpi(act, rec, TAU, data.IF_T + 1)
    early = data.cpi_actual_table().slice(0, 20)  # ends at u = -7
    with pytest.raises(LoadError, match="the actual history ends at u = 0"):
        locked_in_cpi(early, rec, TAU, data.IF_T)
    months: Any = 1.5
    with pytest.raises(LoadError, match="tau0 is a whole number of months"):
        locked_in_cpi(act, rec, months, data.IF_T)


# --------------------------------------------------------------------------- provenance


def _committed() -> dict[str, ReadFile]:
    return {p.stem: read_source(p, p.stem) for p in IF_DIR.glob("*.parquet")}


def _finder(files: dict[str, ReadFile]) -> Any:
    by_sha = {f.sha256: f for f in files.values()}

    def find(sha: str) -> ReadFile:
        if sha not in by_sha:
            raise LoadError(f"no file with sha256 {sha}")
        return by_sha[sha]

    return find


def _check(f: ReadFile, key: str, find_file: Any, **kw: Any) -> ProvenanceCheck:
    def no_run(run_id: str) -> RunResult:
        raise LoadError(f"no run {run_id}")

    defaults: dict[str, Any] = {
        "valuation_date": V,
        "bases": {},
        "previous": None,
        "find_run": no_run,
    }
    defaults.update(kw)
    return ProvenanceCheck(key, f.metadata[key], f, find_file=find_file, **defaults)


def test_the_recomputations_are_registered_and_reproduce_every_derived_file() -> None:
    """Each helper's file is made again from the sources its metadata names by the
    recomputation registered for ``halley.rebased_from``: bit for bit on the laptop that wrote
    it, within ``LIBM_ULPS`` on a machine whose C library rounds ``exp`` the other way (the
    loader's rule); a value changed in the file, or a helper the module does not know, is a
    LoadError for the loader to report."""
    assert {
        "rebased_from": recompute_rebased,
        "derived_from": recompute_derived,
    } == stage_module.PROVENANCE_HELPERS
    files = _committed()
    find = _finder(files)
    derived = ("expense_old", "p_rec", "p_l", "cpi_locked", "df_cur")
    for name in derived:
        f = files[name]
        match = file_match(f, recompute_rebased(_check(f, "rebased_from", find)), ulps=LIBM_ULPS)
        assert match.difference is None, (name, match)
        assert match.ulps <= 4, (name, match)  # the envelope's derivation: four at most
    marked = {n for n, f in files.items() if "rebased_from" in f.metadata}
    assert marked == set(derived)
    p_l = files["p_l"]
    values = p_l.table.column("rf_df").to_pylist()
    values[5] = values[5] * (1 + 1e-15)
    edited = p_l.table.set_column(2, "rf_df", pa.array(values, pa.float64()))
    f = read_source(edited.replace_schema_metadata(p_l.table.schema.metadata), "edited p_l")
    assert file_difference(f, recompute_rebased(_check(f, "rebased_from", find))) == (
        f"column rf_df, row 5: {values[5]!r} against {p_l.table.column('rf_df')[5].as_py()!r}"
    )
    # another machine's exp may round a last bit the other way: within LIBM_ULPS the file is
    # accepted and the distance recorded; beyond it the difference names the room allowed
    original = p_l.table.column("rf_df").to_pylist()
    nudged = list(original)
    nudged[5] = float(np.nextafter(original[5], np.inf))
    near = p_l.table.set_column(2, "rf_df", pa.array(nudged, pa.float64()))
    f = read_source(near.replace_schema_metadata(p_l.table.schema.metadata), "nudged p_l")
    recomputed = recompute_rebased(_check(f, "rebased_from", find))
    assert file_difference(f, recomputed) is not None
    assert file_match(f, recomputed, ulps=LIBM_ULPS) == TableMatch(None, 1)
    off = [v * (1 + 1e-13) for v in original]
    far = p_l.table.set_column(2, "rf_df", pa.array(off, pa.float64()))
    f = read_source(far.replace_schema_metadata(p_l.table.schema.metadata), "far p_l")
    difference = file_difference(f, recompute_rebased(_check(f, "rebased_from", find)), ulps=8)
    assert difference is not None
    assert difference.startswith(f"column rf_df, row 0: {off[0]!r} against {original[0]!r}, ")
    assert difference.endswith(" ulps apart where 8 are allowed")
    # bit for bit is the values, the column types and the halley.* metadata: a file whose values
    # are the recomputation's but whose metadata carries one entry more, or whose month column
    # is another integer type, is not the recomputed file either
    noted = dict(p_l.table.schema.metadata)
    noted[b"halley.note"] = b"typed by hand"
    f = read_source(p_l.table.replace_schema_metadata(noted), "noted p_l")
    difference = file_difference(f, recompute_rebased(_check(f, "rebased_from", find)))
    assert difference is not None
    assert difference.startswith("metadata {")
    assert "typed by hand" in difference
    wide = p_l.table.set_column(1, "t", p_l.table.column("t").cast(pa.int64()))
    f = read_source(wide.replace_schema_metadata(p_l.table.schema.metadata), "wide p_l")
    assert file_difference(f, recompute_rebased(_check(f, "rebased_from", find))) == (
        "column t is int64 against int32"
    )
    by_hand = dict(p_l.table.schema.metadata)
    by_hand[b"halley.helper"] = b"by_hand"
    f = read_source(p_l.table.replace_schema_metadata(by_hand), "by hand")
    with pytest.raises(LoadError, match=r"names no helper of halley\.aggregation\.inforce"):
        recompute_rebased(_check(f, "rebased_from", find))
    with pytest.raises(LoadError, match="no file with sha256"):
        recompute_rebased(_check(files["expense_old"], "rebased_from", _finder({})))


def test_a_source_not_found_is_reported_against_the_file_being_verified() -> None:
    """The finder's refusal names only the sha256 it looked for; the loader reports it against
    the file it was verifying (a later period whose recognition runs moved, say), with the
    metadata entry it recomputes from. A helper's refusal that names the file already is left
    as it is."""
    p_l = read_source(_file("p_l"), "scenario input p_l")
    nowhere = stage_module._Finder([], None, {}, None)
    with pytest.raises(LoadError) as info:
        stage_module._verify_provenance(iter([p_l]), nowhere, V, {}, None)
    message = str(info.value)
    source = p_l.metadata["rebased_from"]
    assert message.startswith(
        f"scenario input p_l ({_file('p_l')}): recomputing it from halley.rebased_from = "
        f"{source}: no file with sha256 {source} in the stage's inputs"
    )
    by_hand = dict(p_l.table.schema.metadata)
    by_hand[b"halley.helper"] = b"by_hand"
    f = read_source(p_l.table.replace_schema_metadata(by_hand), "scenario input p_l")
    with pytest.raises(LoadError) as info:
        stage_module._verify_provenance(iter([f]), nowhere, V, {}, None)
    assert str(info.value).startswith(
        "scenario input p_l: carries halley.rebased_from, but its halley.helper is 'by_hand'"
    )


def test_the_expected_in_force_is_recomputed_from_the_run_it_names(
    compiled_annuity: dict[str, CompiledModel],
    rec_runs: dict[tuple[str, str], RunResult],
    tmp_path: Path,
) -> None:
    """``halley.derived_from``: the OLD model point file is made again from the recognition run
    it names and the actual instalments of the cur basis's model points at V."""
    rec = rec_runs[("P", "level")]
    old = read_source(expected_in_force(rec, TAU, _file("mp_level_P_v")), "old model points")
    cur = _project(
        compiled_annuity["level"],
        _file("mp_level_P_v"),
        tmp_path / "cur",
        at=V,
        tables=CUR_TABLES,
        rf_df=_file("discount_cur"),
        cpi=_file("cpi_cur"),
        horizon=data.IF_T,
    )
    other = rec_runs[("P", "cpi")]

    def find_run(run_id: str) -> RunResult:
        return {rec.run_id: rec, other.run_id: other}[run_id]

    check = _check(old, "derived_from", _finder({}), bases={"cur": [cur]}, find_run=find_run)
    assert file_difference(old, recompute_derived(check)) is None
    with pytest.raises(LoadError, match="the expected in-force takes the actual instalments"):
        recompute_derived(dataclasses.replace(check, bases={}))
    wrong = dataclasses.replace(check, value=other.run_id)
    assert file_difference(old, recompute_derived(wrong)) is not None
    # the file's own halley.tau is held to the period: a file built at month 11 of a 12-month
    # period would rebuild bit for bit from its own metadata, and is refused
    early = read_source(expected_in_force(rec, TAU - 1, _file("mp_level_P_v")), "early")
    assert early.metadata["tau"] == str(TAU - 1)
    with pytest.raises(
        LoadError,
        match=rf"early: halley\.tau is {TAU - 1}, but the run it is derived from, {rec.run_id}, "
        rf"values at {V_REC.isoformat()} and the stage at {V.isoformat()}, {TAU} months later",
    ):
        recompute_derived(dataclasses.replace(check, file=early))
    later = dataclasses.replace(check, valuation_date=dt.date(2027, 1, 31))
    with pytest.raises(LoadError, match=rf"halley\.tau is {TAU}, .* 13 months later"):
        recompute_derived(later)
