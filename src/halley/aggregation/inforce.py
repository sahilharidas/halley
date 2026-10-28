# pyright: basic
"""Cross-run helpers of milestone M9: the inputs of an in-force group's runs at the valuation
date, built from the previous run and the files fixed at recognition (design section 10;
in-force spec sections 3.2, 3.4, 5.1, 5.2 and 5.4).

"The loader re-bases" is read as "a helper writes a hashed file with provenance metadata, which
the ordinary loader reads and re-verifies": every projection run stays ordinary (D4; in-force
C17) and each derived file says where it came from. Each helper here returns a ``pyarrow.Table``
whose ``halley.*`` metadata names its sources by sha256 (or, for the expected in-force, by run
id) and the helper that made it (``halley.helper``); the stage loader recomputes every file that
carries ``halley.rebased_from`` or ``halley.derived_from`` with the recomputations registered at
the end of this module (:func:`recompute_rebased`, :func:`recompute_derived`) and requires
equality within the C library's last bits, bit for bit on one machine
(:mod:`halley.aggregation.stage`, "Provenance verification").

* :func:`expected_in_force`: the ``OLD`` run's model point file, the previous projection run's
  ``pols_if`` at month ``tau`` with the actual instalment at V;
* :func:`check_membership`: G4 between the previous model points and those at V;
* :func:`rebase_expense_table`: the previous expense table in V prices, with actual CPI;
* :func:`composed_curve` and :func:`locked_in_curve`: the composed curve
  ``rf_df(t) x exp(-ilp t / 12)`` and the locked-in curve re-based to V;
* :func:`locked_in_cpi`: actual CPI history to V spliced with the inflation expected at
  recognition.

Every value is formed in plain float64 scalars in the order the in-force spec writes it, so the
files equal ``docs/spec/in_force_group_numbers.py`` bit for bit on the machine that runs both
(another C library's ``exp`` may round a last bit). pyarrow ships no type stubs, so
by the convention of the modules that build or read Arrow tables this one is checked in
pyright's basic mode.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pyarrow as pa

from halley.aggregation.stage import ProvenanceCheck, register_provenance
from halley.authoring.contracts import GROUP_KEY
from halley.errors import LoadError
from halley.runtime.loader import (
    CUR_BASIS,
    ReadFile,
    Source,
    read_copy,
    read_source,
    table_columns,
)
from halley.runtime.run import RunResult, VerifiedRun, verified_run

HELPER = "helper"
"""The metadata key (``halley.helper``) naming the helper that made a file, which its
recomputation dispatches on."""

PRICE_BASE = "price_base"
"""The metadata key of an expense table's price base, ``YYYY-MM`` (in-force 3.5)."""

# ---------------------------------------------------------------------------------- reading


def _read(source: Source, label: str) -> ReadFile:
    """A source as read: a :class:`ReadFile` read before keeps its label (a run's copy names the
    run), anything else is read under ``label``."""
    if isinstance(source, ReadFile):
        return source
    return read_source(source, label)


class _FileScenarios(NamedTuple):
    """A scenario file's number of scenarios (its distinct ``scenario`` ids, 1 without the
    column), for :func:`~halley.aggregation.require_deterministic`."""

    n_scenarios: int


def _require_one_scenario(f: ReadFile, cols: Mapping[str, Any], reason: str) -> None:
    """The file holds one scenario, through the one J = 1 check of D4
    (:func:`~halley.aggregation.require_deterministic`), with the file as what is checked."""
    from halley.aggregation import require_deterministic  # the one J = 1 check (D4)

    n = len(set(cols["scenario"].tolist())) if "scenario" in cols else 1
    require_deterministic(_FileScenarios(n), f.where, reason, LoadError)


def _by_month(f: ReadFile, value: str, reason: str) -> dict[int, float]:
    """A one-scenario file's values by month: the month column is ``t`` or ``u``, as the loader
    reads a scenario file; each month once. ``reason`` says why the helper reads one scenario,
    for the message of :func:`_require_one_scenario`."""
    cols = table_columns(f)
    months = [name for name in ("t", "u") if name in cols]
    if len(months) != 1:
        raise LoadError(f"{f.where}: one month column, t or u, is needed")
    if value not in cols:
        raise LoadError(f"{f.where}: column {value} is missing")
    _require_one_scenario(f, cols, reason)
    out: dict[int, float] = {}
    for m, x in zip(cols[months[0]].tolist(), cols[value].tolist(), strict=True):
        if m in out:
            raise LoadError(f"{f.where}: month {m} appears twice")
        out[int(m)] = float(x)
    return out


def _covering(f: ReadFile, values: Mapping[int, float], lo: int, hi: int, why: str) -> None:
    missing = [m for m in range(lo, hi + 1) if m not in values]
    if missing:
        raise LoadError(
            f"{f.where}: covers no month {missing[0]}; {why} needs every month {lo}..{hi}"
        )


def _one_value(f: ReadFile, column: str) -> float:
    cols = table_columns(f)
    if column not in cols:
        raise LoadError(f"{f.where}: column {column} is missing")
    values = cols[column].tolist()
    if len(values) != 1:
        raise LoadError(f"{f.where}: {len(values)} rows, not one")
    return float(values[0])


def _month(price_base: str, where: str) -> int:
    """``mi`` of a price base written ``YYYY-MM``: ``12 * year + month - 1``."""
    try:
        year, month = (int(x) for x in price_base.split("-"))
    except ValueError:
        raise LoadError(f"{where}: price_base {price_base!r} is not YYYY-MM") from None
    if len(price_base) != 7 or not 1 <= month <= 12:
        raise LoadError(f"{where}: price_base {price_base!r} is not YYYY-MM")
    return 12 * year + month - 1


def _metadata(**entries: str) -> dict[bytes, bytes]:
    return {f"halley.{k}".encode(): v.encode() for k, v in entries.items()}


def _check_tau(tau: object, name: str, lo: int = 0) -> int:
    if not isinstance(tau, int) or isinstance(tau, bool) or tau < lo:
        raise LoadError(f"{name} is a whole number of months, at least {lo}, not {tau!r}")
    return tau


# ------------------------------------------------------------------------------ membership


def check_membership(prev_mps: Source, cur_mps: Source) -> None:
    """G4 between a class's model points at the previous valuation date and at V (design
    section 9; in-force 3.2, 12.4): every policy at V was in the previous file, and its group
    key (``portfolio``, ``cohort``, ``profit_bucket``) is unchanged. A LoadError names the file
    at V, the row and the policy (contracts do not join a group after recognition, para 28; a
    policy's group key does not change, para 24). A policy of the previous file that is absent
    at V had no survivors; it is not an error."""
    prev = _read(prev_mps, "previous model points")
    cur = _read(cur_mps, "model points at V")
    p_cols, c_cols = table_columns(prev), table_columns(cur)
    for f, cols in ((prev, p_cols), (cur, c_cols)):
        for name in ("policy_id", *GROUP_KEY):
            if name not in cols:
                raise LoadError(f"{f.where}: column {name} is missing")
    rows = {pid: row for row, pid in enumerate(p_cols["policy_id"].tolist())}
    keys_now = {name: c_cols[name].tolist() for name in GROUP_KEY}
    keys_then = {name: p_cols[name].tolist() for name in GROUP_KEY}
    for row, pid in enumerate(c_cols["policy_id"].tolist()):
        j = rows.get(pid)
        if j is None:
            raise LoadError(
                f"{cur.where}: row {row}, policy {pid} is not in {prev.where}; contracts do not "
                "join a group after recognition (G4, para 28)"
            )
        for name in GROUP_KEY:
            now, then = keys_now[name][row], keys_then[name][j]
            if now != then:
                raise LoadError(
                    f"{cur.where}: row {row}, policy {pid} has {name} {now!r}, but row {j} of "
                    f"{prev.where} has {then!r}; a policy's group key does not change (G4, "
                    "para 24)"
                )


# ------------------------------------------------------------------------ expected in-force


def _verified(prev: RunResult | str | Path, what: str) -> VerifiedRun:
    """The previous projection run read through its directory, every file verified (D21)."""
    if isinstance(prev, RunResult):
        v = verified_run(prev.run_dir, what)
        if v.manifest.get("run_id") != prev.run_id:
            raise LoadError(
                f"{what} {prev.run_id} was given, but its directory {prev.run_dir} holds run "
                f"{v.manifest.get('run_id')}"
            )
        return v
    return verified_run(Path(prev), what)


def expected_in_force(prev: RunResult | str | Path, tau: int, cur_mps: Source) -> Any:
    """The ``OLD`` run's model point file of one class (in-force 3.4, 5.5; C16): the previous
    projection run's model point copy with ``pols_if_init`` replaced by its expected in-force at
    month ``tau``, ``pols_if(tau)`` read from its outputs at full precision (J = 1), and
    ``instalment_pm`` the actual instalment at V, from ``cur_mps``, for every policy present
    there (in-force 5.5: OLD carries the actual instalment, after every escalation in the
    period). Every policy of the previous run is kept, also one with no survivors at V (C16),
    which keeps the previous file's instalment, the last one known for it: the spec is silent
    on it, design section 10 takes this reading and the spec's author is to confirm it. The
    rows, the columns and their types are the previous copy's; the metadata is
    ``halley.derived_from`` (the previous run's id), ``halley.tau`` and ``halley.helper``.
    ``cur_mps`` is checked against the previous copy first (:func:`check_membership`)."""
    from halley.aggregation import require_deterministic  # the one J = 1 check (D4)

    v = _verified(prev, "previous projection run")
    result = v.result()
    who = f"the previous projection run {result.run_id}"
    tau = _check_tau(tau, "tau")
    if result.kind != "projection":
        raise LoadError(f"{who} is an {result.kind} run; the expected in-force is a projection's")
    require_deterministic(
        result, who, "the expected in-force is read from a deterministic run", LoadError
    )
    if tau > result.H:
        raise LoadError(f"{who} covers months 0..{result.H}, not month tau = {tau}")
    if "pols_if" not in result.arrays:
        raise LoadError(f"{who} did not store pols_if; add it to outputs")
    copy = v.read(v.manifest["inputs"]["model_points"], f"{who}: model points")
    cur = _read(cur_mps, "model points at V")
    check_membership(copy, cur)
    cols, now = table_columns(copy), table_columns(cur)
    for f, c in ((copy, cols), (cur, now)):
        if "instalment_pm" not in c:
            raise LoadError(f"{f.where}: column instalment_pm is missing")
    if "pols_if_init" not in cols:
        raise LoadError(f"{copy.where}: column pols_if_init is missing")
    lane = {pid: j for j, pid in enumerate(result.ids.tolist())}
    pols = np.asarray(result.var("pols_if"))[tau, :, 0]
    actual = dict(zip(now["policy_id"].tolist(), now["instalment_pm"].tolist(), strict=True))
    ids = cols["policy_id"].tolist()
    n_exp = [float(pols[lane[pid]]) for pid in ids]
    instalment = [
        float(actual[pid]) if pid in actual else float(x)
        for pid, x in zip(ids, cols["instalment_pm"].tolist(), strict=True)
    ]
    table = copy.table
    for name, values in (("pols_if_init", n_exp), ("instalment_pm", instalment)):
        i = table.schema.get_field_index(name)
        table = table.set_column(i, name, pa.array(values, pa.float64()))
    meta = _metadata(derived_from=result.run_id, tau=str(tau), helper="expected_in_force")
    return table.replace_schema_metadata(meta)


# ---------------------------------------------------------------------------- expense rebase


def rebase_expense_table(table: Source, cpi_act: Source, tau: int, price_base: str) -> Any:
    """The previous expense table in the prices of V (in-force 5.4; C8): every ``maint_pm`` as
    ``maint_pm x I_act(0) / I_act(-tau)``, in that order, with ``I_act`` the actual CPI history
    as known at V. The table must carry its own ``halley.price_base`` exactly ``tau`` months
    before ``price_base``: a rebase spans the months between the two price bases. The result
    keeps the table's columns and ``halley.*`` metadata, with ``price_base`` set to V's month,
    ``rebased_from`` the table's sha256, ``cpi_act_sha256`` the CPI file's and ``helper``."""
    src = _read(table, "expense table")
    act = _read(cpi_act, "cpi_act")
    tau = _check_tau(tau, "tau", lo=1)
    old = src.metadata.get(PRICE_BASE)
    if old is None:
        raise LoadError(f"{src.where}: carries no metadata halley.{PRICE_BASE}")
    if _month(price_base, "the new price base") - _month(old, src.where) != tau:
        raise LoadError(
            f"{src.where}: priced at {old}, which is not tau = {tau} months before {price_base}"
        )
    index = _by_month(act, "cpi", "the expense rebase reads one actual CPI history")
    _covering(act, index, -tau, 0, "the rebase")
    i_0, i_tau = index[0], index[-tau]
    cols = table_columns(src)
    if "maint_pm" not in cols:
        raise LoadError(f"{src.where}: column maint_pm is missing")
    rebased = [(float(m) * i_0) / i_tau for m in cols["maint_pm"].tolist()]
    i = src.table.schema.get_field_index("maint_pm")
    out = src.table.set_column(i, "maint_pm", pa.array(rebased, pa.float64()))
    meta = {k: v for k, v in (src.table.schema.metadata or {}).items() if k.startswith(b"halley.")}
    meta.update(
        _metadata(
            price_base=price_base,
            rebased_from=src.sha256,
            cpi_act_sha256=act.sha256,
            helper="rebase_expense_table",
        )
    )
    return out.replace_schema_metadata(meta)


# ------------------------------------------------------------------------------------ curves


def _curve_table(values: list[float], meta: dict[bytes, bytes]) -> Any:
    """A one-scenario discount file: ``scenario``, ``t``, ``rf_df`` (a discount file's value
    column, which a projection reads as its ``rf_df`` and a stage as its ``p_l``)."""
    n = len(values)
    return pa.table(
        {
            "scenario": pa.array([0] * n, pa.int32()),
            "t": pa.array(list(range(n)), pa.int32()),
            "rf_df": pa.array(values, pa.float64()),
        }
    ).replace_schema_metadata(meta)


def _composed(discount: ReadFile, ilp_table: ReadFile, last: int) -> tuple[list[float], float]:
    """``P(t) = rf_df(t) x exp(((-ilp) x t) / 12)`` for t = 0..last, in plain float64 scalars
    (the engine's ``cc_discount`` form), and the ILP."""
    rf = _by_month(discount, "rf_df", "a composed or locked-in curve is one deterministic path")
    _covering(discount, rf, 0, last, "the composed curve")
    ilp = _one_value(ilp_table, "ilp")
    return [rf[t] * math.exp(((-ilp) * t) / 12) for t in range(last + 1)], ilp


def composed_curve(discount: Source, ilp_table: Source, horizon: int) -> Any:
    """The composed curve of a discount file and an ILP table for t = 0..horizon (annuity spec,
    "Discounting"): the curve a projection run discounts with, as a file, such as the para 120
    curve ``df_cur`` an aggregation stage discloses. Metadata ``source_sha256`` (the discount
    file), ``ilp_sha256``, ``ilp``, ``rebased_from`` (the discount file) and ``helper``."""
    disc, ilp_f = _read(discount, "discount file"), _read(ilp_table, "ilp")
    horizon = _check_tau(horizon, "horizon", lo=1)
    values, ilp = _composed(disc, ilp_f, horizon)
    meta = _metadata(
        source_sha256=disc.sha256,
        ilp_sha256=ilp_f.sha256,
        ilp=repr(ilp),
        rebased_from=disc.sha256,
        helper="composed_curve",
    )
    return _curve_table(values, meta)


def locked_in_curve(discount_rec: Source, ilp_table: Source, tau0: int, horizon: int) -> Any:
    """The locked-in basis of a group's discount rates (in-force 5.1; B72(b)), as two files:
    ``p_rec``, the composed curve at recognition ``P_rec(t) = rf_df_rec(t) x exp(((-ilp) x t) /
    12)`` for t = 0..tau0 + horizon, and ``p_l``, the curve re-based to V, ``P_L(t) =
    P_rec(tau0 + t) / P_rec(tau0)`` for t = 0..horizon, a ratio of two ``P_rec`` values (so
    ``P_L(0) = 1`` exactly). Both hold their values in ``rf_df``, so the locked-in runs read
    ``p_l`` as their discount file (with an ILP table of 0.0) and a stage as its ``p_l``. Both
    carry ``source_sha256`` (the recognition discount file), ``ilp_sha256``, ``ilp``, ``tau0``,
    ``rebased_from`` (the discount file), ``helper`` and ``curve`` (``p_rec`` or ``p_l``).
    Returns ``(p_rec, p_l)``."""
    disc, ilp_f = _read(discount_rec, "discount_rec"), _read(ilp_table, "ilp")
    tau0 = _check_tau(tau0, "tau0")
    horizon = _check_tau(horizon, "horizon", lo=1)
    p_rec, ilp = _composed(disc, ilp_f, tau0 + horizon)
    base = p_rec[tau0]
    p_l = [p_rec[tau0 + t] / base for t in range(horizon + 1)]
    common = {
        "source_sha256": disc.sha256,
        "ilp_sha256": ilp_f.sha256,
        "ilp": repr(ilp),
        "tau0": str(tau0),
        "rebased_from": disc.sha256,
        "helper": "locked_in_curve",
    }
    return (
        _curve_table(p_rec, _metadata(**common, curve="p_rec")),
        _curve_table(p_l, _metadata(**common, curve="p_l")),
    )


# ---------------------------------------------------------------------------------------- CPI


def locked_in_cpi(cpi_act: Source, cpi_rec: Source, tau0: int, horizon: int) -> Any:
    """The locked-in CPI path at V (in-force 5.2): the actual history as known at V for u <= 0,
    ``I_L(u) = I_act(u)``, then the inflation expected at recognition from the actual level at
    V, ``I_L(u) = (I_act(0) x I_rec(tau0 + u)) / I_rec(tau0)`` for u = 1..horizon, left to right
    as the spec writes it. ``cpi_act`` holds months up to 0 (its first month is the path's);
    ``cpi_rec`` is indexed from the recognition month and covers ``tau0..tau0 + horizon``. One
    scenario, columns ``scenario``, ``u``, ``cpi``; metadata ``act_sha256``, ``rec_sha256``,
    ``estimated_rows`` (the actual file's, ``none`` when it flags none), ``tau0``,
    ``rebased_from`` (the recognition file) and ``helper``."""
    act, rec = _read(cpi_act, "cpi_act"), _read(cpi_rec, "cpi_rec")
    tau0 = _check_tau(tau0, "tau0")
    horizon = _check_tau(horizon, "horizon", lo=1)
    why = "the locked-in CPI path splices one actual history with one expected path"
    i_act = _by_month(act, "cpi", why)
    i_rec = _by_month(rec, "cpi", why)
    if not i_act or max(i_act) != 0:
        raise LoadError(f"{act.where}: the actual history ends at u = 0, the valuation month")
    first = min(i_act)
    _covering(act, i_act, first, 0, "the history")
    _covering(rec, i_rec, tau0, tau0 + horizon, "the splice")
    values = [i_act[u] for u in range(first, 1)]
    values += [(i_act[0] * i_rec[tau0 + u]) / i_rec[tau0] for u in range(1, horizon + 1)]
    n = len(values)
    meta = _metadata(
        act_sha256=act.sha256,
        rec_sha256=rec.sha256,
        estimated_rows=act.metadata.get("estimated_rows", "none"),
        tau0=str(tau0),
        rebased_from=rec.sha256,
        helper="locked_in_cpi",
    )
    return pa.table(
        {
            "scenario": pa.array([0] * n, pa.int32()),
            "u": pa.array(list(range(first, horizon + 1)), pa.int32()),
            "cpi": pa.array(values, pa.float64()),
        }
    ).replace_schema_metadata(meta)


# ---------------------------------------------------------------------------------- provenance


def _last_month(f: ReadFile) -> int:
    cols = table_columns(f)
    months = [name for name in ("t", "u") if name in cols]
    if len(months) != 1 or not len(cols[months[0]]):
        raise LoadError(f"{f.where}: one month column, t or u, with rows, is needed")
    return int(max(cols[months[0]].tolist()))


def _meta(f: ReadFile, key: str) -> str:
    value = f.metadata.get(key)
    if value is None:
        raise LoadError(
            f"{f.where}: carries no metadata halley.{key}, which its recomputation reads"
        )
    return value


def recompute_rebased(check: ProvenanceCheck) -> Any:
    """The recomputation registered for ``halley.rebased_from``: the file is made again, from
    the sources its metadata names by sha256 (found by ``check.find_file``), by the helper its
    ``halley.helper`` names, with the arguments its metadata and rows record (the expense
    table's tau from the two price bases, a curve's or a CPI path's horizon from its last
    month). The stage loader compares the result with the file, bit for bit on one machine and
    within the C library's last bits across machines (``LIBM_ULPS``)."""
    f = check.file
    made_by = f.metadata.get(HELPER)
    if made_by == "rebase_expense_table":
        source = check.find_file(check.value)
        price_base = _meta(f, PRICE_BASE)
        tau = _month(price_base, f.where) - _month(_meta(source, PRICE_BASE), source.where)
        act = check.find_file(_meta(f, "cpi_act_sha256"))
        return rebase_expense_table(source, act, tau, price_base)
    if made_by == "composed_curve":
        disc = check.find_file(_meta(f, "source_sha256"))
        ilp = check.find_file(_meta(f, "ilp_sha256"))
        return composed_curve(disc, ilp, _last_month(f))
    if made_by == "locked_in_curve":
        disc = check.find_file(_meta(f, "source_sha256"))
        ilp = check.find_file(_meta(f, "ilp_sha256"))
        tau0 = int(_meta(f, "tau0"))
        curve = _meta(f, "curve")
        if curve not in ("p_rec", "p_l"):
            raise LoadError(f"{f.where}: halley.curve is p_rec or p_l, not {curve!r}")
        last = _last_month(f)
        p_rec, p_l = locked_in_curve(disc, ilp, tau0, last if curve == "p_l" else last - tau0)
        return p_l if curve == "p_l" else p_rec
    if made_by == "locked_in_cpi":
        act = check.find_file(_meta(f, "act_sha256"))
        rec = check.find_file(_meta(f, "rec_sha256"))
        return locked_in_cpi(act, rec, int(_meta(f, "tau0")), _last_month(f))
    raise LoadError(
        f"{f.where}: carries halley.rebased_from, but its halley.{HELPER} is {made_by!r}, which "
        "names no helper of halley.aggregation.inforce; the file cannot be recomputed"
    )


def _members_at_v(check: ProvenanceCheck, ids: set[Any]) -> Any:
    """The model points at V of the policies ``ids``: the group key and the instalment of every
    row of the ``cur`` basis runs' model point copies (verified) whose policy is in ``ids``."""
    runs = check.bases.get(CUR_BASIS)
    if not runs:
        raise LoadError(
            f"{check.file.where}: the expected in-force takes the actual instalments at V from "
            f"the runs of basis {CUR_BASIS}, which the stage does not read"
        )
    names = ("policy_id", *GROUP_KEY, "instalment_pm")
    rows: dict[str, list[Any]] = {name: [] for name in names}
    for r in runs:
        entry = r.manifest["inputs"]["model_points"]
        f = read_copy(r.run_dir, entry, f"basis {CUR_BASIS} run {r.run_id}: model points")
        cols = table_columns(f)
        for name in names:
            if name not in cols:
                raise LoadError(f"{f.where}: column {name} is missing")
        values = {name: cols[name].tolist() for name in names}
        for j, pid in enumerate(values["policy_id"]):
            if pid in ids:
                for name in names:
                    rows[name].append(values[name][j])
    found = rows["policy_id"]
    id_type = pa.string() if any(isinstance(pid, str) for pid in found) else pa.int64()
    return pa.table(
        {
            "policy_id": pa.array(found, id_type),
            "portfolio": pa.array(rows["portfolio"], pa.string()),
            "cohort": pa.array(rows["cohort"], pa.int64()),
            "profit_bucket": pa.array(rows["profit_bucket"], pa.string()),
            "instalment_pm": pa.array(rows["instalment_pm"], pa.float64()),
        }
    )


def _month_of(d: dt.date) -> int:
    """``mi(d) = 12 * year(d) + month(d) - 1``."""
    return 12 * d.year + d.month - 1


def recompute_derived(check: ProvenanceCheck) -> Any:
    """The recomputation registered for ``halley.derived_from``: an ``OLD`` model point file is
    made again by :func:`expected_in_force` from the run its ``derived_from`` names (found by
    ``check.find_run``), its ``halley.tau`` and the actual instalments of the ``cur`` basis's
    model points at V. The file's own ``halley.tau`` is not taken on trust: it must be the
    period's length, ``mi(V) - mi(V_prev)`` with V the stage's valuation date and V_prev the
    named run's (in-force 3.4: the expected in-force at V), or a file built at another month
    would rebuild from its own metadata and pass."""
    f = check.file
    made_by = f.metadata.get(HELPER)
    if made_by != "expected_in_force":
        raise LoadError(
            f"{f.where}: carries halley.derived_from, but its halley.{HELPER} is {made_by!r}, "
            "which names no helper of halley.aggregation.inforce; the file cannot be recomputed"
        )
    prev = check.find_run(check.value)
    tau = int(_meta(f, "tau"))
    then = dt.date.fromisoformat(str(prev.manifest["settings"]["valuation_date"]))
    period = _month_of(check.valuation_date) - _month_of(then)
    if tau != period:
        raise LoadError(
            f"{f.where}: halley.tau is {tau}, but the run it is derived from, {prev.run_id}, "
            f"values at {then.isoformat()} and the stage at {check.valuation_date.isoformat()}, "
            f"{period} months later; the expected in-force at V is that run's pols_if at month "
            "tau of the period (in-force 3.4)"
        )
    return expected_in_force(prev, tau, _members_at_v(check, set(prev.ids.tolist())))


register_provenance("rebased_from", recompute_rebased)
register_provenance("derived_from", recompute_derived)

__all__ = [
    "check_membership",
    "composed_curve",
    "expected_in_force",
    "locked_in_cpi",
    "locked_in_curve",
    "rebase_expense_table",
    "recompute_derived",
    "recompute_rebased",
]
