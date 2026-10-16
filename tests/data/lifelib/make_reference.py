# pyright: basic, reportMissingImports=false, reportMissingModuleSource=false
"""Regenerate the lifelib fixtures of reference case 2 (tests/data/lifelib).

Reads lifelib's own BasicTerm_ME inputs from the installed ``lifelib`` package, writes them as
Parquet in Halley's table layout, runs BasicTerm_ME with modelx, and writes its results:

* ``model_point_table.parquet``, ``mort_table.parquet``, ``disc_rate_ann.parquet`` and
  ``premium_table.parquet``: lifelib's inputs, values unchanged (the mortality table is unpivoted
  to one row per age and duration, the premium table's multi-index is written as two columns);
* ``results_pv.parquet``: lifelib's per-policy scalars for every model point;
* ``results_monthly.parquet``: lifelib's monthly results for a fixed sample of model points, for
  lifelib's months ``t = 0 .. 12 * policy_term - duration_mth``;
* ``results_totals.parquet``: lifelib's ``result_cf()`` and ``result_pols()`` portfolio totals by
  month, with the discount and inflation factors.

Needs lifelib, modelx, pandas and openpyxl (``uv sync --group lifelib``). lifelib is MIT; modelx
is GPL-3.0 and is used here only to produce the fixtures, which the test suite then reads without
it. Usage: ``python tests/data/lifelib/make_reference.py [--out DIR]``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import sys
import time
from pathlib import Path

import lifelib
import modelx as mx
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

MODEL = "BasicTerm_ME"
LIFELIB_VERSION = importlib.metadata.version("lifelib")
SAMPLE_STEP = 200
PV_CELLS = (
    "pv_premiums",
    "pv_claims",
    "pv_expenses",
    "pv_commissions",
    "pv_net_cf",
    "pv_pols_if",
    "net_premium_pp",
)
MONTHLY_CELLS = (
    "premiums",
    "claims",
    "expenses",
    "commissions",
    "net_cf",
    "pols_if",
    "pols_maturity",
    "pols_new_biz",
    "pols_death",
    "pols_lapse",
    "mort_rate_mth",
    "lapse_rate",
)


def _meta(**extra: str) -> dict[bytes, bytes]:
    meta = {
        b"halley.source": f"lifelib {LIFELIB_VERSION} basiclife/{MODEL} (MIT)".encode(),
    }
    meta.update({f"halley.{k}".encode(): v.encode() for k, v in extra.items()})
    return meta


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_inputs(model_dir: Path, out: Path) -> dict[str, str]:
    """lifelib's four input workbooks as Parquet. Returns the sha256 of each source workbook."""
    version = f"lifelib-{LIFELIB_VERSION}"
    hashes: dict[str, str] = {}

    src = model_dir / "model_point_table.xlsx"
    mp = pd.read_excel(src, index_col=0)
    hashes[src.name] = _sha(src)
    table = pa.table(
        {
            "policy_id": pa.array(mp.index.to_numpy(), pa.int64()),
            "age_at_entry": pa.array(mp["age_at_entry"].to_numpy(), pa.int64()),
            "sex": pa.array(mp["sex"].tolist(), pa.string()),
            "policy_term": pa.array(mp["policy_term"].to_numpy(), pa.int64()),
            "policy_count": pa.array(mp["policy_count"].to_numpy(), pa.int64()),
            "sum_assured": pa.array(mp["sum_assured"].to_numpy(), pa.int64()),
            "duration_mth": pa.array(mp["duration_mth"].to_numpy(), pa.int64()),
        }
    ).replace_schema_metadata(_meta(version=version, file=src.name))
    pq.write_table(table, out / "model_point_table.parquet")

    src = model_dir / "mort_table.xlsx"
    mt = pd.read_excel(src, index_col=0)
    hashes[src.name] = _sha(src)
    ages = [int(a) for a in mt.index for _ in mt.columns]
    durs = [int(c) for _ in mt.index for c in mt.columns]
    rates = [float(mt.loc[a, c]) for a in mt.index for c in mt.columns]
    table = pa.table(
        {
            "age": pa.array(ages, pa.int64()),
            "duration": pa.array(durs, pa.int64()),
            "mort_rate": pa.array(rates, pa.float64()),
        }
    ).replace_schema_metadata(_meta(version=version, file=src.name))
    pq.write_table(table, out / "mort_table.parquet")

    src = model_dir / "disc_rate_ann.xlsx"
    dr = pd.read_excel(src, index_col=0)
    hashes[src.name] = _sha(src)
    table = pa.table(
        {
            "year": pa.array(dr.index.to_numpy(), pa.int64()),
            "zero_spot": pa.array(dr["zero_spot"].to_numpy(), pa.float64()),
        }
    ).replace_schema_metadata(_meta(version=version, file=src.name))
    pq.write_table(table, out / "disc_rate_ann.parquet")

    src = model_dir / "premium_table.xlsx"
    pt = pd.read_excel(src, index_col=[0, 1])
    hashes[src.name] = _sha(src)
    table = pa.table(
        {
            "age_at_entry": pa.array(pt.index.get_level_values(0).to_numpy(), pa.int64()),
            "policy_term": pa.array(pt.index.get_level_values(1).to_numpy(), pa.int64()),
            "premium_rate": pa.array(pt["premium_rate"].to_numpy(), pa.float64()),
        }
    ).replace_schema_metadata(_meta(version=version, file=src.name))
    pq.write_table(table, out / "premium_table.parquet")
    return hashes


def sample_ids(mp: pd.DataFrame) -> list[int]:
    """Every SAMPLE_STEP-th policy plus the edge cases: the earliest future new business, a
    policy issued at t = 0, one maturing at t = 0, one with no policies, the oldest at the
    horizon and the one with the longest projection."""
    ids = list(mp.index[::SAMPLE_STEP])
    rem = 12 * mp["policy_term"] - mp["duration_mth"]
    extras = [
        mp["duration_mth"].idxmin(),
        mp.index[mp["duration_mth"] == 0][0],
        mp.index[rem == 0][0],
        mp.index[mp["policy_count"] == 0][0],
        (mp["age_at_entry"] + mp["policy_term"]).idxmax(),
        rem.idxmax(),
    ]
    for pid in extras:
        if int(pid) not in ids:
            ids.append(int(pid))
    return sorted(int(i) for i in ids)


def run_lifelib(model_dir: Path, out: Path) -> None:
    t0 = time.perf_counter()
    model = mx.read_model(str(model_dir))
    p = model.Projection
    mp = p.model_point()
    ids = np.asarray(mp.index.to_numpy(), dtype=np.int64)
    n = int(p.max_proj_len())
    proj_len = np.asarray(p.proj_len(), dtype=np.int64)

    # per-policy scalars
    scalars: dict[str, object] = {"policy_id": pa.array(ids, pa.int64())}
    for cell in PV_CELLS:
        scalars[cell] = pa.array(np.asarray(getattr(p, cell)(), dtype=np.float64), pa.float64())
    scalars["premium_pp"] = pa.array(np.asarray(p.premium_pp(), dtype=np.float64), pa.float64())
    scalars["proj_len"] = pa.array(proj_len, pa.int64())
    pq.write_table(
        pa.table(scalars).replace_schema_metadata(_meta(months=str(n))),
        out / "results_pv.parquet",
    )
    print(f"scalars for {len(ids)} policies, max_proj_len {n}: {time.perf_counter() - t0:.1f}s")

    # monthly results for the sample, lifelib months 0 .. proj_len - 1
    sample = sample_ids(mp)
    pos = {int(pid): i for i, pid in enumerate(ids)}
    cols: dict[str, list[object]] = {"policy_id": [], "t": []}
    for cell in MONTHLY_CELLS:
        cols[cell] = []
    cols["pols_if_bef_decr"] = []
    monthly = {
        cell: [np.asarray(getattr(p, cell)(t), dtype=np.float64) for t in range(n)]
        for cell in MONTHLY_CELLS
    }
    monthly["pols_if_bef_decr"] = [
        np.asarray(p.pols_if_at(t, "BEF_DECR"), dtype=np.float64) for t in range(n)
    ]
    for pid in sample:
        i = pos[pid]
        for t in range(int(proj_len[i])):
            cols["policy_id"].append(pid)
            cols["t"].append(t)
            for cell, series in monthly.items():
                cols[cell].append(float(series[t][i]))
    arrays: dict[str, object] = {
        "policy_id": pa.array(cols["policy_id"], pa.int64()),
        "t": pa.array(cols["t"], pa.int64()),
    }
    for cell in monthly:
        arrays[cell] = pa.array(cols[cell], pa.float64())
    pq.write_table(
        pa.table(arrays).replace_schema_metadata(_meta(sample=",".join(str(s) for s in sample))),
        out / "results_monthly.parquet",
    )
    print(f"monthly results for {len(sample)} sampled policies: {time.perf_counter() - t0:.1f}s")

    # portfolio totals by month, lifelib's own result tables
    cf = p.result_cf()
    pols = p.result_pols()
    totals: dict[str, object] = {"t": pa.array(np.arange(n), pa.int64())}
    for name, col in (
        ("premiums", "Premiums"),
        ("claims", "Claims"),
        ("expenses", "Expenses"),
        ("commissions", "Commissions"),
        ("net_cf", "Net Cashflow"),
    ):
        totals[name] = pa.array(np.asarray(cf[col], dtype=np.float64), pa.float64())
    for name in ("pols_if", "pols_maturity", "pols_new_biz", "pols_death", "pols_lapse"):
        totals[name] = pa.array(np.asarray(pols[name], dtype=np.float64), pa.float64())
    totals["disc_factors"] = pa.array(np.asarray(p.disc_factors(), dtype=np.float64)[:n])
    totals["inflation_factor"] = pa.array(
        [float(p.inflation_factor(t)) for t in range(n)], pa.float64()
    )
    pq.write_table(
        pa.table(totals).replace_schema_metadata(_meta()), out / "results_totals.parquet"
    )
    print(f"totals: {time.perf_counter() - t0:.1f}s")
    print("pv_net_cf total", float(np.sum(np.asarray(p.pv_net_cf()))))


def model_dir() -> Path:
    """lifelib's BasicTerm_ME folder inside the installed package."""
    file = lifelib.__file__
    assert file is not None, "lifelib is not installed as a package here"
    return Path(file).resolve().parent / "libraries" / "basiclife" / MODEL


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate the lifelib fixtures of reference case 2 (tests/data/lifelib)."
    )
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    folder = model_dir()
    hashes = write_inputs(folder, out)
    for name, digest in hashes.items():
        print(f"{name}: sha256 {digest}")
    run_lifelib(folder, out)
    print(
        f"lifelib {LIFELIB_VERSION}, modelx {mx.__version__}, numpy {np.__version__}, "
        f"pandas {pd.__version__}, python {sys.version.split()[0]}"
    )


if __name__ == "__main__":
    main()
