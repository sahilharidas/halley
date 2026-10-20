"""Write the reference inputs of docs/spec/in-force-group.md as Parquet files.

The model point files of each variant (P and O) and class at recognition and at V, the groups
rows at recognition and at V (with the three locked-in hashes), the assumption tables of
versions 1 and 2, the curves and CPI paths of section 3.6, and the files the cross-run helpers
build from them (the expense table rebased to V prices, the composed curve at recognition and
the locked-in curve re-based to V, the locked-in CPI splice, the composed current curve) go to
``docs/spec/in-force-group/`` beside the workbooks. Each file is written from the bytes the
loader hashes for an in-memory table, so a hash one file records for another (a locked-in hash,
``rebased_from``) is the hash of the file beside it. The groups rows at V leave out what the
recognition run supplies (``prev_run_id``, ``csm_prev``, ``lc_prev``): those are read from that
run, never retyped (``halley.products.annuity_data.with_previous_run``). Needs pyarrow and the
halley package:

    uv run --frozen python docs/spec/in_force_group_inputs.py
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from halley.products.annuity_data import in_force_files, parquet_bytes

OUT = Path(__file__).with_name("in-force-group")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name, table in in_force_files().items():
        path = OUT / f"{name}.parquet"
        raw = parquet_bytes(table)
        path.write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        print(f"{path.relative_to(OUT.parent)}: {table.num_rows} rows, sha256 {digest[:16]}")


if __name__ == "__main__":
    main()
