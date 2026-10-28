"""Write the reference inputs of docs/spec/annuity.md as Parquet files.

The three class files of the worked example (``mp_level``, ``mp_fixed``, ``mp_cpi``), the six
assumption tables the model reads and the improvement table ``mort_cif`` is built from, the
discount file and the CPI file go to ``docs/spec/annuity/``. Each file is written from the same
bytes the loader hashes for an in-memory table, so the hash recorded in ``mort_cif``'s metadata
for ``mort_improvement`` is the hash of the file beside it. Their hashes appear in the manifest
of every reconciliation run. Needs pyarrow and the halley package:

    uv run --frozen python docs/spec/annuity_inputs.py
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from halley.products.annuity_data import (
    example_model_points,
    mort_improvement_table,
    parquet_bytes,
    reference_scenarios,
    reference_tables,
)

OUT = Path(__file__).with_name("annuity")

SCENARIO_FILES = {"rf_df": "discount", "cpi": "cpi"}
"""The scenario inputs' file names: the discount file holds the value column ``rf_df``."""


def main() -> None:
    OUT.mkdir(exist_ok=True)
    files = {
        **example_model_points(),
        **reference_tables(),
        "mort_improvement": mort_improvement_table(),
        **{SCENARIO_FILES[name]: table for name, table in reference_scenarios().items()},
    }
    for name, table in files.items():
        path = OUT / f"{name}.parquet"
        raw = parquet_bytes(table)
        path.write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        print(f"{path.relative_to(OUT.parent)}: {table.num_rows} rows, sha256 {digest[:16]}")


if __name__ == "__main__":
    main()
