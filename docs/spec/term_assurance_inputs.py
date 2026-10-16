"""Write the reference inputs of docs/spec/term-assurance.md as Parquet files (C37).

The ten model points of section 13 and the five assumption tables of section 6.8 go to
``docs/spec/term-assurance/``. Their hashes appear in the manifest of every reconciliation run.
Needs pyarrow and the halley package:

    uv run python docs/spec/term_assurance_inputs.py
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pyarrow.parquet as pq

from halley.products.term_assurance_data import reference_model_points, reference_tables

OUT = Path(__file__).with_name("term-assurance")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    files = {"model_points": reference_model_points(), **reference_tables()}
    for name, table in files.items():
        path = OUT / f"{name}.parquet"
        pq.write_table(table, path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{path.relative_to(OUT.parent)}: {table.num_rows} rows, sha256 {digest[:16]}")


if __name__ == "__main__":
    main()
