# pyright: basic
"""Comparisons that allow for another machine's C library.

The committed reference files under ``docs/spec`` were written on the development laptop, and
the oracle scripts recompute their formulas on the machine the tests run on. Where a formula
takes ``exp`` or ``pow``, another C library (glibc against the Windows CRT) or NumPy's own
AVX-512 kernels may round a last bit the other way, and a few such bits compound through a
projection to about 1e-14 relative. On the machine that wrote a file the comparison is bit for
bit; elsewhere it is within 1e-12 relative, the tolerance the specs give two implementations of
one formula (annuity spec "Invariants and tests", in-force spec section 12), with an exact zero
on one side an exact zero on the other. A formula or convention difference shows at 1e-6 or
more, so the point of each check survives. pyarrow ships no type stubs, so by the convention of
the modules that read Arrow tables this one is checked in pyright's basic mode.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from halley.products import annuity_data as data

RELATIVE = 1e-12


def same_values(got: Any, want: Any) -> bool:
    """Equal bit for bit, or within :data:`RELATIVE` of the larger magnitude with every zero
    exact on both sides."""
    a = np.ascontiguousarray(np.asarray(got, dtype=np.float64))
    b = np.ascontiguousarray(np.asarray(want, dtype=np.float64))
    if a.shape != b.shape:
        return False
    if np.array_equal(a.view(np.uint64), b.view(np.uint64)):
        return True
    zero = (a == 0) | (b == 0)
    if not np.array_equal(a[zero], b[zero]):
        return False
    bound = RELATIVE * np.maximum(np.abs(a[~zero]), np.abs(b[~zero]))
    return bool((np.abs(a[~zero] - b[~zero]) <= bound).all())


def assert_generated(path: Path, table: Any, name: str) -> None:
    """The committed file at ``path`` is what its generator writes, ``table``: bit for bit on
    the machine that wrote it, where its bytes then hash as the generator's table does (a
    pyarrow upgrade may change the bytes without a value: rerun the script then); on another
    machine its float columns within :data:`RELATIVE`, every other column and the schema
    metadata exactly."""
    on_disk = pq.read_table(path)
    if on_disk.equals(table, check_metadata=True):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == data.sha256(table), name
        return
    assert on_disk.schema.equals(table.schema, check_metadata=True), name
    for column in table.column_names:
        x, y = on_disk.column(column), table.column(column)
        if pa.types.is_floating(x.type):
            assert same_values(x.to_numpy(), y.to_numpy()), (name, column)
        else:
            assert x.equals(y), (name, column)
