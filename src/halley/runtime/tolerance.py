"""The reconciliation and invariant tolerance (term spec section 1, C33).

Two float64 values agree when ``|a - b| <= 1e-9 * max(|a|, |b|, scale)``, with ``scale`` the
natural magnitude of the quantity's unit: for a policy, ``sum_assured * pols_if_init`` for in-force
money, ``sum_assured`` per policy, ``pols_if_init`` for counts and 1 for dimensionless and
integer quantities. A model declares the scale fields in ``tolerance_scales``; a portfolio total
takes the sum of the per-policy scales.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

RELATIVE = 1e-9


def within(a: Any, b: Any, scale: Any = 1.0, rel: float = RELATIVE) -> NDArray[np.bool_]:
    """Elementwise agreement under the rule above. Broadcasts."""
    a_ = np.asarray(a, dtype=np.float64)
    b_ = np.asarray(b, dtype=np.float64)
    scale_ = np.asarray(scale, dtype=np.float64)
    bound = rel * np.maximum(np.maximum(np.abs(a_), np.abs(b_)), scale_)
    return np.asarray(np.abs(a_ - b_) <= bound)


def all_within(a: Any, b: Any, scale: Any = 1.0, rel: float = RELATIVE) -> bool:
    return bool(within(a, b, scale, rel).all())


def worst_residual(a: Any, b: Any) -> float:
    """The largest absolute difference, for reports."""
    diff = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
    return float(diff.max()) if diff.size else 0.0


LIBM_ULPS = 8
"""How far apart, in units in the last place, two float64 evaluations of one formula may be
when only the C library's ``exp``, ``log`` and ``pow`` differ between them. glibc, the Windows
CRT and NumPy's own AVX-512 kernels each stay within an ulp of the true value, so two of them
differ by at most an ulp per transcendental, and each product or ratio formed from the results
adds at most an ulp of rounding: a composed curve is one ``exp`` and one product, a locked-in
curve the ratio of two such values, four ulps at most, held to eight (about 2e-15 relative). A
file built from another source, at another month or with another illiquidity premium is off by
orders of magnitude more, so a check within this envelope still tells a wrong file from a right
one built on another machine. On one machine with one NumPy reruns stay bit-identical (D19);
the envelope is for a file that crosses machines."""


def ulps_apart(a: Any, b: Any) -> NDArray[np.float64]:
    """Elementwise distance between two float64 arrays in units in the last place of the larger
    magnitude: 0 where the 64-bit patterns are equal (so a NaN against the same NaN is 0),
    ``|a - b| / spacing(max(|a|, |b|))`` where both are finite (so ``-0.0`` against ``0.0`` is
    0), and infinite where either is not finite. Broadcasts."""
    a_, b_ = np.broadcast_arrays(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))
    same = a_.view(np.uint64) == b_.view(np.uint64)
    finite = np.isfinite(a_) & np.isfinite(b_)
    with np.errstate(all="ignore"):
        apart = np.abs(a_ - b_) / np.spacing(np.maximum(np.abs(a_), np.abs(b_)))
    return np.asarray(np.where(same, 0.0, np.where(finite, apart, np.inf)), dtype=np.float64)


def within_ulps(a: Any, b: Any, ulps: int = LIBM_ULPS) -> NDArray[np.bool_]:
    """Elementwise agreement within ``ulps`` units in the last place (:func:`ulps_apart`):
    bit-for-bit values always agree, and ``ulps = 0`` is bit-for-bit agreement. Broadcasts."""
    return np.asarray(ulps_apart(a, b) <= ulps)
