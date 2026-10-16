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
