"""Runtime: load and validate data, run batches, write outputs and the run manifest, and
read runs back; trace, explain and diff."""

from halley.runtime.explain import Difference, Explanation, diff, explain, recompute
from halley.runtime.invariants import InvariantResult
from halley.runtime.loader import Inputs, load_inputs, load_model_points, load_table
from halley.runtime.run import OUTPUT_SCHEMA_VERSION, RunResult, RunSettings, load_run, run
from halley.runtime.tolerance import RELATIVE, all_within, within, worst_residual

__all__ = [
    "OUTPUT_SCHEMA_VERSION",
    "RELATIVE",
    "Difference",
    "Explanation",
    "Inputs",
    "InvariantResult",
    "RunResult",
    "RunSettings",
    "all_within",
    "diff",
    "explain",
    "load_inputs",
    "load_model_points",
    "load_run",
    "load_table",
    "recompute",
    "run",
    "within",
    "worst_residual",
]
