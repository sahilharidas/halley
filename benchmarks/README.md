# Benchmarks

The engine Spec's two workloads, timed on each backend:

| Workload | Size |
|---|---|
| W1, deterministic | 100k policies x 600 months |
| W2, stochastic | 1,000 scenarios x 10k policies x 600 months |

A fast backend justifies itself at 10x or more over the NumPy reference on W2 (Spec, Benchmarks).

The harness (`run_benchmarks.py`) and the measured results (`RESULTS.md`) are not here yet: they
come with milestone M6's benchmark stage, which replaces this note with how to run them. The
directory exists before them so that the lint and format commands over `src`, `tests`,
`docs/spec` and `benchmarks` have every path they name.
