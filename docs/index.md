# Halley

Actuarial projection engine in Python. Models are checked before they run and compiled when they do.

A model is a Python class. Building it traces the formulas into a typed dependency graph, checks
units, time references and calculation order, then generates vectorised code that runs over every
policy and scenario. Every run writes a manifest so any result can be reproduced exactly.

**Status:** prototype, milestone M0 (repository skeleton). Nothing runs a model yet. What is here
so far is the written specification of the first reference model.

## Install

```bash
pip install halley
```

Python 3.12 or later. Licensed under Apache 2.0.

```{toctree}
:maxdepth: 2
:caption: Specifications

spec/README
spec/term-assurance
```
