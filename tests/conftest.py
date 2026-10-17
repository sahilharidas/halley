"""Shared fixtures: the specs' number scripts as oracles, the reference inputs, one run."""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from halley import CompiledModel, RunResult, RunSettings, build, run
from halley.products.annuity import (
    CPILinkedAnnuity,
    FixedEscalationAnnuity,
    LevelAnnuity,
)
from halley.products.term_assurance import VALUATION_DATE, TermAssurance
from halley.products.term_assurance_data import reference_model_points, reference_tables

ROOT = Path(__file__).resolve().parent.parent
SPEC_DIR = ROOT / "docs" / "spec"
REFERENCE_INPUTS = SPEC_DIR / "term-assurance"
ANNUITY_INPUTS = SPEC_DIR / "annuity"

sys.path.insert(0, str(Path(__file__).parent))


def _spec_module(name: str) -> ModuleType:
    """A number script under ``docs/spec``, loaded by path (it is not a package)."""
    spec = importlib.util.spec_from_file_location(name, SPEC_DIR / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def oracle() -> ModuleType:
    """``docs/spec/term_assurance_numbers.py``: the spec's formulas in plain float64 scalars.
    Not the reference (the spreadsheet is), but the best check available before it exists."""
    return _spec_module("term_assurance_numbers")


@pytest.fixture(scope="session")
def inforce_oracle() -> ModuleType:
    """``docs/spec/in_force_group_numbers.py``: the in-force group spec's formulas, whose
    projection of one policy (``project``) restates the annuity spec's model and reproduces its
    worked example. The annuity's oracle until its spreadsheet exists."""
    return _spec_module("in_force_group_numbers")


@pytest.fixture(scope="session")
def compiled_term() -> CompiledModel:
    return build(TermAssurance)


@pytest.fixture(scope="session")
def compiled_annuity() -> dict[str, CompiledModel]:
    """The three annuity classes, built with every build warning an error (no variable of the
    reference model is unused)."""
    classes = {"level": LevelAnnuity, "fixed": FixedEscalationAnnuity, "cpi": CPILinkedAnnuity}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return {name: build(cls) for name, cls in classes.items()}


@pytest.fixture(scope="session")
def reference_sources() -> dict[str, Any]:
    """The reference inputs as in-memory Arrow tables."""
    return {"model_points": reference_model_points(), "tables": reference_tables()}


@pytest.fixture(scope="session")
def term_run(
    compiled_term: CompiledModel, reference_sources: dict[str, Any], tmp_path_factory: Any
) -> RunResult:
    """One run of the reference case from the committed Parquet files, batch of ten."""
    tables = {name: REFERENCE_INPUTS / f"{name}.parquet" for name in reference_sources["tables"]}
    return run(
        compiled_term,
        model_points=REFERENCE_INPUTS / "model_points.parquet",
        tables=tables,
        settings=RunSettings(valuation_date=VALUATION_DATE, label="reference case 1"),
        run_dir=tmp_path_factory.mktemp("runs") / "term-reference",
    )
