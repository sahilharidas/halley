"""M0: the package installs, imports and reports one version."""

import importlib
import importlib.metadata

import pytest

import halley

PIPELINE_STAGES = [
    "authoring",
    "tracing",
    "ir",
    "checker",
    "scheduler",
    "codegen",
    "runtime",
    "aggregation",
]


def test_version_matches_installed_metadata() -> None:
    assert halley.__version__ == importlib.metadata.version("halley")


@pytest.mark.parametrize("stage", PIPELINE_STAGES)
def test_pipeline_stage_imports(stage: str) -> None:
    module = importlib.import_module(f"halley.{stage}")
    assert module.__doc__, f"halley.{stage} should say what it is for"
