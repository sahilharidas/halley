# pyright: basic
"""The reference inputs the GUI offers as one-click examples, one set per shipped product.

Each example is the inputs a reference test or the tutorial runs the product on. The term
assurance and the annuities build theirs in memory from their specifications' formulas, so they
work from an installed package; lifelib's ``BasicTerm`` reads lifelib's inputs, which only a
checkout of the repository holds (``tests/data/lifelib``). pyarrow has no type stubs, so this
module is checked in pyright's basic mode.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

import halley
from halley.products import annuity, annuity_data, lifelib_basic_term, term_assurance
from halley.products import term_assurance_data as term_data

Inputs = tuple[Any, dict[str, Any], dict[str, Any]]
"""The model points, the tables by name and the scenario inputs by name."""

LIFELIB_FIXTURES = Path(halley.__file__).resolve().parents[2] / "tests" / "data" / "lifelib"
"""lifelib's inputs in a checkout of the repository: ``src/halley`` is two folders below it."""


@dataclass(frozen=True, slots=True)
class Example:
    """A set of inputs for one model and the settings its reference run uses."""

    id: str
    model: str
    """The id of the model the inputs are for, ``<module>:<qualname>``."""
    label: str
    description: str
    valuation_date: dt.date
    make: Callable[[], Inputs]
    batch_size: int = 10_000
    missing: Callable[[], str | None] = lambda: None
    """Why the example cannot run here, or None when it can."""

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "model": self.model,
            "label": self.label,
            "description": self.description,
            "valuation_date": self.valuation_date.isoformat(),
            "batch_size": self.batch_size,
            "missing": self.missing(),
        }


def _term() -> Inputs:
    return term_data.reference_model_points(), term_data.reference_tables(), {}


def _annuity(file: str) -> Callable[[], Inputs]:
    def make() -> Inputs:
        points = annuity_data.example_model_points()[file]
        return points, annuity_data.reference_tables(), annuity_data.reference_scenarios()

    return make


def _lifelib_missing() -> str | None:
    names = ("model_point_table", "premium_table", "mort_table", "disc_rate_ann")
    absent = [n for n in names if not (LIFELIB_FIXTURES / f"{n}.parquet").is_file()]
    if not absent:
        return None
    return (
        f"lifelib's inputs are not here ({LIFELIB_FIXTURES} lacks {', '.join(absent)}); they "
        "come with a checkout of the repository, not with the installed package"
    )


def _lifelib() -> Inputs:
    """lifelib's model points with lifelib's premium rate joined on, as its reconciliation test
    joins it (``premium_rates``), and lifelib's two tables."""
    mp = pq.read_table(LIFELIB_FIXTURES / "model_point_table.parquet")
    premiums = pq.read_table(LIFELIB_FIXTURES / "premium_table.parquet")
    rate = lifelib_basic_term.premium_rates(
        mp.column("age_at_entry").to_numpy(),
        mp.column("policy_term").to_numpy(),
        premiums.column("age_at_entry").to_numpy(),
        premiums.column("policy_term").to_numpy(),
        premiums.column("premium_rate").to_numpy(),
    )
    points = mp.append_column("premium_rate", pa.array(rate, pa.float64()))
    tables = {n: LIFELIB_FIXTURES / f"{n}.parquet" for n in ("mort_table", "disc_rate_ann")}
    return points, tables, {}


_ANNUITY_TEXT = (
    "the annuity spec's worked example: the six assumption tables of its reference basis and "
    "its discount and CPI scenario files (one scenario), built from the spec's formulas"
)

EXAMPLES: tuple[Example, ...] = (
    Example(
        "term-reference",
        "halley.products.term_assurance:TermAssurance",
        "Reference case 1",
        "The ten model points of the term assurance spec (section 13) on its synthetic basis "
        "(section 6), built from the spec's formulas; the tutorial's first run.",
        term_assurance.VALUATION_DATE,
        _term,
    ),
    Example(
        "annuity-level",
        "halley.products.annuity:LevelAnnuity",
        "Worked example: policies 1 and 4",
        f"The level annuities of {_ANNUITY_TEXT}.",
        annuity.VALUATION_DATE,
        _annuity("mp_level"),
    ),
    Example(
        "annuity-fixed",
        "halley.products.annuity:FixedEscalationAnnuity",
        "Worked example: policy 2",
        f"The fixed-escalation annuity of {_ANNUITY_TEXT}.",
        annuity.VALUATION_DATE,
        _annuity("mp_fixed"),
    ),
    Example(
        "annuity-cpi",
        "halley.products.annuity:CPILinkedAnnuity",
        "Worked example: policy 3",
        f"The CPI-linked annuity of {_ANNUITY_TEXT}.",
        annuity.VALUATION_DATE,
        _annuity("mp_cpi"),
    ),
    Example(
        "lifelib-basic-term",
        "halley.products.lifelib_basic_term:BasicTerm",
        "lifelib's BasicTerm_ME inputs",
        "lifelib's 10,000 model points with its premium table joined on, its mortality table and "
        "its discount curve (reference case 2), as the reconciliation test runs them.",
        lifelib_basic_term.VALUATION_DATE,
        _lifelib,
        batch_size=2000,
        missing=_lifelib_missing,
    ),
)


def examples_for(model_id: str) -> list[Example]:
    return [e for e in EXAMPLES if e.model == model_id]


def example(example_id: str) -> Example | None:
    return next((e for e in EXAMPLES if e.id == example_id), None)
