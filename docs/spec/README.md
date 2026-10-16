# Product specifications

Each spec here is a written product definition that a reference spreadsheet is built from by hand. A reviewer checks the spreadsheet against the spec, and the engine reconciles to the spreadsheet.

| File | What it is |
|---|---|
| `term-assurance.md` | Level term assurance, reference case 1 of the validation chain (spec, spreadsheet, reviewer check, engine reconciliation). |
| `term_assurance_numbers.py` | Regenerates every number printed in `term-assurance.md` and rechecks its invariants. Standard library only: `python docs/spec/term_assurance_numbers.py`. |
| `term_assurance_inputs.py` | Writes the reference inputs below from the spec's formulas (needs pyarrow and the package): `uv run python docs/spec/term_assurance_inputs.py`. |
| `term-assurance/` | The reference inputs as Parquet: the ten model points of section 13 and the five assumption tables of section 6.8. Their hashes appear in the manifest of every reconciliation run. |

The engine's model of the term assurance spec is `halley.products.term_assurance.TermAssurance`; `tests/test_term_assurance.py` reconciles it to the number script until the spreadsheet exists.

The design notes these specs depend on (Spec, Decisions, Roadmap) live in the author's vault, not in this repository.
