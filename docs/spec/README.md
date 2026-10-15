# Product specifications

Each spec here is a written product definition that a reference spreadsheet is built from by hand. A reviewer checks the spreadsheet against the spec, and the engine reconciles to the spreadsheet.

| File | What it is |
|---|---|
| `term-assurance.md` | Level term assurance, reference case 1 of the validation chain (spec, spreadsheet, reviewer check, engine reconciliation). |
| `term_assurance_numbers.py` | Regenerates every number printed in `term-assurance.md` and rechecks its invariants. Standard library only: `python docs/spec/term_assurance_numbers.py`. |

The design notes these specs depend on (Spec, Decisions, Roadmap) live in the author's vault, not in this repository.
