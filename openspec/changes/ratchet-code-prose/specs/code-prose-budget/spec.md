## ADDED Requirements

### Requirement: Prose in code is measured and bounded per file
Explanatory prose carried in source — comment lines and the docstrings of
internal names — SHALL be counted per file and held against a recorded budget.
A file SHALL NOT exceed its budget. A file with no recorded budget SHALL be
treated as having a budget of zero, so new files start clean and cannot inherit
an allowance nobody granted.

The count SHALL exclude machine-directed comments, which carry instructions to a
tool rather than prose to a reader, and SHALL exclude the docstrings of names
that form the library's public interface.

#### Scenario: A file over its budget fails
- **WHEN** a file carries more counted prose lines than its recorded budget
- **THEN** the guard fails, naming the file, the budget and the count

#### Scenario: A new file starts at zero
- **WHEN** a source file with no entry in the budget gains a comment or an
  internal docstring
- **THEN** the guard fails, because an unlisted file is budgeted at zero rather
  than unbounded

#### Scenario: Machine-directed comments are not prose
- **WHEN** a file carries `noqa`, type-checker, linter or interpreter directives
  in comment form
- **THEN** those lines are not counted, because they instruct a tool and removing
  them changes behaviour

### Requirement: The budget only shrinks
A recorded budget SHALL be debt, not an allowance. Raising any file's budget
SHALL require the same deliberate act as any other recorded decision, and the
guard SHALL make a budget that has drifted above its file's actual count
visible so that slack is reclaimed rather than banked.

#### Scenario: Slack is reclaimed, not banked
- **WHEN** a file's counted prose falls below its recorded budget
- **THEN** the guard reports the slack, so the budget is lowered to what the file
  now carries rather than left as room for future prose

#### Scenario: The budget is per file, not a pooled total
- **WHEN** prose is added to one file and removed from another
- **THEN** the addition still fails its own file's budget, because a repository
  total would let one file's cleanup pay for another file's narration

### Requirement: A docstring states a contract, not a derivation
A docstring SHALL state what a name guarantees, not re-derive why the guarantee
holds. Where reasoning belongs to a change, it belongs in that change's planning
artifacts, which are versioned with the decision and reviewed as a claim; a
derivation copied into a docstring is a second, unchecked copy that drifts from
the first.

#### Scenario: A derivation in a docstring is refused
- **WHEN** a docstring carries the arithmetic of a derivation rather than the
  condition it produces
- **THEN** the guard fails, naming the docstring, so the reasoning moves to the
  change that owns it and the docstring keeps the condition

#### Scenario: Stating a condition is not deriving it
- **WHEN** a docstring names a bound, a condition or a relationship that the
  caller must satisfy, without reconstructing the argument for it
- **THEN** the guard accepts it, because that is the contract the caller needs

### Requirement: The public interface is exempt and the exemption is stated
Docstrings on the library's public interface SHALL NOT be counted against the
budget, because they are how a reader of the published work uses the library.
The boundary between public and internal SHALL be structural rather than a
judgement recorded per name, so that it cannot be argued case by case.

#### Scenario: Public and internal are told apart structurally
- **WHEN** the guard decides whether a docstring counts
- **THEN** it reads the name's own form and whether the package exports it, not
  an annotation or a list someone maintains

#### Scenario: The exemption is bounded
- **WHEN** an internal name carries a docstring
- **THEN** it counts against the budget, however useful it is, because the
  exemption covers the interface a caller outside the library meets and nothing
  further
