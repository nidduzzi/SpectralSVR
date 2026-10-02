# code-prose-budget Specification

## Purpose
Explanatory prose carried in source -- comments, internal docstrings and docstrings outside the
numpy shape -- is measured per file and held to a recorded budget that only shrinks, so the rule
that intent lives in names and structure is checked rather than left to review.

## Requirements

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

### Requirement: A docstring has a shape that leaves no room for a derivation
Every docstring SHALL be one summary sentence, optionally followed by numpydoc
sections drawn from a closed set -- Parameters, Other Parameters, Receives,
Returns, Yields, Raises, Warns, Attributes, Methods, See Also and Examples --
with each entry's description at most one sentence. An Examples section SHALL
hold only doctest lines and their output. Free paragraphs and the Notes,
Warnings and References sections SHALL NOT appear. A derivation is a
chain of sentences reconstructing why a guarantee holds; the shape gives it
nowhere to go, so the reasoning lives in the change that owns it while the
docstring keeps the condition, stated in a Parameters or Raises entry. The shape
SHALL be checked by parsing structure and counting sentences, never by judging
what the text says. A docstring outside the shape SHALL count against its file's
budget like any other prose, so existing ones are recorded as debt that only
shrinks.

#### Scenario: A docstring outside the shape is refused
- **WHEN** a docstring carries a second sentence in its summary, a free
  paragraph, a section outside the closed set, an entry longer than one
  sentence, or prose inside an Examples section, in a file whose budget does not
  already count it
- **THEN** the guard fails, naming the file, so the docstring is reshaped and any
  reasoning it carried moves to the change that owns it

#### Scenario: A contract in the shape is accepted
- **WHEN** a docstring states its summary in one sentence and states each
  condition, parameter and refusal in its own one-sentence section entry
- **THEN** the guard accepts it, however much mathematics those sentences state,
  because the shape bounds the structure and not the content

### Requirement: The public interface is exempt and the exemption is stated
Docstring lines on the library's public interface SHALL NOT be counted against
the budget, because they are how a reader of the published work uses the
library. The exemption covers their length only: public docstrings SHALL be held
to the docstring shape like every other. The boundary between public and
internal SHALL be structural rather than a judgement recorded per name, so that
it cannot be argued case by case.

#### Scenario: Public and internal are told apart structurally
- **WHEN** the guard decides whether a docstring counts
- **THEN** it reads the name's own form and whether the package exports it, not
  an annotation or a list someone maintains

#### Scenario: The exemption is bounded
- **WHEN** an internal name carries a docstring
- **THEN** it counts against the budget, however useful it is, because the
  exemption covers the interface a caller outside the library meets and nothing
  further

#### Scenario: The exemption covers length, not shape
- **WHEN** a public docstring falls outside the docstring shape
- **THEN** it counts against its file's budget of misshapen docstrings, because
  the exemption spares the public interface a length limit and nothing else
