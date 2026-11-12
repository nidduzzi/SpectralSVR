## Purpose

Mutation testing shows whether the suite would notice a change to the code. This capability fixes how a mutant judged equivalent is declared, so that each judgement is recorded once, with its reason, and is refused as soon as it stops describing the code.

## Requirements

### Requirement: An equivalent mutant is declared with its reason
A mutant judged equivalent SHALL be declared in one tracked registry. Each entry SHALL name the module, the enclosing definition, the source line it mutates, that line as the mutant rewrites it, and a reason. The entry SHALL NOT be keyed by line number, and SHALL match exactly one mutant. A mutation run SHALL skip every declared mutant and SHALL report how many it skipped.

#### Scenario: A declared mutant is skipped
- **WHEN** a mutation run enumerates a mutant whose module, definition, source line and rewritten line match a registry entry
- **THEN** the run skips that mutant, and its report counts it among the skipped

#### Scenario: Other mutants on the same line still run
- **WHEN** a registry entry declares one mutant of a line that has other mutants
- **THEN** only the declared mutant is skipped, and the others run

#### Scenario: An entry without a reason is refused
- **WHEN** a registry entry has an empty reason
- **THEN** the suite fails, naming the entry

### Requirement: A declaration that no longer matches the source is refused
The suite SHALL fail when a registry entry matches no mutation of the current source, so that an edit to a declared line invalidates its judgement instead of silently keeping it.

#### Scenario: The declared line was edited
- **WHEN** the source line named by a registry entry no longer appears in its module
- **THEN** the suite fails, naming the entry and its module

#### Scenario: The declared mutant no longer arises
- **WHEN** the declared line still appears but no mutation of it reads as the declared mutant
- **THEN** the suite fails, naming the entry
