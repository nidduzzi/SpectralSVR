## Purpose

Example notebooks show how the library is used. This capability makes sure every committed example still runs against the current code and carries no stale outputs that could be mistaken for results.

## ADDED Requirements

### Requirement: Every example runs in the suite on CPU
Every notebook under the examples directory SHALL be executed from a clean kernel by the test suite, with no accelerator visible to the kernel. A notebook whose execution raises SHALL fail the suite.

#### Scenario: An example breaks against the library
- **WHEN** a change to the library makes a cell of an example notebook raise
- **THEN** the suite fails, naming the notebook and the cell

#### Scenario: An example needs an accelerator
- **WHEN** an example requires a GPU
- **THEN** it fails in the suite, because its kernel sees no accelerator

### Requirement: Examples are committed without outputs
No notebook under the examples directory SHALL be committed with cell outputs or execution counts.

#### Scenario: A notebook with outputs is committed
- **WHEN** an example notebook in the tree carries a cell output or an execution count
- **THEN** the suite fails, naming the notebook

### Requirement: Notebooks live only under the examples directory
The tracked tree SHALL contain notebooks only under the examples directory.

#### Scenario: A notebook is added elsewhere
- **WHEN** a notebook is tracked outside the examples directory
- **THEN** the suite fails, naming its path
