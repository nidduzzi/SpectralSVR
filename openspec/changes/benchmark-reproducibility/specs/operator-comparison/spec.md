## ADDED Requirements

### Requirement: Results are reported over repeated seeds
The comparison SHALL be run over a fixed, recorded set of seeds, and SHALL report a centre and a spread for every metric rather than a single draw. A reader SHALL be able to tell whether a difference between two operators is larger than the variation between runs of the same operator.

#### Scenario: Spread accompanies every metric
- **WHEN** the comparison table is produced
- **THEN** each metric carries both a centre and a measure of spread across the seed set, and the seeds used are recorded

#### Scenario: A single-seed table is not presented as a result
- **WHEN** the benchmark is run at one seed, for a quick check
- **THEN** its output is marked as such and is not the artifact the documentation points a reader at

### Requirement: Every run records what produced it
Each set of results SHALL carry the provenance needed to reproduce or challenge it: the device each stage ran on, the coefficient precision, the versions of the numerical libraries used, and the revision of the code.

#### Scenario: Provenance travels with the numbers
- **WHEN** a results table is read
- **THEN** the device each stage ran on, the precision, library versions and code revision that produced it are available from the same artifact, not from memory

#### Scenario: A changed condition is visible
- **WHEN** the same benchmark is run at a different precision or on a different device
- **THEN** the two result sets are distinguishable by their recorded provenance rather than only by their numbers

### Requirement: The error statistic is unambiguous
Where an error statistic has more than one convention in the literature, the table SHALL report the convention a reader is likely to assume alongside the one the library computes, and SHALL name each so they cannot be confused.

#### Scenario: Both relative-error conventions appear
- **WHEN** the table reports relative error
- **THEN** it carries both the per-sample form averaged over the test set, which neural-operator papers use, and the global form the library computes over all elements at once, each under a distinct name

#### Scenario: Neither replaces the other
- **WHEN** the additional statistic is introduced
- **THEN** the existing metric definitions are unchanged, so previously reported values remain comparable

### Requirement: The timing column states what it measures
Fit time SHALL be reported with an explicit statement of what work it covers for each operator, because a direct solve and a fixed-iteration training loop are not comparable quantities even when both are the time spent in fitting.

#### Scenario: Times are qualified where reported
- **WHEN** fit times for two operators are shown together
- **THEN** the accompanying documentation says what differs between them beyond the method, so the column is not read as a like-for-like cost comparison

### Requirement: Regeneration is a single documented command
Reproducing the table SHALL be one recorded command, and the analysis that presents the results SHALL read the committed table rather than recomputing it, so that presentation and record cannot drift apart.

#### Scenario: One command regenerates
- **WHEN** a reader wants to reproduce the table
- **THEN** a single documented command does it, with the seed set and datasets as defaults rather than as arguments to be remembered

#### Scenario: The notebook reads the record
- **WHEN** the comparison notebook is run
- **THEN** it loads the committed results rather than recomputing them, so what is shown is what was recorded
