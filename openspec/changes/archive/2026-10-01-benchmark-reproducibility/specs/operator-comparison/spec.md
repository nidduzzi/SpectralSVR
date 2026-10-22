## ADDED Requirements

### Requirement: Every run records what produced it
Each set of results SHALL carry the provenance needed to reproduce or challenge it: the device each stage ran on, the coefficient precision, the versions of the numerical libraries used, and the revision of the code.

#### Scenario: Provenance travels with the numbers
- **WHEN** a results table is read
- **THEN** the device each stage ran on, the precision, library versions and code revision that produced it are available from the same artifact, not from memory

#### Scenario: Every stage runs where the run says
- **WHEN** a run states a device
- **THEN** every operator fits on that device, whatever the process default names after an optional baseline's import, and each row's recorded fit device equals its data device

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

### Requirement: Regeneration is a single documented command
Reproducing the table SHALL be one recorded command, and the analysis that presents the results SHALL read the committed table rather than recomputing it, so that presentation and record cannot drift apart.

#### Scenario: One command regenerates
- **WHEN** a reader wants to reproduce the table
- **THEN** a single documented command does it, with the seed set and datasets as defaults rather than as arguments to be remembered

#### Scenario: The notebook reads the record
- **WHEN** the comparison notebook is run
- **THEN** it loads the committed results rather than recomputing them, so what is shown is what was recorded

### Requirement: A baseline is fitted reproducibly and by its reference protocol
Each baseline SHALL be fitted so that the same seed and the same data give the same predictions wherever the fit falls in a run. It SHALL follow the training protocol of the reference it is compared as, and the table SHALL record that protocol with the results, so a departure from the reference is stated rather than silent.

#### Scenario: A fit does not depend on its place in the run
- **WHEN** a baseline is fitted with the same seed on the same data, once as the first fit in a process and once after other fits have run
- **THEN** its predictions are identical, because the seed governs every random draw the fit makes, its initial weights included

#### Scenario: The reference protocol is followed and recorded
- **WHEN** DeepONet is fitted for the table
- **THEN** it is trained by a named recipe taken from a reference implementation for that kind of problem, and the table records the recipe, the source it was taken from, and each stated departure from it, such as a shorter training length or keeping a simpler recipe where the reference's measured worse

#### Scenario: A baseline predicts where its query is
- **WHEN** a fitted baseline is queried at points on a stated device
- **THEN** its predictions are on that device, whatever the process default names

## MODIFIED Requirements

### Requirement: The table states what it does and does not establish
Documentation accompanying the comparison SHALL state what the harness holds fixed and what it does not, and SHALL NOT present the table as a controlled comparison of the methods while conditions remain that prevent it. The harness controls the data and the metrics. It does not control the tuning, which stays a stated limitation. It no longer fails to control the seed: results are reported over a fixed, recorded set of seeds with a centre and a spread for every metric, so a reader can tell whether a difference between two operators is larger than the variation between runs of the same operator.

#### Scenario: Scope of the claim is stated
- **WHEN** a reader consults the benchmark's documentation
- **THEN** it says the comparison holds data and metrics fixed, that hyperparameters are untuned on both sides, and which seed set the reported figures come from

#### Scenario: Fit times are qualified
- **WHEN** fit times for different operators are compared
- **THEN** the documentation notes what differs between them beyond the method itself, so the numbers are not read as a like-for-like cost comparison

#### Scenario: Spread accompanies every metric
- **WHEN** the comparison table is produced
- **THEN** each metric carries both a centre and a measure of spread across the seed set, and the seeds used are recorded

#### Scenario: A single-seed table is not presented as a result
- **WHEN** the benchmark is run at one seed, for a quick check
- **THEN** its output is marked as such and is not the artifact the documentation points a reader at

#### Scenario: Run-to-run spread is stated per operator
- **WHEN** the table reports an operator's spread across the seed set
- **THEN** the documentation also states that operator's spread between repeated runs at one fixed seed, so a seed spread is not read as the method's variation where repeated runs of one seed vary as much

### Requirement: A row says how its forcing was obtained
Where the comparison offers more than one dataset mapping a forcing to a solution, each SHALL make clear whether its forcing was drawn from a stated distribution or derived from a solution that was drawn first. The two are not interchangeable: a forcing derived as a solution's residual carries whatever structure that solution had, so an operator scoring well on it has not been shown to handle a forcing anyone would choose.

#### Scenario: The derived and drawn forcings are distinguishable
- **WHEN** a reader sees more than one forcing-to-solution row in the table
- **THEN** each row's label says which of the two it is, so neither is mistaken for the other

#### Scenario: Adding a row leaves the others alone
- **WHEN** a dataset is added to the comparison
- **THEN** the rows already reported are unchanged, and that is verified by comparing each metric's centre and spread against a previous run over the same seed set, rather than by comparing single draws that differ between runs anyway
