## ADDED Requirements

### Requirement: A forced-Burgers dataset is available to the benchmark
The operator benchmark SHALL offer a dataset whose input is a sampled time-and-space forcing and whose output is the numerically integrated Burgers solution from rest, alongside the existing manufactured-solution dataset, so the two forcing distributions can be compared.

#### Scenario: Forced dataset is registered
- **WHEN** the benchmark script is listed
- **THEN** a forced-Burgers dataset appears among its choices with a label distinguishing it from the manufactured-solution dataset

#### Scenario: Forced dataset builds and scores
- **WHEN** the benchmark is run on that dataset with the SpectralSVR operator in quick mode
- **THEN** it completes and writes a row for the dataset with finite metrics

### Requirement: A row says how its forcing was obtained
Where the comparison offers more than one dataset mapping a forcing to a solution, each SHALL make clear whether its forcing was drawn from a stated distribution or derived from a solution that was drawn first. The two are not interchangeable: a forcing derived as a solution's residual carries whatever structure that solution had, so an operator scoring well on it has not been shown to handle a forcing anyone would choose.

#### Scenario: The derived and drawn forcings are distinguishable
- **WHEN** a reader sees more than one forcing-to-solution row in the table
- **THEN** each row's label says which of the two it is, so neither is mistaken for the other

#### Scenario: Adding a row leaves the others alone
- **WHEN** a dataset is added to the comparison
- **THEN** the rows already reported are unchanged, and that is verified against a previous run rather than assumed from the size of the change
