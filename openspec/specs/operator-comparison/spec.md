# operator-comparison Specification

## Purpose
Defines what the operator comparison harness guarantees — the interface every method implements, the shared data every method is given, and the table the comparison produces — and states plainly what conclusions the reported numbers do and do not support.

## Requirements

### Requirement: Every operator implements the same fit-then-predict contract
An operator SHALL be trainable from a dataset, and fitting SHALL yield a distinct fitted value that is the only thing able to predict. An unfitted operator SHALL NOT carry a prediction method at all, so calling one before fitting is a type error rather than a runtime check. Prediction SHALL be a function of the input functions and the query points alone, so that two operators given the same inputs and the same query points return comparable outputs.

#### Scenario: Predict after fit
- **WHEN** an operator is fitted on a dataset and then asked to predict from held-out input functions at a set of query points
- **THEN** it returns one value per input function per query point

#### Scenario: Predicting before fitting cannot be stated
- **WHEN** a caller holds an operator that has not been fitted
- **THEN** there is no prediction method on it to call, so the mistake is rejected where the call is written rather than by a check inside it

#### Scenario: Fitting yields the thing that predicts
- **WHEN** an operator is fitted
- **THEN** it returns a fitted value carrying the prediction method, and that value is the only route to a prediction

#### Scenario: Query points are not restricted to the training grid
- **WHEN** an operator is queried at points it was not trained on
- **THEN** it returns values there, because the comparison depends on evaluating every method at the same arbitrary points

### Requirement: Every operator draws from one dataset object
The dataset object SHALL be the single source of the comparison's data: the same input functions sampled at the same sensor locations, and the same output functions held as an evaluable field rather than as pre-sampled values. Each operator SHALL take from that object the view its method needs, so that a difference between two rows is a difference of method rather than of data. A dataset whose parts do not agree, and a train/test pair that shares functions, SHALL NOT be constructible.

#### Scenario: Shared inputs and targets
- **WHEN** two operators are benchmarked on one dataset
- **THEN** both are fitted on the same sensor-sampled inputs and both are scored against targets drawn from the same output field at the same query points

#### Scenario: Each method takes the view it needs
- **WHEN** one operator trains on the output field's spectral coefficients and another on that field sampled at its own training points
- **THEN** both views come from the same stored field, so the data is shared even though what each method consumes is not identical

#### Scenario: Targets are evaluated, not pre-sampled
- **WHEN** targets are requested at query points
- **THEN** they are evaluated from the stored output field at those points, so the choice of query points is not baked into the dataset

#### Scenario: Overlapping train and test cannot be stated
- **WHEN** a caller supplies the data a comparison is run on
- **THEN** the training and scoring halves come from partitioning one source, so a split that shares functions cannot be constructed and the benchmark needs no check of its own

#### Scenario: An inconsistent dataset cannot be built
- **WHEN** a dataset is constructed whose sampled inputs do not match its sensor count, or whose number of input functions does not match its output field
- **THEN** construction fails, because no operator can be fitted on such a dataset and the mismatch has no meaning to carry forward

### Requirement: The comparison table is produced under one measurement protocol
The benchmark SHALL score every operator against the same targets at the same query points using the same metric suite, returning one row per operator. The reported fit time SHALL measure fitting alone, excluding dataset construction and prediction. Scoring SHALL take place on a single device so that a prediction produced elsewhere cannot alter a metric.

#### Scenario: One row per operator, same metrics
- **WHEN** the benchmark is run over a set of operators
- **THEN** every row carries the same metric columns, computed the same way, against the same targets

#### Scenario: Fit time excludes prediction
- **WHEN** the reported fit time is read
- **THEN** it covers only the fitting call, not dataset construction or prediction

#### Scenario: A failing operator is not silently omitted from a run
- **WHEN** an operator raises during fit or predict
- **THEN** the benchmark lets the failure propagate rather than returning a table with that operator quietly missing

#### Scenario: A short table requires an explicit opt-in
- **WHEN** a requested operator cannot be constructed because an optional dependency is absent
- **THEN** resolving the operator set fails, and a table missing a requested row is produced only when the caller has said in so many words that missing operators are acceptable

### Requirement: The table states what it does and does not establish
Documentation accompanying the comparison SHALL state that the harness controls the data and the metrics but not the tuning, and that results are produced at a single seed. It SHALL NOT present the table as a controlled comparison of the methods while those conditions hold.

#### Scenario: Scope of the claim is stated
- **WHEN** a reader consults the benchmark's documentation
- **THEN** it says the comparison holds data and metrics fixed, that hyperparameters are untuned on both sides, and that each figure comes from one seed

#### Scenario: Fit times are qualified
- **WHEN** fit times for different operators are compared
- **THEN** the documentation notes what differs between them beyond the method itself, so the numbers are not read as a like-for-like cost comparison

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
- **THEN** every case already offered builds exactly as it did before, whether or not the new one is built alongside it, and the rows already reported are verified unchanged against a previous run rather than assumed from the size of the change
