## Purpose

Defines what the operator comparison harness guarantees — the interface every method implements, the shared data every method is given, and the table the comparison produces — and states plainly what conclusions the reported numbers do and do not support.

## ADDED Requirements

### Requirement: Every operator implements the same fit-then-predict contract
An operator SHALL be trainable from a dataset and then queryable at arbitrary points in the output domain. Prediction SHALL be a function of the input functions and the query points alone, so that two operators given the same inputs and the same query points return comparable outputs.

#### Scenario: Predict after fit
- **WHEN** an operator is fitted on a dataset and then asked to predict from held-out input functions at a set of query points
- **THEN** it returns one value per input function per query point

#### Scenario: Predict before fit is refused
- **WHEN** predict is called on an operator that has not been fitted
- **THEN** it raises rather than returning an untrained result

#### Scenario: Query points are not restricted to the training grid
- **WHEN** an operator is queried at points it was not trained on
- **THEN** it returns values there, because the comparison depends on evaluating every method at the same arbitrary points

### Requirement: Every operator sees identical data
The dataset object SHALL be the single source of the comparison's data: the same input functions sampled at the same sensor locations, and the same output functions held as an evaluable field rather than as pre-sampled values. No operator SHALL receive data any other operator did not.

#### Scenario: Shared inputs and targets
- **WHEN** two operators are benchmarked on one dataset
- **THEN** both are fitted on the same sensor-sampled inputs and both are scored against targets drawn from the same output field at the same query points

#### Scenario: Targets are evaluated, not pre-sampled
- **WHEN** targets are requested at query points
- **THEN** they are evaluated from the stored output field at those points, so the choice of query points is not baked into the dataset

#### Scenario: Train and test are disjoint
- **WHEN** a dataset is split for benchmarking
- **THEN** the functions used to fit and the functions used to score share no members

### Requirement: The comparison table is produced under one measurement protocol
The benchmark SHALL score every operator against the same targets at the same query points using the same metric suite, returning one row per operator. The reported fit time SHALL measure fitting alone, excluding dataset construction and prediction. Scoring SHALL take place on a single device so that a prediction produced elsewhere cannot alter a metric.

#### Scenario: One row per operator, same metrics
- **WHEN** the benchmark is run over a set of operators
- **THEN** every row carries the same metric columns, computed the same way, against the same targets

#### Scenario: Fit time excludes prediction
- **WHEN** the reported fit time is read
- **THEN** it covers only the fitting call, not dataset construction or prediction

#### Scenario: A failing operator is not silently omitted
- **WHEN** an operator raises during fit or predict
- **THEN** the benchmark surfaces the failure rather than returning a table with that operator quietly missing

### Requirement: The table states what it does and does not establish
Documentation accompanying the comparison SHALL state that the harness controls the data and the metrics but not the tuning, and that results are produced at a single seed. It SHALL NOT present the table as a controlled comparison of the methods while those conditions hold.

#### Scenario: Scope of the claim is stated
- **WHEN** a reader consults the benchmark's documentation
- **THEN** it says the comparison holds data and metrics fixed, that hyperparameters are untuned on both sides, and that each figure comes from one seed

#### Scenario: Fit times are qualified
- **WHEN** fit times for different operators are compared
- **THEN** the documentation notes what differs between them beyond the method itself, so the numbers are not read as a like-for-like cost comparison
