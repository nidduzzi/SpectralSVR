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
- **THEN** the rows already reported are unchanged, and that is verified by comparing each metric's centre and spread against a previous run over the same seed set, rather than by comparing single draws that differ between runs anyway

### Requirement: A time-dependent output is scored across its whole time interval
Where an output field varies in time, the comparison SHALL score it at query points that state a time and a place, with the time first. The targets and every operator's predictions SHALL give one value per query point, in the same layout, so predictions are compared with targets point for point. The whole time interval SHALL be scorable, including its end.

An operator that can be fitted to a time-varying output but cannot be evaluated on one gives the comparison nothing to score, and a comparison that scores only part of the interval claims more than it measured. Evaluating the time axis by any convention other than the field's own, for example by treating time as periodic, reads the end of the interval as its start.

Where the output does not vary in time, the targets and predictions SHALL be exactly as they were before this requirement.

#### Scenario: Targets of a time-varying output are read at a time and a place
- **WHEN** targets are requested for a time-varying output at query points whose first coordinate is time
- **THEN** each query point gives the stored field's value at that time and place, and at a time the field was stored at, that value is the stored one, to rounding

#### Scenario: Predictions of a time-varying output are read at the same points
- **WHEN** an operator fitted to a time-varying output predicts at those query points
- **THEN** it gives one value per query point in the layout of the targets, so it is scored point for point against them

#### Scenario: The end of the interval is scored
- **WHEN** a forcing-to-solution row whose solution varies in time is scored
- **THEN** its query points include the final time, and the value there is the final state rather than the initial one

#### Scenario: A time-invariant output scores as before
- **WHEN** the output field does not vary in time
- **THEN** its targets and predictions are bit-identical to what they were before time-varying outputs could be scored

#### Scenario: A query point without a time is refused for a time-varying output
- **WHEN** query points for a time-varying output carry only the spatial coordinates
- **THEN** evaluation is refused with a message saying time must be the first coordinate, rather than a value being returned for some time the caller did not state

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
