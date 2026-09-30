## Purpose
Defines who decides where the library places work: the order in which an explicit argument, a process-wide setting and the library's own opinion are consulted, and what happens when a caller's random source disagrees with the result. States that the library carries no device preference of its own, so a caller who says nothing gets what the surrounding process says rather than what the library would prefer.

## Requirements

### Requirement: The stated device wins, and the library states none
Resolving a device SHALL consult exactly two sources, in order: a device given at the call site, then the process-wide default. The library SHALL NOT introduce a third answer of its own, and SHALL NOT prefer an accelerator because one happens to be present.

A caller who names no device is still making a choice — that the surrounding process should decide — and that choice SHALL be honoured. Where the process has expressed no preference, the answer is whatever the platform's own default is, not a device the library selects on the caller's behalf.

This is a deliberate refusal to be clever. A per-component preference cannot be honoured together with a process-wide one, because a process default naming the host is indistinguishable from no preference at all; any rule that accelerated "by default" would silently override a caller who asked for the host. The library therefore offers no defaults to be overridden.

#### Scenario: A stated device is used
- **WHEN** a device is given at a call site
- **THEN** work is placed there, whatever the process-wide setting says

#### Scenario: A process-wide setting is followed
- **WHEN** no device is given at the call site and the process has set one
- **THEN** work is placed on the process's device, for every device the platform supports rather than only the one the library was written against

#### Scenario: Stating a device does not change what is drawn
- **WHEN** the same seed and the same device are used, once by naming the device at the call site and once by setting it as the process default
- **THEN** the values drawn are identical, so a caller may move between the two ways of saying where work goes without changing a result

#### Scenario: The host is not overridden
- **WHEN** the process states the host as its device and an accelerator is present
- **THEN** work is placed on the host, because a stated preference is not a gap to be filled

### Requirement: Placement is statable everywhere work is placed
Every entry point that places work SHALL accept a device, so that a caller can direct one part of a workload without directing all of it. An entry point that resolves a device but offers no way to state one SHALL be treated as incomplete, because it forces the caller to move the whole process to move one computation.

#### Scenario: Parts of a workload are directed separately
- **WHEN** a caller wants one stage of a workload on an accelerator and another on the host
- **THEN** each is stated at its own call site, with no process-wide setting required and no effect on the other

#### Scenario: Generating a problem can be directed
- **WHEN** a caller generates a problem and states a device
- **THEN** the generation is placed there, so a workload large enough to benefit from an accelerator can reach one

#### Scenario: Generating coefficients can be directed
- **WHEN** a caller generates coefficients directly, without going through a problem
- **THEN** the generation is placed where they state, so the lower entry point is directable on the same terms as the higher one rather than reachable only by moving the whole process

### Requirement: Work derived for placed data follows that data
Where the library builds a value for data that already exists -- an index, a scale factor, a comparison constant, the points a field is sampled at -- it SHALL take the device from that data rather than from the process default. Asking the process again is only correct while everything follows the process; once placement is statable, the answer can name a device the data is not on.

This is distinct from a factory that materialises a caller's data. Such a factory correctly follows the process default when the caller states nothing, and SHALL keep doing so.

#### Scenario: A workload placed against the process default still runs
- **WHEN** a caller states one device for their work while the process default names another
- **THEN** generation, indexing, evaluation and resizing all complete, because each derived value is built where the data it meets already is

#### Scenario: Sensors are sampled where their field is
- **WHEN** a dataset samples an input field at a sensor grid
- **THEN** the grid and the sampled values are on the field's device, whatever the process default names

#### Scenario: A case does not depend on the process default
- **WHEN** the same case is built with the same seed and stated device under different process defaults
- **THEN** every value it holds is bit-identical and on the same device, so a reported number depends on the stated device alone

#### Scenario: A slice keeps its field's device
- **WHEN** one time slice of a placed field is taken as a field over space
- **THEN** the slice is on the device of the field it came from

#### Scenario: A dataset whose parts sit on different devices is refused
- **WHEN** a dataset is built from an input field and an output field on different devices, or its sensors and sampled values are given on different devices
- **THEN** it is refused, and the refusal names both devices, rather than a comparison later running across them

### Requirement: The accelerator is named, not detected ad hoc
The library SHALL expose one way to name the platform's accelerator, resolving to the host where none is present. Callers SHALL NOT be expected to write their own availability check, and the library SHALL NOT contain more than one.

#### Scenario: Naming the accelerator is a statement, not a search
- **WHEN** a caller wants the accelerator without knowing which kind the platform has
- **THEN** naming it yields whichever accelerator is present, and the host where none is, without the caller testing for a particular vendor

### Requirement: A random source that disagrees is refused clearly
Where a caller supplies a random source, it SHALL be placed on the device the work is placed on. A disagreement SHALL be refused, and the refusal SHALL name both devices and which of the two the caller is expected to move.

Refusing is correct rather than unhelpful: the two cannot be reconciled without either relocating the caller's work or drawing different numbers from the same seed, and both are decisions the caller must make knowingly.

Where the library builds a random source for itself, because the caller supplied none, it SHALL build it on the device the work is placed on. A library that refused its own source would be reporting its own defect to a caller who chose nothing.

#### Scenario: A source the library builds for itself agrees by construction
- **WHEN** a caller invokes an operation that needs a random source, supplies none, and states a device or has one stated for the process
- **THEN** the source the library builds is on that device, and the refusal above cannot fire against a choice the caller did not make

#### Scenario: A mismatched random source names both devices
- **WHEN** a caller supplies a random source on one device while work is placed on another
- **THEN** the generation is refused before any work is done, and the message names the device the source is on, the device the work is on, and that the source is what moves

#### Scenario: A stated index is part of the statement
- **WHEN** the source and the work both name an index and the two differ
- **THEN** the generation is refused, because they are separate memories and a draw on one cannot be read from the other; where either leaves its index unstated, the two agree, since an unstated index means whichever the process is using

#### Scenario: An agreeing random source is unremarkable
- **WHEN** the random source and the placement agree
- **THEN** generation proceeds with no diagnostic, whether they agree because the caller stated both or because both followed the process

### Requirement: A model never reports a device its fitted state is not on
A model SHALL NOT report a device that the values it holds are not on. Where a device change cannot be honoured, it SHALL be refused at the point the change is asked for, not noticed later by whatever next touches the values.

Refusing is correct rather than unhelpful. The two ways to honour such a change are both worse: moving the fitted values is a side effect on a model the caller handed over, applied to state only each kind of model knows the extent of, so one kind that overlooks a value reintroduces the same untruth silently; and reporting the new device without moving anything is the untruth itself. A refusal is one rule that no kind of model can weaken, and it leaves the caller holding a model that still works.

The refusal SHALL name the device the model was fitted on, the device it was asked to move to, and where the device can be stated instead: when the model is built, and when it is loaded. A caller who reaches this has already fitted or loaded a model on the wrong device, so the message has to say what to do differently next time, not only that this attempt failed, and the remedy it names SHALL be one the caller can actually take.

Where the model holds no fitted values, there is nothing to disagree with and the change SHALL be accepted. Where the change names the device the fitted values are already on, nothing moves and the change SHALL be accepted. Devices are compared as the memory they name, so an unstated index means the one in use, never any index of that kind.

A fit that does not complete SHALL leave no partial fitted state behind, since values a model holds without counting itself fitted would escape the refusal.

#### Scenario: A fitted model refuses to be moved
- **WHEN** a caller changes the device of a model that has been fitted
- **THEN** the change is refused, and the refusal names the device it was fitted on, the device it was asked to move to, and that the device is stated when the model is built or loaded

#### Scenario: An unfitted model is directed freely
- **WHEN** a caller changes the device of a model that holds no fitted values
- **THEN** the change is accepted, because there is nothing whose placement could contradict it

#### Scenario: Every kind of model refuses alike
- **WHEN** the model is of any kind the library offers, each holding fitted values of a different shape
- **THEN** the refusal is the same one, so a caller learns a single behaviour rather than one per kind and a diagnostic that varies with what the model happens to hold

#### Scenario: A move to where the model already is is accepted
- **WHEN** a caller directs a fitted model to the device its fitted values are on, naming it with or without an index
- **THEN** the direction is accepted, and what the model predicts is unchanged

#### Scenario: A fit that fails leaves the model as it was
- **WHEN** a fit fails before it completes
- **THEN** the model holds exactly the fitted values it held before that fit, or none, so no partial state can disagree with the device it reports

#### Scenario: A loaded model is placed where the caller states
- **WHEN** a caller loads a saved model and states a device
- **THEN** its fitted values are on that device and it reports that device, so the refusal's remedy is reachable for a model that was never built by hand
