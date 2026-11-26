## ADDED Requirements

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
