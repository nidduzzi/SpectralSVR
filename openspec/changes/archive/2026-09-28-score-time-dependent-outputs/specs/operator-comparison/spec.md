## ADDED Requirements

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
