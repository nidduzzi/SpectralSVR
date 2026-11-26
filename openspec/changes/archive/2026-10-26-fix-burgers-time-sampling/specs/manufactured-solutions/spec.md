## ADDED Requirements

### Requirement: A time-dependent manufactured pair holds the field at its slice times
A manufactured pair generated as time-dependent SHALL hold, in each slice, the manufactured field and forcing at the time that slice is read at, those times spread over the time interval inclusively from its start to its end. The time-dependent pair SHALL agree with the same pair generated without time dependence, evaluated at the slice times, to the rounding of the coefficient precision, at every slice count of at least two and every mode count.

#### Scenario: Each slice reads as the manufactured field at its time
- **WHEN** a manufactured pair is generated as time-dependent at any slice count of at least two, and the same pair is generated without time dependence
- **THEN** each slice of the field and of the forcing matches the non-time-dependent pair evaluated at that slice's time, the interval's end included, to the rounding of the coefficient precision
