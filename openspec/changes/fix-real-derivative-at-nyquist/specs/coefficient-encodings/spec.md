## ADDED Requirements

### Requirement: A real field reads its boundary mode as a cosine
A field stated as real, on an axis with an even number of modes, SHALL read the coefficient at that axis's boundary mode as a cosine shared evenly between the two mode numbers the axis cannot distinguish. It SHALL do so wherever the field is read: evaluated off its own grid, by exact or approximate evaluation, through time, or on a grid of a different resolution. Differentiating or integrating such a field to an odd order SHALL contribute nothing at the boundary mode, because the cosine's odd derivatives vanish at every node. To an even order it SHALL keep the boundary mode. A field stated as complex SHALL keep its boundary mode as a single wave, as before. On a field's own grid, both readings agree and the values SHALL be unchanged.

#### Scenario: A real field reads real everywhere at every mode count
- **WHEN** a field stated as real is read at any points of its domain, by any evaluation, at an even or odd mode count on each axis
- **THEN** its values are real to rounding

#### Scenario: An odd-order derivative of a real field is real
- **WHEN** a field stated as real is differentiated to an odd order along an axis with an even number of modes
- **THEN** the derivative's values are real to rounding, its boundary mode along that axis is zero, and an even-order derivative keeps the boundary mode

#### Scenario: Integration drops the boundary mode as it drops the mean
- **WHEN** a field stated as real is integrated along an axis with an even number of modes
- **THEN** the result carries neither a mean nor a boundary mode along that axis, and integrating the derivative of a field with neither returns the field to rounding

#### Scenario: A complex field keeps its single boundary wave
- **WHEN** a field stated as complex is read off its grid, differentiated or integrated
- **THEN** the result is the same as it was before the boundary mode of real fields was read as a cosine

#### Scenario: On its own grid a real field reads as before
- **WHEN** a field stated as real is read on its own grid at its own resolution
- **THEN** the values are the same as they were before the boundary mode was read as a cosine
