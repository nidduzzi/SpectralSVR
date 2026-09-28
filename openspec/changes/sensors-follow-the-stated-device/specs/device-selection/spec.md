## MODIFIED Requirements

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
