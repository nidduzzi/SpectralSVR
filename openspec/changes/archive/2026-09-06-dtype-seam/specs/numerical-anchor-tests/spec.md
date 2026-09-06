## ADDED Requirements

### Requirement: An anchor may request the precision it needs
An anchor whose oracle requires more precision than the library's default SHALL be able to request it through the precision setting, rather than being relocated to a lower layer or abandoned. Where an anchor's measurement would otherwise saturate at the default precision, the test SHALL state which precision it runs at and why.

#### Scenario: A convergence anchor runs at double precision
- **WHEN** an anchor measures a convergence rate whose error range extends below single precision's resolution
- **THEN** it selects double precision explicitly, and records that the default would floor the measurement

#### Scenario: Saturation is distinguished from order loss
- **WHEN** a convergence measurement flattens
- **THEN** the test makes clear whether the flattening is the method's behaviour or the precision floor, rather than reporting the measured order alone
