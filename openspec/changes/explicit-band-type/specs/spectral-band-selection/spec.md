## Purpose

Defines how a caller states which modes an operation keeps when it band-limits a coefficient grid, what each form of that statement guarantees about whether the result describes a real field, and which statements can be made at all.

## ADDED Requirements

### Requirement: A band is stated, not counted
An operation that keeps some modes of a coefficient grid and discards the rest SHALL take a band, not a bare count. The band SHALL carry which modes it selects, so that a caller reading the call site can tell what is kept without knowing the operation's internal convention for turning a number into a set of modes.

#### Scenario: The kept set is evident at the call site
- **WHEN** a caller states a band and applies it to a coefficient grid
- **THEN** the modes kept follow from the band alone, not from an unstated convention about how a count is distributed across positive and negative modes

#### Scenario: A bare count is not accepted
- **WHEN** a caller passes a plain number where a band is expected
- **THEN** the call is rejected, because a number alone does not say which modes it means

### Requirement: A band that promises a real result cannot be stated wrongly
A band that guarantees a real-valued result SHALL be constructible only in forms that actually deliver one. It SHALL NOT be possible to state such a band that selects one member of a conjugate pair without the other, because no result satisfies that request: modes pair around a lone zero mode, so a kept set with a real inverse transform is either odd-sized or the whole grid.

#### Scenario: A symmetric band about the zero mode
- **WHEN** a caller states a band by the highest magnitude it keeps
- **THEN** the kept set is that magnitude's positive and negative modes together with the zero mode, and applying it to a conjugate-symmetric grid yields a grid that is still conjugate-symmetric

#### Scenario: The whole grid is a statable band
- **WHEN** a caller states the band that keeps everything
- **THEN** it is accepted and the operation is a no-op, including on an even-sized grid where the highest magnitude has no positive counterpart and is its own conjugate partner

#### Scenario: An unsatisfiable band cannot be stated
- **WHEN** a caller tries to state a real-guaranteeing band that would keep an even number of modes short of the whole grid
- **THEN** no such band can be constructed, rather than one being constructed and silently returning a grid that is not conjugate-symmetric

### Requirement: Truncation without a realness promise remains available
A band form that keeps an exact number of modes SHALL remain available, and SHALL state that its result is not guaranteed to describe a real field. Coefficient grids that are not conjugate-symmetric are legitimate — a predicted grid is not symmetrised — so an operation that truncates one is not defective merely because its result is complex.

#### Scenario: An exact count is honoured
- **WHEN** a caller states a band by an exact number of modes and applies it
- **THEN** exactly that many modes are kept, whatever the parity of the number

#### Scenario: The absent guarantee is stated
- **WHEN** a caller reads what that form promises
- **THEN** it says the result is not guaranteed to be conjugate-symmetric, and says which other form to state when a real result is required

#### Scenario: Choosing it is deliberate
- **WHEN** a caller wants the behaviour that does not guarantee a real result
- **THEN** they reach it by naming that form explicitly, so it cannot be selected by accident from an ambiguous argument
