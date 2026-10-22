# basis-evaluation Specification

## Purpose
Defines how a Fourier basis evaluates a field at arbitrary points under its evaluation strategy, and what the approximate fast path guarantees relative to exact evaluation, so choosing the fast path never changes which fields can be evaluated.

## Requirements

### Requirement: Approximate evaluation accepts every field the exact path accepts
Evaluating a field at arbitrary points under a strategy that takes the approximate path SHALL succeed for every mode count the exact path accepts, on every axis, including axes of one and two modes. Which path a strategy selects SHALL NOT decide whether a field can be evaluated.

#### Scenario: A field with a one- or two-mode axis evaluates approximately
- **WHEN** a field with one, two or three mode axes, any of them holding one or two modes, is evaluated at arbitrary points under a strategy that takes the approximate path
- **THEN** the evaluation returns one value per function and point, as exact evaluation does, rather than raising

### Requirement: Approximate evaluation agrees with exact evaluation within its stated bound
Approximate evaluation SHALL agree with exact evaluation of the same field at the same points to within one twentieth of the largest exact value, for every mode count on every axis, including one and two modes.

#### Scenario: Approximate and exact values agree at every mode count
- **WHEN** the same field is evaluated at the same points under an approximate and under an exact strategy, at any mode count per axis from one upward and on one, two or three axes
- **THEN** the largest difference between the two is at most one twentieth of the largest exact value
