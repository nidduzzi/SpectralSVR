## ADDED Requirements

### Requirement: A generator refuses an argument it does not take
A problem generator and a basis generator SHALL accept only the parameters they name and SHALL refuse any other argument at the call, so a precision, a period or any other setting a caller states is either honoured or refused, never silently dropped. A generator that hands work to another generator SHALL pass on only parameters it names.

#### Scenario: A stated precision a generator does not take is refused
- **WHEN** a Burgers, antiderivative or basis generator is called with a precision argument it does not name, such as `dtype` or `coeff_dtype`
- **THEN** the call is refused with a `TypeError` naming the argument, rather than returning coefficients at the default precision

#### Scenario: Any argument a generator does not name is refused
- **WHEN** a Burgers, antiderivative or basis generator is called with any keyword it does not name
- **THEN** the call is refused with a `TypeError` naming the keyword
