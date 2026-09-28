"""Lab tests: the paste path must speak the engine's native event language.

The Lab had a defect that only a test of this shape would have caught: it accepted
flattened pseudo-events shaped like ``{"event_type": ..., "subject": ...}`` and reported
pydantic's raw union errors, so a caller who pasted a *correct* event was told
``payload: Field required`` and ``subject: Extra inputs are not permitted``. Every test
here exists to make that class of divergence impossible to reintroduce silently.

The governing rule, asserted repeatedly below: **the engine decides validity.** The Lab
decides what to *say* about a refusal. Nothing in this package re-declares a schema,
relaxes a constraint, or accepts an event the engine would reject.
"""
