"""The outcome layer.

The policy decides whether to act. This layer answers the question that comes after: what
happened next, and what may honestly be said about it from the events that exist.

**It measures; it does not conclude.** A learner who answers the next question correctly
after a recap is not thereby a learner the recap helped, and the events after the prompt
would very likely have looked the same without it. Every record produced here is therefore
stamped as an observation and is structurally unable to be stamped as anything else, and
every measure is paired with the standing of its reading: a measure that reports
``deteriorated`` is one that was read, and a measure that could not be read says so and
says why. Reading a causal conclusion out of these records is the feedback layer's job,
on the other side of a wall.

**Absence is reported as absence, and never as a negative finding.** A window with no
activity is not evidence that a learner disengaged, a session with no end event is not
evidence that they left, and a question that was not answered is not an answer that was
wrong. Each of those is reported as insufficient data with the reason stated, and the model
refuses to construct a direction for an unread measure — so the rule is a construction
constraint rather than something a reader has to notice.

**No change is a finding, and it is not the same as no finding.** A measured pair that
moved less than the configured tolerance is reported as ``no_change``: a positive claim
that something was read and did not move. Collapsing that into "unknown" would leave a
system with no way to state that a stable learner is stable, and collapsing it into
"improved" would let sampling noise accumulate into a claimed pattern.

Importing this package pulls in the outcome layer and its declared dependencies. It does
not import the feedback, evaluation, or API layers, and a test enforces that statically
rather than leaving it to review.
"""

from __future__ import annotations

from focus_engine.outcomes.engine import (
    DEFAULT_POLICY_VERSION,
    OutcomeEngine,
    OutcomeError,
)
from focus_engine.outcomes.measures import ACTIVITY_EVENTS, direction_of
from focus_engine.outcomes.models import (
    OUTCOME_MEASURES,
    OUTCOME_V1,
    Measurement,
    MeasurementStatus,
    OutcomeDirection,
    OutcomeMeasure,
    OutcomeRecord,
    OutcomeWindow,
    WindowKind,
)

__all__ = [
    "ACTIVITY_EVENTS",
    "DEFAULT_POLICY_VERSION",
    "OUTCOME_MEASURES",
    "OUTCOME_V1",
    "Measurement",
    "MeasurementStatus",
    "OutcomeDirection",
    "OutcomeEngine",
    "OutcomeError",
    "OutcomeMeasure",
    "OutcomeRecord",
    "OutcomeWindow",
    "WindowKind",
    "direction_of",
]
