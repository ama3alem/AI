"""Human feedback: what a person said, and what the system is allowed to do about it.

**A correction is evidence, not a training signal.** This is the most important property
of the module. When a teacher disputes a prediction, the honest responses are to record
the disagreement, reduce authority in that context, and route the case for review. Feeding
the correction straight into a model update is the tempting alternative and it is wrong for
three separate reasons: a single disputed prediction is not a representative sample, the
correct label is a teacher's judgement rather than an observation, and a system that
retrains on its own corrections can be walked into any conclusion by anyone who can file
feedback. Nothing here writes to a model. The feedback produces a
:class:`~focus_engine.authority.memory.MemoryEntry`, and the model is retrained only by a
deliberate, versioned, offline process that this layer has no access to.

**Availability is an input, and its absence is not consent.** When no human is reachable
the correct outcome is deferral, never autonomous action. A queued approval that nobody
can give must expire into "still not approved", not into "approved by default", because
the second reading turns an unreachable teacher into a blank cheque. The engine treats
:attr:`HumanAvailability.UNAVAILABLE` and :attr:`HumanAvailability.UNKNOWN` identically
here, and both produce the same restriction.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from focus_engine.authority.memory import MemoryEntry
from focus_engine.schemas.primitives import DataOrigin

__all__ = [
    "DisagreementRecord",
    "FeedbackOutcome",
    "FeedbackType",
    "HumanAvailability",
    "HumanFeedback",
]


class HumanAvailability(StrEnum):
    """Whether a person who could authorise an action can actually be reached.

    :attr:`UNKNOWN` is a first-class member rather than folded into
    :attr:`UNAVAILABLE` because the two demand different reporting. ``UNKNOWN`` means the
    system never asked, which is a gap in the deployment; ``UNAVAILABLE`` means it asked
    and was refused or is out of hours. Both restrict authority identically, because
    neither licenses autonomous action, but only one of them is a bug to fix.
    """

    AVAILABLE = "available"
    """A named approver is reachable within the decision's own time budget."""

    UNAVAILABLE = "unavailable"
    """An approver exists and was asked, and cannot respond now."""

    UNKNOWN = "unknown"
    """No approver was ever identified for this context."""


class FeedbackType(StrEnum):
    """What a person did in response to a decision.

    These are the four responses the system models. Each is distinct because each implies
    a different update to authority: confirmation leaves it alone, dispute reduces it, a
    correction both reduces it and invalidates the recorded outcome, and an override is a
    decision that was made over the system's objection.
    """

    CONFIRMED = "confirmed"
    """The person agreed with the system's reading. No authority change."""

    DISPUTED = "disputed"
    """The person disagreed with the reading. Reduces authority in this context."""

    CORRECTED = "corrected"
    """The person supplied a different reading. Reduces authority and records the
    correction for review."""

    OVERRIDE = "override"
    """The person took or forbade an action the system had decided on. The strongest
    signal available, and always reduces authority."""


class FeedbackOutcome(StrEnum):
    """What the system did with a piece of feedback.

    :attr:`REJECTED` is a real outcome and not an error. Feedback that fails validation -
    a correction with no reading, an override for a different learner - is rejected with a
    reason, and the rejection is recorded. Silently dropping it would make the ledger
    incomplete in exactly the cases where a reviewer is most likely to be asking.
    """

    ACCEPTED = "accepted"
    """Recorded, and applied to authority and memory."""

    REJECTED = "rejected"
    """Not applicable, with a reason. Still recorded."""

    NEEDS_REVIEW = "needs_review"
    """Recorded but not applied automatically; a human must adjudicate."""


@dataclass(frozen=True, slots=True)
class HumanFeedback:
    """One piece of feedback from a named person about one decision.

    Attributes:
        feedback_id: Stable identifier, used in the ledger and for deduplication.
        decision_ref: The decision or envelope this feedback is about. Required: feedback
            that cannot be traced to a decision is unusable, because there is nothing to
            attach the consequence to.
        feedback_type: What the person did.
        actor_role: The role of the person giving it, such as ``course_coordinator``. A
            role rather than a name so that a record survives staff turnover without
            becoming personally identifying.
        submitted_at: When the feedback was given.
        context_type: The context the decision was made in. Feedback about a graded
            assessment does not reduce authority in free practice.
        corrected_reading: The replacement reading, required when
            :attr:`feedback_type` is ``CORRECTED`` and forbidden otherwise.
        note: Free text. Never parsed, never load-bearing.
        data_origin: Whether the underlying decision came from real or synthetic data.
    """

    feedback_id: str
    decision_ref: str
    feedback_type: FeedbackType
    actor_role: str
    submitted_at: datetime
    context_type: str
    corrected_reading: str | None = None
    note: str = ""
    data_origin: DataOrigin = DataOrigin.SYNTHETIC

    def __post_init__(self) -> None:
        """Validate that a correction carries a reading and nothing else claims one.

        Raises:
            ValueError: If ``feedback_id``, ``decision_ref``, ``actor_role``, or
                ``context_type`` is empty, or if a ``CORRECTED`` feedback has no reading
                while a non-correction does. Accepting a correction with nothing to
                correct would record a disagreement while losing the only information that
                made it actionable.
        """
        for name in ("feedback_id", "decision_ref", "actor_role", "context_type"):
            value = str(getattr(self, name))
            if not value.strip():
                raise ValueError(f"{name} is required and must be non-empty")
        if (
            self.feedback_type is FeedbackType.CORRECTED
            and not (self.corrected_reading or "").strip()
        ):
            raise ValueError(
                "a CORRECTED feedback must carry corrected_reading; without it the "
                "disagreement is recorded and the replacement is lost"
            )
        if self.feedback_type is not FeedbackType.CORRECTED and self.corrected_reading:
            raise ValueError(
                f"corrected_reading is only valid for CORRECTED feedback; got "
                f"{self.feedback_type.value!r} with reading {self.corrected_reading!r}"
            )

    def reduces_authority(self) -> bool:
        """Report whether this feedback should lower authority.

        Returns:
            ``True`` for every type except :attr:`FeedbackType.CONFIRMED`. Confirmation
            is deliberately inert: a system that gained authority for being agreed with
            would be incentivised to seek agreement rather than to be right.
        """
        return self.feedback_type is not FeedbackType.CONFIRMED

    def to_memory_entry(self, action: str) -> MemoryEntry:
        """Convert this feedback into a decayed memory entry.

        Args:
            action: The action type the decision concerned, used as the memory key.

        Returns:
            A :class:`~focus_engine.authority.memory.MemoryEntry` whose outcome is
            ``disputed`` for every reducing type. Overrides and corrections are both
            recorded as ``disputed`` because the decay engine's job is to price the
            evidence, and both carry the same weight of human contradiction. The
            distinction is preserved in :attr:`feedback_type` on the ledger entry, not
            collapsed in the memory.

        Raises:
            ValueError: If this feedback does not reduce authority, and therefore has no
                memory effect. Calling this on a confirmation would create a memory entry
                for an event that should leave the system untouched.
        """
        if not self.reduces_authority():
            raise ValueError(
                f"feedback {self.feedback_id!r} is {self.feedback_type.value!r}, which does "
                "not reduce authority and therefore has no memory entry"
            )
        return MemoryEntry(
            action=action,
            context_type=self.context_type,
            recorded_at=self.submitted_at,
            outcome="disputed",
            weight=1.0,
        )

    def to_summary_dict(self) -> dict[str, object]:
        """Render the feedback as flat, loggable fields.

        Returns:
            A dict carrying every field, with enums and datetimes flattened.
        """
        return {
            "feedback_id": self.feedback_id,
            "decision_ref": self.decision_ref,
            "feedback_type": self.feedback_type.value,
            "actor_role": self.actor_role,
            "submitted_at": self.submitted_at.isoformat(),
            "context_type": self.context_type,
            "corrected_reading": self.corrected_reading,
            "note": self.note,
            "data_origin": self.data_origin.value,
            "reduces_authority": self.reduces_authority(),
        }


@dataclass(frozen=True, slots=True)
class DisagreementRecord:
    """A logged human disagreement, kept for review rather than for scoring.

    The authority layer prices a disagreement immediately. This record exists for the
    slower question: *is the system systematically wrong about this kind of learner?*
    Answering that requires reading the disputes together, which no single decision can
    do, and which must never happen as a side effect of a live decision.

    Attributes:
        disagreement_id: Stable identifier.
        feedback_id: The feedback this came from.
        learner_id: The learner affected.
        context_type: The context the disagreement was about.
        actor_role: The role that disagreed.
        disputed_at: When the disagreement was recorded.
        reason: Why the person disagreed, as recorded.
        requires_model_review: Set when the disagreement is severe enough that the model,
            not just the authority weight, should be examined.
    """

    disagreement_id: str
    feedback_id: str
    learner_id: str
    context_type: str
    actor_role: str
    disputed_at: datetime
    reason: str = ""
    requires_model_review: bool = False

    def to_summary_dict(self) -> dict[str, object]:
        """Render the record as flat, loggable fields.

        Returns:
            A dict carrying every field, with the datetime flattened.
        """
        return {
            "disagreement_id": self.disagreement_id,
            "feedback_id": self.feedback_id,
            "learner_id": self.learner_id,
            "context_type": self.context_type,
            "actor_role": self.actor_role,
            "disputed_at": self.disputed_at.isoformat(),
            "reason": self.reason,
            "requires_model_review": self.requires_model_review,
        }
