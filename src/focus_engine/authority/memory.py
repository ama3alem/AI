"""What the system remembers about its own past interventions.

The authority layer needs one thing the rest of the engine does not provide: a memory of
whether the system has been *useful* in this specific context, and whether the people it
serves have *agreed* with it. Neither is a property of the current prediction. A model can
be beautifully calibrated in aggregate and still be wrong every time about one learner in
one context, and a system that only reads the current probability cannot tell the
difference.

**Memory is keyed by context, not by learner alone.** ``(learner, context, action)`` is the
key because the useful question is not "does this intervention work" but "does *this*
intervention work *here*". A recap that reliably helps during free practice and reliably
fails immediately before a graded submission is one intervention type, two facts, and a
learner-scoped memory would average them into a number that is true of neither.

**Memory never raises authority.** It can only reduce it, or leave it alone. This is the
structural reason a "the system has done well lately, let it act more freely" feature does
not exist here: an intervention that improved an outcome is evidence that the
intervention was appropriate, not that the system's judgement about *other* things is
better. Improvement is recorded as credit against decay, below the base weight, and
cannot lift a context above what an uncorrected first impression would have earned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

__all__ = [
    "ContextMemory",
    "InterventionOutcomeMemory",
    "MemoryEntry",
]


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    """One recorded intervention and what came of it.

    Attributes:
        action: The action type that was delivered.
        context_type: The context it was delivered in. Part of the key, not a label.
        recorded_at: When the outcome became known.
        outcome: A short machine-readable result: ``improved``, ``deteriorated``,
            ``no_response``, ``confirmed``, or ``disputed``. Free-form on purpose: the
            decay engine treats unknown values conservatively, so a new outcome type
            degrades trust rather than being silently ignored.
        weight: How much this entry counts, in ``[0, 1]``. Lets a caller discount a
            weak observation without discarding it, so that a record of what happened is
            never lost.
    """

    action: str
    context_type: str
    recorded_at: datetime
    outcome: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        """Reject an entry whose weight is outside the unit interval.

        Raises:
            ValueError: If ``weight`` is not in ``[0, 1]``. A negative weight would let a
                caller cancel a recorded failure out of the memory, and a weight above 1
                would let one entry dominate a decay calculation.
        """
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"weight must lie in [0.0, 1.0]; got {self.weight!r}")

    def age_days(self, at: datetime) -> float:
        """Return how many days before ``at`` this entry was recorded.

        Args:
            at: The instant to measure from.

        Returns:
            Elapsed days, never negative. A future-dated entry is treated as having zero
            age rather than a negative one, so a clock that moves backwards degrades to
            "no decay yet" instead of amplifying a penalty.
        """
        delta = at - self.recorded_at
        return max(0.0, delta.total_seconds() / 86400.0)


@dataclass(frozen=True, slots=True)
class ContextMemory:
    """The recorded history for one ``(learner, context)`` pair.

    Attributes:
        learner_id: The learner this memory belongs to.
        context_type: The context this memory is valid for.
        entries: Every recorded entry, in insertion order.
    """

    learner_id: str
    context_type: str
    entries: tuple[MemoryEntry, ...] = ()

    def with_entry(self, entry: MemoryEntry) -> ContextMemory:
        """Return a new memory with ``entry`` appended.

        Args:
            entry: The entry to record.

        Returns:
            A new :class:`ContextMemory`. The original is unchanged, so a caller cannot
            retroactively alter what was known at an earlier decision.
        """
        return ContextMemory(
            learner_id=self.learner_id,
            context_type=self.context_type,
            entries=(*self.entries, entry),
        )

    def for_action(self, action: str) -> tuple[MemoryEntry, ...]:
        """Return the entries recorded for one action type.

        Args:
            action: The action type to filter by.

        Returns:
            Matching entries in insertion order.
        """
        return tuple(entry for entry in self.entries if entry.action == action)

    def counts(self) -> dict[str, int]:
        """Tally the entries by outcome.

        Returns:
            A mapping from outcome name to count, including only outcomes present.
        """
        tally: dict[str, int] = {}
        for entry in self.entries:
            tally[entry.outcome] = tally.get(entry.outcome, 0) + 1
        return tally

    def is_empty(self) -> bool:
        """Report whether anything has been recorded.

        Returns:
            ``True`` when there are no entries.
        """
        return not self.entries


@dataclass(frozen=True, slots=True)
class InterventionOutcomeMemory:
    """Every context memory held for one learner.

    The authority engine holds one of these per learner and looks up the context it needs.
    Keeping the contexts separate here - rather than one blended history - is what
    prevents a strong record in a permissive context from lending permission in a
    high-stakes one.

    Attributes:
        learner_id: The learner these memories belong to.
        memories: One :class:`ContextMemory` per context type.
    """

    learner_id: str
    memories: dict[str, ContextMemory] = field(default_factory=dict)

    def for_context(self, context_type: str) -> ContextMemory:
        """Return the memory for a context, creating an empty one if absent.

        Args:
            context_type: The context to look up.

        Returns:
            The stored :class:`ContextMemory`, or a new empty one. Returning an empty
            memory rather than ``None`` is what makes "no history" and "history with
            nothing recorded" behave identically downstream, which is the correct reading
            of both.
        """
        existing = self.memories.get(context_type)
        if existing is not None:
            return existing
        return ContextMemory(learner_id=self.learner_id, context_type=context_type)

    def record(self, entry: MemoryEntry) -> InterventionOutcomeMemory:
        """Return a new memory with ``entry`` filed under its context.

        Args:
            entry: The entry to record.

        Returns:
            A new :class:`InterventionOutcomeMemory` with the entry appended to the
            matching context. The original is unchanged.
        """
        target = self.for_context(entry.context_type)
        updated = self.memories.copy()
        updated[entry.context_type] = target.with_entry(entry)
        return InterventionOutcomeMemory(learner_id=self.learner_id, memories=updated)

    def contexts(self) -> tuple[str, ...]:
        """List the context types with recorded history.

        Returns:
            The context names in insertion order.
        """
        return tuple(self.memories)

    def total_entries(self) -> int:
        """Count every entry across every context.

        Returns:
            The total number of recorded entries.
        """
        return sum(len(memory.entries) for memory in self.memories.values())

    def is_empty(self) -> bool:
        """Report whether any context has recorded history.

        Returns:
            ``True`` when no entries exist anywhere.
        """
        return self.total_entries() == 0
