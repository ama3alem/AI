"""The authority layer: who is permitted to do what, in which context, and for how long.

**The question this layer answers.** Every layer before it produces evidence. Features
describe what happened, the temporal engine describes the pattern, the model produces a
probability, and the uncertainty engine says how much that probability should be trusted.
None of them answers the question a person actually cares about, which is: *given all of
that, is the system allowed to do anything about it?* A probability is not permission.
This layer is the boundary between the two.

**The three properties that make it a boundary rather than a score.**

- *Permission is scoped, never global.* The envelope names the context it applies to and
  refuses to be constructed without one. A grant earned in free practice does not travel
  to a graded assessment, because the harm available in the two places is not comparable.
- *Permission expires.* Every envelope carries an expiry and is recomputed from the clock
  on every decision. There is no cached "current authority" field, because a number that
  silently ages is the one thing an authority figure must never be.
- *Permission can be taken away, but not granted, by history.* Recorded human
  disagreements and failed outcomes decay a context's weight. Good outcomes restore it
  toward - but never past - the weight an untracked context starts from. A track record
  earns the right to act in a low-risk room. It never buys the right to write a mark.

**What this layer deliberately cannot do.** It cannot override a safety refusal, it cannot
act when no human is reachable, and it cannot learn from feedback. All three of those are
absences rather than omissions: each corresponds to a way a system with an authority
number in it tends to do real harm, and each is enforced structurally so that a future
caller cannot route around it with good intentions.

**Where it sits in the pipeline.** It runs after uncertainty and before policy. Policy
decides *whether* to act; the safety gate that follows decides *whether that specific
action may happen now*, and only the two of them together authorise anything.
"""

from __future__ import annotations

from focus_engine.authority.decay import DecayEngine, DecayReport
from focus_engine.authority.engine import (
    ESCALATION_CATALOGUE,
    AuthorityAssessment,
    AuthorityEngine,
)
from focus_engine.authority.feedback import (
    DisagreementRecord,
    FeedbackOutcome,
    FeedbackType,
    HumanAvailability,
    HumanFeedback,
)
from focus_engine.authority.ledger import AuthorityLedger, LedgerEntry, LedgerIntegrity
from focus_engine.authority.memory import (
    ContextMemory,
    InterventionOutcomeMemory,
    MemoryEntry,
)
from focus_engine.authority.safety import FinalAction, FinalActionOutcome, SafetyGate
from focus_engine.authority.types import (
    DEFAULT_PERMITTED_ACTIONS,
    ActionPermission,
    AuthorityEnvelope,
    AuthorityLevel,
    ContextDescriptor,
    ContextRisk,
    RestrictedAction,
    WithheldAction,
    confidence_ceiling,
)

__all__ = [
    "DEFAULT_PERMITTED_ACTIONS",
    "ESCALATION_CATALOGUE",
    "ActionPermission",
    "AuthorityAssessment",
    "AuthorityEngine",
    "AuthorityEnvelope",
    "AuthorityLedger",
    "AuthorityLevel",
    "ContextDescriptor",
    "ContextMemory",
    "ContextRisk",
    "DecayEngine",
    "DecayReport",
    "DisagreementRecord",
    "FeedbackOutcome",
    "FeedbackType",
    "FinalAction",
    "FinalActionOutcome",
    "HumanAvailability",
    "HumanFeedback",
    "InterventionOutcomeMemory",
    "LedgerEntry",
    "LedgerIntegrity",
    "MemoryEntry",
    "RestrictedAction",
    "SafetyGate",
    "WithheldAction",
    "confidence_ceiling",
]
