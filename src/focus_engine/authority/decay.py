"""Deterministic decay of authority over time, disagreement, and outcomes.

**Why decay rather than a score.** A single "trust score" that only ever moves on explicit
feedback is a counter, not a judgement: it remembers that a human disagreed eleven weeks
ago with exactly the same force as one from yesterday, and it never notices that the
learner's context has quietly changed underneath it. Decay here is the standard
exponential half-life, applied *per entry* rather than to an aggregate, so a long history
of mixed results approaches a bounded floor instead of diverging, and a single old
failure genuinely stops counting.

**The ordering is the substance.** The weight is a minimum of three quantities, in this
order:

1. a **ceiling** derived from the uncertainty band, so a low-confidence prediction cannot
   buy high authority no matter what the history says;
2. the **base** weight, which is what a learner with no history in this context starts
   from, below ``1.0`` because a first impression is not a track record;
3. a **decayed** value obtained by subtracting age-weighted penalties and adding
   age-weighted credit.

Taking a minimum rather than multiplying is what makes the layer fail safe. A history of
excellent outcomes can restore a learner to the base weight but never past the ceiling
imposed by a low-confidence prediction, and no amount of good history makes a
``HUMAN_REQUIRED`` context autonomous.

**Unknown outcomes are penalised, not ignored.** An outcome string this module does not
recognise is treated as "no evidence of help". A system that silently discarded outcomes it
could not parse would be most trusting precisely when its logging has broken.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from focus_engine.authority.memory import ContextMemory
from focus_engine.authority.types import ContextDescriptor, confidence_ceiling
from focus_engine.configuration.thresholds import AuthoritySettings
from focus_engine.schemas.primitives import ConfidenceLevel
from focus_engine.utils.clock import Clock

__all__ = ["DecayEngine", "DecayReport"]

#: Outcomes that count against authority, mapped to the settings field that prices them.
_PENALTY_OUTCOMES: dict[str, str] = {
    "disputed": "disagreement_penalty",
    "deteriorated": "deterioration_penalty",
    "no_response": "nonresponse_penalty",
}

#: The one outcome that earns credit. Improvement is *not* treated as evidence that the
#: system's judgement is sound, only that the last action was not harmful.
_CREDIT_OUTCOME = "improved"


@dataclass(frozen=True, slots=True)
class DecayReport:
    """The full derivation of one authority weight.

    A single number is not auditable, so the intermediate quantities are returned
    alongside it. A reviewer who thinks the weight is wrong needs to see which of the
    three terms bound it, and that is only possible if the terms survive the calculation.

    Attributes:
        weight: The final authority weight. Bounded below by
            ``settings.min_authority_weight``, above by the band ceiling, and above by the
            base weight however good the history, so it always lies in
            ``[min_authority_weight, min(base, ceiling)]``.
        ceiling: The maximum the uncertainty band allowed.
        base: The weight a context with no history starts from.
        decayed: The age-weighted value before clamping to the floor and the ceiling.
        penalty_total: The summed age-weighted penalty applied.
        credit_total: The summed age-weighted credit applied.
        binding: Which term set the final value: ``ceiling``, ``base``, ``decayed``, or
            ``floor``.
        applied_penalties: The counted outcomes that cost authority, for the ledger.
        applied_credits: The counted outcomes that earned credit, for the ledger.
    """

    weight: float
    ceiling: float
    base: float
    decayed: float
    penalty_total: float
    credit_total: float
    binding: str
    applied_penalties: tuple[str, ...]
    applied_credits: tuple[str, ...]

    def to_summary_dict(self) -> dict[str, object]:
        """Render the report as flat, loggable fields.

        Returns:
            A dict of every field, with the outcome tallies flattened to lists.
        """
        return {
            "weight": self.weight,
            "ceiling": self.ceiling,
            "base": self.base,
            "decayed": self.decayed,
            "penalty_total": self.penalty_total,
            "credit_total": self.credit_total,
            "binding": self.binding,
            "applied_penalties": list(self.applied_penalties),
            "applied_credits": list(self.applied_credits),
        }


class DecayEngine:
    """Computes how much authority survives, given a band, a history, and a clock.

    The engine is stateless apart from its settings and clock. It reads a
    :class:`~focus_engine.authority.memory.ContextMemory` and returns a report; it never
    mutates the memory it is given. That separation is what allows a decision to be
    replayed: the same memory and the same clock produce the same weight, forever.

    Args:
        settings: The authority thresholds governing penalties, credit, and the floor.
        clock: The time source. Injected so decay is testable and reproducible.
    """

    __slots__ = ("_clock", "_settings")

    def __init__(self, settings: AuthoritySettings | None = None, *, clock: Clock) -> None:
        """Initialise the engine.

        Args:
            settings: Authority thresholds. Defaults to
                :class:`~focus_engine.configuration.thresholds.AuthoritySettings`.
            clock: The time source used to age entries.
        """
        self._settings = AuthoritySettings() if settings is None else settings
        self._clock = clock

    def evaluate(
        self,
        memory: ContextMemory,
        context: ContextDescriptor,
        confidence: ConfidenceLevel | None,
    ) -> DecayReport:
        """Compute the authority weight for one context and history.

        Args:
            memory: The recorded history for this learner *in this context*. Passing
                another context's memory here is the single most damaging mistake a caller
                can make, so the context is also an explicit argument and is checked
                against the memory's own key below.
            context: The context the decision would take effect in.
            confidence: The uncertainty band reported upstream.

        Returns:
            A :class:`DecayReport` giving the final weight and every term behind it.

        Raises:
            ValueError: If ``memory`` was built for a different context than the one
                supplied. Catching this at the boundary is the point: a caller that mixes
                contexts would otherwise produce a plausible, wrong, fully-auditable
                number.
        """
        if memory.context_type != context.context_type:
            raise ValueError(
                "authority memory and context must describe the same context; memory is "
                f"for {memory.context_type!r} but the decision is in "
                f"{context.context_type!r}. Authority does not transfer between contexts."
            )

        settings = self._settings
        now = self._clock.now()
        ceiling = confidence_ceiling(confidence)
        base = settings.base_confidence_weight

        penalty_total = 0.0
        credit_total = 0.0
        applied_penalties: list[str] = []
        applied_credits: list[str] = []

        for entry in memory.entries:
            survivorship = self._survivorship(entry.age_days(now), settings)
            effective = entry.weight * survivorship
            if effective <= 0.0:
                continue
            if entry.outcome == _CREDIT_OUTCOME:
                credit_total += settings.improvement_credit * effective
                applied_credits.append(f"{entry.action}:{entry.outcome}")
                continue
            field_name = _PENALTY_OUTCOMES.get(entry.outcome)
            if field_name is None:
                # An unrecognised outcome is evidence of nothing, and in a system deciding
                # how far to trust itself, "unknown" is not "harmless".
                field_name = "nonresponse_penalty"
            penalty_total += float(getattr(settings, field_name)) * effective
            applied_penalties.append(f"{entry.action}:{entry.outcome}")

        decayed = base - penalty_total + credit_total
        floor = settings.min_authority_weight

        # **Credit restores toward base and never past it.** A run of good outcomes is
        # evidence that the last actions were appropriate, not that the system's judgement
        # has become unimpeachable, so the base weight is a hard ceiling on what history
        # alone can achieve. Without this clamp, three improvements would lift a context
        # to full authority and the layer would be manufacturing permission - the exact
        # failure it exists to prevent.
        restored = min(decayed, base)

        # Clamp order matters. The floor bounds a long bad history so authority degrades to
        # "distrust" rather than to an unreadable negative; the ceiling then bounds the
        # result by what the uncertainty band allowed, and the floor is never allowed to
        # exceed the ceiling, or a permanently distrusted learner would regain authority
        # the moment a thin prediction arrived.
        if restored <= floor:
            weight, binding = floor, "floor"
        elif restored >= ceiling:
            weight, binding = ceiling, "ceiling"
        else:
            weight, binding = restored, "decayed"
        if weight > ceiling:
            weight, binding = ceiling, "ceiling"
        if credit_total > 0.0 and decayed > base and weight >= base:
            # History tried to lift this context past base and the cap stopped it. Report
            # the cap as the binding term, because that is the interesting fact.
            weight = min(weight, base)
            binding = "base"
        if not 0.0 <= weight <= 1.0:  # pragma: no cover - defensive
            raise ValueError(f"computed authority weight escaped [0, 1]: {weight!r}")

        return DecayReport(
            weight=weight,
            ceiling=ceiling,
            base=base,
            decayed=decayed,
            penalty_total=penalty_total,
            credit_total=credit_total,
            binding=binding,
            applied_penalties=tuple(applied_penalties),
            applied_credits=tuple(applied_credits),
        )

    def _survivorship(self, age_days: float, settings: AuthoritySettings) -> float:
        """Return the fraction of an entry's effect that still counts after ``age_days``.

        A textbook half-life, computed in log space for stability at extreme ages. At
        ``0`` it is ``1.0`` and at one half-life it is ``0.5``.

        Args:
            age_days: Age of the entry in days. Negative ages are treated as zero by the
                caller.
            settings: Supplies ``decay_half_life_hours``.

        Returns:
            A factor in ``(0.0, 1.0]``.
        """
        half_life_days = settings.decay_half_life_hours / 24.0
        return math.exp(-math.log(2.0) * age_days / half_life_days)
