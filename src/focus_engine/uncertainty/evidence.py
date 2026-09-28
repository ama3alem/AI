"""Evidence volume and the maturity ladder as the two evidence-based limits on confidence.

This module supplies the other two of the three inputs to a confidence value. Where
:mod:`~focus_engine.uncertainty.calibration` asks *is the model honest about its own
output*, this asks *is there enough here to say anything, and is what there is old enough
to be about this learner*.

**Evidence volume and maturity are different questions, and conflating them is a trap.**
Evidence volume is a count: how many observations stand behind this particular
inference. Maturity is a judgement about that count relative to a ladder, and it exists
because a count on its own does not say whether the count belongs to one learner's long
history or is pooled across many thin ones. The gate in this module is expressed in
volume, the ceiling in maturity, and they are allowed to disagree - a vector with plenty
of volume behind it but a ``NEW`` baseline is held to the strictest confidence ceiling,
because volume pooled across learners is not personal evidence.

**The maturity ceiling is applied to the confidence, never to the probability.** This is
the distinction the whole ladder rests on. Refusing to state a probability because a
learner is new would throw away a real model output and produce a system that learns
nothing from its first week. The number is kept, and what is capped is how much the system
is willing to *claim* about it. A confident model therefore cannot launder a thin history
into a confident prediction: it can report ``0.97`` and be capped at ``0.30``, and both
facts are recorded together so the gap is visible rather than hidden.

**Personalisation has to be earned, and the basis records whether it has been.** The
:func:`~focus_engine.baseline.models.basis_for` mapping is reused unchanged rather than
re-derived here, so that the answer to "is this personal yet" is one definition in the
system rather than a convention repeated per call site.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from focus_engine.baseline.models import BaselineMaturity, basis_for
from focus_engine.baseline.statistics import maturity_for
from focus_engine.configuration.thresholds import BaselineSettings, UncertaintySettings
from focus_engine.schemas.primitives import InferenceBasis

__all__ = [
    "EvidenceVolume",
    "maturity_ceiling",
]


def maturity_ceiling(
    maturity: BaselineMaturity,
    settings: UncertaintySettings,
) -> float:
    """The highest confidence permitted by a baseline's maturity, alone.

    This is a pure function of maturity and configuration, with no reference to any model
    output, and that isolation is the point. A ceiling that consulted the model's own
    confidence could not be a ceiling: a model reporting ``0.99`` would raise the limit
    that exists to hold it down. Keeping the two apart is what makes the cap load-bearing
    rather than decorative, and it is what lets a test assert that an extremely confident
    model is still capped.

    ``ESTABLISHED`` has no ceiling. That is not a claim that an established learner's
    predictions are always reliable - the calibration and evidence ceilings still apply -
    but that history length has stopped being the limiting factor, and the baseline
    settings already state that the evidence-count cap no longer applies at this rung.

    Args:
        maturity: The baseline maturity to constrain.
        settings: The uncertainty configuration holding the ladder.

    Returns:
        The ceiling, in ``(0, 1]``. ``1.0`` for ``ESTABLISHED``.
    """
    match maturity:
        case BaselineMaturity.NEW:
            return settings.max_confidence_when_new
        case BaselineMaturity.EARLY:
            return settings.max_confidence_when_early
        case BaselineMaturity.DEVELOPING:
            return settings.max_confidence_when_developing
        case BaselineMaturity.ESTABLISHED:
            return 1.0


@dataclass(frozen=True, slots=True)
class EvidenceVolume:
    """How much evidence stands behind one inference, and how mature it is.

    Constructed from a count and the ladder rather than by guessing, so that the
    ``maturity`` field can never disagree with the volume it claims to summarise. Where a
    caller has a baseline engine's observation count,
    :meth:`from_observation_count` is the intended constructor.

    Attributes:
        units: Observations standing behind the inference.
        maturity: The maturity implied by those observations.
    """

    units: int
    maturity: BaselineMaturity

    def __post_init__(self) -> None:
        """Validate that the volume is a real count.

        Only the strongest contradiction is checked here. Whether a count is ``NEW`` or
        ``EARLY`` depends on the configured ladder, which this object does not hold, so
        a full consistency check belongs in :meth:`from_observation_count`, which does have
        the settings. What can be caught without them is a volume that claims an
        established record while carrying no observations at all, since no ladder puts
        zero observations at the top rung.

        Raises:
            ValueError: If the count is negative, or if a zero-observation volume claims
                to be ``ESTABLISHED``.
        """
        if self.units < 0:
            raise ValueError(f"evidence units must be non-negative; got {self.units}")
        if self.maturity is BaselineMaturity.ESTABLISHED and self.units == 0:
            raise ValueError(
                "an evidence volume of zero observations cannot be ESTABLISHED; the two "
                "fields contradict each other"
            )

    @classmethod
    def from_observation_count(
        cls,
        observation_count: int,
        *,
        baseline: BaselineSettings,
    ) -> EvidenceVolume:
        """Derive the volume and its maturity together from a raw count.

        Args:
            observation_count: Accepted observations behind the inference.
            baseline: The baseline maturity ladder in force.

        Returns:
            The evidence volume, with maturity derived by the baseline layer's own
            ``maturity_for`` so there is a single definition of the ladder.

        Raises:
            ValueError: If the count is negative.
        """
        return cls(
            units=observation_count,
            maturity=maturity_for(observation_count, baseline),
        )

    def is_sufficient(self, settings: UncertaintySettings) -> bool:
        """Whether enough evidence exists to emit a probability at all.

        Args:
            settings: The uncertainty configuration.

        Returns:
            ``True`` at or above ``min_evidence_for_prediction``.
        """
        return self.units >= settings.min_evidence_for_prediction

    def basis(self) -> InferenceBasis:
        """How personal this inference is permitted to be.

        Returns:
            The inference basis the baseline layer assigns to this maturity. ``NEW`` falls
            back to a population prior, and only ``DEVELOPING`` and above may be called
            personal.
        """
        return basis_for(self.maturity)

    def ceiling(self, settings: UncertaintySettings) -> float:
        """The confidence ceiling implied by evidence volume alone.

        The ramp is logarithmic between the gate and the saturation point, starting at
        ``0.5``. The shape is not a claim about how confidence should grow; it is the
        shape that is monotone, saturating, and does not require a count at which a
        learner is suddenly fully evidenced. Linear growth would say that twenty
        observations is half as trustworthy as forty, which is not a defensible statement
        about any particular measurement, and a step at the gate would produce a
        discontinuity where a learner crosses it and receives a large unexplained jump in
        claimed confidence.

        The floor of ``0.5`` is the honest minimum: at the gate, the evidence says only
        that this is worth saying, and not more than that. Above the saturation point
        evidence stops constraining confidence, and the calibration and maturity ceilings
        are what remain.

        Args:
            settings: The uncertainty configuration.

        Returns:
            A ceiling in ``[0.5, 1.0]`` at or above the gate, or ``0.0`` below it - below
            the gate there is no defensible confidence at all, which is what
            ``INSUFFICIENT_DATA`` reports.
        """
        gate = settings.min_evidence_for_prediction
        full = settings.full_confidence_evidence_units
        if self.units < gate:
            return 0.0
        if self.units >= full:
            return 1.0
        if gate == full:
            return 1.0
        return 0.5 + 0.5 * math.log(self.units / gate) / math.log(full / gate)

    def describe(self) -> str:
        """Render the volume as a short human-readable line.

        Returns:
            A one-line description of the count, its maturity, and its basis.
        """
        return (
            f"{self.units} observation(s), baseline {self.maturity.value}, "
            f"basis {self.basis().value}"
        )
