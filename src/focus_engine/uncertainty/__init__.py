"""The uncertainty engine: a probability is not an answer until it has been checked.

The modelling layer produces a number and stops, deliberately, because a model that
certified its own output would be a model whose confidence could not be questioned. This
package supplies what the modelling layer declines to: whether that number should be
believed, how far, and on what basis.

The three inputs to a confidence value are kept in separate modules because they fail
independently and are repaired independently. Calibration asks whether the model reports
its confidence honestly. Evidence volume asks whether there is enough here to say anything
at all. Baseline maturity asks whether what there is belongs to this learner's own record
rather than a pool. A caller that needed to know *which* of these was binding - because
that determines whether the answer changes with time or with engineering work - can read it
from :attr:`ConfidenceAssessment.binding`.

The public surface is small on purpose. :class:`UncertaintyEngine` is the composition point,
:class:`PredictionOutcome` is the answer, and the three ceiling functions are exposed so
they can be tested and reasoned about independently.
"""

from __future__ import annotations

from focus_engine.uncertainty.calibration import CalibrationReport, ReliabilityBin
from focus_engine.uncertainty.engine import UncertaintyEngine, band_confidence
from focus_engine.uncertainty.evidence import EvidenceVolume, maturity_ceiling
from focus_engine.uncertainty.explanation import (
    PredictionExplanation,
    SignalContribution,
    occlusion_contributions,
)
from focus_engine.uncertainty.outcomes import (
    REMEDIES,
    ConfidenceAssessment,
    Constraint,
    OutcomeIdentity,
    PredictionOutcome,
    Remedy,
    Verdict,
)

__all__ = [
    "REMEDIES",
    "CalibrationReport",
    "ConfidenceAssessment",
    "Constraint",
    "EvidenceVolume",
    "OutcomeIdentity",
    "PredictionExplanation",
    "PredictionOutcome",
    "ReliabilityBin",
    "Remedy",
    "SignalContribution",
    "UncertaintyEngine",
    "Verdict",
    "band_confidence",
    "maturity_ceiling",
    "occlusion_contributions",
]
