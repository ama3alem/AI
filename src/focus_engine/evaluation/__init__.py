"""The evaluation layer: was the prediction right, and what may be said about it.

This package answers a question no earlier layer asks. The baseline layer measures a learner,
the temporal layer describes how they are behaving, the uncertainty layer issues a claim, the
policy layer decides whether to act on it, and the outcome layer measures what a delivery was
followed by. None of them knows whether the claim was right, because knowing that requires
reading the events that came *after* the claim, and every one of those layers is in the
middle of producing them.

Two records come out of here.

A :class:`GroundTruth` is what the post-prediction window shows. Its
:class:`~focus_engine.evaluation.models.PredictionHorizon` is the only thing that decides
which events it may read, and the window is half-open -- the prediction instant is inside it
and the horizon boundary is outside it -- so two adjacent predictions cannot both claim the
same event. A ground truth that could not be read says so, and distinguishes *too few events*
from *events present but not characterisable*, because those two need different remedies and
collapsing them would let a corpus of unmeasurable windows report as a corpus of learners who
did not decline.

An :class:`EvaluationRecord` scores a prediction against that ground truth. It is the only
place in the project where the words correct and incorrect are applied to a claim about a
learner, and it applies them to the frozen claim rather than to a revised one.

What this package will not do is the one thing that would make its numbers impressive and
wrong. It does not rank learners, does not weight metrics into a single score, does not
revise a prediction, and does not claim a delivery caused anything it measured. Every record
it produces is stamped ``OBSERVED``: the events following a prediction are not a source of
truth external to the model, and this repository has no such source.
"""

from focus_engine.evaluation.aggregate import (
    ConfusionCounts,
    EvaluationSummary,
    group_by,
    group_by_intervention_response,
    group_by_learner,
    group_by_model_version,
    group_by_target,
    summarise,
)
from focus_engine.evaluation.engine import DEFAULT_EVALUATION_VERSION, EvaluationEngine
from focus_engine.evaluation.models import (
    EVALUATION_V1,
    BinaryVerdict,
    EvaluationRecord,
    EvaluationVerdict,
    GroundTruth,
    GroundTruthStatus,
    InterventionResponse,
    ObservationWindow,
    PredictionHorizon,
    PredictionSnapshot,
    PredictionTarget,
    claim_value,
    is_declining,
    response_from_outcome,
)

__all__ = [
    "DEFAULT_EVALUATION_VERSION",
    "EVALUATION_V1",
    "BinaryVerdict",
    "ConfusionCounts",
    "EvaluationEngine",
    "EvaluationRecord",
    "EvaluationSummary",
    "EvaluationVerdict",
    "GroundTruth",
    "GroundTruthStatus",
    "InterventionResponse",
    "ObservationWindow",
    "PredictionHorizon",
    "PredictionSnapshot",
    "PredictionTarget",
    "claim_value",
    "group_by",
    "group_by_intervention_response",
    "group_by_learner",
    "group_by_model_version",
    "group_by_target",
    "is_declining",
    "response_from_outcome",
    "summarise",
]
