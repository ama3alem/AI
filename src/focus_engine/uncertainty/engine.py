"""Where a probability, a confidence, and an explanation are decided to belong together.

This is the layer between "the model produced a number" and "something may act on it". Its
whole purpose is to answer a question the modelling layer deliberately declines to: not
*what did the model say*, but *should anyone believe it, and how much of it is there*.

**Confidence is the lowest of four independent ceilings, and the lowest one wins.** A
model's own confidence in its emitted class, the ceiling implied by its measured
calibration error, the ceiling implied by how much evidence stands behind this particular
inference, and the ceiling implied by the maturity of the baseline. They are combined by
minimum rather than by multiplication because a confidence derived by multiplication
invents an interaction that was not measured: it is not known that a miscalibrated model
on a thin history is quadratically less trustworthy, only that it is not more trustworthy
than either constraint alone allows. The minimum claims only what each input justifies.

**The maturity ceiling never sees the model's output.** It is a function of maturity and
configuration alone, so a model reporting ``0.99`` cannot raise a limit that exists to hold
it down. That is why :func:`~focus_engine.uncertainty.evidence.maturity_ceiling` takes no
prediction, and why the cap is applied to the confidence rather than to the probability: a
new learner's probability is still a real model output and worth recording, while a
confident *claim* about a learner with no history is not something this system is willing
to make.

**The order of the checks encodes which remedy applies.** Calibration is checked before
evidence volume, and the reason is not precedence for its own sake. An untrustworthy model
is a property of the system that no amount of data about this particular learner can
repair, so a learner with both problems is told ``UNKNOWN``: reporting
``INSUFFICIENT_DATA`` there would send an operator to collect data for a learner who
already has plenty while the model that cannot read it stays broken. Every negative answer
names its remedy, and the remedy is the part a caller acts on.

**This layer does not choose an intervention.** It reports a probability, a derived
confidence, and an explanation, and stops. Whether that should change a learner's
experience is a policy decision with its own restraints, and a layer holding a probability
has no standing to make it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from focus_engine.baseline.models import BaselineMaturity
from focus_engine.configuration.thresholds import BaselineSettings, UncertaintySettings
from focus_engine.features.models import FeatureValue
from focus_engine.models.artifacts import ModelArtifact
from focus_engine.models.predictors import (
    DEFAULT_THRESHOLD,
    Prediction,
    PredictionRefusal,
    predict_with_artifact,
)
from focus_engine.schemas.primitives import ConfidenceLevel
from focus_engine.uncertainty.calibration import CalibrationReport
from focus_engine.uncertainty.evidence import EvidenceVolume, maturity_ceiling
from focus_engine.uncertainty.explanation import (
    DEFAULT_MIN_CONTRIBUTION,
    PredictionExplanation,
    occlusion_contributions,
)
from focus_engine.uncertainty.outcomes import (
    ConfidenceAssessment,
    Constraint,
    OutcomeIdentity,
    PredictionOutcome,
    Verdict,
)
from focus_engine.utils.clock import Clock, SystemClock

__all__ = [
    "UncertaintyEngine",
    "band_confidence",
]


def band_confidence(
    value: float,
    settings: UncertaintySettings,
) -> ConfidenceLevel:
    """Place a confidence value in one of the three numeric bands.

    Bands are half-open and exhaust ``[0, 1]``, so every value has exactly one home and
    there is no residual case that silently gets no label.

    Args:
        value: The derived confidence.
        settings: The configuration holding the band boundaries.

    Returns:
        ``HIGH``, ``MEDIUM``, or ``LOW``.

    Raises:
        ValueError: If the value is outside ``[0, 1]``.
    """
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"confidence must lie in [0.0, 1.0]; got {value}")
    if value >= settings.high_confidence_threshold:
        return ConfidenceLevel.HIGH
    if value >= settings.medium_confidence_threshold:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW


@dataclass(frozen=True, slots=True)
class UncertaintyEngine:
    """Composes calibration, evidence, and maturity into one auditable answer.

    Attributes:
        uncertainty: The confidence thresholds and the evidence and calibration limits.
        baseline: The maturity ladder, so that a caller's evidence volume derives its
            maturity from the same definition the baseline engine uses rather than from a
            second one that could drift.
        clock: The source of the instant stamped on each outcome. Injected for the same
            reason the temporal, policy and outcome engines take one: a decision dated
            from wall-clock time cannot be replayed, and a replayed decision is what makes
            an auditable record possible.
    """

    uncertainty: UncertaintySettings = field(default_factory=UncertaintySettings)
    baseline: BaselineSettings = field(default_factory=BaselineSettings)
    clock: Clock = field(default_factory=SystemClock)

    def assess(
        self,
        prediction: Prediction | PredictionRefusal,
        *,
        artifact: ModelArtifact,
        evidence: EvidenceVolume,
        vector: FeatureValue | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        min_contribution: float = DEFAULT_MIN_CONTRIBUTION,
    ) -> PredictionOutcome:
        """Turn one model output into a complete, auditable answer.

        Args:
            prediction: What the modelling layer returned: a prediction or a refusal.
            artifact: The artifact that produced it, for identity and calibration.
            evidence: How much evidence stands behind this inference.
            vector: The scored vector, needed only to explain the prediction. Omitted
                yields an outcome whose explanation is unavailable rather than a guess.
            threshold: The threshold in force, recorded on the outcome.
            min_contribution: Smallest signal movement to report as a contribution.

        Returns:
            A :class:`~focus_engine.uncertainty.outcomes.PredictionOutcome`. A prediction
            is returned with a probability whenever the model produced one, the evidence
            supported it, and the model's calibration was believable. Otherwise the outcome
            names the verdict and the remedy that would change it.

        Raises:
            ValueError: If the evidence volume's maturity contradicts the baseline ladder
                in force, or if an explanation is requested for a vector whose schema
                disagrees with the artifact's.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must lie in [0.0, 1.0]; got {threshold}")

        record = artifact.record
        identity: OutcomeIdentity = {
            "model_id": record.model_id,
            "model_version": record.model_version,
            "feature_set_version": record.feature_set_version,
            "target_definition_version": record.target_definition_version,
            "data_origin": record.data_origin,
            "computed_at": self.clock.now(),
        }

        if isinstance(prediction, PredictionRefusal):
            return PredictionOutcome(
                verdict=Verdict.REFUSED,
                threshold=threshold,
                refusal_reason=prediction.reason,
                missing_features=prediction.missing_features,
                evidence_units=evidence.units,
                maturity=evidence.maturity,
                basis=evidence.basis(),
                **identity,
            )

        # Calibration is checked first, and deliberately so. It is the only one of the
        # three inputs that more data about this learner cannot fix, so when it fails the
        # remedy is a different model regardless of how much history the learner has.
        calibration = CalibrationReport.from_record(record)
        if not calibration.is_measured:
            return PredictionOutcome(
                verdict=Verdict.UNKNOWN,
                confidence=ConfidenceLevel.UNKNOWN,
                threshold=threshold,
                refusal_reason=(
                    f"{calibration.reason} The evidence behind this inference is not the "
                    f"problem: there are {evidence.units} observation(s) available, which is "
                    f"{'at or above' if evidence.is_sufficient(self.uncertainty) else 'below'} "
                    f"the {self.uncertainty.min_evidence_for_prediction} required to predict. "
                    "The model simply has not been shown to report its confidence honestly, "
                    "and an uncalibrated number is not worth restating at any confidence."
                ),
                evidence_units=evidence.units,
                maturity=evidence.maturity,
                basis=evidence.basis(),
                **identity,
            )
        if not calibration.is_believable(self.uncertainty.max_calibration_error):
            return PredictionOutcome(
                verdict=Verdict.UNKNOWN,
                confidence=ConfidenceLevel.UNKNOWN,
                threshold=threshold,
                refusal_reason=(
                    f"the model's measured calibration error is "
                    f"{calibration.expected_calibration_error:.4f} on {calibration.n_samples} "
                    f"held-out observations, above the {self.uncertainty.max_calibration_error} "
                    "above which its stated confidence is not usable. This is a property of "
                    "the model rather than of the learner, so collecting more data about this "
                    "learner would not change the answer; the model needs replacing or "
                    "retraining before its probabilities can be acted on."
                ),
                evidence_units=evidence.units,
                maturity=evidence.maturity,
                basis=evidence.basis(),
                **identity,
            )

        if not evidence.is_sufficient(self.uncertainty):
            return PredictionOutcome(
                verdict=Verdict.INSUFFICIENT_DATA,
                confidence=ConfidenceLevel.INSUFFICIENT_DATA,
                threshold=threshold,
                refusal_reason=(
                    f"the model produced a probability of {prediction.probability:.4f}, but it "
                    f"is not issued: only {evidence.units} observation(s) stand "
                    f"behind this inference, below the "
                    f"{self.uncertainty.min_evidence_for_prediction} required before any "
                    "probability is treated as meaningful. The remedy is more evidence about "
                    "this learner, not a different model: the model itself is calibrated and "
                    "would answer the same way with more history."
                ),
                evidence_units=evidence.units,
                maturity=evidence.maturity,
                basis=evidence.basis(),
                **identity,
            )

        explanation = self._explain(artifact, vector, prediction.probability, min_contribution)
        assessment = self._assess_confidence(prediction, evidence, calibration, explanation)
        # A prediction that no feature moved is not a finding about a learner, and the
        # system will not band it above LOW. The probability is left exactly as the model
        # produced it: the number is a fact about the model, while the band is this
        # system's claim about how far to trust it, and only the second is ours to give up.
        level = (
            ConfidenceLevel.LOW
            if explanation is not None and explanation.is_empty
            else assessment.level
        )

        return PredictionOutcome(
            verdict=Verdict.RESOLVED,
            confidence=level,
            assessment=assessment,
            probability=prediction.probability,
            is_positive=prediction.is_positive,
            threshold=threshold,
            basis=evidence.basis(),
            maturity=evidence.maturity,
            evidence_units=evidence.units,
            explanation=explanation,
            **identity,
        )

    def _assess_confidence(
        self,
        prediction: Prediction,
        evidence: EvidenceVolume,
        calibration: CalibrationReport,
        explanation: PredictionExplanation | None,
    ) -> ConfidenceAssessment:
        """Derive the confidence from the four independent ceilings.

        Args:
            prediction: The model output whose emitted class the confidence is about.
            evidence: The evidence volume behind the inference.
            calibration: The measured calibration report.
            explanation: The explanation, consulted only for the band and not for the
                number.

        Returns:
            The assessment, with every ceiling recorded and the binding one identified.
        """
        settings = self.uncertainty
        assertion = (
            prediction.probability if prediction.is_positive else 1.0 - prediction.probability
        )
        calibration_cap = calibration.ceiling()
        if calibration_cap is None:
            # Unreachable: the caller has already refused every unmeasured report. Asserted
            # rather than defaulted, because a silent 1.0 here would be an invented ceiling.
            raise ValueError(
                "an unmeasured calibration report reached confidence derivation, which "
                "should have been refused as UNKNOWN before this point"
            )

        evidence_cap = evidence.ceiling(settings)
        maturity_cap = maturity_ceiling(evidence.maturity, settings)

        ceilings = {
            Constraint.MODEL_ASSERTION: assertion,
            Constraint.CALIBRATION: calibration_cap,
            Constraint.EVIDENCE: evidence_cap,
            Constraint.MATURITY: maturity_cap,
        }
        # Ties are broken by insertion order, which follows the order above. A confidence
        # held down equally by two constraints is reported against the model's own claim
        # first, since that is the one an operator will think to check.
        binding = min(ceilings, key=lambda constraint: ceilings[constraint])
        value = ceilings[binding]

        return ConfidenceAssessment(
            value=value,
            level=band_confidence(value, settings),
            model_assertion=assertion,
            calibration_ceiling=calibration_cap,
            evidence_ceiling=evidence_cap,
            maturity_ceiling=maturity_cap,
            binding=binding,
            explanation_capped=explanation is not None and explanation.is_empty,
        )

    def _explain(
        self,
        artifact: ModelArtifact,
        vector: FeatureValue | None,
        probability: float,
        min_contribution: float,
    ) -> PredictionExplanation | None:
        """Produce the explanation, or record that one could not be.

        Args:
            artifact: The artifact that produced the probability.
            vector: The scored vector, or ``None`` if the caller did not supply one.
            probability: The probability the contributions are measured against.
            min_contribution: Smallest movement to report.

        Returns:
            The explanation, or ``None`` when no vector was supplied. A supplied vector
            always yields an explanation, which may itself be marked unavailable when the
            artifact carries no reference values.
        """
        if vector is None:
            return None
        return occlusion_contributions(
            artifact,
            vector,
            base_probability=probability,
            min_contribution=min_contribution,
        )

    def assess_vector(
        self,
        artifact: ModelArtifact,
        vector: FeatureValue,
        *,
        evidence: EvidenceVolume,
        threshold: float = DEFAULT_THRESHOLD,
        min_contribution: float = DEFAULT_MIN_CONTRIBUTION,
    ) -> PredictionOutcome:
        """Score a vector and assess the result in one call.

        This is the entry point a serving layer should use. It keeps the prediction and
        the explanation measured against the same encoded row, which is the failure a
        two-call arrangement invites: scoring the vector once and occluding it against a
        second, separately encoded copy would attribute the prediction to whichever copy
        happened to be encoded last.

        Args:
            artifact: The fitted artifact to score with.
            vector: The feature vector to score.
            evidence: How much evidence stands behind this inference.
            threshold: The probability at or above which the prediction is positive.
            min_contribution: Smallest signal movement to report as a contribution.

        Returns:
            The outcome, resolved or refused with its remedy.
        """
        return self.assess(
            predict_with_artifact(artifact, vector, threshold=threshold),
            artifact=artifact,
            evidence=evidence,
            vector=vector,
            threshold=threshold,
            min_contribution=min_contribution,
        )

    def maturity_for_units(self, observation_count: int) -> BaselineMaturity:
        """Derive a baseline maturity from an observation count.

        Exposed so a caller building an :class:`EvidenceVolume` uses this engine's
        configured ladder rather than a separately constructed settings object, and the
        two can never disagree.

        Args:
            observation_count: Accepted observations behind the inference.

        Returns:
            The maturity that count implies.
        """
        return EvidenceVolume.from_observation_count(
            observation_count, baseline=self.baseline
        ).maturity
