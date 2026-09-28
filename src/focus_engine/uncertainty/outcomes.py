"""The four answers the uncertainty engine can give, and what each one costs.

This module holds the vocabulary that makes the distinction the research principles
require: ``UNKNOWN`` and ``INSUFFICIENT_DATA`` are not two words for "no", and a system
that treats them as interchangeable has thrown away the only actionable part of a negative
answer. One needs more data about this learner; the other needs a better or different
model. Sending both to the same handler means one of them never gets the response that
would have helped.

**Four states, not two.** :class:`Verdict` adds :attr:`Verdict.REFUSED` alongside
``UNKNOWN`` and ``INSUFFICIENT_DATA``, because an input that cannot be encoded is neither
a thin history nor an untrustworthy model - it is a caller error, and its remedy is to fix
the payload. Collapsing a refusal into ``INSUFFICIENT_DATA`` would send an engineering
defect to a data-collection path, where nothing would ever be fixed.

**Confidence is a record of its own derivation, not a bare number.** A confidence of
``0.30`` presented on its own invites the reader to assume the model simply felt cautious.
:class:`ConfidenceAssessment` records all four ceilings that were applied and, critically,
which one *bound* - the constraint that actually set the value. "Capped at 0.30 because
the baseline is ``NEW``" is a different operational situation from "capped at 0.30 because
the model is miscalibrated", and only the first is fixed by waiting.

**A confidence without a probability is not withheld, it is absent.** For every verdict
other than ``RESOLVED`` the probability field is ``None``, and the dataclass refuses to be
constructed with a number in that position. A downstream consumer cannot accidentally read
a probability out of a verdict that declined to issue one, and a log line cannot present
``0.0`` as though the model had reported certainty of no risk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import TypedDict

from focus_engine.baseline.models import BaselineMaturity
from focus_engine.schemas.primitives import ConfidenceLevel, DataOrigin, InferenceBasis
from focus_engine.schemas.versioning import FeatureSetVersion, ModelVersion
from focus_engine.uncertainty.explanation import PredictionExplanation

__all__ = [
    "REMEDIES",
    "ConfidenceAssessment",
    "Constraint",
    "OutcomeIdentity",
    "PredictionOutcome",
    "Remedy",
    "Verdict",
]


class OutcomeIdentity(TypedDict):
    """The provenance fields every outcome carries, whatever verdict it reached.

    These six are set once from the artifact and are identical on all five return paths in
    the engine, which is why they are splatted rather than repeated. Declaring them as a
    :class:`~typing.TypedDict` rather than a plain ``dict`` is what lets a type checker
    confirm that a splat into :class:`PredictionOutcome` supplies exactly the right keys
    with exactly the right types - with a plain dict the values widen to ``object`` and the
    whole call goes unchecked, including the fields that matter.
    """

    model_id: str
    model_version: ModelVersion
    feature_set_version: FeatureSetVersion
    target_definition_version: str
    data_origin: DataOrigin
    computed_at: datetime | None


class Verdict(StrEnum):
    """The state an inference ended in.

    Three of the four are refusals to answer, and they are refusals for different reasons.
    """

    RESOLVED = "resolved"
    """A probability was issued, with a derived confidence and an explanation."""

    INSUFFICIENT_DATA = "insufficient_data"
    """There is not enough evidence about this learner to say anything.

    The remedy is more observations. Nothing about the model needs to change, and a
    different model would produce the same refusal, because the constraint is the volume of
    evidence behind this particular inference rather than anything to do with modelling.
    """

    UNKNOWN = "unknown"
    """The evidence is sufficient but the model cannot be trusted with it.

    The remedy is a different or better model, or a model whose calibration has actually
    been measured. Waiting will not help here, which is precisely why this state is kept
    apart from ``INSUFFICIENT_DATA``: a system that answered ``INSUFFICIENT_DATA`` here
    would tell an operator to collect more data until the learner had a very long record
    that the same broken model would still misjudge.
    """

    REFUSED = "refused"
    """The input was not usable, and no inference was attempted.

    The remedy is to fix the payload: a schema mismatch or an absent required feature. This
    is an engineering fault, not a finding about a learner, and it is counted separately
    for that reason.
    """


class Remedy(StrEnum):
    """What would have to change for a different answer to be available."""

    NONE = "none"
    """The verdict is ``RESOLVED``; there is nothing outstanding."""

    COLLECT_MORE_DATA = "collect_more_data"
    """More evidence about this learner would change the answer."""

    REPLACE_OR_RETRAIN_MODEL = "replace_or_retrain_model"
    """A model whose stated confidence can be believed would change the answer."""

    FIX_THE_INPUT = "fix_the_input"
    """A payload matching the model's schema, with its required features present, would
    change the answer."""


#: The single remedy that applies to each verdict, as a first-class mapping. Exposed so
#: that a caller resolving a verdict to a remedy cannot pick a mismatched one, and so a
#: test can assert the two vocabularies stay in step.
REMEDIES: dict[Verdict, Remedy] = {
    Verdict.RESOLVED: Remedy.NONE,
    Verdict.INSUFFICIENT_DATA: Remedy.COLLECT_MORE_DATA,
    Verdict.UNKNOWN: Remedy.REPLACE_OR_RETRAIN_MODEL,
    Verdict.REFUSED: Remedy.FIX_THE_INPUT,
}


class Constraint(StrEnum):
    """Which ceiling set the confidence value.

    Recorded because a confidence that cannot be attributed to a cause cannot be acted on.
    The same number arises from a waiting problem and an engineering problem, and the
    response differs completely.
    """

    MODEL_ASSERTION = "model_assertion"
    """The model's own confidence in the class it emitted, uncapped by anything."""

    CALIBRATION = "calibration"
    """The model has been shown to misstate its confidence by more than the configured
    tolerance."""

    EVIDENCE = "evidence"
    """Too little evidence stands behind this inference for higher confidence to be
    defensible."""

    MATURITY = "maturity"
    """The baseline is too thin to support a confident claim about an individual."""

    EXPLANATION = "explanation"
    """No feature measurably moved the prediction, so the system will not be confident
    about a number it cannot attribute to any signal."""


@dataclass(frozen=True, slots=True)
class ConfidenceAssessment:
    """A confidence value together with every ceiling that was applied to reach it.

    The four ceilings are stored rather than only the resulting value so that the value
    can be audited after the fact. Four separate mechanisms can produce the same number,
    and a system that reported only the number would leave an operator unable to tell a
    patient condition from a broken one.

    Attributes:
        value: The final confidence, which is the lowest of the four ceilings.
        level: The band the value falls in.
        model_assertion: The model's own confidence in the class it *emitted*, which is
            not always the class it favours. At a threshold above 0.5 a probability of 0.60
            emits a negative decision, and the model's confidence in that decision is
            ``0.40`` - it still believes the positive side is more likely. This is the
            normal state of a prediction near its own threshold, and it is recorded as it
            stands rather than replaced with ``max(p, 1 - p)``, which would report 0.60 and
            quietly describe a decision the system did not make.
        calibration_ceiling: The ceiling from measured calibration error, or ``1.0`` when
            calibration is good or unconstrained.
        evidence_ceiling: The ceiling from evidence volume.
        maturity_ceiling: The ceiling from baseline maturity. Independent of every other
            value here, and specifically not a function of ``model_assertion``.
        binding: The ceiling that actually set ``value``.
        explanation_capped: Whether an unexplicable prediction forced the band down to
            ``LOW`` regardless of the value. Recorded separately because it is a decision
            about how much to *claim*, not a ceiling on the number itself.
    """

    value: float
    level: ConfidenceLevel
    model_assertion: float
    calibration_ceiling: float
    evidence_ceiling: float
    maturity_ceiling: float
    binding: Constraint
    explanation_capped: bool = False

    def __post_init__(self) -> None:
        """Validate that the recorded ceilings actually produce the recorded value.

        Raises:
            ValueError: If the value or any ceiling is outside ``[0, 1]``, or if ``value``
                is not the minimum of the ceilings. A confidence that disagrees with its
                own derivation is the exact defect this record exists to make impossible,
                so it is refused rather than stored.
        """
        for name, ceiling in (
            ("value", self.value),
            ("calibration_ceiling", self.calibration_ceiling),
            ("evidence_ceiling", self.evidence_ceiling),
            ("maturity_ceiling", self.maturity_ceiling),
            ("model_assertion", self.model_assertion),
        ):
            if not 0.0 <= ceiling <= 1.0:
                raise ValueError(f"{name} must lie in [0.0, 1.0]; got {ceiling}")
        lowest = min(
            self.model_assertion,
            self.calibration_ceiling,
            self.evidence_ceiling,
            self.maturity_ceiling,
        )
        if abs(self.value - lowest) > 1e-9:
            raise ValueError(
                f"the recorded confidence {self.value} is not the lowest of its ceilings "
                f"({lowest}). A confidence that disagrees with its own derivation is "
                "untraceable, and the point of recording the ceilings is that the value can "
                "always be re-derived from them."
            )

    def describe(self) -> str:
        """Render the assessment as a short human-readable line.

        Returns:
            A one-line description naming the value, its band, and what bound it.
        """
        capped = ", band capped because nothing was attributable" if self.explanation_capped else ""
        return (
            f"confidence {self.value:.3f} ({self.level.value}), bound by "
            f"{self.binding.value}; model said {self.model_assertion:.3f}, calibration cap "
            f"{self.calibration_ceiling:.3f}, evidence cap {self.evidence_ceiling:.3f}, "
            f"maturity cap {self.maturity_ceiling:.3f}{capped}"
        )


@dataclass(frozen=True, slots=True)
class PredictionOutcome:
    """One complete answer from the uncertainty engine, including when it declines.

    Every field that could carry a number is optional, and the combination that is
    permitted is enforced rather than described. A ``RESOLVED`` outcome must carry a
    probability and an assessment; any other verdict must carry neither, and must say why
    it declined.

    Attributes:
        verdict: The state the inference ended in.
        confidence: The band, using the shared vocabulary. ``None`` only for a refusal,
            where nothing was scored and so no confidence was assessed.
        assessment: The full confidence derivation, when one was performed.
        probability: The emitted probability, when one was emitted.
        is_positive: Whether the probability met the threshold, when one was emitted.
        threshold: The threshold in force, always recorded so a threshold change is visible
            on outcomes produced under the old one.
        basis: How personal the inference was permitted to be.
        maturity: The baseline maturity the caps were derived from.
        evidence_units: Observations standing behind the inference.
        explanation: Which signals moved the prediction, when one could be produced.
        refusal_reason: Why no inference was attempted, for a refusal.
        missing_features: Required features that were absent, for a refusal.
        model_id: Identity of the model, when one was involved.
        model_version: Identity of the specific artifact, when one was involved.
        feature_set_version: The feature definitions in force.
        target_definition_version: The label definition the probability refers to. Recorded
            because ``0.8`` is uninterpretable without knowing what it is ``0.8`` *of*.
        data_origin: Whether the model behind this outcome was trained on real or
            synthetic data.
        computed_at: When the outcome was produced.
    """

    verdict: Verdict
    confidence: ConfidenceLevel | None = None
    assessment: ConfidenceAssessment | None = None
    probability: float | None = None
    is_positive: bool | None = None
    threshold: float = 0.5
    basis: InferenceBasis | None = None
    maturity: BaselineMaturity | None = None
    evidence_units: int = 0
    explanation: PredictionExplanation | None = None
    refusal_reason: str | None = None
    missing_features: tuple[str, ...] = field(default_factory=tuple)
    model_id: str = ""
    model_version: ModelVersion = ""
    feature_set_version: FeatureSetVersion = ""
    target_definition_version: str = ""
    data_origin: DataOrigin = DataOrigin.SYNTHETIC
    computed_at: datetime | None = None

    def __post_init__(self) -> None:
        """Validate that the fields agree with the verdict.

        Raises:
            ValueError: If a number is present for a verdict that did not emit one, if
                ``RESOLVED`` is missing its probability or assessment, or if the confidence
                band does not match the verdict.
        """
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must lie in [0.0, 1.0]; got {self.threshold}")

        if self.verdict is Verdict.RESOLVED:
            if self.probability is None or self.assessment is None:
                raise ValueError(
                    "a RESOLVED outcome must carry both a probability and a confidence "
                    f"assessment; got probability={self.probability}, "
                    f"assessment={self.assessment}"
                )
            if self.confidence is None:
                raise ValueError("a RESOLVED outcome must carry a confidence band")
            if self.confidence not in (
                ConfidenceLevel.HIGH,
                ConfidenceLevel.MEDIUM,
                ConfidenceLevel.LOW,
            ):
                raise ValueError(
                    f"a RESOLVED outcome must be banded high, medium, or low; got {self.confidence}"
                )
            if self.is_positive is None:
                raise ValueError("a RESOLVED outcome must record whether the threshold was met")
            return

        for name, value in (
            ("probability", self.probability),
            ("assessment", self.assessment),
            ("is_positive", self.is_positive),
        ):
            if value is not None:
                raise ValueError(
                    f"a {self.verdict.value} outcome must not carry {name}; it declined to "
                    f"issue one, and carrying {value!r} would let a consumer read a value "
                    "the engine never asserted."
                )
        if self.verdict is Verdict.REFUSED and self.confidence is not None:
            raise ValueError(
                "a refused outcome must carry no confidence band: nothing was scored, so no "
                f"confidence was assessed. Got {self.confidence}."
            )
        if self.verdict is Verdict.INSUFFICIENT_DATA and self.confidence is not (
            ConfidenceLevel.INSUFFICIENT_DATA
        ):
            raise ValueError(
                "an INSUFFICIENT_DATA outcome must be banded insufficient_data; got "
                f"{self.confidence}"
            )
        if self.verdict is Verdict.UNKNOWN and self.confidence is not ConfidenceLevel.UNKNOWN:
            raise ValueError(f"an UNKNOWN outcome must be banded unknown; got {self.confidence}")
        if self.verdict in (Verdict.UNKNOWN, Verdict.INSUFFICIENT_DATA) and not (
            self.refusal_reason
        ):
            raise ValueError(
                f"a {self.verdict.value} outcome must state why it declined; an unexplained "
                "negative answer cannot be acted on"
            )

    @property
    def remedy(self) -> Remedy:
        """What would have to change for a different answer to be available."""
        return REMEDIES[self.verdict]

    @property
    def is_resolved(self) -> bool:
        """Whether a probability was issued."""
        return self.verdict is Verdict.RESOLVED

    def describe(self) -> str:
        """Render the outcome as a short human-readable block.

        Returns:
            A multi-line summary of the verdict and, when resolved, the number behind it.
        """
        header = f"{self.verdict.value} (remedy: {self.remedy.value})"
        if not self.is_resolved:
            return f"{header}: {self.refusal_reason}"
        lines = [
            header,
            f"  probability {self.probability:.4f} against threshold {self.threshold:.2f} "
            f"-> {'positive' if self.is_positive else 'negative'}",
            f"  {self.assessment.describe()}" if self.assessment else "",
            f"  basis {self.basis.value if self.basis else 'unstated'}"
            f", {self.evidence_units} evidence unit(s)",
        ]
        lines = [line for line in lines if line]
        if self.explanation is not None:
            lines.append(f"  {self.explanation.describe()}")
        return "\n".join(lines)
