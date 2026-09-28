"""Learner archetypes for the synthetic simulator.

An archetype is a *parameterisation of a behavioural signature*, not a personality type and
not a claim about any real learner. Each one describes how a synthetic learner's
observable interaction metrics move across the questions of a session: response latency,
accuracy, and session persistence.

Three constraints shape this module.

**Distinctness must be measurable, not asserted.** Each archetype exposes its signature
as a small set of numbers — baseline latency, drift shape, difficulty sensitivity, noise
scale. Two archetypes that would generate statistically indistinguishable output are a
defect, because an evaluation over indistinguishable populations cannot show whether a
model learned anything. :meth:`ArchetypeProfile.signature` returns those coefficients so
a test can compare them directly instead of trusting the labels.

**Unused parameters are defects.** Every field below is consumed by the generator or by an
explicitly named method. A coefficient that exists but does nothing would misrepresent the
simulator's behaviour to a reader of the generated data.

**No archetype is a mental state.** The names describe *patterns in generated interaction
data*. ``SLOW_BUT_ENGAGED`` means "generates longer response latencies while maintaining
accuracy and completing the requested question count". It does not mean the learner is
engaged, attentive, or motivated. These are descriptors of synthetic data, not of people.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

__all__ = [
    "ARCHETYPE_PROFILES",
    "ArchetypeProfile",
    "LatencyShape",
    "LearnerArchetype",
]


class LearnerArchetype(StrEnum):
    """Synthetic behavioural archetypes.

    Names describe the interaction pattern a simulated learner produces. They are
    engineering labels for test fixtures and carry no psychological meaning.
    """

    STABLE = "stable"
    """Latency and accuracy flat across the session; only per-item difficulty and noise
    move the values."""

    GRADUAL_DECLINE = "gradual_decline"
    """Latency rises and accuracy falls monotonically with question index."""

    NOISY = "noisy"
    """Large variance in every metric with no consistent direction. Mean behaviour is flat
    but individual observations scatter widely."""

    RECOVERY = "recovery"
    """Latency and accuracy degrade over the early questions, then return toward baseline
    by the end of the session."""

    INTERVENTION_RESISTANT = "intervention_resistant"
    """Degrades monotonically and shows no latency reduction after an intervention event."""

    FAST_BUT_INACCURATE = "fast_but_inaccurate"
    """Short response latencies coupled with low accuracy."""

    SLOW_BUT_ENGAGED = "slow_but_engaged"
    """Long response latencies coupled with high accuracy and near-certain completion."""

    CONTEXT_SENSITIVE = "context_sensitive"
    """Latency and accuracy track per-item difficulty far more strongly than other
    archetypes, with only mild drift across the session."""


class LatencyShape(StrEnum):
    """Shape of the degradation curve across a session.

    The shape decides how question index maps to a drift multiplier. It is a property of
    the archetype, not of the config, so that the eight archetypes remain a closed and
    exhaustive set of behaviours.
    """

    FLAT = "flat"
    """No index-dependent drift. Only difficulty and noise move the metrics."""

    LINEAR_DECLINE = "linear_decline"
    """Drift grows in proportion to question index."""

    DIP_AND_RECOVER = "dip_and_recover"
    """Drift grows to a peak partway through the session, then returns to zero by the
    final question."""


@dataclass(frozen=True, slots=True)
class ArchetypeProfile:
    """Behavioural coefficients for one archetype.

    Every field is a generator parameter, not a threshold and not a claim. Units are given
    in each attribute docstring so a reader can see the scale without reading the
    generator.

    Attributes:
        base_response_seconds: Latency multiplier numerator at neutral difficulty and zero
            drift, in seconds.
        base_accuracy: Accuracy at neutral difficulty and zero drift, in ``[0, 1]``.
        latency_shape: How drift accumulates across question index.
        latency_drift_per_question: Latency added per unit of drift progress, as a
            fraction of ``base_response_seconds``. ``0.02`` means a fully drifted item
            takes 2% longer per unit of progress.
        accuracy_drift_per_question: Accuracy change per unit of drift progress, in
            probability units. Negative values model degradation.
        difficulty_sensitivity: How strongly metrics track per-item difficulty, in
            ``[0, 0.9]``. At ``0.0`` difficulty is ignored; at ``0.6`` the easiest and
            hardest items differ substantially in both latency and accuracy.
        noise_scale: Standard deviation of multiplicative latency jitter, as a fraction of
            the latency value. ``0.0`` would disable jitter, so the floor is above zero.
        accuracy_noise: Standard deviation of additive accuracy jitter, in probability
            units.
        min_questions: Smallest session length at which this archetype's pattern is
            separable from noise. Sessions shorter than this are *generated* but must not
            be used to test pattern-based claims; see
            :meth:`pattern_is_measurable`.
        session_completion_rate: Probability of completing the full requested question
            count when dropout is enabled in the config. Never consulted when dropout is
            disabled, so a default run is exactly reproducible.
        response_to_intervention: Multiplier applied to drift progress for questions after
            an intervention. ``1.0`` means the intervention changes nothing, which is what
            ``INTERVENTION_RESISTANT`` requires by definition.
    """

    base_response_seconds: float
    base_accuracy: float
    latency_shape: LatencyShape
    latency_drift_per_question: float
    accuracy_drift_per_question: float
    difficulty_sensitivity: float
    noise_scale: float
    accuracy_noise: float
    min_questions: int
    session_completion_rate: float
    response_to_intervention: float = 1.0

    def signature(
        self,
    ) -> tuple[float, float, str, float, float, float, float, float, float, float]:
        """Return the comparable coefficient vector for this profile.

        Returns:
            A tuple of the behavioural coefficients, in a fixed order. Two profiles with
            equal signatures would be unable to produce distinguishable output.
        """
        return (
            self.base_response_seconds,
            self.base_accuracy,
            self.latency_shape.value,
            self.latency_drift_per_question,
            self.accuracy_drift_per_question,
            self.difficulty_sensitivity,
            self.noise_scale,
            self.accuracy_noise,
            self.session_completion_rate,
            self.response_to_intervention,
        )

    def pattern_is_measurable(self, question_count: int) -> bool:
        """Report whether a session of this length can support pattern-based claims.

        This is a *consumer-side* check, not a generation constraint. The simulator will
        happily emit a three-question session for an archetype whose pattern needs twelve;
        the point of this method is that analysis code must notice when it is about to
        draw a conclusion from too little data.

        Args:
            question_count: Number of answered questions in the session.

        Returns:
            ``True`` when ``question_count`` meets ``min_questions``.
        """
        return question_count >= self.min_questions


#: Per-archetype coefficients.
#:
#: These are **synthetic test-fixture parameters, not empirical estimates** of any human
#: population. They exist so later phases have data with known, documented structure to
#: test against. No inference about real learners may be drawn from them, and any model
#: fitted on this output is fitted on a simulator, not on behaviour.
#:
#: Drift coefficients are chosen so that a shaped archetype's peak departure from its own
#: baseline is several times its ``noise_scale``. A pattern buried under the noise it is
#: meant to be distinguishable from would make the distinctness requirement unsatisfiable
#: for reasons that have nothing to do with the consumer under test.
ARCHETYPE_PROFILES: Final[dict[LearnerArchetype, ArchetypeProfile]] = {
    LearnerArchetype.STABLE: ArchetypeProfile(
        base_response_seconds=20.0,
        base_accuracy=0.78,
        latency_shape=LatencyShape.FLAT,
        latency_drift_per_question=0.0,
        accuracy_drift_per_question=0.0,
        difficulty_sensitivity=0.25,
        noise_scale=0.08,
        accuracy_noise=0.03,
        min_questions=6,
        session_completion_rate=0.95,
    ),
    LearnerArchetype.GRADUAL_DECLINE: ArchetypeProfile(
        base_response_seconds=18.0,
        base_accuracy=0.80,
        latency_shape=LatencyShape.LINEAR_DECLINE,
        latency_drift_per_question=0.030,
        accuracy_drift_per_question=-0.010,
        difficulty_sensitivity=0.25,
        noise_scale=0.07,
        accuracy_noise=0.03,
        min_questions=10,
        session_completion_rate=0.85,
    ),
    LearnerArchetype.NOISY: ArchetypeProfile(
        base_response_seconds=20.0,
        base_accuracy=0.75,
        latency_shape=LatencyShape.FLAT,
        latency_drift_per_question=0.0,
        accuracy_drift_per_question=0.0,
        difficulty_sensitivity=0.25,
        noise_scale=0.45,
        accuracy_noise=0.18,
        min_questions=8,
        session_completion_rate=0.80,
    ),
    LearnerArchetype.RECOVERY: ArchetypeProfile(
        base_response_seconds=18.0,
        base_accuracy=0.78,
        latency_shape=LatencyShape.DIP_AND_RECOVER,
        latency_drift_per_question=0.09,
        accuracy_drift_per_question=-0.015,
        difficulty_sensitivity=0.25,
        noise_scale=0.08,
        accuracy_noise=0.03,
        min_questions=12,
        session_completion_rate=0.90,
        response_to_intervention=0.5,
    ),
    LearnerArchetype.INTERVENTION_RESISTANT: ArchetypeProfile(
        base_response_seconds=19.0,
        base_accuracy=0.77,
        latency_shape=LatencyShape.LINEAR_DECLINE,
        latency_drift_per_question=0.032,
        accuracy_drift_per_question=-0.011,
        difficulty_sensitivity=0.25,
        noise_scale=0.08,
        accuracy_noise=0.03,
        min_questions=10,
        session_completion_rate=0.85,
        response_to_intervention=1.0,
    ),
    LearnerArchetype.FAST_BUT_INACCURATE: ArchetypeProfile(
        base_response_seconds=6.0,
        base_accuracy=0.35,
        latency_shape=LatencyShape.FLAT,
        latency_drift_per_question=0.0,
        accuracy_drift_per_question=0.0,
        difficulty_sensitivity=0.20,
        noise_scale=0.12,
        accuracy_noise=0.06,
        min_questions=6,
        session_completion_rate=0.70,
    ),
    LearnerArchetype.SLOW_BUT_ENGAGED: ArchetypeProfile(
        base_response_seconds=45.0,
        base_accuracy=0.88,
        latency_shape=LatencyShape.FLAT,
        latency_drift_per_question=0.0,
        accuracy_drift_per_question=0.0,
        difficulty_sensitivity=0.25,
        noise_scale=0.06,
        accuracy_noise=0.02,
        min_questions=6,
        session_completion_rate=0.98,
    ),
    LearnerArchetype.CONTEXT_SENSITIVE: ArchetypeProfile(
        base_response_seconds=22.0,
        base_accuracy=0.75,
        latency_shape=LatencyShape.LINEAR_DECLINE,
        latency_drift_per_question=0.006,
        accuracy_drift_per_question=-0.002,
        difficulty_sensitivity=0.60,
        noise_scale=0.10,
        accuracy_noise=0.04,
        min_questions=8,
        session_completion_rate=0.88,
    ),
}
