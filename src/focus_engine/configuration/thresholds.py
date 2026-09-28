"""Layer-scoped configuration.

Every tunable number in the engine lives here as a named, documented field. No module
anywhere else is permitted to embed a bare numeric literal in a decision: if a
threshold influences a prediction, a policy decision, or an uncertainty verdict, it is
declared in one of the settings objects below and can be traced to a rationale.

Two rules govern this module:

1. **Rationale over tuning.** Each field's docstring states why the initial value was
   chosen and what evidence would justify changing it. A number with no stated
   derivation is a defect, not a default.
2. **No hidden state.** These are frozen dataclasses. Overrides are explicit
   constructor arguments or explicit environment settings, never mutation of a module
   singleton. This is what makes a run reproducible from its configuration alone.

Initial values are *engineering starting points, not empirical findings*. Where a
value genuinely cannot be justified without data, the docstring says so and the field
is marked accordingly. Nothing here is a validated threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise

__all__ = [
    "AuthoritySettings",
    "BaselineSettings",
    "EvaluationSettings",
    "FeatureSettings",
    "InterventionSettings",
    "OutcomeSettings",
    "SimulationSettings",
    "TemporalSettings",
    "UncertaintySettings",
]


@dataclass(frozen=True, slots=True)
class FeatureSettings:
    """Configuration for the feature engine (Phase 5).

    Attributes:
        short_window_minutes: Width of the recent window used for fast-moving features
            such as trend and rate features. Initial value is a judgement call: it is
            intended to be long enough to average out single-question noise and short
            enough to react within a few questions. To be revised against measured
            feature stability, not tuned for a better score.
        long_window_minutes: Width of the slower comparison window. Must exceed
            ``short_window_minutes``; the feature engine cross-references short-window
            values against long-window values to express change.
        interaction_gap_seconds_floor: Lower bound applied to interaction gaps before
            rate features are derived. Prevents a division by ~0 when two events share a
            timestamp and produces an unbounded rate. Not a behavioural judgement.
        min_events_for_window: A window with fewer events than this yields
            ``INSUFFICIENT_DATA`` for rate-based features rather than a computed value.
            Guards against confident-looking statistics from two data points.
    """

    short_window_minutes: float = 15.0
    long_window_minutes: float = 60.0
    interaction_gap_seconds_floor: float = 0.5
    min_events_for_window: int = 3

    def __post_init__(self) -> None:
        """Validate cross-field constraints.

        Raises:
            ValueError: If the long window is not strictly longer than the short window,
                or if any value is non-positive.
        """
        if self.long_window_minutes <= self.short_window_minutes:
            raise ValueError(
                f"long_window_minutes ({self.long_window_minutes}) must exceed "
                f"short_window_minutes ({self.short_window_minutes}); the long window exists "
                "to provide a comparison horizon for the short window"
            )
        if self.short_window_minutes <= 0:
            raise ValueError("short_window_minutes must be positive")
        if self.interaction_gap_seconds_floor <= 0:
            raise ValueError("interaction_gap_seconds_floor must be positive")
        if self.min_events_for_window < 1:
            raise ValueError("min_events_for_window must be at least 1")


@dataclass(frozen=True, slots=True)
class BaselineSettings:
    """Configuration for the personal baseline engine (Phase 6).

    The maturity ladder encodes the central cold-start requirement: a learner observed
    for thirty seconds must not receive a baseline confidence comparable to one observed
    for three weeks.

    Attributes:
        min_samples_new: Below this many observations the baseline is ``NEW`` and
            inference falls back to the population prior.
        min_samples_early: At or above this, ``EARLY``. Personal statistics are computed
            but treated as provisional, and predictions are capped at low confidence.
        min_samples_developing: At or above this, ``DEVELOPING``. Personal statistics
            are used and moderate confidence is permitted.
        min_samples_established: At or above this, ``ESTABLISHED``. No evidence-count
            cap on confidence is applied; confidence is then driven by model
            calibration and data quality rather than history length.
        ewma_alpha: Exponential-weighting factor for the streaming update, in
            ``(0, 1]``. Higher values react faster and forget faster; lower values are
            stabler and adapt slower. The initial value favours stability because an
            unstable baseline injects noise into every downstream deviation calculation.
        use_robust_statistics: When true, both the centre and the spread are estimated
            robustly: the centre update is winsorised, and the spread is a median absolute
            deviation rather than a standard deviation. When false, the centre is a plain
            exponential mean and the spread a standard deviation. Recommended whenever
            observation counts are small or heavy-tailed.
        winsorisation_limit_spreads: Bounds the per-observation step of the centre, in
            units of the current spread, when ``use_robust_statistics`` is set. This is the
            control that stops one extreme observation from redefining "normal": an
            exponentially weighted mean is not robust, so without a bound on its step a
            single 300-second outlier drags the baseline far enough that the following
            hour of ordinary behaviour reads as a change. The initial value of five is a
            compromise between two failure modes. Set it too low and a genuine change of
            a few dispersions takes many observations to register; set it too high and an
            extreme value passes through almost unattenuated. Five allows a genuine shift
            to register within a handful of observations while bounding a value hundreds
            of dispersions out to a barely perceptible move. Revisit against measured
            response-time distributions, not for a better score.
        max_observations_retained: Cap on retained observations per dimension, bounding
            memory for long-lived learners. Older observations are dropped by a
            documented decimation rule rather than silently discarded.
    """

    min_samples_new: int = 0
    min_samples_early: int = 20
    min_samples_developing: int = 100
    min_samples_established: int = 500
    ewma_alpha: float = 0.05
    use_robust_statistics: bool = True
    winsorisation_limit_spreads: float = 5.0
    max_observations_retained: int = 2000

    def __post_init__(self) -> None:
        """Validate that the maturity ladder is strictly increasing and usable.

        Raises:
            ValueError: If thresholds are non-increasing, negative, or if ``ewma_alpha``
                is outside ``(0, 1]``.
        """
        ladder = [
            self.min_samples_new,
            self.min_samples_early,
            self.min_samples_developing,
            self.min_samples_established,
        ]
        if any(count < 0 for count in ladder):
            raise ValueError("maturity thresholds must be non-negative")
        if any(b <= a for a, b in pairwise(ladder)):
            raise ValueError(f"maturity thresholds must be strictly increasing; got {ladder}")
        if not 0.0 < self.ewma_alpha <= 1.0:
            raise ValueError(f"ewma_alpha must lie in (0.0, 1.0]; got {self.ewma_alpha}")
        if self.winsorisation_limit_spreads <= 0.0:
            raise ValueError(
                "winsorisation_limit_spreads must be positive; a limit of zero would freeze "
                "the centre permanently and a negative limit would invert the update"
            )
        if self.max_observations_retained < 1:
            raise ValueError("max_observations_retained must be at least 1")


@dataclass(frozen=True, slots=True)
class TemporalSettings:
    """Configuration for the temporal state engine (Phase 7).

    The engine's central job is separating a temporary anomaly from a sustained
    trajectory change. These parameters are what make that distinction possible, and
    they are the parameters most likely to need empirical revision once real
    longitudinal data exists.

    Attributes:
        state_min_observations: Number of consecutive state observations required before
            a state other than ``STABLE`` or ``INSUFFICIENT_DATA`` may be asserted. One
            anomalous observation is an anomaly; a run of them is a trajectory.
        sustained_change_min_observations: Consecutive observations of directional
            deviation required to classify a change as sustained rather than temporary.
        rate_of_change_window: Number of trailing observations used to estimate the rate
            of change, so a single large step does not read as a high rate.
        stability_window: Number of trailing observations used to measure behavioural
            stability as inverse dispersion. A wider window measures stability more
            reliably and reacts more slowly.
        recovery_epsilon: A deviation must fall back below this fraction of its prior
            magnitude to count as recovering. Prevents oscillation around the baseline
            from being labelled as alternating decline and recovery.
        material_deviation_min: A standardised deviation smaller than this, in
            dispersions, is treated as arithmetic noise rather than a behavioural fact.
            Its job is to exclude numerical dust, not to decide what counts as a change,
            which is why the value is tiny: a learner behaving exactly as their own
            baseline should not be able to satisfy a persistence requirement through
            rounding alone.
        high_deviation_standardised: The standardised magnitude at or above which a
            sustained run is reported as ``HIGH_DEVIATION`` rather than as a developing
            or modest one. Two dispersions is an engineering starting point chosen to sit
            clearly outside ordinary within-person variability, not an empirically
            derived boundary. Revisit against measured within-learner distributions.
        max_points_retained: Cap on retained deviation points per dimension, bounding
            memory for long-lived learners. The value must be large enough to hold the
            longest window the other thresholds define, and a reviewer can check it is by
            inspecting ``max(ratio_change_window * 4, stability_window * 4)``.
        max_trajectories_retained: Cap on recorded trajectories per dimension. Trajectories
            are retained separately from the points behind them, because a run that
            outlasts the point window must still be reportable: the points age out, and the
            record that the departure happened does not.
    """

    state_min_observations: int = 2
    sustained_change_min_observations: int = 3
    rate_of_change_window: int = 5
    stability_window: int = 10
    recovery_epsilon: float = 0.5
    material_deviation_min: float = 0.01
    high_deviation_standardised: float = 2.0
    max_points_retained: int = 40
    max_trajectories_retained: int = 50

    def __post_init__(self) -> None:
        """Validate the temporal parameters.

        Raises:
            ValueError: If any window is smaller than one observation, if the recovery
                threshold is outside ``(0, 1]``, or if any derived quantity is invalid.
        """
        if self.state_min_observations < 1:
            raise ValueError("state_min_observations must be at least 1")
        if self.sustained_change_min_observations < 2:
            raise ValueError(
                "sustained_change_min_observations must be at least 2; a run of one "
                "observation has no direction and no rate, so a trajectory built from it "
                "would assert a change from a single measurement"
            )
        if self.rate_of_change_window < 1:
            raise ValueError("rate_of_change_window must be at least 1")
        if self.stability_window < 1:
            raise ValueError("stability_window must be at least 1")
        if not 0.0 < self.recovery_epsilon <= 1.0:
            raise ValueError(
                f"recovery_epsilon must lie in (0.0, 1.0]; got {self.recovery_epsilon}"
            )
        if self.material_deviation_min <= 0.0:
            raise ValueError(
                "material_deviation_min must be positive; a value of zero would let exact "
                "zero deviations register as behavioural facts"
            )
        if self.high_deviation_standardised <= 0.0:
            raise ValueError(
                "high_deviation_standardised must be positive; a non-positive threshold "
                "would classify every run as a high deviation"
            )
        min_retained = max(self.rate_of_change_window * 4, self.stability_window * 4)
        if self.max_points_retained < min_retained:
            raise ValueError(
                f"max_points_retained ({self.max_points_retained}) must be at least "
                f"{min_retained}; the retention cap must be large enough to hold the longest "
                "window defined by the other parameters"
            )
        if self.sustained_change_min_observations > self.max_points_retained:
            raise ValueError(
                f"sustained_change_min_observations "
                f"({self.sustained_change_min_observations}) exceeds max_points_retained "
                f"({self.max_points_retained}); a run is measured from the retained window, "
                "so a minimum above the retention cap could never be reached and no "
                "trajectory could ever be recorded"
            )
        if self.max_trajectories_retained < 1:
            raise ValueError(
                "max_trajectories_retained must be at least 1; a cap of zero would discard "
                "the record of every departure as soon as it was observed"
            )


@dataclass(frozen=True, slots=True)
class UncertaintySettings:
    """Configuration for the uncertainty engine (Phase 9).

    The engine must be able to answer ``UNKNOWN`` and ``INSUFFICIENT_DATA``. These
    settings define when, and they are the guard against a fabricated confidence number.

    Attributes:
        min_evidence_for_prediction: Minimum evidence units required before any
            probability is emitted at all. Below this the engine returns
            ``INSUFFICIENT_DATA`` rather than a number. The single most important
            anti-fabrication control in the system.
        calibration_bins: Number of equal-width bins used for expected calibration error
            and reliability reporting.
        high_confidence_threshold: Lower bound of the highest confidence band, applied to
            the calibrated probability of the emitted class together with the evidence
            and baseline-maturity caps below.
        medium_confidence_threshold: Lower bound of the medium confidence band.
        max_confidence_when_new: Ceiling applied while the personal baseline is ``NEW``,
            regardless of model output. The strictest rung of the ladder, and separate
            from the ``EARLY`` ceiling on purpose: ``NEW`` means inference is falling back
            to a population prior, and a population prior cannot support a confident claim
            about an individual no matter how thin or thick their own record is. Capping
            ``NEW`` at the ``EARLY`` value would make a learner with no usable history
            indistinguishable from one with twenty observations.
        max_confidence_when_early: Ceiling applied while the personal baseline is
            ``EARLY``, regardless of model output.
        max_confidence_when_developing: Ceiling applied while the baseline is
            ``DEVELOPING``.
        full_confidence_evidence_units: Evidence volume at which the evidence ceiling
            reaches 1.0. Below ``min_evidence_for_prediction`` the engine refuses to emit
            a probability at all; between the two the evidence ceiling rises
            logarithmically, and at or above this value evidence stops constraining
            confidence and the calibration and maturity ceilings are what remain. This is
            a *saturation point*, not a target: it says how much history is enough for
            evidence volume to stop being the limiting factor, and a larger value means
            more of the ceiling comes from how well calibrated the model is.
        max_calibration_error: Expected calibration error above which the model's own
            stated confidence cannot be believed at any evidence volume, and the engine
            answers ``UNKNOWN`` rather than a number. This is a line of *unbelief*, not a
            quality target: an ECE of 0.25 means the model's stated confidence is off by
            twenty-five percentage points on average, and no amount of additional data
            about one learner repairs a model that reports itself that inaccurately.
            ``UNKNOWN`` is correct there because the remedy is a better or different
            model, which is exactly what the distinction from ``INSUFFICIENT_DATA`` is
            for.
        probability_floor: Numerical floor applied to an emitted probability. Prevents
            log-loss divergence and reflects that no behavioural probability estimated
            from a finite sample is effectively exactly zero.
    """

    min_evidence_for_prediction: int = 10
    calibration_bins: int = 10
    high_confidence_threshold: float = 0.80
    medium_confidence_threshold: float = 0.60
    max_confidence_when_new: float = 0.30
    max_confidence_when_early: float = 0.50
    max_confidence_when_developing: float = 0.75
    full_confidence_evidence_units: int = 500
    max_calibration_error: float = 0.25
    probability_floor: float = 1e-6

    def __post_init__(self) -> None:
        """Validate the uncertainty thresholds and their ordering.

        Raises:
            ValueError: If thresholds are non-monotonic, out of range, or if
                ``probability_floor`` is not in ``(0, 0.5)``.
        """
        if not 0.0 < self.medium_confidence_threshold < self.high_confidence_threshold <= 1.0:
            raise ValueError(
                "confidence thresholds must satisfy "
                "0 < medium < high <= 1; got "
                f"medium={self.medium_confidence_threshold}, high={self.high_confidence_threshold}"
            )
        if self.min_evidence_for_prediction < 1:
            raise ValueError("min_evidence_for_prediction must be at least 1")
        if self.calibration_bins < 2:
            raise ValueError("calibration_bins must be at least 2")
        if self.full_confidence_evidence_units < self.min_evidence_for_prediction:
            raise ValueError(
                f"full_confidence_evidence_units ({self.full_confidence_evidence_units}) must be "
                f"at least min_evidence_for_prediction ({self.min_evidence_for_prediction}). The "
                "saturation point cannot sit below the gate that permits a prediction at all: "
                "evidence would stop constraining confidence before it was sufficient to "
                "produce one, and the evidence ceiling would never be consulted at all."
            )
        if not 0.0 < self.max_calibration_error < 1.0:
            raise ValueError(
                f"max_calibration_error must lie in (0.0, 1.0); got {self.max_calibration_error}"
            )
        if not 0.0 < self.probability_floor < 0.5:
            raise ValueError(
                f"probability_floor must lie in (0.0, 0.5); got {self.probability_floor}"
            )

        # The maturity caps form a ladder, and the ladder is the mechanism by which a
        # confident model is prevented from laundering a thin history into a confident
        # prediction. It only does that if the rungs ascend, so the ordering is validated
        # rather than documented-and-hoped-for.
        ladder = (
            ("NEW", self.max_confidence_when_new),
            ("EARLY", self.max_confidence_when_early),
            ("DEVELOPING", self.max_confidence_when_developing),
        )
        for name, cap in ladder:
            if not 0.0 < cap <= 1.0:
                raise ValueError(
                    f"max_confidence_when_{name.lower()} must lie in (0.0, 1.0]; got {cap}"
                )
        for (lower_name, lower), (upper_name, upper) in pairwise(ladder):
            if lower > upper:
                raise ValueError(
                    "the maturity confidence ladder must be non-decreasing: "
                    f"max_confidence_when_{lower_name.lower()} ({lower}) must not exceed "
                    f"max_confidence_when_{upper_name.lower()} ({upper}). A more established "
                    "baseline may not be capped lower than a less mature one, or the ladder "
                    "would narrow confidence as evidence accumulates."
                )


@dataclass(frozen=True, slots=True)
class InterventionSettings:
    """Configuration for the intervention policy (Phase 10).

    The policy optimises for *useful* intervention, not maximal intervention. Every
    field below is a restraint on acting, not an encouragement.

    Attributes:
        min_probability_to_act: Minimum calibrated decline probability before any
            intervention may be considered. Prevents acting on noise.
        cooldown_minutes: Minimum gap between two interventions for the same learner. The
            primary protection against repeatedly interrupting someone who is working.
        max_interventions_per_session: Hard per-session cap.
        max_interventions_per_hour: Sliding-window cap, which bounds the rate across
            session boundaries.
        max_repeat_same_type: Maximum consecutive deliveries of the same intervention
            type before a different type must be tried. Prevents a stuck loop.
        min_response_observations: Minimum measured outcomes for the same learner and
            intervention type before response statistics are treated as informative.
            Below this, response data is recorded but not used to select an intervention.
        abandon_after_no_response_count: Number of consecutive non-responses after which
            that intervention type is retired for the learner.
        min_evidence_to_act: Minimum observations the policy requires behind a decision
            before it will act. This is a *policy* floor and is deliberately separate from
            the uncertainty engine's ``min_evidence_for_prediction``: a probability may be
            worth reporting at five observations and still be too thin to justify
            interrupting a learner. Named separately because the two ceilings answer
            different questions, and collapsing them would make the stricter one silently
            disappear.
    """

    min_probability_to_act: float = 0.65
    cooldown_minutes: float = 10.0
    max_interventions_per_session: int = 3
    max_interventions_per_hour: int = 5
    max_repeat_same_type: int = 2
    min_response_observations: int = 3
    abandon_after_no_response_count: int = 2
    min_evidence_to_act: int = 5

    def __post_init__(self) -> None:
        """Validate the policy restraints.

        Raises:
            ValueError: If the action threshold is out of range, the cooldown is not
                positive, or any count limit is below one.
        """
        if not 0.0 < self.min_probability_to_act < 1.0:
            raise ValueError(
                f"min_probability_to_act must lie in (0.0, 1.0); got {self.min_probability_to_act}"
            )
        if self.cooldown_minutes <= 0:
            raise ValueError("cooldown_minutes must be positive")
        for name, count in (
            ("max_interventions_per_session", self.max_interventions_per_session),
            ("max_interventions_per_hour", self.max_interventions_per_hour),
            ("max_repeat_same_type", self.max_repeat_same_type),
            ("min_response_observations", self.min_response_observations),
            ("abandon_after_no_response_count", self.abandon_after_no_response_count),
            ("min_evidence_to_act", self.min_evidence_to_act),
        ):
            if count < 1:
                raise ValueError(f"{name} must be at least 1; got {count}")


@dataclass(frozen=True, slots=True)
class OutcomeSettings:
    """Configuration for the outcome engine (Phase 11).

    Every field here defines *what gets measured*, not how a measurement is judged to
    have succeeded. The engine makes no causal claim and sets no target, so there is
    deliberately no field for "how much improvement counts as a win" — a value like that
    would be a policy decision wearing a measurement's clothes, and the layer that owns
    such a decision is Phase 12, not this one.

    The window widths are the consequential fields. They determine which events are
    inside a measurement at all, which is why they are versioned alongside the measure
    set rather than treated as presentation.

    Attributes:
        before_window_minutes: Width of the comparison window immediately *preceding*
            delivery. Long enough to describe the learner's ordinary rate rather than
            one event, and short enough that the behaviour is plausibly comparable to the
            minutes after the prompt. Initial value is an engineering starting point: it
            assumes an intervention is issued in the middle of an activity burst, so the
            preceding half hour is representative. To be revised against measured
            within-learner stability, not tuned until outcomes look better.
        after_window_minutes: Width of the post-delivery window. Shorter than the before
            window because the layer deliberately does not claim credit or blame for
            behaviour minutes or hours after a prompt; a longer window would make
            ordinary session drift look like an intervention effect.
        immediate_window_minutes: Width of the short window at the very start of the
            after window, used for the immediate-interaction measure. It exists to
            separate a burst of resumed interaction from sustained engagement, which are
            different outcomes and which a single after-window rate cannot tell apart.
            Must not exceed ``after_window_minutes``.
        min_samples_for_comparison: Minimum activity events required in *both* windows
            before a rate comparison is reported at all. Below this a rate is arithmetic
            on one, two, or three events and reads as a behavioural fact. A window with
            fewer events is reported as ``INSUFFICIENT_DATA`` with a reason, never as a
            rate of zero, because a rate of zero derived from no observations is
            indistinguishable from a rate of zero derived from a learner who did nothing.
            The default of four is the point at which a half-hour window stops describing a
            rhythm: three events over thirty minutes is a learner who paused, and calling
            the pause a rate would hand the comparison an apparent baseline that the
            learner never actually had. One question alone is worth three activity events,
            because answering it starts the question and opens an interaction around it, so
            a single question answered inside a thirty-minute window is one click and not a
            rate.
        min_relative_change: Smallest relative difference between the before and after
            values that is reported as a change rather than as no change. A rate of
            100.0 and 103.0 events per minute is noise, and reporting it as an
            improvement would let ordinary sampling variation accumulate into a claimed
            pattern. Compared against ``max(|before|, 1.0)`` so the test stays defined
            when the before value is exactly zero.
        tail_minutes: **Reserved and not consumed by any measure in OUTCOME_V1.** It was
            specified as the length of the end of the after window to be treated as "still
            engaged" for the continued-activity measure, so that one inactive minute at the
            close of a window would not read as having disengaged. No measure reads it:
            continued activity is currently a straight rate over the whole after window, and
            a reading that a learner went quiet for the final minute of a fifteen-minute
            window cannot be derived from the events at all, because the window's end says
            nothing about what followed it. The field is kept, and validated, so that a
            future ``OUTCOME_V2`` can add the behaviour without a settings change that
            silently alters what an existing configuration meant - changing a field's
            meaning under a version that claims not to have changed is the failure this
            repository's versioning rules exist to prevent. Until a measure reads it, a
            value set here has no effect on any reading, and that is stated rather than
            left to be discovered from a measurement that ignored it.
    """

    before_window_minutes: float = 30.0
    after_window_minutes: float = 15.0
    immediate_window_minutes: float = 2.0
    min_samples_for_comparison: int = 4
    min_relative_change: float = 0.05
    tail_minutes: float = 1.0

    def __post_init__(self) -> None:
        """Validate the measurement geometry.

        Raises:
            ValueError: If any window is not positive, if the immediate window exceeds
                the after window, if the change tolerance is outside ``[0.0, 1.0)``, or
                if the sample floor is below one.
        """
        for name, minutes in (
            ("before_window_minutes", self.before_window_minutes),
            ("after_window_minutes", self.after_window_minutes),
            ("immediate_window_minutes", self.immediate_window_minutes),
            ("tail_minutes", self.tail_minutes),
        ):
            if minutes <= 0.0:
                raise ValueError(f"{name} must be positive; got {minutes}")
        if self.immediate_window_minutes > self.after_window_minutes:
            raise ValueError(
                f"immediate_window_minutes ({self.immediate_window_minutes}) must not exceed "
                f"after_window_minutes ({self.after_window_minutes}); the immediate window is "
                "a prefix of the after window, and a longer prefix would make the immediate "
                "measure read events the after measure cannot see"
            )
        if not 0.0 <= self.min_relative_change < 1.0:
            raise ValueError(
                f"min_relative_change must lie within [0.0, 1.0); got "
                f"{self.min_relative_change}. A tolerance of 1.0 or more would classify no "
                "measurable change as a change at all."
            )
        if self.min_samples_for_comparison < 1:
            raise ValueError(
                f"min_samples_for_comparison must be at least 1; got "
                f"{self.min_samples_for_comparison}"
            )


@dataclass(frozen=True, slots=True)
class EvaluationSettings:
    """Configuration for the evaluation framework (Phase 13).

    The phase has two halves and this object holds the thresholds for both. The first three
    fields govern *split construction* — how a corpus is divided into the part used to fit a
    model and the part held out to score it. The last two govern *prediction evaluation* —
    when a single recorded prediction may be compared against what followed it. They are
    kept together because they are one phase's configuration and a reader reaching for
    either should not have to know which half of the phase the other belongs to.

    Attributes:
        temporal_split_quantile: Fraction of each learner's timeline reserved as the
            temporal holdout, ordered by timestamp. Cross-learner random splits are
            unusable for forecasting tasks because they leak each learner's own future
            into training.
        min_learners_per_split: Minimum learners required for a split-level metric to be
            reported. Below this, learner-level averages are too noisy to interpret and
            the metric is reported as undefined rather than as a number.
        min_positive_rate: Minimum positive-class rate for a split to be evaluated with
            ranking metrics at all. Below this, PR-AUC and ROC-AUC are unstable and are
            reported as undefined.
        min_evidence_for_evaluation: Minimum activity events inside a prediction's
            observation window before that prediction may be scored at all. Below this the
            window cannot support a ground truth and the prediction is recorded
            ``NOT_ASSESSABLE``. The default of four reuses the reasoning the outcome layer
            states for its own sample floor: three events over a short window is a learner
            who paused, and calling the pause a behavioural state hands the comparison a
            finding about a person that the log does not contain. A prediction of "deterioration
            within five minutes" that is scored against two events has been evaluated against
            a gap in a log, and the resulting ``NOT_ASSESSABLE`` is the honest reading.
        require_horizon_inclusive_start: Whether the observation window includes the
            prediction instant itself. Retained as ``True`` and named explicitly because the
            rule is load-bearing: the *end* of the window is exclusive and the *start* is
            inclusive, so an event at exactly the prediction instant is evidence about what
            followed and an event at exactly the horizon boundary is not. Flipping either
            edge would move events across the prediction boundary, which is why the
            asymmetry is a documented field rather than a literal in an expression.
    """

    temporal_split_quantile: float = 0.8
    min_learners_per_split: int = 5
    min_positive_rate: float = 0.01
    min_evidence_for_evaluation: int = 4
    require_horizon_inclusive_start: bool = True

    def __post_init__(self) -> None:
        """Validate the evaluation parameters.

        Raises:
            ValueError: If the split quantile is outside ``(0, 1)``, if the minimum
                counts are below one, or if the evidence floor is below one.
        """
        if not 0.0 < self.temporal_split_quantile < 1.0:
            raise ValueError(
                f"temporal_split_quantile must lie in (0.0, 1.0); got {self.temporal_split_quantile}"
            )
        if self.min_learners_per_split < 1:
            raise ValueError("min_learners_per_split must be at least 1")
        if not 0.0 <= self.min_positive_rate < 0.5:
            raise ValueError(
                f"min_positive_rate must lie in [0.0, 0.5); got {self.min_positive_rate}"
            )
        if self.min_evidence_for_evaluation < 1:
            raise ValueError(
                "min_evidence_for_evaluation must be at least 1; got "
                f"{self.min_evidence_for_evaluation}. A floor of zero would let an empty "
                "observation window be scored, and an empty window is the absence of a "
                "finding rather than a finding of absence."
            )
        if not self.require_horizon_inclusive_start:
            raise ValueError(
                "require_horizon_inclusive_start must be True. The observation window starts "
                "at the prediction instant and includes it: the delivery or session event "
                "recorded at the instant a prediction was made is the first observation "
                "after it. Excluding it would leave a window with no start-adjacent evidence "
                "for a prediction made mid-activity, and the asymmetry is a versioning "
                "matter rather than a preference, so it is not configurable."
            )


@dataclass(frozen=True, slots=True)
class SimulationSettings:
    """Configuration for the synthetic learner simulator (Phase 4).

    Attributes:
        default_seed: Master seed. Every simulator run must record the seed it used, so
            that a synthetic dataset is exactly regenerable.
        min_session_events: A generated session shorter than this is discarded, because
            it cannot support window-based feature calculation.
        enforce_monotonic_timestamps: Generated events are emitted in non-decreasing
            timestamp order. A simulator that emits out-of-order timestamps would test
            the ingestion layer rather than the intelligence layers.
        emit_synthetic_stamp: Attach the in-band synthetic warning stamp to generated
            data. Disabling this is permitted only for tests of the rejection path.
    """

    default_seed: int = 20260926
    min_session_events: int = 8
    enforce_monotonic_timestamps: bool = True
    emit_synthetic_stamp: bool = True
    generator_version: str = "SIMULATOR_V1"

    def __post_init__(self) -> None:
        """Validate the simulator parameters.

        Raises:
            ValueError: If the seed is negative or the minimum session length is too
                small to support feature calculation.
        """
        if self.default_seed < 0:
            raise ValueError("default_seed must be non-negative")
        if self.min_session_events < 2:
            raise ValueError(
                "min_session_events must be at least 2; a session of one event carries no "
                "temporal structure"
            )


@dataclass(frozen=True, slots=True)
class AuthoritySettings:
    """Configuration for the authority layer.

    The authority layer answers a different question from the policy layer. Policy asks
    "should something be done?"; authority asks "who is permitted to do it, in this
    context, and how much of the system's belief survives the last time a human disagreed
    with it?". These fields therefore govern *permission* and *trust decay*, not
    behaviour detection.

    Attributes:
        base_confidence_weight: The authority weight assigned to a learner in a context
            with no recorded history at all. Deliberately below ``1.0``: a system that has
            never been corrected has not been *vindicated*, and a first impression is
            weaker evidence than a track record.
        min_authority_weight: The floor authority weight decays toward but never reaches
            ``0.0``. A weight of exactly zero would make the envelope unreadable and would
            make ``HUMAN_REQUIRED`` indistinguishable from a bug.
        decay_half_life_hours: Hours after which one unit of penalty is half-spent. Sets
            the pace at which a stale disagreement stops counting against the system.
            Chosen to span a typical study week rather than a single session, so authority
            reflects whether a pattern is *current* rather than whether it happened today.
        disagreement_penalty: Weight removed per recorded human disagreement. Larger than
            the per-outcome penalty because a human contradicting the system is direct
            evidence about the system, whereas a null outcome is only indirect.
        nonresponse_penalty: Weight removed per intervention the learner did not respond
            to. Indirect evidence, so smaller than ``disagreement_penalty``.
        deterioration_penalty: Weight removed per measured post-intervention
            deterioration. Also indirect, and deliberately equal to ``nonresponse_penalty``
            because "it did not help" and "they ignored it" are the same magnitude of
            evidence about an intervention's usefulness.
        improvement_credit: Weight returned per measured post-intervention improvement.
            Strictly less than ``1.0`` so that a track record of good outcomes approaches
            but never attains the weight of an uncorrected first impression: history is
            evidence, not proof.
        expiry_hours: Hours after which an authority envelope is stale. A stale envelope
            must be recalculated rather than reused, because the evidence behind it has
            moved on. Compared against ``decay_half_life_hours`` in
            :meth:`__post_init__`; expiry is normally the longer of the two so that an
            envelope does not expire while its own decay is still in its first half-life.
        high_impact_min_authority: The minimum authority weight at which a
            ``HIGH_IMPACT`` action may be considered for autonomous execution at all.
            Above ``base_confidence_weight`` by default and above ``0.0``: an action that
            changes a learner's record is never something a system does on first contact,
            and it should require a track record of outcomes that went well. Reachable
            because the weight range is ``[min_authority_weight, 1.0]`` and measured
            improvement raises the weight - not because the base is generous.
        max_ledger_entries: Upper bound on retained authority ledger records. The ledger is
            append-only; this is a *retention* bound, and trimming is reported in the
            ledger summary rather than performed silently.
        autonomy_min_weight: The minimum authority weight at which a low-risk, reversible,
            non-high-stakes context may grant autonomous execution. Every other context
            shape escalates on its own, so this floor governs only the most permissive case
            and exists to answer the question context risk cannot: whether the system has
            been contradicted often enough that it should stop acting alone even where
            acting alone would be harmless. Constrained to
            ``(min_authority_weight, base_confidence_weight]`` so that a fresh context
            starts above it and a distrusted one falls below it.
    """

    base_confidence_weight: float = 0.60
    min_authority_weight: float = 0.05
    decay_half_life_hours: float = 168.0
    disagreement_penalty: float = 0.12
    nonresponse_penalty: float = 0.05
    deterioration_penalty: float = 0.05
    improvement_credit: float = 0.30
    expiry_hours: float = 336.0
    high_impact_min_authority: float = 0.75
    max_ledger_entries: int = 1000
    autonomy_min_weight: float = 0.50

    def __post_init__(self) -> None:
        """Validate the decay and retention parameters.

        Raises:
            ValueError: If the weight floor is not strictly below the base weight, if a
                penalty is negative, if the half-life is non-positive, if the expiry is not
                at least the half-life, if the high-impact floor leaves the top of the
                range unreachable, if the autonomy floor lies outside
                ``(min_authority_weight, base_confidence_weight]``, or if the ledger bound
                is not positive.
        """
        if not 0.0 <= self.min_authority_weight < self.base_confidence_weight <= 1.0:
            raise ValueError(
                "require 0.0 <= min_authority_weight < base_confidence_weight <= 1.0; got "
                f"min={self.min_authority_weight!r}, base={self.base_confidence_weight!r}"
            )
        if self.min_authority_weight >= 1.0:
            raise ValueError(
                "min_authority_weight must be strictly below 1.0; a floor of 1.0 would make "
                "authority undecayable and a permanent grant is the opposite of the point"
            )
        for name in (
            "disagreement_penalty",
            "nonresponse_penalty",
            "deterioration_penalty",
            "improvement_credit",
        ):
            value = float(getattr(self, name))
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative; got {value!r}")
        if self.improvement_credit > 1.0:
            raise ValueError(
                "improvement_credit must not exceed 1.0; a credit of 1.0 or more would let "
                "a track record manufacture authority from nothing, which is the failure "
                "this layer exists to prevent"
            )
        if self.decay_half_life_hours <= 0.0:
            raise ValueError(
                f"decay_half_life_hours must be positive; got {self.decay_half_life_hours!r}"
            )
        if self.expiry_hours < self.decay_half_life_hours:
            raise ValueError(
                "expiry_hours must be at least decay_half_life_hours; an envelope that "
                "expires inside its own first half-life would be discarded while its "
                f"disagreements still count at full strength (expiry={self.expiry_hours!r}, "
                f"half_life={self.decay_half_life_hours!r})"
            )
        if not self.min_authority_weight <= self.high_impact_min_authority <= 1.0:
            raise ValueError(
                "high_impact_min_authority must lie in [min_authority_weight, 1.0]; a "
                "floor above 1.0 would make every high-impact action permanently "
                f"unavailable (floor={self.high_impact_min_authority!r})"
            )
        if self.high_impact_min_authority <= self.base_confidence_weight:
            raise ValueError(
                "high_impact_min_authority must exceed base_confidence_weight; a floor at "
                "or below the weight a brand-new context starts at would let a system take "
                "a high-impact action on a learner it has never observed "
                f"(floor={self.high_impact_min_authority!r}, "
                f"base={self.base_confidence_weight!r})"
            )
        if self.max_ledger_entries <= 0:
            raise ValueError(
                f"max_ledger_entries must be positive; got {self.max_ledger_entries!r}"
            )
        if not self.min_authority_weight < self.autonomy_min_weight <= self.base_confidence_weight:
            raise ValueError(
                "autonomy_min_weight must lie in (min_authority_weight, "
                "base_confidence_weight]; a floor at or below the weight floor would make "
                "autonomy unreachable in every context, and a floor above the base would "
                "stop a never-contradicted learner from ever acting alone even in free "
                f"practice (autonomy_min={self.autonomy_min_weight!r}, "
                f"min={self.min_authority_weight!r}, base={self.base_confidence_weight!r})"
            )


@dataclass(frozen=True, slots=True)
class EngineSettings:
    """Aggregate configuration for a single engine instance.

    Layers receive only the sub-settings object they need. Passing this aggregate
    everywhere would give every layer read access to thresholds it has no business
    knowing, which is how a policy threshold ends up silently influencing a feature.

    Attributes:
        features: Feature engine configuration.
        baseline: Personal baseline engine configuration.
        temporal: Temporal state engine configuration.
        uncertainty: Uncertainty engine configuration.
        authority: Authority layer configuration.
        interventions: Intervention policy configuration.
        evaluation: Evaluation framework configuration.
        simulation: Synthetic simulator configuration.
        metadata: Free-form provenance for the configuration itself, used to record who
            changed which threshold and why.
    """

    features: FeatureSettings = field(default_factory=FeatureSettings)
    baseline: BaselineSettings = field(default_factory=BaselineSettings)
    temporal: TemporalSettings = field(default_factory=TemporalSettings)
    uncertainty: UncertaintySettings = field(default_factory=UncertaintySettings)
    authority: AuthoritySettings = field(default_factory=AuthoritySettings)
    interventions: InterventionSettings = field(default_factory=InterventionSettings)
    evaluation: EvaluationSettings = field(default_factory=EvaluationSettings)
    simulation: SimulationSettings = field(default_factory=SimulationSettings)
    metadata: dict[str, str] = field(default_factory=dict)

    def describe(self) -> str:
        """Render the configuration as a stable, loggable summary.

        Returns:
            A deterministic multi-line description. Stable so that two runs can be
            compared by diffing this string.
        """
        lines = [
            f"features={self.features!r}",
            f"baseline={self.baseline!r}",
            f"temporal={self.temporal!r}",
            f"uncertainty={self.uncertainty!r}",
            f"authority={self.authority!r}",
            f"interventions={self.interventions!r}",
            f"evaluation={self.evaluation!r}",
            f"simulation={self.simulation!r}",
        ]
        return "\n".join(lines)
