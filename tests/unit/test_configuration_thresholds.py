"""Unit tests for :mod:`focus_engine.configuration.thresholds`.

These tests assert two things: that each settings object rejects incoherent
configuration, and that the defaults are internally consistent. They deliberately do
*not* assert that a default is empirically optimal, because no empirical evidence exists
for any of these values yet.
"""

from __future__ import annotations

import pytest

from focus_engine.configuration.thresholds import (
    BaselineSettings,
    EngineSettings,
    EvaluationSettings,
    FeatureSettings,
    InterventionSettings,
    SimulationSettings,
    TemporalSettings,
    UncertaintySettings,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------------------
# Feature settings
# --------------------------------------------------------------------------------------


def test_feature_settings_defaults_are_coherent() -> None:
    settings = FeatureSettings()
    assert settings.long_window_minutes > settings.short_window_minutes
    assert settings.min_events_for_window >= 1


def test_long_window_must_exceed_short_window() -> None:
    with pytest.raises(ValueError, match="must exceed"):
        FeatureSettings(short_window_minutes=60.0, long_window_minutes=60.0)


def test_inverted_feature_windows_are_rejected() -> None:
    with pytest.raises(ValueError, match="must exceed"):
        FeatureSettings(short_window_minutes=60.0, long_window_minutes=15.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"short_window_minutes": 0.0, "long_window_minutes": 10.0},
        {"short_window_minutes": -5.0, "long_window_minutes": 10.0},
        {"interaction_gap_seconds_floor": 0.0},
        {"interaction_gap_seconds_floor": -1.0},
        {"min_events_for_window": 0},
    ],
)
def test_non_positive_feature_settings_are_rejected(kwargs: dict[str, float | int]) -> None:
    with pytest.raises(ValueError):
        FeatureSettings(**kwargs)


# --------------------------------------------------------------------------------------
# Baseline settings
# --------------------------------------------------------------------------------------


def test_baseline_maturity_ladder_is_strictly_increasing() -> None:
    settings = BaselineSettings()
    ladder = [
        settings.min_samples_new,
        settings.min_samples_early,
        settings.min_samples_developing,
        settings.min_samples_established,
    ]
    assert ladder == sorted(ladder)
    assert len(set(ladder)) == len(ladder)


def test_baseline_defaults_begin_at_a_cold_start() -> None:
    """A learner with no observations must fall to the population prior, not a baseline."""
    assert BaselineSettings().min_samples_new == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_samples_early": 100, "min_samples_developing": 50},
        {"min_samples_early": 50, "min_samples_developing": 50},
        {"min_samples_early": 10, "min_samples_developing": 20, "min_samples_established": 15},
    ],
)
def test_non_increasing_maturity_ladder_is_rejected(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        BaselineSettings(**kwargs)


def test_negative_maturity_threshold_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        BaselineSettings(min_samples_early=-1)


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.1, 2.0])
def test_ewma_alpha_must_lie_in_unit_half_open_interval(alpha: float) -> None:
    with pytest.raises(ValueError, match=r"ewma_alpha"):
        BaselineSettings(ewma_alpha=alpha)


def test_ewma_alpha_of_one_is_permitted() -> None:
    """alpha == 1.0 means every new observation replaces history. Valid but degenerate."""
    assert BaselineSettings(ewma_alpha=1.0).ewma_alpha == 1.0


def test_robust_statistics_enabled_by_default() -> None:
    """Robust statistics are the default because small samples are heavy-tailed here."""
    assert BaselineSettings().use_robust_statistics is True


def test_baseline_settings_are_frozen() -> None:
    settings = BaselineSettings()
    with pytest.raises(AttributeError):
        settings.ewma_alpha = 0.5  # type: ignore[misc]


# --------------------------------------------------------------------------------------
# Temporal settings
# --------------------------------------------------------------------------------------


def test_temporal_defaults_require_multiple_observations() -> None:
    """One anomalous observation is an anomaly, not a trajectory."""
    settings = TemporalSettings()
    assert settings.state_min_observations >= 2
    assert settings.sustained_change_min_observations >= 3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"state_min_observations": 0},
        {"sustained_change_min_observations": 0},
        {"rate_of_change_window": 0},
        {"stability_window": 0},
        {"recovery_epsilon": 0.0},
        {"recovery_epsilon": 1.5},
        {"recovery_epsilon": -0.2},
    ],
)
def test_invalid_temporal_settings_are_rejected(kwargs: dict[str, int | float]) -> None:
    with pytest.raises(ValueError):
        TemporalSettings(**kwargs)


# --------------------------------------------------------------------------------------
# Uncertainty settings
# --------------------------------------------------------------------------------------


def test_confidence_thresholds_are_ordered() -> None:
    settings = UncertaintySettings()
    assert 0.0 < settings.medium_confidence_threshold < settings.high_confidence_threshold <= 1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"medium_confidence_threshold": 0.9, "high_confidence_threshold": 0.5},
        {"medium_confidence_threshold": 0.0},
        {"high_confidence_threshold": 1.5},
        {"medium_confidence_threshold": 0.95, "high_confidence_threshold": 0.95},
    ],
)
def test_unordered_confidence_thresholds_are_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="0 < medium < high"):
        UncertaintySettings(**kwargs)


def test_early_maturity_confidence_cap_may_not_exceed_developing_cap() -> None:
    """A more established baseline must never be capped lower than a less mature one."""
    with pytest.raises(ValueError, match="must not exceed"):
        UncertaintySettings(max_confidence_when_early=0.9, max_confidence_when_developing=0.5)


def test_minimum_evidence_is_at_least_one() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        UncertaintySettings(min_evidence_for_prediction=0)


def test_probability_floor_must_be_a_small_positive_fraction() -> None:
    with pytest.raises(ValueError, match="probability_floor"):
        UncertaintySettings(probability_floor=0.0)
    with pytest.raises(ValueError, match="probability_floor"):
        UncertaintySettings(probability_floor=0.6)


def test_defaults_permit_insufficient_data_verdict() -> None:
    """The engine must be able to refuse to predict; that requires a positive threshold."""
    assert UncertaintySettings().min_evidence_for_prediction >= 1


# --------------------------------------------------------------------------------------
# Intervention settings
# --------------------------------------------------------------------------------------


def test_intervention_defaults_are_restraintive() -> None:
    settings = InterventionSettings()
    assert 0.0 < settings.min_probability_to_act < 1.0
    assert settings.cooldown_minutes > 0
    assert settings.max_interventions_per_session >= 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_probability_to_act": 0.0},
        {"min_probability_to_act": 1.0},
        {"min_probability_to_act": -0.1},
        {"cooldown_minutes": 0.0},
        {"max_interventions_per_session": 0},
        {"max_repeat_same_type": 0},
        {"min_response_observations": 0},
        {"abandon_after_no_response_count": 0},
    ],
)
def test_invalid_intervention_settings_are_rejected(kwargs: dict[str, float | int]) -> None:
    with pytest.raises(ValueError):
        InterventionSettings(**kwargs)


def test_action_threshold_is_not_trivially_low() -> None:
    """A default that fires on a coin flip would make the policy noise-driven."""
    assert InterventionSettings().min_probability_to_act >= 0.5


# --------------------------------------------------------------------------------------
# Evaluation settings
# --------------------------------------------------------------------------------------


def test_evaluation_split_keeps_a_holdout() -> None:
    assert 0.5 < EvaluationSettings().temporal_split_quantile < 1.0


def test_prediction_evaluation_defaults_reuse_the_outcome_sample_floor() -> None:
    """Four events is the floor for reading a window as a state, not a pause.

    The evaluation floor matches the outcome layer's own floor deliberately: scoring a
    forecast against the same thin evidence the outcome layer refused to characterise
    would make the two layers disagree about what counts as evidence.
    """
    assert EvaluationSettings().min_evidence_for_evaluation == 4


def test_observation_window_start_is_inclusive_by_default() -> None:
    """An event at exactly the prediction instant is the first event after it."""
    assert EvaluationSettings().require_horizon_inclusive_start is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temporal_split_quantile": 0.0},
        {"temporal_split_quantile": 1.0},
        {"temporal_split_quantile": -0.2},
        {"min_learners_per_split": 0},
        {"min_positive_rate": 0.5},
        {"min_positive_rate": 0.9},
        {"min_evidence_for_evaluation": 0},
        {"min_evidence_for_evaluation": -1},
        {"require_horizon_inclusive_start": False},
    ],
)
def test_invalid_evaluation_settings_are_rejected(kwargs: dict[str, float | int | bool]) -> None:
    with pytest.raises(ValueError):
        EvaluationSettings(**kwargs)


def test_zero_evidence_floor_is_rejected() -> None:
    """A floor of zero would score an empty window, which is absence, not a finding."""
    with pytest.raises(ValueError, match="at least 1"):
        EvaluationSettings(min_evidence_for_evaluation=0)


def test_exclusive_window_start_may_not_be_configured() -> None:
    """The start-inclusive/end-exclusive rule is a versioning matter, not a preference.

    The source explains why it refuses, so the message is asserted to keep the rationale
    attached to the behaviour rather than to a bare rejection.
    """
    with pytest.raises(ValueError, match="must be True"):
        EvaluationSettings(require_horizon_inclusive_start=False)


def test_evidence_floor_of_one_is_permitted() -> None:
    """The validator forbids zero, not small; one is a coherent floor if chosen."""
    assert EvaluationSettings(min_evidence_for_evaluation=1).min_evidence_for_evaluation == 1


# --------------------------------------------------------------------------------------
# Simulation settings
# --------------------------------------------------------------------------------------


def test_simulation_defaults_are_reproducible() -> None:
    settings = SimulationSettings()
    assert settings.default_seed >= 0
    assert settings.enforce_monotonic_timestamps is True
    assert settings.emit_synthetic_stamp is True


def test_negative_simulation_seed_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        SimulationSettings(default_seed=-1)


def test_session_shorter_than_two_events_is_rejected() -> None:
    with pytest.raises(ValueError, match="no temporal structure"):
        SimulationSettings(min_session_events=1)


# --------------------------------------------------------------------------------------
# Aggregate settings
# --------------------------------------------------------------------------------------


def test_engine_settings_constructs_with_independent_defaults() -> None:
    settings = EngineSettings()
    assert isinstance(settings.features, FeatureSettings)
    assert isinstance(settings.baseline, BaselineSettings)
    assert isinstance(settings.temporal, TemporalSettings)
    assert isinstance(settings.uncertainty, UncertaintySettings)
    assert isinstance(settings.interventions, InterventionSettings)
    assert isinstance(settings.evaluation, EvaluationSettings)
    assert isinstance(settings.simulation, SimulationSettings)


def test_engine_settings_defaults_are_not_shared_mutable_state() -> None:
    """Two instances must not alias the same sub-settings object."""
    first = EngineSettings()
    second = EngineSettings()
    assert first.baseline is not second.baseline
    assert first.metadata is not second.metadata


def test_engine_settings_describe_is_deterministic() -> None:
    """Two identical configurations must render identically so runs can be diffed."""
    assert EngineSettings().describe() == EngineSettings().describe()


def test_engine_settings_describe_mentions_every_layer() -> None:
    described = EngineSettings().describe()
    for layer in (
        "features",
        "baseline",
        "temporal",
        "uncertainty",
        "interventions",
        "evaluation",
        "simulation",
    ):
        assert f"{layer}=" in described


def test_engine_settings_accepts_partial_overrides() -> None:
    settings = EngineSettings(baseline=BaselineSettings(ewma_alpha=0.2))
    assert settings.baseline.ewma_alpha == 0.2
    assert settings.features == FeatureSettings()
