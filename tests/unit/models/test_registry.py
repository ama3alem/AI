"""Tests for the append-only model registry (Phase 8).

The registry exists to make the path a model took to production inspectable after the
fact. These tests pin the properties that make it worth having: registration is the only
entry point and always starts at ``EXPERIMENTAL``; transitions follow a declared graph and
carry an attribution; history is extended and never rewritten; and a synthetic-only model
cannot be marked validated, whoever asks and however they ask.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from focus_engine.models.registry import (
    ALLOWED_TRANSITIONS,
    ModelRegistry,
    ModelRegistryError,
    ModelRegistryNotFoundError,
    assert_monotonic_versions,
)
from focus_engine.schemas.primitives import DataOrigin, Provenance
from focus_engine.schemas.versioning import ModelMetrics, ModelRecord, ModelStatus
from focus_engine.utils.clock import FixedClock

pytestmark = pytest.mark.unit

AUTHOR = "tester"
RATIONALE = "because the test says so"


def make_record(
    *,
    model_id: str = "risk-logreg",
    model_version: str = "RISK_MODEL_V1",
    status: ModelStatus = ModelStatus.EXPERIMENTAL,
    data_origin: DataOrigin = DataOrigin.REAL,
    label_provenance: Provenance = Provenance.OBSERVED,
    metrics: tuple[ModelMetrics, ...] = (),
    trained_at: datetime | None = None,
) -> ModelRecord:
    """Build a valid registry record for a test."""
    return ModelRecord(
        model_id=model_id,
        model_version=model_version,
        status=status,
        algorithm="sklearn.linear_model.LogisticRegression",
        hyperparameters={"max_iter": 1000},
        feature_set_version="FEATURE_SET_V1",
        dataset_version="DATASET_REAL_V1",
        trained_at=trained_at or datetime(2026, 1, 1, tzinfo=UTC),
        metrics=metrics,
        target_definition_version="td-1.0.0",
        label_provenance=label_provenance,
        data_origin=data_origin,
        seed=17,
    )


def real_metrics() -> tuple[ModelMetrics, ...]:
    """Build a metric set that is unambiguously real, for promotion tests."""
    return (
        ModelMetrics(
            split_name="holdout",
            n_samples=500,
            n_positive=180,
            roc_auc=0.79,
            pr_auc=0.61,
            f1=0.55,
            precision=0.58,
            recall=0.52,
            brier=0.17,
            expected_calibration_error=0.06,
            accuracy=0.74,
        ),
    )


class TestRegistration:
    """Registration is the only entry point, and it starts at the bottom."""

    def test_registration_records_the_model(self) -> None:
        """A registered model is immediately visible."""
        registry = ModelRegistry()

        registry.register(make_record(), author=AUTHOR, rationale=RATIONALE)

        assert registry.is_registered("risk-logreg", "RISK_MODEL_V1")
        assert registry.status("risk-logreg", "RISK_MODEL_V1") is ModelStatus.EXPERIMENTAL

    @pytest.mark.parametrize(
        "status",
        [ModelStatus.VALIDATED, ModelStatus.CANDIDATE, ModelStatus.PRODUCTION],
    )
    def test_a_model_cannot_be_registered_above_experimental(self, status: ModelStatus) -> None:
        """Registration cannot be used to skip the lifecycle.

        The single most damaging thing this registry could allow is a record that appears
        in production without ever having been validated, and the cheapest defence is to
        make that path raise.
        """
        registry = ModelRegistry()
        record = make_record(status=status, metrics=real_metrics())

        with pytest.raises(ModelRegistryError, match="must be registered as"):
            registry.register(record, author=AUTHOR, rationale=RATIONALE)

    def test_duplicate_registration_is_refused(self) -> None:
        """A version is minted once; its history is extended, not replaced."""
        registry = ModelRegistry()
        registry.register(make_record(), author=AUTHOR, rationale=RATIONALE)

        with pytest.raises(ModelRegistryError, match="already registered"):
            registry.register(make_record(), author=AUTHOR, rationale=RATIONALE)

    def test_registration_appends_a_transition(self) -> None:
        """The entry into the registry is itself recorded, with an attribution."""
        registry = ModelRegistry()

        registry.register(make_record(), author=AUTHOR, rationale=RATIONALE)

        events = registry.transitions("risk-logreg", "RISK_MODEL_V1")
        assert len(events) == 1
        assert events[0].to_status is ModelStatus.EXPERIMENTAL
        assert events[0].from_status is None
        assert events[0].author == AUTHOR
        assert events[0].rationale == RATIONALE

    def test_the_same_model_id_accepts_a_new_version(self) -> None:
        """A version is a version: a re-trained model is a new entry, not a replacement."""
        registry = ModelRegistry()
        registry.register(make_record(), author=AUTHOR, rationale=RATIONALE)

        registry.register(
            make_record(model_version="RISK_MODEL_V2", trained_at=datetime(2026, 2, 1, tzinfo=UTC)),
            author=AUTHOR,
            rationale=RATIONALE,
        )

        assert len(registry.history("risk-logreg", "RISK_MODEL_V1")) == 1
        assert len(registry.history("risk-logreg", "RISK_MODEL_V2")) == 1
        assert len(registry.all_records()) == 2


class TestTransitionGraph:
    """A status changes only along a declared edge, and only with an attribution."""

    @pytest.mark.parametrize(
        ("start", "target"),
        [
            (ModelStatus.EXPERIMENTAL, ModelStatus.VALIDATED),
            (ModelStatus.EXPERIMENTAL, ModelStatus.RETIRED),
            (ModelStatus.VALIDATED, ModelStatus.CANDIDATE),
            (ModelStatus.VALIDATED, ModelStatus.RETIRED),
            (ModelStatus.CANDIDATE, ModelStatus.PRODUCTION),
            (ModelStatus.CANDIDATE, ModelStatus.RETIRED),
        ],
    )
    def test_permitted_moves_are_accepted(self, start: ModelStatus, target: ModelStatus) -> None:
        """Every edge in the declared graph is usable."""
        registry = ModelRegistry()
        registry.register(make_record(metrics=real_metrics()), author=AUTHOR, rationale=RATIONALE)

        path = {
            ModelStatus.EXPERIMENTAL: [],
            ModelStatus.VALIDATED: [ModelStatus.VALIDATED],
            ModelStatus.CANDIDATE: [ModelStatus.VALIDATED, ModelStatus.CANDIDATE],
            ModelStatus.PRODUCTION: [
                ModelStatus.VALIDATED,
                ModelStatus.CANDIDATE,
                ModelStatus.PRODUCTION,
            ],
            ModelStatus.RETIRED: [],
        }[start]
        for status in path:
            if status is not target:
                registry.transition(
                    "risk-logreg", "RISK_MODEL_V1", status, author=AUTHOR, rationale=RATIONALE
                )

        result = registry.transition(
            "risk-logreg", "RISK_MODEL_V1", target, author=AUTHOR, rationale=RATIONALE
        )

        assert result.status is target

    @pytest.mark.parametrize(
        ("start", "target"),
        [
            (ModelStatus.EXPERIMENTAL, ModelStatus.CANDIDATE),
            (ModelStatus.EXPERIMENTAL, ModelStatus.PRODUCTION),
            (ModelStatus.VALIDATED, ModelStatus.PRODUCTION),
        ],
    )
    def test_skipped_stages_are_refused(self, start: ModelStatus, target: ModelStatus) -> None:
        """A model cannot jump from experimental straight to serving learners."""
        registry = ModelRegistry()
        registry.register(make_record(metrics=real_metrics()), author=AUTHOR, rationale=RATIONALE)
        if start is not ModelStatus.EXPERIMENTAL:
            registry.transition(
                "risk-logreg", "RISK_MODEL_V1", start, author=AUTHOR, rationale=RATIONALE
            )

        with pytest.raises(ModelRegistryError, match="cannot move"):
            registry.transition(
                "risk-logreg", "RISK_MODEL_V1", target, author=AUTHOR, rationale=RATIONALE
            )

    def test_retired_is_terminal(self) -> None:
        """A retired model cannot be brought back to life by a transition."""
        registry = ModelRegistry()
        registry.register(make_record(), author=AUTHOR, rationale=RATIONALE)
        registry.transition(
            "risk-logreg", "RISK_MODEL_V1", ModelStatus.RETIRED, author=AUTHOR, rationale=RATIONALE
        )

        for status in ModelStatus:
            if status is ModelStatus.RETIRED:
                continue
            with pytest.raises(ModelRegistryError, match="terminal"):
                registry.transition(
                    "risk-logreg", "RISK_MODEL_V1", status, author=AUTHOR, rationale=RATIONALE
                )

    def test_a_no_op_transition_is_refused(self) -> None:
        """Re-declaring the current status is not a transition.

        Allowing it would let a caller manufacture the appearance of a decision.
        """
        registry = ModelRegistry()
        registry.register(make_record(), author=AUTHOR, rationale=RATIONALE)

        with pytest.raises(ModelRegistryError):
            registry.transition(
                "risk-logreg",
                "RISK_MODEL_V1",
                ModelStatus.EXPERIMENTAL,
                author=AUTHOR,
                rationale=RATIONALE,
            )

    def test_transitioning_an_unregistered_model_is_refused(self) -> None:
        """A status can only change for a model the registry has seen."""
        registry = ModelRegistry()

        with pytest.raises(ModelRegistryNotFoundError, match="is not registered"):
            registry.transition(
                "ghost", "m-1", ModelStatus.VALIDATED, author=AUTHOR, rationale=RATIONALE
            )

    def test_author_and_rationale_are_mandatory(self) -> None:
        """An unattributed transition is refused by the type, not by convention."""
        registry = ModelRegistry()
        registry.register(make_record(), author=AUTHOR, rationale=RATIONALE)

        with pytest.raises(TypeError):
            registry.transition(  # type: ignore[call-arg]
                "risk-logreg", "RISK_MODEL_V1", ModelStatus.VALIDATED, author=AUTHOR
            )

    def test_a_blank_author_is_refused(self) -> None:
        """``"   "`` is not an author."""
        registry = ModelRegistry()
        registry.register(make_record(), author=AUTHOR, rationale=RATIONALE)

        with pytest.raises(Exception, match="author|rationale"):
            registry.transition(
                "risk-logreg",
                "RISK_MODEL_V1",
                ModelStatus.VALIDATED,
                author="   ",
                rationale=RATIONALE,
            )


class TestSyntheticGuard:
    """A synthetic-only model can never be marked validated."""

    @pytest.mark.parametrize("target", [ModelStatus.VALIDATED, ModelStatus.PRODUCTION])
    def test_synthetic_only_model_cannot_be_promoted(self, target: ModelStatus) -> None:
        """The registry enforces the guard, not merely the record validator.

        Relying on the caller to have constructed a valid record would put the only
        defence against the system's worst false claim in the hands of whoever most wants
        to bypass it.
        """
        registry = ModelRegistry()
        registry.register(
            make_record(
                data_origin=DataOrigin.SYNTHETIC,
                label_provenance=Provenance.SYNTHETIC_LABEL,
            ),
            author=AUTHOR,
            rationale=RATIONALE,
        )

        with pytest.raises(ModelRegistryError):
            registry.transition(
                "risk-logreg", "RISK_MODEL_V1", target, author=AUTHOR, rationale=RATIONALE
            )

    def test_a_failed_promotion_does_not_change_the_recorded_status(self) -> None:
        """A refused transition leaves the model where it was.

        A registry that recorded the attempt as a successful move would make the refusal
        invisible in the history, which is the one place the refusal has to be visible.
        """
        registry = ModelRegistry()
        registry.register(
            make_record(
                data_origin=DataOrigin.SYNTHETIC,
                label_provenance=Provenance.SYNTHETIC_LABEL,
            ),
            author=AUTHOR,
            rationale=RATIONALE,
        )

        with pytest.raises(ModelRegistryError):
            registry.transition(
                "risk-logreg",
                "RISK_MODEL_V1",
                ModelStatus.VALIDATED,
                author=AUTHOR,
                rationale=RATIONALE,
            )

        assert registry.status("risk-logreg", "RISK_MODEL_V1") is ModelStatus.EXPERIMENTAL
        assert len(registry.history("risk-logreg", "RISK_MODEL_V1")) == 1

    def test_a_synthetic_model_may_still_be_retired(self) -> None:
        """Retiring is always permitted, including for a synthetic model.

        Being unable to retire a model is a way of keeping a bad one in service.
        """
        registry = ModelRegistry()
        registry.register(
            make_record(
                data_origin=DataOrigin.SYNTHETIC,
                label_provenance=Provenance.SYNTHETIC_LABEL,
            ),
            author=AUTHOR,
            rationale=RATIONALE,
        )

        result = registry.transition(
            "risk-logreg", "RISK_MODEL_V1", ModelStatus.RETIRED, author=AUTHOR, rationale=RATIONALE
        )

        assert result.status is ModelStatus.RETIRED


class TestAppendOnlyHistory:
    """History is extended, never rewritten."""

    def test_every_transition_appends_a_record(self) -> None:
        """The full path to production remains inspectable."""
        registry = ModelRegistry()
        registry.register(make_record(metrics=real_metrics()), author=AUTHOR, rationale=RATIONALE)
        for status in (ModelStatus.VALIDATED, ModelStatus.CANDIDATE, ModelStatus.PRODUCTION):
            registry.transition(
                "risk-logreg", "RISK_MODEL_V1", status, author=AUTHOR, rationale=RATIONALE
            )

        history = registry.history("risk-logreg", "RISK_MODEL_V1")

        assert [entry.status for entry in history] == [
            ModelStatus.EXPERIMENTAL,
            ModelStatus.VALIDATED,
            ModelStatus.CANDIDATE,
            ModelStatus.PRODUCTION,
        ]

    def test_the_transition_log_names_every_decision(self) -> None:
        """Each step records who authorised it and on what grounds."""
        registry = ModelRegistry()
        registry.register(make_record(metrics=real_metrics()), author=AUTHOR, rationale=RATIONALE)
        registry.transition(
            "risk-logreg",
            "RISK_MODEL_V1",
            ModelStatus.VALIDATED,
            author="reviewer-a",
            rationale="held-out metrics reviewed",
        )

        events = registry.transitions("risk-logreg", "RISK_MODEL_V1")

        assert [event.author for event in events] == [AUTHOR, "reviewer-a"]
        assert events[1].from_status is ModelStatus.EXPERIMENTAL
        assert events[1].rationale == "held-out metrics reviewed"
        assert "reviewer-a" in events[1].describe()

    def test_transition_timestamps_come_from_the_injected_clock(self) -> None:
        """A test cannot depend on the wall clock, and neither can an audit."""
        clock = FixedClock(datetime(2030, 5, 1, 12, 0, tzinfo=UTC))
        registry = ModelRegistry(clock=clock)
        registry.register(make_record(), author=AUTHOR, rationale=RATIONALE)

        events = registry.transitions("risk-logreg", "RISK_MODEL_V1")

        assert events[0].recorded_at == datetime(2030, 5, 1, 12, 0, tzinfo=UTC)

    def test_asking_about_an_unknown_model_raises(self) -> None:
        """The registry is authoritative: an unknown model has no status and no history.

        Returning an empty history for a model nobody registered would let a caller treat
        "this model was never registered" and "this model was registered and then
        deregistered" as the same observation. ``is_registered`` is the non-raising probe.
        """
        registry = ModelRegistry()

        assert registry.is_registered("ghost", "RISK_MODEL_V1") is False
        with pytest.raises(ModelRegistryNotFoundError):
            registry.history("ghost", "RISK_MODEL_V1")
        with pytest.raises(ModelRegistryNotFoundError):
            registry.status("ghost", "RISK_MODEL_V1")


class TestProductionView:
    """The production view reports only what is actually in production."""

    def test_production_models_is_empty_before_any_promotion(self) -> None:
        """A freshly built registry serves nothing."""
        registry = ModelRegistry()
        registry.register(make_record(metrics=real_metrics()), author=AUTHOR, rationale=RATIONALE)

        assert registry.production_models() == ()

    def test_a_promoted_model_appears_in_the_production_view(self) -> None:
        """Only a model that reached production is listed."""
        registry = ModelRegistry()
        registry.register(make_record(metrics=real_metrics()), author=AUTHOR, rationale=RATIONALE)
        for status in (ModelStatus.VALIDATED, ModelStatus.CANDIDATE, ModelStatus.PRODUCTION):
            registry.transition(
                "risk-logreg", "RISK_MODEL_V1", status, author=AUTHOR, rationale=RATIONALE
            )

        serving = registry.production_models()

        assert len(serving) == 1
        assert serving[0].model_id == "risk-logreg"

    def test_a_retired_model_leaves_the_production_view(self) -> None:
        """Retirement takes effect immediately, without rewriting history."""
        registry = ModelRegistry()
        registry.register(make_record(metrics=real_metrics()), author=AUTHOR, rationale=RATIONALE)
        for status in (ModelStatus.VALIDATED, ModelStatus.CANDIDATE, ModelStatus.PRODUCTION):
            registry.transition(
                "risk-logreg", "RISK_MODEL_V1", status, author=AUTHOR, rationale=RATIONALE
            )
        registry.transition(
            "risk-logreg", "RISK_MODEL_V1", ModelStatus.RETIRED, author=AUTHOR, rationale=RATIONALE
        )

        assert registry.production_models() == ()
        assert len(registry.history("risk-logreg", "RISK_MODEL_V1")) == 5


class TestRegistryDigest:
    """The digest identifies which models exist and in what state."""

    def test_digest_is_stable_for_the_same_contents(self) -> None:
        """Two registries built the same way agree."""
        first, second = ModelRegistry(), ModelRegistry()
        for registry in (first, second):
            registry.register(make_record(), author=AUTHOR, rationale=RATIONALE)

        assert first.digest() == second.digest()

    def test_digest_changes_when_a_status_changes(self) -> None:
        """A promotion is a change to what the registry knows."""
        first, second = ModelRegistry(), ModelRegistry()
        first.register(make_record(), author=AUTHOR, rationale=RATIONALE)
        second.register(make_record(), author=AUTHOR, rationale=RATIONALE)
        second.transition(
            "risk-logreg", "RISK_MODEL_V1", ModelStatus.RETIRED, author=AUTHOR, rationale=RATIONALE
        )

        assert first.digest() != second.digest()

    def test_digest_does_not_depend_on_the_clock(self) -> None:
        """The digest is over what is registered, not over when it was registered.

        Timestamps are excluded deliberately: a digest that changed with every run would
        be useless for comparing two registries' contents.
        """
        early = ModelRegistry(clock=FixedClock(datetime(2026, 1, 1, tzinfo=UTC)))
        late = ModelRegistry(clock=FixedClock(datetime(2031, 12, 31, tzinfo=UTC)))
        early.register(make_record(), author=AUTHOR, rationale=RATIONALE)
        late.register(make_record(), author=AUTHOR, rationale=RATIONALE)

        assert early.digest() == late.digest()


class TestTransitionGraphShape:
    """The declared graph itself is sane."""

    def test_retired_has_no_outgoing_edges(self) -> None:
        """Retirement is terminal."""
        assert ALLOWED_TRANSITIONS[ModelStatus.RETIRED] == frozenset()

    def test_no_status_is_its_own_successor(self) -> None:
        """The graph contains no self-loops."""
        for source, targets in ALLOWED_TRANSITIONS.items():
            assert source not in targets

    def test_production_is_only_reachable_from_candidate(self) -> None:
        """There is exactly one edge into production, and it is the honest one."""
        sources = [
            source
            for source, targets in ALLOWED_TRANSITIONS.items()
            if ModelStatus.PRODUCTION in targets
        ]
        assert sources == [ModelStatus.CANDIDATE]

    def test_every_source_is_reachable_from_experimental(self) -> None:
        """No status can be entered that the lifecycle does not actually lead to.

        An orphaned status in the graph is a status that can never legitimately be
        reached, which means its existence is a lie about the process.
        """
        reachable = {ModelStatus.EXPERIMENTAL}
        frontier = [ModelStatus.EXPERIMENTAL]
        while frontier:
            source = frontier.pop()
            for target in ALLOWED_TRANSITIONS.get(source, frozenset()):
                if target not in reachable:
                    reachable.add(target)
                    frontier.append(target)

        assert set(ALLOWED_TRANSITIONS) <= reachable


class TestVersionOrdering:
    """Versions within one model family must increase."""

    def test_a_malformed_version_is_refused_at_construction(self) -> None:
        """A version nobody can order is not a version.

        The grammar is enforced by the schema, so an unparseable version cannot reach the
        registry in the first place.
        """
        with pytest.raises(Exception, match="invalid version identifier"):
            make_record(model_version="2026-01-01")

    def test_versions_may_repeat_the_number_across_families(self) -> None:
        """Ordering is per family, so two models may both be at version 1."""
        assert_monotonic_versions(
            [make_record(model_id="risk-logreg"), make_record(model_id="risk-forest")]
        )

    def test_a_decreasing_version_within_a_family_is_refused(self) -> None:
        """A later artifact cannot claim to predate an earlier one.

        If it could, "the latest version" would have no defined meaning, and a rollback
        would be indistinguishable from a forward move.
        """
        records = [
            make_record(model_id="risk-logreg", model_version="RISK_MODEL_V2"),
            make_record(model_id="risk-logreg", model_version="RISK_MODEL_V1"),
        ]

        with pytest.raises(ModelRegistryError, match="must increase"):
            assert_monotonic_versions(records)

    def test_an_increasing_sequence_is_accepted(self) -> None:
        """The ordinary case passes."""
        records = [
            make_record(model_id="risk-logreg", model_version="RISK_MODEL_V1"),
            make_record(model_id="risk-logreg", model_version="RISK_MODEL_V2"),
        ]

        assert list(assert_monotonic_versions(records)) == ["RISK_MODEL_V1", "RISK_MODEL_V2"]
