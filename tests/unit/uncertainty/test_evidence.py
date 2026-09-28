"""Tests for evidence volume and the maturity ceiling (Phase 9).

The property under test throughout is that a confident model cannot raise a limit that
exists to hold it down. Every test that checks a cap also checks that the cap is a function
of maturity alone.
"""

from __future__ import annotations

import pytest

from focus_engine.baseline.models import BaselineMaturity
from focus_engine.configuration.thresholds import (
    BaselineSettings,
    UncertaintySettings,
)
from focus_engine.schemas.primitives import InferenceBasis
from focus_engine.uncertainty.evidence import EvidenceVolume, maturity_ceiling
from tests.unit.uncertainty.scenarios import (
    UNITS_DEVELOPING,
    UNITS_EARLY,
    UNITS_ESTABLISHED,
    UNITS_NEW,
    evidence_at,
)

pytestmark = pytest.mark.unit

SETTINGS = UncertaintySettings()


class TestMaturityCeiling:
    """The cap that a model's own confidence cannot influence."""

    def test_each_rung_is_capped_at_its_configured_ceiling(self) -> None:
        assert maturity_ceiling(BaselineMaturity.NEW, SETTINGS) == (
            SETTINGS.max_confidence_when_new
        )
        assert maturity_ceiling(BaselineMaturity.EARLY, SETTINGS) == (
            SETTINGS.max_confidence_when_early
        )
        assert maturity_ceiling(BaselineMaturity.DEVELOPING, SETTINGS) == (
            SETTINGS.max_confidence_when_developing
        )

    def test_an_established_baseline_is_uncapped(self) -> None:
        assert maturity_ceiling(BaselineMaturity.ESTABLISHED, SETTINGS) == 1.0

    def test_the_ladder_ascends(self) -> None:
        """Confidence must not narrow as evidence accumulates."""
        ceilings = [maturity_ceiling(maturity, SETTINGS) for maturity in BaselineMaturity]
        assert ceilings == sorted(ceilings)

    def test_a_new_learner_is_capped_strictly_below_an_early_one(self) -> None:
        """Cold start is not the same situation as a thin history.

        Both are "not much data", but ``NEW`` means inference is falling back to a
        population prior, and a population prior cannot support a confident claim about an
        individual. Giving the two rungs the same cap would make them indistinguishable.
        """
        assert maturity_ceiling(BaselineMaturity.NEW, SETTINGS) < maturity_ceiling(
            BaselineMaturity.EARLY, SETTINGS
        )

    def test_the_ceiling_ignores_anything_a_model_might_say(self) -> None:
        """The function takes no prediction, which is the point.

        This is asserted by signature as well as by behaviour: there is no parameter
        through which a model output could reach this value.
        """
        import inspect

        parameters = list(inspect.signature(maturity_ceiling).parameters)
        assert parameters == ["maturity", "settings"]

    def test_a_tighter_configuration_tightens_the_ceiling(self) -> None:
        """The whole ladder has to move together.

        Lowering only the ``DEVELOPING`` rung is refused by the configuration itself: a cap
        below the ``EARLY`` one would mean confidence narrows as evidence accumulates,
        which is the opposite of what the ladder is for.
        """
        strict = UncertaintySettings(
            max_confidence_when_new=0.05,
            max_confidence_when_early=0.10,
            max_confidence_when_developing=0.20,
        )
        assert maturity_ceiling(BaselineMaturity.DEVELOPING, strict) == 0.20
        with pytest.raises(ValueError, match="non-decreasing"):
            UncertaintySettings(max_confidence_when_developing=0.10)


class TestEvidenceVolume:
    """How much evidence there is, and what it is permitted to support."""

    def test_maturity_is_derived_not_asserted(self) -> None:
        volume = evidence_at(UNITS_EARLY)
        assert volume.maturity is BaselineMaturity.EARLY
        assert volume.units == UNITS_EARLY

    def test_each_rung_lands_where_the_ladder_says(self) -> None:
        assert evidence_at(UNITS_NEW).maturity is BaselineMaturity.NEW
        assert evidence_at(UNITS_EARLY).maturity is BaselineMaturity.EARLY
        assert evidence_at(UNITS_DEVELOPING).maturity is BaselineMaturity.DEVELOPING
        assert evidence_at(UNITS_ESTABLISHED).maturity is BaselineMaturity.ESTABLISHED

    def test_basis_matches_the_baseline_layers_own_mapping(self) -> None:
        """Reused rather than re-derived, so the two answers cannot drift apart."""
        assert evidence_at(0).basis() is InferenceBasis.POPULATION_PRIOR
        assert evidence_at(UNITS_EARLY).basis() is InferenceBasis.PARTIAL_PERSONAL
        assert evidence_at(UNITS_ESTABLISHED).basis() is InferenceBasis.PERSONAL

    def test_personalisation_is_earned(self) -> None:
        """No rung below ``DEVELOPING`` may call itself personal."""
        for units in (0, UNITS_NEW, UNITS_EARLY):
            assert evidence_at(units).basis() is not InferenceBasis.PERSONAL

    def test_sufficiency_is_exactly_the_configured_gate(self) -> None:
        gate = SETTINGS.min_evidence_for_prediction
        assert not evidence_at(gate - 1).is_sufficient(SETTINGS)
        assert evidence_at(gate).is_sufficient(SETTINGS)

    def test_below_the_gate_there_is_no_defensible_confidence(self) -> None:
        assert evidence_at(0).ceiling(SETTINGS) == 0.0
        assert evidence_at(SETTINGS.min_evidence_for_prediction - 1).ceiling(SETTINGS) == 0.0

    def test_at_the_gate_the_ceiling_is_the_honest_minimum(self) -> None:
        assert evidence_at(SETTINGS.min_evidence_for_prediction).ceiling(SETTINGS) == 0.5

    def test_the_ceiling_rises_monotonically_between_gate_and_saturation(self) -> None:
        gate = SETTINGS.min_evidence_for_prediction
        ceilings = [
            evidence_at(units).ceiling(SETTINGS)
            for units in range(gate, SETTINGS.full_confidence_evidence_units, 7)
        ]
        assert ceilings == sorted(ceilings)
        assert ceilings[-1] < 1.0

    def test_evidence_stops_constraining_at_the_saturation_point(self) -> None:
        full = SETTINGS.full_confidence_evidence_units
        assert evidence_at(full).ceiling(SETTINGS) == 1.0
        assert evidence_at(full * 10).ceiling(SETTINGS) == 1.0

    def test_the_ramp_is_logarithmic_so_the_gate_is_not_a_cliff(self) -> None:
        """Confidence must not jump because a count crossed a threshold.

        Crossing the gate moves the ceiling from nothing at all to 0.5, and that step is
        justified - the verdict changes from INSUFFICIENT_DATA to a number. What must not
        happen is a second discontinuity: twenty observations should not suddenly be
        trusted as much as four hundred.
        """
        gate = SETTINGS.min_evidence_for_prediction
        full = SETTINGS.full_confidence_evidence_units
        first = evidence_at(gate + 1).ceiling(SETTINGS)
        a_quarter = evidence_at(gate + (full - gate) // 4).ceiling(SETTINGS)
        halfway = evidence_at(gate + (full - gate) // 2).ceiling(SETTINGS)
        assert 0.5 < first < a_quarter < halfway < 1.0

    def test_a_negative_count_is_refused(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            EvidenceVolume(units=-1, maturity=BaselineMaturity.NEW)

    def test_zero_observations_cannot_be_established(self) -> None:
        with pytest.raises(ValueError, match="contradict each other"):
            EvidenceVolume(units=0, maturity=BaselineMaturity.ESTABLISHED)

    def test_maturity_follows_a_different_configured_ladder(self) -> None:
        """The engine's ladder is the one that decides, not a second definition."""
        eager = BaselineSettings(min_samples_early=1, min_samples_developing=2)
        assert evidence_at(5, baseline=eager).maturity is BaselineMaturity.DEVELOPING
        assert evidence_at(5, baseline=BaselineSettings()).maturity is BaselineMaturity.NEW

    def test_a_negative_count_is_refused_by_the_factories_too(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            evidence_at(-3)

    def test_describe_names_the_count_maturity_and_basis(self) -> None:
        text = evidence_at(UNITS_EARLY).describe()
        assert str(UNITS_EARLY) in text
        assert "early" in text
        assert "partial_personal" in text
