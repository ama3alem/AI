"""Tests for the feature-to-design-matrix encoding (Phase 8).

The encoding layer is the boundary between measured features and the numbers a model is
fitted on, and it is where imputation would enter the system if it were going to. These
tests pin the contract that it does not: an absent feature produces a refusal with a
reason, a categorical value outside its declared vocabulary is refused rather than
folded into a catch-all, and a vector nobody can attribute to a learner is not quietly
accepted into a per-learner evaluation set.
"""

from __future__ import annotations

import math

import pytest

from focus_engine.features import FeatureName, FeatureValueType
from focus_engine.models.encoding import (
    CATEGORICAL_VOCABULARIES,
    UNATTRIBUTED,
    DesignMatrix,
    EncodingError,
    default_features,
    encode_rows,
    requirements_from_schema,
)
from tests.unit.models.scenarios import make_labelled_rows, make_vector

pytestmark = pytest.mark.unit


class TestEncodingShape:
    """The matrix has the shape its schema promises."""

    def test_numeric_and_categorical_columns_are_expanded(self) -> None:
        """One-hot columns replace each categorical feature's single slot."""
        matrix = encode_rows(make_labelled_rows(4, positive=2))

        feature_count = len(default_features())
        one_hot_count = sum(len(vocabulary) for vocabulary in CATEGORICAL_VOCABULARIES.values())

        # 14 features, of which 4 are categorical and expand to 4 + 2 + 4 + 2 columns.
        assert feature_count == 14
        assert matrix.n_columns == feature_count - len(CATEGORICAL_VOCABULARIES) + one_hot_count
        assert matrix.n_columns == 24
        assert matrix.n_rows == 4

    def test_columns_are_named_and_ordered(self) -> None:
        """Column names carry the feature and, for one-hot, the category.

        A bare positional matrix is not interpretable, and the naming is what lets a
        reloaded model check that it is being handed the columns it learned on.
        """
        matrix = encode_rows(make_labelled_rows(2, positive=1))

        assert "performance_recent_accuracy" in matrix.columns
        assert "content_type_encoded=multiple_choice" in matrix.columns
        assert "intervention_cooldown_active=false" in matrix.columns
        # Categorical blocks are contiguous, so a reader can see where one ends.
        start = matrix.columns.index("content_type_encoded=multiple_choice")
        block = matrix.columns[start : start + 4]
        assert all(column.startswith("content_type_encoded=") for column in block)

    def test_labels_and_learner_ids_survive_encoding(self) -> None:
        """The label and the attribution are carried alongside the row."""
        rows = make_labelled_rows(4, positive=1, start_index=7)
        matrix = encode_rows(rows)

        assert list(matrix.labels) == [1, 0, 0, 0]
        assert matrix.learner_ids == (
            "learner-0007",
            "learner-0008",
            "learner-0009",
            "learner-0010",
        )

    def test_single_row_matrix_keeps_two_dimensions(self) -> None:
        """A one-row batch is still two-dimensional.

        ``numpy`` would squeeze this to a 1-D array, and an estimator would then raise an
        error that names its own internals rather than the real problem.
        """
        matrix = encode_rows(make_labelled_rows(1, positive=1))

        assert matrix.matrix.ndim == 2
        assert matrix.matrix.shape == (1, matrix.n_columns)

    def test_empty_input_produces_an_empty_matrix(self) -> None:
        """No rows in, no rows out, with the schema still intact."""
        matrix = encode_rows([])

        assert matrix.n_rows == 0
        assert matrix.matrix.shape[0] == 0
        assert matrix.n_columns > 0
        assert matrix.rejected == ()


class TestMissingData:
    """Absent data is refused, never invented."""

    def test_absent_feature_is_rejected_not_imputed(self) -> None:
        """A row missing a required feature is dropped with a named reason."""
        usable = make_vector(learner_id="learner-0001")
        broken = make_vector(
            learner_id="learner-0002",
            absent=(FeatureName.PERFORMANCE_RECENT_ACCURACY,),
        )

        matrix = encode_rows([(usable, 0), (broken, 1)])

        assert matrix.n_rows == 1
        assert matrix.learner_ids == ("learner-0001",)
        assert len(matrix.rejected) == 1
        rejection = matrix.rejected[0]
        assert rejection.learner_id == "learner-0002"
        assert rejection.missing_features == (FeatureName.PERFORMANCE_RECENT_ACCURACY,)
        assert "performance_recent_accuracy" in rejection.reason

    def test_rejection_states_the_underlying_reason(self) -> None:
        """The rejection carries the feature's own explanation, not a generic message."""
        broken = make_vector(
            learner_id="learner-0002",
            absent=(FeatureName.PERFORMANCE_RECENT_RESPONSE_SECONDS,),
        )

        matrix = encode_rows([(broken, 1)])

        assert "the test declared this feature unavailable" in matrix.rejected[0].reason

    def test_no_placeholder_value_survives_encoding(self) -> None:
        """No rejected row contributes a row, so no marker can be fitted on.

        This is the property that would break if a future change added imputation: the
        encoded matrix would contain a sentinel the model would learn to rely on.
        """
        broken = make_vector(absent=(FeatureName.SESSION_EVENT_COUNT,))

        matrix = encode_rows([(broken, 1)])

        assert matrix.matrix.size == 0
        assert matrix.n_rows == 0

    def test_synthetic_only_values_are_accepted(self) -> None:
        """A synthetic value is a real measurement and is used.

        Refusing it would conflate "not real" with "not there". The distinction is
        recorded on the vector and enforced at promotion time, where it belongs.
        """
        from focus_engine.features import FeatureAvailability, FeatureValue, TypedFeatureValue
        from tests.unit.models.scenarios import BASE_TIME, DEFAULT_VALUES, FEATURE_SET_V1

        values = tuple(
            TypedFeatureValue(
                name=name,
                value=DEFAULT_VALUES[name],
                availability=FeatureAvailability.SYNTHETIC_ONLY,
                computed_at=BASE_TIME,
                feature_set_version=FEATURE_SET_V1,
                spec_version="spec-v1",
            )
            for name in FeatureName
        )
        vector = FeatureValue(
            feature_set_version=FEATURE_SET_V1,
            computed_at=BASE_TIME,
            values=values,
            learner_id="learner-0001",
            session_id="session-0001",
        )

        matrix = encode_rows([(vector, 1)])

        assert matrix.n_rows == 1
        assert matrix.rejected == ()


class TestAttribution:
    """A vector that cannot be attributed is not quietly used."""

    def test_unattributed_vector_is_rejected(self) -> None:
        """A vector with no learner is refused and reported.

        An unattributed vector is a genuine measurement, but it cannot be sliced per
        learner, held out per learner, or traced when an evaluation result is questioned.
        """
        vector = make_vector(learner_id=None)

        matrix = encode_rows([(vector, 1)])

        assert matrix.n_rows == 0
        assert len(matrix.rejected) == 1
        assert matrix.rejected[0].learner_id == UNATTRIBUTED
        assert "learner_id" in matrix.rejected[0].reason

    def test_unattributed_sentinel_is_not_a_plausible_learner_id(self) -> None:
        """The sentinel cannot be mistaken for a real identifier."""
        assert UNATTRIBUTED == "unattributed"
        assert not UNATTRIBUTED.startswith("learner-")


class TestCategoricalVocabulary:
    """Categories come from a declared set, not from the data."""

    def test_unknown_category_is_refused(self) -> None:
        """A value outside the declared vocabulary raises rather than falling back.

        Folding it into an ``other`` bucket would make it indistinguishable from a real
        ``other``, which is how a model ends up confidently wrong about a case type it has
        never seen.
        """
        vector = make_vector(overrides={FeatureName.TRAJECTORY_ENCODED: "exploding"})

        with pytest.raises(EncodingError, match="outside its declared vocabulary"):
            encode_rows([(vector, 1)])

    def test_every_declared_category_is_encoded(self) -> None:
        """Each category in the vocabulary lights exactly its own column."""
        for category in CATEGORICAL_VOCABULARIES[FeatureName.TRAJECTORY_ENCODED]:
            vector = make_vector(overrides={FeatureName.TRAJECTORY_ENCODED: category})

            matrix = encode_rows([(vector, 1)])

            column = matrix.columns.index(f"trajectory_encoded={category}")
            assert matrix.matrix[0, column] == 1.0
            # Exactly one column of the block is set.
            start = matrix.columns.index("trajectory_encoded=improving")
            block = matrix.matrix[0, start : start + 4]
            assert block.sum() == 1.0

    def test_categorical_features_declare_a_string_type(self) -> None:
        """The requirement for a categorical feature is not a numeric one."""
        matrix = encode_rows(make_labelled_rows(1, positive=1))
        requirements = {requirement.name: requirement for requirement in matrix.requirements}

        for name in CATEGORICAL_VOCABULARIES:
            assert requirements[name].value_type is FeatureValueType.STR
            assert requirements[name].vocabulary == CATEGORICAL_VOCABULARIES[name]


class TestNumericValidation:
    """Numbers are numbers, or the row is refused."""

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_non_finite_numeric_is_refused(self, bad: float) -> None:
        """A NaN or infinity is not a measurement.

        It would propagate through the estimator and surface later as a meaningless metric
        rather than as the bad input that caused it.
        """
        vector = make_vector(overrides={FeatureName.PERFORMANCE_RECENT_ACCURACY: bad})

        with pytest.raises(EncodingError, match="finite"):
            encode_rows([(vector, 1)])

    def test_categorical_feature_must_carry_a_string(self) -> None:
        """A number where a category belongs is refused, not coerced."""
        vector = make_vector(overrides={FeatureName.CONTENT_TYPE_ENCODED: 3})

        with pytest.raises(EncodingError, match="not a string"):
            encode_rows([(vector, 1)])

    def test_integer_feature_rejects_a_fractional_value(self) -> None:
        """A count that is not a whole number is refused.

        Rounding it would quietly change a quantity the learner actually produced.
        """
        vector = make_vector(overrides={FeatureName.SESSION_EVENT_COUNT: 4.5})

        with pytest.raises(EncodingError, match="whole number"):
            encode_rows([(vector, 1)])

    def test_numeric_feature_rejects_a_string(self) -> None:
        """A string where a measurement belongs is refused rather than parsed."""
        vector = make_vector(overrides={FeatureName.PERFORMANCE_RECENT_ACCURACY: "high"})

        with pytest.raises(EncodingError, match="is declared 'float' but carries 'high'"):
            encode_rows([(vector, 1)])


class TestSchemaDigest:
    """The digest identifies an interpretation, not a serialisation."""

    def test_identical_schemas_share_a_digest(self) -> None:
        """The same requirements produce the same digest."""
        first = encode_rows(make_labelled_rows(2, positive=1, start_index=0))
        second = encode_rows(make_labelled_rows(2, positive=1, start_index=50))

        assert first.schema_digest() == second.schema_digest()

    def test_digest_ignores_the_data(self) -> None:
        """The digest is over the schema, so different rows do not change it."""
        low = encode_rows(make_labelled_rows(2, positive=1, start_index=0))
        high = encode_rows(
            [
                (
                    make_vector(
                        learner_id="learner-9999",
                        overrides={FeatureName.PERFORMANCE_RECENT_ACCURACY: 0.99},
                    ),
                    1,
                )
            ]
        )

        assert low.schema_digest() == high.schema_digest()

    def test_requirements_order_changes_the_digest(self) -> None:
        """Column order is part of the meaning, so it is part of the digest."""
        matrix = encode_rows(make_labelled_rows(1, positive=1))
        reversed_requirements = tuple(reversed(matrix.requirements))
        reversed_matrix = DesignMatrix(
            matrix=matrix.matrix,
            labels=matrix.labels,
            columns=tuple(column.split("=")[0] for column in reversed(matrix.columns)),
            requirements=reversed_requirements,
        )

        assert reversed_matrix.schema_digest() != matrix.schema_digest()

    def test_a_subset_of_features_has_a_different_digest(self) -> None:
        """A model trained on fewer features is a different model."""
        full = encode_rows(make_labelled_rows(1, positive=1))
        subset = encode_rows(
            [(make_vector(), 1)],
            requirements=full.requirements[:3],
        )

        assert subset.schema_digest() != full.schema_digest()


class TestRequirementsRoundTrip:
    """A stored schema can be rebuilt, or its divergence is reported."""

    def test_round_trip_preserves_the_schema(self) -> None:
        """Rebuilt requirements encode to the same digest as the original."""
        matrix = encode_rows(make_labelled_rows(2, positive=1))
        schema = matrix.schema()

        rebuilt = requirements_from_schema(schema)

        assert rebuilt == matrix.requirements
        assert encode_rows(
            make_labelled_rows(2, positive=1), requirements=rebuilt
        ).schema_digest() == (matrix.schema_digest())

    def test_unknown_feature_name_is_refused(self) -> None:
        """A schema naming a feature the code does not have is a divergence.

        Encoding against either side would be a guess, and one-hot columns are positional
        enough that a guess would be silently wrong.
        """
        schema = {
            "columns": ["mystery"],
            "requirements": [{"name": "mystery", "value_type": "float", "vocabulary": None}],
        }

        with pytest.raises(EncodingError, match="not a known feature"):
            requirements_from_schema(schema)

    def test_changed_vocabulary_is_refused(self) -> None:
        """A stored vocabulary that no longer matches the code is refused.

        Reordering a vocabulary reinterprets every one-hot column the model was trained on,
        so the model would keep producing numbers and the numbers would be wrong.
        """
        matrix = encode_rows(make_labelled_rows(1, positive=1))
        schema = matrix.schema()
        schema["requirements"] = [
            {**entry, "vocabulary": ["reversed", *reversed(entry["vocabulary"][1:])]}
            if entry["vocabulary"] is not None
            else entry
            for entry in schema["requirements"]
        ]

        with pytest.raises(EncodingError, match="but the code declares"):
            requirements_from_schema(schema)

    def test_schema_without_requirements_is_refused(self) -> None:
        """A schema that describes no requirements cannot be encoded against."""
        with pytest.raises(EncodingError, match="no requirements"):
            requirements_from_schema({"columns": [], "requirements": []})

    def test_malformed_requirement_entry_is_refused(self) -> None:
        """A requirement that is not a mapping is refused rather than skipped."""
        with pytest.raises(EncodingError, match="malformed requirement"):
            requirements_from_schema({"columns": [], "requirements": ["not-a-mapping"]})
