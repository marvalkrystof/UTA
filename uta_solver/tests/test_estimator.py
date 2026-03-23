"""
Tests for UTAEstimator – end-to-end fitting, prediction, and correctness proofs.

These tests verify that:
1. The estimator fits without errors.
2. Predicted utilities respect the given preference order.
3. Marginal utility variables are non-negative where required.
4. Kendall-tau score is meaningful.
5. Different criterion types can be mixed.
6. Edge-cases (ties, single criterion, minimal data) are handled.
"""

import pytest
import numpy as np
import pandas as pd

from uta_solver.estimator import UTAEstimator
from uta_solver.criteria import CardinalCriterion, OrdinalCriterion, NominalCriterion



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_data():
    """Four alternatives, two cardinal criteria, clear ranking."""
    X = pd.DataFrame({
        "price": [10.0, 7.0, 1.0, 3.0],
        "size":  [10.0, 6.0, 2.0, 2.0],
    })
    y = np.array([1, 2, 4, 3])
    return X, y


def _utilities_respect_ranking(utilities, ranks):
    """Return True if higher-ranked alts have >= utility than lower-ranked."""
    for i in range(len(ranks)):
        for j in range(len(ranks)):
            if ranks[i] < ranks[j]:
                # i is strictly preferred → should have higher utility
                if utilities[i] < utilities[j] - 1e-6:
                    return False
            elif ranks[i] == ranks[j]:
                # tie → utilities should be approximately equal
                if abs(utilities[i] - utilities[j]) > 1e-4:
                    return False
    return True



# Basic fit / predict


class TestBasicFitPredict:
    """Smoke tests and basic contract checks."""

    def test_fit_returns_self(self):
        X, y = _simple_data()
        model = UTAEstimator(n_segments=2)
        result = model.fit(X, y)
        assert result is model

    def test_marginal_utilities_populated(self):
        X, y = _simple_data()
        model = UTAEstimator(n_segments=2).fit(X, y)
        assert len(model.marginal_utilities_) > 0

    def test_criteria_created(self):
        X, y = _simple_data()
        model = UTAEstimator(n_segments=2).fit(X, y)
        assert len(model.criteria_) == 2

    def test_predict_shape(self):
        X, y = _simple_data()
        model = UTAEstimator(n_segments=2).fit(X, y)
        utilities = model.predict(X)
        assert utilities.shape == (4,)

    def test_predict_before_fit_raises(self):
        model = UTAEstimator()
        X, _ = _simple_data()
        with pytest.raises(Exception):
            model.predict(X)

    def test_feature_names_stored(self):
        X, y = _simple_data()
        model = UTAEstimator(n_segments=2).fit(X, y)
        assert list(model.feature_names_in_) == ["price", "size"]

    def test_n_features_stored(self):
        X, y = _simple_data()
        model = UTAEstimator(n_segments=2).fit(X, y)
        assert model.n_features_in_ == 2



# Correctness – preference order preservation


class TestPreferenceOrderPreservation:
    """The core property: learned utilities must respect input rankings."""

    def test_strict_ranking_two_criteria(self):
        X, y = _simple_data()
        model = UTAEstimator(n_segments=2).fit(X, y)
        utilities = model.predict(X)
        assert _utilities_respect_ranking(utilities, y)

    def test_strict_ranking_more_segments(self):
        X, y = _simple_data()
        model = UTAEstimator(n_segments=4).fit(X, y)
        utilities = model.predict(X)
        assert _utilities_respect_ranking(utilities, y)

    def test_five_alternatives(self):
        X = pd.DataFrame({
            "a": [10.0, 8.0, 6.0, 4.0, 2.0],
            "b": [9.0, 7.0, 5.0, 3.0, 1.0],
        })
        y = np.array([1, 2, 3, 4, 5])
        model = UTAEstimator(n_segments=3).fit(X, y)
        utilities = model.predict(X)
        assert _utilities_respect_ranking(utilities, y)

    def test_reversed_ranking(self):
        """Best alternative has highest criterion values but rank 5."""
        X = pd.DataFrame({
            "a": [2.0, 4.0, 6.0, 8.0, 10.0],
            "b": [2.0, 4.0, 6.0, 8.0, 10.0],
        })
        y = np.array([5, 4, 3, 2, 1])
        model = UTAEstimator(n_segments=2).fit(X, y)
        utilities = model.predict(X)
        assert _utilities_respect_ranking(utilities, y)



# Ties


class TestTiedRankings:

    def test_two_tied(self):
        X = pd.DataFrame({"x": [10.0, 5.0, 5.0, 1.0]})
        y = np.array([1, 2, 2, 3])
        model = UTAEstimator(n_segments=2).fit(X, y)
        utilities = model.predict(X)
        # Tied alternatives should have ≈ equal utility
        assert abs(utilities[1] - utilities[2]) < 1e-4
        # Strict preferences preserved
        assert utilities[0] > utilities[1] - 1e-6
        assert utilities[1] > utilities[3] - 1e-6

    def test_all_tied(self):
        X = pd.DataFrame({"x": [3.0, 7.0, 5.0]})
        y = np.array([1, 1, 1])
        model = UTAEstimator(n_segments=2).fit(X, y)
        utilities = model.predict(X)
        # All should be approximately equal
        assert np.std(utilities) < 1e-4



# Mixed criterion types


class TestMixedCriteria:

    def test_cardinal_and_nominal(self):
        X = pd.DataFrame({
            "price": [10.0, 7.0, 1.0, 3.0],
            "color": ["red", "blue", "green", "red"],
        })
        y = np.array([1, 2, 4, 3])
        model = UTAEstimator(
            criteria=[
                CardinalCriterion("price", n_segments=2),
                NominalCriterion("color", categories=["red", "blue", "green"]),
            ],
        ).fit(X, y)
        utilities = model.predict(X)
        assert _utilities_respect_ranking(utilities, y)

    def test_cardinal_and_ordinal(self):
        X = pd.DataFrame({
            "price": [10.0, 7.0, 3.0, 1.0],
            "quality": ["excellent", "good", "fair", "poor"],
        })
        y = np.array([1, 2, 3, 4])
        model = UTAEstimator(
            criteria=[
                CardinalCriterion("price", n_segments=2),
                OrdinalCriterion("quality", categories=["poor", "fair", "good", "excellent"]),
            ],
        ).fit(X, y)
        utilities = model.predict(X)
        assert _utilities_respect_ranking(utilities, y)

    def test_all_three_types(self):
        X = pd.DataFrame({
            "price":   [10.0, 7.0, 3.0, 1.0],
            "quality": ["excellent", "good", "fair", "poor"],
            "brand":   ["A", "B", "A", "C"],
        })
        y = np.array([1, 2, 3, 4])
        model = UTAEstimator(
            criteria=[
                CardinalCriterion("price", n_segments=2),
                OrdinalCriterion("quality", categories=["poor", "fair", "good", "excellent"]),
                NominalCriterion("brand", categories=["A", "B", "C"]),
            ],
        ).fit(X, y)
        utilities = model.predict(X)
        assert _utilities_respect_ranking(utilities, y)



# Criterion type auto-detection


class TestAutoDetection:

    def test_numeric_detected_as_cardinal(self):
        X = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        y = np.array([3, 2, 1])
        model = UTAEstimator(n_segments=2).fit(X, y)
        assert model.criteria_[0].type == "cardinal"

    def test_string_column_without_criterion_raises(self):
        """Non-numeric column missing from criteria list should raise a helpful error."""
        X = pd.DataFrame({"x": ["a", "b", "c"]})
        y = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="OrdinalCriterion or NominalCriterion"):
            UTAEstimator(n_segments=2).fit(X, y)

    def test_auto_nominal_numeric_categories_preserve_types(self):
        X = pd.DataFrame(
            {
                "price": [10.0, 7.0, 3.0, 1.0],
                "brand": [1, 2, 1, 3],
            }
        )
        y = np.array([1, 2, 3, 4])

        model = UTAEstimator(
            criterion_types={"price": "cardinal", "brand": "nominal"},
            n_segments=2,
        ).fit(X, y)

        utilities = model.predict(X)
        assert _utilities_respect_ranking(utilities, y)
        brand_categories = model.criteria_[1].categories
        assert all(not isinstance(level, str) for level in brand_categories)

    def test_auto_ordinal_numeric_categories_preserve_types(self):
        X = pd.DataFrame({"quality": [1, 2, 3, 4]})
        y = np.array([4, 3, 2, 1])

        model = UTAEstimator(
            criterion_types={"quality": "ordinal"},
            n_segments=2,
        ).fit(X, y)

        utilities = model.predict(X)
        assert _utilities_respect_ranking(utilities, y)
        quality_categories = model.criteria_[0].categories
        assert all(not isinstance(level, str) for level in quality_categories)


class TestCategoryTypeHandling:

    def test_explicit_string_categories_with_numeric_values_raise_clear_error(self):
        X = pd.DataFrame({"quality": [1, 2, 3]})
        y = np.array([3, 2, 1])
        model = UTAEstimator(
            criteria=[OrdinalCriterion("quality", categories=["1", "2", "3"])],
            n_segments=2,
        )

        with pytest.raises(ValueError) as exc:
            model.fit(X, y)

        err = str(exc.value)
        assert "Unknown category/value" in err
        assert "type=" in err



# Providing Criterion objects directly


class TestCustomCriterionObjects:

    def test_pass_criterion_objects(self):
        crit = CardinalCriterion("price", n_segments=3, shape="gain",
                                 min_val=0, max_val=10)
        X = pd.DataFrame({"price": [10.0, 5.0, 1.0]})
        y = np.array([1, 2, 3])
        model = UTAEstimator(criteria=[crit]).fit(X, y)
        assert model.criteria_[0].n_segments == 3
        utilities = model.predict(X)
        assert _utilities_respect_ranking(utilities, y)



# Score (Kendall's tau)


class TestScore:

    def test_perfect_ranking_score(self):
        """When preferences are perfectly learnable, |tau| should be high."""
        X = pd.DataFrame({
            "a": [10.0, 8.0, 6.0, 4.0, 2.0],
            "b": [9.0, 7.0, 5.0, 3.0, 1.0],
        })
        y = np.array([1, 2, 3, 4, 5])
        model = UTAEstimator(n_segments=3).fit(X, y)
        tau = model.score(X, y)
        assert abs(tau) > 0.8

    def test_score_is_float(self):
        X, y = _simple_data()
        model = UTAEstimator(n_segments=2).fit(X, y)
        tau = model.score(X, y)
        assert isinstance(tau, float)



# Objective value (error minimisation)


class TestObjectiveValue:

    def test_objective_is_non_negative(self):
        X, y = _simple_data()
        model = UTAEstimator(n_segments=2).fit(X, y)
        assert model.objective_value_ >= -1e-10

    def test_zero_error_for_simple_case(self):
        """Monotonically consistent data should have near-zero error."""
        X = pd.DataFrame({"x": [10.0, 5.0, 1.0]})
        y = np.array([1, 2, 3])
        model = UTAEstimator(n_segments=2).fit(X, y)
        assert model.objective_value_ < 0.01



# Sigma parameter


class TestSigma:

    def test_default_sigma(self):
        model = UTAEstimator()
        assert model.sigma == 0.001

    def test_custom_sigma(self):
        model = UTAEstimator(sigma=0.05)
        assert model.sigma == 0.05

    def test_larger_sigma_still_correct(self):
        X, y = _simple_data()
        model = UTAEstimator(n_segments=2, sigma=0.01).fit(X, y)
        utilities = model.predict(X)
        assert _utilities_respect_ranking(utilities, y)



# Edge cases


class TestEdgeCases:

    def test_two_alternatives(self):
        """Minimum viable problem: two alternatives, one criterion."""
        X = pd.DataFrame({"x": [10.0, 1.0]})
        y = np.array([1, 2])
        model = UTAEstimator(n_segments=2).fit(X, y)
        utilities = model.predict(X)
        assert utilities[0] >= utilities[1] - 1e-6

    def test_single_criterion(self):
        X = pd.DataFrame({"x": [10.0, 7.0, 3.0, 1.0]})
        y = np.array([1, 2, 3, 4])
        model = UTAEstimator(n_segments=3).fit(X, y)
        utilities = model.predict(X)
        assert _utilities_respect_ranking(utilities, y)

    def test_many_segments(self):
        X = pd.DataFrame({"x": [10.0, 7.0, 3.0, 1.0]})
        y = np.array([1, 2, 3, 4])
        model = UTAEstimator(n_segments=10).fit(X, y)
        utilities = model.predict(X)
        assert _utilities_respect_ranking(utilities, y)

    def test_numpy_input(self):
        """Accepts raw numpy array (auto-names columns)."""
        X = np.array([[10, 8], [5, 4], [1, 2]])
        y = np.array([1, 2, 3])
        model = UTAEstimator(n_segments=2).fit(X, y)
        utilities = model.predict(X)
        assert _utilities_respect_ranking(utilities, y)

    def test_invalid_criterion_type_raises(self):
        X = pd.DataFrame({"x": [1.0, 2.0]})
        y = np.array([1, 2])
        model = UTAEstimator(criteria=["not_a_criterion"])
        with pytest.raises(ValueError, match="Criterion instances"):
            model.fit(X, y)



# Reproducibility


class TestReproducibility:

    def test_same_data_same_result(self):
        """Fitting twice on the same data yields identical utilities."""
        X, y = _simple_data()
        u1 = UTAEstimator(n_segments=2).fit(X, y).predict(X)
        u2 = UTAEstimator(n_segments=2).fit(X, y).predict(X)
        np.testing.assert_allclose(u1, u2)



# Mathematical property: monotonicity of marginal utilities for gain


class TestMathProperties:

    def test_utility_increases_with_value_for_gain(self):
        """For a single gain criterion, utility should be non-decreasing."""
        X = pd.DataFrame({"x": [1.0, 3.0, 5.0, 7.0, 10.0]})
        y = np.array([5, 4, 3, 2, 1])
        model = UTAEstimator(n_segments=3).fit(X, y)
        utilities = model.predict(X)
        # Utility should increase with x
        for i in range(len(utilities) - 1):
            assert utilities[i] <= utilities[i + 1] + 1e-6

    def test_prediction_on_unseen_data(self):
        """Model can predict on values not in training set."""
        X_train = pd.DataFrame({"x": [0.0, 5.0, 10.0]})
        y_train = np.array([3, 2, 1])
        model = UTAEstimator(n_segments=2).fit(X_train, y_train)

        X_test = pd.DataFrame({"x": [2.5, 7.5]})
        utilities = model.predict(X_test)
        assert utilities.shape == (2,)
        # 2.5 < 7.5 value-wise; for gain, higher value = higher utility
        assert utilities[0] <= utilities[1] + 1e-6

    def test_error_vars_are_non_negative(self):
        """All Sp and Sm variables in solution must be >= 0."""
        X, y = _simple_data()
        model = UTAEstimator(n_segments=2).fit(X, y)
        for var, val in model.marginal_utilities_.items():
            if var.startswith("Sp") or var.startswith("Sm"):
                assert val >= -1e-10, f"{var} = {val} is negative"

    def test_cardinal_breakpoints_prune_empty_segments_with_custom_points(self):
        X = pd.DataFrame({"x": [0.0, 1.0, 5.0, 5.0]})
        y = np.array([4, 3, 1, 2])
        model = UTAEstimator(
            n_segments=5,
            breakpoints={"x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]},
        ).fit(X, y)

        positions = [float(bp.position) for bp in model.criteria_[0].breakpoints]
        assert positions == [0.0, 1.0, 5.0]

    def test_cardinal_breakpoints_keep_minimum_two_segments(self):
        X = pd.DataFrame({"x": [0.0, 0.0, 0.0, 5.0]})
        y = np.array([1, 1, 1, 2])
        model = UTAEstimator(
            n_segments=5,
            breakpoints={"x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]},
        ).fit(X, y)

        positions = [float(bp.position) for bp in model.criteria_[0].breakpoints]
        assert len(positions) == 2
        assert positions == [0.0, 5.0]

    def test_quantile_breakpoints_respect_cardinal_min_max_bounds(self):
        X = pd.DataFrame({"x": [20.0, 40.0, 60.0, 80.0]})
        y = np.array([4, 3, 2, 1])
        criteria = [CardinalCriterion("x", n_segments=3, min_val=0.0, max_val=100.0)]

        model = UTAEstimator(
            criteria=criteria,
            n_segments=3,
            breakpoints="quantile",
        ).fit(X, y)

        positions = [float(bp.position) for bp in model.criteria_[0].breakpoints]
        assert positions[0] == 0.0
        assert positions[-1] == 100.0



# Algorithm switching (UTASTAR vs UTANM)


class TestAlgorithmSwitching:

    def test_invalid_algorithm_raises(self):
        X, y = _simple_data()
        model = UTAEstimator(algorithm="INVALID")
        with pytest.raises(ValueError, match="Unknown algorithm"):
            model.fit(X, y)

    def test_utastar_marginal_utilities_nonnegative(self):
        """UTASTAR enforces monotonic segment increments."""
        X, y = _simple_data()
        model = UTAEstimator(n_segments=2, algorithm="UTASTAR").fit(X, y)
        for var, val in model.marginal_utilities_.items():
            if var.startswith("w_"):
                assert val >= -1e-10, f"{var} = {val} is negative under UTASTAR"

    def test_utastar_nominal_first_category_is_zero_baseline(self):
        X = pd.DataFrame({"brand": ["A", "B", "C", "B"]})
        y = np.array([4, 2, 1, 3])
        model = UTAEstimator(
            algorithm="UTASTAR",
            criteria=[NominalCriterion("brand", categories=["A", "B", "C"])],
        ).fit(X, y)

        nominal_values = model.partial_values_["brand"]
        assert abs(nominal_values[0]) < 1e-9
        assert np.all(nominal_values >= -1e-10)

    def test_utanm_allows_negative_marginal_utilities(self):
        """UTANM allows w_i_j < 0 (non-monotonic utility)."""
        # Reversed ranking: smaller values preferred → needs negative w
        X = pd.DataFrame({"x": [10.0, 7.0, 3.0, 1.0]})
        y = np.array([4, 3, 2, 1])
        model = UTAEstimator(n_segments=2, algorithm="UTANM").fit(X, y)
        w_vars = {k: v for k, v in model.marginal_utilities_.items()
                  if k.startswith("w_")}
        # At least one w should be negative for reversed preference
        assert any(v < -1e-10 for v in w_vars.values()), \
            "Expected negative marginal utilities under UTANM with reversed ranking"

    def test_utastar_preserves_ranking(self):
        X, y = _simple_data()
        model = UTAEstimator(n_segments=2, algorithm="UTASTAR").fit(X, y)
        utilities = model.predict(X)
        assert _utilities_respect_ranking(utilities, y)

    def test_utanm_preserves_ranking(self):
        X, y = _simple_data()
        model = UTAEstimator(n_segments=2, algorithm="UTANM").fit(X, y)
        utilities = model.predict(X)
        assert _utilities_respect_ranking(utilities, y)

    def test_default_algorithm_is_utastar(self):
        model = UTAEstimator()
        assert model.algorithm == "UTASTAR"

    def test_utanm_exposes_iteration_metadata(self):
        X, y = _simple_data()
        model = UTAEstimator(
            n_segments=2,
            algorithm="UTANM",
            max_nonmonotonicity_degree=2,
            objective_threshold=0.0,
            minimum_improvement=0.0,
        ).fit(X, y)

        assert hasattr(model, "selected_nonmonotonicity_degree_")
        assert hasattr(model, "utanm_iteration_trace_")
        assert model.selected_nonmonotonicity_degree_ >= 0
        assert isinstance(model.utanm_iteration_trace_, list)
        assert len(model.utanm_iteration_trace_) >= 1

    def test_utanm_respects_nonmonotonicity_cap_zero(self):
        X, y = _simple_data()
        model = UTAEstimator(
            n_segments=2,
            algorithm="UTANM",
            max_nonmonotonicity_degree=0,
            objective_threshold=0.0,
            minimum_improvement=0.0,
        ).fit(X, y)

        total_shape_changes = sum(int(np.sum(v)) for v in model.shape_change_flags_.values())
        assert total_shape_changes <= 0
