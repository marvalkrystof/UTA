import numpy as np
import pandas as pd
import pytest

from uta_solver import UTAEstimator, UTAStarRegressor, UTANMRegressor
from uta_solver.criteria import CardinalCriterion, NominalCriterion


def _dataset():
    X = pd.DataFrame(
        {
            "price": [10.0, 8.0, 7.0, 5.0, 3.0, 2.0],
            "size": [9.0, 8.0, 7.0, 5.0, 4.0, 2.0],
            "brand": ["A", "A", "B", "B", "C", "C"],
        }
    )
    return X


def test_reference_subset_fit_and_predict_all():
    X = _dataset()
    mask = np.array([True, True, False, True, False, True])
    X_ref = X[mask].reset_index(drop=True)
    y_ref = np.array([1, 2, 3, 4])

    model = UTAEstimator(
        algorithm="UTASTAR",
        criteria=[
            CardinalCriterion("price", n_segments=3, shape="cost"),
            CardinalCriterion("size", n_segments=3, shape="gain"),
            NominalCriterion("brand", categories=["A", "B", "C"]),
        ],
    ).fit(X_ref, y_ref)

    scores = model.predict(X)
    assert scores.shape == (len(X),)
    assert np.all(np.isfinite(scores))


def test_reference_mask_equivalence_and_constraint_count():
    X = _dataset()
    mask = np.array([True, True, False, True, False, True])
    X_ref = X[mask].reset_index(drop=True)
    y_ref = np.array([1, 2, 3, 4])

    criterion_types = {"price": "cardinal", "size": "cardinal", "brand": "nominal"}
    direct = UTAEstimator(algorithm="UTASTAR", criterion_types=criterion_types).fit(X_ref, y_ref)
    masked = UTAEstimator(algorithm="UTASTAR", criterion_types=criterion_types).fit(X, y_ref, reference_mask=mask)

    np.testing.assert_allclose(direct._u_values_, masked._u_values_, atol=1e-6)
    assert masked.n_reference_constraints_ == direct.n_reference_constraints_


def test_reference_mask_validation():
    X = _dataset()
    y_ref = np.array([1, 2, 3, 4])
    with pytest.raises(ValueError, match="boolean"):
        UTAEstimator().fit(X, y_ref, reference_mask=np.array([1, 0, 1, 0, 1, 0]))


def test_utastar_solver_metadata_fixed_backend():
    X = pd.DataFrame({"x": [10.0, 7.0, 2.0], "y": [9.0, 6.0, 1.0]})
    y = np.array([1, 2, 3])
    model = UTAStarRegressor().fit(X, y)
    assert model.solver_ == "highs"
    assert model.solver_status_ == "optimal"
    assert isinstance(model.solver_message_, str)


def test_utanm_solver_metadata_fixed_backend():
    X = pd.DataFrame({"x": [10.0, 7.0, 2.0, 1.0]})
    y = np.array([1, 2, 3, 4])

    model = UTANMRegressor(theta=0.1).fit(X, y)
    assert model.solver_ == "scipy_milp"
    assert model.solver_status_ == "optimal"
    assert model.mip_gap_ is None or model.mip_gap_ >= 0.0


def test_utanm_paper_normalization_properties():
    X = pd.DataFrame(
        {
            "x": [0.0, 2.0, 4.0, 6.0, 8.0],
            "y": [8.0, 6.0, 4.0, 2.0, 0.0],
        }
    )
    ranks = np.array([3, 2, 1, 4, 5])

    model = UTANMRegressor(theta=0.2).fit(X, ranks)

    sum_max = sum(v["m_max"] for v in model.criterion_extrema_.values())
    assert abs(sum_max - 1.0) < 1e-4

    for criterion_name, values in model.partial_values_.items():
        assert np.any(np.isclose(values, 0.0, atol=1e-6)), criterion_name


def test_utanm_normalization_mode_argument_removed():
    with pytest.raises(TypeError):
        UTANMRegressor(normalization="paper")


def test_nominal_unseen_category_policy_and_no_shape_flags():
    X = pd.DataFrame({"x": [10.0, 7.0, 2.0], "brand": ["A", "B", "C"]})
    y = np.array([1, 2, 3])

    model = UTAEstimator(
        algorithm="UTANM",
        criteria=[
            CardinalCriterion("x", n_segments=2),
            NominalCriterion("brand", categories=["A", "B", "C"]),
        ],
        handle_unknown="error",
    ).fit(X, y)

    assert model.shape_change_flags_["brand"].size == 0

    with pytest.raises(ValueError, match="Unknown category"):
        model.predict(pd.DataFrame({"x": [5.0], "brand": ["D"]}))

    model_ignore = UTAEstimator(
        algorithm="UTANM",
        criteria=[
            CardinalCriterion("x", n_segments=2),
            NominalCriterion("brand", categories=["A", "B", "C"]),
        ],
        handle_unknown="ignore",
    ).fit(X, y)

    out = model_ignore.predict(pd.DataFrame({"x": [5.0], "brand": ["D"]}))
    assert np.isfinite(out[0])


def test_utanm_degree_tie_keeps_lowest_degree(monkeypatch):
    X = pd.DataFrame({"x": [10.0, 7.0, 2.0, 1.0]})
    y = np.array([1, 2, 3, 4])

    def fake_solve(self, X, y, shape_cap):
        objective = 1.0
        return {
            "solver": "mock",
            "solver_status": "optimal",
            "solver_message": "mock",
            "mip_gap": None,
            "objective_value": objective,
            "u": np.zeros(self._n_u_vars_),
            "sp": np.zeros(len(X)),
            "sm": np.zeros(len(X)),
            "shape_change_flags": {
                criterion.name: np.zeros(max(len(criterion.breakpoints) - 2, 0), dtype=int)
                for criterion in self.criteria_
            },
            "shape_penalty_value": 0.0,
            "criterion_extrema": {
                criterion.name: {"min": 0.0, "max": 0.0, "m_max": 0.0}
                for criterion in self.criteria_
            },
            "n_reference_constraints": 0,
        }

    monkeypatch.setattr(UTANMRegressor, "_solve_with_shape_cap", fake_solve)

    model = UTANMRegressor(
        max_nonmonotonicity_degree=2,
        minimum_improvement=0.0,
        objective_threshold=0.0,
    ).fit(X, y)

    assert model.selected_nonmonotonicity_degree_ == 0


def test_utanm_degree_strict_improvement_updates_selection(monkeypatch):
    X = pd.DataFrame({"x": [10.0, 7.0, 2.0, 1.0]})
    y = np.array([1, 2, 3, 4])

    def fake_solve(self, X, y, shape_cap):
        objective_by_degree = {0: 1.0, 1: 0.5, 2: 0.5}
        objective = objective_by_degree[int(shape_cap)]
        return {
            "solver": "mock",
            "solver_status": "optimal",
            "solver_message": "mock",
            "mip_gap": None,
            "objective_value": objective,
            "u": np.zeros(self._n_u_vars_),
            "sp": np.zeros(len(X)),
            "sm": np.zeros(len(X)),
            "shape_change_flags": {
                criterion.name: np.zeros(max(len(criterion.breakpoints) - 2, 0), dtype=int)
                for criterion in self.criteria_
            },
            "shape_penalty_value": 0.0,
            "criterion_extrema": {
                criterion.name: {"min": 0.0, "max": 0.0, "m_max": 0.0}
                for criterion in self.criteria_
            },
            "n_reference_constraints": 0,
        }

    monkeypatch.setattr(UTANMRegressor, "_solve_with_shape_cap", fake_solve)

    model = UTANMRegressor(
        max_nonmonotonicity_degree=2,
        minimum_improvement=0.0,
        objective_threshold=0.0,
    ).fit(X, y)

    assert model.selected_nonmonotonicity_degree_ == 1
