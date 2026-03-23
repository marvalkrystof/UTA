"""
Shared fixtures for uta-solver tests.
"""

import pytest
import numpy as np
import pandas as pd
import importlib

from uta_solver.criteria import CardinalCriterion, OrdinalCriterion, NominalCriterion

@pytest.fixture
def cardinal_criterion():
    """A simple cardinal (gain) criterion with 2 segments over [0, 10]."""
    crit = CardinalCriterion("price", n_segments=2, shape="gain")
    crit.order = 0
    crit.create_breakpoints(2, np.array([0, 5, 10]))
    return crit


@pytest.fixture
def cost_criterion():
    """A cardinal cost criterion with 2 segments over [0, 10]."""
    crit = CardinalCriterion("cost", n_segments=2, shape="cost")
    crit.order = 1
    crit.create_breakpoints(2, np.array([0, 5, 10]))
    return crit


@pytest.fixture
def ordinal_criterion():
    """An ordinal criterion with categories: poor < fair < good < excellent."""
    crit = OrdinalCriterion("quality", categories=["poor", "fair", "good", "excellent"])
    crit.order = 0
    crit.create_breakpoints(3, np.array(["poor", "fair", "good", "excellent"]))
    return crit


@pytest.fixture
def nominal_criterion():
    """A nominal criterion with categories: red, green, blue."""
    crit = NominalCriterion("color", categories=["red", "green", "blue"])
    crit.order = 0
    crit.create_breakpoints(2, np.array(["red", "green", "blue"]))
    return crit


@pytest.fixture
def simple_dataframe():
    """Small dataset with two cardinal criteria for basic testing."""
    return pd.DataFrame({
        "price": [10.0, 7.0, 1.0, 3.0],
        "size":  [10.0, 6.0, 2.0, 2.0],
    })


@pytest.fixture
def simple_rankings():
    """Rankings for simple_dataframe (1 = best)."""
    return np.array([1, 2, 4, 3])


@pytest.fixture
def mixed_dataframe():
    """Dataset mixing cardinal, ordinal and nominal criteria."""
    return pd.DataFrame({
        "price":     [10.0, 7.0, 1.0, 3.0],
        "size":      [10.0, 6.0, 2.0, 2.0],
        "condition": ["excellent", "good", "poor", "poor"],
    })


@pytest.fixture
def mixed_rankings():
    """Rankings for mixed_dataframe."""
    return np.array([1, 2, 4, 3])
