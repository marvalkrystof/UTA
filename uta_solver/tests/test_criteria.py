"""
Tests for criterion classes (CardinalCriterion, OrdinalCriterion, NominalCriterion).
"""

import pytest
import numpy as np

from uta_solver.criteria import CardinalCriterion, OrdinalCriterion, NominalCriterion



# CardinalCriterion


class TestCardinalCriterionCreation:
    """Creation and breakpoint initialisation."""

    def test_type_is_cardinal(self):
        c = CardinalCriterion("x")
        assert c.type == "cardinal"

    def test_default_segments(self):
        c = CardinalCriterion("x")
        assert c.n_segments == 2

    def test_create_breakpoints_count(self):
        c = CardinalCriterion("x", n_segments=3)
        c.order = 0
        c.create_breakpoints(3, np.array([0, 10, 20, 30]))
        # n_segments + 1 breakpoints
        assert len(c.breakpoints) == 4

    def test_breakpoints_evenly_spaced(self):
        c = CardinalCriterion("x", n_segments=4)
        c.order = 0
        c.create_breakpoints(4, np.array([0.0, 100.0]))
        positions = [bp.position for bp in c.breakpoints]
        np.testing.assert_allclose(positions, [0, 25, 50, 75, 100])

    def test_min_max_override(self):
        """Explicit min/max take precedence over data."""
        c = CardinalCriterion("x", min_val=0, max_val=100)
        c.order = 0
        c.create_breakpoints(2, np.array([20, 50, 80]))
        assert c.breakpoints[0].position == 0
        assert c.breakpoints[-1].position == 100



# OrdinalCriterion


class TestOrdinalCriterion:

    def test_type_is_ordinal(self):
        c = OrdinalCriterion("q", categories=["low", "high"])
        assert c.type == "ordinal"

    def test_breakpoint_count_equals_categories(self):
        cats = ["low", "medium", "high"]
        c = OrdinalCriterion("q", categories=cats)
        c.order = 0
        c.create_breakpoints(2, np.array(cats))
        assert len(c.breakpoints) == 3



# NominalCriterion


class TestNominalCriterion:

    def test_type_is_nominal(self):
        c = NominalCriterion("c", categories=["a", "b"])
        assert c.type == "nominal"

    def test_breakpoint_count(self, nominal_criterion):
        assert len(nominal_criterion.breakpoints) == 3
