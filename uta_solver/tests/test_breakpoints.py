"""
Tests for breakpoint classes.
"""

import pytest
import numpy as np

from uta_solver.breakpoints import CardinalBreakpoint, OrdinalBreakpoint, NominalBreakpoint




# CardinalBreakpoint


class TestCardinalBreakpoint:
    """Unit tests for CardinalBreakpoint."""

    def test_variable_name(self, cardinal_criterion):
        """Each breakpoint's variable name encodes criterion order and bp index."""
        names = [bp.get_marginal_utility_var_name() for bp in cardinal_criterion.breakpoints]
        assert names == ["w_0_0", "w_0_1", "w_0_2"]

    def test_position_values(self, cardinal_criterion):
        """Breakpoints are evenly spaced across the value range."""
        positions = [bp.position for bp in cardinal_criterion.breakpoints]
        np.testing.assert_allclose(positions, [0.0, 5.0, 10.0])

    def test_repr(self, cardinal_criterion):
        bp = cardinal_criterion.breakpoints[0]
        assert "CardinalBreakpoint" in repr(bp)
        assert "price" in repr(bp)



# OrdinalBreakpoint


class TestOrdinalBreakpoint:
    """Unit tests for OrdinalBreakpoint."""

    def test_positions_are_categories(self, ordinal_criterion):
        positions = [bp.position for bp in ordinal_criterion.breakpoints]
        assert positions == ["poor", "fair", "good", "excellent"]

    def test_variable_names(self, ordinal_criterion):
        names = [bp.get_marginal_utility_var_name() for bp in ordinal_criterion.breakpoints]
        assert names == ["w_0_0", "w_0_1", "w_0_2", "w_0_3"]

    def test_repr(self, ordinal_criterion):
        bp = ordinal_criterion.breakpoints[1]
        assert "OrdinalBreakpoint" in repr(bp)
        assert "fair" in repr(bp)



# NominalBreakpoint


class TestNominalBreakpoint:
    """Unit tests for NominalBreakpoint."""

    def test_positions_are_categories(self, nominal_criterion):
        positions = [bp.position for bp in nominal_criterion.breakpoints]
        assert positions == ["red", "green", "blue"]

    def test_repr(self, nominal_criterion):
        bp = nominal_criterion.breakpoints[2]
        assert "NominalBreakpoint" in repr(bp)
        assert "blue" in repr(bp)
