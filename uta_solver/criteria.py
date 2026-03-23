"""
Criterion classes for different types of evaluation dimensions.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np


class Criterion(ABC):
    """
    Base class for all criterion types.
    
    A criterion represents an evaluation dimension (feature) in the
    multi-criteria decision problem.
    """
    
    def __init__(self, name: str, criterion_type: str):
        self.name = name
        self.type = criterion_type
        self.breakpoints: List = []
        self._order: int = 0
        
    @abstractmethod
    def create_breakpoints(self, n_segments: int, values: np.ndarray):
        """Create breakpoints for this criterion based on data."""
        pass
    
    @abstractmethod
    def get_utility_coefficients(self, value) -> List[Tuple[int, float]]:
        """
        Get breakpoint index/coefficient pairs for a criterion value.

        Parameters
        ----------
        value : float or str
            The criterion value

        Returns
        -------
        List[Tuple[int, float]]
            List of (breakpoint_index, coefficient) pairs
        """
        pass
    
    @property
    def order(self) -> int:
        """Position in the criteria list (0-indexed)."""
        return self._order
    
    @order.setter
    def order(self, value: int):
        self._order = value


class CardinalCriterion(Criterion):
    """
    Numeric criterion with continuous values.
    
    Examples: price, age, temperature, distance
    
    Utility is piecewise linear with breakpoints dividing the range
    into segments.
    """
    
    def __init__(self, name: str, n_segments: int = 2, 
                 shape: str = 'gain', min_val: float = None, 
                 max_val: float = None):
        """
        Parameters
        ----------
        name : str
            Name of the criterion
        n_segments : int, default=2
            Number of segments (will create n_segments+1 breakpoints)
        shape : {'gain', 'cost'}, default='gain'
            Whether higher values are better (gain) or worse (cost)
        min_val : float, optional
            Minimum value (inferred from data if not provided)
        max_val : float, optional
            Maximum value (inferred from data if not provided)
        """
        super().__init__(name, 'cardinal')
        self.n_segments = n_segments
        self.shape = shape.lower()
        self.min_val = min_val
        self.max_val = max_val
        
    def create_breakpoints(self, n_segments: int, values: np.ndarray):
        """Create evenly-spaced breakpoints across the value range."""
        from .breakpoints import CardinalBreakpoint
        
        # Determine range
        if self.min_val is None:
            self.min_val = float(np.min(values))
        if self.max_val is None:
            self.max_val = float(np.max(values))
        
        # Create n_segments + 1 breakpoints
        positions = np.linspace(self.min_val, self.max_val, n_segments + 1)
        
        self.breakpoints = []
        for i, pos in enumerate(positions):
            bp = CardinalBreakpoint(self, pos, i)
            self.breakpoints.append(bp)

    def get_utility_coefficients(self, value: float) -> List[Tuple[int, float]]:
        """Get interpolation coefficients on neighbouring breakpoints."""
        value = float(value)

        if not self.breakpoints:
            return []

        if value <= self.breakpoints[0].position:
            return [(0, 1.0)]

        last_idx = len(self.breakpoints) - 1
        if value >= self.breakpoints[last_idx].position:
            return [(last_idx, 1.0)]

        for i in range(last_idx):
            left = self.breakpoints[i].position
            right = self.breakpoints[i + 1].position
            if left <= value <= right:
                denom = right - left
                if denom <= 0:
                    return [(i, 1.0)]
                alpha = (value - left) / denom
                return [(i, 1.0 - alpha), (i + 1, alpha)]

        return [(last_idx, 1.0)]


class OrdinalCriterion(Criterion):
    """
    Ordered categorical criterion.
    
    Examples: education level (high school < bachelor < master < PhD),
             size (S < M < L < XL), rating (poor < fair < good < excellent)
    
    Values have a natural order, but distances between levels are not meaningful.
    """
    
    def __init__(self, name: str, categories: List[str], n_segments: int = None):
        """
        Parameters
        ----------
        name : str
            Name of the criterion
        categories : List[str]
            Ordered list of category values (from worst to best for gain)
        n_segments : int, optional
            Number of segments (defaults to len(categories) - 1)
        """
        super().__init__(name, 'ordinal')
        self.categories = categories
        self.n_segments = n_segments or (len(categories) - 1)
        
    def create_breakpoints(self, n_segments: int, values: np.ndarray):
        """Create one breakpoint per category."""
        from .breakpoints import OrdinalBreakpoint
        
        self.breakpoints = []
        for i, category in enumerate(self.categories):
            bp = OrdinalBreakpoint(self, category, i)
            self.breakpoints.append(bp)

    def get_utility_coefficients(self, value: str) -> List[Tuple[int, float]]:
        """Return one-hot coefficient for the matching category."""
        for i, bp in enumerate(self.breakpoints):
            if bp.position == value:
                return [(i, 1.0)]
        return []


class NominalCriterion(Criterion):
    """
    Unordered categorical criterion.
    
    Examples: color, brand, location type, department
    
    No natural ordering between categories - each has independent utility.
    """
    
    def __init__(self, name: str, categories: List[str]):
        """
        Parameters
        ----------
        name : str
            Name of the criterion
        categories : List[str]
            List of category values (order doesn't matter)
        """
        super().__init__(name, 'nominal')
        self.categories = categories
        self.n_segments = len(categories) - 1
        
    def create_breakpoints(self, n_segments: int, values: np.ndarray):
        """Create one breakpoint per category."""
        from .breakpoints import NominalBreakpoint
        
        # Infer categories from data if not provided
        unique_values = list(np.unique(values))
        if not self.categories:
            self.categories = unique_values
        
        self.breakpoints = []
        for i, category in enumerate(self.categories):
            bp = NominalBreakpoint(self, category, i)
            self.breakpoints.append(bp)

    def get_utility_coefficients(self, value: str) -> List[Tuple[int, float]]:
        """Return coefficient 1.0 only for the matching category."""
        for i, bp in enumerate(self.breakpoints):
            if bp.position == value:
                return [(i, 1.0)]
        return []
