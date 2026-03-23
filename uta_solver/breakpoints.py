"""
Breakpoints for piecewise utility functions.
"""

from typing import Union


class Breakpoint:
    """A point in a criterion's utility function with an associated marginal utility variable."""

    def __init__(self, criterion: 'Criterion', position: Union[float, str], index: int):
        self.criterion = criterion
        self.position = position
        self.index = index

    def get_marginal_utility_var_name(self) -> str:
        """Variable name format: w_{criterion_order}_{breakpoint_index}"""
        return f"w_{self.criterion.order}_{self.index}"


class CardinalBreakpoint(Breakpoint):
    """Breakpoint for cardinal (numeric) criteria."""
    
    def __init__(self, criterion: 'CardinalCriterion', position: float, index: int):
        super().__init__(criterion, float(position), index)
        
    def __repr__(self):
        return f"CardinalBreakpoint({self.criterion.name}, pos={self.position:.2f})"


class OrdinalBreakpoint(Breakpoint):
    """Breakpoint for ordinal (ordered categorical) criteria."""
    
    def __init__(self, criterion: 'OrdinalCriterion', position: str, index: int):
        super().__init__(criterion, position, index)
        
    def __repr__(self):
        return f"OrdinalBreakpoint({self.criterion.name}, pos='{self.position}')"


class NominalBreakpoint(Breakpoint):
    """Breakpoint for nominal (unordered categorical) criteria."""
    
    def __init__(self, criterion: 'NominalCriterion', position: str, index: int):
        super().__init__(criterion, position, index)
        
    def __repr__(self):
        return f"NominalBreakpoint({self.criterion.name}, pos='{self.position}')"
