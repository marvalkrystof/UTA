"""
UTA (UTilités Additives) - Multi-criteria decision analysis library

This library implements the UTA family of algorithms for learning additive utility
functions from preference rankings.
"""

from .estimator import UTAEstimator, UTAStarRegressor, UTANMRegressor
from .criteria import CardinalCriterion, OrdinalCriterion, NominalCriterion

__version__ = '0.1.0'
__all__ = [
    'UTAEstimator',
    'UTAStarRegressor',
    'UTANMRegressor',
    'CardinalCriterion',
    'OrdinalCriterion', 
    'NominalCriterion',
]
