"""
Base class for portfolio optimizers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np
import pandas as pd


class PortfolioOptimizer(ABC):
    """Abstract base class for portfolio optimization methods."""
    
    def __init__(
        self,
        long_only: bool = True,
        max_position_size: float = 0.15,
        min_position_size: float = 0.01,
        max_stocks: Optional[int] = None
    ):
        """
        Initialize optimizer.
        
        Parameters
        ----------
        long_only : bool
            Whether to enforce long-only constraint
        max_position_size : float
            Maximum weight per position
        min_position_size : float
            Minimum weight per position (to avoid dust)
        max_stocks : Optional[int]
            Maximum number of positions
        """
        self.long_only = long_only
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        self.max_stocks = max_stocks
    
    @abstractmethod
    def optimize(
        self,
        returns: pd.DataFrame,
        **kwargs
    ) -> pd.Series:
        """
        Optimize portfolio weights.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Historical returns
        **kwargs
            Method-specific parameters
            
        Returns
        -------
        pd.Series
            Optimal weights indexed by ticker
        """
        pass
    
    def apply_constraints(self, weights: pd.Series) -> pd.Series:
        """
        Apply position constraints to weights iteratively.
        
        Parameters
        ----------
        weights : pd.Series
            Unconstrained weights
            
        Returns
        -------
        pd.Series
            Constrained weights
        """
        # Clip negative weights if long-only
        if self.long_only:
            weights = weights.clip(lower=0)
        
        # Remove positions below minimum size
        weights[weights < self.min_position_size] = 0
        
        # If max_stocks specified, keep only top N positions
        if self.max_stocks is not None and (weights > 0).sum() > self.max_stocks:
            top_stocks = weights.nlargest(self.max_stocks)
            weights = pd.Series(0.0, index=weights.index)
            weights.loc[top_stocks.index] = top_stocks.values
        
        # Iteratively apply max position constraint
        # This handles cases where after renormalization some weights exceed max again
        max_iterations = 50
        for iteration in range(max_iterations):
            # Check which weights exceed max
            violations = weights > self.max_position_size + 1e-10
            
            if not violations.any():
                break  # Converged
            
            # Cap violating weights
            excess_total = (weights[violations] - self.max_position_size).sum()
            weights[violations] = self.max_position_size
            
            # Redistribute excess to non-violating, non-zero weights
            can_absorb = (weights < self.max_position_size - 1e-10) & (weights > 1e-10)
            
            if can_absorb.any():
                # Proportionally distribute excess
                absorb_weights = weights[can_absorb]
                absorb_capacity = (self.max_position_size - absorb_weights).sum()
                
                if absorb_capacity > excess_total:
                    # Can absorb proportionally
                    weights[can_absorb] += excess_total * (absorb_weights / absorb_weights.sum())
                else:
                    # Max out absorbers and spread remaining
                    weights[can_absorb] = self.max_position_size
            
            # Renormalize
            if weights.sum() > 1e-10:
                weights = weights / weights.sum()
        
        # Final safety normalization
        if weights.sum() > 1e-10:
            weights = weights / weights.sum()
        else:
            weights = pd.Series(1.0 / len(weights), index=weights.index)
        
        # Absolute final clip (should be no-op if converged)
        weights = weights.clip(upper=self.max_position_size)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        
        return weights
    
    def validate_weights(self, weights: pd.Series) -> bool:
        """
        Validate that weights satisfy constraints.
        
        Parameters
        ----------
        weights : pd.Series
            Portfolio weights
            
        Returns
        -------
        bool
            True if valid
        """
        # Check sum to 1
        if not np.isclose(weights.sum(), 1.0, atol=1e-6):
            return False
        
        # Check long-only
        if self.long_only and (weights < -1e-6).any():
            return False
        
        # Check position limits
        if (weights > self.max_position_size + 1e-6).any():
            return False
        
        return True
