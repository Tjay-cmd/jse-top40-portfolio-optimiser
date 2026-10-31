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
        
        Position caps are enforced at rebalancing. Between rebalances, weights may
        drift due to price changes, which is allowed in this implementation.
        
        Parameters
        ----------
        weights : pd.Series
            Unconstrained weights
            
        Returns
        -------
        pd.Series
            Constrained weights with all constraints satisfied
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
        
        # Iteratively apply max position constraint with tighter tolerance
        # Ensures caps are strictly enforced at rebalance time
        max_iterations = 100
        tolerance = 1e-8  # Stricter tolerance for position caps
        
        for iteration in range(max_iterations):
            # Normalize first to ensure we're working with weights that sum to 1
            if weights.sum() > 1e-10:
                weights = weights / weights.sum()
            
            # Check which weights exceed max (with tight tolerance)
            violations = weights > self.max_position_size + tolerance
            
            if not violations.any():
                break  # Converged
            
            # Cap violating weights strictly
            excess_total = (weights[violations] - self.max_position_size).sum()
            weights[violations] = self.max_position_size
            
            # Redistribute excess to non-violating, non-zero weights
            can_absorb = (weights < self.max_position_size - tolerance) & (weights > tolerance)
            
            if can_absorb.any():
                # Proportionally distribute excess
                absorb_weights = weights[can_absorb]
                absorb_capacity = (self.max_position_size - absorb_weights).sum()
                
                if absorb_capacity > excess_total + tolerance:
                    # Can absorb proportionally
                    weights[can_absorb] += excess_total * (absorb_weights / absorb_weights.sum())
                else:
                    # Not enough capacity - cap absorbers and iterate
                    # Distribute what we can proportionally
                    if absorb_capacity > 0:
                        scale_factor = min(1.0, absorb_capacity / excess_total)
                        weights[can_absorb] += scale_factor * excess_total * (absorb_weights / absorb_weights.sum())
            
            # Check for convergence failure
            if iteration == max_iterations - 1:
                print(f"Warning: Position constraint did not converge after {max_iterations} iterations")
                # Force strict cap
                weights = weights.clip(upper=self.max_position_size)
        
        # Final strict normalization and capping
        if weights.sum() > 1e-10:
            weights = weights / weights.sum()
        else:
            # Fallback to equal weight
            weights = pd.Series(1.0 / len(weights), index=weights.index)
        
        # Absolute final strict clip to ensure no position exceeds max
        weights = weights.clip(upper=self.max_position_size)
        
        # Final renormalization after clipping
        if weights.sum() > 0:
            weights = weights / weights.sum()
        
        # Verify constraints are met
        max_weight = weights.max()
        if max_weight > self.max_position_size + tolerance:
            print(f"ERROR: Max weight {max_weight:.6f} exceeds limit {self.max_position_size:.6f}")
        
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
