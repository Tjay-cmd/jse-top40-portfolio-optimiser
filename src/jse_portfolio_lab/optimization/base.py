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
        
        # Iteratively apply max position constraint and renormalize
        for _ in range(20):
            # Find weights exceeding max
            excess_mask = weights > self.max_position_size
            
            if not excess_mask.any():
                break
            
            # Set excess weights to max
            weights[excess_mask] = self.max_position_size
            
            # Rescale remaining weights
            excess_weight = weights[excess_mask].sum()
            remaining_mask = ~excess_mask & (weights > 0)
            remaining_weight = weights[remaining_mask].sum()
            
            if remaining_weight > 1e-10:
                # Distribute deficit across remaining weights
                deficit = 1.0 - excess_weight
                weights[remaining_mask] *= deficit / remaining_weight
            else:
                # All constrained, just normalize
                break
        
        # Final normalization
        if weights.sum() > 1e-10:
            weights = weights / weights.sum()
        else:
            # If all weights are zero, equal weight
            weights = pd.Series(1.0 / len(weights), index=weights.index)
        
        # Final clip to ensure no fp errors
        weights = weights.clip(upper=self.max_position_size)
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
