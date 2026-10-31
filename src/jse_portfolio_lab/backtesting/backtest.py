"""
Walk-forward backtesting with transaction costs.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..optimization.base import PortfolioOptimizer


@dataclass
class BacktestResult:
    """Results from a backtest."""
    
    portfolio_value: pd.Series
    weights_history: pd.DataFrame
    turnover: pd.Series
    transaction_costs: pd.Series
    returns: pd.Series
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'portfolio_value': self.portfolio_value,
            'weights_history': self.weights_history,
            'turnover': self.turnover,
            'transaction_costs': self.transaction_costs,
            'returns': self.returns
        }


class Backtest:
    """Walk-forward backtesting engine."""
    
    def __init__(
        self,
        optimizer: PortfolioOptimizer,
        rebalance_frequency: str = 'monthly',
        transaction_cost_bps: float = 30.0,
        initial_capital: float = 1000000.0,
        min_history_days: int = 252
    ):
        """
        Initialize backtest.
        
        Parameters
        ----------
        optimizer : PortfolioOptimizer
            Portfolio optimization method
        rebalance_frequency : str
            Rebalancing frequency ('monthly', 'quarterly', etc.)
        transaction_cost_bps : float
            Transaction costs in basis points
        initial_capital : float
            Initial portfolio value
        min_history_days : int
            Minimum history required before first allocation
        """
        self.optimizer = optimizer
        self.rebalance_frequency = rebalance_frequency
        self.transaction_cost_bps = transaction_cost_bps
        self.initial_capital = initial_capital
        self.min_history_days = min_history_days
    
    def _get_rebalance_dates(self, returns: pd.DataFrame) -> List[pd.Timestamp]:
        """Get rebalancing dates based on frequency."""
        dates = returns.index
        
        if self.rebalance_frequency == 'monthly':
            # Last trading day of each month
            rebalance_dates = dates.to_series().groupby(
                [dates.year, dates.month]
            ).last().tolist()
        elif self.rebalance_frequency == 'quarterly':
            rebalance_dates = dates.to_series().groupby(
                [dates.year, dates.quarter]
            ).last().tolist()
        else:
            raise ValueError(f"Unknown frequency: {self.rebalance_frequency}")
        
        # Filter out dates too early in the history
        min_date = dates[self.min_history_days]
        rebalance_dates = [d for d in rebalance_dates if d >= min_date]
        
        return rebalance_dates
    
    def _compute_turnover(
        self,
        old_weights: pd.Series,
        new_weights: pd.Series
    ) -> float:
        """
        Compute turnover between two weight vectors.
        
        Turnover = sum(|new_weight - old_weight|) / 2
        """
        # Align indices
        all_tickers = old_weights.index.union(new_weights.index)
        old_aligned = old_weights.reindex(all_tickers, fill_value=0)
        new_aligned = new_weights.reindex(all_tickers, fill_value=0)
        
        return np.abs(new_aligned - old_aligned).sum() / 2
    
    def run(
        self,
        returns: pd.DataFrame,
        prices: pd.DataFrame,
        verbose: bool = True
    ) -> BacktestResult:
        """
        Run walk-forward backtest.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Historical returns (daily)
        prices : pd.DataFrame
            Historical prices
        verbose : bool
            Whether to show progress
            
        Returns
        -------
        BacktestResult
            Backtest results
        """
        rebalance_dates = self._get_rebalance_dates(returns)
        
        # Initialize tracking
        portfolio_value = pd.Series(index=returns.index, dtype=float)
        weights_history = pd.DataFrame(index=rebalance_dates, columns=returns.columns)
        turnover_series = pd.Series(index=rebalance_dates, dtype=float)
        transaction_costs_series = pd.Series(index=rebalance_dates, dtype=float)
        
        current_value = self.initial_capital
        current_weights = pd.Series(0, index=returns.columns)
        
        portfolio_value.iloc[0] = self.initial_capital
        
        iterator = tqdm(rebalance_dates, desc="Backtesting") if verbose else rebalance_dates
        
        for i, rebalance_date in enumerate(iterator):
            # Get historical data up to rebalance date
            idx = returns.index.get_loc(rebalance_date)
            hist_returns = returns.iloc[:idx]
            
            if len(hist_returns) < self.min_history_days:
                continue
            
            # Optimize portfolio
            try:
                new_weights = self.optimizer.optimize(hist_returns)
            except Exception as e:
                if verbose:
                    print(f"Optimization failed at {rebalance_date}: {e}")
                # Keep current weights
                new_weights = current_weights
            
            # Compute turnover
            turnover = self._compute_turnover(current_weights, new_weights)
            turnover_series[rebalance_date] = turnover
            
            # Compute transaction costs
            tc = turnover * current_value * (self.transaction_cost_bps / 10000)
            transaction_costs_series[rebalance_date] = tc
            
            # Apply transaction costs
            current_value -= tc
            
            # Update weights
            current_weights = new_weights
            weights_history.loc[rebalance_date] = new_weights
            
            # Compute returns until next rebalance
            if i < len(rebalance_dates) - 1:
                next_rebalance = rebalance_dates[i + 1]
            else:
                next_rebalance = returns.index[-1]
            
            # Get period returns
            period_start_idx = idx
            period_end_idx = returns.index.get_loc(next_rebalance)
            
            for j in range(period_start_idx, period_end_idx + 1):
                date = returns.index[j]
                
                if j == period_start_idx:
                    portfolio_value[date] = current_value
                else:
                    # Compute portfolio return
                    daily_return = returns.iloc[j]
                    portfolio_return = (current_weights * daily_return).sum()
                    
                    # Update value (can't go below 0 for long-only)
                    current_value *= (1 + portfolio_return)
                    if current_value < 0:
                        current_value = 0  # Wipeout - portfolio worth nothing
                    portfolio_value[date] = current_value
                    
                    # Update weights due to price changes (drift)
                    # Clip returns to prevent data errors from causing negative weights
                    daily_return_clipped = daily_return.clip(lower=-0.95, upper=3.0)
                    current_weights = current_weights * (1 + daily_return_clipped)
                    
                    # Remove negative weights (shouldn't happen but safety check)
                    current_weights = current_weights.clip(lower=0)
                    
                    # Renormalize (handle case where all weights are zero)
                    weight_sum = current_weights.sum()
                    if weight_sum > 1e-10:
                        current_weights = current_weights / weight_sum
                    else:
                        # Portfolio wiped out, set to equal weight of remaining value
                        current_weights = pd.Series(
                            1.0 / len(current_weights),
                            index=current_weights.index
                        )
        
        # Fill any remaining NaN values
        portfolio_value = portfolio_value.ffill().bfill()
        
        # Compute portfolio returns
        portfolio_returns = portfolio_value.pct_change().fillna(0)
        
        return BacktestResult(
            portfolio_value=portfolio_value,
            weights_history=weights_history,
            turnover=turnover_series,
            transaction_costs=transaction_costs_series,
            returns=portfolio_returns
        )
