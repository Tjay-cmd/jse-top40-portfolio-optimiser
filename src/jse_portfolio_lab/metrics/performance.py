"""
Performance metrics for portfolio evaluation.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats


class PerformanceMetrics:
    """Calculate portfolio performance metrics."""
    
    def __init__(self, risk_free_rate: float = 0.07):
        """
        Initialize metrics calculator.
        
        Parameters
        ----------
        risk_free_rate : float
            Annual risk-free rate (as decimal, e.g., 0.07 for 7%)
        """
        self.risk_free_rate = risk_free_rate
    
    def annualized_return(self, returns: pd.Series) -> float:
        """
        Compute annualized return.
        
        Parameters
        ----------
        returns : pd.Series
            Daily returns
            
        Returns
        -------
        float
            Annualized return
        """
        if len(returns) == 0:
            return 0.0
        
        total_return = (1 + returns).prod() - 1
        n_years = len(returns) / 252
        
        if n_years <= 0:
            return 0.0
        
        # Handle negative total returns
        if total_return <= -1:
            return -1.0  # Total loss
        
        try:
            ann_return = (1 + total_return) ** (1 / n_years) - 1
            return ann_return if np.isfinite(ann_return) else 0.0
        except:
            return 0.0
    
    def annualized_volatility(self, returns: pd.Series) -> float:
        """
        Compute annualized volatility.
        
        Parameters
        ----------
        returns : pd.Series
            Daily returns
            
        Returns
        -------
        float
            Annualized volatility
        """
        return returns.std() * np.sqrt(252)
    
    def sharpe_ratio(self, returns: pd.Series) -> float:
        """
        Compute Sharpe ratio.
        
        Parameters
        ----------
        returns : pd.Series
            Daily returns
            
        Returns
        -------
        float
            Sharpe ratio
        """
        ann_return = self.annualized_return(returns)
        ann_vol = self.annualized_volatility(returns)
        
        if ann_vol == 0:
            return 0
        
        return (ann_return - self.risk_free_rate) / ann_vol
    
    def deflated_sharpe_ratio(
        self,
        returns: pd.Series,
        n_trials: int = 1,
        skew: Optional[float] = None,
        kurtosis: Optional[float] = None
    ) -> float:
        """
        Compute Deflated Sharpe Ratio (Bailey and Lopez de Prado, 2014).
        
        Adjusts Sharpe ratio for multiple testing and non-normality.
        Uses per-period (daily) Sharpe ratio as in the original paper.
        
        Parameters
        ----------
        returns : pd.Series
            Daily returns
        n_trials : int
            Number of strategies tested (including variations)
        skew : Optional[float]
            Skewness of returns
        kurtosis : Optional[float]
            Excess kurtosis of returns
            
        Returns
        -------
        float
            Deflated Sharpe ratio (probability that true SR > 0)
        """
        # Compute per-period (daily) Sharpe ratio, not annualized
        daily_rf = self.risk_free_rate / 252
        excess_returns = returns - daily_rf
        
        if excess_returns.std() == 0:
            return 0.0
        
        sr_daily = excess_returns.mean() / excess_returns.std()
        
        if skew is None:
            skew = returns.skew()
        if kurtosis is None:
            kurtosis = returns.kurtosis()
        
        T = len(returns)  # Number of observations
        
        # Expected maximum Sharpe under null hypothesis (Bailey & López de Prado, 2014)
        # SR_0 ≈ sqrt((1/T) * ln(N))
        sr_0 = np.sqrt((1 / T) * np.log(n_trials))
        
        # Variance of Sharpe ratio estimator (Bailey & López de Prado, 2014, Eq. 6)
        # Var[SR] = (1/T) * [1 + 0.5*SR^2 - skew*SR + (kurt/4)*SR^2]
        var_sr = (1 / T) * (1 + 0.5 * sr_daily**2 - skew * sr_daily + (kurtosis / 4) * sr_daily**2)
        
        if var_sr <= 0:
            return 0.0
        
        # Deflated Sharpe: Φ((SR - SR_0) / sqrt(Var[SR]))
        dsr = stats.norm.cdf((sr_daily - sr_0) / np.sqrt(var_sr))
        
        return dsr
    
    def sortino_ratio(self, returns: pd.Series, target: float = 0) -> float:
        """
        Compute Sortino ratio.
        
        Parameters
        ----------
        returns : pd.Series
            Daily returns
        target : float
            Target return (default 0)
            
        Returns
        -------
        float
            Sortino ratio
        """
        ann_return = self.annualized_return(returns)
        
        # Downside deviation
        downside_returns = returns[returns < target]
        if len(downside_returns) == 0:
            return np.inf
        
        downside_std = downside_returns.std() * np.sqrt(252)
        
        if downside_std == 0:
            return 0
        
        return (ann_return - self.risk_free_rate) / downside_std
    
    def max_drawdown(self, returns: pd.Series) -> float:
        """
        Compute maximum drawdown.
        
        Parameters
        ----------
        returns : pd.Series
            Daily returns
            
        Returns
        -------
        float
            Maximum drawdown (positive number, between 0 and 1)
        """
        if len(returns) == 0:
            return 0.0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = abs(drawdown.min())
        
        # Sanity check: drawdown should not exceed 100% for long-only
        if max_dd > 1.0:
            print(f"ERROR: Max drawdown {max_dd:.2%} exceeds 100% - data quality issue!")
        
        return min(max_dd, 1.0)  # Cap at 100% as mathematical limit
    
    def calmar_ratio(self, returns: pd.Series) -> float:
        """
        Compute Calmar ratio (return / max drawdown).
        
        Parameters
        ----------
        returns : pd.Series
            Daily returns
            
        Returns
        -------
        float
            Calmar ratio
        """
        ann_return = self.annualized_return(returns)
        max_dd = self.max_drawdown(returns)
        
        if max_dd == 0:
            return 0
        
        return ann_return / max_dd
    
    def cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """
        Compute Conditional Value at Risk (CVaR / Expected Shortfall).
        
        Parameters
        ----------
        returns : pd.Series
            Daily returns
        confidence : float
            Confidence level
            
        Returns
        -------
        float
            CVaR (positive number for loss)
        """
        var = returns.quantile(1 - confidence)
        cvar = returns[returns <= var].mean()
        return abs(cvar)
    
    def tracking_error(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> float:
        """
        Compute tracking error vs benchmark.
        
        Parameters
        ----------
        returns : pd.Series
            Portfolio returns
        benchmark_returns : pd.Series
            Benchmark returns
            
        Returns
        -------
        float
            Annualized tracking error
        """
        # Align dates
        aligned_ret, aligned_bench = returns.align(benchmark_returns, join='inner')
        
        active_returns = aligned_ret - aligned_bench
        return active_returns.std() * np.sqrt(252)
    
    def information_ratio(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> float:
        """
        Compute information ratio vs benchmark.
        
        Parameters
        ----------
        returns : pd.Series
            Portfolio returns
        benchmark_returns : pd.Series
            Benchmark returns
            
        Returns
        -------
        float
            Information ratio
        """
        # Align dates
        aligned_ret, aligned_bench = returns.align(benchmark_returns, join='inner')
        
        active_returns = aligned_ret - aligned_bench
        
        ann_active_return = self.annualized_return(active_returns)
        te = self.tracking_error(returns, benchmark_returns)
        
        if te == 0:
            return 0
        
        return ann_active_return / te
    
    def compute_all_metrics(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        turnover: Optional[pd.Series] = None,
        n_trials: int = 1
    ) -> Dict[str, float]:
        """
        Compute all performance metrics.
        
        Parameters
        ----------
        returns : pd.Series
            Portfolio returns
        benchmark_returns : Optional[pd.Series]
            Benchmark returns
        turnover : Optional[pd.Series]
            Turnover series
        n_trials : int
            Number of strategies tested (for Deflated Sharpe Ratio)
            
        Returns
        -------
        Dict[str, float]
            Dictionary of all metrics
        """
        metrics = {
            'Annualized Return': self.annualized_return(returns),
            'Annualized Volatility': self.annualized_volatility(returns),
            'Sharpe Ratio': self.sharpe_ratio(returns),
            'Deflated Sharpe Ratio': self.deflated_sharpe_ratio(returns, n_trials=n_trials),
            'Sortino Ratio': self.sortino_ratio(returns),
            'Max Drawdown': self.max_drawdown(returns),
            'Calmar Ratio': self.calmar_ratio(returns),
            'CVaR 95%': self.cvar(returns, 0.95),
            'Skewness': returns.skew(),
            'Kurtosis': returns.kurtosis(),
        }
        
        if turnover is not None and len(turnover) > 0:
            metrics['Avg Turnover'] = turnover.mean()
        
        if benchmark_returns is not None:
            metrics['Tracking Error'] = self.tracking_error(returns, benchmark_returns)
            metrics['Information Ratio'] = self.information_ratio(returns, benchmark_returns)
            
            # Alpha/Beta
            aligned_ret, aligned_bench = returns.align(benchmark_returns, join='inner')
            if len(aligned_ret) > 0 and aligned_bench.std() > 0:
                beta = aligned_ret.cov(aligned_bench) / aligned_bench.var()
                alpha = (
                    self.annualized_return(aligned_ret) -
                    self.risk_free_rate -
                    beta * (self.annualized_return(aligned_bench) - self.risk_free_rate)
                )
                metrics['Beta'] = beta
                metrics['Alpha'] = alpha
        
        return metrics
