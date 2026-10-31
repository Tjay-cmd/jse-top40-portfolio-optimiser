"""
Portfolio optimization methods implementation.
"""

from typing import Optional

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf

from .base import PortfolioOptimizer


class EqualWeightOptimizer(PortfolioOptimizer):
    """Equal weight portfolio (1/N)."""
    
    def optimize(self, returns: pd.DataFrame, **kwargs) -> pd.Series:
        """Return equal weights for all assets."""
        n = len(returns.columns)
        weights = pd.Series(1.0 / n, index=returns.columns)
        return self.apply_constraints(weights)


class MinVarianceOptimizer(PortfolioOptimizer):
    """Minimum variance portfolio."""
    
    def __init__(
        self,
        covariance_method: str = 'sample',
        **kwargs
    ):
        """
        Initialize minimum variance optimizer.
        
        Parameters
        ----------
        covariance_method : str
            'sample' or 'ledoit_wolf'
        """
        super().__init__(**kwargs)
        self.covariance_method = covariance_method
    
    def _estimate_covariance(self, returns: pd.DataFrame) -> np.ndarray:
        """Estimate covariance matrix."""
        if self.covariance_method == 'ledoit_wolf':
            lw = LedoitWolf()
            cov_matrix = lw.fit(returns).covariance_
        else:
            cov_matrix = returns.cov().values
        
        return cov_matrix
    
    def optimize(self, returns: pd.DataFrame, **kwargs) -> pd.Series:
        """Optimize for minimum variance."""
        n = len(returns.columns)
        cov_matrix = self._estimate_covariance(returns)
        
        # Quadratic program: minimize w'Σw subject to constraints
        w = cp.Variable(n)
        risk = cp.quad_form(w, cov_matrix)
        
        constraints = [cp.sum(w) == 1]
        
        if self.long_only:
            constraints.append(w >= 0)
        
        if self.max_position_size is not None:
            constraints.append(w <= self.max_position_size)
        
        prob = cp.Problem(cp.Minimize(risk), constraints)
        
        try:
            prob.solve(solver=cp.OSQP, verbose=False)
        except:
            try:
                prob.solve(solver=cp.SCS, verbose=False)
            except:
                pass
        
        if prob.status not in ['optimal', 'optimal_inaccurate']:
            print(f"Warning: Optimization status: {prob.status}")
            # Fallback to inverse volatility, but apply constraints properly
            vols = returns.std()
            weights_raw = 1.0 / vols
            weights = pd.Series(weights_raw.values / weights_raw.sum(), index=returns.columns)
        else:
            weights = pd.Series(w.value, index=returns.columns)
        
        # Always apply constraints (handles normalization too)
        return self.apply_constraints(weights)


class MeanVarianceOptimizer(PortfolioOptimizer):
    """Mean-variance optimization (Markowitz)."""
    
    def __init__(
        self,
        objective: str = 'max_sharpe',
        covariance_method: str = 'sample',
        return_lookback: int = 252,
        risk_free_rate: float = 0.07,
        **kwargs
    ):
        """
        Initialize mean-variance optimizer.
        
        Parameters
        ----------
        objective : str
            'max_sharpe' or 'max_return'
        covariance_method : str
            'sample' or 'ledoit_wolf'
        return_lookback : int
            Lookback period for expected returns
        risk_free_rate : float
            Annual risk-free rate
        """
        super().__init__(**kwargs)
        self.objective = objective
        self.covariance_method = covariance_method
        self.return_lookback = return_lookback
        self.risk_free_rate = risk_free_rate
    
    def _estimate_covariance(self, returns: pd.DataFrame) -> np.ndarray:
        """Estimate covariance matrix."""
        if self.covariance_method == 'ledoit_wolf':
            lw = LedoitWolf()
            cov_matrix = lw.fit(returns).covariance_
        else:
            cov_matrix = returns.cov().values
        
        return cov_matrix
    
    def _estimate_returns(self, returns: pd.DataFrame) -> np.ndarray:
        """Estimate expected returns."""
        # Use recent returns as proxy for expected returns
        mean_returns = returns.tail(self.return_lookback).mean().values
        return mean_returns * 252  # Annualize
    
    def optimize(self, returns: pd.DataFrame, **kwargs) -> pd.Series:
        """Optimize mean-variance portfolio."""
        n = len(returns.columns)
        cov_matrix = self._estimate_covariance(returns)
        exp_returns = self._estimate_returns(returns)
        
        w = cp.Variable(n)
        portfolio_return = exp_returns @ w
        portfolio_risk = cp.quad_form(w, cov_matrix)
        
        constraints = [cp.sum(w) == 1]
        
        if self.long_only:
            constraints.append(w >= 0)
        
        if self.max_position_size is not None:
            constraints.append(w <= self.max_position_size)
        
        if self.objective == 'max_sharpe':
            # Simplified: maximize return subject to constraints
            # This is a reasonable approximation for max Sharpe
            objective = cp.Maximize(portfolio_return - self.risk_free_rate * 252 - 0.5 * portfolio_risk)
            prob = cp.Problem(objective, constraints)
            
            try:
                prob.solve(solver=cp.OSQP, verbose=False)
            except:
                try:
                    prob.solve(solver=cp.SCS, verbose=False)
                except:
                    pass
            
            if prob.status in ['optimal', 'optimal_inaccurate']:
                weights = pd.Series(w.value, index=returns.columns)
            else:
                print(f"Warning: Sharpe optimization failed: {prob.status}")
                # Fallback to mean-vol weighting
                sharpes = (exp_returns - self.risk_free_rate) / returns.std()
                sharpes = sharpes.clip(lower=0)
                if sharpes.sum() > 0:
                    weights = pd.Series(sharpes.values / sharpes.sum(), index=returns.columns)
                else:
                    weights = pd.Series(1.0 / n, index=returns.columns)
        else:
            # Simple max return with risk constraint
            prob = cp.Problem(cp.Maximize(portfolio_return), constraints)
            
            try:
                prob.solve(solver=cp.OSQP, verbose=False)
            except:
                try:
                    prob.solve(solver=cp.SCS, verbose=False)
                except:
                    pass
            
            if prob.status in ['optimal', 'optimal_inaccurate']:
                weights = pd.Series(w.value, index=returns.columns)
            else:
                print(f"Warning: Optimization failed: {prob.status}")
                weights = pd.Series(1.0 / n, index=returns.columns)
        
        return self.apply_constraints(weights)


class RiskParityOptimizer(PortfolioOptimizer):
    """Risk parity (Equal Risk Contribution) portfolio."""
    
    def __init__(
        self,
        covariance_method: str = 'sample',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.covariance_method = covariance_method
    
    def _estimate_covariance(self, returns: pd.DataFrame) -> np.ndarray:
        """Estimate covariance matrix."""
        if self.covariance_method == 'ledoit_wolf':
            lw = LedoitWolf()
            cov_matrix = lw.fit(returns).covariance_
        else:
            cov_matrix = returns.cov().values
        
        return cov_matrix
    
    def optimize(self, returns: pd.DataFrame, **kwargs) -> pd.Series:
        """Optimize for equal risk contribution."""
        n = len(returns.columns)
        cov_matrix = self._estimate_covariance(returns)
        
        # Start with inverse volatility weights
        vols = np.sqrt(np.diag(cov_matrix))
        weights_init = 1.0 / vols
        weights_init = weights_init / weights_init.sum()
        
        # Iterative risk parity (simple heuristic)
        w = weights_init.copy()
        for _ in range(100):
            portfolio_vol = np.sqrt(w @ cov_matrix @ w)
            marginal_contrib = cov_matrix @ w / portfolio_vol
            risk_contrib = w * marginal_contrib
            
            # Target equal risk contribution
            target_rc = portfolio_vol / n
            
            # Adjust weights
            w_new = w * (target_rc / risk_contrib)
            w_new = np.maximum(w_new, 0)  # Ensure non-negative
            w_new = w_new / w_new.sum()  # Normalize
            
            # Check convergence
            if np.allclose(w, w_new, atol=1e-6):
                break
            
            w = 0.5 * w + 0.5 * w_new  # Damping for stability
        
        weights = pd.Series(w, index=returns.columns)
        return self.apply_constraints(weights)


class HRPOptimizer(PortfolioOptimizer):
    """Hierarchical Risk Parity portfolio."""
    
    def __init__(
        self,
        covariance_method: str = 'sample',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.covariance_method = covariance_method
    
    def _estimate_covariance(self, returns: pd.DataFrame) -> np.ndarray:
        """Estimate covariance matrix."""
        if self.covariance_method == 'ledoit_wolf':
            lw = LedoitWolf()
            cov_matrix = lw.fit(returns).covariance_
        else:
            cov_matrix = returns.cov().values
        
        return cov_matrix
    
    def _get_quasi_diag(self, link):
        """Get quasi-diagonal matrix from linkage."""
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]
        
        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
            df0 = sort_ix[sort_ix >= num_items]
            i = df0.index
            j = df0.values - num_items
            sort_ix[i] = link[j, 0]
            df0 = pd.Series(link[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df0]).sort_index()
            sort_ix.index = range(sort_ix.shape[0])
        
        return sort_ix.tolist()
    
    def _get_cluster_var(self, cov, cluster_items):
        """Get cluster variance."""
        cov_slice = cov.loc[cluster_items, cluster_items]
        w_ = 1.0 / np.diag(cov_slice.values)
        w_ = w_ / w_.sum()
        return np.dot(w_, np.dot(cov_slice.values, w_))
    
    def _get_rec_bipart(self, cov, sort_ix):
        """Get recursive bisection weights."""
        w = pd.Series(1.0, index=sort_ix)
        cluster_items = [sort_ix]
        
        while len(cluster_items) > 0:
            cluster_items = [
                i[j:k]
                for i in cluster_items
                for j, k in ((0, len(i) // 2), (len(i) // 2, len(i)))
                if len(i) > 1
            ]
            
            for i in range(0, len(cluster_items), 2):
                cluster0 = cluster_items[i]
                cluster1 = cluster_items[i + 1]
                
                var0 = self._get_cluster_var(cov, cluster0)
                var1 = self._get_cluster_var(cov, cluster1)
                
                alpha = 1 - var0 / (var0 + var1)
                
                w.loc[cluster0] *= alpha
                w.loc[cluster1] *= 1 - alpha
        
        return w
    
    def optimize(self, returns: pd.DataFrame, **kwargs) -> pd.Series:
        """Optimize using Hierarchical Risk Parity."""
        cov_matrix = self._estimate_covariance(returns)
        cov = pd.DataFrame(cov_matrix, index=returns.columns, columns=returns.columns)
        
        # Compute distance matrix
        corr = returns.corr()
        dist = np.sqrt((1 - corr) / 2)
        
        # Hierarchical clustering
        link = linkage(squareform(dist.values), method='single')
        
        # Get quasi-diagonal matrix
        sort_ix = self._get_quasi_diag(link)
        sort_ix = [returns.columns[i] for i in sort_ix]
        
        # Get recursive bisection weights
        weights = self._get_rec_bipart(cov, sort_ix)
        
        return self.apply_constraints(weights)


class BlackLittermanOptimizer(PortfolioOptimizer):
    """Black-Litterman portfolio optimization."""
    
    def __init__(
        self,
        tau: float = 0.05,
        risk_aversion: float = 2.5,
        market_cap_weight: bool = True,
        **kwargs
    ):
        """
        Initialize Black-Litterman optimizer.
        
        Parameters
        ----------
        tau : float
            Scaling factor for uncertainty in prior
        risk_aversion : float
            Market risk aversion parameter
        market_cap_weight : bool
            Whether to use market cap weighting for prior
        """
        super().__init__(**kwargs)
        self.tau = tau
        self.risk_aversion = risk_aversion
        self.market_cap_weight = market_cap_weight
    
    def optimize(self, returns: pd.DataFrame, **kwargs) -> pd.Series:
        """Optimize using Black-Litterman model."""
        n = len(returns.columns)
        cov_matrix = returns.cov().values
        
        # Prior: market-implied returns
        if self.market_cap_weight:
            # Use equal weight as proxy for market weights
            w_mkt = np.ones(n) / n
        else:
            w_mkt = np.ones(n) / n
        
        # Implied returns: Π = λΣw_mkt
        implied_returns = self.risk_aversion * cov_matrix @ w_mkt
        
        # In absence of views, posterior = prior
        # With views, we would combine: Σ_post = [(τΣ)^-1 + P'ΩP]^-1
        # μ_post = Σ_post[(τΣ)^-1 Π + P'Ω^-1 Q]
        
        # For now, use implied returns directly (no views)
        exp_returns = implied_returns
        
        # Mean-variance optimization with posterior distribution
        w = cp.Variable(n)
        portfolio_return = exp_returns @ w
        portfolio_risk = cp.quad_form(w, cov_matrix)
        
        constraints = [cp.sum(w) == 1]
        
        if self.long_only:
            constraints.append(w >= 0)
        
        if self.max_position_size is not None:
            constraints.append(w <= self.max_position_size)
        
        # Maximize return - risk_aversion * risk
        objective = cp.Maximize(portfolio_return - self.risk_aversion * portfolio_risk)
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP, verbose=False)
        
        if prob.status in ['optimal', 'optimal_inaccurate']:
            weights = pd.Series(w.value, index=returns.columns)
        else:
            print(f"Warning: Black-Litterman optimization failed: {prob.status}")
            weights = pd.Series(1.0 / n, index=returns.columns)
        
        return self.apply_constraints(weights)
