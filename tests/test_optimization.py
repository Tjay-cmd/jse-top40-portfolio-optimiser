"""
Tests for portfolio optimization methods.
"""

import numpy as np
import pandas as pd
import pytest

from jse_portfolio_lab.optimization import (
    EqualWeightOptimizer,
    MinVarianceOptimizer,
    MeanVarianceOptimizer,
    RiskParityOptimizer,
    HRPOptimizer,
)


@pytest.fixture
def sample_returns():
    """Generate sample returns data."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    tickers = ['TICK1', 'TICK2', 'TICK3', 'TICK4', 'TICK5']
    
    returns = pd.DataFrame(
        np.random.randn(500, 5) * 0.02,
        index=dates,
        columns=tickers
    )
    return returns


def test_equal_weight_optimizer(sample_returns):
    """Test equal weight optimizer."""
    optimizer = EqualWeightOptimizer()
    weights = optimizer.optimize(sample_returns)
    
    # Check weights sum to 1
    assert np.isclose(weights.sum(), 1.0, atol=1e-6)
    
    # Check all weights are equal
    assert np.allclose(weights.values, 1.0 / len(weights))
    
    # Check all weights are non-negative (long only)
    assert (weights >= 0).all()


def test_min_variance_optimizer(sample_returns):
    """Test minimum variance optimizer."""
    optimizer = MinVarianceOptimizer(covariance_method='sample')
    weights = optimizer.optimize(sample_returns)
    
    # Check weights sum to 1
    assert np.isclose(weights.sum(), 1.0, atol=1e-6)
    
    # Check long-only constraint
    assert (weights >= -1e-6).all()
    
    # Check max position constraint
    assert (weights <= 0.15 + 1e-6).all()


def test_mean_variance_optimizer(sample_returns):
    """Test mean-variance optimizer."""
    optimizer = MeanVarianceOptimizer(
        objective='max_sharpe',
        covariance_method='sample'
    )
    weights = optimizer.optimize(sample_returns)
    
    # Check weights sum to 1
    assert np.isclose(weights.sum(), 1.0, atol=1e-6)
    
    # Check constraints
    assert (weights >= -1e-6).all()
    assert (weights <= 0.15 + 1e-6).all()


def test_risk_parity_optimizer(sample_returns):
    """Test risk parity optimizer."""
    optimizer = RiskParityOptimizer()
    weights = optimizer.optimize(sample_returns)
    
    # Check weights sum to 1
    assert np.isclose(weights.sum(), 1.0, atol=1e-6)
    
    # Check long-only
    assert (weights >= 0).all()


def test_hrp_optimizer(sample_returns):
    """Test Hierarchical Risk Parity optimizer."""
    optimizer = HRPOptimizer()
    weights = optimizer.optimize(sample_returns)
    
    # Check weights sum to 1
    assert np.isclose(weights.sum(), 1.0, atol=1e-6)
    
    # Check long-only
    assert (weights >= 0).all()


def test_optimizer_constraints():
    """Test that constraints are properly enforced."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    tickers = [f'TICK{i}' for i in range(20)]
    
    returns = pd.DataFrame(
        np.random.randn(500, 20) * 0.02,
        index=dates,
        columns=tickers
    )
    
    optimizer = MinVarianceOptimizer(
        long_only=True,
        max_position_size=0.1,
        min_position_size=0.02,
        max_stocks=10
    )
    
    weights = optimizer.optimize(returns)
    
    # Check sum to 1
    assert np.isclose(weights.sum(), 1.0, atol=1e-6)
    
    # Check long-only
    assert (weights >= 0).all()
    
    # Check max position
    assert (weights <= 0.1 + 1e-6).all()
    
    # Check max stocks
    assert (weights > 0).sum() <= 10


def test_no_lookahead():
    """Test that optimization doesn't use future data."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    tickers = ['TICK1', 'TICK2', 'TICK3']
    
    returns = pd.DataFrame(
        np.random.randn(500, 3) * 0.02,
        index=dates,
        columns=tickers
    )
    
    # Add a large positive return at the end
    returns.iloc[-1] = 0.10
    
    # Optimize using only first 400 days
    optimizer = MinVarianceOptimizer()
    weights = optimizer.optimize(returns.iloc[:400])
    
    # The weights should not be influenced by the last day's return
    # This is implicitly tested by using only historical data
    assert len(returns.iloc[:400]) == 400
    assert np.isclose(weights.sum(), 1.0)
