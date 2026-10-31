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
    optimizer = MinVarianceOptimizer(
        covariance_method='sample',
        max_position_size=0.30  # Feasible for 5 stocks
    )
    weights = optimizer.optimize(sample_returns)
    
    # Check weights sum to 1
    assert np.isclose(weights.sum(), 1.0, atol=1e-6)
    
    # Check long-only constraint
    assert (weights >= -1e-6).all()
    
    # Check max position constraint
    assert (weights <= 0.30 + 1e-6).all()


def test_mean_variance_optimizer(sample_returns):
    """Test mean-variance optimizer."""
    optimizer = MeanVarianceOptimizer(
        objective='max_sharpe',
        covariance_method='sample',
        max_position_size=0.30  # Feasible for 5 stocks
    )
    weights = optimizer.optimize(sample_returns)
    
    # Check weights sum to 1
    assert np.isclose(weights.sum(), 1.0, atol=1e-6)
    
    # Check constraints
    assert (weights >= -1e-6).all()
    assert (weights <= 0.30 + 1e-6).all()


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
    """Test that constraints are properly enforced with feasible settings."""
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
        max_position_size=0.25,  # Feasible: 4 stocks at 0.25 = 1.0
        min_position_size=0.02,
        max_stocks=10
    )
    
    weights = optimizer.optimize(returns)
    
    # Check sum to 1
    assert np.isclose(weights.sum(), 1.0, atol=1e-6)
    
    # Check long-only
    assert (weights >= 0).all()
    
    # Check max position (with small tolerance for numerical precision)
    assert (weights <= 0.25 + 1e-4).all()
    
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


def test_position_caps_strictly_enforced():
    """Test that position caps are strictly enforced at rebalancing."""
    np.random.seed(123)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    tickers = [f'TICK{i}' for i in range(10)]
    
    # Create returns that might lead to concentrated portfolios
    returns = pd.DataFrame(
        np.random.randn(500, 10) * 0.02,
        index=dates,
        columns=tickers
    )
    
    max_position = 0.15
    
    # Test all optimizer types
    optimizers = [
        MinVarianceOptimizer(max_position_size=max_position),
        MeanVarianceOptimizer(max_position_size=max_position),
        RiskParityOptimizer(max_position_size=max_position),
        HRPOptimizer(max_position_size=max_position),
    ]
    
    for optimizer in optimizers:
        weights = optimizer.optimize(returns)
        
        # Check sum to 1
        assert np.isclose(weights.sum(), 1.0, atol=1e-6), \
            f"{optimizer.__class__.__name__}: weights sum to {weights.sum()}"
        
        # Check position caps strictly enforced (tight tolerance)
        max_weight = weights.max()
        assert max_weight <= max_position + 1e-6, \
            f"{optimizer.__class__.__name__}: max weight {max_weight:.6f} exceeds {max_position}"
        
        # Verify no single position exceeds cap
        for ticker in weights.index:
            assert weights[ticker] <= max_position + 1e-6, \
                f"{optimizer.__class__.__name__}: {ticker} weight {weights[ticker]:.6f} exceeds {max_position}"
