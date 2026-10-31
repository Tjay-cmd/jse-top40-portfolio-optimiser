"""
Tests for backtesting framework.
"""

import numpy as np
import pandas as pd
import pytest

from jse_portfolio_lab.optimization import EqualWeightOptimizer
from jse_portfolio_lab.backtesting import Backtest


@pytest.fixture
def sample_data():
    """Generate sample returns and prices."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    tickers = ['TICK1', 'TICK2', 'TICK3', 'TICK4', 'TICK5']
    
    # Generate returns
    returns = pd.DataFrame(
        np.random.randn(500, 5) * 0.01,
        index=dates,
        columns=tickers
    )
    
    # Generate prices
    prices = (1 + returns).cumprod() * 100
    
    return prices, returns


def test_backtest_runs(sample_data):
    """Test that backtest runs without errors."""
    prices, returns = sample_data
    
    optimizer = EqualWeightOptimizer()
    backtest = Backtest(
        optimizer=optimizer,
        rebalance_frequency='monthly',
        transaction_cost_bps=30.0,
        initial_capital=1000000.0,
        min_history_days=100
    )
    
    result = backtest.run(returns, prices, verbose=False)
    
    # Check that results are returned
    assert result.portfolio_value is not None
    assert len(result.portfolio_value) > 0
    
    # Check that portfolio value starts at initial capital
    assert np.isclose(
        result.portfolio_value.iloc[0],
        1000000.0,
        rtol=0.01
    )


def test_transaction_costs_reduce_returns(sample_data):
    """Test that transaction costs reduce portfolio returns."""
    prices, returns = sample_data
    
    optimizer = EqualWeightOptimizer()
    
    # Backtest with no costs
    backtest_no_cost = Backtest(
        optimizer=optimizer,
        rebalance_frequency='monthly',
        transaction_cost_bps=0.0,
        initial_capital=1000000.0,
        min_history_days=100
    )
    result_no_cost = backtest_no_cost.run(returns, prices, verbose=False)
    
    # Backtest with costs
    backtest_with_cost = Backtest(
        optimizer=optimizer,
        rebalance_frequency='monthly',
        transaction_cost_bps=50.0,
        initial_capital=1000000.0,
        min_history_days=100
    )
    result_with_cost = backtest_with_cost.run(returns, prices, verbose=False)
    
    # Portfolio with costs should have lower final value
    assert result_with_cost.portfolio_value.iloc[-1] < result_no_cost.portfolio_value.iloc[-1]


def test_turnover_calculation(sample_data):
    """Test turnover calculation."""
    prices, returns = sample_data
    
    optimizer = EqualWeightOptimizer()
    backtest = Backtest(
        optimizer=optimizer,
        rebalance_frequency='monthly',
        transaction_cost_bps=30.0,
        initial_capital=1000000.0,
        min_history_days=100
    )
    
    result = backtest.run(returns, prices, verbose=False)
    
    # Check that turnover is calculated
    assert result.turnover is not None
    assert len(result.turnover) > 0
    
    # Turnover should be between 0 and 2 (0 = no change, 2 = complete turnover)
    assert (result.turnover >= 0).all()
    assert (result.turnover <= 2).all()


def test_weights_sum_to_one(sample_data):
    """Test that weights always sum to one."""
    prices, returns = sample_data
    
    optimizer = EqualWeightOptimizer()
    backtest = Backtest(
        optimizer=optimizer,
        rebalance_frequency='monthly',
        transaction_cost_bps=30.0,
        initial_capital=1000000.0,
        min_history_days=100
    )
    
    result = backtest.run(returns, prices, verbose=False)
    
    # Check that weights sum to 1 at each rebalance
    for date, weights in result.weights_history.iterrows():
        assert np.isclose(weights.sum(), 1.0, atol=1e-6)
