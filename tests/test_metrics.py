"""
Tests for performance metrics.
"""

import numpy as np
import pandas as pd
import pytest

from jse_portfolio_lab.metrics import PerformanceMetrics


@pytest.fixture
def sample_returns():
    """Generate sample returns."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=252, freq='D')
    returns = pd.Series(np.random.randn(252) * 0.01, index=dates)
    return returns


def test_annualized_return(sample_returns):
    """Test annualized return calculation."""
    metrics = PerformanceMetrics()
    ann_return = metrics.annualized_return(sample_returns)
    
    # Should be a reasonable number
    assert -1 < ann_return < 1


def test_annualized_volatility(sample_returns):
    """Test annualized volatility calculation."""
    metrics = PerformanceMetrics()
    ann_vol = metrics.annualized_volatility(sample_returns)
    
    # Should be positive
    assert ann_vol > 0
    
    # Should be roughly sqrt(252) * daily vol
    daily_vol = sample_returns.std()
    expected_vol = daily_vol * np.sqrt(252)
    assert np.isclose(ann_vol, expected_vol)


def test_sharpe_ratio(sample_returns):
    """Test Sharpe ratio calculation."""
    metrics = PerformanceMetrics(risk_free_rate=0.0)
    sharpe = metrics.sharpe_ratio(sample_returns)
    
    # Should be finite
    assert np.isfinite(sharpe)


def test_max_drawdown():
    """Test maximum drawdown with known values."""
    # Create a simple return series with known drawdown
    returns = pd.Series([0.1, 0.05, -0.2, -0.1, 0.15, 0.05])
    
    metrics = PerformanceMetrics()
    max_dd = metrics.max_drawdown(returns)
    
    # Should be positive
    assert max_dd > 0
    
    # Should be less than 1
    assert max_dd < 1


def test_cvar():
    """Test CVaR calculation."""
    np.random.seed(42)
    returns = pd.Series(np.random.randn(1000) * 0.01)
    
    metrics = PerformanceMetrics()
    cvar = metrics.cvar(returns, confidence=0.95)
    
    # Should be positive (we report as positive loss)
    assert cvar > 0


def test_tracking_error():
    """Test tracking error calculation."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=252, freq='D')
    
    portfolio_returns = pd.Series(np.random.randn(252) * 0.01, index=dates)
    benchmark_returns = pd.Series(np.random.randn(252) * 0.01, index=dates)
    
    metrics = PerformanceMetrics()
    te = metrics.tracking_error(portfolio_returns, benchmark_returns)
    
    # Should be positive
    assert te > 0


def test_information_ratio():
    """Test information ratio calculation."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=252, freq='D')
    
    # Portfolio that outperforms benchmark
    benchmark_returns = pd.Series(np.random.randn(252) * 0.01, index=dates)
    portfolio_returns = benchmark_returns + 0.001  # Small outperformance
    
    metrics = PerformanceMetrics()
    ir = metrics.information_ratio(portfolio_returns, benchmark_returns)
    
    # Should be positive (portfolio outperforms)
    assert ir > 0


def test_compute_all_metrics(sample_returns):
    """Test that all metrics can be computed."""
    np.random.seed(42)
    benchmark_returns = pd.Series(
        np.random.randn(len(sample_returns)) * 0.01,
        index=sample_returns.index
    )
    
    metrics = PerformanceMetrics()
    all_metrics = metrics.compute_all_metrics(
        sample_returns,
        benchmark_returns=benchmark_returns
    )
    
    # Check that all expected metrics are present
    expected_keys = [
        'Annualized Return',
        'Annualized Volatility',
        'Sharpe Ratio',
        'Sortino Ratio',
        'Max Drawdown',
        'CVaR 95%',
        'Tracking Error',
        'Information Ratio',
    ]
    
    for key in expected_keys:
        assert key in all_metrics
        assert np.isfinite(all_metrics[key])
