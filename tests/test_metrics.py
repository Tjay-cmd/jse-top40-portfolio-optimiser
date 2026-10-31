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


def test_deflated_sharpe_ratio():
    """Test Deflated Sharpe Ratio implementation against expected behavior."""
    pm = PerformanceMetrics(risk_free_rate=0.0)  # Zero RF for simplicity
    
    # Create synthetic returns with known properties
    np.random.seed(42)
    # Strategy with positive Sharpe
    good_returns = pd.Series(np.random.randn(252) * 0.01 + 0.0005)  # ~12.6% annual return, ~16% vol
    
    # Strategy with zero mean
    null_returns = pd.Series(np.random.randn(252) * 0.01)
    
    # Test 1: Single strategy should have high DSR if Sharpe is good
    dsr_good_single = pm.deflated_sharpe_ratio(good_returns, n_trials=1)
    assert 0.5 < dsr_good_single <= 1.0, f"DSR for good strategy with 1 trial should be high, got {dsr_good_single}"
    
    # Test 2: Same strategy with many trials should have lower DSR (multiple testing penalty)
    dsr_good_multi = pm.deflated_sharpe_ratio(good_returns, n_trials=100)
    assert dsr_good_multi < dsr_good_single, "DSR should decrease with more trials"
    
    # Test 3: Null strategy (zero Sharpe) should have DSR near 0.5 with few trials
    dsr_null = pm.deflated_sharpe_ratio(null_returns, n_trials=1)
    assert 0.3 < dsr_null < 0.7, f"DSR for null strategy should be near 0.5, got {dsr_null}"
    
    # Test 4: DSR should be between 0 and 1
    dsr_test = pm.deflated_sharpe_ratio(good_returns, n_trials=10)
    assert 0 <= dsr_test <= 1, f"DSR must be in [0, 1], got {dsr_test}"
    
    # Test 5: More trials should reduce DSR (unless Sharpe is very high)
    dsr_10 = pm.deflated_sharpe_ratio(good_returns, n_trials=10)
    dsr_50 = pm.deflated_sharpe_ratio(good_returns, n_trials=50)
    # With modest Sharpe, more trials should penalize more
    assert dsr_50 <= dsr_10 or dsr_50 > 0.9, "More trials should reduce DSR unless strategy is very strong"


def test_annualized_return_trading_day_convention():
    """Annualization is (1+r).prod()**(252/n)-1, not calendar-day or PV ratio."""
    dates = pd.bdate_range("2020-01-01", periods=252)
    returns = pd.Series(0.001, index=dates)
    metrics = PerformanceMetrics()

    expected = (1.001 ** 252) - 1
    assert np.isclose(metrics.annualized_return(returns), expected, rtol=1e-12)

    # A shorter window must use n=len(returns), not 365.25 calendar days
    short = returns.iloc[:126]
    expected_short = (1.001 ** 126) ** (252 / 126) - 1
    assert np.isclose(metrics.annualized_return(short), expected_short, rtol=1e-12)
