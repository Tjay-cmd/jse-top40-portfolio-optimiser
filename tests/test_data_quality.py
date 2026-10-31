"""
Tests for data quality and cleaning.
"""

import numpy as np
import pandas as pd
import pytest
from jse_portfolio_lab.data.cleaner import JSEDataCleaner


def test_benchmark_volatility_range():
    """Test that STX40 benchmark volatility is in a plausible range (13-22%)."""
    pytest.skip("Integration test - run manually with real data")


def test_unit_jump_detection():
    """Test that unit jump detection and correction works."""
    cleaner = JSEDataCleaner()
    
    # Create a price series with a 100x unit jump
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    prices = pd.Series(100.0, index=dates)
    
    # Introduce a 100x jump at day 50 (cents -> rand)
    prices.iloc[50:] = prices.iloc[50:] / 100
    
    # Clean the series (returns tuple: cleaned_series, issue_count)
    cleaned, issues = cleaner.clean_series(prices, ticker="TEST")
    
    # After cleaning, returns should be small (unit jump corrected)
    returns = cleaned.pct_change().dropna()
    
    # Maximum absolute return should be small (unit jump was corrected)
    max_abs_return = returns.abs().max()
    assert max_abs_return < 0.02, \
        f"Unit jump not corrected: max return {max_abs_return:.2%}"


def test_no_extreme_daily_returns():
    """Test that cleaned data has no extreme single-day returns."""
    cleaner = JSEDataCleaner()
    
    # Create normal price series
    dates = pd.date_range('2020-01-01', periods=252, freq='D')
    np.random.seed(42)
    returns = np.random.randn(252) * 0.02  # 2% daily vol
    prices = pd.Series(100 * (1 + returns).cumprod(), index=dates)
    
    # Clean the series
    cleaned, _ = cleaner.clean_series(prices, ticker="TEST")
    
    # Calculate returns
    clean_returns = cleaned.pct_change().dropna()
    
    # No return should exceed 40% (after cleaning)
    max_abs_return = clean_returns.abs().max()
    assert max_abs_return < 0.40, \
        f"Extreme return found: {max_abs_return:.2%}"
