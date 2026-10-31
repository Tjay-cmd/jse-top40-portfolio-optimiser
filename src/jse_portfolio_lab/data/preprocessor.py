"""
Data preprocessing and cleaning for portfolio optimization.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class DataPreprocessor:
    """Clean and preprocess market data for portfolio optimization."""
    
    def __init__(
        self,
        min_trading_days: int = 500,
        price_scale: float = 100.0
    ):
        """
        Initialize preprocessor.
        
        Parameters
        ----------
        min_trading_days : int
            Minimum required trading days for inclusion
        price_scale : float
            Scale factor for prices (JSE prices in cents, use 100)
        """
        self.min_trading_days = min_trading_days
        self.price_scale = price_scale
    
    def filter_liquid_stocks(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Keep all stocks with any data (unbalanced panel support).
        Individual rebalances will check for sufficient history.
        
        Parameters
        ----------
        prices : pd.DataFrame
            Raw price data
            
        Returns
        -------
        pd.DataFrame
            Prices (may have NaNs for late-listed stocks)
        """
        # Don't filter here - allow unbalanced panel
        # Each stock will be included once it has min_trading_days of history
        print(f"Keeping all {len(prices.columns)} stocks (unbalanced panel)")
        
        return prices
    
    def handle_missing_data(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing data using forward fill then drop remaining NaNs.
        
        Parameters
        ----------
        prices : pd.DataFrame
            Price data with potential missing values
            
        Returns
        -------
        pd.DataFrame
            Cleaned prices
        """
        # Forward fill missing prices (stocks not trading on certain days)
        prices_filled = prices.ffill()
        
        # Count how many NaNs remain
        remaining_nans = prices_filled.isna().sum().sum()
        if remaining_nans > 0:
            print(f"Warning: {remaining_nans} NaNs remaining after forward fill")
            # Drop rows with any remaining NaNs
            prices_filled = prices_filled.dropna()
        
        return prices_filled
    
    def scale_prices(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Scale prices (JSE prices are in cents).
        
        Parameters
        ----------
        prices : pd.DataFrame
            Raw prices in cents
            
        Returns
        -------
        pd.DataFrame
            Prices in rands
        """
        return prices / self.price_scale
    
    def compute_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute simple returns from prices (preserves NaNs for unbalanced panel).
        
        Parameters
        ----------
        prices : pd.DataFrame
            Price data
            
        Returns
        -------
        pd.DataFrame
            Simple returns (may contain NaNs where stocks haven't listed yet)
        """
        returns = prices.pct_change()
        
        # Don't dropna() - preserve NaNs for unbalanced panel
        # Each stock's returns start when that stock has data
        
        # Sanity check: after cleaning, no single return should exceed ±40%
        extreme = (returns.abs() > 0.40).sum().sum()
        if extreme > 0:
            print(f"WARNING: {extreme} returns still > 40% after cleaning - data quality issue!")
        
        return returns
    
    def align_data(
        self,
        prices: pd.DataFrame,
        benchmark: pd.Series,
        fx: pd.Series,
        rf_rate: pd.Series = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Align all data to common dates.
        
        Parameters
        ----------
        prices : pd.DataFrame
            Stock prices
        benchmark : pd.Series
            Benchmark prices
        fx : pd.Series
            FX rates
        rf_rate : pd.Series
            Risk-free rates
            
        Returns
        -------
        Dict[str, pd.DataFrame]
            Aligned data
        """
        # Get common dates
        common_dates = prices.index
        
        # Align benchmark
        benchmark_aligned = benchmark.reindex(common_dates, method='ffill')
        
        # Align FX
        fx_aligned = fx.reindex(common_dates, method='ffill')
        
        # Align risk-free rate if provided
        if rf_rate is not None and len(rf_rate) > 0:
            rf_rate_aligned = rf_rate.reindex(common_dates, method='ffill')
        else:
            rf_rate_aligned = None
        
        return {
            'prices': prices,
            'benchmark': benchmark_aligned,
            'fx': fx_aligned,
            'rf_rate': rf_rate_aligned
        }
    
    def preprocess_all(
        self,
        raw_data: Dict[str, pd.DataFrame]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.Series]]:
        """
        Complete preprocessing pipeline.
        
        Parameters
        ----------
        raw_data : Dict[str, pd.DataFrame]
            Raw data from DataLoader
            
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.Series]]
            (prices, returns, factors) where factors contains benchmark, fx, rf_rate
        """
        # Filter liquid stocks
        prices = self.filter_liquid_stocks(raw_data['prices'])
        
        # No additional filtering - preserve NaNs for unbalanced panel
        
        # Scale prices (JSE prices in cents)
        # Note: Some tickers may already be in correct scale, but dividing by 100
        # is standard for JSE
        # Actually, yfinance may already provide adjusted prices in rands
        # Let's check the magnitude and only scale if needed
        median_price = prices.median().median()
        if median_price > 100:
            print(f"Scaling prices (median={median_price:.0f})")
            prices = self.scale_prices(prices)
        else:
            print(f"Prices already scaled (median={median_price:.2f})")
        
        # Align all data
        aligned = self.align_data(
            prices,
            raw_data['benchmark'],
            raw_data['fx'],
            raw_data.get('rf_rate')
        )
        
        # Compute returns (preserves NaNs for unbalanced panel - each stock starts when it has data)
        returns = self.compute_returns(aligned['prices'])
        
        # Remove first row (all NaN from pct_change)
        returns = returns.iloc[1:]
        prices_trimmed = aligned['prices'].iloc[1:]
        
        # Compute factor returns  
        benchmark_returns = aligned['benchmark'].pct_change().iloc[1:]
        fx_returns = aligned['fx'].pct_change().iloc[1:] if aligned['fx'] is not None else None
        
        # Build factors dict
        factors = {
            'benchmark': benchmark_returns,
            'fx': fx_returns,
            'risk_free_rate': aligned['rf_rate'].iloc[1:] if aligned['rf_rate'] is not None else None,
        }
        
        # Compute factor returns
        factors = {}
        factors['benchmark'] = self.compute_returns(aligned['benchmark'].to_frame()).iloc[:, 0]
        factors['fx'] = self.compute_returns(aligned['fx'].to_frame()).iloc[:, 0]
        
        if aligned['rf_rate'] is not None:
            # Convert annual yield to daily rate
            factors['rf_rate'] = aligned['rf_rate'] / 252 / 100
        
        # Report stock availability over time
        available_per_date = returns.notna().sum(axis=1)
        print(f"Final dataset: {len(returns)} days, {len(returns.columns)} stocks total")
        print(f"  Stock availability: {available_per_date.min()}-{available_per_date.max()} per day (avg {available_per_date.mean():.1f})")
        print(f"  Date range: {returns.index.min().date()} to {returns.index.max().date()}")
        
        return prices_trimmed, returns, factors
