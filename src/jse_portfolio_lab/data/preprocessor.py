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
        Filter stocks with sufficient trading history.
        
        Parameters
        ----------
        prices : pd.DataFrame
            Raw price data
            
        Returns
        -------
        pd.DataFrame
            Filtered prices
        """
        valid_counts = prices.notna().sum()
        liquid_tickers = valid_counts[valid_counts >= self.min_trading_days].index.tolist()
        
        print(f"Filtered to {len(liquid_tickers)}/{len(prices.columns)} liquid stocks")
        
        return prices[liquid_tickers]
    
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
        Compute log returns from prices.
        
        Parameters
        ----------
        prices : pd.DataFrame
            Price data
            
        Returns
        -------
        pd.DataFrame
            Log returns
        """
        returns = np.log(prices / prices.shift(1))
        return returns.dropna()
    
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
        
        # Handle missing data
        prices = self.handle_missing_data(prices)
        
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
        
        # Compute returns
        returns = self.compute_returns(aligned['prices'])
        
        # Compute factor returns
        factors = {}
        factors['benchmark'] = self.compute_returns(aligned['benchmark'].to_frame()).iloc[:, 0]
        factors['fx'] = self.compute_returns(aligned['fx'].to_frame()).iloc[:, 0]
        
        if aligned['rf_rate'] is not None:
            # Convert annual yield to daily rate
            factors['rf_rate'] = aligned['rf_rate'] / 252 / 100
        
        # Align returns and factors
        common_dates = returns.index.intersection(factors['benchmark'].index)
        returns = returns.loc[common_dates]
        for key in factors:
            if factors[key] is not None:
                factors[key] = factors[key].reindex(common_dates, method='ffill')
        
        print(f"Final dataset: {len(returns)} days, {len(returns.columns)} stocks")
        
        return aligned['prices'].loc[returns.index], returns, factors
