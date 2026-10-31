"""
Data loading and caching for JSE equities and factors.
"""

import os
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
from tqdm import tqdm


class DataLoader:
    """Load and cache market data from yfinance and FRED."""
    
    def __init__(self, cache_dir: str = "data/cache"):
        """
        Initialize data loader.
        
        Parameters
        ----------
        cache_dir : str
            Directory to cache downloaded data
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def _get_cache_path(self, key: str) -> str:
        """Get cache file path for a given key."""
        return os.path.join(self.cache_dir, f"{key}.pkl")
    
    def _load_from_cache(self, key: str, max_age_days: int = 1) -> Optional[pd.DataFrame]:
        """Load data from cache if fresh enough."""
        cache_path = self._get_cache_path(key)
        
        if not os.path.exists(cache_path):
            return None
        
        # Check cache age
        cache_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
        if (datetime.now() - cache_time).days > max_age_days:
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Warning: Failed to load cache for {key}: {e}")
            return None
    
    def _save_to_cache(self, key: str, data: pd.DataFrame):
        """Save data to cache."""
        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Warning: Failed to save cache for {key}: {e}")
    
    def fetch_prices(
        self,
        tickers: List[str],
        start_date: str,
        end_date: Optional[str] = None,
        use_cache: bool = True,
        min_data_points: int = 500
    ) -> pd.DataFrame:
        """
        Fetch adjusted close prices for a list of tickers.
        
        Parameters
        ----------
        tickers : List[str]
            List of ticker symbols (e.g., ['NPN.JO', 'SBK.JO'])
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : Optional[str]
            End date in 'YYYY-MM-DD' format, None for today
        use_cache : bool
            Whether to use cached data
            
        Returns
        -------
        pd.DataFrame
            DataFrame with dates as index and tickers as columns
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        cache_key = f"prices_{start_date}_{end_date}_{'_'.join(sorted(tickers))}"
        
        if use_cache:
            cached = self._load_from_cache(cache_key)
            if cached is not None:
                print(f"Loaded {len(tickers)} tickers from cache")
                return cached
        
        print(f"Fetching {len(tickers)} tickers from yfinance...")
        prices_dict = {}
        
        for ticker in tqdm(tickers, desc="Downloading prices"):
            try:
                data = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    auto_adjust=True
                )
                
                # Skip if empty
                if data is None or (hasattr(data, 'empty') and data.empty):
                    print(f"Warning: No data for {ticker}")
                    continue
                
                # Extract close prices based on data type
                close_data = None
                if isinstance(data, pd.DataFrame):
                    if 'Close' in data.columns:
                        close_data = data['Close']
                        # If Close is still a DataFrame, get its first column
                        if isinstance(close_data, pd.DataFrame):
                            close_data = close_data.iloc[:, 0]
                    elif len(data.columns) > 0:
                        close_data = data.iloc[:, 0]
                elif isinstance(data, pd.Series):
                    close_data = data
                
                # Validate the data
                if close_data is not None:
                    # Ensure it's a series
                    if isinstance(close_data, pd.Series):
                        if len(close_data) >= min_data_points:
                            prices_dict[ticker] = close_data
                        else:
                            print(f"Warning: Insufficient data for {ticker} ({len(close_data)} points, need {min_data_points})")
                    elif isinstance(close_data, (int, float)):
                        print(f"Warning: Scalar data for {ticker}")
                    else:
                        print(f"Warning: Unexpected data type for {ticker}: {type(close_data)}")
                else:
                    print(f"Warning: Could not extract close data for {ticker}")
                    
            except Exception as e:
                print(f"Error fetching {ticker}: {e}")
        
        if not prices_dict:
            raise ValueError("Failed to fetch any price data")
        
        prices = pd.DataFrame(prices_dict)
        prices.index = pd.to_datetime(prices.index)
        
        if use_cache:
            self._save_to_cache(cache_key, prices)
        
        return prices
    
    def fetch_benchmark(
        self,
        benchmark: str,
        start_date: str,
        end_date: Optional[str] = None,
        use_cache: bool = True
    ) -> pd.Series:
        """
        Fetch benchmark index prices.
        
        Parameters
        ----------
        benchmark : str
            Benchmark ticker (e.g., 'STX40.JO')
        start_date : str
            Start date
        end_date : Optional[str]
            End date
        use_cache : bool
            Whether to use cache
            
        Returns
        -------
        pd.Series
            Benchmark prices
        """
        prices = self.fetch_prices([benchmark], start_date, end_date, use_cache)
        return prices[benchmark]
    
    def fetch_fred_data(
        self,
        series_id: str,
        start_date: str,
        end_date: Optional[str] = None,
        use_cache: bool = True
    ) -> pd.Series:
        """
        Fetch data from FRED.
        
        Parameters
        ----------
        series_id : str
            FRED series ID
        start_date : str
            Start date
        end_date : Optional[str]
            End date
        use_cache : bool
            Whether to use cache
            
        Returns
        -------
        pd.Series
            FRED series data
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        cache_key = f"fred_{series_id}_{start_date}_{end_date}"
        
        if use_cache:
            cached = self._load_from_cache(cache_key)
            if cached is not None:
                return cached.iloc[:, 0] if isinstance(cached, pd.DataFrame) else cached
        
        print(f"Fetching FRED series {series_id}...")
        try:
            data = pdr.DataReader(series_id, 'fred', start_date, end_date)
            series = data.iloc[:, 0]
            series.index = pd.to_datetime(series.index)
            
            if use_cache:
                self._save_to_cache(cache_key, pd.DataFrame(series))
            
            return series
        except Exception as e:
            print(f"Warning: Failed to fetch FRED series {series_id}: {e}")
            # Return empty series with proper index
            return pd.Series(dtype=float)
    
    def fetch_all_data(
        self,
        tickers: List[str],
        benchmark: str,
        usd_zar: str,
        start_date: str,
        end_date: Optional[str] = None,
        sa_10y_yield_fred: Optional[str] = None,
        use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch all required data: prices, benchmark, FX, and risk-free rate.
        
        Parameters
        ----------
        tickers : List[str]
            Stock tickers
        benchmark : str
            Benchmark ticker
        usd_zar : str
            USD/ZAR ticker
        start_date : str
            Start date
        end_date : Optional[str]
            End date
        sa_10y_yield_fred : Optional[str]
            FRED series for SA 10y yield
        use_cache : bool
            Whether to use cache
            
        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary with keys: 'prices', 'benchmark', 'fx', 'rf_rate'
        """
        data = {}
        
        # Fetch equity prices
        data['prices'] = self.fetch_prices(tickers, start_date, end_date, use_cache)
        
        # Fetch benchmark
        data['benchmark'] = self.fetch_benchmark(benchmark, start_date, end_date, use_cache)
        
        # Fetch USD/ZAR
        data['fx'] = self.fetch_benchmark(usd_zar, start_date, end_date, use_cache)
        
        # Fetch risk-free rate if available
        if sa_10y_yield_fred:
            data['rf_rate'] = self.fetch_fred_data(
                sa_10y_yield_fred, start_date, end_date, use_cache
            )
        
        return data
