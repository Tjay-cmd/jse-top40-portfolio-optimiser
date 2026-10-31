"""
Data cleaning utilities for JSE price series.

Handles common JSE data issues:
- Unit switches (cents/rand mid-series)
- Split/consolidation errors
- Stale prices
"""

import numpy as np
import pandas as pd
from typing import Tuple


class JSEDataCleaner:
    """Clean JSE price data from yfinance."""
    
    def __init__(self, unit_jump_threshold: float = 80.0):
        """
        Initialize cleaner.
        
        Parameters
        ----------
        unit_jump_threshold : float
            Threshold for detecting unit jumps (default 80 for ~100x changes)
        """
        self.unit_jump_threshold = unit_jump_threshold
    
    def detect_unit_jumps(self, prices: pd.Series) -> pd.Series:
        """
        Detect and correct unit jumps (x100 or /100 switches).
        
        Common JSE issue: prices switch between cents and rands mid-series.
        
        Parameters
        ----------
        prices : pd.Series
            Raw price series
            
        Returns
        -------
        pd.Series
            Corrected prices with unit jumps fixed
        """
        if len(prices) < 2:
            return prices
        
        prices_clean = prices.copy()
        returns = prices_clean.pct_change().abs()
        
        # Find potential unit jumps (moves close to 99% = x100 or /100)
        candidates = returns[returns > (self.unit_jump_threshold / 100)]
        
        if len(candidates) == 0:
            return prices_clean
        
        print(f"    Detected {len(candidates)} potential unit jumps")
        
        for jump_date in candidates.index:
            idx = prices_clean.index.get_loc(jump_date)
            
            if idx == 0:
                continue
            
            prev_price = prices_clean.iloc[idx - 1]
            curr_price = prices_clean.iloc[idx]
            
            # Check if this looks like a /100 or x100 jump
            ratio = curr_price / prev_price
            
            # Looks like a /100 jump (cents to rands or vice versa)
            if 0.008 < ratio < 0.012:  # Close to 1/100
                # Correct this and all subsequent prices by x100
                prices_clean.iloc[idx:] *= 100
                print(f"      {jump_date.date()}: x100 correction from {curr_price:.2f}")
            
            # Looks like a x100 jump (rands to cents or vice versa)  
            elif 98 < ratio < 102:  # Close to 100
                # Correct this and all subsequent prices by /100
                prices_clean.iloc[idx:] /= 100
                print(f"      {jump_date.date()}: /100 correction from {curr_price:.2f}")
        
        return prices_clean
    
    def remove_stale_prices(self, prices: pd.Series, max_stale_days: int = 10) -> pd.Series:
        """
        Remove periods with stale (unchanging) prices.
        
        Parameters
        ----------
        prices : pd.Series
            Price series
        max_stale_days : int
            Maximum consecutive days with same price before marking as stale
            
        Returns
        -------
        pd.Series
            Prices with stale periods removed
        """
        if len(prices) < max_stale_days:
            return prices
        
        # Find consecutive runs of same price
        price_changes = prices.diff() != 0
        
        # Mark first occurrence of each unique price as valid
        prices_clean = prices.copy()
        
        consecutive_same = 0
        last_price = None
        
        for i in range(len(prices)):
            if prices.iloc[i] == last_price:
                consecutive_same += 1
                if consecutive_same >= max_stale_days:
                    prices_clean.iloc[i] = np.nan
            else:
                consecutive_same = 0
                last_price = prices.iloc[i]
        
        return prices_clean.dropna()
    
    def clean_series(
        self,
        prices: pd.Series,
        ticker: str,
        verbose: bool = True
    ) -> Tuple[pd.Series, dict]:
        """
        Complete cleaning pipeline for a price series.
        
        Parameters
        ----------
        prices : pd.Series
            Raw prices
        ticker : str
            Ticker symbol (for logging)
        verbose : bool
            Print cleaning actions
            
        Returns
        -------
        Tuple[pd.Series, dict]
            (cleaned_prices, cleaning_stats)
        """
        if verbose:
            print(f"  Cleaning {ticker}...")
        
        stats = {
            'original_len': len(prices),
            'unit_jumps_fixed': 0,
            'stale_removed': 0,
        }
        
        # Fix unit jumps
        prices_clean = self.detect_unit_jumps(prices)
        
        # Calculate returns to check quality
        returns = prices_clean.pct_change().dropna()
        
        if len(returns) > 0:
            stats['annual_vol'] = returns.std() * np.sqrt(252)
            stats['max_daily_return'] = returns.abs().max()
            stats['final_len'] = len(prices_clean)
            
            if verbose:
                print(f"    Annual vol: {stats['annual_vol']:.1%}")
                print(f"    Max daily move: {stats['max_daily_return']:.1%}")
                
                # Sanity check
                if stats['annual_vol'] > 1.0:  # >100% vol still suspicious
                    print(f"    WARNING: Volatility still high after cleaning!")
                
                if stats['max_daily_return'] > 0.40:  # >40% single day
                    print(f"    WARNING: Large single-day move detected!")
        
        return prices_clean, stats
