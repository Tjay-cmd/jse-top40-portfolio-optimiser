"""
Visualization utilities for portfolio analysis.
"""

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class Visualizer:
    """Create charts and plots for portfolio analysis."""
    
    def __init__(
        self,
        output_dir: str = "reports",
        plot_format: str = "png",
        plot_dpi: int = 150
    ):
        """
        Initialize visualizer.
        
        Parameters
        ----------
        output_dir : str
            Directory to save plots
        plot_format : str
            Plot file format
        plot_dpi : int
            Plot resolution
        """
        self.output_dir = output_dir
        self.plot_format = plot_format
        self.plot_dpi = plot_dpi
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
    
    def plot_equity_curves(
        self,
        results: Dict[str, pd.Series],
        benchmark: Optional[pd.Series] = None,
        filename: str = "equity_curves.png"
    ):
        """
        Plot equity curves for multiple strategies.
        
        Parameters
        ----------
        results : Dict[str, pd.Series]
            Dictionary of portfolio values
        benchmark : Optional[pd.Series]
            Benchmark portfolio values
        filename : str
            Output filename
        """
        fig, ax = plt.subplots(figsize=(14, 7))
        
        for name, values in results.items():
            # Normalize to start at 1
            normalized = values / values.iloc[0]
            ax.plot(normalized.index, normalized.values, label=name, linewidth=2)
        
        if benchmark is not None:
            normalized_bench = benchmark / benchmark.iloc[0]
            ax.plot(
                normalized_bench.index,
                normalized_bench.values,
                label='Benchmark (STX40)',
                linewidth=2,
                linestyle='--',
                color='black',
                alpha=0.7
            )
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Cumulative Return', fontsize=12)
        ax.set_title('Portfolio Equity Curves', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=self.plot_dpi)
        plt.close()
    
    def plot_drawdowns(
        self,
        results: Dict[str, pd.Series],
        filename: str = "drawdowns.png"
    ):
        """
        Plot drawdown series.
        
        Parameters
        ----------
        results : Dict[str, pd.Series]
            Dictionary of portfolio returns
        filename : str
            Output filename
        """
        fig, ax = plt.subplots(figsize=(14, 7))
        
        for name, returns in results.items():
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            
            ax.fill_between(
                drawdown.index,
                0,
                drawdown.values * 100,
                label=name,
                alpha=0.6
            )
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.set_title('Portfolio Drawdowns', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=self.plot_dpi)
        plt.close()
    
    def plot_weights_heatmap(
        self,
        weights_history: pd.DataFrame,
        method_name: str,
        filename: Optional[str] = None
    ):
        """
        Plot weight evolution heatmap.
        
        Parameters
        ----------
        weights_history : pd.DataFrame
            Historical weights (dates × tickers)
        method_name : str
            Name of the method
        filename : Optional[str]
            Output filename
        """
        if filename is None:
            filename = f"weights_{method_name}.png"
        
        # Select top N holdings by average weight
        avg_weights = weights_history.mean()
        # Ensure numeric dtype
        avg_weights = pd.to_numeric(avg_weights, errors='coerce').fillna(0)
        top_holdings = avg_weights.nlargest(min(15, len(avg_weights))).index
        
        weights_subset = weights_history[top_holdings].T
        
        # Ensure all values are numeric
        weights_subset = weights_subset.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        sns.heatmap(
            weights_subset * 100,
            cmap='RdYlGn',
            center=0,
            cbar_kws={'label': 'Weight (%)'},
            ax=ax,
            linewidths=0.5,
            linecolor='white'
        )
        
        ax.set_title(
            f'Portfolio Weights Over Time: {method_name}',
            fontsize=14,
            fontweight='bold'
        )
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Ticker', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=self.plot_dpi)
        plt.close()
    
    def plot_risk_return_scatter(
        self,
        metrics_df: pd.DataFrame,
        filename: str = "risk_return_scatter.png"
    ):
        """
        Plot risk-return scatter.
        
        Parameters
        ----------
        metrics_df : pd.DataFrame
            DataFrame with metrics (should have 'Annualized Return' and 'Annualized Volatility')
        filename : str
            Output filename
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        returns = metrics_df['Annualized Return'] * 100
        vols = metrics_df['Annualized Volatility'] * 100
        
        # Scatter plot
        ax.scatter(vols, returns, s=200, alpha=0.6, edgecolors='black', linewidth=2)
        
        # Annotate points
        for idx, (vol, ret) in enumerate(zip(vols, returns)):
            ax.annotate(
                metrics_df.index[idx],
                (vol, ret),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=10,
                fontweight='bold'
            )
        
        ax.set_xlabel('Annualized Volatility (%)', fontsize=12)
        ax.set_ylabel('Annualized Return (%)', fontsize=12)
        ax.set_title('Risk-Return Profile', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=self.plot_dpi)
        plt.close()
    
    def plot_cost_sensitivity(
        self,
        sensitivity_results: Dict[int, Dict[str, float]],
        metric: str = 'Sharpe Ratio',
        filename: str = "cost_sensitivity.png"
    ):
        """
        Plot sensitivity to transaction costs.
        
        Parameters
        ----------
        sensitivity_results : Dict[int, Dict[str, float]]
            Results for different cost levels
        metric : str
            Metric to plot
        filename : str
            Output filename
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Organize data
        cost_levels = sorted(sensitivity_results.keys())
        methods = list(next(iter(sensitivity_results.values())).keys())
        
        for method in methods:
            values = [sensitivity_results[cost][method] for cost in cost_levels]
            ax.plot(cost_levels, values, marker='o', label=method, linewidth=2)
        
        ax.set_xlabel('Transaction Cost (bps)', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f'Sensitivity to Transaction Costs: {metric}', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=self.plot_dpi)
        plt.close()
    
    def plot_rolling_sharpe(
        self,
        results: Dict[str, pd.Series],
        window: int = 252,
        filename: str = "rolling_sharpe.png"
    ):
        """
        Plot rolling Sharpe ratios.
        
        Parameters
        ----------
        results : Dict[str, pd.Series]
            Dictionary of portfolio returns
        window : int
            Rolling window size
        filename : str
            Output filename
        """
        fig, ax = plt.subplots(figsize=(14, 7))
        
        for name, returns in results.items():
            rolling_sharpe = (
                returns.rolling(window).mean() / returns.rolling(window).std()
            ) * np.sqrt(252)
            
            ax.plot(rolling_sharpe.index, rolling_sharpe.values, label=name, linewidth=2)
        
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Rolling Sharpe Ratio', fontsize=12)
        ax.set_title(f'{window}-Day Rolling Sharpe Ratio', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=self.plot_dpi)
        plt.close()
