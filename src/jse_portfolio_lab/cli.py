"""
Command-line interface for JSE Portfolio Optimizer.
"""

import os
import pickle
from datetime import datetime
from typing import Dict

import click
import numpy as np
import pandas as pd

from .data import DataLoader, DataPreprocessor
from .optimization import (
    EqualWeightOptimizer,
    MinVarianceOptimizer,
    MeanVarianceOptimizer,
    RiskParityOptimizer,
    HRPOptimizer,
    BlackLittermanOptimizer,
)
from .backtesting import Backtest
from .metrics import PerformanceMetrics
from .utils import Visualizer, load_config


def create_optimizer(method_config: Dict, constraints: Dict):
    """Create optimizer from config."""
    method_type = method_config['type']
    params = method_config.get('params', {})
    
    common_params = {
        'long_only': constraints['long_only'],
        'max_position_size': constraints['max_position_size'],
        'min_position_size': constraints['min_position_size'],
        'max_stocks': constraints['max_stocks'],
    }
    
    if method_type == 'equal_weight':
        return EqualWeightOptimizer(**common_params)
    elif method_type == 'min_variance':
        return MinVarianceOptimizer(
            covariance_method=params.get('covariance', 'sample'),
            **common_params
        )
    elif method_type == 'mean_variance':
        return MeanVarianceOptimizer(
            objective=params.get('objective', 'max_sharpe'),
            covariance_method=params.get('covariance', 'sample'),
            return_lookback=params.get('return_lookback', 252),
            **common_params
        )
    elif method_type == 'risk_parity':
        return RiskParityOptimizer(
            covariance_method=params.get('covariance', 'sample'),
            **common_params
        )
    elif method_type == 'hrp':
        return HRPOptimizer(
            covariance_method=params.get('covariance', 'sample'),
            **common_params
        )
    elif method_type == 'black_litterman':
        return BlackLittermanOptimizer(
            tau=params.get('tau', 0.05),
            risk_aversion=params.get('risk_aversion', 2.5),
            market_cap_weight=params.get('market_cap_weight', True),
            **common_params
        )
    else:
        raise ValueError(f"Unknown method type: {method_type}")


@click.command()
@click.option(
    '--config',
    default='config/config.yaml',
    help='Path to configuration file'
)
@click.option(
    '--use-cache/--no-cache',
    default=True,
    help='Use cached data'
)
@click.option(
    '--verbose/--quiet',
    default=True,
    help='Verbose output'
)
def main(config: str, use_cache: bool, verbose: bool):
    """
    Run JSE Portfolio Optimizer pipeline.
    
    This will:
    1. Load and preprocess data
    2. Run backtests for all methods
    3. Compute metrics
    4. Generate visualizations
    5. Save results
    """
    click.echo("=" * 80)
    click.echo("JSE Portfolio Optimizer Lab")
    click.echo("=" * 80)
    
    # Load configuration
    if verbose:
        click.echo(f"\nLoading configuration from {config}...")
    cfg = load_config(config)
    
    # Initialize components
    data_loader = DataLoader(cache_dir=cfg['data']['cache_dir'])
    preprocessor = DataPreprocessor(
        min_trading_days=cfg['data']['min_trading_days'],
        price_scale=cfg['data']['price_scale']
    )
    
    # Load data
    if verbose:
        click.echo("\n" + "=" * 80)
        click.echo("STEP 1: Loading Data")
        click.echo("=" * 80)
    
    raw_data = data_loader.fetch_all_data(
        tickers=cfg['data']['tickers'],
        benchmark=cfg['data']['benchmark'],
        usd_zar=cfg['data']['usd_zar'],
        start_date=cfg['data']['start_date'],
        end_date=cfg['data']['end_date'],
        sa_10y_yield_fred=cfg['data'].get('sa_10y_yield_fred'),
        use_cache=use_cache
    )
    
    # Preprocess data
    if verbose:
        click.echo("\n" + "=" * 80)
        click.echo("STEP 2: Preprocessing Data")
        click.echo("=" * 80)
    
    prices, returns, factors = preprocessor.preprocess_all(raw_data)
    
    # Get clean benchmark returns from factors
    benchmark_returns = factors['benchmark']
    
    if verbose:
        click.echo(f"\nDataset summary:")
        click.echo(f"  Date range: {returns.index[0].date()} to {returns.index[-1].date()}")
        click.echo(f"  Trading days: {len(returns)}")
        click.echo(f"  Stocks: {len(returns.columns)}")
        click.echo(f"  Tickers: {', '.join(returns.columns[:10].tolist())}...")
    
    # Run backtests
    if verbose:
        click.echo("\n" + "=" * 80)
        click.echo("STEP 3: Running Backtests")
        click.echo("=" * 80)
    
    results = {}
    metrics_list = []
    
    enabled_methods = [m for m in cfg['methods'] if m.get('enabled', True)]
    
    for method_config in enabled_methods:
        method_name = method_config['name']
        
        if verbose:
            click.echo(f"\nBacktesting: {method_name}")
        
        # Create optimizer
        optimizer = create_optimizer(method_config, cfg['constraints'])
        
        # Create backtest
        backtest = Backtest(
            optimizer=optimizer,
            rebalance_frequency=cfg['backtest']['rebalance_frequency'],
            transaction_cost_bps=cfg['backtest']['transaction_cost_bps'],
            initial_capital=cfg['backtest']['initial_capital'],
            min_history_days=cfg['backtest']['min_history_days']
        )
        
        # Run backtest
        result = backtest.run(returns, prices, verbose=False)
        results[method_name] = result
    
    # Compute metrics
    if verbose:
        click.echo("\n" + "=" * 80)
        click.echo("STEP 4: Computing Performance Metrics")
        click.echo("=" * 80)
    
    metrics_calc = PerformanceMetrics(risk_free_rate=cfg['risk_free_rate'])
    
    # Number of strategies tested (for Deflated Sharpe Ratio)
    n_trials = len(results)
    
    # Determine first allocation date across all strategies
    first_dates = [res.first_allocation_date for res in results.values() if res.first_allocation_date is not None]
    if first_dates:
        evaluation_start_date = max(first_dates)  # Use latest first allocation to ensure all strategies are active
        if verbose:
            click.echo(f"\nEvaluation period starts at first allocation: {evaluation_start_date.date()}")
    else:
        evaluation_start_date = returns.index[0]
        if verbose:
            click.echo(f"\nWarning: No allocation dates found, using full period from {evaluation_start_date.date()}")
    
    # Trim returns and benchmark to evaluation period
    benchmark_returns_eval = benchmark_returns[benchmark_returns.index >= evaluation_start_date]
    
    # Compute portfolio metrics
    for method_name, result in results.items():
        # Trim returns to evaluation period
        returns_eval = result.returns[result.returns.index >= evaluation_start_date]
        turnover_eval = result.turnover[result.turnover.index >= evaluation_start_date] if len(result.turnover) > 0 else result.turnover
        
        metrics = metrics_calc.compute_all_metrics(
            returns_eval,
            benchmark_returns=benchmark_returns_eval,
            turnover=turnover_eval,
            n_trials=n_trials
        )
        metrics['Method'] = method_name
        
        # Compute actual cost drag from backtest
        total_costs = result.transaction_costs[result.transaction_costs.index >= evaluation_start_date].sum()
        n_years = len(returns_eval) / 252
        if n_years > 0:
            annual_cost_drag = (total_costs / cfg['backtest']['initial_capital']) / n_years
            metrics['Annual Cost Drag'] = annual_cost_drag
        else:
            metrics['Annual Cost Drag'] = 0.0
        
        metrics_list.append(metrics)
    
    # Add benchmark metrics
    benchmark_metrics = metrics_calc.compute_all_metrics(
        benchmark_returns_eval,
        n_trials=n_trials
    )
    benchmark_metrics['Method'] = 'STX40 (Benchmark)'
    benchmark_metrics['Avg Turnover'] = 0.0
    benchmark_metrics['Annual Cost Drag'] = 0.0
    metrics_list.append(benchmark_metrics)
    
    metrics_df = pd.DataFrame(metrics_list).set_index('Method')
    
    if verbose:
        click.echo("\nPerformance Summary:")
        click.echo(metrics_df.to_string())
    
    # Save results
    results_dir = cfg['reporting']['output_dir']
    os.makedirs(results_dir, exist_ok=True)
    
    # Save metrics table
    metrics_df.to_csv(os.path.join(results_dir, 'metrics.csv'))
    
    # Save benchmark data for reproducibility
    benchmark_df = pd.DataFrame({
        'Date': benchmark_returns.index,
        'STX40_Return': benchmark_returns.values
    })
    benchmark_df.to_csv(os.path.join(results_dir, 'benchmark_stx40.csv'), index=False)
    if verbose:
        click.echo(f"\nSaved benchmark data to {results_dir}/benchmark_stx40.csv")
    
    # Save detailed results
    with open(os.path.join(results_dir, 'backtest_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    # Generate visualizations
    if verbose:
        click.echo("\n" + "=" * 80)
        click.echo("STEP 5: Generating Visualizations")
        click.echo("=" * 80)
    
    viz = Visualizer(
        output_dir=results_dir,
        plot_format=cfg['reporting']['plot_format'],
        plot_dpi=cfg['reporting']['plot_dpi']
    )
    
    # Equity curves (reconstruct benchmark portfolio value from returns)
    portfolio_values = {name: res.portfolio_value for name, res in results.items()}
    # Align benchmark to evaluation period
    benchmark_portfolio_value = (1 + benchmark_returns_eval).cumprod() * cfg['backtest']['initial_capital']
    viz.plot_equity_curves(portfolio_values, benchmark_portfolio_value, "equity_curves.png")
    
    # Drawdowns
    portfolio_returns = {name: res.returns for name, res in results.items()}
    viz.plot_drawdowns(portfolio_returns, "drawdowns.png")
    
    # Risk-return scatter
    viz.plot_risk_return_scatter(metrics_df, "risk_return_scatter.png")
    
    # Weight heatmaps for selected methods
    for method_name in ['equal_weight', 'min_variance', 'hrp']:
        if method_name in results:
            viz.plot_weights_heatmap(
                results[method_name].weights_history,
                method_name,
                f"weights_{method_name}.png"
            )
    
    # Rolling Sharpe
    viz.plot_rolling_sharpe(portfolio_returns, window=252, filename="rolling_sharpe.png")
    
    # Cost sensitivity analysis
    if verbose:
        click.echo("\n" + "=" * 80)
        click.echo("STEP 6: Cost Sensitivity Analysis")
        click.echo("=" * 80)
    
    sensitivity_results = {}
    
    for cost_bps in cfg['sensitivity']['cost_bps_range']:
        if verbose:
            click.echo(f"\nTesting cost: {cost_bps} bps")
        
        sensitivity_results[cost_bps] = {}
        
        # Run for ALL enabled methods, not just first 3
        for method_config in enabled_methods:
            method_name = method_config['name']
            
            optimizer = create_optimizer(method_config, cfg['constraints'])
            backtest = Backtest(
                optimizer=optimizer,
                rebalance_frequency=cfg['backtest']['rebalance_frequency'],
                transaction_cost_bps=cost_bps,
                initial_capital=cfg['backtest']['initial_capital'],
                min_history_days=cfg['backtest']['min_history_days']
            )
            
            result = backtest.run(returns, prices, verbose=False)
            # Use evaluation period for consistency
            returns_eval = result.returns[result.returns.index >= evaluation_start_date]
            sharpe = metrics_calc.sharpe_ratio(returns_eval)
            sensitivity_results[cost_bps][method_name] = sharpe
    
    viz.plot_cost_sensitivity(sensitivity_results, metric='Sharpe Ratio')
    
    # Save sensitivity results
    sensitivity_df = pd.DataFrame(sensitivity_results).T
    sensitivity_df.to_csv(os.path.join(results_dir, 'cost_sensitivity.csv'))
    
    # Final summary
    if verbose:
        click.echo("\n" + "=" * 80)
        click.echo("PIPELINE COMPLETE")
        click.echo("=" * 80)
        click.echo(f"\nResults saved to: {results_dir}/")
        click.echo("\nKey findings:")
        
        # Best Sharpe
        best_sharpe_method = metrics_df['Sharpe Ratio'].idxmax()
        best_sharpe = metrics_df.loc[best_sharpe_method, 'Sharpe Ratio']
        click.echo(f"  Best Sharpe Ratio: {best_sharpe_method} ({best_sharpe:.3f})")
        
        # Best return
        best_return_method = metrics_df['Annualized Return'].idxmax()
        best_return = metrics_df.loc[best_return_method, 'Annualized Return']
        click.echo(f"  Best Return: {best_return_method} ({best_return:.2%})")
        
        # Lowest drawdown
        best_dd_method = metrics_df['Max Drawdown'].idxmin()
        best_dd = metrics_df.loc[best_dd_method, 'Max Drawdown']
        click.echo(f"  Lowest Max Drawdown: {best_dd_method} ({best_dd:.2%})")
        
        click.echo(f"\nView results:")
        click.echo(f"  - Metrics table: {results_dir}/metrics.csv")
        click.echo(f"  - Equity curves: {results_dir}/equity_curves.png")
        click.echo(f"  - Risk-return plot: {results_dir}/risk_return_scatter.png")
    
    click.echo("\n✓ Done!")


if __name__ == '__main__':
    main()
