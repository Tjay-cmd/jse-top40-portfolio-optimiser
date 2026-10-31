# JSE Top-40 Portfolio Optimiser Lab

**Author:** Tjay Burger  
**GitHub:** [Tjay-cmd](https://github.com/Tjay-cmd)

## Research Question

Which portfolio construction method delivers the best out-of-sample risk-adjusted returns on JSE equities after accounting for transaction costs, turnover, and ZAR exchange rate risk?

## Key Findings

<!-- METRICS:KEY_FINDINGS:START -->
1. **Mean-Variance optimization achieved the highest return (15.88%) but with proportionally higher risk (22.96% vol)** and significantly higher turnover (21.72%). Its Sharpe ratio is 0.387 and its Deflated Sharpe Ratio is 0.660 — the highest among strategies tested, but still below the usual 0.95 threshold. This is promising, not proven outperformance.

2. **Equal Weight (1/N) outperformed the passive benchmark** with 10.33% return vs 8.34% for STX40, while maintaining lower turnover (3.25%) and comparable volatility. This confirms the findings of DeMiguel et al. (2009).

3. **Deflated Sharpe Ratio correctly penalizes for multiple testing**: When testing 7 strategies, even good Sharpe ratios receive DSR < 0.7. The DSR is the probability that the true Sharpe exceeds the expected maximum Sharpe from luck across the N=7 trials, not the probability that it exceeds zero.

4. **Transaction costs significantly impact high-turnover strategies**: Mean-Variance's annual cost drag is 1.57% (21.72% monthly one-way turnover × 2 sides × 30 bps × 12).

5. **Hierarchical Risk Parity underperformed on return** (8.18% vs 8.34% for STX40) with DSR 0.195. HRP's Sharpe (0.079) still slightly beats STX40 (0.075) because HRP's volatility (14.98%) is lower than the benchmark (17.88%). That is a volatility effect, not higher absolute return.
<!-- METRICS:KEY_FINDINGS:END -->

## Methodology

### Universe

JSE Top-40 constituent equities, sourced via yfinance with `.JO` suffix. The configuration lists **35 tickers**. After download and quality filters, **31 series load** (some names have insufficient history or fail the unit-jump / volatility checks). The backtest uses an **unbalanced panel**: stocks enter once they have 252 days of trailing history.

Ticker labels: **BHG.JO is BHP Group** (JSE listing); **BVT.JO is Bidvest**. Bidvest is listed once. BHP.JO / BIL.JO are additional BHP-related listings in the configured set.

### Sample Period

**Data range:** 2010-01-04 to 2026-10-30 (4,289 trading days)  
**Evaluation period:** 2010-12-31 to 2026-10-30 (from first allocation date, ~16 years)

This period captures:
- The 2015-2016 commodity downturn
- The 2020 COVID-19 crash and recovery
- Multiple ZAR volatility cycles
- Varying interest rate environments

On any trading day, **25–31 of the 31 loaded names have a valid price (average 27.5)**. Rebalance holdings can be lower than that because of the 30-name cap and the 1% minimum weight. Late listings such as PRX.JO (September 2019) enter when they have enough history; earlier history for long-listed names is kept.

### Portfolio Construction Methods

1. **Equal Weight (1/N)** - Baseline diversification
2. **Minimum Variance (Sample)** - Minimize portfolio volatility using sample covariance
3. **Minimum Variance (Ledoit-Wolf Shrinkage)** - Shrinkage estimator for more stable covariance
4. **Mean-Variance (Max Sharpe)** - Maximize Sharpe ratio with CVXPY optimization
5. **Risk Parity / Equal Risk Contribution (ERC)** - Equal volatility contribution from each asset
6. **Hierarchical Risk Parity (HRP)** - Machine-learning based clustering + inverse-variance allocation
7. **Black-Litterman** - Market-implied equilibrium returns with Bayesian views

The backtest tests **7 portfolio construction methods** against the JSE Top-40 index benchmark.

All methods respect:
- Long-only constraint
- Maximum 15% position size at rebalancing (inspired by Regulation 28; weights may drift between rebalances)
- Minimum 1% position size (avoid dust holdings)
- Maximum 30 holdings

### Walk-Forward Backtest

- **Rebalance frequency:** Monthly
- **Estimation window:** Expanding (all prior data)
- **Minimum history:** 252 trading days before first allocation
- **Transaction costs:** 30 basis points (0.3%) per trade
- **Initial capital:** ZAR 1,000,000

At each rebalance, only stocks with at least 252 days of trailing returns are eligible for inclusion.

#### Turnover and Cost Measurement

**Turnover** is measured as **one-way turnover**: 
```
Turnover = sum(|new_weight - old_weight|) / 2
```

This represents the fraction of the portfolio that changes. For example, going from 50% A + 50% B to 100% A + 0% B has 50% one-way turnover (you sell 50% and buy 50%).

**Transaction costs** are charged on **both buying and selling**:
```
Cost = Turnover × 2 × Cost_per_side × Portfolio_value
```

With 30bps cost per side and 20% monthly turnover:
- Monthly cost = 20% × 2 × 0.3% = 0.12% of portfolio
- Annual cost drag = 0.12% × 12 = 1.44%

**Annual cost drag** is computed as the average cost rate (costs as % of portfolio value) at each rebalance, multiplied by 12. This correctly accounts for portfolio growth over time.

## Data Cleaning

JSE price series from yfinance exhibit **unit jumps** where mid-series the reported unit switches between cents and rands (or vice versa), producing spurious ~99% single-day moves. For example:

- **STX40.JO on 2025-04-25**: Price jumped from ~84 to ~8500 (x100), then on 2025-04-29 dropped back to ~84 (/100 correction)
- **SBK.JO** and other tickers showed similar patterns

The `JSEDataCleaner` class in `src/jse_portfolio_lab/data/cleaner.py`:

1. **Detects unit jumps** by identifying returns near ±99% (threshold 80%)
2. **Corrects** the series by multiplying or dividing all subsequent prices by 100
3. **Removes stale prices** (10+ consecutive identical values)
4. **Validates** cleaned series: rejects if annualized volatility >150% or max daily return >50%

After cleaning, benchmark (STX40) volatility dropped from >2400% to ~17.9% annualized, confirming data integrity.

## Results

<!-- METRICS:RESULTS:START -->
All metrics annualized, risk-free rate 7% p.a. **Evaluation period: 2010-12-31 to 2026-10-30** (from first allocation date). **Deflated Sharpe Ratio** (Bailey & López de Prado, 2014) adjusts for multiple testing across 7 strategies.

Annualized return uses one convention everywhere: `(1 + r).prod() ** (252 / n) - 1` on daily returns in the evaluation window (`n` = number of trading days). That is the figure in `reports/metrics.csv`. Comparing terminal portfolio value to the value already observed on the first allocation date omits the first-day transaction-cost return and is not used.

| **Method**               | **Return** | **Vol** | **Sharpe** | **Deflated Sharpe** | **Max DD** | **Avg Turnover** | **Annual Cost Drag** |
|--------------------------|------------|---------|------------|---------------------|------------|------------------|----------------------|
| **Mean-Variance**        | **15.88%** | **22.96%** | **0.387**  | **0.660**           | **40.40%**     | **21.72%**        | **1.57%**            |
| Equal Weight             | 10.33%     | 17.56%  | 0.189      | 0.345               | 41.08%     | 3.25%            | 0.23%                |
| Risk Parity              | 9.90%      | 15.88%  | 0.183      | 0.323               | 41.53%     | 3.49%            | 0.25%                |
| Black-Litterman          | 9.61%      | 14.69%  | 0.178      | 0.307               | 44.20%     | 4.98%            | 0.36%                |
| Min Variance (Shrunk)    | 9.06%      | 14.27%  | 0.144      | 0.262               | 44.74%     | 5.06%            | 0.36%                |
| Min Variance (Sample)    | 9.03%      | 14.29%  | 0.142      | 0.259               | 44.98%     | 5.23%            | 0.38%                |
| **STX40 (Benchmark)**    | **8.34%** | **17.88%** | **0.075**  | **0.209**           | **37.29%**     | **0.00%**        | **0.00%**            |
| HRP                      | 8.18%      | 14.98%  | 0.079      | 0.195               | 49.72%     | 6.25%            | 0.45%                |

**Cost Reconciliation:** For Mean-Variance with 21.72% monthly one-way turnover: Annual cost ≈ 21.72% × 2 (both sides) × 0.3% (per side) × 12 (months) = 1.56%. The reported drag (1.57%) is the average of (cost / portfolio value) at each rebalance, times 12.

### Interpretation

1. **Mean-Variance's DSR of 0.660** is the highest among all strategies: the probability that its true Sharpe exceeds the expected maximum Sharpe from luck across N=7 trials (not the probability it exceeds zero). That is promising, not a claim of proven skill. Annual cost drag is **1.57%**.

2. **Equal Weight (DSR 0.345), Risk Parity (DSR 0.323), and Black-Litterman (DSR 0.307)** form a lower-turnover cluster. Their annual cost drag is 0.23%–0.36%.

3. **HRP return (8.18%) is below STX40 (8.34%)**, but HRP's Sharpe (0.079) is slightly above STX40 (0.075) because HRP volatility (14.98%) is lower than the benchmark (17.88%). DSR 0.195 still indicates the edge is not distinguishable from luck across 7 trials.

4. **The benchmark's DSR (0.209)** is lower than Equal Weight, Risk Parity, Black-Litterman, and Mean-Variance.
<!-- METRICS:RESULTS:END -->

## Visualizations

![Equity Curves](reports/equity_curves.png)
*Portfolio value evolution from first allocation (2010-12-31) to 2026-10-30. Mean-Variance shows highest terminal wealth but with greater volatility.*

![Risk-Return Scatter](reports/risk_return_scatter.png)
*Risk-return tradeoff: Mean-Variance sits on the upper right (high return, high risk), while Minimum Variance strategies occupy the lower left.*

![Drawdowns](reports/drawdowns.png)
*Maximum drawdown periods. All strategies experienced 40-50% drawdowns during major market dislocations (2015-16, 2020 COVID).*

![Rolling Sharpe](reports/rolling_sharpe.png)
*252-day rolling Sharpe ratios show significant time-variation. Mean-Variance dominates in bull markets but suffers in volatile periods.*

![Cost Sensitivity](reports/cost_sensitivity.png)
*Sharpe ratio sensitivity to transaction costs (10-50 bps). All 7 methods included to compare cost impact across strategies.*

## Limitations

1. **Survivorship bias**: The universe includes only current or recent Top-40 constituents. Delisted stocks (bankruptcies, takeovers) are excluded, likely inflating returns.

2. **Liquidity assumptions**: The backtest assumes all trades execute at closing prices with fixed 30 bps cost. In reality, larger positions in less liquid stocks face market impact and wider bid-ask spreads.

3. **Look-ahead bias in universe selection**: Using the "current" Top-40 list introduces subtle look-ahead bias. A fully rigorous study would reconstruct the Top-40 membership at each historical point.

4. **ZAR risk not hedged**: All returns are in ZAR. International investors face additional currency risk not captured here.

5. **No borrowing costs or short constraints**: Portfolios are long-only, but in practice, even slight leverage (e.g., for rebalancing) incurs margin costs.

## Repository Structure

```
.
├── src/jse_portfolio_lab/
│   ├── data/               # Data loading, cleaning, preprocessing
│   ├── optimization/       # Portfolio optimizers
│   ├── backtesting/        # Walk-forward backtest engine
│   ├── metrics/            # Performance metrics (incl. Deflated Sharpe)
│   ├── utils/              # Visualization, config
│   └── cli.py              # CLI orchestrator
├── tests/                  # Pytest suite (one live-data integration test is skipped)
├── config/config.yaml      # Backtest configuration
├── reports/                # Generated charts and metrics
├── requirements.txt        # Pinned dependencies
└── README.md
```

## Installation

```bash
# Clone repository
git clone https://github.com/Tjay-cmd/jse-top40-portfolio-optimiser.git
cd jse-top40-portfolio-optimiser

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .

# Run backtest
python -m jse_portfolio_lab.cli
```

Requirements: Python 3.12+, see `requirements.txt` for dependency ranges.

## Testing

```bash
pytest tests/ -v
```

One integration test is skipped (requires a live download). Do not hardcode a pass count here — `pytest tests/ -v` is the source of truth.

## Configuration

Edit `config/config.yaml` to:
- Change stock universe (default: JSE Top-40 constituents)
- Adjust transaction costs (default: 30 bps)
- Modify constraints (max position, max holdings)
- Enable/disable optimization methods
- Set date range

## References

- Bailey, D. H., & López de Prado, M. (2014). The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality. *Journal of Portfolio Management*, 40(5), 94-107.
- DeMiguel, V., Garlappi, L., & Uppal, R. (2009). Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy? *Review of Financial Studies*, 22(5), 1915-1953.
- Ledoit, O., & Wolf, M. (2004). Honey, I Shrunk the Sample Covariance Matrix. *Journal of Portfolio Management*, 30(4), 110-119.
- López de Prado, M. (2016). Building Diversified Portfolios that Outperform Out of Sample. *Journal of Portfolio Management*, 42(4), 59-69.

## License

MIT

## Author

Tjay Burger  
[github.com/Tjay-cmd](https://github.com/Tjay-cmd)

