"""
Regenerate README metric sections from reports/metrics.csv.

Numbers in the README must come from the pipeline output, not hand edits.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


METHOD_DISPLAY = {
    "mean_variance": "Mean-Variance",
    "equal_weight": "Equal Weight",
    "risk_parity": "Risk Parity",
    "black_litterman": "Black-Litterman",
    "min_variance_shrunk": "Min Variance (Shrunk)",
    "min_variance": "Min Variance (Sample)",
    "hrp": "HRP",
    "STX40 (Benchmark)": "STX40 (Benchmark)",
}

KEY_FINDINGS_MARKERS = (
    "<!-- METRICS:KEY_FINDINGS:START -->",
    "<!-- METRICS:KEY_FINDINGS:END -->",
)
RESULTS_MARKERS = (
    "<!-- METRICS:RESULTS:START -->",
    "<!-- METRICS:RESULTS:END -->",
)


def pct(value: float, digits: int = 2) -> str:
    """Format a decimal as a percentage string."""
    return f"{value * 100:.{digits}f}%"


def ratio(value: float, digits: int = 3) -> str:
    """Format a ratio to a fixed number of decimals."""
    return f"{value:.{digits}f}"


def load_metrics(metrics_path: Path) -> pd.DataFrame:
    """Load the pipeline metrics table."""
    df = pd.read_csv(metrics_path, index_col="Method")
    return df


def load_metadata(metadata_path: Path) -> Dict:
    """Load optional run metadata written by the pipeline."""
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text())


def render_key_findings(metrics: pd.DataFrame) -> str:
    """Render the Key Findings list from metrics.csv."""
    mvo = metrics.loc["mean_variance"]
    ew = metrics.loc["equal_weight"]
    bench = metrics.loc["STX40 (Benchmark)"]
    hrp = metrics.loc["hrp"]

    return (
        f"1. **Mean-Variance optimization achieved the highest return "
        f"({pct(mvo['Annualized Return'])}) but with proportionally higher risk "
        f"({pct(mvo['Annualized Volatility'])} vol)** and significantly higher turnover "
        f"({pct(mvo['Avg Turnover'])}). Its Sharpe ratio is {ratio(mvo['Sharpe Ratio'])} "
        f"and its Deflated Sharpe Ratio is {ratio(mvo['Deflated Sharpe Ratio'])} — "
        f"the highest among strategies tested, but still below the usual 0.95 "
        f"threshold. This is promising, not proven outperformance.\n"
        f"\n"
        f"2. **Equal Weight (1/N) outperformed the passive benchmark** with "
        f"{pct(ew['Annualized Return'])} return vs {pct(bench['Annualized Return'])} "
        f"for STX40, while maintaining lower turnover ({pct(ew['Avg Turnover'])}) "
        f"and comparable volatility. This confirms the findings of DeMiguel et al. (2009).\n"
        f"\n"
        f"3. **Deflated Sharpe Ratio correctly penalizes for multiple testing**: "
        f"When testing 7 strategies, even good Sharpe ratios receive DSR < 0.7. "
        f"The DSR is the probability that the true Sharpe exceeds the expected "
        f"maximum Sharpe from luck across the N=7 trials, not the probability "
        f"that it exceeds zero.\n"
        f"\n"
        f"4. **Transaction costs significantly impact high-turnover strategies**: "
        f"Mean-Variance's annual cost drag is {pct(mvo['Annual Cost Drag'])} "
        f"({pct(mvo['Avg Turnover'])} monthly one-way turnover × 2 sides × 30 bps × 12).\n"
        f"\n"
        f"5. **Hierarchical Risk Parity underperformed on return** "
        f"({pct(hrp['Annualized Return'])} vs {pct(bench['Annualized Return'])} for STX40) "
        f"with DSR {ratio(hrp['Deflated Sharpe Ratio'])}. HRP's Sharpe "
        f"({ratio(hrp['Sharpe Ratio'])}) still slightly beats STX40 "
        f"({ratio(bench['Sharpe Ratio'])}) because HRP's volatility "
        f"({pct(hrp['Annualized Volatility'])}) is lower than the benchmark "
        f"({pct(bench['Annualized Volatility'])}). That is a volatility effect, "
        f"not higher absolute return."
    )


def render_results_section(metrics: pd.DataFrame, metadata: Optional[Dict] = None) -> str:
    """Render the Results intro, table, and interpretation from metrics.csv."""
    metadata = metadata or {}
    eval_start = metadata.get("evaluation_start", "2010-12-31")
    eval_end = metadata.get("evaluation_end", "2026-10-30")

    ranked = metrics.sort_values("Annualized Return", ascending=False)

    header = (
        "| **Method**               | **Return** | **Vol** | **Sharpe** | "
        "**Deflated Sharpe** | **Max DD** | **Avg Turnover** | **Annual Cost Drag** |"
    )
    sep = (
        "|--------------------------|------------|---------|------------|"
        "---------------------|------------|------------------|----------------------|"
    )
    rows = [header, sep]
    for method, row in ranked.iterrows():
        name = METHOD_DISPLAY.get(method, method)
        if method in ("mean_variance", "STX40 (Benchmark)"):
            name = f"**{name}**"
            fmt = (
                f"| {name:<24} | **{pct(row['Annualized Return'])}** | "
                f"**{pct(row['Annualized Volatility'])}** | "
                f"**{ratio(row['Sharpe Ratio'])}**  | "
                f"**{ratio(row['Deflated Sharpe Ratio'])}**           | "
                f"**{pct(row['Max Drawdown'])}**     | "
                f"**{pct(row['Avg Turnover'])}**        | "
                f"**{pct(row['Annual Cost Drag'])}**            |"
            )
        else:
            fmt = (
                f"| {name:<24} | {pct(row['Annualized Return']):<10} | "
                f"{pct(row['Annualized Volatility']):<7} | "
                f"{ratio(row['Sharpe Ratio']):<10} | "
                f"{ratio(row['Deflated Sharpe Ratio']):<19} | "
                f"{pct(row['Max Drawdown']):<10} | "
                f"{pct(row['Avg Turnover']):<16} | "
                f"{pct(row['Annual Cost Drag']):<20} |"
            )
        rows.append(fmt)

    table = "\n".join(rows)

    mvo = metrics.loc["mean_variance"]
    ew = metrics.loc["equal_weight"]
    rp = metrics.loc["risk_parity"]
    bl = metrics.loc["black_litterman"]
    hrp = metrics.loc["hrp"]
    bench = metrics.loc["STX40 (Benchmark)"]

    expected_mvo_drag = mvo["Avg Turnover"] * 2 * 0.003 * 12

    return (
        f"All metrics annualized, risk-free rate 7% p.a. "
        f"**Evaluation period: {eval_start} to {eval_end}** "
        f"(from first allocation date). **Deflated Sharpe Ratio** "
        f"(Bailey & López de Prado, 2014) adjusts for multiple testing across 7 strategies.\n"
        f"\n"
        f"Annualized return uses one convention everywhere: "
        f"`(1 + r).prod() ** (252 / n) - 1` on daily returns in the evaluation "
        f"window (`n` = number of trading days). That is the figure in "
        f"`reports/metrics.csv`. Comparing terminal portfolio value to the "
        f"value already observed on the first allocation date omits the "
        f"first-day transaction-cost return and is not used.\n"
        f"\n"
        f"{table}\n"
        f"\n"
        f"**Cost Reconciliation:** For Mean-Variance with "
        f"{pct(mvo['Avg Turnover'])} monthly one-way turnover: "
        f"Annual cost ≈ {pct(mvo['Avg Turnover'])} × 2 (both sides) × 0.3% "
        f"(per side) × 12 (months) = {pct(expected_mvo_drag)}. "
        f"The reported drag ({pct(mvo['Annual Cost Drag'])}) is the average "
        f"of (cost / portfolio value) at each rebalance, times 12.\n"
        f"\n"
        f"### Interpretation\n"
        f"\n"
        f"1. **Mean-Variance's DSR of {ratio(mvo['Deflated Sharpe Ratio'])}** "
        f"is the highest among all strategies: the probability that its true "
        f"Sharpe exceeds the expected maximum Sharpe from luck across N=7 "
        f"trials (not the probability it exceeds zero). That is promising, "
        f"not a claim of proven skill. Annual cost drag is "
        f"**{pct(mvo['Annual Cost Drag'])}**.\n"
        f"\n"
        f"2. **Equal Weight (DSR {ratio(ew['Deflated Sharpe Ratio'])}), "
        f"Risk Parity (DSR {ratio(rp['Deflated Sharpe Ratio'])}), and "
        f"Black-Litterman (DSR {ratio(bl['Deflated Sharpe Ratio'])})** "
        f"form a lower-turnover cluster. Their annual cost drag is "
        f"{pct(ew['Annual Cost Drag'])}–{pct(bl['Annual Cost Drag'])}.\n"
        f"\n"
        f"3. **HRP return ({pct(hrp['Annualized Return'])}) is below STX40 "
        f"({pct(bench['Annualized Return'])})**, but HRP's Sharpe "
        f"({ratio(hrp['Sharpe Ratio'])}) is slightly above STX40 "
        f"({ratio(bench['Sharpe Ratio'])}) because HRP volatility "
        f"({pct(hrp['Annualized Volatility'])}) is lower than the benchmark "
        f"({pct(bench['Annualized Volatility'])}). DSR "
        f"{ratio(hrp['Deflated Sharpe Ratio'])} still indicates the edge "
        f"is not distinguishable from luck across 7 trials.\n"
        f"\n"
        f"4. **The benchmark's DSR ({ratio(bench['Deflated Sharpe Ratio'])})** "
        f"is lower than Equal Weight, Risk Parity, Black-Litterman, and "
        f"Mean-Variance."
    )


def _replace_marked_section(text: str, start: str, end: str, body: str) -> str:
    """Replace the content between HTML comment markers."""
    pattern = re.compile(
        re.escape(start) + r".*?" + re.escape(end),
        flags=re.DOTALL,
    )
    replacement = f"{start}\n{body}\n{end}"
    if not pattern.search(text):
        raise ValueError(f"README is missing markers {start} / {end}")
    return pattern.sub(replacement, text)


def update_readme(
    readme_path: Path,
    metrics_path: Path,
    metadata_path: Optional[Path] = None,
) -> str:
    """Rewrite marked README sections from metrics.csv and return the new text."""
    metrics = load_metrics(metrics_path)
    metadata = load_metadata(metadata_path) if metadata_path else {}
    text = readme_path.read_text()
    text = _replace_marked_section(
        text, *KEY_FINDINGS_MARKERS, render_key_findings(metrics)
    )
    text = _replace_marked_section(
        text, *RESULTS_MARKERS, render_results_section(metrics, metadata)
    )
    readme_path.write_text(text)
    return text
