"""README numbers must come from reports/metrics.csv."""

from pathlib import Path

import pandas as pd

from jse_portfolio_lab.utils.readme_sync import pct, ratio, update_readme


ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "reports" / "metrics.csv"
README_PATH = ROOT / "README.md"
METADATA_PATH = ROOT / "reports" / "run_metadata.json"


def test_readme_key_findings_match_metrics_csv():
    """Key Findings and Results must quote the current metrics.csv figures."""
    metrics = pd.read_csv(METRICS_PATH, index_col="Method")
    readme = README_PATH.read_text()
    mvo = metrics.loc["mean_variance"]
    hrp = metrics.loc["hrp"]
    bench = metrics.loc["STX40 (Benchmark)"]

    assert pct(mvo["Annualized Return"]) in readme
    assert ratio(mvo["Sharpe Ratio"]) in readme
    assert ratio(mvo["Deflated Sharpe Ratio"]) in readme
    assert pct(hrp["Annualized Return"]) in readme
    assert ratio(hrp["Sharpe Ratio"]) in readme
    assert ratio(bench["Sharpe Ratio"]) in readme
    assert "lower" in readme.lower() and "volatil" in readme.lower()


def test_readme_sync_is_idempotent(tmp_path):
    """Regenerating marked sections from metrics.csv must be stable."""
    dest = tmp_path / "README.md"
    dest.write_text(README_PATH.read_text())
    update_readme(dest, METRICS_PATH, METADATA_PATH)
    first = dest.read_text()
    update_readme(dest, METRICS_PATH, METADATA_PATH)
    assert dest.read_text() == first
