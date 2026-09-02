"""Small, data-agnostic helpers for final tables and figures."""

from pathlib import Path
from collections.abc import Mapping
import pandas as pd

def write_table(table: pd.DataFrame, path: str | Path) -> None:
    """Write a caller-specified result table; directories are not created implicitly."""
    table.to_csv(Path(path), index=False)

def plot_cumulative_squared_error_difference(
    sample: pd.DataFrame,
    model: str,
    *,
    benchmark: str = "benchmark",
    path: str | Path,
) -> None:
    """Save cumulative benchmark-minus-model squared-error differences."""
    import matplotlib.pyplot as plt
    daily = (
        ((sample["realized"] - sample[benchmark]) ** 2
         - (sample["realized"] - sample[model]) ** 2)
        .groupby(sample["trading_date"]).mean().sort_index().cumsum()
    )
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(daily.index, daily.values)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(xlabel="Trading date", ylabel="Cumulative squared-error difference")
    figure.tight_layout()
    figure.savefig(Path(path), dpi=200)
    plt.close(figure)

FINAL_OUTPUT_MAP: Mapping[str, str] = {
    "headline_regression": "Headline fixed-effects regression table",
    "stability_tests": "Period-interaction stability and Wald-test table",
    "oos_metrics": "Baseline and rolling-window-sensitivity OOS tables",
    "clark_west": "Clark--West comparison table",
    "daily_mean_adjustment": "Daily cross-sectional mean-adjustment table",
    "winsor_sensitivity": "Common-control winsorization sensitivity table",
    "cumulative_error_figure": "Cumulative OOS error-difference figure",
}

