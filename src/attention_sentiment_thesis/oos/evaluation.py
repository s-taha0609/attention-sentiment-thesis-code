"""Strict-sample OOS metrics, Clark--West tests, and mean adjustment."""

from math import erf, sqrt
from collections.abc import Mapping
import numpy as np
import pandas as pd

KEYS = ["firm_id", "trading_date"]

def strict_common_sample(
    realized: pd.DataFrame,
    benchmark: pd.DataFrame,
    model_predictions: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Inner-join realized, benchmark, and every model forecast."""
    out = realized[KEYS + ["realized"]].copy()
    out = out.merge(benchmark[KEYS + ["benchmark"]], on=KEYS, how="inner")
    for name, frame in model_predictions.items():
        out = out.merge(
            frame[KEYS + ["prediction"]].rename(columns={"prediction": name}),
            on=KEYS,
            how="inner",
        )
    return out.dropna().sort_values(KEYS).reset_index(drop=True)

def evaluate_forecasts(sample: pd.DataFrame, model_columns) -> pd.DataFrame:
    benchmark_mspe = float(np.mean((sample["realized"] - sample["benchmark"]) ** 2))
    rows = []
    for model in model_columns:
        mspe = float(np.mean((sample["realized"] - sample[model]) ** 2))
        rows.append({
            "model": model,
            "mspe": mspe,
            "oos_r2": 1.0 - mspe / benchmark_mspe,
        })
    return pd.DataFrame(rows)

def daily_mean_adjust(sample: pd.DataFrame, forecast_columns) -> pd.DataFrame:
    """Remove each date's cross-sectional mean from realized and forecasts."""
    out = sample.copy()
    columns = ["realized", *forecast_columns]
    out[columns] = out[columns] - out.groupby("trading_date")[columns].transform("mean")
    return out

def _newey_west_mean_se(values: np.ndarray, lag: int) -> float:
    values = np.asarray(values, dtype=float)
    centered = values - values.mean()
    t = len(values)
    long_run = float(centered @ centered / t)
    for order in range(1, min(lag, t - 1) + 1):
        covariance = float(centered[order:] @ centered[:-order] / t)
        long_run += 2.0 * (1.0 - order / (lag + 1.0)) * covariance
    return sqrt(max(long_run, 0.0) / t)

def clark_west_test(
    sample: pd.DataFrame,
    small_model: str,
    large_model: str,
) -> dict[str, float]:
    """One-sided Clark--West test after date-level cross-sectional averaging."""
    y = sample["realized"]
    adjusted = (
        (y - sample[small_model]) ** 2
        - (y - sample[large_model]) ** 2
        + (sample[large_model] - sample[small_model]) ** 2
    )
    daily = adjusted.groupby(sample["trading_date"]).mean().to_numpy(float)
    lag = max(1, int(len(daily) ** (1 / 3)))
    se = _newey_west_mean_se(daily, lag)
    statistic = float(daily.mean() / se) if se > 0 else float("nan")
    p_value = 0.5 * (1.0 - erf(statistic / sqrt(2.0)))
    return {
        "mean_adjusted_loss": float(daily.mean()),
        "statistic": statistic,
        "one_sided_p_value": float(p_value),
        "newey_west_lag": float(lag),
    }

def bonferroni_holm(p_values: Mapping[str, float]) -> dict[str, float]:
    """Bonferroni--Holm adjustment across the supplied valid p-values."""
    names = list(p_values)
    values = np.array([p_values[name] for name in names], dtype=float)
    valid_positions = np.flatnonzero(np.isfinite(values))
    adjusted = np.full(len(values), np.nan)
    if len(valid_positions):
        order = valid_positions[np.argsort(values[valid_positions])]
        running = 0.0
        for rank, position in enumerate(order):
            running = max(running, (len(order) - rank) * values[position])
            adjusted[position] = min(running, 1.0)
    return {name: float(adjusted[index]) for index, name in enumerate(names)}

def clark_west_comparisons(
    sample: pd.DataFrame,
    comparisons,
) -> pd.DataFrame:
    """Run the final six nested comparisons and Holm-adjust their p-values."""
    rows = []
    raw = {}
    for small, large in comparisons:
        result = clark_west_test(sample, small, large)
        name = f"{large}_vs_{small}"
        raw[name] = result["one_sided_p_value"]
        rows.append({"comparison": name, "small": small, "large": large, **result})
    adjusted = bonferroni_holm(raw)
    for row in rows:
        row["holm_p_value"] = adjusted[row["comparison"]]
    return pd.DataFrame(rows)
