"""Rolling within-firm OLS forecasts and historical-mean benchmark."""

from collections.abc import Sequence
import numpy as np
import pandas as pd
from ..schemas import require_columns
from .window_utils import generate_refit_schedule

def _fit_within(
    training: pd.DataFrame,
    target: str,
    predictors: Sequence[str],
) -> tuple[np.ndarray, pd.DataFrame]:
    means = training.groupby("firm_id")[[target, *predictors]].mean()
    centered = training.join(means, on="firm_id", rsuffix="__mean")
    y = centered[target] - centered[f"{target}__mean"]
    x = np.column_stack([
        centered[column] - centered[f"{column}__mean"] for column in predictors
    ])
    beta = np.linalg.lstsq(x, y.to_numpy(float), rcond=None)[0]
    return beta, means

def rolling_forecasts(
    panel: pd.DataFrame,
    predictors: Sequence[str],
    *,
    target: str = "target_return",
    window_dates: int = 504,
) -> pd.DataFrame:
    """Daily refits; sentiment missing is zero and other missing values listwise."""
    require_columns(panel, {"firm_id", "trading_date", target, *predictors}, "panel")
    source = panel.copy()
    source["trading_date"] = pd.to_datetime(source["trading_date"]).dt.normalize()
    if "sentiment" in predictors:
        source["sentiment"] = source["sentiment"].fillna(0.0)
    non_sentiment = [column for column in predictors if column != "sentiment"]
    results = []
    for window in generate_refit_schedule(source["trading_date"], window_dates):
        training = source[source["trading_date"].isin(window.training_dates)]
        training = training.dropna(subset=[target, *non_sentiment])
        origin = source[source["trading_date"].eq(window.origin)].dropna(
            subset=non_sentiment
        )
        if training.empty or origin.empty:
            continue
        beta, means = _fit_within(training, target, predictors)
        for row in origin.itertuples(index=False):
            firm = getattr(row, "firm_id")
            if firm not in means.index:
                continue
            x = np.array([getattr(row, column) for column in predictors], dtype=float)
            firm_means = means.loc[firm]
            prediction = float(
                firm_means[target]
                + beta @ (x - firm_means[list(predictors)].to_numpy(float))
            )
            results.append({
                "firm_id": firm,
                "trading_date": window.origin,
                "realized": getattr(row, target),
                "prediction": prediction,
            })
    return pd.DataFrame(results)

def rolling_historical_mean(
    panel: pd.DataFrame,
    *,
    target: str = "target_return",
    window_dates: int = 504,
) -> pd.DataFrame:
    """Firm-level historical mean over the same prior-date rolling window."""
    require_columns(panel, {"firm_id", "trading_date", target}, "panel")
    source = panel.copy()
    source["trading_date"] = pd.to_datetime(source["trading_date"]).dt.normalize()
    results = []
    for window in generate_refit_schedule(source["trading_date"], window_dates):
        training = source[source["trading_date"].isin(window.training_dates)]
        means = training.groupby("firm_id")[target].mean()
        origin = source[source["trading_date"].eq(window.origin)]
        for row in origin.itertuples(index=False):
            firm = getattr(row, "firm_id")
            if firm in means.index and pd.notna(means.loc[firm]):
                results.append({
                    "firm_id": firm,
                    "trading_date": window.origin,
                    "benchmark": float(means.loc[firm]),
                })
    return pd.DataFrame(results)

