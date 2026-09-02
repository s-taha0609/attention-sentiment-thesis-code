"""Panel assembly helpers with information-set-safe transformations."""

from collections.abc import Iterable
import numpy as np
import pandas as pd
from ..schemas import require_columns

def lag_wpv_by_trading_observation(panel: pd.DataFrame) -> pd.DataFrame:
    require_columns(panel, {"firm_id", "trading_date", "wpv_raw"}, "panel")
    out = panel.sort_values(["firm_id", "trading_date"]).copy()
    out["wpv_lag1"] = out.groupby("firm_id", sort=False)["wpv_raw"].shift(1)
    return out

def fit_winsor_limits(
    in_sample: pd.DataFrame,
    columns: Iterable[str],
    lower: float = 0.01,
    upper: float = 0.99,
) -> dict[str, tuple[float, float]]:
    """Fit quantile limits using only the supplied in-sample rows."""
    limits = {}
    for column in columns:
        values = pd.to_numeric(in_sample[column], errors="coerce")
        limits[column] = (float(values.quantile(lower)), float(values.quantile(upper)))
    return limits

def apply_winsor_limits(
    frame: pd.DataFrame,
    limits: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    out = frame.copy()
    for column, (low, high) in limits.items():
        out[column] = pd.to_numeric(out[column], errors="coerce").clip(low, high)
    return out

def build_analysis_panel(
    returns: pd.DataFrame,
    sentiment: pd.DataFrame,
    attention: pd.DataFrame,
    controls: pd.DataFrame,
) -> pd.DataFrame:
    """One-to-one merge of prepared components on firm and trading date."""
    keys = ["firm_id", "trading_date"]
    for name, frame in {
        "returns": returns, "sentiment": sentiment,
        "attention": attention, "controls": controls,
    }.items():
        require_columns(frame, keys, name)
        if frame.duplicated(keys).any():
            raise ValueError(f"{name} is not unique on {keys}")
    out = returns.merge(sentiment, on=keys, how="left", validate="one_to_one")
    out = out.merge(attention, on=keys, how="left", validate="one_to_one")
    out = out.merge(controls, on=keys, how="left", validate="one_to_one")
    out["sentiment"] = out["sentiment"].fillna(0.0)
    return lag_wpv_by_trading_observation(out)
