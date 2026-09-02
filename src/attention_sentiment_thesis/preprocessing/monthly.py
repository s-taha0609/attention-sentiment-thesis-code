"""Release-aware alignment of monthly controls to observed trading dates."""

import numpy as np
import pandas as pd
from ..schemas import require_columns

def availability_date(
    reference_month: pd.Timestamp,
    trading_dates: pd.DatetimeIndex,
    trading_day_number: int = 40,
) -> pd.Timestamp:
    month_end = pd.Timestamp(reference_month).to_period("M").end_time.normalize()
    later = trading_dates[trading_dates > month_end]
    if len(later) < trading_day_number:
        return pd.NaT
    return pd.Timestamp(later[trading_day_number - 1])

def align_monthly_controls(
    monthly: pd.DataFrame,
    trading_dates,
    *,
    month_col: str = "reference_month",
    trading_day_number: int = 40,
) -> pd.DataFrame:
    """Make each monthly observation available on the 40th later trading date.

    Values are forward-filled after availability; interpolation and backfill are
    deliberately absent.
    """
    require_columns(monthly, {month_col}, "monthly_controls")
    dates = pd.DatetimeIndex(pd.to_datetime(trading_dates)).normalize().unique().sort_values()
    values = monthly.copy()
    values[month_col] = pd.to_datetime(values[month_col], errors="raise")
    values["availability_date"] = values[month_col].map(
        lambda m: availability_date(m, dates, trading_day_number)
    )
    values = values.dropna(subset=["availability_date"]).sort_values("availability_date")
    calendar = pd.DataFrame({"trading_date": dates})
    aligned = pd.merge_asof(
        calendar,
        values,
        left_on="trading_date",
        right_on="availability_date",
        direction="backward",
        allow_exact_matches=True,
    )
    return aligned

def construct_monthly_controls(
    monthly: pd.DataFrame,
    *,
    growth_columns=("cpi_jp", "ip_jp", "m1_jp"),
    level_columns=("unemployment_jp", "jgb_10y", "jp_3m"),
    month_col: str = "reference_month",
) -> pd.DataFrame:
    """Log-difference CPI/IP/M1 and retain unemployment/rate variables in levels."""
    require_columns(monthly, {month_col, *growth_columns, *level_columns}, "monthly_controls")
    out = monthly.sort_values(month_col).copy()
    for column in growth_columns:
        values = pd.to_numeric(out[column], errors="coerce")
        if (values.dropna() <= 0).any():
            raise ValueError(f"{column} must be positive for log differences")
        out[column] = np.log(values).diff()
    for column in level_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out
