"""Abnormal trading-volume construction."""

import numpy as np
import pandas as pd
from ..schemas import require_columns

def compute_atv(
    frame: pd.DataFrame,
    *,
    volume_col: str = "volume",
    window: int = 60,
    min_observations: int = 30,
) -> pd.DataFrame:
    """Standardize volume using at most 60 prior ticker observations."""
    require_columns(frame, {"firm_id", "trading_date", volume_col}, "volume")
    out = frame.sort_values(["firm_id", "trading_date"]).copy()
    volume = pd.to_numeric(out[volume_col], errors="coerce")
    grouped = volume.groupby(out["firm_id"], sort=False)
    prior_mean = grouped.transform(
        lambda x: x.shift(1).rolling(window, min_periods=min_observations).mean()
    )
    prior_std = grouped.transform(
        lambda x: x.shift(1).rolling(window, min_periods=min_observations).std()
    )
    out["atv"] = (volume - prior_mean) / prior_std
    out.loc[(prior_std == 0) | ~np.isfinite(out["atv"]), "atv"] = np.nan
    return out

