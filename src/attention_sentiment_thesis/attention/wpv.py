"""Wikipedia page-view attention measure."""

import numpy as np
import pandas as pd
from ..schemas import require_columns

def compute_raw_wpv(
    frame: pd.DataFrame,
    *,
    views_col: str = "page_views",
    window: int = 8,
    min_observations: int = 4,
) -> pd.DataFrame:
    """Compute log views minus the median log baseline of prior observations.

    The current observation is excluded from the eight-observation benchmark.
    Winsorization and the final ticker-level lag are separate, IS-fitted steps.
    """
    require_columns(frame, {"firm_id", "trading_date", views_col}, "page_views")
    out = frame.sort_values(["firm_id", "trading_date"]).copy()
    views = pd.to_numeric(out[views_col], errors="coerce")
    views = views.where(views > 0)
    prior_median = (
        views.groupby(out["firm_id"], sort=False)
        .transform(lambda x: x.shift(1).rolling(window, min_periods=min_observations).median())
    )
    out["wpv_raw"] = np.log(views) - np.log(prior_median)
    return out

