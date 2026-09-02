"""Construct close-to-close and within-day return components."""

import numpy as np
import pandas as pd
from ..schemas import require_columns

def construct_returns(
    prices: pd.DataFrame,
    *,
    firm_col: str = "firm_id",
    date_col: str = "trading_date",
    open_col: str = "open_price",
    close_col: str = "close_price",
) -> pd.DataFrame:
    """Return sorted rows with r_cc, r_oc, and next-observation target_return.

    Returns are log returns. The forecast target at row t is the ticker's
    close-to-close return at its next observed trading date.
    """
    require_columns(prices, {firm_col, date_col, open_col, close_col}, "prices")
    out = prices.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="raise").dt.normalize()
    out = out.sort_values([firm_col, date_col]).reset_index(drop=True)
    if (out[[open_col, close_col]] <= 0).any().any():
        raise ValueError("prices must be strictly positive")
    groups = out.groupby(firm_col, sort=False)
    out["r_cc"] = groups[close_col].transform(lambda x: np.log(x).diff())
    out["r_oc"] = np.log(out[close_col]) - np.log(out[open_col])
    out["target_return"] = groups["r_cc"].shift(-1)
    return out

