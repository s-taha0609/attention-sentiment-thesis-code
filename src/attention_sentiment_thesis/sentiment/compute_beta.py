"""Rolling market-beta and market-premium calculation."""

import numpy as np
import pandas as pd
from ..schemas import require_columns
from ..spec import FINAL_SPEC

def compute_rolling_beta(
    returns: pd.DataFrame,
    *,
    firm_return_col: str = "firm_return",
    market_return_col: str = "market_return",
    window: int = FINAL_SPEC.beta_window,
    min_observations: int = FINAL_SPEC.beta_min_observations,
) -> pd.DataFrame:
    """Estimate beta from preceding observations, excluding the current row."""
    require_columns(
        returns,
        {"firm_id", "trading_date", firm_return_col, market_return_col},
        "returns",
    )
    out = returns.sort_values(["firm_id", "trading_date"]).copy()
    beta = pd.Series(np.nan, index=out.index, dtype=float)
    for _, group in out.groupby("firm_id", sort=False):
        indices = group.index.to_numpy()
        firm = pd.to_numeric(group[firm_return_col], errors="coerce").to_numpy(float)
        market = pd.to_numeric(group[market_return_col], errors="coerce").to_numpy(float)
        for position in range(len(group)):
            start = max(0, position - window)
            x = market[start:position]
            y = firm[start:position]
            valid = np.isfinite(x) & np.isfinite(y)
            if valid.sum() < min_observations:
                continue
            variance = np.var(x[valid], ddof=1)
            if variance > 0 and np.isfinite(variance):
                beta.loc[indices[position]] = np.cov(y[valid], x[valid], ddof=1)[0, 1] / variance
    out["beta"] = beta
    out["market_premium"] = (
        pd.to_numeric(out[firm_return_col], errors="coerce")
        - out["beta"] * pd.to_numeric(out[market_return_col], errors="coerce")
    )
    return out

