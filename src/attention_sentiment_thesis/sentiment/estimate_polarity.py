"""Annual recursive expression selection and polarity estimation."""

from collections.abc import Sequence
import numpy as np
import pandas as pd
from ..schemas import require_columns
from ..spec import FINAL_SPEC

def _fit_no_intercept(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(x, y, rcond=None)[0]

def estimate_annual_polarities(
    events: pd.DataFrame,
    vocabulary: Sequence[str],
    scoring_years: Sequence[int],
    *,
    premium_col: str = "market_premium",
    threshold: float = FINAL_SPEC.expression_premium_threshold,
) -> pd.DataFrame:
    """Use observations through year y-1 to estimate polarities for year y."""
    vocab = tuple(vocabulary)
    require_columns(events, {"publication_timestamp", premium_col, *vocab}, "events")
    source = events.copy()
    source["publication_timestamp"] = pd.to_datetime(source["publication_timestamp"])
    records: list[dict] = []
    for year in scoring_years:
        history = source[source["publication_timestamp"].dt.year < int(year)]
        selected = []
        for expression in vocab:
            present = history[expression].eq(1)
            mean_premium = pd.to_numeric(
                history.loc[present, premium_col], errors="coerce"
            ).mean()
            if pd.notna(mean_premium) and abs(float(mean_premium)) > threshold:
                selected.append(expression)
        if not selected:
            continue
        fit = history[[premium_col, *selected]].apply(pd.to_numeric, errors="coerce").dropna()
        if fit.empty:
            continue
        coefficients = _fit_no_intercept(
            fit[selected].to_numpy(float), fit[premium_col].to_numpy(float)
        )
        records.extend(
            {
                "scoring_year": int(year),
                "expression_id": expression,
                "polarity": float(value),
            }
            for expression, value in zip(selected, coefficients)
        )
    return pd.DataFrame(
        records, columns=["scoring_year", "expression_id", "polarity"]
    )

