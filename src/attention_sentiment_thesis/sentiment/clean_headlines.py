"""Provider-neutral event normalization.

Despite the historical module name, this public implementation never accepts or
processes article text. It validates expression identifiers produced upstream.
"""

import pandas as pd
from ..schemas import validate_expression_events

def normalize_expression_events(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize schema and de-duplicate expression presence within each event."""
    out = validate_expression_events(frame)
    out["firm_id"] = out["firm_id"].astype(str)
    out["expressions"] = out["expressions"].map(
        lambda values: tuple(dict.fromkeys(str(value) for value in values))
    )
    out["eligible_for_daily_score"] = out["eligible_for_daily_score"].astype(bool)
    return out.sort_values(["publication_timestamp", "firm_id"]).reset_index(drop=True)

