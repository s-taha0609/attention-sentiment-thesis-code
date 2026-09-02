"""Aggregate estimated expression polarity to firm-trading-date sentiment."""

import pandas as pd
from ..schemas import require_columns, validate_expression_events

def aggregate_daily_sentiment(
    events: pd.DataFrame,
    polarities: pd.DataFrame,
) -> pd.DataFrame:
    """Sum each expression's polarity once per eligible event.

    A zero score means no measured non-zero net signal: it can reflect no
    qualifying event, no retained expression, or offsetting polarities.
    """
    source = validate_expression_events(events)
    require_columns(
        polarities, {"scoring_year", "expression_id", "polarity"}, "polarities"
    )
    lookup = {
        (int(row.scoring_year), str(row.expression_id)): float(row.polarity)
        for row in polarities.itertuples(index=False)
    }
    eligible = source[source["eligible_for_daily_score"]].copy()
    eligible["scoring_year"] = eligible["effective_trading_date"].dt.year
    eligible["event_sentiment"] = eligible.apply(
        lambda row: sum(
            lookup.get((int(row["scoring_year"]), str(expression)), 0.0)
            for expression in set(row["expressions"])
        ),
        axis=1,
    )
    daily = (
        eligible.groupby(["firm_id", "effective_trading_date"], as_index=False)
        ["event_sentiment"].sum()
        .rename(columns={
            "effective_trading_date": "trading_date",
            "event_sentiment": "sentiment",
        })
    )
    return daily
