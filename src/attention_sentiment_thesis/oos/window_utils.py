"""Trading-date rolling-window utilities."""

from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class RefitWindow:
    origin: pd.Timestamp
    training_dates: tuple[pd.Timestamp, ...]

def generate_refit_schedule(trading_dates, window_dates: int = 504) -> list[RefitWindow]:
    """Return windows [d-window_dates, ..., d-1]; origin d is never in training."""
    dates = pd.DatetimeIndex(pd.to_datetime(trading_dates)).normalize().unique().sort_values()
    if window_dates <= 0:
        raise ValueError("window_dates must be positive")
    return [
        RefitWindow(
            origin=pd.Timestamp(dates[position]),
            training_dates=tuple(pd.Timestamp(value) for value in dates[position-window_dates:position]),
        )
        for position in range(window_dates, len(dates))
    ]

