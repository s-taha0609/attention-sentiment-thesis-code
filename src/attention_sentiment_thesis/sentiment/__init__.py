"""Provider-neutral sentiment algorithms.

The package accepts expression identifiers, never article text.
"""

from .build_dictionary import build_presence_matrix, validate_candidate_vocabulary
from .compute_beta import compute_rolling_beta
from .estimate_polarity import estimate_annual_polarities
from .compute_sentiment import aggregate_daily_sentiment

__all__ = [
    "build_presence_matrix", "validate_candidate_vocabulary",
    "compute_rolling_beta", "estimate_annual_polarities",
    "aggregate_daily_sentiment",
]

