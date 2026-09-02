"""Build occurrence indicators from externally supplied expression identifiers."""

from collections.abc import Sequence
import pandas as pd
from ..schemas import SchemaError, validate_expression_events
from ..spec import FINAL_SPEC

def validate_candidate_vocabulary(
    vocabulary: Sequence[str],
    *,
    required_size: int = FINAL_SPEC.candidate_vocabulary_size,
) -> tuple[str, ...]:
    values = tuple(str(value) for value in vocabulary)
    if len(values) != required_size:
        raise SchemaError(f"candidate vocabulary must contain exactly {required_size} items")
    if len(set(values)) != len(values):
        raise SchemaError("candidate vocabulary identifiers must be unique")
    return values

def build_presence_matrix(
    events: pd.DataFrame,
    vocabulary: Sequence[str],
) -> pd.DataFrame:
    """Append one binary presence column per candidate expression."""
    out = validate_expression_events(events)
    vocab = validate_candidate_vocabulary(vocabulary)
    expression_sets = out["expressions"].map(set)
    indicators = {
        expression: expression_sets.map(lambda values, e=expression: int(e in values))
        for expression in vocab
    }
    return pd.concat([out.reset_index(drop=True), pd.DataFrame(indicators)], axis=1)

