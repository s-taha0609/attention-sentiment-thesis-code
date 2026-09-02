"""Input-schema validation without provider-specific fields."""

from collections.abc import Iterable
import pandas as pd

class SchemaError(ValueError):
    """Raised when an input table violates the public schema."""

def require_columns(frame: pd.DataFrame, required: Iterable[str], table: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise SchemaError(f"{table} is missing required columns: {missing}")

def validate_expression_events(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "firm_id", "publication_timestamp", "effective_trading_date",
        "expressions", "eligible_for_daily_score",
    }
    require_columns(frame, required, "expression_events")
    out = frame.copy()
    out["publication_timestamp"] = pd.to_datetime(out["publication_timestamp"], errors="raise")
    out["effective_trading_date"] = pd.to_datetime(
        out["effective_trading_date"], errors="raise"
    ).dt.normalize()
    if out[["firm_id", "publication_timestamp", "effective_trading_date"]].isna().any().any():
        raise SchemaError("expression_events contains null identifiers or dates")
    if not out["expressions"].map(lambda value: isinstance(value, (list, tuple, set))).all():
        raise SchemaError("expressions must contain a sequence of expression identifiers")
    return out

def validate_panel(frame: pd.DataFrame) -> pd.DataFrame:
    require_columns(frame, {"firm_id", "trading_date", "target_return"}, "panel")
    out = frame.copy()
    out["trading_date"] = pd.to_datetime(out["trading_date"], errors="raise").dt.normalize()
    if out.duplicated(["firm_id", "trading_date"]).any():
        raise SchemaError("panel must be unique by firm_id and trading_date")
    return out

