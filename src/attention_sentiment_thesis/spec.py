"""Single source of truth for the final thesis numerical specification."""

from dataclasses import dataclass

FIRM_VARYING_REGRESSORS = ("sentiment", "wpv_lag1", "atv", "r_cc", "r_oc")

CONTROL_COLUMNS = (
    "r_cc",
    "r_oc",
    "market_excess_return",
    "smb",
    "hml",
    "rmw",
    "cma",
    "market_volatility",
    "jgb_10y",
    "jp_3m",
    "us_default_spread",
    "usdjpy_return",
    "eurjpy_return",
    "gbpjpy_return",
    "cpi_jp",
    "ip_jp",
    "m1_jp",
    "unemployment_jp",
)

MODEL_PREDICTORS = {
    "M_ctrl": CONTROL_COLUMNS,
    "M_sent": CONTROL_COLUMNS + ("sentiment",),
    "M_att": CONTROL_COLUMNS + ("wpv_lag1", "atv"),
    "M_full": CONTROL_COLUMNS + ("sentiment", "wpv_lag1", "atv"),
}

SIMPLIFIED_MODEL_PREDICTORS = {
    "S_ctrl": ("r_cc", "r_oc"),
    "S_sent": ("r_cc", "r_oc", "sentiment"),
    "S_att": ("r_cc", "r_oc", "wpv_lag1", "atv"),
    "S_full": ("r_cc", "r_oc", "sentiment", "wpv_lag1", "atv"),
}

CLARK_WEST_COMPARISONS = (
    ("benchmark", "M_ctrl"),
    ("M_ctrl", "M_sent"),
    ("M_ctrl", "M_att"),
    ("M_ctrl", "M_full"),
    ("M_sent", "M_full"),
    ("M_att", "M_full"),
)

@dataclass(frozen=True)
class FinalSpecification:
    candidate_vocabulary_size: int = 88
    expression_premium_threshold: float = 0.0015
    beta_window: int = 260
    beta_min_observations: int = 60
    wpv_median_window: int = 8
    wpv_min_observations: int = 4
    atv_window: int = 60
    atv_min_observations: int = 30
    monthly_availability_trading_day: int = 40
    winsor_lower_quantile: float = 0.01
    winsor_upper_quantile: float = 0.99
    baseline_oos_window: int = 504
    oos_window_sensitivities: tuple[int, int] = (378, 630)
    is_start: str = "2021-01-04"
    is_end: str = "2023-12-29"
    oos_start: str = "2024-01-04"
    oos_end: str = "2025-12-30"

FINAL_SPEC = FinalSpecification()
