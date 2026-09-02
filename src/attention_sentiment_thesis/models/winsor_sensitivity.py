"""Common-control winsorization sensitivity preparation."""

import pandas as pd
import numpy as np
from ..schemas import require_columns
from ..spec import CONTROL_COLUMNS
from .fixed_effects import RegressionResult, _cluster_df, _validate_covariance

COMMON_CONTROL_COLUMNS = tuple(
    column for column in CONTROL_COLUMNS if column not in {"r_cc", "r_oc"}
)

def prepare_common_control_sensitivity(
    capped: pd.DataFrame,
    uncapped: pd.DataFrame,
) -> pd.DataFrame:
    """Replace only the 16 common controls with release-aware uncapped values."""
    keys = ["firm_id", "trading_date"]
    replacement = uncapped[keys + list(COMMON_CONTROL_COLUMNS)].copy()
    renamed = {column: f"{column}__uncapped" for column in COMMON_CONTROL_COLUMNS}
    replacement = replacement.rename(columns=renamed)
    out = capped.merge(replacement, on=keys, how="left", validate="one_to_one")
    for column in COMMON_CONTROL_COLUMNS:
        out[column] = out.pop(f"{column}__uncapped")
    return out

def fit_winsor_sensitivity_regression(
    frame: pd.DataFrame,
    *,
    target: str = "target_return",
) -> RegressionResult:
    """Three focal variables plus 18 controls, firm FE, and two-way clusters."""
    from linearmodels.panel import PanelOLS

    focal = ("sentiment", "wpv_lag1", "atv")
    regressors = (*focal, *CONTROL_COLUMNS)
    columns = [target, "firm_id", "trading_date", *regressors]
    require_columns(frame, columns, "winsor_sensitivity_panel")
    fit = frame[columns].dropna().copy()
    fit["trading_date"] = pd.to_datetime(fit["trading_date"]).dt.normalize()
    fit = fit.set_index(["firm_id", "trading_date"]).sort_index()
    result = PanelOLS(
        fit[target],
        fit[list(regressors)],
        entity_effects=True,
        time_effects=False,
        drop_absorbed=False,
        check_rank=True,
    ).fit(
        cov_type="clustered",
        cluster_entity=True,
        cluster_time=True,
        debiased=True,
    )
    covariance = _validate_covariance(
        result.cov.reindex(index=regressors, columns=regressors).to_numpy(float)
    )
    return RegressionResult(
        tuple(regressors), result.params.reindex(regressors).to_numpy(float),
        covariance, len(fit), np.asarray(result.resids),
        _cluster_df(fit.index.get_level_values(0), fit.index.get_level_values(1)),
    )
