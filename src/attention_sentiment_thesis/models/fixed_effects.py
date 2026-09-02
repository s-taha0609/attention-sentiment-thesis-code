"""Final fixed-effects estimators and two-way clustered inference."""

from dataclasses import dataclass
from collections.abc import Sequence
import numpy as np
import pandas as pd
from ..schemas import require_columns
from ..spec import FIRM_VARYING_REGRESSORS

@dataclass(frozen=True)
class RegressionResult:
    names: tuple[str, ...]
    coefficients: np.ndarray
    covariance: np.ndarray
    nobs: int
    residuals: np.ndarray
    cluster_df: int

    def coefficient(self, name: str) -> float:
        return float(self.coefficients[self.names.index(name)])

def _cluster_df(firm, date) -> int:
    value = min(pd.Series(firm).nunique() - 1, pd.Series(date).nunique() - 1)
    if value < 1:
        raise ValueError("two-way clustering requires at least two firms and dates")
    return int(value)

def _validate_covariance(covariance: np.ndarray) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=float)
    covariance = (covariance + covariance.T) / 2.0
    diagonal = np.diag(covariance).copy()
    diagonal[(diagonal < 0) & (diagonal > -1e-18)] = 0.0
    if not np.isfinite(covariance).all() or (diagonal < 0).any():
        raise RuntimeError("invalid clustered covariance")
    return covariance

def fit_headline_regression(
    frame: pd.DataFrame,
    *,
    target: str = "target_return",
) -> RegressionResult:
    """Five regressors, firm FE, date FE, and debiased two-way clustering."""
    from linearmodels.panel import PanelOLS

    columns = [target, "firm_id", "trading_date", *FIRM_VARYING_REGRESSORS]
    require_columns(frame, columns, "headline_panel")
    fit = frame[columns].dropna().copy()
    fit["trading_date"] = pd.to_datetime(fit["trading_date"]).dt.normalize()
    fit = fit.set_index(["firm_id", "trading_date"]).sort_index()
    model = PanelOLS(
        dependent=fit[target],
        exog=fit[list(FIRM_VARYING_REGRESSORS)],
        entity_effects=True,
        time_effects=True,
        drop_absorbed=False,
        check_rank=True,
    )
    result = model.fit(
        cov_type="clustered",
        cluster_entity=True,
        cluster_time=True,
        debiased=True,
    )
    names = tuple(FIRM_VARYING_REGRESSORS)
    coefficients = result.params.reindex(names).to_numpy(float)
    covariance = _validate_covariance(
        result.cov.reindex(index=names, columns=names).to_numpy(float)
    )
    return RegressionResult(
        names, coefficients, covariance, len(fit),
        np.asarray(result.resids), _cluster_df(
            fit.index.get_level_values(0), fit.index.get_level_values(1)
        ),
    )

def _subtract_group_means(matrix: np.ndarray, codes: np.ndarray, groups: int) -> float:
    counts = np.bincount(codes, minlength=groups).astype(float)
    if (counts == 0).any():
        raise ValueError("empty fixed-effect group")
    maximum = 0.0
    for index in range(matrix.shape[1]):
        means = np.bincount(
            codes, weights=matrix[:, index], minlength=groups
        ) / counts
        maximum = max(maximum, float(np.max(np.abs(means))))
        matrix[:, index] -= means[codes]
    return maximum

def _maximum_group_mean(matrix: np.ndarray, codes: np.ndarray, groups: int) -> float:
    counts = np.bincount(codes, minlength=groups).astype(float)
    return max(
        float(np.max(np.abs(
            np.bincount(codes, weights=matrix[:, index], minlength=groups) / counts
        )))
        for index in range(matrix.shape[1])
    )

def _absorb_two_effects(
    values: np.ndarray,
    first_codes: np.ndarray,
    second_codes: np.ndarray,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> np.ndarray:
    first_groups = int(first_codes.max()) + 1
    second_groups = int(second_codes.max()) + 1
    for _ in range(max_iterations):
        adjustment = max(
            _subtract_group_means(values, first_codes, first_groups),
            _subtract_group_means(values, second_codes, second_groups),
        )
        if adjustment < tolerance:
            break
    else:
        raise RuntimeError("fixed-effect absorption did not converge")
    if (
        _maximum_group_mean(values, first_codes, first_groups) > 1e-8
        or _maximum_group_mean(values, second_codes, second_groups) > 1e-8
    ):
        raise RuntimeError("fixed-effect absorption validation failed")
    return values

def fit_stability_regression(
    frame: pd.DataFrame,
    *,
    target: str = "target_return",
    period_col: str = "is_oos_period",
) -> RegressionResult:
    """Firm-by-period/date FE with OOS interactions and corrected clusters."""
    import statsmodels.api as sm
    from statsmodels.stats.sandwich_covariance import cov_cluster_2groups

    columns = [target, "firm_id", "trading_date", period_col, *FIRM_VARYING_REGRESSORS]
    require_columns(frame, columns, "stability_panel")
    fit = frame[columns].dropna().copy()
    fit["trading_date"] = pd.to_datetime(fit["trading_date"]).dt.normalize()
    is_oos = fit[period_col].astype(str).str.upper().eq("OOS").astype(float)
    interactions = []
    for column in FIRM_VARYING_REGRESSORS:
        name = f"{column}:OOS"
        fit[name] = fit[column].astype(float) * is_oos
        interactions.append(name)
    names = (*FIRM_VARYING_REGRESSORS, *interactions)
    numeric = [target, *names]
    values = fit[numeric].to_numpy(float, copy=True)
    firm_period_codes = pd.factorize(
        fit["firm_id"].astype(str) + "::" + fit[period_col].astype(str), sort=True
    )[0].astype(np.int32)
    firm_codes = pd.factorize(fit["firm_id"].astype(str), sort=True)[0].astype(np.int32)
    date_codes = pd.factorize(fit["trading_date"], sort=True)[0].astype(np.int32)
    values = _absorb_two_effects(values, firm_period_codes, date_codes)
    y, x = values[:, 0], values[:, 1:]
    if np.linalg.matrix_rank(x) != x.shape[1]:
        raise RuntimeError("stability regressor matrix is rank deficient")
    result = sm.OLS(y, x, hasconst=False).fit()
    covariance = _validate_covariance(
        cov_cluster_2groups(
            result, firm_codes, date_codes, use_correction=True
        )[0]
    )
    return RegressionResult(
        tuple(names), np.asarray(result.params), covariance, len(fit),
        np.asarray(result.resid), _cluster_df(firm_codes, date_codes),
    )

def wald_test(result: RegressionResult, names: Sequence[str]) -> dict[str, float]:
    """Final Wald F test with cluster degrees of freedom."""
    from scipy.stats import f

    indices = [result.names.index(name) for name in names]
    beta = result.coefficients[indices]
    covariance = result.covariance[np.ix_(indices, indices)]
    if np.linalg.matrix_rank(covariance, tol=1e-14) != len(indices):
        raise RuntimeError("Wald covariance is rank deficient")
    chi_square = float(beta.T @ np.linalg.pinv(covariance, rcond=1e-14) @ beta)
    statistic = chi_square / len(indices)
    return {
        "statistic": statistic,
        "df_num": float(len(indices)),
        "df_den": float(result.cluster_df),
        "p_value": float(f.sf(statistic, len(indices), result.cluster_df)),
    }

