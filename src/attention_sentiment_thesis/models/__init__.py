"""Final fixed-effects and sensitivity specifications."""

from .fixed_effects import fit_headline_regression, fit_stability_regression, wald_test
from .winsor_sensitivity import (
    prepare_common_control_sensitivity, fit_winsor_sensitivity_regression,
)

__all__ = [
    "fit_headline_regression", "fit_stability_regression", "wald_test",
    "prepare_common_control_sensitivity", "fit_winsor_sensitivity_regression",
]
