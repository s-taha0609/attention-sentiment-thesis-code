import unittest
import numpy as np
import pandas as pd

from attention_sentiment_thesis.models.fixed_effects import (
    fit_headline_regression, fit_stability_regression, wald_test,
)
from attention_sentiment_thesis.spec import FIRM_VARYING_REGRESSORS

class ModelTests(unittest.TestCase):
    @staticmethod
    def synthetic_panel():
        random = np.random.default_rng(42)
        firms = [f"firm_{index:02d}" for index in range(10)]
        dates = pd.bdate_range("2024-01-01", periods=20)
        rows = []
        slopes = np.array([0.2, -0.1, 0.05, 0.3, -0.2])
        for firm_index, firm in enumerate(firms):
            for date_index, date in enumerate(dates):
                x = random.normal(size=5)
                period = "IS" if date_index < 10 else "OOS"
                change = 0.03 * x.sum() if period == "OOS" else 0.0
                y = (
                    x @ slopes + change + firm_index * 0.01
                    + date_index * 0.002 + random.normal(scale=0.1)
                )
                row = {
                    "firm_id": firm, "trading_date": date,
                    "is_oos_period": period, "target_return": y,
                }
                row.update(dict(zip(FIRM_VARYING_REGRESSORS, x)))
                rows.append(row)
        return pd.DataFrame(rows)

    def test_headline_final_effects_and_clusters(self):
        result = fit_headline_regression(self.synthetic_panel())
        self.assertEqual(result.names, FIRM_VARYING_REGRESSORS)
        self.assertEqual(result.covariance.shape, (5, 5))
        self.assertEqual(result.cluster_df, 9)

    def test_stability_interactions_and_wald(self):
        result = fit_stability_regression(self.synthetic_panel())
        names = ("sentiment:OOS", "wpv_lag1:OOS", "atv:OOS")
        test = wald_test(result, names)
        self.assertEqual(test["df_num"], 3.0)
        self.assertEqual(test["df_den"], 9.0)

if __name__ == "__main__":
    unittest.main()
