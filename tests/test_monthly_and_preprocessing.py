import unittest
import numpy as np
import pandas as pd

from attention_sentiment_thesis.preprocessing.monthly import align_monthly_controls
from attention_sentiment_thesis.preprocessing.panel import (
    fit_winsor_limits, apply_winsor_limits,
)

class MonthlyAndPreprocessingTests(unittest.TestCase):
    def test_release_aware_alignment_has_no_future_data(self):
        dates = pd.bdate_range("2024-02-01", "2024-05-31")
        monthly = pd.DataFrame({
            "reference_month": ["2024-01-01", "2024-02-01"],
            "macro_value": [10.0, 20.0],
        })
        aligned = align_monthly_controls(monthly, dates)
        jan_release = dates[dates > pd.Timestamp("2024-01-31")][39]
        feb_release = dates[dates > pd.Timestamp("2024-02-29")][39]
        before = aligned.loc[aligned["trading_date"] < jan_release, "macro_value"]
        self.assertTrue(before.isna().all())
        self.assertEqual(
            aligned.loc[aligned["trading_date"].eq(jan_release), "macro_value"].iloc[0],
            10.0,
        )
        between = aligned[
            aligned["trading_date"].ge(jan_release)
            & aligned["trading_date"].lt(feb_release)
        ]
        self.assertTrue(between["macro_value"].eq(10.0).all())

    def test_is_fitted_preprocessing_ignores_oos_extreme(self):
        training = pd.DataFrame({"x": np.arange(100.0)})
        oos = pd.DataFrame({"x": [1000000.0]})
        limits = fit_winsor_limits(training, ["x"])
        transformed = apply_winsor_limits(oos, limits)
        self.assertEqual(limits["x"], (0.99, 98.01))
        self.assertEqual(transformed.loc[0, "x"], 98.01)

if __name__ == "__main__":
    unittest.main()

