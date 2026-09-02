import unittest
import pandas as pd

from attention_sentiment_thesis.oos.window_utils import generate_refit_schedule
from attention_sentiment_thesis.oos.rolling import rolling_historical_mean
from attention_sentiment_thesis.oos.evaluation import daily_mean_adjust

class OOSTests(unittest.TestCase):
    def test_window_excludes_forecast_origin(self):
        dates = pd.bdate_range("2024-01-01", periods=6)
        schedule = generate_refit_schedule(dates, window_dates=3)
        self.assertEqual(schedule[0].origin, dates[3])
        self.assertNotIn(schedule[0].origin, schedule[0].training_dates)
        self.assertEqual(schedule[0].training_dates, tuple(dates[:3]))

    def test_historical_mean_excludes_current(self):
        dates = pd.bdate_range("2024-01-01", periods=4)
        panel = pd.DataFrame({
            "firm_id": ["a"] * 4,
            "trading_date": dates,
            "target_return": [1.0, 3.0, 100.0, 9.0],
        })
        result = rolling_historical_mean(panel, window_dates=2)
        first = result[result["trading_date"].eq(dates[2])]
        self.assertEqual(first["benchmark"].iloc[0], 2.0)

    def test_daily_cross_sectional_mean_removal(self):
        sample = pd.DataFrame({
            "firm_id": ["a", "b", "a", "b"],
            "trading_date": pd.to_datetime(
                ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]
            ),
            "realized": [1.0, 3.0, 2.0, 6.0],
            "benchmark": [0.0, 2.0, 1.0, 3.0],
            "M_full": [0.5, 1.5, 1.0, 5.0],
        })
        adjusted = daily_mean_adjust(sample, ["benchmark", "M_full"])
        means = adjusted.groupby("trading_date")[["realized", "benchmark", "M_full"]].mean()
        self.assertTrue((means.abs() < 1e-12).all().all())

if __name__ == "__main__":
    unittest.main()

